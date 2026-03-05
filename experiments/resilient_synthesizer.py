"""
experiments/resilient_synthesizer.py
--------------------------------------
Drop-in replacement for TextSynthesizer for the high-volume Global Confusion
Index build.  Overrides _single_pass only — all gate logic (boilerplate filter,
multi-pass for dense blocks, PostHocAligner, useful-fact filter) lives in the
base TextSynthesizer and runs unchanged.

Design:
  • Overrides _single_pass to use tenacity retry wrappers with correct backoff.
  • Both _call_raw and _call_instructor carry max_tokens=4096 to prevent the
    mid-JSON truncation that the base class fixed in its own _single_pass.
  • Calls _preprocess_json before Pydantic construction to coerce scale:null→1.0
    and truncate overlong grounding_quotes, matching base-class behaviour.
  • Key rotation via settings.GROQ_KEYS with a unified itertools.cycle.
  • Output cap and strict grounding rule injected into prompt_template.
  • extract_facts_chunked splits oversized blocks and aggregates sub-results.
    Returns the 3-tuple (accepted_rows, proposed_count, any_failed) matching
    the updated extract_facts_with_proposals signature.

CRITICAL ALIGNMENT WITH synthesis.py (changes vs previous version):
  1. extract_facts_with_proposals now returns (rows, proposals, extraction_failed).
     extract_facts_chunked now unpacks all 3 values and propagates any_failed.
  2. _call_raw and _call_instructor both pass max_tokens=4096.  The base class
     added this to fix finish_reason='length' truncation; without it our override
     regressed that fix.
  3. _single_pass now imports and calls _preprocess_json() after json.loads.
     The base class uses this to coerce scale:null→1.0 and truncate quotes.
     Without it the ResilientTextSynthesizer still hit the deterministic
     ValidationError retry cascade on qualitative facts.
  4. _call_raw now checks finish_reason=='length' and logs a warning, matching
     the base class visibility.
  5. Retry policy kept at 5 attempts / 90s max (base uses 2 / 30s) because
     this runs as an unattended batch job where a 6-min per-block stall is
     acceptable; the base class runs interactively where it is not.
  6. BUG 4 FIX: Sub-chunk IDs now use MD5 instead of Python's built-in hash().
     Python's hash() is randomised per-process by PYTHONHASHSEED, so sub-chunk
     IDs differed between the original Phase 1 run and any checkpoint-resume run.
     This caused UFLRow.source_chunk_id to carry a stale suffix, which made the
     schema_summary FIX 15 parent-metadata lookup fail silently — the metric was
     never added to metrics_by_record, so the Navigator lost knowledge of it at
     query time.  MD5 is deterministic across all runs and its first 4 hex chars
     match the existing r"_[0-9a-f]{4}$" strip pattern without any other changes.
"""

import asyncio
import hashlib
import itertools
import json
import os
import re
import sys
from typing import List, Tuple

import instructor
import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from venra.models import ScrapedFact, FactExtractionResponse
from venra.synthesis import TextSynthesizer, _preprocess_json
from venra.logging_config import logger
from venra.config import settings

# think-tag stripper — import public alias, private fallback, then local def
try:
    from venra.synthesis import strip_think_tags as _strip_think_tags
except ImportError:
    try:
        from venra.synthesis import _strip_think_tags
    except ImportError:
        def _strip_think_tags(text: str) -> str:
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

# Max tokens sent to the SLM — must match synthesis._SLM_MAX_OUTPUT_TOKENS.
# Keeps our override aligned with the base class truncation fix.
_SLM_MAX_OUTPUT_TOKENS = 4096


# ── Table detection ───────────────────────────────────────────────────────────

def is_table_aggressive(content: str) -> bool:
    """
    Multi-signal table detector.  Routes content to TableMelter (deterministic,
    free) rather than the SLM path whenever the block is clearly tabular.

    Signals (any one triggers True):
      1. Markdown pipe-table with separator row.
      2. CSV-style: 3+ commas on the majority of lines.
      3. TSV-style: 3+ tabs on the majority of lines.
      4. Columnar numeric: 3+ whitespace-separated numbers on 3+ lines.
      5. Consistent pipe-column count across lines.
    """
    if not content or len(content.strip()) < 10:
        return False
    if "|" in content and "---" in content:
        return True
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    csv_lines = sum(1 for ln in lines if ln.count(",") >= 3)
    if csv_lines >= len(lines) * 0.6:
        return True
    tsv_lines = sum(1 for ln in lines if ln.count("\t") >= 3)
    if tsv_lines >= len(lines) * 0.6:
        return True
    numeric_row_re = re.compile(r"-?\d[\d,\.]*(?:\s+-?\d[\d,\.]*){2,}")
    if sum(1 for ln in lines if numeric_row_re.search(ln)) >= 3:
        return True
    pipe_counts = [ln.count("|") for ln in lines if "|" in ln]
    if len(pipe_counts) >= 3:
        most_common = max(set(pipe_counts), key=pipe_counts.count)
        if pipe_counts.count(most_common) >= len(pipe_counts) * 0.7 and most_common >= 2:
            return True
    return False


# ── Chunk splitter ────────────────────────────────────────────────────────────

def split_large_block(content: str, max_chars: int = 1500) -> List[str]:
    """
    Splits a text block into sub-chunks ≤ max_chars at sentence boundaries.
    Prevents SLM max-token truncation mid-JSON on dense financial paragraphs.
    """
    if len(content) <= max_chars:
        return [content]
    sentences = re.split(r"(?<=[.!?])\s+", content)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                words, current = sentence.split(), ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current = (current + " " + word).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = sentence
    if current:
        chunks.append(current)
    return chunks if chunks else [content]


def _sub_chunk_suffix(chunk_text: str) -> str:
    """
    BUG 4 FIX: Deterministic 4-hex-char suffix for sub-chunk IDs.

    Previously: hash(chunk_text) & 0xFFFF
      → non-deterministic across Python restarts (PYTHONHASHSEED randomised)
      → on checkpoint resume, sub-chunk ID suffix changes
      → UFLRow.source_chunk_id carries stale suffix
      → schema_summary FIX 15 parent-metadata lookup fails silently
      → metric dropped from metrics_by_record
      → Navigator loses knowledge of that metric at query time

    Now: hashlib.md5(chunk_text.encode()).hexdigest()[:4]
      → always the same string for the same chunk text
      → still exactly 4 hex chars, matching the r"_[0-9a-f]{4}$" strip pattern
      → zero changes required anywhere else
    """
    return hashlib.md5(chunk_text.encode()).hexdigest()[:4]


# ── Retryable exception types ─────────────────────────────────────────────────

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.APITimeoutError,
    TimeoutError,
)


# ── ResilientTextSynthesizer ──────────────────────────────────────────────────

class ResilientTextSynthesizer(TextSynthesizer):
    """
    Wraps TextSynthesizer for unattended batch runs.
    Adds: per-block output cap, key rotation, tenacity retry wrappers,
    max_tokens alignment, _preprocess_json coercion, and chunk splitting.

    Only _single_pass is overridden.  All gate logic (boilerplate filter,
    multi-pass for dense blocks, PostHocAligner, useful-fact filter) is
    inherited from TextSynthesizer and executes unchanged.
    """

    def __init__(self, entity_id: str, entity_name_raw: str = "Unknown Entity"):
        super().__init__(entity_id, entity_name_raw)

        # Top-5 cap prevents JSON explosions on dense paragraphs.
        # Grounding rule reduces Semantic Rejection noise from the
        # Double-Lock Aligner — the LLM is told not to propose metric names
        # that don't appear verbatim in the source text.
        self.prompt_template = (
            self.prompt_template
            + "\n\nCRITICAL CONSTRAINT: Extract ONLY the TOP 5 most important "
              "financial metrics from the text. Ignore the rest."
            + "\n\nSTRICT GROUNDING RULE: Only extract a metric if BOTH its "
              "exact label AND a specific numerical value are explicitly present "
              "as literal words in the source text. Do NOT infer, generalise, "
              "or use category names (e.g. 'Revenue', 'Expenses', 'Comparability') "
              "unless that exact word is followed by a specific number in the text. "
              "If the metric name is not verbatim in the source, omit it entirely."
        )

        keys = settings.GROQ_KEYS
        if not keys:
            raise ValueError(
                "settings.GROQ_KEYS is empty. "
                "Populate GROQ_API_KEY / GROQ_API_KEY_2…N in .env before running."
            )

        raw_clients, instructor_clients = [], []
        for k in keys:
            raw = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=k,
                timeout=30.0,
            )
            raw_clients.append(raw)
            instructor_clients.append(
                instructor.from_openai(raw, mode=instructor.Mode.JSON)
            )

        # Unified cycle: instructor and raw clients rotate in lockstep so a
        # rate-limit advance on one path carries over to the other.
        self._client_cycle = itertools.cycle(zip(instructor_clients, raw_clients))

    def _next_clients(self) -> Tuple[instructor.AsyncInstructor, AsyncOpenAI]:
        return next(self._client_cycle)

    # ── Retry-wrapped API calls ────────────────────────────────────────────────
    # 5 attempts / 90s max: appropriate for an unattended batch job.
    # Groq TPM 429 responses ask for ~30s cooldown; 5 × 90s = up to 6 min per
    # block which is acceptable here.  The base class uses 2 / 30s for its
    # interactive path.

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=5, max=90),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _call_raw(
        self,
        filled_prompt: str,
        target_model: str,
        raw_client: AsyncOpenAI,
        schema_str: str,
    ) -> str:
        completion = await raw_client.chat.completions.create(
            model=target_model,
            max_tokens=_SLM_MAX_OUTPUT_TOKENS,   # aligned with synthesis._SLM_MAX_OUTPUT_TOKENS
            messages=[
                {"role": "system", "content": filled_prompt},
                {
                    "role": "user",
                    "content": (
                        "Extract facts. Respond ONLY with a valid JSON object "
                        f"matching this exact schema:\n{schema_str}\n\n"
                        "Do NOT include any preamble, markdown fences, or reasoning."
                    ),
                },
            ],
            temperature=0.0,
        )
        # Warn on length truncation — the caller will get a partial JSON
        # string that may fail json.loads and trigger the instructor fallback.
        if completion.choices[0].finish_reason == "length":
            logger.warning(
                "SLM hit max_tokens limit (finish_reason='length'). "
                "Response may be truncated JSON — instructor fallback will be attempted."
            )
        return completion.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=5, max=90),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _call_instructor(
        self,
        filled_prompt: str,
        target_model: str,
        instructor_client: instructor.AsyncInstructor,
    ) -> List[ScrapedFact]:
        resp = await instructor_client.chat.completions.create(
            model=target_model,
            max_tokens=_SLM_MAX_OUTPUT_TOKENS,   # aligned with synthesis._SLM_MAX_OUTPUT_TOKENS
            response_model=FactExtractionResponse,
            messages=[
                {"role": "system", "content": filled_prompt},
                {"role": "user",   "content": "Extract facts."},
            ],
            temperature=0.0,
        )
        return resp.facts

    # ── _single_pass override ─────────────────────────────────────────────────

    async def _single_pass(self, filled_prompt: str, target_model: str) -> List[ScrapedFact]:
        """
        Raw JSON first, instructor fallback on parse failure.

        Aligned with base TextSynthesizer._single_pass:
          • max_tokens=_SLM_MAX_OUTPUT_TOKENS on both paths (truncation fix).
          • _preprocess_json called before Pydantic (scale:null coercion,
            quote truncation) so qualitative facts don't trigger the
            deterministic ValidationError retry cascade.
          • Uses our tenacity wrappers (5 attempts / 90s) instead of the
            base class AsyncRetrying (2 attempts / 30s).
        """
        instructor_client, raw_client = self._next_clients()

        try:
            import pydantic
            if int(pydantic.VERSION.split(".")[0]) >= 2:
                schema = FactExtractionResponse.model_json_schema(mode="serialization")
            else:
                schema = FactExtractionResponse.schema()
        except Exception:
            schema = {"type": "object", "properties": {"facts": {"type": "array"}}}

        try:
            if "$defs" in schema and "ScrapedFact" in schema.get("$defs", {}):
                schema["properties"]["facts"]["items"] = schema["$defs"]["ScrapedFact"]
                del schema["$defs"]
        except (KeyError, TypeError):
            pass

        schema_str = json.dumps(schema, indent=2)

        try:
            raw_text   = await self._call_raw(filled_prompt, target_model, raw_client, schema_str)
            clean_text = _strip_think_tags(raw_text)
            clean_text = re.sub(r"```(?:json)?|```", "", clean_text).strip()
            data       = json.loads(clean_text)
            data       = _preprocess_json(data)   # scale:null→1.0, quote truncation
            return FactExtractionResponse(**data).facts
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON decode error (likely truncation) — using instructor fallback. {e}"
            )
            return await self._call_instructor(filled_prompt, target_model, instructor_client)
        except Exception as e:
            logger.error(f"Raw pass failed ({type(e).__name__}): {str(e)[:120]}")
            return []

    # ── Chunk-aware extraction ─────────────────────────────────────────────────

    async def extract_facts_chunked(
        self,
        block: "DocBlock",   # type: ignore[name-defined]
        context_str: str = "",
        max_chars: int = 1500,
    ) -> Tuple[List["UFLRow"], int, bool]:  # type: ignore[name-defined]
        """
        Splits oversized blocks and merges extraction results.

        Returns (accepted_rows, proposed_count, any_failed).

        any_failed=True means at least one sub-chunk exhausted all retries.
        Callers should NOT checkpoint a partial result — log and skip the block
        so it will be retried on the next run.

        The 3-tuple return aligns with extract_facts_with_proposals which now
        returns (accepted_rows, raw_proposals, extraction_failed).  The old
        2-tuple unpack `accepted, proposals = await ...` raised ValueError
        at runtime on the very first text block.

        BUG 4 FIX: Sub-chunk IDs use _sub_chunk_suffix() (MD5-based) instead
        of the previous hash()-based inline expression.  See _sub_chunk_suffix
        docstring for the full rationale.
        """
        sub_chunks = split_large_block(block.content, max_chars=max_chars)
        all_accepted:  List = []
        total_proposed: int = 0
        any_failed:    bool = False

        for chunk_text in sub_chunks:
            from venra.models import DocBlock
            # BUG 4 FIX: use deterministic MD5 suffix instead of hash()
            sub_block = DocBlock(
                id=block.id + f"_{_sub_chunk_suffix(chunk_text)}",
                content=chunk_text,
                block_type=block.block_type,
                section_path=block.section_path,
                page_num=block.page_num,
            )

            if hasattr(self, "extract_facts_with_proposals"):
                accepted, proposals, failed = await self.extract_facts_with_proposals(
                    sub_block, context_str=context_str
                )
                all_accepted.extend(accepted)
                total_proposed += len(proposals)
                if failed:
                    any_failed = True
            else:
                # Safety fallback for any base class that only exposes extract_facts
                accepted = await self.extract_facts(sub_block, context_str=context_str)
                all_accepted.extend(accepted)
                total_proposed += len(accepted)

        return all_accepted, total_proposed, any_failed