"""
venra/synthesis.py
Knowledge synthesis: Entity resolution, Table melting, Text-to-Fact extraction,
and Post-Hoc Alignment verification.

Changes (Phase 3 & 4):
  Phase 3 — SLM Text-to-Fact Extraction Upgrades:
    • Think-tag stripping: regex removes <think>…</think> reasoning traces
      before the response reaches Pydantic / Instructor.
    • Multi-pass extraction: extract_facts() runs the SLM 2-3 times in
      parallel via asyncio.gather for dense text blocks, then deduplicates
      by metric_name + period overlap.

  Phase 4 — Post-Hoc Aligner (Hallucination Killer):
    • PostHocAligner class: reverse-grounds every ScrapedFact against the
      source TextBlock using three tiers:
        Tier 1 — difflib.SequenceMatcher exact-match pass.
        Tier 2 — sliding-window Counter token-intersection fuzzy pass.
      Sets char_interval, alignment_status, and confidence_score on each
      UFLRow before it enters the ledger.
"""

from __future__ import annotations

import asyncio
import collections
import difflib
import hashlib
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import instructor
import chromadb
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

from venra.models import (
    DocBlock,
    EntityMetadata,
    FactExtractionResponse,
    ScrapedFact,
    TableBlock,
    TextBlock,
    UFLRow,
)
from venra.config import settings
from venra.prompt_loader import load_prompt
from venra.logging_config import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum character length for a text block to be worth extracting from
_MIN_BLOCK_CHARS: int = 10

# Number of parallel SLM passes for dense text blocks
_MULTI_PASS_COUNT: int = 3

# Character threshold above which multi-pass is triggered
_DENSE_BLOCK_CHARS: int = 800

# Confidence scores by tier (matches UFL_spec §1)
_CONFIDENCE_TABLE: float = 0.95
_CONFIDENCE_TEXT_ALIGNED: float = 0.70
_CONFIDENCE_UNALIGNED: float = 0.0

# Fuzzy sliding-window parameters
_FUZZY_WINDOW_TOKENS: int = 20     # token window width for sliding search
_FUZZY_MIN_OVERLAP: float = 0.55   # Counter recall pre-check threshold (fast gate)
_FUZZY_MIN_RECALL: float = 0.85    # difflib recall threshold (fraction of raw found in window)


# ---------------------------------------------------------------------------
# Utility: think-tag stripping
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> reasoning traces from raw LLM output."""
    return _THINK_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------

class EntityResolver:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found. EntityResolver might fail against real API.")

        self.client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=self.api_key or "dummy_key"),
            mode=instructor.Mode.JSON,
        )
        self.model = settings.SLM_MODEL_PRECISION

    async def resolve_entity(self, blocks: List[DocBlock]) -> EntityMetadata:
        """Analyse the first few blocks (Cover Page) to extract canonical entity info."""
        context_text = ""
        for block in blocks[:20]:
            context_text += (
                f"[{block.block_type.value.upper()}] Path: {block.section_path}\n"
                f"Content: {block.content}\n---\n"
            )

        logger.info("Resolving Entity from Cover Page context…")

        resp = self.client.chat.completions.create(
            model=self.model,
            response_model=EntityMetadata,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial data extraction engine. You will be given "
                        "the raw text of a 10-K cover page. Your job is to extract the "
                        "exact legal name, CIK (if present), and create a Canonical ID "
                        "(e.g. 'ID_AAPL') and list of common aliases (e.g. 'The Company')."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract Entity Metadata from this cover page content:\n\n{context_text}",
                },
            ],
            temperature=0.0,
        )

        logger.info(f"Resolved Entity: {resp.canonical_id} ({resp.official_name})")
        return resp


# ---------------------------------------------------------------------------
# TableMelter
# ---------------------------------------------------------------------------

class TableMelter:
    def __init__(
        self,
        entity_id: str,
        entity_name_raw: str = "Unknown Entity",
        api_key: Optional[str] = None,
    ):
        self.entity_id = entity_id
        self.entity_name_raw = entity_name_raw
        self.api_key = api_key or settings.GROQ_API_KEY

        self.client = instructor.from_openai(
            OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key or "dummy_key",
            ),
            mode=instructor.Mode.JSON,
        )

    def melt(self, block: TableBlock) -> List[UFLRow]:
        content = block.content.strip()
        lines = content.split("\n")
        if not lines:
            return []

        # 0. Header Merging Heuristic
        header_rows: List[List[str]] = []
        data_start_idx = 0
        for i, line in enumerate(lines[:5]):
            if re.match(r"^\|?\s*:?-+:?\s*\|", line.strip()):
                data_start_idx = i + 1
                break
            header_rows.append([p.strip() for p in line.strip().strip("|").split("|")])

        merged_headers: List[str] = []
        if header_rows:
            num_cols = max(len(r) for r in header_rows)
            for col_idx in range(num_cols):
                parts = [
                    row[col_idx]
                    for row in header_rows
                    if col_idx < len(row) and row[col_idx]
                ]
                merged_headers.append(" ".join(parts).strip())

        # 1. Hierarchical Disambiguation
        hierarchy_lines: List[str] = []
        parent_stack: List[str] = []

        for line in lines[data_start_idx:]:
            if not line.strip() or re.match(r"^\|?\s*:?-+:?\s*\|", line.strip()):
                continue

            indent_match = re.match(r"\|?\s*((?:&nbsp;|\s)*)([^|]+)", line)
            if indent_match:
                indent_str = indent_match.group(1)
                metric_text = indent_match.group(2).strip()

                if not metric_text or metric_text.lower() in ["item", "value", "metric"]:
                    continue

                normalized_indent = indent_str.replace("&nbsp;", "  ")
                depth = len(normalized_indent) // 2
                parent_stack = parent_stack[:depth]

                full_name = (
                    " > ".join(parent_stack + [metric_text]) if parent_stack else metric_text
                )

                inner_content = line.strip().strip("|")
                parts = [p.strip() for p in inner_content.split("|")]

                is_parent = len(parts) > 1 and all(not p for p in parts[1:])
                if is_parent or len(parts) == 1:
                    parent_stack.append(metric_text)

                parts[0] = full_name
                if len(parts) < len(merged_headers):
                    parts.extend([""] * (len(merged_headers) - len(parts)))

                hierarchy_lines.append("| " + " | ".join(parts) + " |")

        # 2. DataFrame Conversion
        csv_header = "| " + " | ".join(merged_headers) + " |"
        csv_content = csv_header + "\n" + "\n".join(hierarchy_lines)

        try:
            df = pd.read_csv(
                io.StringIO(csv_content), sep=r"\s*\|\s*", engine="python"
            )
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            logger.error(f"Pandas parsing failed: {e}")
            return []

        if df.columns.empty:
            return []

        table_scale_factor = self._detect_scale(block)
        id_col = df.columns[0]
        period_cols = [c for c in df.columns[1:] if self._is_period_col(c)]
        if not period_cols:
            period_cols = [df.columns[1]] if len(df.columns) > 1 else []

        ufl_rows: List[UFLRow] = []
        for _, row in df.iterrows():
            metric_raw = str(row[id_col]).strip()
            if not metric_raw or metric_raw.lower() == "nan":
                continue

            metric_clean = re.sub(r"\s*\([\d\w]+\)", "", metric_raw).strip()
            row_scale_factor = table_scale_factor
            unit = "USD"
            if any(kw in metric_clean.lower() for kw in ["per share", "eps"]):
                row_scale_factor = 1.0
                unit = "USD/Share"
            elif any(kw in metric_clean.lower() for kw in ["ratio", "percentage", "margin"]):
                row_scale_factor = 1.0
                unit = "Ratio"

            for period in period_cols:
                raw_val = str(row[period]).strip()
                val, nuance = self._parse_numeric(raw_val)
                scaled_val = val * row_scale_factor if val is not None else None

                if "restated" in period.lower():
                    nuance = (nuance + " (Restated)") if nuance else "Restated"

                row_id_seed = f"{self.entity_id}_{metric_clean}_{period}_{block.id}_{scaled_val}"
                row_id = hashlib.md5(row_id_seed.encode()).hexdigest()

                # Map legacy period string to period_end (best-effort)
                period_end = period if re.search(r"20\d{2}", period) else None

                ufl_rows.append(
                    UFLRow(
                        row_id=row_id,
                        canonical_entity_id=self.entity_id,
                        entity_name_raw=self.entity_name_raw,
                        metric_name=metric_clean,
                        num_value=scaled_val,
                        grounding_quote=raw_val,        # verbatim cell text
                        unit_normalized=unit,
                        scale=row_scale_factor,
                        period_end=period_end,
                        doc_section=" > ".join(block.section_path),
                        source_chunk_id=block.id,
                        text_nuance=nuance,
                        related_entity_id=None,
                        # Table melts are deterministic — mark EXACT immediately
                        alignment_status="EXACT",
                        confidence_score=_CONFIDENCE_TABLE,
                    )
                )

        return ufl_rows

    async def normalize_headers_with_slm(self, headers: List[str]) -> Dict[str, str]:
        """Use SLM to normalise messy column headers to ISO-8601 dates."""
        prompt = (
            "Convert these financial table column headers into ISO 8601 dates "
            f"(YYYY-MM-DD) or standardised period names. Headers: {headers}"
        )

        class HeaderMap(BaseModel):
            mapping: Dict[str, str]

        resp = self.client.chat.completions.create(
            model=settings.SLM_MODEL_FAST,
            response_model=HeaderMap,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.mapping

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_scale(self, block: TableBlock) -> float:
        context = " ".join(block.section_path).lower() + " " + block.content.split("\n")[0].lower()
        if "millions" in context:
            return 1_000_000.0
        if "thousands" in context or "000s" in context:
            return 1_000.0
        return 1.0

    @staticmethod
    def _is_period_col(col_name: str) -> bool:
        return bool(re.search(r"20\d{2}", str(col_name)))

    @staticmethod
    def _parse_numeric(val_str: str) -> Tuple[Optional[float], Optional[str]]:
        s = val_str.strip().replace("&nbsp;", " ").strip()
        if s in ["—", "-", "–"]:
            return 0.0, "Dash treated as zero"

        is_neg_parens = s.startswith("(") and s.endswith(")")
        if is_neg_parens:
            s = s[1:-1]

        s = re.sub(r"\s*\([\d\w]+\)", "", s)
        s = re.sub(r"([0-9])[a-z]$", r"\1", s)
        s = s.replace(",", "").replace("$", "").strip()

        if not s or s.lower() in ["n/a", "nan"]:
            return None, None

        if is_neg_parens:
            s = "-" + s
            nuance: Optional[str] = "Negative (parentheses)"
        else:
            nuance = None

        try:
            return float(s), nuance
        except ValueError:
            return None, None


# ---------------------------------------------------------------------------
# TextSynthesizer  (Phase 3 upgrades)
# ---------------------------------------------------------------------------

class TextSynthesizer:
    def __init__(
        self,
        entity_id: str,
        entity_name_raw: str = "Unknown Entity",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.entity_id = entity_id
        self.entity_name_raw = entity_name_raw
        self.api_key = api_key or settings.GROQ_API_KEY

        self.client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=self.api_key or "dummy_key"),
            mode=instructor.Mode.JSON,
        )
        self.model = settings.SLM_MODEL_FAST
        self.prompt_template = (
            load_prompt("extract_financial_facts")
            or "You are a financial analyst. Extract facts from: {{text_content}}"
        )

        # PostHocAligner is constructed once per synthesizer instance
        self._aligner = PostHocAligner()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def extract_facts(
        self,
        block: TextBlock,
        context_str: str = "",
        model_name: Optional[str] = None,
    ) -> List[UFLRow]:
        """
        Extract UFLRows from a TextBlock.

        Phase-3 changes:
        1. Think-tag stripping applied after every raw SLM response.
        2. Multi-pass extraction for dense blocks (>_DENSE_BLOCK_CHARS chars)
           using asyncio.gather, then deduplication by (metric_name, period_end).
        3. PostHocAligner is applied to every extracted ScrapedFact before
           converting it to a UFLRow (Phase 4).
        """
        if len(block.content.strip()) < _MIN_BLOCK_CHARS:
            return []

        target_model = model_name or self.model
        filled_prompt = (
            self.prompt_template
            .replace("{{section_path}}", str(block.section_path))
            .replace("{{context_str}}", context_str)
            .replace("{{text_content}}", block.content)
        )

        # ----------------------------------------------------------------
        # Phase-3 (2): Multi-pass for dense blocks
        # ----------------------------------------------------------------
        is_dense = len(block.content.strip()) > _DENSE_BLOCK_CHARS
        n_passes = _MULTI_PASS_COUNT if is_dense else 1

        raw_fact_lists: List[List[ScrapedFact]] = await asyncio.gather(
            *[
                self._single_pass(filled_prompt, target_model)
                for _ in range(n_passes)
            ]
        )

        # ----------------------------------------------------------------
        # Phase-3 (3): Merge & deduplicate passes
        # ----------------------------------------------------------------
        merged_facts = self._merge_passes(raw_fact_lists)

        # ----------------------------------------------------------------
        # Phase-4: Post-hoc alignment
        # ----------------------------------------------------------------
        aligned_facts = self._aligner.align(
            merged_facts, 
            source_text=block.content,
            section_path=block.section_path
        )

        # ----------------------------------------------------------------
        # Convert ScrapedFact → UFLRow
        # ----------------------------------------------------------------
        ufl_rows: List[UFLRow] = []
        for fact in aligned_facts:
            if fact.confidence < settings.CONFIDENCE_TEXT_LOW:
                continue

            # Resolve final float value
            final_value: Optional[float] = None
            final_nuance = fact.text_nuance
            if isinstance(fact.num_value, (int, float)):
                final_value = float(fact.num_value)
            elif fact.num_value is None:
                final_value = None

            val_str = str(final_value) if final_value is not None else "None"
            row_id_seed = (
                f"{self.entity_id}_{fact.metric_name}_{fact.period_end}_{block.id}_{val_str}"
            )
            row_id = hashlib.md5(row_id_seed.encode()).hexdigest()

            # related_entity is a raw name string; it requires Entity Registry
            # resolution before it can become a canonical related_entity_id.
            # We preserve it in text_nuance to avoid storing a name string in
            # a field that the Code Agent treats as a canonical ID.
            extra_nuance = final_nuance or ""
            raw_related = fact.related_entity
            if raw_related:
                rel_note = f"Related entity (unresolved): {raw_related}"
                extra_nuance = f"{extra_nuance}; {rel_note}".lstrip("; ")

            ufl_rows.append(
                UFLRow(
                    row_id=row_id,
                    canonical_entity_id=self.entity_id,
                    entity_name_raw=self.entity_name_raw,
                    metric_name=fact.metric_name,
                    num_value=final_value,
                    grounding_quote=fact.grounding_quote,
                    unit_normalized=fact.unit_normalized,
                    scale=fact.scale,
                    period_start=fact.period_start,
                    period_end=fact.period_end,
                    period_type=fact.period_type,
                    doc_section=" > ".join(block.section_path),
                    source_chunk_id=block.id,
                    text_nuance=extra_nuance or None,
                    # related_entity_id left None until EntityResolver can map it
                    related_entity_id=None,
                    # Alignment fields now live directly on ScrapedFact
                    char_interval=fact.char_interval,
                    alignment_status=fact.alignment_status,
                    confidence_score=fact.alignment_confidence,
                )
            )

        return ufl_rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _single_pass(
        self, filled_prompt: str, target_model: str
    ) -> List[ScrapedFact]:
        """
        Run one SLM extraction pass.

        Phase-3 (1) — Think-tag stripping.

        We always use a raw OpenAI call so we can intercept the response text
        *before* it reaches Pydantic. This is the correct primary strategy:
        `instructor.Hooks` is not a real Instructor API and silently raises
        TypeError, so it cannot be used as a hook mechanism. Stripping think
        tags on the raw string before JSON parsing is both simpler and
        guaranteed to work regardless of instructor version.
        """
        try:
            raw_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key or "dummy_key",
            )
            schema = FactExtractionResponse.model_json_schema(mode='serialization')
            if "$defs" in schema and "ScrapedFact" in schema["$defs"]:
                schema["properties"]["facts"]["items"] = schema["$defs"]["ScrapedFact"]
                del schema["$defs"]
            schema_str = json.dumps(schema, indent=2)
            completion = raw_client.chat.completions.create(
                model=target_model,
                messages=[
                    {
                        "role": "system",
                        "content": filled_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Extract facts. Respond ONLY with a valid JSON object matching this exact schema:\n"
                            f"{schema_str}\n\n"
                            "Do NOT include any preamble, markdown fences, or reasoning."
                        ),
                    },
                ],
                temperature=0.0,
            )
            raw_text = completion.choices[0].message.content or ""

            # Phase-3 (1): strip <think>…</think> reasoning traces
            clean_text = _strip_think_tags(raw_text)
            # Strip markdown code fences if present
            clean_text = re.sub(r"```(?:json)?|```", "", clean_text).strip()

            data = json.loads(clean_text)
            return FactExtractionResponse(**data).facts

        except json.JSONDecodeError as e:
            logger.error(f"SLM returned non-JSON in extraction pass: {e}")
            # Second attempt: try instructor's structured output as a recovery path
            return await self._single_pass_instructor(filled_prompt, target_model)
        except Exception as e:
            logger.error(f"SLM extraction pass failed: {e}")
            return []

    async def _single_pass_instructor(
        self, filled_prompt: str, target_model: str
    ) -> List[ScrapedFact]:
        """
        Recovery path using Instructor's structured-output mode.
        Used only when the primary raw-JSON path fails (e.g. model returns
        natural language instead of JSON despite the system prompt).
        """
        try:
            resp = self.client.chat.completions.create(
                model=target_model,
                response_model=FactExtractionResponse,
                messages=[
                    {"role": "system", "content": filled_prompt},
                    {"role": "user", "content": "Extract facts."},
                ],
                temperature=0.0,
            )
            return resp.facts
        except Exception as e:
            logger.error(f"Instructor recovery extraction pass also failed: {e}")
            return []

    @staticmethod
    def _merge_passes(
        fact_lists: List[List[ScrapedFact]],
    ) -> List[ScrapedFact]:
        """
        Merge multiple extraction passes, deduplicating on
        (metric_name, period_start, period_end, related_entity).

        Using all three temporal fields plus related_entity prevents over-collapsing:
        • Facts with period_end=None (qualitative/undated) are still
          distinguished by metric_name + period_start.
        • Facts about the same metric across different periods are kept.
        • Relationships (edges) to different entities under the same metric are kept.

        First-pass wins on conflict (highest-priority pass is index 0).
        """
        seen: dict = {}
        merged: List[ScrapedFact] = []

        for facts in fact_lists:
            for fact in facts:
                key = (
                    fact.metric_name.strip().lower(),
                    fact.period_start,
                    fact.period_end,
                    (fact.related_entity.strip().lower() if fact.related_entity else None),
                )
                if key not in seen:
                    seen[key] = True
                    merged.append(fact)

        return merged


# ---------------------------------------------------------------------------
# PostHocAligner  (Phase 4)
# ---------------------------------------------------------------------------

class PostHocAligner:
    """
    Reverse-grounds extracted ScrapedFacts against the source text using
    a three-tier strategy (UFL_spec §2 Step D):

      Tier 1 (EXACT)   — direct + whitespace-normalised substring search.
      Tier 2 (PARTIAL) — key numeric token found verbatim in source.
      Tier 3 (FUZZY)   — sliding-window Counter token-intersection.
      Rejection        — UNALIGNED if no tier matches.

    After alignment the following fields are set on each ScrapedFact
    (declared as real model fields, not private attributes):
        char_interval       : Optional[Tuple[int, int]]
        alignment_status    : Literal["EXACT", "PARTIAL", "FUZZY", "UNALIGNED"]
        alignment_confidence: float
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def align(
        self,
        facts: List[ScrapedFact],
        source_text: str,
        section_path: Optional[List[str]] = None,
    ) -> List[ScrapedFact]:
        """
        Align all facts against `source_text` in a single batch pass.
        Returns the same list with alignment attributes populated.
        """
        if not facts or not source_text:
            for fact in facts:
                self._mark(fact, None, "UNALIGNED", _CONFIDENCE_UNALIGNED)
            return facts

        # Build token index once — maps (token_string, char_start) pairs
        token_index = self._build_token_index(source_text)
        source_tokens = [tok for tok, _ in token_index]

        # Prepare context for semantic verification
        context_blob = source_text.lower()
        if section_path:
            context_blob += " " + " ".join(section_path).lower()

        # Generic financial terms to ignore during semantic gating
        stop_words = {
            "net", "total", "of", "and", "the", "a", "in", "by", "per", "at", 
            "for", "is", "was", "were", "to", "from", "on", "with", "as", 
            "ratio", "value", "metric", "amount", "fact", "item", "percentage", "percent",
            "increase", "decrease", "growth", "change", "period", "year", "quarter",
            "fiscal", "ended", "current", "prior", "last"
        }

        for fact in facts:
            raw_quote = (fact.grounding_quote or "").strip()
            if not raw_quote:
                self._mark(fact, None, "UNALIGNED", _CONFIDENCE_UNALIGNED)
                continue

            # --- Lock 1: Quote Grounding (Verbatim matching) ---
            interval = self._tier1_exact(raw_quote, source_text)
            status = "EXACT"
            if interval is None:
                interval = self._tier2_partial(raw_quote, source_text)
                status = "PARTIAL"
            if interval is None:
                interval = self._tier3_fuzzy(raw_quote, source_text, token_index, source_tokens)
                status = "FUZZY"

            if interval is None:
                self._mark(fact, None, "UNALIGNED", _CONFIDENCE_UNALIGNED)
                continue

            # --- Lock 2: Semantic Grounding (Metric & Nuance validation) ---
            # Verify that the core identifying words of the metric/nuance actually appear 
            # in the context. This stops LLMs from assigning a real number to a fake metric.
            metric_text = fact.metric_name.lower()
            if fact.text_nuance:
                metric_text += " " + fact.text_nuance.lower()
            
            # Tokenize and filter (remove stop words AND numeric tokens)
            all_tokens = re.findall(r"\w+", metric_text)
            core_tokens = [t for t in all_tokens if t not in stop_words and not t.isdigit()]
            
            if core_tokens:
                # How many core tokens are present in the context?
                # Using set for fast exact word match
                context_tokens = set(re.findall(r"\w+", context_blob))
                matched_tokens = [t for t in core_tokens if t in context_tokens]
                match_ratio = len(matched_tokens) / len(core_tokens)
                
                # If we have core tokens but NONE are in the text, it is a semantic hallucination.
                # Threshold of 0.30 (similar to retriever's lexical overlap gate)
                if match_ratio < 0.30:
                    logger.warning(f"Semantic Rejection: Metric '{fact.metric_name}' tokens {core_tokens} not grounded in context.")
                    self._mark(fact, None, "UNALIGNED", _CONFIDENCE_UNALIGNED)
                    continue
                
                # Scale confidence based on semantic grounding
                semantic_multiplier = 0.5 + (match_ratio * 0.5) # [0.65 to 1.0]
            else:
                # Generic or purely numeric metric - assign very low confidence
                # as it is likely an orphan number or boilerplate extraction.
                semantic_multiplier = 0.1

            # --- Final Confidence Assignment ---
            base_confidence = _CONFIDENCE_TEXT_ALIGNED
            if status == "PARTIAL":
                base_confidence *= 0.90
            elif status == "FUZZY":
                base_confidence *= 0.75
            
            final_conf = base_confidence * semantic_multiplier
            self._mark(fact, interval, status, final_conf)

        return facts

    # ------------------------------------------------------------------
    # Tier 1: exact substring (direct + whitespace-normalised)
    # ------------------------------------------------------------------

    @staticmethod
    def _tier1_exact(raw: str, source: str) -> Optional[Tuple[int, int]]:
        """
        Two-step exact search:

        Step 1 — character-exact: straight str.find() — O(n), fastest.
        Step 2 — whitespace-normalised: collapses any whitespace run to a
                 single space before comparison, catching cases where the LLM
                 reproduced the value but normalised spacing (e.g. "$615\\n
                 million" vs "$615 million").

        The old difflib.ratio(raw, entire_source) approach has been removed.
        ratio() on the full source asymptotically approaches 0 as source
        grows, making it functionally dead code for any real 10-K document.
        """
        # Step 1: character-exact
        idx = source.find(raw)
        if idx != -1:
            return (idx, idx + len(raw))

        # Step 2: whitespace-normalised
        raw_norm = re.sub(r"\s+", " ", raw).strip()
        source_norm = re.sub(r"\s+", " ", source)
        idx_norm = source_norm.find(raw_norm)
        if idx_norm != -1:
            end = min(idx_norm + len(raw_norm), len(source))
            return (idx_norm, end)

        return None

    # ------------------------------------------------------------------
    # Tier 2: partial — key numeric token found verbatim
    # ------------------------------------------------------------------

    @staticmethod
    def _tier2_partial(raw: str, source: str) -> Optional[Tuple[int, int]]:
        """
        Extracts the most significant numeric token from grounding_quote
        (e.g. "615" from "$615 million") and searches for it verbatim in
        the source.  A PARTIAL match means the LLM found the right number
        but paraphrased the surrounding text.

        Only considers tokens of 2+ digits to avoid false matches against
        footnote numbers, page numbers, or single-digit values.
        """
        numeric_tokens = re.findall(r"\d[\d,.]+", raw)
        if not numeric_tokens:
            return None

        # Use the longest numeric token as the most discriminative signal
        key_num = max(numeric_tokens, key=len)
        if len(key_num) < 2:
            return None

        idx = source.find(key_num)
        if idx != -1:
            return (idx, idx + len(key_num))

        return None

    # ------------------------------------------------------------------
    # Tier 3: fuzzy sliding-window with Counter pre-check
    # ------------------------------------------------------------------

    def _tier3_fuzzy(
        self,
        raw: str,
        source: str,
        token_index: List[Tuple[str, int]],
        source_tokens: List[str],
    ) -> Optional[Tuple[int, int]]:
        """
        Slide a token window over the source and use Counter intersection
        as a rapid pre-check before committing to a difflib refinement.

        Char positions are derived from real token offsets captured by
        _build_token_index (re.finditer), NOT from
        len(" ".join(tokens[:i])) which produces wrong offsets whenever the
        source text has multi-char whitespace, tabs, or newlines.
        """
        raw_tokens = self._tokenize(raw)
        if not raw_tokens:
            return None

        raw_counter = collections.Counter(t.lower() for t in raw_tokens)
        n_raw = len(raw_tokens)
        n_source = len(token_index)
        window_size = min(max(n_raw, _FUZZY_WINDOW_TOKENS), n_source)

        if window_size == 0:
            return None

        best_ratio = 0.0
        best_interval: Optional[Tuple[int, int]] = None

        for i in range(n_source - window_size + 1):
            window_toks = source_tokens[i : i + window_size]
            window_counter = collections.Counter(t.lower() for t in window_toks)

            # Counter intersection pre-check (recall-based: what fraction of
            # raw tokens appear in the window?).
            # Jaccard was wrong here: with window_size >> n_raw, the union is
            # dominated by window-only tokens and ratio → 0 even for a perfect
            # match. Recall = intersection / raw_total is the correct signal.
            raw_total = sum(raw_counter.values())
            intersection = sum(
                min(raw_counter[t], window_counter[t]) for t in raw_counter
            )
            if raw_total == 0 or intersection / raw_total < _FUZZY_MIN_OVERLAP:
                continue

            # Refine with difflib recall (matching_chars / len(raw)).
            # We deliberately do NOT use ratio() here because ratio() =
            # 2M/(len_a+len_b) is artificially low when window >> raw,
            # even when raw is perfectly contained in the window.
            # Recall = M/len(raw) is the correct signal: "how much of raw
            # did we find inside this window?"
            window_text = " ".join(window_toks)
            matcher = difflib.SequenceMatcher(
                None, raw, window_text, autojunk=False
            )
            matching_chars = sum(
                size for _, _, size in matcher.get_matching_blocks()
            )
            recall = matching_chars / len(raw) if raw else 0.0
            if recall > best_ratio:
                best_ratio = recall
                char_start = token_index[i][1]
                last_tok, last_start = token_index[i + window_size - 1]
                char_end = last_start + len(last_tok)
                best_interval = (char_start, char_end)

        if best_ratio >= _FUZZY_MIN_RECALL and best_interval is not None:
            return best_interval

        return None

    # ------------------------------------------------------------------
    # Token index builder — with real char offsets (re.finditer, not findall)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_token_index(text: str) -> List[Tuple[str, int]]:
        """
        Tokenize `text` and return (token, char_start) pairs.

        Uses re.finditer so every token carries its exact character position
        in the original string — critical for producing correct char_interval
        values.  This replaces the previous approach of computing positions
        via len(" ".join(tokens[:i])), which was systematically wrong for any
        text containing newlines, tabs, or multi-space runs.
        """
        return [(m.group(), m.start()) for m in re.finditer(r"\w+|\S", text)]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Token strings only (no offsets). Used for grounding_quote."""
        return re.findall(r"\w+|\S", text)

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _mark(
        fact: ScrapedFact,
        interval: Optional[Tuple[int, int]],
        status: str,
        score: float,
    ) -> None:
        """
        Write alignment results directly onto the ScrapedFact model fields.
        No object.__setattr__ hack needed — the fields are declared on
        ScrapedFact with defaults, so normal assignment works in both
        Pydantic v1 and v2.
        """
        fact.char_interval = interval
        fact.alignment_status = status        # type: ignore[assignment]
        fact.alignment_confidence = score




# ---------------------------------------------------------------------------
# ContextIndexer  (unchanged from original)
# ---------------------------------------------------------------------------

class ContextIndexer:
    def __init__(
        self,
        db_path: str = settings.CHROMA_DB_PATH,
        embedding_fn: Optional[Any] = None,
    ):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_fn
        self.text_collection = self.client.get_or_create_collection(
            name="venra_text_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.schema_collection = self.client.get_or_create_collection(
            name="venra_metric_schema",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def index_blocks(self, blocks: List[DocBlock]):
        if not blocks:
            return
        documents = [b.content for b in blocks]
        ids = [b.id for b in blocks]
        metadatas = [
            {
                "block_type": b.block_type.value,
                "section_path": json.dumps(b.section_path),
                "page_num": b.page_num or 0,
            }
            for b in blocks
        ]
        self.text_collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(blocks)} blocks in ChromaDB.")

    def index_ufl_schema(self, rows: List[UFLRow]):
        if not rows:
            return
        unique_metrics: Dict[str, Dict[str, str]] = {}
        for r in rows:
            key = f"{r.canonical_entity_id}_{r.metric_name}"
            if key not in unique_metrics:
                unique_metrics[key] = {
                    "id": hashlib.md5(key.encode()).hexdigest(),
                    "metric_name": r.metric_name,
                    "entity_id": r.canonical_entity_id,
                }
        ids = [m["id"] for m in unique_metrics.values()]
        documents = [m["metric_name"] for m in unique_metrics.values()]
        metadatas = [
            {"entity_id": m["entity_id"], "metric_name": m["metric_name"]}
            for m in unique_metrics.values()
        ]
        self.schema_collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(unique_metrics)} unique metrics for schema mapping.")

    def update_chunk_linkage(self, chunk_id: str, row_ids: List[str]):
        if not row_ids:
            return
        self.text_collection.update(
            ids=[chunk_id],
            metadatas=[{"contains_rows": json.dumps(row_ids)}],
        )