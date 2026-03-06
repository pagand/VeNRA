"""
venra/synthesis.py
Knowledge synthesis: Entity resolution, Table melting, Text-to-Fact extraction,
and Post-Hoc Alignment verification.

Changes (Phase 3 & 4):
  Phase 3 — SLM Text-to-Fact Extraction Upgrades:
    • Think-tag stripping: regex removes <think>…</think> reasoning traces
      before the response reaches Pydantic / Instructor.
    • Multi-pass extraction: extract_facts() runs the SLM 2 times in
      parallel via asyncio.gather for dense text blocks, then deduplicates
      by metric_name + period overlap.

  Phase 4 — Post-Hoc Aligner (Hallucination Killer):
    • PostHocAligner class: reverse-grounds every ScrapedFact against the
      source TextBlock using three tiers:
        Tier 1 — difflib.SequenceMatcher exact-match pass.
        Tier 2 — sliding-window Counter token-intersection fuzzy pass.
      Sets char_interval, alignment_status, and confidence_score on each
      UFLRow before it enters the ledger.

  Bug Fixes (P0 & P1):
    P0-1 — Confidence filter now gates on fact.alignment_confidence (set by
            PostHocAligner) rather than fact.confidence (SLM self-report,
            always high and therefore useless as a gate).
    P0-2 — Schema-leakage guard: grounding_quotes that contain instructor
            system-prompt fragments (e.g. "genius expert", "json_schema") are
            dropped. Quotes exceeding _MAX_GROUNDING_QUOTE_CHARS are now
            TRUNCATED at a word boundary rather than dropped entirely.
    P0-3 — scale: null coercion: The SLM legitimately returns null for scale
            on qualitative facts. The JSON pre-processor now coerces null → 1.0
            before Pydantic validation, preventing a deterministic retry cascade
            that could stall the pipeline for 10+ minutes per block.
    P0-4 — max_tokens on raw API call: Added explicit max_tokens to prevent
            mid-JSON truncation (finish_reason='length') that triggers the
            retry/fallback cascade. Retries capped at 2 (was 5) — deterministic
            failures should not be retried 5 times with exponential backoff.
    P0-5 — Chunk failure vs zero-facts distinction: extract_facts now returns
            a dedicated sentinel so callers can distinguish "SLM returned
            no facts" from "SLM call completely failed for this chunk".
    P1-1 — Upstream boilerplate filter: _is_boilerplate_block() detects SEC
            cover-page checkboxes, URL-only blocks, timestamp-only lines, and
            content with no financial signal. Such blocks are skipped entirely
            before any SLM call is made.
    P1-2 — Prompt updated (PROMPTS.md) to enforce ≤15-word minimal
            grounding_quote snippets, with BAD/GOOD examples and explicit
            boilerplate-rejection rule.
    P1-3 — Useful-fact filter: UFLRows where only metric_name is populated
            (all value/period/entity/nuance fields are None) are dropped
            post-extraction. Qualitative facts with a related_entity, period,
            or meaningful text_nuance are retained as they support retrieval.
    P1-4 — Previous-context XML wrapping: The [Previous Context: ...] prefix
            is reformatted as <previous_context>...</previous_context> in the
            prompt so the SLM clearly distinguishes context from source text.
    P1-5 — Confidence dead-zone fix: Facts that pass both Lock 1 (quote
            grounding) and Lock 2 (semantic match ≥ 30%) now receive a minimum
            confidence of CONFIDENCE_TEXT_LOW + ε, eliminating the dead zone
            where 30-85% semantic match on an EXACT span still fell below the
            acceptance gate.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import difflib
import hashlib
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import instructor
import chromadb
import openai
import pandas as pd
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

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
_MULTI_PASS_COUNT: int = 2

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

# Max tokens to request from the SLM — prevents mid-JSON truncation.
# Groq llama-3.1-8b-instant hard cap is 8192; 4096 leaves headroom for
# a dense block response without triggering finish_reason='length'.
_SLM_MAX_OUTPUT_TOKENS: int = 4096

# ---------------------------------------------------------------------------
# P1-1: Boilerplate detection constants
# ---------------------------------------------------------------------------

_BOILERPLATE_SECTION_SIGNALS: frozenset = frozenset([
    "indicate by check mark",
    "check mark whether",
    "large accelerated filer",
    "accelerated filer",
    "emerging growth company",
    "smaller reporting company",
    "shell company",
    "registrant's telephone",
    "exchange act",
    "incorporated by reference",
])

# Regex: block content that contains ONLY a URL, a timestamp, or a bare page ref
_BOILERPLATE_CONTENT_RE = re.compile(
    r"^(https?://\S+\s*)+$"              # URL-only
    r"|^\s*\d{1,2}/\d{1,2}/\d{2,4}"     # timestamp (e.g. "1/28/26, 3:13 PM …")
    r"\s*,\s*\d{1,2}:\d{2}\s*(AM|PM)"
    r"(\s+\S+)?\s*$",
    re.IGNORECASE,
)

# Presence of ANY of these signals rescues a block from boilerplate rejection.
# NOTE: "interest" intentionally omitted — too common in non-financial text
# (e.g. "in the interest of", cybersecurity "interest groups"). Add only
# terms that are unambiguously financial in SEC filings.
_FINANCIAL_SIGNAL_RE = re.compile(
    r"\$[\d,]|(?<!\w)revenue(?!\w)|(?<!\w)sales(?!\w)|net income|gross profit"
    r"|(?<!\w)loss(?!\w)|operating expense|capital expenditure"
    r"|(?<!\w)asset(?!\w)|liabilit|stockholder|shareholder"
    r"|(?<!\w)equity(?!\w)|long.term debt|short.term debt"
    r"|(?<!\w)cash(?!\w)|(?<!\w)dividend(?!\w)|earnings per share"
    r"|(?<!\w)margin(?!\w)|goodwill|amortiz|depreciat|impairment"
    r"|(?<!\w)principal(?!\w)",
    re.IGNORECASE,
)

# Checkbox unicode characters that indicate SEC form content
_CHECKBOX_CHARS = frozenset("☒☐✓✗")

# ---------------------------------------------------------------------------
# P0-2 / P1-3: Schema-leakage and quote-length constants
#
# Reduced from 200 → 120 chars (~15 words as stated in spec).
# Quotes exceeding this are now TRUNCATED at the last word boundary rather
# than dropped, so the fact still participates in alignment (with reduced
# quote precision) rather than being silently discarded.
# ---------------------------------------------------------------------------

_SCHEMA_LEAK_SIGNALS: Tuple[str, ...] = (
    "json_schema",
    "parsed objects",
    "genius expert",
    "your task is to",
    "json that match",
    "match the following",
    "following json_schema",
    "provide the parsed",
)

# Hard cap for grounding quotes: ≤15 words ≈ ≤120 chars.
# Quotes longer than this are truncated (not dropped) unless they contain
# schema-leak signals, which ARE dropped.
_MAX_GROUNDING_QUOTE_CHARS: int = 120

# Retry configuration — increased to 4 to ensure cycling through multiple keys 
# if one organization is restricted.
_MAX_RETRIES: int = 4

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.BadRequestError,      # Added to handle organization_restricted failover
    TimeoutError,
    ValidationError,
    json.JSONDecodeError,
)

# Regex to detect the [Previous Context: ...] prefix injected by the chunker
_PREV_CONTEXT_RE = re.compile(r"^\[Previous Context:(.*?)\]\n\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Utility: think-tag stripping
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> reasoning traces from raw LLM output."""
    return _THINK_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# P1-4: Previous-context XML reformatter
#
# The chunker injects "[Previous Context: <tail>]\n\n" at the start of each
# TextBlock. Sending this verbatim in text_content confuses the SLM: it may
# treat the context tail as source text and quote it back in grounding_quote.
#
# Fix: extract the previous-context prefix and inject it as an XML-tagged
# element so the SLM clearly separates context from source.
# ---------------------------------------------------------------------------

def _reformat_previous_context(content: str) -> Tuple[str, str]:
    """
    Split a TextBlock's content into (previous_context, source_text).

    If no [Previous Context: ...] prefix is present, returns ("", content).
    The caller is responsible for injecting both into the prompt template.
    """
    m = _PREV_CONTEXT_RE.match(content)
    if m:
        prev_ctx = m.group(1).strip()
        source = content[m.end():]
        return prev_ctx, source
    return "", content


# ---------------------------------------------------------------------------
# P1-1: Boilerplate block filter
# ---------------------------------------------------------------------------

def _is_boilerplate_block(block: TextBlock) -> bool:
    """
    Return True if this TextBlock should be skipped before any SLM call.

    A block is boilerplate when it carries no extractable financial or
    regulatory fact.  Two fast heuristics are applied in order:

    1. Section path signal — matches known SEC cover-page headings.
    2. Content signal — URL-only blocks, timestamp-only lines, checkbox forms.

    The _FINANCIAL_SIGNAL_RE rescue clause ensures we never drop a block
    that explicitly mentions dollar amounts with digits, revenue, sales, etc.
    The regex has been tightened (vs original) to require more specific signals;
    bare "interest" or "cost" alone no longer rescue a block since these appear
    in non-financial boilerplate.
    """
    content = block.content

    # Fast rescue: any SPECIFIC financial signal keeps the block alive
    has_financial = bool(_FINANCIAL_SIGNAL_RE.search(content))
    if has_financial:
        return False

    # 1. Section path boilerplate signal
    path_lower = " ".join(block.section_path).lower()
    if any(signal in path_lower for signal in _BOILERPLATE_SECTION_SIGNALS):
        logger.debug(f"Boilerplate (section path): {block.section_path}")
        return True

    # 2a. URL-only or timestamp-only content
    stripped = content.strip()
    if _BOILERPLATE_CONTENT_RE.match(stripped):
        logger.debug(f"Boilerplate (URL/timestamp pattern): {stripped[:60]!r}")
        return True

    # 2b. Checkbox-heavy content (SEC filer-status forms)
    checkbox_count = sum(1 for ch in content if ch in _CHECKBOX_CHARS)
    if checkbox_count >= 2:
        logger.debug(f"Boilerplate (checkbox count={checkbox_count}): {stripped[:60]!r}")
        return True

    # 2c. Very short block with no financial substance
    if len(stripped) < 10:
        logger.debug(f"Boilerplate (too short, no financial signal): {stripped!r}")
        return True

    return False


# ---------------------------------------------------------------------------
# P0-2 / P0-3: JSON pre-processor
#
# Applied to the raw parsed JSON dict before Pydantic validation.
# Two coercions:
#   1. scale: null → 1.0  (the SLM returns null for qualitative facts;
#      Pydantic rejects null on a float field, causing a deterministic
#      ValidationError → retry cascade → pipeline stall).
#   2. grounding_quote truncation: quotes exceeding _MAX_GROUNDING_QUOTE_CHARS
#      are truncated at a word boundary rather than dropped.  If a quote
#      contains schema-leak signals it is set to "" so the aligner will
#      mark the fact UNALIGNED and it will be filtered by Gate 6.
# ---------------------------------------------------------------------------

def _sanitize_fact_dict(fact: dict) -> dict:
    """
    In-place sanitise a single fact dict before Pydantic construction.

    Returns the (mutated) dict.
    """
    # 1. Coerce null scale → 1.0
    if fact.get("scale") is None:
        fact["scale"] = 1.0

    # 2. Grounding-quote sanitation
    gq: Optional[str] = fact.get("grounding_quote")
    if gq:
        gq_lower = gq.lower()
        # Drop if contains schema-leak signals (sets to "" → UNALIGNED)
        if any(signal in gq_lower for signal in _SCHEMA_LEAK_SIGNALS):
            logger.warning(
                f"Schema leakage in grounding_quote (first 80 chars): {gq[:80]!r}"
            )
            fact["grounding_quote"] = ""
        elif len(gq) > _MAX_GROUNDING_QUOTE_CHARS:
            # Truncate at last word boundary within limit
            truncated = gq[:_MAX_GROUNDING_QUOTE_CHARS]
            last_space = truncated.rfind(" ")
            if last_space > _MAX_GROUNDING_QUOTE_CHARS // 2:
                truncated = truncated[:last_space]
            logger.debug(
                f"grounding_quote truncated from {len(gq)} → {len(truncated)} chars: "
                f"{truncated!r}"
            )
            fact["grounding_quote"] = truncated

    return fact


def _preprocess_json(data: dict) -> dict:
    """Apply _sanitize_fact_dict to every fact in a raw parsed JSON dict."""
    if "facts" in data and isinstance(data["facts"], list):
        data["facts"] = [
            _sanitize_fact_dict(f) if isinstance(f, dict) else f
            for f in data["facts"]
        ]
    return data


# ---------------------------------------------------------------------------
# P1-3: Useful-fact filter
#
# A UFLRow is "useful" if it carries at least one signal field beyond just a
# metric_name.  A row with ONLY metric_name (num_value=None, no period, no
# related entity, no nuance) cannot support numeric analysis or retrieval and
# is dropped here.
#
# Retained cases (examples):
#   • num_value is set                → numeric fact (always keep)
#   • period_end or period_start set  → time-anchored qualitative fact (keep)
#   • text_nuance is meaningful       → e.g. formula, condition, compliance
#   • grounding_quote has a number    → partial numeric signal (keep)
#   • related_entity was set on fact  → graph-edge fact (keep via nuance field)
#
# Dropped cases (examples):
#   • "Cybersecurity Program: None"  — no value, period, nuance, or entity
#   • "Board of Directors Meetings: None" — ditto
# ---------------------------------------------------------------------------

_MEANINGLESS_NUANCE_RE = re.compile(
    r"^(qualitative|n/?a|none|null|N/A)$",
    re.IGNORECASE,
)

# Minimum length for text_nuance to be considered meaningful
_MIN_NUANCE_LEN: int = 3


def _is_useful_row(row: UFLRow) -> bool:
    """
    Return True if the UFLRow carries enough signal to be worth storing.
    """
    # num_value is the strongest signal
    if row.num_value is not None:
        return True

    # Time-anchored facts support period-based retrieval
    if row.period_end or row.period_start:
        return True

    # Meaningful nuance (formula, compliance note, condition, entity mention)
    nuance = row.text_nuance or ""
    if len(nuance) >= _MIN_NUANCE_LEN and not _MEANINGLESS_NUANCE_RE.match(nuance.strip()):
        return True

    # Grounding quote contains a numeric token → partial value signal
    gq = row.grounding_quote or ""
    if re.search(r"\d", gq):
        return True
    
    # some information exist in addition to 
    if (row.metric_name) and (row.unit_normalized or row.unit_normalized):
        return True

    return False


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
        # BUG FIX: Use only internal pipes as delimiters to prevent ghost columns 
        # (parts before first pipe or after last pipe being interpreted as empty cols)
        csv_header = "|".join(merged_headers)
        # hierarchy_lines Segments were built with leading/trailing pipes; strip them
        clean_lines = [l.strip().strip("|") for l in hierarchy_lines]
        csv_content = csv_header + "\n" + "\n".join(clean_lines)

        try:
            # dtype=str prevents auto-conversion of years to floats/ints
            df = pd.read_csv(
                io.StringIO(csv_content), sep="|", engine="python", dtype=str
            )
            
            # If the first column header was empty in markdown, it becomes "Unnamed: 0"
            if df.columns[0].startswith("Unnamed"):
                df.rename(columns={df.columns[0]: "Row_Label"}, inplace=True)
            
            # Clean column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            logger.error(f"Pandas parsing failed: {e}")
            return []

        if df.columns.empty:
            return []

        table_scale_factor = self._detect_scale(block)
        id_col = df.columns[0]

        # BUG C FIX: Dynamic Table Orientation Detection (Performance Tables)
        # If the first column contains "Entity" signals (Corporation, Index, Inc)
        # and headers are metrics ($ 100), we pivot the naming.
        entity_keywords = ["corp", "inc", "index", "company", "group", "peer", "industry"]
        is_entity_first_col = any(
            any(kw in str(val).lower() for kw in entity_keywords)
            for val in df[id_col].head(10)
        )
        
        # Select period columns: All columns EXCEPT the ID column
        period_cols = [c for i, c in enumerate(df.columns) if i > 0]

        ufl_rows: List[UFLRow] = []
        for _, row in df.iterrows():
            metric_raw = str(row[id_col]).strip()
            # Skip empty rows or generic 'nan'
            if not metric_raw or metric_raw.lower() in ["nan", "null", "none"]:
                continue

            # P1-2: Footnote removal
            metric_clean = re.sub(r"\s*\([\w\d]+\)", "", metric_raw).strip()

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
                
                # Keep the row if we have a value OR if it's a known placeholder
                # or if the header itself is a year (qualitative fact)
                is_placeholder = raw_val.lower() in ["n/a", "—", "-", "nan", ""]
                is_year_header = bool(re.search(r"20\d{2}", str(period)))
                
                if val is None and not is_placeholder and not is_year_header:
                    continue
                    
                scaled_val = val * row_scale_factor if val is not None else None

                if "restated" in str(period).lower():
                    nuance = (nuance + " (Restated)") if nuance else "Restated"

                # BUG C FIX: Construct semantic metric name for performance tables
                actual_metric_name = metric_clean
                related_entity = None
                if is_entity_first_col:
                    related_entity = metric_clean
                    p_lower = str(period).lower()
                    if "$" in p_lower or "return" in p_lower or "value" in p_lower:
                        actual_metric_name = f"{period} of {metric_clean}"
                    elif not re.search(r"20\d{2}", str(period)):
                        actual_metric_name = f"{metric_clean}: {period}"
                    else:
                        actual_metric_name = metric_clean

                row_id_seed = f"{self.entity_id}_{actual_metric_name}_{period}_{block.id}_{scaled_val}"
                row_id = hashlib.md5(row_id_seed.encode()).hexdigest()

                period_end = str(period) if re.search(r"20\d{2}", str(period)) else None

                ufl_rows.append(
                    UFLRow(
                        row_id=row_id,
                        canonical_entity_id=self.entity_id,
                        entity_name_raw=self.entity_name_raw,
                        metric_name=actual_metric_name,
                        num_value=scaled_val,
                        grounding_quote=raw_val,
                        unit_normalized=unit,
                        scale=row_scale_factor,
                        period_end=period_end,
                        doc_section=" > ".join(block.section_path),
                        source_chunk_id=block.id,
                        text_nuance=nuance,
                        related_entity_id=related_entity,
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

        s = re.sub(r"\s*\(\d+[\w]*\)", "", s)
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
# TextSynthesizer  (Phase 3 upgrades + P0/P1 bug fixes)
# ---------------------------------------------------------------------------

class TextSynthesizer:
    def __init__(
        self,
        entity_id: str,
        entity_name_raw: str = "Unknown Entity",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        max_retries: int = _MAX_RETRIES,
    ):
        self.entity_id = entity_id
        self.entity_name_raw = entity_name_raw
        self.api_key = api_key or settings.GROQ_API_KEY
        self.max_retries = max_retries

        keys = settings.GROQ_KEYS
        if not keys:
            keys = [self.api_key or "dummy_key"]

        raw_clients = []
        instructor_clients = []
        for k in keys:
            raw = AsyncOpenAI(base_url=base_url, api_key=k, timeout=30.0)
            raw_clients.append(raw)
            instructor_clients.append(instructor.from_openai(raw, mode=instructor.Mode.JSON))

        self._client_cycle = itertools.cycle(zip(instructor_clients, raw_clients))

        self.model = settings.SLM_MODEL_FAST
        self.prompt_template = (
            load_prompt("extract_financial_facts")
            or "You are a financial analyst. Extract facts from: {{text_content}}"
        )

        self._aligner = PostHocAligner()

    def _next_clients(self) -> Tuple[instructor.AsyncInstructor, AsyncOpenAI]:
        return next(self._client_cycle)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def extract_facts_with_proposals(
        self,
        block: TextBlock,
        context_str: str = "",
        model_name: Optional[str] = None,
    ) -> Tuple[List[UFLRow], List[ScrapedFact], bool]:
        """
        Extract UFLRows from a TextBlock.

        Returns:
            (accepted_rows, raw_proposals, extraction_failed)

        extraction_failed=True means the SLM call itself failed (all retries
        exhausted).  This is distinct from extraction_failed=False with an
        empty accepted_rows list, which means the SLM ran but found no facts
        worth keeping (or the block was boilerplate-filtered).

        Callers should log / surface the distinction so pipeline monitoring
        can identify problematic chunks rather than silently treating them as
        empty.

        Gate order (fail-fast, cheapest first):
          1. Minimum length check (pure string len — O(1))
          2. P1-1: Boilerplate detection (regex + string ops — O(k))
          3. SLM extraction (API call — expensive)
          4. P0-2: Schema-leakage + quote truncation (via _preprocess_json)
          5. PostHocAligner (sliding window — O(n·m))
          6. P0-1: alignment_confidence filter (float compare — O(1))
          7. P1-3: Useful-fact filter (field presence check — O(1))
        """
        # Gate 1: minimum length
        if len(block.content.strip()) < _MIN_BLOCK_CHARS:
            return [], [], False

        # Gate 2 (P1-1): boilerplate detection — skip SLM call entirely
        if _is_boilerplate_block(block):
            logger.info(
                f"Boilerplate block skipped (no SLM call): {block.section_path}"
            )
            return [], [], False

        target_model = model_name or self.model

        # P1-4: Reformat previous-context prefix into XML tag so the SLM
        # clearly distinguishes injected context from source text.
        prev_ctx, source_text = _reformat_previous_context(block.content)
        prev_ctx_xml = (
            f"<previous_context>\n{prev_ctx}\n</previous_context>\n\n"
            if prev_ctx else ""
        )

        filled_prompt = (
            self.prompt_template
            .replace("{{section_path}}", str(block.section_path))
            .replace("{{context_str}}", context_str)
            .replace("{{text_content}}", prev_ctx_xml + source_text)
        )

        # ----------------------------------------------------------------
        # Phase-3 (2): Multi-pass for dense blocks
        # ----------------------------------------------------------------
        is_dense = len(block.content.strip()) > _DENSE_BLOCK_CHARS
        n_passes = _MULTI_PASS_COUNT if is_dense else 1

        raw_fact_lists: List[List[ScrapedFact]]
        extraction_failed = False

        raw_fact_lists, extraction_failed = await self._gather_passes(
            filled_prompt, target_model, n_passes, block.id
        )

        if extraction_failed:
            # Return early; caller should log this as a chunk failure
            return [], [], True

        # ----------------------------------------------------------------
        # Phase-3 (3): Merge & deduplicate passes
        # ----------------------------------------------------------------
        merged_facts = self._merge_passes(raw_fact_lists)

        # ----------------------------------------------------------------
        # Gate 5: Post-hoc alignment
        # ----------------------------------------------------------------
        aligned_facts = self._aligner.align(
            merged_facts,
            source_text=block.content,
            section_path=block.section_path,
        )

        # ----------------------------------------------------------------
        # Convert ScrapedFact → UFLRow
        # ----------------------------------------------------------------
        ufl_rows: List[UFLRow] = []
        for fact in aligned_facts:

            # Gate 6 (P0-1): Filter on alignment_confidence from PostHocAligner,
            # NOT the SLM's self-reported confidence (which is always high).
            if fact.alignment_confidence < settings.CONFIDENCE_TEXT_LOW:
                logger.debug(
                    f"Dropped fact '{fact.metric_name}' — alignment_confidence "
                    f"{fact.alignment_confidence:.3f} < threshold "
                    f"{settings.CONFIDENCE_TEXT_LOW} "
                    f"(alignment_status={fact.alignment_status}, "
                    f"SLM self-confidence={fact.confidence:.2f})"
                )
                continue

            final_value: Optional[float] = None
            final_nuance = fact.text_nuance
            if isinstance(fact.num_value, (int, float)):
                final_value = float(fact.num_value)

            val_str = str(final_value) if final_value is not None else "None"
            row_id_seed = (
                f"{self.entity_id}_{fact.metric_name}_{fact.period_end}_{block.id}_{val_str}"
            )
            row_id = hashlib.md5(row_id_seed.encode()).hexdigest()

            extra_nuance = final_nuance or ""
            raw_related = fact.related_entity
            if raw_related:
                rel_note = f"Related entity (unresolved): {raw_related}"
                extra_nuance = f"{extra_nuance}; {rel_note}".lstrip("; ")

            candidate = UFLRow(
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
                related_entity_id=None,
                char_interval=fact.char_interval,
                alignment_status=fact.alignment_status,
                confidence_score=fact.alignment_confidence,
            )

            # Gate 7 (P1-3): Drop rows that carry no actionable signal.
            # A row with only metric_name set cannot support numeric analysis
            # or meaningful retrieval expansion.
            if not _is_useful_row(candidate):
                logger.debug(
                    f"Dropped low-signal row '{fact.metric_name}' — "
                    f"no num_value, period, nuance, unit, related_entity, or numeric grounding_quote."
                )
                continue

            ufl_rows.append(candidate)

        return ufl_rows, merged_facts, False

    async def extract_facts(
        self,
        block: TextBlock,
        context_str: str = "",
        model_name: Optional[str] = None,
    ) -> List[UFLRow]:
        """
        Backward-compatible wrapper around extract_facts_with_proposals.

        Logs a WARNING if the extraction failed (vs returning zero facts),
        so callers that don't inspect the third return value still get
        visibility into chunk-level failures.
        """
        accepted_rows, _, extraction_failed = await self.extract_facts_with_proposals(
            block=block, context_str=context_str, model_name=model_name
        )
        if extraction_failed:
            logger.warning(
                f"CHUNK EXTRACTION FAILED (all {self.max_retries} retries exhausted) "
                f"for block {block.id!r} — section: {block.section_path}. "
                f"This chunk will have NO facts in the ledger. "
                f"Consider re-running or investigating the block content."
            )
        return accepted_rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _gather_passes(
        self,
        filled_prompt: str,
        target_model: str,
        n_passes: int,
        block_id: str,
    ) -> Tuple[List[List[ScrapedFact]], bool]:
        """
        Run n_passes SLM extraction passes in parallel.

        Returns (fact_lists, extraction_failed).
        extraction_failed=True only if ALL passes failed completely.
        Partial pass failures (some passes succeed) are treated as non-fatal;
        the successful pass results are used.
        """
        results = await asyncio.gather(
            *[self._single_pass(filled_prompt, target_model) for _ in range(n_passes)],
            return_exceptions=True,
        )

        fact_lists: List[List[ScrapedFact]] = []
        all_failed = True

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Pass {i+1}/{n_passes} completely failed for block {block_id!r}: {result}"
                )
            elif result is None:
                # _single_pass returns [] on total failure, None signals fatal
                logger.error(
                    f"Pass {i+1}/{n_passes} returned None for block {block_id!r}"
                )
            else:
                fact_lists.append(result)
                all_failed = False

        return fact_lists, (all_failed and n_passes > 0 and len(fact_lists) == 0)

    async def _single_pass(
        self, filled_prompt: str, target_model: str
    ) -> List[ScrapedFact]:
        """
        Run one SLM extraction pass with resilience.

        Key fixes vs original:
          • max_tokens=_SLM_MAX_OUTPUT_TOKENS on the raw API call prevents
            mid-JSON truncation (finish_reason='length').
          • _preprocess_json() coerces scale: null → 1.0 before Pydantic,
            eliminating the deterministic ValidationError retry cascade.
          • Retries capped at _MAX_RETRIES=2; deterministic failures should
            not burn 90-second backoff slots.
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_random_exponential(min=2, max=30),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            ):
                with attempt:
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

                    completion = await raw_client.chat.completions.create(
                        model=target_model,
                        max_tokens=_SLM_MAX_OUTPUT_TOKENS,   # P0-4: prevent truncation
                        messages=[
                            {"role": "system", "content": filled_prompt},
                            {
                                "role": "user",
                                "content": (
                                    "Extract facts. Respond ONLY with a valid JSON object "
                                    "matching this exact schema:\n"
                                    f"{schema_str}\n\n"
                                    "Do NOT include any preamble, markdown fences, or reasoning."
                                ),
                            },
                        ],
                        temperature=0.0,
                    )

                    finish_reason = completion.choices[0].finish_reason
                    if finish_reason == "length":
                        logger.warning(
                            f"SLM hit max_tokens limit (finish_reason='length') — "
                            f"response may be truncated JSON. "
                            f"Block may be too dense; consider splitting."
                        )

                    raw_text = completion.choices[0].message.content or ""

                    clean_text = _strip_think_tags(raw_text)
                    clean_text = re.sub(r"```(?:json)?|```", "", clean_text).strip()

                    try:
                        data = json.loads(clean_text)
                        data = _preprocess_json(data)   # P0-3: coerce scale + truncate quotes
                        return FactExtractionResponse(**data).facts
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.warning(
                            f"JSON/Validation error (attempt "
                            f"{attempt.retry_state.attempt_number}): {e}. "
                            f"Falling back to instructor for this attempt."
                        )
                        try:
                            resp = await instructor_client.chat.completions.create(
                                model=target_model,
                                max_tokens=_SLM_MAX_OUTPUT_TOKENS,
                                response_model=FactExtractionResponse,
                                messages=[
                                    {"role": "system", "content": filled_prompt},
                                    {"role": "user", "content": "Extract facts."},
                                ],
                                temperature=0.0,
                            )
                            return resp.facts
                        except Exception as inner_e:
                            logger.error(f"Instructor fallback also failed: {inner_e}")
                            raise e

        except Exception as e:
            logger.critical(
                f"SLM extraction completely failed after {self.max_retries} attempts: {e}"
            )
            return []

    async def _single_pass_instructor(
        self, filled_prompt: str, target_model: str
    ) -> List[ScrapedFact]:
        """Recovery path using Instructor's structured-output mode. (Legacy)"""
        instructor_client, _ = self._next_clients()
        try:
            resp = await instructor_client.chat.completions.create(
                model=target_model,
                max_tokens=_SLM_MAX_OUTPUT_TOKENS,
                response_model=FactExtractionResponse,
                messages=[
                    {"role": "system", "content": filled_prompt},
                    {"role": "user", "content": "Extract facts."},
                ],
                temperature=0.0,
            )
            return resp.facts
        except Exception as e:
            logger.error(f"Standalone Instructor recovery pass failed: {e}")
            return []

    @staticmethod
    def _merge_passes(
        fact_lists: List[List[ScrapedFact]],
    ) -> List[ScrapedFact]:
        """
        Merge multiple extraction passes, deduplicating on
        (metric_name, period_start, period_end, related_entity).

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

    P1-5: Confidence dead-zone fix
    --------------------------------
    The original formula `base * (0.5 + 0.5 * match_ratio)` created a dead
    zone: facts with 30-85% semantic match on an EXACT span produced final
    confidence 0.595–0.693, straddling the 0.6 acceptance threshold unpredictably.

    Fix: facts that pass BOTH Lock 1 (quote grounded) AND Lock 2 (semantic ≥ 30%)
    receive `max(computed_confidence, CONFIDENCE_TEXT_LOW + 0.01)`.
    This eliminates silent drops of legitimately grounded facts while keeping
    the semantic gate itself strict (the 30% floor still rejects phantom metrics).
    """

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

        token_index = self._build_token_index(source_text)
        source_tokens = [tok for tok, _ in token_index]

        context_blob = source_text.lower()
        if section_path:
            context_blob += " " + " ".join(section_path).lower()

        stop_words = {
            "net", "total", "of", "and", "the", "a", "in", "by", "per", "at",
            "for", "is", "was", "were", "to", "from", "on", "with", "as",
            "ratio", "value", "metric", "amount", "fact", "item", "percentage", "percent",
            "increase", "decrease", "growth", "change", "period", "year", "quarter",
            "fiscal", "ended", "current", "prior", "last",
        }

        # Minimum confidence floor for facts that pass both locks (P1-5).
        # Ensures no dead zone between the semantic rejection floor (30%)
        # and the acceptance gate (CONFIDENCE_TEXT_LOW).
        confidence_floor = settings.CONFIDENCE_TEXT_LOW + 0.01

        for fact in facts:
            raw_quote = (fact.grounding_quote or "").strip()
            if not raw_quote:
                self._mark(fact, None, "UNALIGNED", _CONFIDENCE_UNALIGNED)
                continue

            # --- Lock 1: Quote Grounding ---
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

            # --- Lock 2: Semantic Grounding ---
            metric_text = fact.metric_name.lower()
            if fact.text_nuance:
                metric_text += " " + fact.text_nuance.lower()

            all_tokens = re.findall(r"\w+", metric_text)
            core_tokens = [t for t in all_tokens if t not in stop_words and not t.isdigit()]

            if core_tokens:
                # P1-6: Improved context normalization (I.R.S. -> IRS)
                normalized_context = re.sub(r"(?<=[a-zA-Z])\.(?=[a-zA-Z])", "", context_blob)
                context_tokens = set(re.findall(r"\w+", normalized_context))
                
                # Basic pluralization/stemming fallback for 's' suffix
                matched_tokens = []
                for t in core_tokens:
                    if t in context_tokens:
                        matched_tokens.append(t)
                    elif t.endswith("s") and t[:-1] in context_tokens:
                        matched_tokens.append(t)
                    elif t + "s" in context_tokens:
                        matched_tokens.append(t)

                match_ratio = len(matched_tokens) / len(core_tokens)

                if match_ratio < 0.30:
                    logger.warning(
                        f"Semantic Rejection: Metric '{fact.metric_name}' tokens "
                        f"{core_tokens} not grounded in context."
                    )
                    self._mark(fact, None, "UNALIGNED", _CONFIDENCE_UNALIGNED)
                    continue

                semantic_multiplier = 0.5 + (match_ratio * 0.5)
            else:
                # No meaningful tokens → apply penalty but don't auto-reject
                semantic_multiplier = 0.1

            base_confidence = _CONFIDENCE_TEXT_ALIGNED
            if status == "PARTIAL":
                base_confidence *= 0.90
            elif status == "FUZZY":
                base_confidence *= 0.75

            computed_conf = base_confidence * semantic_multiplier

            # P1-5: apply floor so passing facts never fall below threshold
            final_conf = max(computed_conf, confidence_floor) if semantic_multiplier > 0.1 else computed_conf

            self._mark(fact, interval, status, final_conf)

        return facts

    @staticmethod
    def _tier1_exact(raw: str, source: str) -> Optional[Tuple[int, int]]:
        idx = source.find(raw)
        if idx != -1:
            return (idx, idx + len(raw))

        raw_norm = re.sub(r"\s+", " ", raw).strip()
        source_norm = re.sub(r"\s+", " ", source)
        idx_norm = source_norm.find(raw_norm)
        if idx_norm != -1:
            end = min(idx_norm + len(raw_norm), len(source))
            return (idx_norm, end)

        return None

    @staticmethod
    def _tier2_partial(raw: str, source: str) -> Optional[Tuple[int, int]]:
        numeric_tokens = re.findall(r"\d[\d,.]+", raw)
        if not numeric_tokens:
            return None

        key_num = max(numeric_tokens, key=len)
        if len(key_num) < 2:
            return None

        idx = source.find(key_num)
        if idx != -1:
            return (idx, idx + len(key_num))

        return None

    def _tier3_fuzzy(
        self,
        raw: str,
        source: str,
        token_index: List[Tuple[str, int]],
        source_tokens: List[str],
    ) -> Optional[Tuple[int, int]]:
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

            raw_total = sum(raw_counter.values())
            intersection = sum(
                min(raw_counter[t], window_counter[t]) for t in raw_counter
            )
            if raw_total == 0 or intersection / raw_total < _FUZZY_MIN_OVERLAP:
                continue

            window_text = " ".join(window_toks)
            matcher = difflib.SequenceMatcher(None, raw, window_text, autojunk=False)
            matching_chars = sum(size for _, _, size in matcher.get_matching_blocks())
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

    @staticmethod
    def _build_token_index(text: str) -> List[Tuple[str, int]]:
        return [(m.group(), m.start()) for m in re.finditer(r"\w+|\S", text)]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|\S", text)

    @staticmethod
    def _mark(
        fact: ScrapedFact,
        interval: Optional[Tuple[int, int]],
        status: str,
        score: float,
    ) -> None:
        fact.char_interval = interval
        fact.alignment_status = status          # type: ignore[assignment]
        fact.alignment_confidence = score


# ---------------------------------------------------------------------------
# ContextIndexer  (unchanged)
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

    def index_blocks(self, blocks: List[DocBlock], record_map: Optional[Dict[str, str]] = None):
        if not blocks:
            return
        documents = [b.content for b in blocks]
        ids = [b.id for b in blocks]
        metadatas = []
        for b in blocks:
            # Extract entity ID from section path (last element is usually the company)
            company = b.section_path[-1] if b.section_path else "Global_Entity"
            # FIX: Use the same canonicalization as the indexer
            entity_id = self._company_to_id(company)
            
            meta = {
                "block_type": b.block_type.value,
                "section_path": json.dumps(b.section_path),
                "page_num": b.page_num or 0,
                "canonical_entity_id": entity_id,
            }
            # Add source_record if provided (for strict doc-level scoping)
            if record_map and b.id in record_map:
                meta["source_record"] = record_map[b.id]
            
            metadatas.append(meta)
            
        self.text_collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(blocks)} blocks in ChromaDB (with entity/record metadata).")

    @staticmethod
    def _company_to_id(company: str) -> str:
        if not company or company in ("Global_Entity", "Unknown Entity", ""):
            return "EXP_GLOBAL"
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", company)
        clean = re.sub(r"\s+", "_", clean.strip()).upper()
        return f"ID_{clean}"

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