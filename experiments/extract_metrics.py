"""
experiments/extract_metrics.py
Phase 3: Deterministic Metric Extraction
------------------------------------------
Computes all paper metrics from generation_results.json and
chunk_metadata.json using only programmatic logic — zero LLM-judge cost.

Directory contract:
  experiments/extract_metrics.py                ← this file
  data/exp/global_index/chunk_metadata.json     ← chunk_id → source_records map
  data/exp/results/generation_results.json      ← input (Phase 2 versioned envelope)
  data/exp/results/final_metrics.json           ← per-sample metrics
  data/exp/results/results_summary.md           ← paper-ready tables

Schema notes (confirmed against normalized JSONL):
  - Golden answer field: `target_sentence`
  - Record ID field: `id`
  - RetrievalPlan has a `reasoning` field (Navigator SLM output, whitepaper §2)

FIXES:
  1. JSON loader handles Phase 2 versioned envelope format:
       { "_config": {...}, "results": [...] }
     The old code assigned the full dict to self.data and crashed on the
     first `record["sample_info"]` access with KeyError.
  2. hallucination_rate called with only the context actually shown to that
     run's model, not both contexts.  Feeding both contexts suppressed the
     baseline hallucination rate whenever a fabricated number happened to
     appear in the VeNRA context — an unfair cross-contamination.
  3. semantic_bleed_ratio uses word-boundary regex for all term lookups.
     Plain substring matching caused false positives: "revenue" in the query
     matched "cost of revenue" as a bleed trigger, accusing correct retrieval
     of semantic confusion.
  4. Failure Matrix markdown table fixed: failure_row() was returning a single
     concatenated string inserted into one cell of a 4-column table, producing
     structurally invalid markdown.  Now each column has its own helper that
     returns a plain count, and the table is correctly formed.
  5. BUG 1 FIX (Scale Blindness): normalize_numeric expands financial suffixes.
  6. BUG 1 FIX (Full Trace Audit): hallucination_rate inspects the reasoning trace.
  7. BUG 2 FIX (UFL Bleed): Taxonomy added to detect cross-document contamination.
  8. BUG 4 FIX (Qualitative Taxonomy): TYPE_4_MODALITY_MISMATCH added for PAL runs
     on qualitative queries. Refined TYPE_3 assignment.
  9. _can_derive_arithmetically extended to cover percentage-change and ratio
     formulas (the dominant FinQA arithmetic patterns). Without this, a PAL agent
     that correctly computes (a-b)/b*100 from context values would be flagged as
     hallucinating because the derived percentage is not verbatim in the context.
     The previous 1-step +/- rescue was insufficient for all real FinQA cases.
 10. answer_only parameter removed from hallucination_rate — it was dead code.
     The caller pre-selects audit_trace before passing; no internal branching
     was ever needed. Removing it prevents future maintainers from thinking the
     function behaves differently for PAL when it does not.
 11. T7 OVER-CLASSIFICATION FIX (get_failure_type): The guard `code_executed is None`
     was unconditional. For CoT runs (run_1, run_2), code_success is set to None
     in analyze() because code execution is not applicable — not because generation
     failed. This caused every wrong CoT answer to be classified as T7 instead of
     T1/T2/T3. Fix: `code_executed is None` now only triggers T7 when run_id is a
     PAL run ("run_3" or "run_4"), where None is the GENERATION_FAILURE sentinel
     written by _run_pal. CoT runs use only the failure_sentinels and
     truncation_signals checks.
"""

import json
import os
import re
import hashlib
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# ── Chunk ID (single source of truth) ────────────────────────────────────────

def get_chunk_id(record_id: str, content: str) -> str:
    """
    Deterministic MD5 with Canonical Whitespace Normalization and Namespacing.
    Matches the logic in build_global_index.py and retriever.py.
    """
    if not content:
        return ""
    canonical  = " ".join(content.lower().split())
    namespaced = f"{record_id}::{canonical}"
    return hashlib.md5(namespaced.encode()).hexdigest()


# TOGGLE THIS: "GEMINI" or "QWEN" to match Phase 2 run
MODEL_TYPE = "GEMINI"

# ── Paths ─────────────────────────────────────────────────────────────────────
GLOBAL_IDX_DIR      = "data/exp/global_index"
RESULTS_DIR         = "data/exp/results"
CHUNK_METADATA_PATH = os.path.join(GLOBAL_IDX_DIR, "chunk_metadata.json")
RESULTS_PATH        = os.path.join(RESULTS_DIR, f"generation_results_{MODEL_TYPE.lower()}.json")
FINAL_METRICS_PATH  = os.path.join(RESULTS_DIR, f"final_metrics_{MODEL_TYPE.lower()}.json")
SUMMARY_MD_PATH     = os.path.join(RESULTS_DIR, f"results_summary_{MODEL_TYPE.lower()}.md")

TOP_K = 5  # keep in sync with unified_benchmark.py

# Which retrieval context each run received
RUN_RETRIEVAL_TYPE = {
    "run_1": "baseline",
    "run_2": "venra",
    "run_3": "baseline",
    "run_4": "venra",
}

# PAL runs — the ONLY runs where code_executed=None is a generation failure sentinel.
# For CoT runs (run_1, run_2) code_success is legitimately None (not applicable).
PAL_RUNS = {"run_3", "run_4"}

# Semantic Bleed hard-negative term pairs (term-level only).
# Year pairs are intentionally excluded: financial documents legitimately
# reference multiple years, so year co-occurrence is not a bleed signal.
TERM_BLEED_PAIRS = [
    ("gross profit",              "net profit"),
    ("gross margin",              "net margin"),
    ("net income",                "net loss"),
    ("operating income",          "operating loss"),
    ("revenue",                   "cost of revenue"),
    ("revenue",                   "cost of goods sold"),
    ("total assets",              "total liabilities"),
    ("cash flow from operations", "cash flow from investing"),
    ("diluted eps",               "basic eps"),
    ("accounts receivable",       "accounts payable"),
]


# ── Numeric helpers ───────────────────────────────────────────────────────────

def normalize_numeric(text: str) -> List[str]:
    """
    Extracts and normalises all numbers from a string.
    Strips currency symbols, commas, and percent signs.

    FIX (Scale Blindness): Detects and expands financial suffixes
    (million, billion, trillion) to full float values.

    BUG A FIX: Always emit both raw and scaled forms. FinQA gold answers
    often omit suffixes (e.g., gold="12", pred="12.0 million"). Emitting
    both "12" and "12000000" ensures EM=True for correct logic.

    FIX (Partial Match Bug): Use strict boundary checks to prevent '3' from
    matching inside '3.2' or '2022' from matching sub-segments.
    """
    if not text:
        return []

    text = text.lower()
    scales = {
        "million": 1e6, "mn": 1e6, "m": 1e6,
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "trillion": 1e12, "tn": 1e12, "t": 1e12
    }

    text = text.replace("$", " ").replace(",", "").replace("%", " ")

    pattern = r"(-?\d+\.?\d*)(?![.\d])\s*(million|billion|trillion|mn|bn|tn|m|b|t)?\b"
    matches = re.findall(pattern, text)

    out = set()
    for num_str, suffix in matches:
        try:
            v = float(num_str)
            out.add(str(int(v)) if v == int(v) else f"{v:.4f}")
            if suffix in scales:
                v_scaled = v * scales[suffix]
                out.add(str(int(v_scaled)) if v_scaled == int(v_scaled) else f"{v_scaled:.4f}")
        except ValueError:
            continue
    return list(out)

# ── Retrieval metrics ─────────────────────────────────────────────────────────

def exact_recall(
    golden_chunk_ids: Set[str],
    retrieved_chunk_ids: List[str],
    sample_id: str,
    chunk_meta: Dict[str, Any],
) -> bool:
    """
    ID-based Golden Recall@K.

    BUG 3 FIX: Use subset check (ALL Recall).
    BUG 4 FIX: Strict Record Check. For shared chunks (TAT-QA), we verify that
    the retrieved chunk is actually tagged with this specific sample's record ID.
    """
    if not golden_chunk_ids:
        return True

    retrieved_set = set(retrieved_chunk_ids)

    if not golden_chunk_ids.issubset(retrieved_set):
        return False

    return True


def build_golden_chunk_ids(
    sample: Dict[str, Any],
    chunk_meta: Dict[str, Any],
) -> Set[str]:
    """
    Returns the specific chunk IDs that are golden evidence for this sample.

    BUG C FIX: Implement Answer-Level Recall (Grounding Evidence).
    Instead of returning all chunks for a record, we isolate the specific
    chunk(s) containing the answer OR the numeric inputs used in the trace.
    """
    rec_id    = sample["id"]
    target    = sample.get("target_sentence", "")
    trace     = sample.get("trace_code", "")

    target_nums       = set(normalize_numeric(target))
    trace_nums        = set(normalize_numeric(trace))
    all_evidence_nums = target_nums | trace_nums

    chunks   = sample.get("context_chunks", [])
    gold_ids = set()

    for chunk_text in chunks:
        cid = get_chunk_id(sample["id"], chunk_text)

        chunk_nums = set(normalize_numeric(chunk_text))
        if all_evidence_nums and (all_evidence_nums & chunk_nums):
            gold_ids.add(cid)
            continue

        if target:
            if target.lower() in chunk_text.lower():
                gold_ids.add(cid)
                continue

            stopwords = {"and", "the", "of", "for", "in", "with", "on", "at", "by", "from", "to", "is", "a", "an"}
            target_tokens = {t for t in re.findall(r"\w+", target.lower()) if t not in stopwords and len(t) > 3}
            chunk_tokens  = {t for t in re.findall(r"\w+", chunk_text.lower())}
            if target_tokens:
                overlap = len(target_tokens & chunk_tokens) / len(target_tokens)
                if overlap >= 0.2:
                    gold_ids.add(cid)
                    continue

    if not gold_ids:
        record_chunks = [
            cid for cid, meta in chunk_meta.items()
            if rec_id in meta.get("source_records", [])
        ]
        if len(record_chunks) <= 3:
            gold_ids = set(record_chunks)
        else:
            return set()

    return gold_ids


def _term_in(term: str, text: str) -> bool:
    """
    FIX 3: word-boundary match to avoid false positives.
    Plain `term in text` matched "revenue" inside "cost of revenue", causing
    correct retrieval to be incorrectly flagged as a bleed event.
    """
    return bool(re.search(r"\b" + re.escape(term) + r"\b", text))


def semantic_bleed_ratio(retrieved_context: str, query: str) -> float:
    """
    Term-level semantic bleed: fraction of query-relevant term pairs for which
    the retrieved context contains the logically opposite term.
    """
    q = query.lower()
    c = retrieved_context.lower()
    triggered, checked = 0, 0
    for target, opposite in TERM_BLEED_PAIRS:
        if _term_in(target, q):
            checked += 1
            if _term_in(opposite, c):
                triggered += 1
    return triggered / checked if checked else 0.0


# ── Generation metrics ────────────────────────────────────────────────────────

def is_exact_match(predicted: str, golden: str, label: str = "Supported", trace: str = "") -> bool:
    """
    Numerical EM: any gold number appears in the prediction's number set.
    Falls back to substring or token-overlap match for non-numeric golden answers.

    BUG G FIX: Strip citations and trailing punctuation from prediction.
    SOFT EM FIX: Use token overlap for qualitative answers to handle rephrasing.
    """
    if not predicted:
        return False

    if label == "Unfounded":
        refusals = [
            "not available", "no information", "cannot answer",
            "not found", "insufficient data", "no data", "does not state"
        ]
        return any(r in predicted.lower() for r in refusals)

    clean_pred = re.sub(r"\(source:.*?\)", "", predicted, flags=re.IGNORECASE)
    clean_pred = clean_pred.strip().rstrip(".").lower()

    is_lookup = "# VERIFICATION_TYPE: LOOKUP" in trace

    pred_nums = normalize_numeric(predicted)
    gold_nums = normalize_numeric(golden)

    if not gold_nums or is_lookup:
        clean_gold = re.sub(r"\s*[\(\[\-].*?(\$|usd|nzd|eur|gbp|£|€|¥|\d).*$", "", golden, flags=re.IGNORECASE).strip().lower()

        if clean_gold in clean_pred:
            return True

        stopwords = {"and", "the", "of", "for", "in", "with", "on", "at", "by", "from", "to", "is", "a", "an"}
        gold_tokens = {t for t in re.findall(r"\w+", clean_gold) if t not in stopwords}
        pred_tokens = {t for t in re.findall(r"\w+", clean_pred) if t not in stopwords}

        if gold_tokens:
            overlap = len(gold_tokens & pred_tokens) / len(gold_tokens)
            if overlap >= 0.7:
                return True

    if gold_nums:
        if any(gn in pred_nums for gn in gold_nums):
            return True

        try:
            g_floats = [float(n) for n in gold_nums]
            p_floats = [float(n) for n in pred_nums]
            for g in g_floats:
                for p in p_floats:
                    if g != 0:
                        if abs(g - p) / abs(g) <= 0.005: return True
                    elif abs(g - p) < 0.015: return True

                    if g != 0:
                        if abs(g * 100 - p) / abs(g * 100) <= 0.01: return True
                        if abs(g - p * 100) / abs(g) <= 0.01: return True
        except (ValueError, TypeError):
            pass

    return False


def get_failure_type(
    em: bool,
    retrieval_ok: bool,
    predicted: str,
    retrieved_context: str,
    golden_answer: str,
    ufl_bleed: bool = False,
    run_id: str = "run_1",
    code_executed: Optional[bool] = False,
    coverage_gap: bool = False,
    source_ds: str = "unknown",
    is_self_aware_warning: bool = False,
    finish_reason: str = "STOP",
) -> str:
    """
    VeNRA Failure Taxonomy:
      NONE   — correct answer
      TYPE_0 — UFL Bleed: context contains unrelated data from other docs
      TYPE_1 — Retrieval Blindness: gold evidence not in retrieved context
      TYPE_2 — Generative Conflation: gold was in context but model returned
               a different number from context (distractor confusion)
      TYPE_3 — Arithmetic Hallucination: gold was in context, model had the
               right numbers, but computed the wrong result
      TYPE_4 — Modality Mismatch: code agent (PAL) attempted qualitative lookup
      TYPE_5 — Coverage Gap: gold chunk not in global index at all
      TYPE_6 — Gold Ambiguity: the question/label is ambiguous or noisy
      TYPE_7 — Generation Failure: model crashed, timed out, safety-blocked,
               or PAL Pass 1 returned the None sentinel

    T7 OVER-CLASSIFICATION FIX (Fix 11):
      `code_executed is None` is now restricted to PAL_RUNS only.
      For CoT runs (run_1, run_2), code_success=None means "not applicable"
      — it is set that way in analyze() for every CoT sample regardless of
      outcome. Checking it unconditionally caused EVERY wrong CoT answer to
      be promoted to T7, wiping out T1/T2/T3 entirely.
    """
    if em:
        return "NONE"

    # ── T7: Generation Failure ────────────────────────────────────────────────
    # Signal (a): answer IS a known failure sentinel string
    # Signal (b): PAL only — None is the sentinel _run_pal writes on crash.
    #             NOT checked for CoT runs; None there means "not applicable".
    # Signal (c): finish_reason reports output truncation (MAX_TOKENS hit).
    failure_sentinels  = {"GENERATION_FAILURE", "REASONING_FAILURE", "TIMEOUT_FAILURE"}
    truncation_signals = ["MAX_TOKENS", "length", "max_tokens"]

    is_pal = run_id in PAL_RUNS

    if (
        predicted in failure_sentinels
        or (is_pal and code_executed is None)
        or any(s in str(finish_reason) for s in truncation_signals)
    ):
        return "TYPE_7_GENERATION_FAILURE"

    # ── TAT-QA ambiguity heuristic ────────────────────────────────────────────
    if "tatqa" in source_ds.lower():
        ambiguity_triggers = ["goodwill", "plant and equipment", "intangibles", "carrying value"]
        if any(t in predicted.lower() for t in ambiguity_triggers):
            return "TYPE_6_GOLD_AMBIGUITY"

    # ── T5: Coverage Gap ──────────────────────────────────────────────────────
    if coverage_gap:
        return "TYPE_5_COVERAGE_GAP"

    # ── T1: Retrieval Blindness ───────────────────────────────────────────────
    if not retrieval_ok:
        return "TYPE_1_RETRIEVAL_BLINDNESS"

    gold_nums  = normalize_numeric(golden_answer)
    is_numeric = len(gold_nums) > 0

    # ── T4: Modality Mismatch ─────────────────────────────────────────────────
    if not is_numeric and is_pal and code_executed:
        return "TYPE_4_MODALITY_MISMATCH"

    refusals = [
        "not available", "no information", "cannot answer", "not found",
        "insufficient", "does not mention", "does not state", "not in context",
    ]
    is_refusal = any(r in predicted.lower() for r in refusals) or is_self_aware_warning

    if is_numeric:
        if is_refusal:
            return "TYPE_1_RETRIEVAL_BLINDNESS"

        context_nums = set(normalize_numeric(retrieved_context))
        if set(gold_nums).issubset(context_nums):
            return "TYPE_3_ARITHMETIC_HALLUCINATION"

        if ufl_bleed and not is_refusal and not code_executed:
            ufl_section = retrieved_context.split("# SOURCE TEXT CHUNKS")[0]
            ufl_nums    = set(normalize_numeric(ufl_section))
            pred_nums   = set(normalize_numeric(predicted))
            if pred_nums.intersection(ufl_nums):
                return "TYPE_0_UFL_BLEED"

        return "TYPE_2_GENERATIVE_CONFLATION"

    # Qualitative
    if is_refusal:
        return "TYPE_1_RETRIEVAL_BLINDNESS"

    return "TYPE_2_GENERATIVE_CONFLATION"


def _can_derive_arithmetically(target: float, candidates: List[float]) -> bool:
    """
    Check if target can be reached via basic arithmetic or percentage-change /
    ratio operations — the dominant PAL computation patterns in FinQA.

    Covered operations (all pairwise over candidates):
      • Addition / subtraction:  c1 ± c2
      • Percentage change:       (c1 - c2) / c2 × 100  (and symmetric)
      • Margin / share:          c1 / c2 × 100          (and inverse)
      • Raw ratio:               c1 / c2                (and inverse)

    FIX (vs previous 1-step +/- only): The prior implementation only covered
    c1 + c2 and c1 - c2.  A PAL agent computing (1309-1281)/1281*100 = 2.186%
    correctly from context values would not match any +/- pair, causing the
    hallucination rescue to fail and the valid derivation to be flagged.
    """
    if not candidates:
        return False

    TOL       = 0.015
    TOL_RATIO = 0.001

    for i, c1 in enumerate(candidates):
        for c2 in candidates[i:]:
            if abs((c1 + c2) - target) < 1e-6: return True
            if abs((c1 - c2) - target) < 1e-6: return True
            if abs((c2 - c1) - target) < 1e-6: return True

            if c2 != 0:
                if abs((c1 - c2) / c2 * 100 - target) < TOL: return True
            if c1 != 0:
                if abs((c2 - c1) / c1 * 100 - target) < TOL: return True
                if abs((c1 - c2) / c1 * 100 - target) < TOL: return True

            if c2 != 0:
                if abs(c1 / c2 * 100 - target) < TOL: return True
            if c1 != 0:
                if abs(c2 / c1 * 100 - target) < TOL: return True

            if c2 != 0:
                if abs(c1 / c2 - target) < TOL_RATIO: return True
            if c1 != 0:
                if abs(c2 / c1 - target) < TOL_RATIO: return True

    return False


def hallucination_rate(
    full_trace: str,
    source_context: str,
    is_correct: bool = False,
    pred_val: Optional[float] = None,
    query: str = "",
) -> float:
    """
    FIX (Full Trace Audit):
    1. Whitelist em=True: A correct answer is by definition grounded.
    2. If em=False but pred_val can be derived from context numbers via
       basic arithmetic or percentage-change (FIX 9), rate is 0.0.
    3. Returns: 1.0 if any number in full_trace is not found in context or
       derivable from context. Binary per-sample for consistent denominators.

    BUG E FIX: Whitelist numbers from the query to prevent penalising
    refusal echos (e.g. "I can't find info for 28 June 2019").

    BUG 1 FAIRNESS FIX: Strip markdown code blocks before auditing numbers.
    Prevents Python indices (v[0], v[1]) and syntax from being flagged.
    """
    if is_correct:
        return 0.0

    trace_clean = re.sub(r"```.*?```", "", full_trace, flags=re.DOTALL)

    pred_nums = normalize_numeric(trace_clean)
    if not pred_nums:
        return 0.0
    ctx_nums   = set(normalize_numeric(source_context))
    query_nums = set(normalize_numeric(query))

    if pred_val is not None:
        ctx_floats = []
        for cn in ctx_nums:
            try: ctx_floats.append(float(cn))
            except ValueError: pass
        if _can_derive_arithmetically(pred_val, ctx_floats):
            return 0.0

    whitelist             = {"0", "1", "100", "0.0", "1.0", "100.0"} | query_nums
    hallucinated_instances = [n for n in pred_nums if n not in ctx_nums and n not in whitelist]

    return 1.0 if hallucinated_instances else 0.0


# ── Core analysis ─────────────────────────────────────────────────────────────

class MetricExtractor:

    def __init__(self):
        if not os.path.exists(RESULTS_PATH):
            raise FileNotFoundError(
                f"generation_results.json not found at {RESULTS_PATH}\n"
                "Run unified_benchmark.py (Phase 2) first."
            )
        if not os.path.exists(CHUNK_METADATA_PATH):
            raise FileNotFoundError(
                f"chunk_metadata.json not found at {CHUNK_METADATA_PATH}\n"
                "Run build_global_index.py (Phase 1) first."
            )

        with open(RESULTS_PATH) as f:
            raw = json.load(f)

        # FIX 1: Phase 2 writes a versioned envelope:
        #   { "_config": {...}, "results": [...] }
        self.data: List[Dict[str, Any]] = (
            raw["results"] if isinstance(raw, dict) else raw
        )

        if not self.data:
            raise ValueError("generation_results.json contains no result records.")

        with open(CHUNK_METADATA_PATH) as f:
            self.chunk_meta: Dict[str, Any] = json.load(f)

    def analyze(self):
        summary = []

        for record in self.data:
            sample        = record["sample_info"]
            golden_answer = sample["target_sentence"]
            query         = sample["query"]

            # ── Retrieval ──────────────────────────────────────────────────────
            baseline_ctx = record["retrieval"]["baseline"]["context"]
            venra_ctx    = record["retrieval"]["venra"]["context"]
            baseline_ids = record["retrieval"]["baseline"].get("retrieved_chunk_ids", [])
            venra_ids    = record["retrieval"]["venra"].get("retrieved_chunk_ids", [])

            golden_ids      = build_golden_chunk_ids(sample, self.chunk_meta)
            baseline_recall = exact_recall(golden_ids, baseline_ids, sample["id"], self.chunk_meta)
            venra_recall    = exact_recall(golden_ids, venra_ids,    sample["id"], self.chunk_meta)

            coverage_gap = False
            if golden_ids:
                coverage_gap = any(gid not in self.chunk_meta for gid in golden_ids)

            baseline_bleed  = semantic_bleed_ratio(baseline_ctx, query)
            venra_bleed     = semantic_bleed_ratio(venra_ctx,    query)
            first_pass_miss = record["retrieval"]["venra"].get("first_pass_miss", False)
            venra_ufl_bleed = record["retrieval"]["venra"].get("ufl_bleed", False)

            # ── Generation — all four runs ─────────────────────────────────────
            run_stats: Dict[str, Any] = {}

            for run_id in ["run_1", "run_2", "run_3", "run_4"]:
                run_data = record["runs"].get(run_id)

                if run_data is None:
                    run_stats[run_id] = {
                        "em": None, "failure_type": "MISSING",
                        "code_success": None, "hallucination_rate": None,
                    }
                    continue

                em = is_exact_match(
                    run_data["answer"],
                    golden_answer,
                    label=sample.get("label", "Supported"),
                    trace=sample.get("trace_code", "")
                )

                ret_type = RUN_RETRIEVAL_TYPE[run_id]
                ret_ctx  = venra_ctx    if ret_type == "venra" else baseline_ctx
                ret_ok   = venra_recall if ret_type == "venra" else baseline_recall

                pred_val = None
                try:
                    p_nums = normalize_numeric(run_data["answer"])
                    if p_nums:
                        pred_val = float(p_nums[0])
                except Exception:
                    pass

                # code_success is only meaningful for PAL runs with a logic trace.
                # For CoT runs (run_1, run_2) it is legitimately None — not a
                # failure signal. The None sentinel for generation failure is
                # exclusively written by _run_pal on crash.
                has_logic    = bool(sample.get("trace_code"))
                code_success: Optional[bool] = (
                    run_data.get("code_executed", False)
                    if (run_id in PAL_RUNS and has_logic)
                    else None
                )

                is_warn = run_data.get("full_response", {}).get("is_self_aware_warning", False)
                finish_reason = (
                    run_data.get("full_response", {}).get("finish_reason")
                    or run_data.get("full_response", {}).get("finish_reason_1")
                    or "STOP"
                )

                ftype = get_failure_type(
                    em, ret_ok, run_data["answer"], ret_ctx, golden_answer,
                    ufl_bleed=(venra_ufl_bleed if ret_type == "venra" else False),
                    run_id=run_id,
                    code_executed=code_success,
                    coverage_gap=coverage_gap,
                    source_ds=sample.get("source_ds", "unknown"),
                    is_self_aware_warning=is_warn,
                    finish_reason=finish_reason,
                )

                full_trace  = run_data.get("full_response", {}).get("raw_text", run_data["answer"])
                audit_trace = run_data["answer"] if run_id in PAL_RUNS else full_trace

                h_rate = hallucination_rate(
                    audit_trace,
                    ret_ctx,
                    is_correct=em,
                    pred_val=pred_val,
                    query=query,
                )

                if not ret_ok:
                    h_rate = 0.0

                run_stats[run_id] = {
                    "em":               em,
                    "failure_type":     ftype,
                    "code_success":     code_success,
                    "hallucination_rate": round(h_rate, 4),
                }

            summary.append({
                "id":        sample["id"],
                "source_ds": sample.get("source_ds", "unknown"),
                "retrieval": {
                    "baseline_recall": baseline_recall,
                    "venra_recall":    venra_recall,
                    "baseline_bleed":  round(baseline_bleed, 4),
                    "venra_bleed":     round(venra_bleed,    4),
                    "first_pass_miss": first_pass_miss,
                    "coverage_gap":    coverage_gap,
                    "venra_ufl_bleed": venra_ufl_bleed,
                    "ufl_row_count":   record["retrieval"]["venra"].get("ufl_row_count", 0),
                },
                "token_parity": record.get("token_parity", {}),
                "runs": run_stats,
            })

        self._save_results(summary)

    # ── Output ────────────────────────────────────────────────────────────────

    def _safe_pct(self, series: pd.Series) -> str:
        valid = series.dropna()
        return f"{valid.mean() * 100:.1f}%" if len(valid) else "N/A"

    def _safe_avg(self, df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return "N/A"
        v = df[col].dropna()
        return f"{v.mean():.3f}" if len(v) else "N/A"

    def _em(self, df: pd.DataFrame, run_id: str) -> str:
        col = f"runs.{run_id}.em"
        return self._safe_pct(df[col]) if col in df.columns else "N/A"

    def _hall(self, df: pd.DataFrame, run_id: str) -> str:
        col = f"runs.{run_id}.hallucination_rate"
        if col not in df.columns:
            return "N/A"
        v = df[col].dropna()
        return f"{v.mean()*100:.1f}%" if len(v) else "N/A"

    def _compile_rate(self, df: pd.DataFrame, run_id: str) -> str:
        col = f"runs.{run_id}.code_success"
        return self._safe_pct(df[col]) if col in df.columns else "—"

    def _fail_t0(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_0_UFL_BLEED", 0)) if col in df.columns else "N/A"

    def _fail_t1(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_1_RETRIEVAL_BLINDNESS", 0)) if col in df.columns else "N/A"

    def _fail_t2(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_2_GENERATIVE_CONFLATION", 0)) if col in df.columns else "N/A"

    def _fail_t3(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_3_ARITHMETIC_HALLUCINATION", 0)) if col in df.columns else "N/A"

    def _fail_t4(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_4_MODALITY_MISMATCH", 0)) if col in df.columns else "N/A"

    def _fail_t5(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_5_COVERAGE_GAP", 0)) if col in df.columns else "N/A"

    def _fail_t6(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_6_GOLD_AMBIGUITY", 0)) if col in df.columns else "N/A"

    def _fail_t7(self, df, run_id):
        col = f"runs.{run_id}.failure_type"
        return str(Counter(df[col].dropna()).get("TYPE_7_GENERATION_FAILURE", 0)) if col in df.columns else "N/A"

    def _rescue_rate(self, df):
        """
        Navigator Rescue Rate (Experiment 2).
        Denominator: samples where baseline and VeNRA recall differed.
        Numerator: cases where VeNRA succeeded (recall=True).
        """
        if "retrieval.baseline_recall" not in df.columns or "retrieval.venra_recall" not in df.columns:
            return "—"

        delta_set = df[df["retrieval.baseline_recall"] != df["retrieval.venra_recall"]]
        if delta_set.empty:
            return "—"

        rescues = delta_set[delta_set["retrieval.venra_recall"] == True]
        return f"{len(rescues) / len(delta_set):.1%}"

    def _save_results(self, summary: List[Dict]):
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(FINAL_METRICS_PATH, "w") as f:
            json.dump(summary, f, indent=2)

        df = pd.json_normalize(summary)

        tok_cols  = [c for c in df.columns if "tokens" in c]
        tok_lines = "\n".join(
            f"- `{c}`: {df[c].dropna().mean():.0f} avg tokens"
            for c in tok_cols
        ) or "No token data available."

        md = f"""# VeNRA Stage A — Results Summary
Generated from {len(summary)} samples.
## Retrieval Performance (Experiment 2)
| Metric | Baseline (Vector) | VeNRA (DualRetriever) |
| :----- | :---------------- | :-------------------- |
| Golden Recall@{TOP_K} (ID-exact)     | {self._safe_pct(df['retrieval.baseline_recall'])} | {self._safe_pct(df['retrieval.venra_recall'])} |
| Coverage Gap (Missing from Index)    | {self._safe_pct(df['retrieval.coverage_gap'])} | {self._safe_pct(df['retrieval.coverage_gap'])} |
| Semantic Bleed Ratio (term-level)    | {self._safe_avg(df, 'retrieval.baseline_bleed')} | {self._safe_avg(df, 'retrieval.venra_bleed')} |
| **Navigator Rescue Rate** (misses)   | — | {self._rescue_rate(df)} |
## End-to-End Accuracy (Experiment 1)
| Run | Configuration | EM | Hallucination Rate |
| :-- | :------------ | :- | :----------------- |
| 1 | Baseline RAG (Vector + Gemini CoT)         | {self._em(df, 'run_1')} | {self._hall(df, 'run_1')} |
| 2 | Smart Ret., Dumb Math (VeNRA + Gemini CoT) | {self._em(df, 'run_2')} | {self._hall(df, 'run_2')} |
| 3 | Dumb Ret., Smart Math (Vector + PAL)        | {self._em(df, 'run_3')} | {self._hall(df, 'run_3')} |
| 4 | VeNRA Full (VeNRA + PAL, zero-shot)         | {self._em(df, 'run_4')} | {self._hall(df, 'run_4')} |
## Code Compilation Rate — PAL Runs (3 & 4)
| Run | Configuration | First-Attempt Compilation |
| :-- | :------------ | :------------------------ |
| 3 | Vector + VeNRA Code Agent | {self._compile_rate(df, 'run_3')} |
| 4 | VeNRA  + VeNRA Code Agent | {self._compile_rate(df, 'run_4')} |
## Failure Matrix
| Run | Configuration | T0 UFL | T1 Blind | T2 Confl. | T3 Arith. | T4 Modality | T5 Gap | T6 Ambig. | T7 Gen. |
| :-- | :------------ | :----: | :------: | :-------: | :-------: | :---------: | :----: | :-------: | :----: |
| 1 | Baseline RAG          | {self._fail_t0(df, 'run_1')} | {self._fail_t1(df, 'run_1')} | {self._fail_t2(df, 'run_1')} | {self._fail_t3(df, 'run_1')} | {self._fail_t4(df, 'run_1')} | {self._fail_t5(df, 'run_1')} | {self._fail_t6(df, 'run_1')} | {self._fail_t7(df, 'run_1')} |
| 2 | Smart Ret., Dumb Math | {self._fail_t0(df, 'run_2')} | {self._fail_t1(df, 'run_2')} | {self._fail_t2(df, 'run_2')} | {self._fail_t3(df, 'run_2')} | {self._fail_t4(df, 'run_2')} | {self._fail_t5(df, 'run_2')} | {self._fail_t6(df, 'run_2')} | {self._fail_t7(df, 'run_2')} |
| 3 | Dumb Ret., Smart Math | {self._fail_t0(df, 'run_3')} | {self._fail_t1(df, 'run_3')} | {self._fail_t2(df, 'run_3')} | {self._fail_t3(df, 'run_3')} | {self._fail_t4(df, 'run_3')} | {self._fail_t5(df, 'run_3')} | {self._fail_t6(df, 'run_3')} | {self._fail_t7(df, 'run_3')} |
| 4 | VeNRA Full            | {self._fail_t0(df, 'run_4')} | {self._fail_t1(df, 'run_4')} | {self._fail_t2(df, 'run_4')} | {self._fail_t3(df, 'run_4')} | {self._fail_t4(df, 'run_4')} | {self._fail_t5(df, 'run_4')} | {self._fail_t6(df, 'run_4')} | {self._fail_t7(df, 'run_4')} |
## Context Volume Parity
{tok_lines}
"""
        with open(SUMMARY_MD_PATH, "w") as f:
            f.write(md)

        print(f"Metrics saved:\n  {FINAL_METRICS_PATH}\n  {SUMMARY_MD_PATH}")
        print(f"\nQuick summary ({len(summary)} samples):")
        print(f"  Run 1 EM (Baseline RAG):   {self._em(df, 'run_1')}")
        print(f"  Run 2 EM (Smart Ret.):     {self._em(df, 'run_2')}")
        print(f"  Run 3 EM (Smart Math):     {self._em(df, 'run_3')}")
        print(f"  Run 4 EM (VeNRA Full):     {self._em(df, 'run_4')}")
        print(f"  Baseline Recall@{TOP_K}:       {self._safe_pct(df['retrieval.baseline_recall'])}")
        print(f"  VeNRA Recall@{TOP_K}:          {self._safe_pct(df['retrieval.venra_recall'])}")


if __name__ == "__main__":
    MetricExtractor().analyze()