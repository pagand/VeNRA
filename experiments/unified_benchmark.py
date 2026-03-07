"""
experiments/unified_benchmark.py
Phase 2: Unified Benchmark (2×2 Execution Matrix)
---------------------------------------------------
Runs queries across four pipeline configurations to isolate the
causal contribution of each VeNRA component.

2×2 matrix:
  Run 1: Baseline RAG          → Vector retrieval    + Gemini CoT
  Run 2: Smart Ret., Dumb Math → VeNRA DualRetriever + Gemini CoT
  Run 3: Dumb Ret., Smart Math → Vector retrieval    + VeNRA Code Agent (PAL)
  Run 4: VeNRA Full            → VeNRA DualRetriever + VeNRA Code Agent (PAL)

Directory contract:
  experiments/unified_benchmark.py            ← this file
  src/venra/                                  ← library, import only
  data/golden_records/                        ← read-only input datasets
  data/exp/global_index/                      ← ChromaDB + UFL (Phase 1 output)
  data/exp/retrieval_cache/{id}_{type}.json   ← cached retrieval results
  data/exp/results/generation_results.json    ← benchmark output (versioned envelope)

API keys used (from .env):
  GEMINI_API_KEY   → CoT baseline (Runs 1 & 2)
  NVIDIA_API_KEY   → Kimi NIM inside ReasoningAgent code pass (Runs 3 & 4)
  GROQ_API_KEY     → Groq synthesis pass inside ReasoningAgent (Runs 3 & 4)

Checkpoint format:
  { "_config": { gemini_model, sample_counts, random_seed, top_k },
    "results": [ ... per-sample records ... ] }
  On resume, _config is compared to current values. A mismatch warns loudly.

Schema notes:
  - Golden answers are stored in `target_sentence` in all normalized JSONL files.
  - RetrievalPlan includes a `reasoning` field (Navigator SLM output) per whitepaper.

FIXES (all rounds):
  1. METADATA_PATH defined at module level (was missing, caused NameError).
  2. get_chunk_id imported from build_global_index — single source of truth.
  3. Concurrency note: do not run Phase 1 and Phase 2 simultaneously (shared Groq TPM).
  4. Context caching removed — COT_SYSTEM_PROMPT is ~60 tokens, uncacheable.
  5. Gemini client created in __init__, not at import time.
  6. GenerateContentConfig constructed once with all fields — no post-hoc mutation.
  7. user_prompt never rebound — prompt structure identical across every call.
  8. _pydantic_dump helper for Pydantic v1/v2 compatibility.
  9. Checkpoint _config stamp — mismatch emits loud warning.
 10. splitlines()[-1] guarded against empty Gemini response (IndexError on []).
 11. BUG 3 FIX (Bleed detection false-positive): In _baseline_context and
     _venra_context, bleed filter now guards with `meta and` before checking
     chunk_is_from_same_ds.  Previously, a chunk whose ID was absent from
     chunk_metadata.json (e.g. an orphaned ChromaDB vector whose Phase 1
     delete failed) would have meta={} → source_records=[] →
     chunk_is_from_same_ds=False → falsely filtered as cross-dataset bleed.
     This silently dropped valid context chunks from BOTH baseline and VeNRA,
     making both systems look worse and potentially skewing the Run 1/Run 4
     comparison.  The fix: only apply the filter when we have confirmed origin
     evidence; unknown-origin chunks are included with a debug-level log.
 12. PAL FAIL-FAST FIX: _run_pal no longer retries on generation failure.
     A single Pass 1 attempt is made. On any exception reasoning=None and
     GENERATION_FAILURE is returned immediately. The prior 3-attempt loop
     with exponential backoff (2s, 4s) wasted up to 6s per PAL call per
     sample and obscured whether the model can reliably generate at all.
     Retry logic belongs at the orchestration layer, not inside the benchmark.
"""

import json
import os
import asyncio
import random
import sys
import re
from typing import List, Dict, Any

from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from venra.models import RetrievalPlan
from venra.retriever import DualRetriever
from venra.agent import ReasoningAgent
from venra.assembler import ContextAssembler
from venra.navigator import Navigator
from venra.logging_config import logger

# Single source of truth for chunk ID generation.
# Never reimplement inline — a hash mismatch makes every record appear
# "not indexed" and load_samples() returns nothing with no obvious error.
from build_global_index import get_chunk_id

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
GLOBAL_IDX_DIR          = "data/exp/global_index"
CHROMA_DB_PATH          = os.path.join(GLOBAL_IDX_DIR, "chroma_db")
UFL_PATH                = os.path.join(GLOBAL_IDX_DIR, "ufl.parquet")
SCHEMA_PATH             = os.path.join(GLOBAL_IDX_DIR, "schema_summary.json")
METADATA_PATH           = os.path.join(GLOBAL_IDX_DIR, "chunk_metadata.json")
RETRIEVAL_CACHE_DIR     = "data/exp/retrieval_cache"
GENERATION_RESULTS_PATH = "data/exp/results/generation_results.json"
GOLDEN_RECORDS_DIR      = "data/golden_records"

# ── Experiment config ─────────────────────────────────────────────────────────
# TOGGLE THIS: "GEMINI" or "QWEN"
MODEL_TYPE = "GEMINI"

# Stamped into every checkpoint.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
QWEN_MODEL   = "qwen/qwen3-32b"           # Groq ID for Qwen 3 32B
LLAMA_MODEL  = "llama-3.3-70b-versatile"  # The "Small Thinking Model" for Pass 2

# Choose the active Pass 1 model ID
ACTIVE_MODEL = GEMINI_MODEL if MODEL_TYPE == "GEMINI" else QWEN_MODEL

SAMPLE_COUNTS: Dict[str, int] = {
    "financebench_normalized.jsonl":    20,   # production: 75
    "tatqa_normalized_test_gold.jsonl": 20,   # production: 75
    "finqa_normalized.jsonl":           10,   # production: 50
}
RANDOM_SEED = 42
TOP_K       = 5

# Output path changes based on model to prevent overwriting
GENERATION_RESULTS_PATH = f"data/exp/results/generation_results_{MODEL_TYPE.lower()}.json"

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

# ── Helpers ───────────────────────────────────────────────────────────────────

class AgentReasoning(BaseModel):
    """The internal thought process of the agent."""
    plan: str = Field(..., description="Step-by-step logic of how to answer the query.")
    requires_math: bool = Field(..., description="Whether Python code is needed for calculation.")
    python_code: Optional[str] = Field(None, description="The code to run. Use print() to output final numbers.")
    missing_info: List[str] = Field(default_factory=list, description="Any data points expected but not found in context.")

class FinalResponse(BaseModel):
    """The final answer delivered to the user."""
    answer: str = Field(..., description="The definitive answer string. If grounded, mention chunk/row IDs in the text.")
    nuances: Optional[str] = Field(None, description="Important context found in text chunks (e.g. 'adjusted for inflation').")
    data_source_type: Literal["GROUNDED", "INTERNAL_KNOWLEDGE", "MIXED", "NOT_FOUND"] = Field(
        ..., description="The primary source of the information provided."
    )
    citations: List[str] = Field(..., description="Specific IDs of rows or chunks used.")
    groundedness_score: float = Field(..., description="0.0 to 1.0. High for document context, low for internal knowledge.")
    is_self_aware_warning: bool = Field(..., description="True if the agent is guessing, lacks data, or used internal knowledge.")

def _pydantic_dump(model) -> Dict[str, Any]:
    """Pydantic v1 / v2 compatible serialisation."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _estimate_tokens(text: str) -> int:
    """
    Conservative token estimate (no external dependency).
    Financial prose: ~0.75 words/token → divide word count by 0.75.
    """
    return max(1, int(len(text.split()) / 0.75))


# ── System prompts ────────────────────────────────────────────────────────────

COT_SYSTEM_PROMPT = (
    "You are a precise financial analyst. "
    "Answer the question using ONLY the information in the provided context. "
    "Show your arithmetic step-by-step. "
    "Your LAST line must be the final answer as a single number or short phrase, "
    "with no units or currency symbols unless the question requires them."
)

PAL_CONTEXT_HEADER = (
    "[INSTRUCTION] Read the context below. Identify the exact numerical "
    "variables needed, then write and execute a self-contained Python script "
    "that computes the answer. Print the final answer as a plain number.\n\n"
)


# ── Benchmark ─────────────────────────────────────────────────────────────────

class UnifiedBenchmark:

    def __init__(self):
        # Gemini Client
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY not found.")
        self._gemini = genai.Client(api_key=gemini_api_key)

        # Groq Client (using instructor for structured output)
        from openai import AsyncOpenAI
        import instructor
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError("GROQ_API_KEY not found.")
        raw_groq = AsyncOpenAI(api_key=groq_api_key)
        self._groq = instructor.from_openai(raw_groq, mode=instructor.Mode.JSON)

        self.retriever = DualRetriever(ufl_path=UFL_PATH, db_path=CHROMA_DB_PATH)
        self.navigator = Navigator(schema_path=SCHEMA_PATH)
        self.assembler = ContextAssembler()
        self.agent     = ReasoningAgent()

        os.makedirs(RETRIEVAL_CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(GENERATION_RESULTS_PATH), exist_ok=True)

        # BUG B FIX: Load metadata for text chunk bleed detection
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                self.chunk_metadata = json.load(f)
        else:
            self.chunk_metadata = {}

    # ── Sampling ──────────────────────────────────────────────────────────────

    def load_samples(self) -> List[Dict[str, Any]]:
        """
        Reproducible stratified sample restricted to records fully processed
        in Phase 1 (every context chunk present in chunk_metadata.json).

        Uses `target_sentence` as the golden answer field — confirmed in
        all normalized JSONL datasets (FinanceBench, TAT-QA, FinQA).
        """
        random.seed(RANDOM_SEED)
        all_samples: List[Dict[str, Any]] = []

        if not os.path.exists(METADATA_PATH):
            logger.error(
                f"Phase 1 metadata not found at {METADATA_PATH}. "
                "Run build_global_index.py before this script."
            )
            return []

        with open(METADATA_PATH) as f:
            meta = json.load(f)
        indexed_chunk_ids: set = set(meta.keys())
        logger.info(f"Phase 1 index: {len(indexed_chunk_ids)} unique chunk IDs.")

        total_seen     = 0
        total_eligible = 0

        for ds_name, count in SAMPLE_COUNTS.items():
            path = os.path.join(GOLDEN_RECORDS_DIR, ds_name)
            if not os.path.exists(path):
                logger.warning(f"Dataset missing: {path} — skipping.")
                continue

            with open(path) as f:
                lines = f.readlines()
            total_seen += len(lines)

            valid_lines = []
            for line in lines:
                record = json.loads(line)
                rec_id = record["id"]

                # FIX: Robust Company Extraction
                if "company" in record.get("metadata", {}):
                    record["company"] = record["metadata"]["company"]
                elif ds_name.startswith("finqa") and "/" in rec_id:
                    try:
                        record["company"] = rec_id.split("_")[1].split("/")[0]
                    except:
                        record["company"] = None
                else:
                    record["company"] = record.get("company")

                if all(get_chunk_id(rec_id, chunk) in indexed_chunk_ids for chunk in record["context_chunks"]):
                    valid_lines.append(line)
            total_eligible += len(valid_lines)

            if not valid_lines:
                logger.warning(
                    f"{ds_name}: no fully-indexed records found. "
                    "Increase Phase 1 DEBUG_LIMIT or run without a limit."
                )
                continue

            rng = random.Random(RANDOM_SEED)
            rng.shuffle(valid_lines)
            sampled = valid_lines[:min(count, len(valid_lines))]

            n_dropped = len(lines) - len(valid_lines)
            logger.info(
                f"{ds_name}: {len(valid_lines)} eligible / {len(lines)} total "
                f"({n_dropped} dropped). Sampled {len(sampled)}."
            )

            for line in sampled:
                sample = json.loads(line)
                sample["source_ds"] = ds_name
                all_samples.append(sample)

        logger.info("=" * 60)
        logger.info("PHASE 2 ELIGIBILITY SUMMARY")
        logger.info(f"  Total records in gold datasets : {total_seen}")
        logger.info(f"  Fully-indexed (eligible)       : {total_eligible}")
        logger.info(f"  Dropped (chunks not in index)  : {total_seen - total_eligible}")
        logger.info(f"  Sampled for evaluation         : {len(all_samples)}")
        logger.info("=" * 60)

        return all_samples

    # ── Retrieval (cached to disk) ────────────────────────────────────────────

    async def _baseline_context(self, query: str, query_id: str, source_ds: str = "") -> Dict[str, Any]:
        """
        Vector-only retrieval. ufl_query=None skips the UFL leg in DualRetriever.
        Cached to disk — resume or re-run costs zero API calls.

        BUG B FIX: Detects and filters cross-dataset bleed.
        BUG 3 FIX: Only filter chunks whose metadata CONFIRMS a different
        dataset origin.  Chunks absent from chunk_metadata (meta={}) have
        unknown origin and are included rather than falsely dropped.
        """
        safe_id    = query_id.replace("/", "_")
        cache_path = os.path.join(RETRIEVAL_CACHE_DIR, f"{safe_id}_baseline.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        plan = RetrievalPlan(
            ufl_query=None,
            vector_hypothesis=query,
            vector_keywords=query.split()[:6],
            reasoning="Baseline vector-only plan — UFL leg disabled.",
        )
        results = await self.retriever.retrieve(
            plan,
            k=TOP_K,
            doc_id=query_id,
            include_all_ufl_for_chunks=False
        )

        text_chunk_bleed = False
        final_chunks     = []

        ds_prefix = source_ds.split("_")[0]
        if ds_prefix == "financebench":
            ds_prefix = "finbench"

        for c in results.get("text_chunks", []):
            meta = self.chunk_metadata.get(c.id, {})
            chunk_is_from_same_ds = any(
                r.startswith(ds_prefix) for r in meta.get("source_records", [])
            )
            if source_ds and meta and not chunk_is_from_same_ds:
                text_chunk_bleed = True
                logger.warning(f"Text Bleed: Chunk {c.id[:8]} not from {ds_prefix}. Filtering.")
                continue
            if source_ds and not meta:
                logger.debug(f"Chunk {c.id[:8]} absent from metadata; including with unknown origin.")
            final_chunks.append(c)

        filtered_chunks = [
            c for c in final_chunks
            if getattr(c, "relevance_score", 1.0) > 0.0
        ]

        context = self.assembler.assemble({"text_chunks": filtered_chunks, "ufl_rows": []})
        payload = {
            "context":             context,
            "prompt_tokens":       _estimate_tokens(context),
            "retrieved_chunk_ids": [c.id for c in filtered_chunks],
            "text_chunk_bleed":    text_chunk_bleed,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    async def _venra_context(self, query: str, query_id: str, source_ds: str = "", company: Optional[str] = None) -> Dict[str, Any]:
        """
        VeNRA full retrieval: Navigator → DualRetriever (UFL + Lexical Gate)
        → ContextAssembler. Cached to disk.

        BUG B FIX: Detects and filters cross-dataset bleed.
        BUG 3 FIX: Only filter chunks whose metadata CONFIRMS a different
        dataset origin.  Same rationale as _baseline_context above.
        """
        safe_id    = query_id.replace("/", "_")
        cache_path = os.path.join(RETRIEVAL_CACHE_DIR, f"{safe_id}_venra.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        plan = await self.navigator.navigate(query, doc_id=query_id)

        if plan.ufl_query and not plan.ufl_query.entity_ids:
            logger.warning(f"Navigator failed to resolve entity for {query_id}. Scoping to record ID.")

        results = await self.retriever.retrieve(
            plan,
            k=TOP_K,
            doc_id=query_id,
            company=company
        )

        ufl_rows        = results.get("ufl_rows", [])
        ufl_bleed       = results.get("meta", {}).get("ufl_bleed", False)
        first_pass_miss = results.get("meta", {}).get("first_pass_miss", False)

        text_chunk_bleed = False
        final_chunks     = []

        ds_prefix = source_ds.split("_")[0]
        if ds_prefix == "financebench":
            ds_prefix = "finbench"

        for c in results.get("text_chunks", []):
            meta = self.chunk_metadata.get(c.id, {})
            chunk_is_from_same_ds = any(
                r.startswith(ds_prefix) for r in meta.get("source_records", [])
            )
            if source_ds and meta and not chunk_is_from_same_ds:
                text_chunk_bleed = True
                logger.warning(f"Text Bleed: Chunk {c.id[:8]} not from {ds_prefix}. Filtering.")
                continue
            if source_ds and not meta:
                logger.debug(f"Chunk {c.id[:8]} absent from metadata; including with unknown origin.")
            final_chunks.append(c)

        filtered_chunks = [
            c for c in final_chunks
            if getattr(c, "relevance_score", 1.0) > 0.0
        ]

        context = self.assembler.assemble({"text_chunks": filtered_chunks, "ufl_rows": ufl_rows})
        payload = {
            "context":             context,
            "prompt_tokens":       _estimate_tokens(context),
            "retrieved_chunk_ids": [c.id for c in filtered_chunks],
            "ufl_row_ids":         [r.row_id for r in ufl_rows],
            "plan":                _pydantic_dump(plan),
            "ufl_bleed":           ufl_bleed,
            "text_chunk_bleed":    text_chunk_bleed,
            "ufl_row_count":       len(ufl_rows),
            "first_pass_miss":     first_pass_miss,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def _company_to_entity_id(self, company: str) -> str:
        """Mirror the canonicalization logic in build_global_index.py."""
        if not company or company in ("Global_Entity", ""):
            return "EXP_GLOBAL"
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", company)
        clean = re.sub(r"\s+", "_", clean.strip()).upper()
        return f"ID_{clean}"

    # ── Generation ────────────────────────────────────────────────────────────

    async def _run_cot(self, query: str, context: str) -> Dict[str, Any]:
        """
        Runs 1 & 2 — CoT baseline using the ACTIVE_MODEL.
        """
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

        try:
            if MODEL_TYPE == "GEMINI":
                safe_user_prompt = user_prompt[:30000]
                prompt_len = _estimate_tokens(safe_user_prompt)
                logger.info(f"Gemini CoT prompt size: {prompt_len} tokens")

                response = await asyncio.to_thread(
                    self._gemini.models.generate_content,
                    model=ACTIVE_MODEL,
                    contents=safe_user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=COT_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=4096,
                    ),
                )

                finish_reason = "UNKNOWN"
                if response and hasattr(response, "candidates") and response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)

                if not response or not hasattr(response, "text") or not response.text:
                    logger.warning(f"Gemini returned empty response for {ACTIVE_MODEL}. Finish reason: {finish_reason}")
                    raw_text = "ERROR: Empty response or safety block"
                else:
                    raw_text = response.text.strip()
            else:
                response = await self._groq.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=[
                        {"role": "system", "content": COT_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    response_model=None
                )
                raw_text     = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason

            lines  = raw_text.splitlines()
            answer = lines[-1].strip() if lines else "GENERATION_FAILURE"
        except Exception as e:
            logger.error(f"{MODEL_TYPE} CoT error: {e}")
            answer        = "TIMEOUT_FAILURE" if "timeout" in str(e).lower() else "GENERATION_FAILURE"
            raw_text      = "ERROR: " + str(e)
            finish_reason = "ERROR"

        return {
            "answer":        answer,
            "code_executed": False,
            "prompt_tokens": _estimate_tokens(COT_SYSTEM_PROMPT + "\n\n" + user_prompt),
            "full_response": {
                "raw_text":        raw_text,
                "reasoning_trace": raw_text,
                "finish_reason":   finish_reason,
            },
        }

    async def _run_pal(self, query: str, context: str) -> Dict[str, Any]:
        """
        Runs 3 & 4 — The VeNRA Hybrid Architecture.
        Pass 1: Reasoning/Code via ACTIVE_MODEL (Gemini or Qwen).
        Pass 2: Synthesis via LLAMA_MODEL (Llama-3.3-70b via Groq).

        FAIL-FAST: A single Pass 1 attempt is made. On any exception,
        reasoning=None and GENERATION_FAILURE is returned immediately.
        No retries, no sleep. If the model cannot generate on the first
        attempt the sample is marked T7 and the benchmark moves on.
        """
        augmented_context = PAL_CONTEXT_HEADER + context
        pass_1_prompt     = self.agent.pass_1_prompt
        user_prompt_1     = f"CONTEXT:\n{augmented_context}\n\nQUERY: {query}"

        reasoning       = None
        last_error      = "Unknown"
        finish_reason_1 = "UNKNOWN"

        # ── Single attempt — no retry, no sleep ───────────────────────────────
        try:
            if MODEL_TYPE == "GEMINI":
                safe_user_prompt_1 = user_prompt_1[:30000]
                prompt_len_1 = _estimate_tokens(safe_user_prompt_1)
                logger.info(f"Gemini PAL Pass 1 prompt size: {prompt_len_1} tokens")

                reasoning_resp = await asyncio.to_thread(
                    self._gemini.models.generate_content,
                    model=ACTIVE_MODEL,
                    contents=safe_user_prompt_1,
                    config=types.GenerateContentConfig(
                        system_instruction=pass_1_prompt,
                        temperature=0.0,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        response_schema=AgentReasoning,
                    ),
                )

                if reasoning_resp and hasattr(reasoning_resp, "candidates") and reasoning_resp.candidates:
                    finish_reason_1 = str(reasoning_resp.candidates[0].finish_reason)

                if not reasoning_resp or not hasattr(reasoning_resp, "parsed"):
                    raise ValueError(f"response.parsed missing. Finish reason: {finish_reason_1}")
                reasoning = reasoning_resp.parsed
            else:
                reasoning, completion = await self._groq.chat.completions.create_with_completion(
                    model=ACTIVE_MODEL,
                    messages=[
                        {"role": "system", "content": pass_1_prompt},
                        {"role": "user",   "content": user_prompt_1}
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    response_model=AgentReasoning
                )
                finish_reason_1 = completion.choices[0].finish_reason

            if not reasoning:
                raise ValueError("Empty reasoning generated")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"PAL Pass 1 failed (no retry): {e}")
            # reasoning remains None → outer try/except returns GENERATION_FAILURE

        try:
            if not reasoning:
                raise ValueError(
                    f"Pass 1 model ({ACTIVE_MODEL}) failed. Last error: {last_error}"
                )

            # ── Execute Code ──────────────────────────────────────────────────
            code_result = {"output": "No code run", "error": None}
            if reasoning.requires_math and reasoning.python_code:
                code_result = self.agent.executor.execute(reasoning.python_code)

            # ── Pass 2: Synthesis ─────────────────────────────────────────────
            user_prompt_2 = (
                f"QUERY: {query}\n"
                f"CONTEXT: {augmented_context}\n"
                f"REASONING: {reasoning.plan}\n"
                f"CODE_OUTPUT: {code_result['output']}\n"
                f"CODE_ERROR: {code_result['error'] if code_result['error'] else 'None'}\n"
            )
            final = await self.agent._call_fast_synthesis(user_prompt_2)

            if not final:
                raise ValueError(f"Pass 2 model ({LLAMA_MODEL}) failed to synthesize.")

            full = _pydantic_dump(final)
            full["reasoning_plan"]  = reasoning.plan
            full["python_code"]     = reasoning.python_code
            full["finish_reason_1"] = finish_reason_1
            full["raw_text"] = (
                f"PLAN: {reasoning.plan}\n\n"
                f"CODE:\n{reasoning.python_code}\n\n"
                f"FINAL: {final.answer}"
            )

            return {
                "answer":        final.answer,
                "code_executed": bool(reasoning.python_code and not code_result["error"]),
                "prompt_tokens": _estimate_tokens(augmented_context + query),
                "full_response": full,
            }

        except Exception as e:
            logger.error(f"Hybrid PAL error: {e}")
            return {
                "answer":        "GENERATION_FAILURE",
                "code_executed": None,   # Sentinel: tells extract_metrics this is T7
                "prompt_tokens": 0,
                "full_response": {"error": str(e), "raw_text": f"ERROR: {e}"},
            }

    # ── Benchmark loop ────────────────────────────────────────────────────────

    async def run_benchmark(self):
        samples = self.load_samples()
        if not samples:
            logger.error("No eligible samples found. Aborting benchmark.")
            return

        results: List[Dict[str, Any]] = []
        processed_ids: set            = set()

        current_config = {
            "gemini_model":  GEMINI_MODEL,
            "sample_counts": SAMPLE_COUNTS,
            "random_seed":   RANDOM_SEED,
            "top_k":         TOP_K,
        }

        if os.path.exists(GENERATION_RESULTS_PATH):
            try:
                with open(GENERATION_RESULTS_PATH) as f:
                    checkpoint = json.load(f)

                saved_config = checkpoint.get("_config")
                if saved_config is None:
                    logger.warning(
                        "Checkpoint has no _config stamp (older format). "
                        "Cannot verify config compatibility — proceeding anyway."
                    )
                elif saved_config != current_config:
                    logger.warning(
                        "Checkpoint config MISMATCH — existing results were "
                        "produced with a different configuration.\n"
                        f"  Saved  : {json.dumps(saved_config,  indent=4)}\n"
                        f"  Current: {json.dumps(current_config, indent=4)}\n"
                        "To start a clean run: delete the checkpoint file.\n"
                        "To resume safely: restore the original config values."
                    )

                results       = checkpoint.get("results", [])
                processed_ids = {r["sample_info"]["id"] for r in results}
                logger.info(
                    f"Checkpoint loaded: {len(processed_ids)} / {len(samples)} "
                    "samples already processed."
                )
            except Exception as e:
                logger.warning(f"Could not load checkpoint ({e}). Starting fresh.")

        remaining = len(samples) - len(processed_ids)
        logger.info(
            f"Benchmark target : {len(samples)} samples\n"
            f"Already done     : {len(processed_ids)}\n"
            f"Remaining        : {remaining}"
        )

        pbar = tqdm(total=len(samples), desc="Benchmark Progress")

        final_results: List[Dict[str, Any]] = list(results)
        existing_ids = {r["sample_info"]["id"] for r in final_results}

        for i, sample in enumerate(samples):
            query, query_id = sample["query"], sample["id"]
            source_ds = sample.get("source_ds", "")

            if query_id in existing_ids:
                pbar.update(1)
                pbar.set_postfix({"id": query_id[:8], "status": "exists"})
                continue

            raw_company = sample.get("company")
            company_id  = self._company_to_entity_id(raw_company) if raw_company else None

            baseline_ret, venra_ret = await asyncio.gather(
                self._baseline_context(query, query_id, source_ds),
                self._venra_context(query, query_id, source_ds, company=company_id),
            )

            first_pass_miss = (
                len(baseline_ret.get("retrieved_chunk_ids", [])) == 0 and
                (
                    len(venra_ret.get("retrieved_chunk_ids", [])) > 0 or
                    len(venra_ret.get("ufl_row_ids", [])) > 0
                )
            )
            venra_ret["first_pass_miss"] = first_pass_miss

            try:
                run_1, run_2, run_3, run_4 = await asyncio.wait_for(
                    asyncio.gather(
                        self._run_cot(query, baseline_ret["context"]),
                        self._run_cot(query, venra_ret["context"]),
                        self._run_pal(query, baseline_ret["context"]),
                        self._run_pal(query, venra_ret["context"]),
                    ),
                    timeout=120.0
                )
            except asyncio.TimeoutError:
                logger.error(f"Global generation timeout for {query_id}. Marking all as TIMEOUT.")
                timeout_resp = {
                    "answer":        "TIMEOUT_FAILURE",
                    "code_executed": False,
                    "prompt_tokens": 0,
                    "full_response": {"error": "Global timeout"},
                }
                run_1 = run_2 = run_3 = run_4 = timeout_resp

            record = {
                "sample_info": {**sample, "company": sample.get("company")},
                "token_parity": {
                    "baseline_ctx_tokens": baseline_ret.get("prompt_tokens", 0),
                    "venra_ctx_tokens":    venra_ret.get("prompt_tokens", 0),
                    "run_1_prompt_tokens": run_1.get("prompt_tokens", 0),
                    "run_2_prompt_tokens": run_2.get("prompt_tokens", 0),
                    "run_3_prompt_tokens": run_3.get("prompt_tokens", 0),
                    "run_4_prompt_tokens": run_4.get("prompt_tokens", 0),
                },
                "retrieval": {
                    "baseline": baseline_ret,
                    "venra":    venra_ret,
                },
                "runs": {
                    "run_1": run_1,
                    "run_2": run_2,
                    "run_3": run_3,
                    "run_4": run_4,
                },
            }

            final_results.append(record)

            with open(GENERATION_RESULTS_PATH, "w") as f:
                json.dump(
                    {"_config": current_config, "results": final_results},
                    f,
                    indent=2,
                )

            pbar.update(1)
            pbar.set_postfix({"id": query_id[:8]})

        pbar.close()
        logger.info(f"Benchmark complete → {GENERATION_RESULTS_PATH}")


if __name__ == "__main__":
    benchmark = UnifiedBenchmark()
    asyncio.run(benchmark.run_benchmark())