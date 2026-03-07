"""
experiments/phase2/utils.py
Shared utilities for VeNRA Phase 2 evaluation.
Imports only stdlib + numpy — works in both GPU and laptop environments.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Directory layout  (all output under data/exp/phase2/)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent.parent.parent.resolve()
PHASE2_ROOT   = PROJECT_ROOT / "data" / "exp" / "phase2"

DATA_DIR        = PHASE2_ROOT / "data"
PREDICTIONS_DIR = PHASE2_ROOT / "predictions"
LATENCY_DIR     = PHASE2_ROOT / "latency"
METRICS_DIR     = PHASE2_ROOT / "metrics"
ANALYSIS_DIR    = PHASE2_ROOT / "analysis"

MANIFEST_PATH      = DATA_DIR / "test_manifest.json"
PROMPTS_VENRA_PATH = DATA_DIR / "prompts_venra.jsonl"
PROMPTS_FRONTIER_PATH = DATA_DIR / "prompts_frontier.jsonl"

# Prediction file names (key → Path)
PRED_FILES = {
    "venra_salsa":        PREDICTIONS_DIR / "venra_salsa.jsonl",
    "base_qwen_zeroshot": PREDICTIONS_DIR / "base_qwen_zeroshot.jsonl",
    "base_qwen_cot":      PREDICTIONS_DIR / "base_qwen_cot.jsonl",
    "gemini_3_flash":     PREDICTIONS_DIR / "gemini_3_flash.jsonl",
    "kimi_k25_nvidia":    PREDICTIONS_DIR / "kimi_k25_nvidia.jsonl",
    "qwen3_32b_groq":     PREDICTIONS_DIR / "qwen3_32b_groq.jsonl",
    "llama33_70b_groq":   PREDICTIONS_DIR / "llama33_70b_groq.jsonl",
    "gpt_oss_120b_groq":  PREDICTIONS_DIR / "gpt_oss_120b_groq.jsonl",
}

# Label constants (ground truth integers)
GT_SUPPORTED = 0   # "Found"
GT_UNFOUNDED = 1   # "Fake"
GT_GENERAL   = 2   # "General"

# Maps first-token text → ground truth integer
VALID_MAP = {
    "found":     GT_SUPPORTED,
    "supported": GT_SUPPORTED,
    "fake":      GT_UNFOUNDED,
    "unfounded": GT_UNFOUNDED,
    "general":   GT_GENERAL,
}

SABOTAGE_TYPES = [
    "logic_code_lie",
    "numeric_neighbor_trap",
    "irrelevancy_rag",
    "semantic_drift",
]

# Pool tags
PAIR_PARENT_TAGS = {"pair_short_parent", "pair_long_parent"}
PAIR_CHILD_TAGS  = {"pair_short_child",  "pair_long_child"}
ALL_PAIR_TAGS    = PAIR_PARENT_TAGS | PAIR_CHILD_TAGS


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in [DATA_DIR, PREDICTIONS_DIR, LATENCY_DIR, METRICS_DIR, ANALYSIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> Tuple[int, bool]:
    """
    Parse the first-token response from any model.
    Returns (pred_int, is_valid). pred_int=-1 when invalid.
    """
    if not raw or not raw.strip():
        return -1, False
    first_word = raw.strip().split()[0].strip('.,!?:;"\'`').lower()
    pred = VALID_MAP.get(first_word, -1)
    return pred, pred != -1


def parse_cot_response(raw: str) -> Tuple[int, bool]:
    """
    Parse CoT output: find the LAST valid label word in the generated text.
    Returns (pred_int, is_valid).
    """
    if not raw or not raw.strip():
        return -1, False
    words = [w.strip('.,!?:;"\'`').lower() for w in raw.strip().split()]
    for word in reversed(words):
        if word in VALID_MAP:
            return VALID_MAP[word], True
    return -1, False


# ---------------------------------------------------------------------------
# Posterior mean — EXACT copy from train.py SabotageEvalCallback
# ---------------------------------------------------------------------------

def posterior_mean(
    m_times_n: float,
    n: int,
    prior_a: float = 1e-6,
    prior_b: float = 1.0,
) -> float:
    if n <= 0:
        return 0.0
    return (prior_a + m_times_n) / (prior_a + prior_b + n)


# ---------------------------------------------------------------------------
# JSONL I/O with crash-safe caching
# ---------------------------------------------------------------------------

def write_prediction(path: Path, row: Dict[str, Any]) -> None:
    """Append one prediction row to JSONL and flush immediately."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def get_completed_ids(path: Path) -> Set[int]:
    """Return set of row_ids already written to a prediction file."""
    if not path.exists():
        return set()
    ids: Set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["row_id"])
                except Exception:
                    pass
    return ids


# ---------------------------------------------------------------------------
# Manifest / prompt loading
# ---------------------------------------------------------------------------

def load_manifest() -> List[Dict[str, Any]]:
    """Full manifest load — includes input_components/output_components.
    Use only when you need the raw text (e.g. building prompts).
    For metrics, use load_manifest_slim() instead."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_manifest_slim() -> List[Dict[str, Any]]:
    """
    Lightweight manifest loader for metrics computation.
    Reads only the 5 fields needed for pool routing and scoring,
    skipping input_components / output_components (which hold full
    financial contexts and can make the manifest 30-80 MB).

    Fields returned per row:
      row_id, pool, ground_truth, sabotage_type, cot_subsample
    """
    KEEP = {"row_id", "pool", "ground_truth", "sabotage_type", "cot_subsample",
            "family_id", "meta_token_count", "label"}
    with open(MANIFEST_PATH) as f:
        full = json.load(f)
    return [{k: v for k, v in row.items() if k in KEEP} for row in full]


def load_prompts_venra() -> Dict[int, str]:
    """row_id → prompt_text"""
    out: Dict[int, str] = {}
    with open(PROMPTS_VENRA_PATH) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["row_id"]] = row["prompt_text"]
    return out


def load_prompts_frontier() -> Dict[int, Dict[str, str]]:
    """row_id → {system_content, user_content}"""
    out: Dict[int, Dict[str, str]] = {}
    with open(PROMPTS_FRONTIER_PATH) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["row_id"]] = row
    return out


def load_predictions(model_key: str) -> Dict[int, Dict[str, Any]]:
    """Load all predictions for a model keyed by row_id."""
    path = PRED_FILES.get(model_key)
    if path is None or not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                    out[row["row_id"]] = row
                except Exception:
                    pass
    return out


# ---------------------------------------------------------------------------
# Pool index helpers
# ---------------------------------------------------------------------------

def build_pool_arrays(
    manifest: List[Dict],
    preds_by_id: Dict[int, Dict],
) -> Dict[str, Any]:
    """
    Build numpy arrays aligned with manifest row_ids.
    Returns a dict of named index arrays and prediction/truth arrays.
    Invalid or missing predictions are scored as wrong (pred=-1).
    """
    n = len(manifest)
    preds   = np.full(n, -1, dtype=np.int32)
    truths  = np.zeros(n, dtype=np.int32)
    confs   = np.full(n, 0.5, dtype=np.float32)
    valid   = np.zeros(n, dtype=bool)
    pools   = np.array([r["pool"] for r in manifest])
    sab_types = np.array([r.get("sabotage_type", "unknown") for r in manifest])

    for i, row in enumerate(manifest):
        truths[i] = row["ground_truth"]
        pred_row = preds_by_id.get(row["row_id"])
        if pred_row is not None:
            preds[i]  = pred_row.get("pred", -1)
            valid[i]  = pred_row.get("valid", False)
            confs[i]  = pred_row.get("confidence", 0.5)

    idx = {
        "pair_short_parent": np.where(pools == "pair_short_parent")[0],
        "pair_short_child":  np.where(pools == "pair_short_child")[0],
        "pair_long_parent":  np.where(pools == "pair_long_parent")[0],
        "pair_long_child":   np.where(pools == "pair_long_child")[0],
        "natural_fake":      np.where(pools == "natural_fake")[0],
        "natural_true":      np.where(pools == "natural_true")[0],
        "axiom":             np.where(pools == "axiom")[0],
    }

    return {
        "preds": preds, "truths": truths, "confs": confs,
        "valid": valid, "pools": pools, "sab_types": sab_types,
        "idx": idx,
    }