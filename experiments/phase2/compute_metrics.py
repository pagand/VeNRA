"""
experiments/phase2/compute_metrics.py
----------------------------------------
Compute all evaluation metrics from cached prediction files.
Works for partial results — reports metrics for whatever models are complete.

Metric definitions mirror SabotageEvalCallback EXACTLY:
  - Flip Rate: both parent (pred==0) AND child (pred==1) correct simultaneously
  - Recall (natural fake): pred==1 on natural_fake pool
  - TPR (natural true):    pred==0 on natural_true pool
  - FPR (natural true):    pred==1 on natural_true pool  ← Paranoid Sycophant check
  - Axiom Accuracy:        pred==2 on axiom pool
  - Composite:             (pm_fr^0.5) * (pm_recall^0.5) * (pm_tpr^0.5) * (pm_axiom^0.5)
  - posterior_mean() is copied verbatim from train.py

Subsample metrics (CoT comparison) use only rows with cot_subsample=True.

Outputs:
  data/exp/phase2/metrics/all_models_metrics.json
  Prints formatted comparison table to stdout.

Usage (either environment):
  python -m experiments.phase2.compute_metrics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_predictions,
    build_pool_arrays, posterior_mean,
    PRED_FILES, METRICS_DIR, SABOTAGE_TYPES,
    GT_SUPPORTED, GT_UNFOUNDED, GT_GENERAL,
)

# All models in display order
ALL_MODELS = [
    ("venra_salsa",        "VeNRA 3B SALSA"),
    ("base_qwen_zeroshot", "Base Qwen 3B (zero-shot)"),
    ("base_qwen_cot",      "Base Qwen 3B (CoT, subsample)"),
    ("gemini_3_flash",     "Gemini-3-Flash-Preview"),
    ("kimi_k25_nvidia",    "Kimi K2.5 (NVIDIA NIM)"),
    ("qwen3_32b_groq",     "Qwen3-32B (Groq)"),
    ("llama33_70b_groq",   "Llama 3.3 70B (Groq)"),
]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics_for_model(
    model_key: str,
    manifest:  List[Dict],
    use_subsample: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Compute all metrics for one model.
    use_subsample=True restricts to cot_subsample rows (for CoT comparison table).
    Returns None if prediction file not found or empty.
    """
    preds_by_id = load_predictions(model_key)
    if not preds_by_id:
        return None

    # Optionally filter manifest to subsample rows only
    active_manifest = manifest
    if use_subsample:
        active_manifest = [r for r in manifest if r.get("cot_subsample", False)]
        if not active_manifest:
            return None

    arrays = build_pool_arrays(active_manifest, preds_by_id)
    preds   = arrays["preds"]
    truths  = arrays["truths"]
    idx     = arrays["idx"]
    sab_np  = arrays["sab_types"]
    pools   = arrays["pools"]

    # ── Pair pools ────────────────────────────────────────────────────────────
    ps_par = idx["pair_short_parent"]
    ps_chi = idx["pair_short_child"]
    pl_par = idx["pair_long_parent"]
    pl_chi = idx["pair_long_child"]

    n_short = len(ps_par)
    n_long  = len(pl_par)

    def flip_rate(par_idx, chi_idx):
        n = len(par_idx)
        if n == 0:
            return 0.0, 0
        correct = int(np.sum((preds[par_idx] == 0) & (preds[chi_idx] == 1)))
        return correct / n, n

    fr_short, n_fr_short = flip_rate(ps_par, ps_chi)
    fr_long,  n_fr_long  = flip_rate(pl_par, pl_chi)

    n_fr = n_fr_short + n_fr_long
    s_fr = fr_short * n_fr_short + fr_long * n_fr_long
    fr_global = s_fr / n_fr if n_fr > 0 else 0.0

    # Weighted flip rate (same as callback)
    def weighted(a, na, b, nb):
        denom = na + nb
        return (a * na + b * nb) / denom if denom > 0 else 0.0

    # ── Natural pools ─────────────────────────────────────────────────────────
    nat_fake_idx = idx["natural_fake"]
    nat_true_idx = idx["natural_true"]
    axiom_idx    = idx["axiom"]

    n_nf = len(nat_fake_idx)
    n_nt = len(nat_true_idx)
    n_ax = len(axiom_idx)

    recall_natural = float(np.mean(preds[nat_fake_idx] == 1)) if n_nf > 0 else 0.0
    tpr_clean      = float(np.mean(preds[nat_true_idx] == 0)) if n_nt > 0 else 0.0
    fpr_clean      = float(np.mean(preds[nat_true_idx] == 1)) if n_nt > 0 else 0.0
    acc_axiom      = float(np.mean(preds[axiom_idx]    == 2)) if n_ax > 0 else 0.0

    # ── Composite — EXACT from train.py ──────────────────────────────────────
    pm_fr     = posterior_mean(s_fr,              n_fr)
    pm_recall = posterior_mean(recall_natural * n_nf, n_nf)
    pm_tpr    = posterior_mean(tpr_clean * n_nt,  n_nt)
    pm_axiom  = posterior_mean(acc_axiom * n_ax,  n_ax)

    composite = (pm_fr**0.5) * (pm_recall**0.5) * (pm_tpr**0.5) * (pm_axiom**0.5)

    # ── Per-sabotage flip rate ─────────────────────────────────────────────────
    fr_by_sabotage: Dict[str, float] = {}
    for st in SABOTAGE_TYPES:
        # Get child rows of this sabotage type in pairs
        child_mask = (sab_np == st) & np.isin(pools, ["pair_short_child", "pair_long_child"])
        child_idx  = np.where(child_mask)[0]
        # Parent is always immediately before child in flat layout (row_id offset)
        parent_idx = child_idx - 1
        # Sanity: ensure parents are actually parents
        valid_pair = np.isin(pools[parent_idx],
                             ["pair_short_parent", "pair_long_parent"])
        child_idx  = child_idx[valid_pair]
        parent_idx = parent_idx[valid_pair]
        n_st = len(child_idx)
        if n_st > 0:
            fr_st = float(np.sum(
                (preds[parent_idx] == 0) & (preds[child_idx] == 1)
            ) / n_st)
        else:
            fr_st = 0.0
        fr_by_sabotage[st] = round(fr_st, 4)

    # ── Validity ──────────────────────────────────────────────────────────────
    total_coverage = sum(1 for rid in [r["row_id"] for r in active_manifest]
                         if rid in preds_by_id)
    valid_count    = sum(1 for rid in [r["row_id"] for r in active_manifest]
                         if preds_by_id.get(rid, {}).get("valid", False))
    validity_rate  = valid_count / total_coverage if total_coverage > 0 else 0.0

    return {
        "validity_rate":    round(validity_rate, 4),
        "fr_short":         round(fr_short, 4),
        "fr_long":          round(fr_long, 4),
        "fr_global":        round(fr_global, 4),
        "recall_natural":   round(recall_natural, 4),
        "tpr_clean":        round(tpr_clean, 4),
        "fpr_clean":        round(fpr_clean, 4),
        "acc_axiom":        round(acc_axiom, 4),
        "composite":        round(float(composite), 6),
        "fr_by_sabotage":   fr_by_sabotage,
        "n": {
            "pairs_short": n_fr_short,
            "pairs_long":  n_fr_long,
            "natural_fake": n_nf,
            "natural_true": n_nt,
            "axioms":       n_ax,
            "total_coverage": total_coverage,
        },
    }


def compute_cot_budget_stats(manifest: List[Dict]) -> Optional[Dict]:
    """Summarise token budget distribution for base model CoT subsample."""
    preds_by_id = load_predictions("base_qwen_cot")
    if not preds_by_id:
        return None

    cot_ids = {r["row_id"] for r in manifest if r.get("cot_subsample", False)}
    budgets = [
        v["token_budget"]
        for k, v in preds_by_id.items()
        if k in cot_ids and "token_budget" in v
    ]
    if not budgets:
        return None

    arr = np.array(budgets)
    return {
        "n":      len(arr),
        "median": round(float(np.median(arr)), 1),
        "p25":    round(float(np.percentile(arr, 25)), 1),
        "p75":    round(float(np.percentile(arr, 75)), 1),
        "p95":    round(float(np.percentile(arr, 95)), 1),
        "max":    round(float(arr.max()), 1),
        "mean":   round(float(arr.mean()), 1),
    }


def compute_subsample_comparison(manifest: List[Dict]) -> Dict:
    """
    On the 50-pair CoT subsample compare flip rate for:
      - VeNRA SALSA       (1 token, fine-tuned)
      - Base Qwen zeroshot (1 token, Label: conditioning)
      - Base Qwen CoT     (N tokens, free generation)

    IMPORTANT: the subsample manifest contains ONLY pair rows (cot_subsample
    is True exclusively on pair pool rows after the Bug 2 fix in build_manifest).
    That means natural_fake / natural_true / axiom pools are EMPTY in the
    subsample, so posterior_mean(0, 0)=0 for those pillars and composite=0
    for every model — useless and misleading. We therefore report only flip
    rate here; the full composite lives in the main results table.
    """
    results = {}
    for model_key, display_name in [
        ("venra_salsa",        "VeNRA SALSA (1 token)"),
        ("base_qwen_zeroshot", "Base Qwen zero-shot (1 token)"),
        ("base_qwen_cot",      "Base Qwen CoT (N tokens)"),
    ]:
        m = compute_metrics_for_model(model_key, manifest, use_subsample=True)
        if m:
            results[model_key] = {
                "display":   display_name,
                "fr_global": m["fr_global"],
                "fr_short":  m["fr_short"],
                "fr_long":   m["fr_long"],
                # composite intentionally omitted — only pairs available in subsample
                "n_pairs":   m["n"]["pairs_short"] + m["n"]["pairs_long"],
            }
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_main_table(results: Dict[str, Any]) -> None:
    # Maintain the canonical model order from ALL_MODELS
    ordered_keys = [k for k, _ in ALL_MODELS if k in results and k != "base_qwen_cot"]

    print("\n" + "="*110)
    print("VENRA PHASE 2 — JUDGE SHOWDOWN RESULTS")
    print("="*110)
    header = (
        f"{'Model':<32} {'Valid':>6} {'FR(S)':>6} {'FR(L)':>6} "
        f"{'FR':>6} {'Recall':>7} {'TPR':>6} {'FPR':>6} {'Axiom':>6} {'M':>8}"
    )
    print(header)
    print("-"*110)

    for model_key in ordered_keys:
        m    = results[model_key]
        name = m.get("display_name", model_key)[:31]
        print(
            f"{name:<32} "
            f"{_pct(m['validity_rate']):>6} "
            f"{_pct(m['fr_short']):>6} "
            f"{_pct(m['fr_long']):>6} "
            f"{_pct(m['fr_global']):>6} "
            f"{_pct(m['recall_natural']):>7} "
            f"{_pct(m['tpr_clean']):>6} "
            f"{_pct(m['fpr_clean']):>6} "
            f"{_pct(m['acc_axiom']):>6} "
            f"{m['composite']:>8.4f}"
        )

    print("="*110)
    print("Columns: Valid=Validity Rate, FR(S/L)=Flip Rate Short/Long, FR=Global Flip Rate,")
    print("         Recall=Natural Fake Recall, TPR=True Positive Rate (clean), FPR=False Alarm Rate,")
    print("         Axiom=General Knowledge Accuracy, M=Composite Score (posterior-mean form)")


def print_sabotage_table(results: Dict[str, Any]) -> None:
    ordered_keys = [k for k, _ in ALL_MODELS if k in results and k != "base_qwen_cot"]
    print("\n" + "="*90)
    print("PER-SABOTAGE FLIP RATE BREAKDOWN")
    print("="*90)
    header = (
        f"{'Model':<32} "
        + " ".join(f"{st[:14]:>16}" for st in SABOTAGE_TYPES)
    )
    print(header)
    print("-"*90)
    for model_key in ordered_keys:
        m    = results[model_key]
        name = m.get("display_name", model_key)[:31]
        sab  = m.get("fr_by_sabotage", {})
        row  = f"{name:<32} "
        row += " ".join(f"{_pct(sab.get(st, 0.0)):>16}" for st in SABOTAGE_TYPES)
        print(row)
    print("="*90)


def print_cot_comparison(subsample: Dict, budget: Optional[Dict]) -> None:
    print("\n" + "="*70)
    print("SUBSAMPLE CoT COMPARISON (50 pairs, seed=42)")
    print("(Claim: VeNRA SALSA at 1 token ≈ Base CoT at N tokens)")
    print("="*70)
    header = f"{'Model':<40} {'FR':>8} {'Tokens':>10}"
    print(header)
    print("-"*70)
    for model_key, m in subsample.items():
        budget_str = "1" if "salsa" in model_key or "zeroshot" in model_key \
                     else (f"~{budget['median']:.0f}" if budget else "?")
        print(f"{m['display']:<40} "
              f"{_pct(m['fr_global']):>8} "
              f"{budget_str:>10}")
    print("="*70)
    if budget:
        print(f"\n  Base CoT token budget: "
              f"median={budget['median']}, P95={budget['p95']}, max={budget['max']}")
        salsa_fr  = subsample.get("venra_salsa", {}).get("fr_global", 0)
        cot_fr    = subsample.get("base_qwen_cot", {}).get("fr_global", 0)
        if cot_fr > 0:
            efficiency = salsa_fr / cot_fr
            print(f"  VeNRA/CoT flip rate ratio: {efficiency:.2f}× "
                  f"({'better' if efficiency >= 1 else 'worse'} at 1/~{budget['median']:.0f} tokens)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("[load] Loading manifest...")
    manifest = load_manifest()
    print(f"[load] {len(manifest)} test rows")

    # ── Full test set metrics ─────────────────────────────────────────────────
    full_results: Dict[str, Any] = {}

    for model_key, display_name in ALL_MODELS:
        if model_key == "base_qwen_cot":
            continue   # CoT is subsample-only; handled below
        m = compute_metrics_for_model(model_key, manifest, use_subsample=False)
        if m is None:
            print(f"[skip] {display_name} — no prediction file found")
            continue
        m["display_name"] = display_name
        full_results[model_key] = m
        print(f"[ok]   {display_name} — composite={m['composite']:.4f}")

    # ── CoT subsample comparison ──────────────────────────────────────────────
    subsample_results = compute_subsample_comparison(manifest)
    budget_stats      = compute_cot_budget_stats(manifest)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "full_test_set":       full_results,
        "cot_subsample":       subsample_results,
        "cot_budget_stats":    budget_stats,
    }

    out_path = METRICS_DIR / "all_models_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[save] Metrics → {out_path}")

    # ── Print tables ──────────────────────────────────────────────────────────
    if full_results:
        print_main_table(full_results)
        print_sabotage_table(full_results)

    if subsample_results:
        print_cot_comparison(subsample_results, budget_stats)


if __name__ == "__main__":
    main()