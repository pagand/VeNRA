"""
experiments/phase2/compute_metrics.py
----------------------------------------
Compute all evaluation metrics from cached prediction files.
Works with partial results — skips any model whose file is missing.

Metric definitions mirror SabotageEvalCallback EXACTLY:
  - Flip Rate:       both parent (pred==0) AND child (pred==1) correct simultaneously
  - Recall nat-fake: pred==1 on natural_fake pool
  - TPR nat-true:    pred==0 on natural_true pool
  - FPR nat-true:    pred==1 on natural_true pool  ← Paranoid Sycophant check
  - Axiom Accuracy:  pred==2 on axiom pool
  - Composite M:     (pm_fr^0.5)(pm_recall^0.5)(pm_tpr^0.5)(pm_axiom^0.5)
  posterior_mean() copied verbatim from train.py (prior_a=1e-6, prior_b=1.0)

CoT comparison uses base_qwen_cot predictions (full test set, same as all others).
The only CoT-specific metric is token_budget — how many tokens the base model
needed to reach a verdict that VeNRA reaches in 1 token.

Outputs:
  data/exp/phase2/metrics/all_models_metrics.json
  Prints formatted tables to stdout.

NOTE on combining results from two machines:
  GPU server writes:  venra_salsa.jsonl, base_qwen_zeroshot.jsonl,
                      base_qwen_cot.jsonl
  Laptop writes:      gemini_3_flash.jsonl, kimi_k25_nvidia.jsonl,
                      qwen3_32b_groq.jsonl, llama33_70b_groq.jsonl
  Before running this script, copy all prediction files to the same
  data/exp/phase2/predictions/ directory on one machine.

Usage (either machine):
  python -m experiments.phase2.compute_metrics
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.phase2.utils import (
    ALL_PAIR_TAGS,
    METRICS_DIR,
    PRED_FILES,
    SABOTAGE_TYPES,
    build_pool_arrays,
    ensure_dirs,
    load_manifest_slim,
    load_predictions,
    posterior_mean,
)

# Canonical model order for all output tables
ALL_MODELS = [
    ("venra_salsa",        "VeNRA 3B SALSA"),
    ("base_qwen_zeroshot", "Base Qwen 3B (zero-shot)"),
    ("base_qwen_cot",      "Base Qwen 3B (CoT)"),
    ("gemini_3_flash",     "Gemini-3-Flash-Preview"),
    ("kimi_k25_nvidia",    "Kimi K2.5 (NVIDIA NIM)"),
    ("qwen3_32b_groq",     "Qwen3-32B (Groq)"),
    ("llama33_70b_groq",   "Llama 3.3 70B (Groq)"),
]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics_for_model(
    model_key:  str,
    manifest:   List[Dict],
) -> Optional[Dict[str, Any]]:
    """
    Compute all metrics for one model against the full test manifest.
    Returns None if prediction file not found or empty.
    """
    preds_by_id = load_predictions(model_key)
    if not preds_by_id:
        return None

    arrays   = build_pool_arrays(manifest, preds_by_id)
    preds    = arrays["preds"]
    idx      = arrays["idx"]
    sab_np   = arrays["sab_types"]
    pools    = arrays["pools"]

    # ── Pair pools ────────────────────────────────────────────────────────────
    ps_par = idx["pair_short_parent"]
    ps_chi = idx["pair_short_child"]
    pl_par = idx["pair_long_parent"]
    pl_chi = idx["pair_long_child"]

    def _flip(par_idx, chi_idx):
        n = len(par_idx)
        if n == 0:
            return 0.0, 0
        correct = int(np.sum((preds[par_idx] == 0) & (preds[chi_idx] == 1)))
        return correct / n, n

    fr_short, n_short = _flip(ps_par, ps_chi)
    fr_long,  n_long  = _flip(pl_par, pl_chi)
    n_fr = n_short + n_long
    s_fr = fr_short * n_short + fr_long * n_long
    fr_global = s_fr / n_fr if n_fr > 0 else 0.0

    # ── Natural + axiom pools ─────────────────────────────────────────────────
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

    # ── Composite M — EXACT from train.py ────────────────────────────────────
    pm_fr     = posterior_mean(s_fr,                  n_fr)
    pm_recall = posterior_mean(recall_natural * n_nf, n_nf)
    pm_tpr    = posterior_mean(tpr_clean * n_nt,      n_nt)
    pm_axiom  = posterior_mean(acc_axiom * n_ax,      n_ax)
    composite = (pm_fr**0.5) * (pm_recall**0.5) * (pm_tpr**0.5) * (pm_axiom**0.5)

    # ── Per-sabotage flip rate ────────────────────────────────────────────────
    fr_by_sabotage: Dict[str, float] = {}
    for st in SABOTAGE_TYPES:
        child_mask = (sab_np == st) & np.isin(
            pools, ["pair_short_child", "pair_long_child"]
        )
        child_idx  = np.where(child_mask)[0]
        parent_idx = child_idx - 1   # layout guarantee: parent always at child-1
        # Sanity: verify parents are actually parent-tagged
        valid_pair = np.isin(
            pools[parent_idx], ["pair_short_parent", "pair_long_parent"]
        )
        child_idx  = child_idx[valid_pair]
        parent_idx = parent_idx[valid_pair]
        n_st = len(child_idx)
        fr_st = float(
            np.sum((preds[parent_idx] == 0) & (preds[child_idx] == 1)) / n_st
        ) if n_st > 0 else 0.0
        fr_by_sabotage[st] = round(fr_st, 4)

    # ── Validity rate ─────────────────────────────────────────────────────────
    row_ids       = [r["row_id"] for r in manifest]
    total_covered = sum(1 for rid in row_ids if rid in preds_by_id)
    valid_count   = sum(
        1 for rid in row_ids
        if preds_by_id.get(rid, {}).get("valid", False)
    )
    validity_rate = valid_count / total_covered if total_covered > 0 else 0.0

    return {
        "validity_rate":  round(validity_rate, 4),
        "fr_short":       round(fr_short, 4),
        "fr_long":        round(fr_long, 4),
        "fr_global":      round(fr_global, 4),
        "recall_natural": round(recall_natural, 4),
        "tpr_clean":      round(tpr_clean, 4),
        "fpr_clean":      round(fpr_clean, 4),
        "acc_axiom":      round(acc_axiom, 4),
        "composite":      round(float(composite), 6),
        "fr_by_sabotage": fr_by_sabotage,
        "n": {
            "pairs_short":    n_short,
            "pairs_long":     n_long,
            "natural_fake":   n_nf,
            "natural_true":   n_nt,
            "axioms":         n_ax,
            "total_covered":  total_covered,
        },
    }


def compute_cot_budget_stats(manifest: List[Dict]) -> Optional[Dict]:
    """
    Summarise token_budget distribution for base model CoT predictions.
    Every row that has a token_budget entry is included.
    """
    preds_by_id = load_predictions("base_qwen_cot")
    if not preds_by_id:
        return None

    all_ids = {r["row_id"] for r in manifest}
    budgets = [
        v["token_budget"]
        for rid, v in preds_by_id.items()
        if rid in all_ids and "token_budget" in v
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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_main_table(results: Dict[str, Any]) -> None:
    ordered = [k for k, _ in ALL_MODELS if k in results]

    print("\n" + "=" * 113)
    print("VENRA PHASE 2 — JUDGE SHOWDOWN RESULTS")
    print("=" * 113)
    print(
        f"{'Model':<32} {'Valid':>6} {'FR(S)':>6} {'FR(L)':>6} "
        f"{'FR':>6} {'Recall':>7} {'TPR':>6} {'FPR':>6} {'Axiom':>6} {'M':>8}"
    )
    print("-" * 113)
    for key in ordered:
        m    = results[key]
        name = m["display_name"][:31]
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
    print("=" * 113)
    print(
        "FR(S/L)=Flip Rate Short/Long context  FR=Global Flip Rate  "
        "Recall=Natural Hallucination Recall\n"
        "TPR=True Positive Rate (clean docs)   FPR=False Alarm Rate  "
        "Axiom=General Knowledge  M=Composite"
    )


def print_sabotage_table(results: Dict[str, Any]) -> None:
    ordered = [k for k, _ in ALL_MODELS if k in results]
    col_w   = 18
    print("\n" + "=" * (32 + col_w * len(SABOTAGE_TYPES)))
    print("PER-SABOTAGE TYPE FLIP RATE")
    print("=" * (32 + col_w * len(SABOTAGE_TYPES)))
    header = f"{'Model':<32}" + "".join(f"{st:>{col_w}}" for st in SABOTAGE_TYPES)
    print(header)
    print("-" * (32 + col_w * len(SABOTAGE_TYPES)))
    for key in ordered:
        m    = results[key]
        name = m["display_name"][:31]
        sab  = m.get("fr_by_sabotage", {})
        row  = f"{name:<32}"
        row += "".join(f"{_pct(sab.get(st, 0.0)):>{col_w}}" for st in SABOTAGE_TYPES)
        print(row)
    print("=" * (32 + col_w * len(SABOTAGE_TYPES)))


def print_cot_table(
    results: Dict[str, Any],
    budget:  Optional[Dict],
) -> None:
    """
    Side-by-side comparison of VeNRA SALSA vs Base zero-shot vs Base CoT.
    All three now run on the full test set so full M is available.
    The token budget column shows the key difference: 1 vs N tokens.
    """
    COT_MODELS = ["venra_salsa", "base_qwen_zeroshot", "base_qwen_cot"]
    available  = [k for k in COT_MODELS if k in results]

    if not available:
        return

    print("\n" + "=" * 80)
    print("BASE CoT vs SALSA COMPARISON  (1.5 Thinking Claim)")
    print("VeNRA SALSA achieves in 1 token what the base model needs N tokens for")
    print("=" * 80)
    print(f"{'Model':<36} {'Tokens':>8} {'FR':>8} {'M':>10}")
    print("-" * 80)

    for key in available:
        m     = results[key]
        name  = m["display_name"][:35]
        fr    = _pct(m["fr_global"])
        comp  = f"{m['composite']:.4f}"
        if key in ("venra_salsa", "base_qwen_zeroshot"):
            tok_str = "1"
        else:
            tok_str = f"~{budget['median']:.0f}" if budget else "?"
        print(f"{name:<36} {tok_str:>8} {fr:>8} {comp:>10}")

    print("=" * 80)

    if budget:
        salsa_fr = results.get("venra_salsa",        {}).get("fr_global", 0)
        cot_fr   = results.get("base_qwen_cot",      {}).get("fr_global", 0)
        base_fr  = results.get("base_qwen_zeroshot",  {}).get("fr_global", 0)
        print(f"\n  Token budget (Base CoT, n={budget['n']} rows):")
        print(f"    Median {budget['median']:.0f} tokens  |  "
              f"P95 {budget['p95']:.0f} tokens  |  Max {budget['max']:.0f} tokens")
        if cot_fr > 0:
            print(f"\n  VeNRA SALSA flip rate / Base CoT flip rate = "
                  f"{salsa_fr/cot_fr:.2f}×  "
                  f"(at 1 token vs ~{budget['median']:.0f} tokens)")
        if base_fr > 0:
            print(f"  VeNRA SALSA flip rate / Base zero-shot flip rate = "
                  f"{salsa_fr/base_fr:.2f}×  "
                  f"(fine-tuning contribution, both at 1 token)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("[load] Loading manifest (slim, skipping context data)...")
    manifest = load_manifest_slim()
    print(f"[load] {len(manifest)} rows  "
          f"({sum(1 for r in manifest if r['pool'].endswith('parent'))} pairs, "
          f"{sum(1 for r in manifest if r['pool']=='natural_fake')} nat-fake, "
          f"{sum(1 for r in manifest if r['pool']=='natural_true')} nat-true, "
          f"{sum(1 for r in manifest if r['pool']=='axiom')} axioms)")

    full_results: Dict[str, Any] = {}

    for model_key, display_name in ALL_MODELS:
        pred_path = PRED_FILES.get(model_key)
        if pred_path and not pred_path.exists():
            print(f"[skip] {display_name:<35} — prediction file not found")
            continue

        print(f"[calc] {display_name:<35} ...", end=" ", flush=True)
        m = compute_metrics_for_model(model_key, manifest)
        if m is None:
            print("no data")
            continue

        m["display_name"] = display_name
        full_results[model_key] = m
        print(
            f"M={m['composite']:.4f}  "
            f"FR={_pct(m['fr_global'])}  "
            f"covered={m['n']['total_covered']}/{len(manifest)}"
        )

    # ── CoT budget stats ──────────────────────────────────────────────────────
    print("\n[calc] Token budget stats (Base CoT)...", end=" ", flush=True)
    budget_stats = compute_cot_budget_stats(manifest)
    if budget_stats:
        print(f"median={budget_stats['median']:.0f} tokens, n={budget_stats['n']}")
    else:
        print("no data")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = METRICS_DIR / "all_models_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {"full_test_set": full_results, "cot_budget_stats": budget_stats},
            f, indent=2,
        )
    print(f"\n[save] → {out_path}")

    # ── Print tables ──────────────────────────────────────────────────────────
    if full_results:
        print_main_table(full_results)
        print_sabotage_table(full_results)
        print_cot_table(full_results, budget_stats)
    else:
        print("\n[warn] No complete prediction files found. Run inference scripts first.")
        print("       GPU:    run_venra_gpu.py  run_base_model_gpu.py  run_base_cot_gpu.py")
        print("       Laptop: run_frontier_api.py --model {gemini,kimi,qwen3,llama70b}")
        print("       Then copy all predictions/ files to one machine before compute_metrics.")


if __name__ == "__main__":
    main()