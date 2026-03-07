"""
experiments/phase2/generate_report.py
----------------------------------------
Generate all paper-ready figures and tables from cached metrics + latency data.

Reads:
  data/exp/phase2/metrics/all_models_metrics.json
  data/exp/phase2/latency/latency_summary.json
  data/exp/phase2/latency/latency_salsa_gpu.jsonl
  data/exp/phase2/latency/latency_cot_gpu.jsonl
  data/exp/phase2/predictions/base_qwen_cot.jsonl   (for budget histogram)

Outputs (all under data/exp/phase2/figures/):
  fig1_latency_bar.png          — SALSA vs CoT vs CPU reference latency
  fig2_composite_bar.png        — Composite score M across all models
  fig3_sabotage_heatmap.png     — Per-sabotage flip rate heatmap
  fig4_cot_budget_hist.png      — Base-CoT token budget distribution
  fig5_fpr_tpr_scatter.png      — TPR vs FPR per model (Paranoid Sycophant check)
  table_main.txt                — ASCII + LaTeX main results table
  table_sabotage.txt            — ASCII + LaTeX per-sabotage table
  table_cot_comparison.txt      — ASCII + LaTeX CoT subsample comparison

Usage (either environment, needs matplotlib + numpy):
  python -m experiments.phase2.generate_report
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not installed — skipping all figure generation")
    print("       Install with: pip install matplotlib")

from experiments.phase2.utils import (
    ensure_dirs, METRICS_DIR, LATENCY_DIR, PREDICTIONS_DIR,
    PHASE2_ROOT, SABOTAGE_TYPES,
)

FIGURES_DIR = PHASE2_ROOT / "figures"

# ── Display names and canonical order ─────────────────────────────────────────
MODEL_ORDER = [
    "venra_salsa",
    "base_qwen_zeroshot",
    "gemini_3_flash",
    "gemini_25_flash",
    "kimi_k25_nvidia",
    "qwen3_32b_groq",
    "llama33_70b_groq",
]
MODEL_LABELS = {
    "venra_salsa":        "VeNRA 3B\n(SALSA)",
    "base_qwen_zeroshot": "Base Qwen 3B\n(zero-shot)",
    "gemini_3_flash":     "Gemini-3\nFlash",
    "gemini_25_flash":     "Gemini-2.5\nFlash",
    "kimi_k25_nvidia":    "Kimi K2.5\n(NIM)",
    "qwen3_32b_groq":     "Qwen3-32B\n(Groq)",
    "llama33_70b_groq":   "Llama 3.3\n70B (Groq)",
}

# Colour for VeNRA vs others
VENRA_COLOR  = "#2563EB"   # strong blue
OTHER_COLOR  = "#94A3B8"   # muted slate
ACCENT_COLOR = "#DC2626"   # red for CPU reference

# LaTeX safe model name (no special chars)
LATEX_NAMES = {
    "venra_salsa":        r"\textbf{VeNRA 3B (SALSA)}",
    "base_qwen_zeroshot": "Base Qwen 3B (zero-shot)",
    "gemini_3_flash":     "Gemini-3-Flash-Preview",
    "gemini_25_flash":     "Gemini-25-Flash",
    "kimi_k25_nvidia":    "Kimi K2.5 (NIM)",
    "qwen3_32b_groq":     "Qwen3-32B (Groq)",
    "llama33_70b_groq":   "Llama 3.3 70B (Groq)",
}


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_metrics() -> Optional[Dict]:
    p = METRICS_DIR / "all_models_metrics.json"
    if not p.exists():
        print(f"[skip] {p} not found — run compute_metrics.py first")
        return None
    with open(p) as f:
        return json.load(f)


def load_latency_summary() -> Optional[Dict]:
    p = LATENCY_DIR / "latency_summary.json"
    if not p.exists():
        print(f"[skip] {p} not found — run run_latency_gpu.py first")
        return None
    with open(p) as f:
        return json.load(f)


def load_cot_budgets() -> List[int]:
    """Load per-sample token_budget values from base CoT prediction file."""
    p = PREDICTIONS_DIR / "base_qwen_cot.jsonl"
    if not p.exists():
        return []
    budgets = []
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                    if "token_budget" in row:
                        budgets.append(int(row["token_budget"]))
                except Exception:
                    pass
    return budgets


def load_per_sample_latency() -> Tuple[List[float], List[float]]:
    """Load median-per-prompt latency for SALSA and CoT."""
    salsa_ms = []
    cot_ms   = []
    for path, target in [
        (LATENCY_DIR / "latency_salsa_gpu.jsonl", salsa_ms),
        (LATENCY_DIR / "latency_cot_gpu.jsonl",   cot_ms),
    ]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            row = json.loads(line)
                            target.append(float(row["median_ms"]))
                        except Exception:
                            pass
    return salsa_ms, cot_ms


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  Saved → {path}")


def bar_colors(keys: List[str]) -> List[str]:
    return [VENRA_COLOR if k == "venra_salsa" else OTHER_COLOR for k in keys]


# ── Figure 1: Latency bar chart ───────────────────────────────────────────────

def fig_latency(latency_summary: Dict, api_latencies: Optional[Dict]) -> None:
    """
    Grouped bar: SALSA (GPU) | CoT (GPU) | CPU reference
    Proves SALSA is viable; CoT on same hardware is prohibitively slow.
    """
    salsa_med = latency_summary["salsa"]["overall_median_ms"]
    cot_med   = latency_summary["cot"]["overall_median_ms"]
    cot_p95   = latency_summary["cot"]["overall_p95_ms"]
    salsa_p95 = latency_summary["salsa"]["overall_p95_ms"]
    cpu_ref   = latency_summary.get("cpu_reference_ms", 12000)
    speedup   = latency_summary["speedup_x"]

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["SALSA\n(GPU, 1 token)", f"CoT\n(GPU, ~{latency_summary['config']['max_new_tokens_cot']} tokens)", "SALSA\n(CPU reference)"]
    values = [salsa_med, cot_med, cpu_ref]
    errors = [salsa_p95 - salsa_med, cot_p95 - cot_med, 0]
    colors = [VENRA_COLOR, OTHER_COLOR, ACCENT_COLOR]

    bars = ax.bar(labels, values, color=colors, width=0.5,
                  yerr=errors, capsize=5, error_kw={"elinewidth": 1.5})

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Time-to-Verdict: SALSA vs Chain-of-Thought", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}ms" if x < 1000 else f"{x/1000:.1f}s"
    ))
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, values):
        label = f"{val:.0f} ms" if val < 1000 else f"{val/1000:.1f} s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Speedup annotation
    ax.annotate(
        f"{speedup:.0f}× faster",
        xy=(0, salsa_med), xytext=(1, cot_med * 0.4),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, color=VENRA_COLOR, fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color=VENRA_COLOR, label="VeNRA SALSA"),
        mpatches.Patch(color=OTHER_COLOR, label="CoT (same model)"),
        mpatches.Patch(color=ACCENT_COLOR, label="CPU deployment (HF Space)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig1_latency_bar.png")


# ── Figure 2: Composite score bar chart ───────────────────────────────────────

def fig_composite(full_results: Dict) -> None:
    keys   = [k for k in MODEL_ORDER if k in full_results]
    values = [full_results[k]["composite"] for k in keys]
    labels = [MODEL_LABELS[k] for k in keys]
    colors = bar_colors(keys)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Composite Score  $\\mathcal{M}$", fontsize=12)
    ax.set_title("Multiplicative Metric $\\mathcal{M}$ Across Models\n"
                 r"(Flip Rate $\times$ Recall$_\mathrm{Natural}$ $\times$ "
                 r"TPR$_\mathrm{Clean}$ $\times$ Acc$_\mathrm{Axiom}$)$^{1/2}$",
                 fontsize=12)
    ax.set_ylim(0, min(1.05, max(values) * 1.25))
    ax.axhline(y=full_results.get("venra_salsa", {}).get("composite", 0),
               color=VENRA_COLOR, linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=VENRA_COLOR, label="VeNRA 3B (ours)"),
        mpatches.Patch(color=OTHER_COLOR, label="Baseline / Frontier"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig2_composite_bar.png")


# ── Figure 3: Per-sabotage flip rate heatmap ─────────────────────────────────

def fig_sabotage_heatmap(full_results: Dict) -> None:
    keys   = [k for k in MODEL_ORDER if k in full_results]
    labels = [MODEL_LABELS[k].replace("\n", " ") for k in keys]
    sab_labels = [
        "Logic\nCode Lie",
        "Numeric\nNeighbor",
        "Irrelevancy\nRAG",
        "Semantic\nDrift",
    ]

    matrix = np.array([
        [full_results[k]["fr_by_sabotage"].get(st, 0.0)
         for st in SABOTAGE_TYPES]
        for k in keys
    ])

    fig, ax = plt.subplots(figsize=(8, max(4, len(keys) * 0.7 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(SABOTAGE_TYPES)))
    ax.set_xticklabels(sab_labels, fontsize=10)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Flip Rate by Sabotage Type\n(higher = better at catching that failure mode)",
                 fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(keys)):
        for j in range(len(SABOTAGE_TYPES)):
            val = matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Flip Rate", fraction=0.03, pad=0.04)
    fig.tight_layout()
    _save(fig, "fig3_sabotage_heatmap.png")


# ── Figure 4: CoT budget histogram ───────────────────────────────────────────

def fig_cot_budget(budgets: List[int]) -> None:
    if not budgets:
        print("[skip] fig4_cot_budget_hist — no budget data found")
        return

    arr = np.array(budgets)
    fig, ax = plt.subplots(figsize=(8, 4))

    bins = range(0, min(int(arr.max()) + 20, 310), 10)
    ax.hist(arr, bins=bins, color=OTHER_COLOR, edgecolor="white", linewidth=0.5)

    # VeNRA SALSA reference line at 1 token
    ax.axvline(x=1, color=VENRA_COLOR, linewidth=2.5, linestyle="-",
               label=f"VeNRA SALSA (1 token)")
    ax.axvline(x=float(np.median(arr)), color=OTHER_COLOR, linewidth=2,
               linestyle="--", label=f"Base CoT median ({np.median(arr):.0f} tokens)")
    ax.axvline(x=float(np.percentile(arr, 95)), color=ACCENT_COLOR,
               linewidth=1.5, linestyle=":", label=f"P95 ({np.percentile(arr, 95):.0f} tokens)")

    ax.set_xlabel("Tokens generated before verdict", fontsize=12)
    ax.set_ylabel("Count (pairs)", fontsize=12)
    ax.set_title("Base Model CoT: Test-Time Compute Budget\n"
                 "VeNRA bakes the same reasoning into weights (budget = 1)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotation box
    stats_text = (
        f"n={len(arr)}\n"
        f"median={np.median(arr):.0f}\n"
        f"P95={np.percentile(arr, 95):.0f}\n"
        f"max={arr.max():.0f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    fig.tight_layout()
    _save(fig, "fig4_cot_budget_hist.png")


# ── Figure 5: TPR vs FPR scatter (Paranoid Sycophant check) ──────────────────

def fig_tpr_fpr(full_results: Dict) -> None:
    keys   = [k for k in MODEL_ORDER if k in full_results]
    tprs   = [full_results[k]["tpr_clean"]  for k in keys]
    fprs   = [full_results[k]["fpr_clean"]  for k in keys]
    labels = [MODEL_LABELS[k].replace("\n", " ") for k in keys]
    colors = bar_colors(keys)

    fig, ax = plt.subplots(figsize=(7, 6))

    for x, y, lbl, col in zip(fprs, tprs, labels, colors):
        ax.scatter(x, y, color=col, s=160, zorder=3, edgecolors="white", linewidths=1)
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    # Ideal quadrant: top-left = high TPR, low FPR
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.fill_between([0, 0.5], [0.5, 0.5], [1, 1],
                    color="green", alpha=0.05, label="Ideal region")

    ax.set_xlabel("False Positive Rate (FPR) on Clean Docs\n← lower is better", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR) on Clean Docs\nhigher is better →", fontsize=11)
    ax.set_title("Paranoid Sycophant Check\n"
                 "High Flip Rate achieved by blindly saying 'Fake' → high FPR",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    legend_patches = [
        mpatches.Patch(color=VENRA_COLOR, label="VeNRA 3B (ours)"),
        mpatches.Patch(color=OTHER_COLOR, label="Baseline / Frontier"),
        mpatches.Patch(color="green", alpha=0.2, label="Ideal: high TPR, low FPR"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_fpr_tpr_scatter.png")


# ── Text / LaTeX tables ───────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}\\%"


def table_main(full_results: Dict, cot_sub: Dict, budget: Optional[Dict]) -> None:
    """
    Writes both ASCII and LaTeX versions of the main comparison table.
    """
    keys = [k for k in MODEL_ORDER if k in full_results]

    # ── ASCII ──────────────────────────────────────────────────────────────────
    lines = []
    SEP = "=" * 118
    lines.append(SEP)
    lines.append("VENRA PHASE 2 — MAIN RESULTS TABLE")
    lines.append(SEP)
    h = (f"{'Model':<30} {'Valid':>6} {'FR(S)':>6} {'FR(L)':>6} {'FR':>6} "
         f"{'Recall':>7} {'TPR':>6} {'FPR':>6} {'Axiom':>6} {'M':>8}")
    lines.append(h)
    lines.append("-" * 118)

    for k in keys:
        m = full_results[k]
        n = m.get("display_name", k)[:29]
        lines.append(
            f"{n:<30} "
            f"{m['validity_rate']*100:>5.1f}% "
            f"{m['fr_short']*100:>5.1f}% "
            f"{m['fr_long']*100:>5.1f}% "
            f"{m['fr_global']*100:>5.1f}% "
            f"{m['recall_natural']*100:>6.1f}% "
            f"{m['tpr_clean']*100:>5.1f}% "
            f"{m['fpr_clean']*100:>5.1f}% "
            f"{m['acc_axiom']*100:>5.1f}% "
            f"{m['composite']:>8.4f}"
        )
    lines.append(SEP)

    # CoT subsample section
    if cot_sub:
        lines.append("")
        lines.append("CoT SUBSAMPLE (50 pairs — claims VeNRA at 1 token ≈ base model at N tokens)")
        lines.append("-" * 80)
        lines.append(f"{'Model':<42} {'FR':>8} {'Token Budget':>14}")
        lines.append("-" * 80)
        for mk, m in cot_sub.items():
            tok = "1" if "salsa" in mk or "zeroshot" in mk else \
                  (f"~{budget['median']:.0f}" if budget else "?")
            lines.append(f"{m['display']:<42} {m['fr_global']*100:>7.1f}% {tok:>14}")
        if budget:
            lines.append(f"\n  Base CoT budget stats: "
                         f"median={budget['median']}, P95={budget['p95']}, max={budget['max']}")

    out_ascii = FIGURES_DIR / "table_main.txt"
    with open(out_ascii, "w") as f:
        f.write("\n".join(lines))
    print(f"[table] Saved ASCII → {out_ascii}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{VeNRA Phase 2 Judge Showdown Results. "
                 r"FR=Flip Rate (global), Recall=Natural Fake Recall, "
                 r"TPR/FPR on clean supported docs, M=composite metric (Eq.~\ref{eq:metric}).}")
    latex.append(r"\label{tab:judge_showdown}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccccccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Valid & FR & Recall & TPR & FPR & Axiom & $\mathcal{M}$ \\")
    latex.append(r"\midrule")

    for k in keys:
        m = full_results[k]
        name = LATEX_NAMES.get(k, k)
        latex.append(
            f"{name} & "
            f"{m['validity_rate']*100:.0f}\\% & "
            f"{m['fr_global']*100:.1f}\\% & "
            f"{m['recall_natural']*100:.1f}\\% & "
            f"{m['tpr_clean']*100:.1f}\\% & "
            f"{m['fpr_clean']*100:.1f}\\% & "
            f"{m['acc_axiom']*100:.1f}\\% & "
            f"{m['composite']:.4f} \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    out_latex = FIGURES_DIR / "table_main_latex.tex"
    with open(out_latex, "w") as f:
        f.write("\n".join(latex))
    print(f"[table] Saved LaTeX → {out_latex}")


def table_sabotage(full_results: Dict) -> None:
    keys = [k for k in MODEL_ORDER if k in full_results]
    sab_short = ["Logic Code Lie", "Numeric Neighbor", "Irrelevancy RAG", "Semantic Drift"]

    # ASCII
    lines = ["=" * 100, "PER-SABOTAGE FLIP RATE", "=" * 100]
    lines.append(f"{'Model':<30} " + " ".join(f"{s[:16]:>18}" for s in sab_short))
    lines.append("-" * 100)
    for k in keys:
        m    = full_results[k]
        name = m.get("display_name", k)[:29]
        sab  = m.get("fr_by_sabotage", {})
        row  = f"{name:<30} "
        row += " ".join(f"{sab.get(st, 0.0)*100:>17.1f}%" for st in SABOTAGE_TYPES)
        lines.append(row)
    lines.append("=" * 100)

    out_ascii = FIGURES_DIR / "table_sabotage.txt"
    with open(out_ascii, "w") as f:
        f.write("\n".join(lines))
    print(f"[table] Saved ASCII → {out_ascii}")

    # LaTeX
    latex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Flip Rate by Sabotage Type. Each cell shows the fraction of pairs "
        r"where the Sentinel correctly passed the Supported parent and flagged the "
        r"sabotaged child, for that specific sabotage category.}",
        r"\label{tab:sabotage_breakdown}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Logic Code Lie & Numeric Neighbor & Irrelevancy RAG & Semantic Drift \\",
        r"\midrule",
    ]
    for k in keys:
        m   = full_results[k]
        sab = m.get("fr_by_sabotage", {})
        name = LATEX_NAMES.get(k, k)
        latex.append(
            f"{name} & "
            + " & ".join(f"{sab.get(st, 0.0)*100:.1f}\\%" for st in SABOTAGE_TYPES)
            + r" \\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_latex = FIGURES_DIR / "table_sabotage_latex.tex"
    with open(out_latex, "w") as f:
        f.write("\n".join(latex))
    print(f"[table] Saved LaTeX → {out_latex}")


def table_cot_comparison(cot_sub: Dict, budget: Optional[Dict],
                         latency_summary: Optional[Dict]) -> None:
    lines = [
        "=" * 72,
        "SYSTEM 1.5 BENCHMARK: CoT SUBSAMPLE COMPARISON",
        f"(n=50 pairs, seed=42 — all from pair pools only)",
        "=" * 72,
        f"{'Model':<42} {'FR (Flip Rate)':>15} {'Token Budget':>13}",
        "-" * 72,
    ]
    for mk, m in cot_sub.items():
        tok = "1" if "salsa" in mk or "zeroshot" in mk else \
              (f"~{budget['median']:.0f}" if budget else "?")
        lines.append(f"{m['display']:<42} {m['fr_global']*100:>14.1f}% {tok:>13}")

    lines.append("=" * 72)

    if budget:
        lines.append(f"\nBase CoT token budget: "
                     f"median={budget['median']}, "
                     f"P25={budget.get('p25','?')}, "
                     f"P75={budget.get('p75','?')}, "
                     f"P95={budget['p95']}, "
                     f"max={budget['max']}")

    if latency_summary:
        s = latency_summary["salsa"]["overall_median_ms"]
        c = latency_summary["cot"]["overall_median_ms"]
        lines.append(f"\nLatency (GPU RTX 3090):")
        lines.append(f"  SALSA  median: {s:.1f} ms")
        lines.append(f"  CoT    median: {c:.1f} ms")
        lines.append(f"  Speedup:       {latency_summary['speedup_x']:.0f}×")
        lines.append(f"  CPU reference: ~12,000 ms (HF Space, end-to-end)")

    out = FIGURES_DIR / "table_cot_comparison.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[table] Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics         = load_metrics()
    latency_summary = load_latency_summary()
    cot_budgets     = load_cot_budgets()
    salsa_ms, cot_ms = load_per_sample_latency()

    if metrics is None:
        print("[error] Cannot proceed without metrics. Run compute_metrics.py first.")
        return

    full_results = metrics.get("full_test_set", {})
    cot_sub      = metrics.get("cot_subsample", {})
    budget_stats = metrics.get("cot_budget_stats")

    if not full_results:
        print("[warn] full_test_set is empty in metrics file — no figures to generate")

    # ── Tables (always generated, no matplotlib needed) ───────────────────────
    table_main(full_results, cot_sub, budget_stats)
    if full_results:
        table_sabotage(full_results)
    if cot_sub:
        table_cot_comparison(cot_sub, budget_stats, latency_summary)

    # ── Figures ───────────────────────────────────────────────────────────────
    if not HAS_MPL:
        print("\n[done] Tables written. Install matplotlib to also generate figures.")
        return

    if latency_summary:
        fig_latency(latency_summary, api_latencies=None)
    else:
        print("[skip] fig1_latency_bar — no latency_summary.json")

    if full_results:
        fig_composite(full_results)
        fig_sabotage_heatmap(full_results)
        fig_tpr_fpr(full_results)
    else:
        print("[skip] figs 2/3/5 — no full_results data")

    if cot_budgets:
        fig_cot_budget(cot_budgets)
    else:
        print("[skip] fig4_cot_budget_hist — no token_budget data in CoT predictions")

    print(f"\n[done] All outputs written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()