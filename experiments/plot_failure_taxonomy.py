"""
experiments/phase2/plot_failure_taxonomy.py
-------------------------------------------
Grouped horizontal bar chart: Run 1 (Baseline RAG) vs Run 4 (VeNRA Full).
Source: data/exp/results/failure_analysis_gemini.json

Output: data/exp/results/fig_failure_taxonomy.pdf
        data/exp/results/fig_failure_taxonomy.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FAILURE_JSON = PROJECT_ROOT / "data/exp/results/failure_analysis_gemini.json"
OUT_DIR      = PROJECT_ROOT / "data/exp/results"
OUT_PDF      = OUT_DIR / "fig_failure_taxonomy.pdf"
OUT_PNG      = OUT_DIR / "fig_failure_taxonomy.png"

# ── Style — matches other paper figures ────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "text.usetex":      False,
})

COLOR_R1 = "#4878CF"   # blue  — Run 1 Baseline RAG
COLOR_R4 = "#E8793A"   # orange — Run 4 VeNRA Full
ALPHA    = 0.88

# ── Data ───────────────────────────────────────────────────────────────────────
# Loaded directly from failure_analysis_gemini.json; falls back to hardcoded
# values computed from the file for portability.
try:
    with open(FAILURE_JSON) as f:
        fa = json.load(f)
    r1 = fa["per_run_breakdown"]["run_1"]
    r4 = fa["per_run_breakdown"]["run_4"]
    N  = fa["total_samples"]
except FileNotFoundError:
    print(f"[warn] {FAILURE_JSON} not found!")

# ── Category definitions — ordered top-to-bottom by display priority ───────────
# (SUCCESS first, then failures ordered by Run 1 magnitude descending)
CATEGORIES = [
    ("NONE",                          "Success",               "#2ca02c", "#2ca02c"),
    ("TYPE_1_RETRIEVAL_BLINDNESS",    "T1: Retrieval Blindness",  COLOR_R1, COLOR_R4),
    ("TYPE_7_GENERATION_FAILURE",     "T7: Generation Failure",   COLOR_R1, COLOR_R4),
    ("TYPE_2_GENERATIVE_CONFLATION",  "T2: Generative Conflation",COLOR_R1, COLOR_R4),
    ("TYPE_3_ARITHMETIC_HALLUCINATION","T3: Arithmetic Halluc.",  COLOR_R1, COLOR_R4),
    ("TYPE_6_GOLD_AMBIGUITY",         "T6: Gold Ambiguity",       COLOR_R1, COLOR_R4),
]

labels  = [c[1] for c in CATEGORIES]
vals_r1 = [r1.get(c[0], 0) for c in CATEGORIES]
vals_r4 = [r4.get(c[0], 0) for c in CATEGORIES]

n_cats  = len(CATEGORIES)
y       = np.arange(n_cats)
bar_h   = 0.36
gap     = 0.06

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.8))

for i, (cat_key, cat_label, _, _) in enumerate(CATEGORIES):
    v1 = vals_r1[i]
    v4 = vals_r4[i]

    # Run 1 bar (upper)
    b1 = ax.barh(y[i] + bar_h/2 + gap/2, v1, height=bar_h,
                 color=COLOR_R1, alpha=ALPHA, zorder=3)
    # Run 4 bar (lower)
    b4 = ax.barh(y[i] - bar_h/2 - gap/2, v4, height=bar_h,
                 color=COLOR_R4, alpha=ALPHA, zorder=3)

    # Count annotations
    offset = 0.8
    ax.text(v1 + offset, y[i] + bar_h/2 + gap/2, str(v1),
            va="center", ha="left", fontsize=8.5, color="#333333")
    ax.text(v4 + offset, y[i] - bar_h/2 - gap/2, str(v4),
            va="center", ha="left", fontsize=8.5, color="#333333")

    # Delta annotation — show change on the right side
    delta = v4 - v1
    if delta != 0:
        sign  = "+" if delta > 0 else ""
        color = "#c0392b" if delta > 0 else "#27ae60"  # red=worse, green=better
        # For SUCCESS, flip: more is better
        if cat_key == "NONE":
            color = "#27ae60" if delta > 0 else "#c0392b"
        ax.text(max(v1, v4) + 3.5, y[i], f"({sign}{delta})",
                va="center", ha="left", fontsize=8, color=color, style="italic")

# ── Separator line between SUCCESS and failures ────────────────────────────────
ax.axhline(y=n_cats - 1 - 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=2)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Number of queries (out of 171)", fontsize=9)
ax.set_xlim(0, 105)
ax.set_ylim(-0.75, n_cats - 0.25)
ax.invert_yaxis()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax.set_axisbelow(True)

# ── Legend ─────────────────────────────────────────────────────────────────────
patch_r1 = mpatches.Patch(facecolor=COLOR_R1, alpha=ALPHA,
                           label="Run 1 — Baseline RAG (Vector + Gemini CoT)")
patch_r4 = mpatches.Patch(facecolor=COLOR_R4, alpha=ALPHA,
                           label="Run 4 — VeNRA Full (DualRetriever + PAL)")
ax.legend(handles=[patch_r1, patch_r4], loc="lower right",
          framealpha=0.9, edgecolor="#cccccc", fontsize=8.5)

# ── Title & footnote ───────────────────────────────────────────────────────────
ax.set_title(
    "Failure Type Distribution: Baseline RAG vs. VeNRA Full",
    fontsize=11, pad=8, fontweight="normal"
)
fig.text(
    0.01, -0.02,
    "Δ values show Run 4 − Run 1. Green = improvement (fewer failures / more successes). "
    "Red = regression.",
    fontsize=7.5, color="#555555", ha="left"
)

plt.tight_layout(rect=[0, 0.02, 1, 1])

# ── Save ───────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
print(f"[done] {OUT_PDF}")
print(f"[done] {OUT_PNG}")
plt.close()