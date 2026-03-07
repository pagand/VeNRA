"""
experiments/phase2/plot_ablations.py
=====================================
Section 6: Training Stability Ablations — Fetch WandB runs and plot.

Produces:
  - data/exp/phase2/chart2_confusion_matrices.pdf   (Run A vs Run B)
  - data/exp/phase2/chart3_loss_gradnorm_curves.pdf  (Run A vs Run B vs Run C)
  - data/exp/phase2/ablation_metrics_table.csv        (final-checkpoint numbers)

Run mapping (most-recent wins per naming convention):
  Run A — Uniform Weight  (clamped, no 10gen → general_penalty == penalty)
  Run B — Differential    (clamped + 10gen  → general_penalty = 10.0)  [Our Method]
  Run C — No Clamping     (weighted, no clamped tag)
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv
import wandb

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # repo root
OUT  = ROOT / "data" / "exp" / "phase2" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load credentials ──────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

if not WANDB_API_KEY:
    raise EnvironmentError("WANDB_API_KEY missing from .env")
if not WANDB_ENTITY or not WANDB_PROJECT:
    raise EnvironmentError(
        "WANDB_ENTITY and WANDB_PROJECT must be set in .env. "
        "Check your training code / wandb init."
    )

# ── Run name fragments — edit here if needed ──────────────────────────────────
# The script matches runs whose wandb display-name CONTAINS the fragment.
# Most-recent match wins (runs are sorted newest-first by wandb).
RUN_FRAGMENTS = {
    "A_uniform":       "./data/output/sweep2/venra-lr-weighted-clamped-postprompt-1e-4-r128-w0.10",   # sweep2, no 10gen
    "B_differential":  "./data/output/sweep2/venra-final-noinstruct-1e-4-r96-w0.10",  # sweep
    "C_noclamping":    "./data/output/sweep2/venra-lr-weighted-2e-4-r64-w0.10",                # closest no-clamped run; set to None to skip
}

RUN_LABELS = {
    "A_uniform":      "Run A — Uniform Weight",
    "B_differential": "Run B — Differential (Ours)",
    "C_noclamping":   "Run C — No Clamping",
}

COLORS = {
    "A_uniform":      "#e07b39",   # orange
    "B_differential": "#3a7ebf",   # blue  (our method)
    "C_noclamping":   "#c23b22",   # red
}

# Metrics to pull from run history
HISTORY_KEYS = [
    "train/loss",
    "train/grad_norm",
    "eval_audit/composite_score",
    "eval_audit/accuracy_found",
    "eval_audit/accuracy_fake",
    "eval_audit/axiom_accuracy",
    "_step",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. WandB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_api() -> wandb.Api:
    return wandb.Api(api_key=WANDB_API_KEY, timeout=60)


def find_run(api: wandb.Api, name_fragment: str) -> Optional[wandb.apis.public.Run]:
    """
    Return the most-recent run in entity/project whose display name
    contains `name_fragment` (case-insensitive substring match).
    Returns None if nothing found.
    """
    if not name_fragment:
        return None
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    try:
        runs = api.runs(path, order="-created_at")
    except Exception as e:
        print(f"  [WandB] Could not list runs for {path}: {e}")
        return None

    for run in runs:
        if name_fragment.lower() in run.name.lower():
            print(f"  ✓ Matched '{name_fragment}' → run '{run.name}' (id={run.id}, state={run.state})")
            return run

    # Fallback: try matching on run.config['output_dir'] or tags
    for run in runs:
        cfg_dir = run.config.get("output_dir", "")
        if name_fragment.lower() in cfg_dir.lower():
            print(f"  ✓ Matched via config.output_dir '{name_fragment}' → run '{run.name}' (id={run.id})")
            return run

    print(f"  ✗ No run found matching '{name_fragment}'")
    return None


def fetch_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    """Pull scalar history for the keys we care about."""
    df = run.history(keys=HISTORY_KEYS, pandas=True, samples=5000)
    # wandb sometimes returns '_step' as index; normalise
    if "_step" in df.columns:
        df = df.rename(columns={"_step": "step"})
    elif df.index.name == "_step":
        df = df.reset_index().rename(columns={"_step": "step"})
    df = df.sort_values("step").reset_index(drop=True)
    return df


def fetch_summary(run: wandb.apis.public.Run) -> dict:
    """Return the run's summary dict (final / best values)."""
    return dict(run.summary)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_confusion_matrix(summary: dict, history: pd.DataFrame) -> np.ndarray:
    """
    Build a 3×3 confusion matrix from scalar metrics.
    Class order: 0=Found (Supported), 1=Fake (Unfounded), 2=General.

    The diagonal comes from accuracy_{found,fake,axiom}.
    Off-diagonals are estimated from the complementary error split equally
    (we don't have per-cell counts, so this is approximate but visually
    informative — clearly label it as 'approximate' in the figure caption).
    """
    # Use best eval step (max composite score)
    eval_cols = [c for c in history.columns if "eval_audit" in c]
    if eval_cols and "eval_audit/composite_score" in history.columns:
        best_idx = history["eval_audit/composite_score"].idxmax()
        row = history.loc[best_idx]
        acc_found  = float(row.get("eval_audit/accuracy_found", summary.get("eval_audit/accuracy_found", 0)))
        acc_fake   = float(row.get("eval_audit/accuracy_fake",  summary.get("eval_audit/accuracy_fake",  0)))
        acc_axiom  = float(row.get("eval_audit/axiom_accuracy", summary.get("eval_audit/axiom_accuracy", 0)))
    else:
        acc_found  = float(summary.get("eval_audit/accuracy_found", 0))
        acc_fake   = float(summary.get("eval_audit/accuracy_fake",  0))
        acc_axiom  = float(summary.get("eval_audit/axiom_accuracy", 0))

    # Diagonal
    cm = np.zeros((3, 3), dtype=float)
    cm[0, 0] = acc_found
    cm[1, 1] = acc_fake
    cm[2, 2] = acc_axiom

    # Distribute errors equally across non-diagonal columns (rough approximation)
    err_found = 1.0 - acc_found
    err_fake  = 1.0 - acc_fake
    err_axiom = 1.0 - acc_axiom

    cm[0, 1] = err_found / 2
    cm[0, 2] = err_found / 2
    cm[1, 0] = err_fake  / 2
    cm[1, 2] = err_fake  / 2
    cm[2, 0] = err_axiom / 2
    cm[2, 1] = err_axiom / 2

    return cm, acc_found, acc_fake, acc_axiom


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chart 2 — Confusion Matrices (Run A vs Run B)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(data: dict):
    """
    Side-by-side confusion matrices for Run A and Run B.
    data keys: run_key → {"cm": np.ndarray, "accs": tuple}
    """
    keys_to_plot = ["A_uniform", "B_differential"]
    available = [k for k in keys_to_plot if k in data and data[k] is not None]

    if not available:
        print("[Chart 2] No data available — skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    class_names = ["Found\n(Supported)", "Fake\n(Unfounded)", "General"]
    cmap = LinearSegmentedColormap.from_list("venra_blue", ["#ffffff", "#1a5ea8"])

    for ax, key in zip(axes, available):
        cm = data[key]["cm"]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(RUN_LABELS[key], fontsize=11, fontweight="bold", pad=10)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)

        thresh = 0.55
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center", fontsize=10,
                        color=color, fontweight="bold" if i == j else "normal")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Confusion Matrices at Best Checkpoint\n"
        "(off-diagonal values are approximate — see text)",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    out_path = OUT / "chart2_confusion_matrices.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chart 3 — Loss & Grad-Norm Curves (all three runs)
# ─────────────────────────────────────────────────────────────────────────────

def smooth(series: pd.Series, window: int = 20) -> pd.Series:
    """Exponential moving average for readability."""
    return series.ewm(span=window, adjust=False).mean()


def plot_loss_gradnorm(data: dict):
    """
    Two-panel figure: train/loss (top) and train/grad_norm (bottom),
    all three runs overlaid, smoothed.
    """
    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    has_any = False
    for key in ["A_uniform", "B_differential", "C_noclamping"]:
        if key not in data or data[key] is None:
            continue
        hist = data[key]["history"]
        if hist is None or hist.empty:
            continue
        has_any = True
        label  = RUN_LABELS[key]
        color  = COLORS[key]
        lw     = 2.5 if key == "B_differential" else 1.8
        ls     = "-"  if key == "B_differential" else "--" if key == "A_uniform" else ":"

        if "train/loss" in hist.columns:
            raw  = hist.dropna(subset=["train/loss"])
            ax_loss.plot(raw["step"], smooth(raw["train/loss"]),
                         color=color, lw=lw, ls=ls, label=label, alpha=0.9)
            ax_loss.plot(raw["step"], raw["train/loss"],
                         color=color, lw=0.4, alpha=0.2)

        if "train/grad_norm" in hist.columns:
            raw = hist.dropna(subset=["train/grad_norm"])
            ax_grad.plot(raw["step"], smooth(raw["train/grad_norm"]),
                         color=color, lw=lw, ls=ls, label=label, alpha=0.9)
            ax_grad.plot(raw["step"], raw["train/grad_norm"],
                         color=color, lw=0.4, alpha=0.2)

    if not has_any:
        print("[Chart 3] No history data available — skipping.")
        plt.close(fig)
        return

    ax_loss.set_ylabel("Training Loss", fontsize=11)
    ax_loss.set_xlabel("Training Step", fontsize=10)
    ax_loss.set_title("Training Loss Curves", fontsize=11, fontweight="bold")
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.grid(True, alpha=0.3, linestyle=":")

    ax_grad.set_ylabel("Gradient Norm", fontsize=11)
    ax_grad.set_xlabel("Training Step", fontsize=10)
    ax_grad.set_title("Gradient Norm Curves", fontsize=11, fontweight="bold")
    ax_grad.legend(fontsize=9, loc="upper right")
    ax_grad.grid(True, alpha=0.3, linestyle=":")

    # Annotate Run C spike if present
    if "C_noclamping" in data and data["C_noclamping"] is not None:
        hist_c = data["C_noclamping"]["history"]
        if hist_c is not None and "train/grad_norm" in hist_c.columns:
            peak_idx = hist_c["train/grad_norm"].idxmax()
            peak_step = hist_c.loc[peak_idx, "step"]
            peak_val  = hist_c.loc[peak_idx, "train/grad_norm"]
            ax_grad.annotate(
                "Gradient bomb",
                xy=(peak_step, peak_val),
                xytext=(peak_step + 20, peak_val * 0.9),
                fontsize=8, color=COLORS["C_noclamping"],
                arrowprops=dict(arrowstyle="->", color=COLORS["C_noclamping"])
            )

    plt.tight_layout()
    out_path = OUT / "chart3_loss_gradnorm_curves.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bonus: composite score over eval steps
# ─────────────────────────────────────────────────────────────────────────────

def plot_composite_score(data: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    has_any = False
    for key in ["A_uniform", "B_differential", "C_noclamping"]:
        if key not in data or data[key] is None:
            continue
        hist = data[key]["history"]
        if hist is None or hist.empty:
            continue
        col = "eval_audit/composite_score"
        if col not in hist.columns:
            continue
        has_any = True
        sub = hist.dropna(subset=[col])
        ax.plot(sub["step"], sub[col],
                color=COLORS[key], lw=2.2,
                ls="-" if key == "B_differential" else "--" if key == "A_uniform" else ":",
                label=RUN_LABELS[key], marker="o", markersize=3)

    if not has_any:
        plt.close(fig)
        return

    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_title("Eval Composite Score per Checkpoint", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    out_path = OUT / "chart3b_composite_score.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary CSV table
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_table(data: dict):
    rows = []
    metric_keys = [
        "eval_audit/composite_score",
        "eval_audit/flip_rate_global",
        "eval_audit/accuracy_found",
        "eval_audit/accuracy_fake",
        "eval_audit/axiom_accuracy",
        "eval_audit/recall_natural_fake",
        "eval_audit/tpr_natural_true",
        "eval_audit/fpr_natural_true",
        "eval_audit/ece_global",
    ]
    for key in ["A_uniform", "B_differential", "C_noclamping"]:
        if key not in data or data[key] is None:
            continue
        summary = data[key]["summary"]
        hist    = data[key]["history"]

        # Prefer value at best composite step from history
        if hist is not None and "eval_audit/composite_score" in hist.columns:
            best_idx = hist["eval_audit/composite_score"].idxmax()
            best_row = hist.loc[best_idx]
        else:
            best_row = {}

        row = {"run": RUN_LABELS[key]}
        for m in metric_keys:
            val = best_row.get(m, summary.get(m, float("nan")))
            row[m.split("/")[-1]] = round(float(val), 4) if val == val else "—"
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("run")
    csv_path = OUT / "ablation_metrics_table.csv"
    df.to_csv(csv_path)
    print(f"  ✓ Saved {csv_path}")
    print("\n" + df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VeNRA Section 6 Ablation Plotter")
    print(f"  Entity:  {WANDB_ENTITY}")
    print(f"  Project: {WANDB_PROJECT}")
    print(f"  Output:  {OUT}")
    print("=" * 70)

    api = get_api()
    data = {}

    for key, fragment in RUN_FRAGMENTS.items():
        if not fragment:
            print(f"\n[{key}] Skipped (no fragment specified)")
            data[key] = None
            continue

        print(f"\n[{key}] Searching for: '{fragment}'")
        run = find_run(api, fragment)
        if run is None:
            data[key] = None
            continue

        print(f"  Fetching history...")
        try:
            hist = fetch_history(run)
            print(f"  History rows: {len(hist)}  |  Columns: {[c for c in hist.columns if c in HISTORY_KEYS]}")
        except Exception as e:
            print(f"  [warn] Could not fetch history: {e}")
            hist = pd.DataFrame()

        summary = fetch_summary(run)
        cm, acc_found, acc_fake, acc_axiom = reconstruct_confusion_matrix(summary, hist)

        data[key] = {
            "run":     run,
            "history": hist,
            "summary": summary,
            "cm":      cm,
            "accs":    (acc_found, acc_fake, acc_axiom),
        }
        print(f"  Accuracy → Found={acc_found:.2%}  Fake={acc_fake:.2%}  Axiom={acc_axiom:.2%}")

    print("\n" + "=" * 70)
    print("Plotting Chart 2 — Confusion Matrices (A vs B)")
    plot_confusion_matrices(data)

    print("Plotting Chart 3 — Loss / Grad Norm curves (A vs B vs C)")
    plot_loss_gradnorm(data)

    print("Plotting Chart 3b — Composite Score over training")
    plot_composite_score(data)

    print("Saving metrics table")
    save_metrics_table(data)

    print("\n✅  All outputs written to:", OUT)


if __name__ == "__main__":
    main()