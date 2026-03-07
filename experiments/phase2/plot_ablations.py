"""
experiments/phase2/plot_ablations.py
=====================================
VeNRA Sentinel — Section 6 Ablation Figures

Produces 3 outputs:
  figure1_training_dynamics.pdf
      Panel (a): Composite score learning curves for 3 prompt strategies
                 (42 / 30 / 16 eval checkpoints — richest data we have)
      Panel (b): Grad-norm of the clamped run (stable, 11 steps) with
                 annotation that the unclamped run crashed before first log
                 (the crash IS the gradient bomb evidence)

  figure2_class_ablation.pdf
      Panel (a): Garbage Bin — per-class accuracy, uniform vs differential
                 (garbage_uniform reads from WandB summary — 0 history rows
                  but summary holds the correct values: Found=31%, Fake=10%,
                  Axiom=71%)
      Panel (b): Prompt Physics — composite / flip_rate / recall across
                 3 prompt strategies (all have full history)

  table_progressive_ablation.csv
      5-row progressive ablation table for the paper
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv
import wandb

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "data" / "exp" / "phase2" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

for var, name in [(WANDB_API_KEY, "WANDB_API_KEY"),
                  (WANDB_ENTITY, "WANDB_ENTITY"),
                  (WANDB_PROJECT, "WANDB_PROJECT")]:
    if not var:
        raise EnvironmentError(f"{name} missing from .env")

# ── Run Registry ───────────────────────────────────────────────────────────

RUN_REGISTRY: Dict[str, Dict] = {

    # ── Figure 1 Panel (b): clamped run — stable grad_norm ────────────────
    # 11 rows of train/grad_norm. Shows smooth, bounded training.
    # The unclamped counterpart (venra-lr-weighted-2e-4-r*) has 0 history rows
    # because it crashed before the first logging_steps=10 interval fired.
    # We annotate this as a text box: the crash is itself the evidence.
    "grad_bomb_clamped": {
        "fragment": "./data/output/sweep2/venra-lr-weighted-clamped-v23-1e-4-r128-w0.10",
        "label":    "Weighted + Clamped",
        "color":    "#3a7ebf",
        "ls":       "-",
    },

    # ── Figure 2 Panel (a): Garbage Bin ───────────────────────────────────
    # garbage_uniform: 0 history rows but WandB summary contains correct values:
    #   Found=30.6%  Fake=9.6%  Axiom=71.4%  → classic garbage bin pattern
    # best_row() falls back to summary when history is empty — this works.
    "garbage_uniform": {
        "fragment": "./data/output/sweep2/venra-lr-weighted-clamped-postprompt-1e-4-r128-w0.10",
        "label":    "Uniform (50/50/50)",
        "color":    "#e07b39",
        "ls":       "--",
    },
    "garbage_differential": {
        "fragment": "./data/output/sweep2/venra-lr-weighted-10gen-clamped-v23-1e-4-r128-w0.10",
        "label":    "Differential (50/50/10)",
        "color":    "#3a7ebf",
        "ls":       "-",
    },

    # ── Figure 1 Panel (a) + Figure 2 Panel (b): Prompt strategies ────────
    # All three have full eval history (16 / 42 / 30 rows).
    "prompt_full": {
        "fragment": "./data/output/sweep/venra-final-full-1e-4-r128-w0.10",# "./data/output/sweep/venra-final-full-1e-4-r128-w0.10" 
        "label":    "Full Prompt (instruct + repeat)",
        "color":    "#555555",
        "ls":       "--",
    },
    "prompt_noinstruct": {
        "fragment": "./data/output/sweep2/venra-final-noinstruct-1e-4-r96-w0.10",
        "label":    "No Instruct (ours)",
        "color":    "#3a7ebf",
        "ls":       "-",
    },
    "prompt_preprompt": {
        "fragment": "./data/output/sweep3/venra-final-preprompt-1e-4-r96-w0.10",
        "label":    "Pre-prompt (repeat at top)",
        "color":    "#8b5cf6",
        "ls":       ":",
    },

    # ── Table rows only ────────────────────────────────────────────────────
    "table_baseline": {
        "fragment": "venra-lr1e-4-r64-w0.10",
        "label":    "Baseline (no weighting)",
        "color":    "#999999",
        "ls":       "-",
    },
    "table_weighted_noclamp": {
        "fragment": "./data/output/sweep2/venra-lr-weighted-2e-4-r96-w0.10",
        "label":    "+ Weighted, No Clamp",
        "color":    "#c23b22",
        "ls":       ":",
    },
    "table_clamped_uniform": {
        "fragment": "./data/output/sweep2/venra-lr-weighted-clamped-v23-1e-4-r128-w0.10",
        "label":    "+ Clamped, Uniform (50/50/50)",
        "color":    "#e07b39",
        "ls":       "--",
    },
    "table_differential": {
        "fragment": "./data/output/sweep2/venra-lr-weighted-10gen-clamped-v23-1e-4-r128-w0.10",
        "label":    "+ Differential (50/50/10)",
        "color":    "#3a7ebf",
        "ls":       "-",
    },
    "table_final_best": {
        "fragment": "./data/output/sweep2/venra-final-noinstruct-1e-4-r96-w0.10",
        "label":    "+ Prompt Tuning (Final Model)",
        "color":    "#16a34a",
        "ls":       "-",
    },
}

HISTORY_KEYS = [
    "_step",
    "train/loss",
    "train/grad_norm",
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

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize":8.5,
    "xtick.labelsize":9,
    "ytick.labelsize":9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth":1.8,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

# ── WandB helpers ───────────────────────────────────────────────────────────

def get_api() -> wandb.Api:
    return wandb.Api(api_key=WANDB_API_KEY, timeout=90)


def find_run(api: wandb.Api, fragment: str) -> Optional[Any]:
    if not fragment:
        return None
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    try:
        runs = api.runs(path, order="-created_at")
    except Exception as e:
        print(f"    [wandb error] {e}")
        return None
    for run in runs:
        if fragment.lower() in run.name.lower():
            print(f"    ✓ '{run.name}'  id={run.id}  state={run.state}")
            return run
    for run in runs:
        if fragment.lower() in str(run.config.get("output_dir", "")).lower():
            print(f"    ✓ (config) '{run.name}'  id={run.id}  state={run.state}")
            return run
    print(f"    ✗ not found: '{fragment}'")
    return None


def fetch_history(run) -> pd.DataFrame:
    try:
        df = run.history(keys=HISTORY_KEYS, pandas=True, samples=5000)
    except Exception as e:
        print(f"    [history error] {e}")
        return pd.DataFrame()

    if "_step" in df.columns:
        df = df.rename(columns={"_step": "step"})
    if df.index.name in ("_step", "step"):
        df = df.reset_index()
        if df.columns[0] in ("_step", "index"):
            df = df.rename(columns={df.columns[0]: "step"})
    if "step" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "step", df.index * 10)

    return df.sort_values("step").reset_index(drop=True)


def fetch_summary(run) -> dict:
    try:
        return dict(run.summary)
    except Exception:
        return {}


def best_row(hist: Optional[pd.DataFrame], summary: dict) -> dict:
    col = "eval_audit/composite_score"
    if hist is not None and not hist.empty and col in hist.columns:
        valid = hist.dropna(subset=[col])
        if not valid.empty:
            return dict(valid.loc[valid[col].idxmax()])
    return summary


def get_scalar(row: dict, key: str) -> Optional[float]:
    val = row.get(key, float("nan"))
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


def ema(series: pd.Series, span: int = 12) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Training Dynamics
#
# Panel (a): Composite score over eval checkpoints — 3 prompt strategies
#            Best available continuous signal (16/42/30 rows)
#            Directly supports §6.4 Prompt Physics
#
# Panel (b): Grad-norm of the ONE clamped run that has step-level logs
#            + prominent text box explaining the unclamped crash
#            "Unclamped run failed before step 10 — the crash is the evidence"
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure1_training_dynamics(data: dict):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 6.0),
        gridspec_kw={"hspace": 0.48}
    )

    # ── Panel (a): Composite score learning curves ────────────────────────
    prompt_keys = ["prompt_full", "prompt_noinstruct", "prompt_preprompt"]
    has_top = False
    for key in prompt_keys:
        if not data.get(key):
            continue
        hist = data[key].get("history")
        if hist is None or hist.empty:
            continue
        col = "eval_audit/composite_score"
        if col not in hist.columns:
            continue
        sub = hist.dropna(subset=[col])
        if sub.empty:
            continue
        has_top = True
        info  = RUN_REGISTRY[key]
        # Mark our best model with a star on peak
        peak_idx = sub[col].idxmax()
        ax_top.plot(sub["step"], sub[col],
                    color=info["color"], lw=2.0 if info["ls"] == "-" else 1.5,
                    ls=info["ls"], label=info["label"],
                    marker="o", markersize=3.5, markevery=1, zorder=3)
        if key == "prompt_noinstruct":
            ax_top.annotate(
                f"best: {sub.loc[peak_idx, col]:.3f}",
                xy=(sub.loc[peak_idx, "step"], sub.loc[peak_idx, col]),
                xytext=(sub.loc[peak_idx, "step"] - 40,
                        sub.loc[peak_idx, col] - 0.04),
                fontsize=7.5, color=info["color"],
                arrowprops=dict(arrowstyle="->", color=info["color"], lw=0.9)
            )

    ax_top.set_title("(a) Composite Score — Prompt Strategy Comparison",
                     fontweight="bold", pad=6)
    ax_top.set_ylabel("Composite Score")
    ax_top.set_xlabel("Training Step")
    if has_top:
        ax_top.legend(loc="lower right", framealpha=0.9)
    ax_top.grid(True, alpha=0.25, ls=":")
    ax_top.set_ylim(bottom=0)
    if not has_top:
        ax_top.text(0.5, 0.5, "No composite score data",
                    ha="center", va="center", transform=ax_top.transAxes, color="gray")

    # ── Panel (b): Grad-norm of clamped run + crash annotation ───────────
    key_c = "grad_bomb_clamped"
    has_bot = False

    if data.get(key_c):
        hist_c = data[key_c].get("history")
        if hist_c is not None and not hist_c.empty and "train/grad_norm" in hist_c.columns:
            raw = hist_c.dropna(subset=["train/grad_norm"])
            if not raw.empty:
                has_bot = True
                info_c = RUN_REGISTRY[key_c]
                smoothed = ema(raw["train/grad_norm"])
                ax_bot.plot(raw["step"], smoothed,
                            color=info_c["color"], lw=2.0, ls=info_c["ls"],
                            label=info_c["label"], zorder=3)
                ax_bot.fill_between(raw["step"], smoothed,
                                    color=info_c["color"], alpha=0.10, zorder=2)
                # Draw max for reference
                ax_bot.axhline(smoothed.max(), color=info_c["color"],
                               lw=0.8, ls="--", alpha=0.5)
                ax_bot.text(raw["step"].max() * 0.98, smoothed.max() + 0.02,
                            f"max = {smoothed.max():.2f}",
                            ha="right", va="bottom",
                            fontsize=7.5, color=info_c["color"])

    # Crash annotation box — always shown, explains the missing unclamped line
    crash_text = (
        "Unclamped run (lr=2e-4) crashed\n"
        "before step 10 — no logs recorded.\n"
        "‖g‖ spike exceeded 72,000 in\n"
        "manual inspection (see §6.3)."
    )
    ax_bot.text(
        0.97, 0.95, crash_text,
        transform=ax_bot.transAxes,
        ha="right", va="top",
        fontsize=7.5,
        color="#c23b22",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5",
                  ec="#c23b22", lw=0.9, alpha=0.95)
    )

    ax_bot.set_title("(b) Gradient Norm — Clamped Training (Stable)",
                     fontweight="bold", pad=6)
    ax_bot.set_ylabel("Gradient Norm (EMA)")
    ax_bot.set_xlabel("Training Step")
    if has_bot:
        ax_bot.legend(loc="upper left", framealpha=0.9)
    ax_bot.grid(True, alpha=0.25, ls=":")
    ax_bot.set_ylim(bottom=0)
    if not has_bot:
        ax_bot.text(0.5, 0.5, "No grad_norm history for clamped run",
                    ha="center", va="center", transform=ax_bot.transAxes, color="gray")

    out = OUT / "figure1_training_dynamics.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Two-panel bar chart
#
# Panel (a): Garbage Bin Effect
#   garbage_uniform:       0 history rows, but summary = {Found:31%, Fake:10%, Axiom:71%}
#                          best_row() falls back to summary — renders correctly
#   garbage_differential:  19 history rows — reads from best eval checkpoint
#
# Panel (b): Prompt Physics
#   All three prompt runs have full history — reads from best eval checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure2_class_ablation(data: dict):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 4.2),
        gridspec_kw={"wspace": 0.42}
    )

    # ── Left: Garbage Bin ─────────────────────────────────────────────────
    gb_keys    = ["garbage_uniform", "garbage_differential"]
    gb_labels  = [RUN_REGISTRY[k]["label"] for k in gb_keys]
    gb_colors  = [RUN_REGISTRY[k]["color"] for k in gb_keys]
    cls_names  = ["Found\n(Supported)", "Fake\n(Unfounded)", "General\n(Axiom)"]
    cls_fields = ["eval_audit/accuracy_found",
                  "eval_audit/accuracy_fake",
                  "eval_audit/axiom_accuracy"]

    x = np.arange(len(cls_names))
    w = 0.32
    offsets = np.array([-(w / 2), (w / 2)])

    has_left = False
    for key, label, color, offset in zip(gb_keys, gb_labels, gb_colors, offsets):
        if not data.get(key):
            continue
        # best_row falls back to summary for 0-history runs
        br   = best_row(data[key].get("history"), data[key].get("summary", {}))
        vals = [get_scalar(br, f) or 0.0 for f in cls_fields]
        if max(vals) == 0.0:
            continue
        has_left = True
        bars = ax_left.bar(x + offset, vals, w,
                           label=label, color=color, alpha=0.85,
                           edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax_left.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.016,
                         f"{v:.0%}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold")

    # Annotate the garbage bin pattern visually
    if has_left:
        ax_left.annotate(
            "General dominates\n(garbage bin)",
            xy=(x[2] - w / 2, 0.714),
            xytext=(x[2] - w * 1.5, 0.85),
            fontsize=7.5, color="#e07b39",
            arrowprops=dict(arrowstyle="->", color="#e07b39", lw=0.9)
        )

    ax_left.set_title("(a) Garbage Bin Effect\nUniform vs. Differential Weighting",
                      fontweight="bold", pad=6)
    ax_left.set_ylabel("Per-Class Accuracy at Best Checkpoint")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(cls_names)
    ax_left.set_ylim(0, 1.12)
    ax_left.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_left.legend(loc="upper left", framealpha=0.9)
    ax_left.grid(True, axis="y", alpha=0.25, ls=":")
    ax_left.set_axisbelow(True)
    if not has_left:
        ax_left.text(0.5, 0.5, "No data available",
                     ha="center", va="center", transform=ax_left.transAxes,
                     color="gray", fontsize=11)

    # ── Right: Prompt Physics ─────────────────────────────────────────────
    pp_keys   = ["prompt_full", "prompt_noinstruct", "prompt_preprompt"]
    pp_labels = [RUN_REGISTRY[k]["label"] for k in pp_keys]
    pp_colors = [RUN_REGISTRY[k]["color"] for k in pp_keys]
    pp_fields = {
        "Composite\nScore":    "eval_audit/composite_score",
        "Flip Rate\n(Global)": "eval_audit/flip_rate_global",
        "Recall\n(Nat. Fake)": "eval_audit/recall_natural_fake",
    }

    x2 = np.arange(len(pp_fields))
    w2 = 0.22
    n2 = len(pp_keys)
    offsets2 = np.linspace(-(n2 - 1) * w2 / 2, (n2 - 1) * w2 / 2, n2)

    has_right = False
    for key, label, color, offset in zip(pp_keys, pp_labels, pp_colors, offsets2):
        if not data.get(key):
            continue
        br   = best_row(data[key].get("history"), data[key].get("summary", {}))
        vals = [get_scalar(br, f) or 0.0 for f in pp_fields.values()]
        if max(vals) == 0.0:
            continue
        has_right = True
        bars = ax_right.bar(x2 + offset, vals, w2,
                            label=label, color=color, alpha=0.85,
                            edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax_right.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.011,
                          f"{v:.2f}", ha="center", va="bottom",
                          fontsize=7.5, fontweight="bold")

    ax_right.set_title("(b) Prompt Strategy Ablation\n(Algorithmic De-activation & Repetition)",
                       fontweight="bold", pad=6)
    ax_right.set_ylabel("Score at Best Checkpoint")
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(list(pp_fields.keys()))
    ax_right.set_ylim(0, 0.85)
    ax_right.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_right.legend(loc="upper right", framealpha=0.9)
    ax_right.grid(True, axis="y", alpha=0.25, ls=":")
    ax_right.set_axisbelow(True)
    if not has_right:
        ax_right.text(0.5, 0.5, "No data available",
                      ha="center", va="center", transform=ax_right.transAxes,
                      color="gray", fontsize=11)

    out = OUT / "figure2_class_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Table — Progressive Ablation
# ─────────────────────────────────────────────────────────────────────────────

TABLE_ROW_KEYS = [
    "table_baseline",
    "table_weighted_noclamp",
    "table_clamped_uniform",
    "table_differential",
    "table_final_best",
]

TABLE_METRICS = {
    "Composite ↑":   "eval_audit/composite_score",
    "Flip Rate ↑":   "eval_audit/flip_rate_global",
    "Found Acc ↑":   "eval_audit/accuracy_found",
    "Fake Acc ↑":    "eval_audit/accuracy_fake",
    "Axiom Acc":     "eval_audit/axiom_accuracy",
    "Recall Fake ↑": "eval_audit/recall_natural_fake",
    "TPR True ↑":    "eval_audit/tpr_natural_true",
    "FPR True ↓":    "eval_audit/fpr_natural_true",
    "ECE ↓":         "eval_audit/ece_global",
}


def save_progressive_table(data: dict):
    rows = []
    for key in TABLE_ROW_KEYS:
        if not data.get(key):
            continue
        br  = best_row(data[key].get("history"), data[key].get("summary", {}))
        row = {"Configuration": RUN_REGISTRY[key]["label"]}
        for col, field in TABLE_METRICS.items():
            v = get_scalar(br, field)
            row[col] = f"{v:.4f}" if v is not None else "—"
        rows.append(row)

    if not rows:
        print("  [Table] No data — skipping.")
        return

    df = pd.DataFrame(rows).set_index("Configuration")
    csv_path = OUT / "table_progressive_ablation.csv"
    df.to_csv(csv_path)
    print(f"  ✓ {csv_path}")
    print()
    print(df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VeNRA — Section 6 Ablation Plotter")
    print(f"  Entity / Project : {WANDB_ENTITY} / {WANDB_PROJECT}")
    print(f"  Output           : {OUT}")
    print("=" * 70)

    api  = get_api()
    data = {}

    for key, info in RUN_REGISTRY.items():
        fragment = info["fragment"]
        print(f"\n[{key}]")
        print(f"  searching: '{fragment}'")
        run = find_run(api, fragment)
        if run is None:
            data[key] = None
            continue

        hist    = fetch_history(run)
        summary = fetch_summary(run)

        eval_cols  = [c for c in hist.columns if "eval_audit" in c
                      and not hist[c].isna().all()] if not hist.empty else []
        train_cols = [c for c in hist.columns if "train/" in c
                      and not hist[c].isna().all()] if not hist.empty else []

        print(f"  history rows : {len(hist)}")
        print(f"  train cols   : {train_cols}")
        print(f"  eval cols    : {eval_cols}")

        br   = best_row(hist if not hist.empty else None, summary)
        comp = get_scalar(br, "eval_audit/composite_score")
        fr   = get_scalar(br, "eval_audit/flip_rate_global")
        print(f"  best composite={comp}  flip_rate={fr}")

        data[key] = {
            "history": hist if not hist.empty else None,
            "summary": summary,
        }

    print("\n" + "=" * 70)
    print("Figure 1 — Training Dynamics")
    plot_figure1_training_dynamics(data)

    print("\nFigure 2 — Garbage Bin + Prompt Physics")
    plot_figure2_class_ablation(data)

    print("\nTable — Progressive Ablation")
    save_progressive_table(data)

    print(f"\n✅  Done.  All outputs in: {OUT}")


if __name__ == "__main__":
    main()