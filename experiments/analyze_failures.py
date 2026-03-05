"""
experiments/analyze_failures.py
Phase 4: Failure Matrix Construction
--------------------------------------
Sweeps through final_metrics.json and produces:
  1. Per-run failure breakdown (all four runs, paper table).
  2. Per-dataset breakdown (FinanceBench / TAT-QA / FinQA) comparing
     Baseline vs. VeNRA Full — reveals WHERE each system fails.
  3. Cross-run survivor analysis with four mutually-exclusive, correctly
     defined groups.
  4. Hallucination rate summary per run.

Directory contract:
  experiments/analyze_failures.py          ← this file
  data/exp/results/final_metrics.json      ← input (from extract_metrics.py)
  data/exp/results/failure_analysis.json   ← detailed JSON output
  data/exp/results/failure_summary.md      ← paper-ready markdown tables
"""

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any

# TOGGLE THIS: "GEMINI" or "QWEN" to match previous phases
MODEL_TYPE = "GEMINI"

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR           = "data/exp/results"
FINAL_METRICS_PATH    = os.path.join(RESULTS_DIR, f"final_metrics_{MODEL_TYPE.lower()}.json")
FAILURE_ANALYSIS_PATH = os.path.join(RESULTS_DIR, f"failure_analysis_{MODEL_TYPE.lower()}.json")
FAILURE_SUMMARY_PATH  = os.path.join(RESULTS_DIR, f"failure_summary_{MODEL_TYPE.lower()}.md")

RUNS = ["run_1", "run_2", "run_3", "run_4"]
RUN_LABEL = {
    "run_1": f"Baseline RAG (Vector + {MODEL_TYPE.capitalize()} CoT)",
    "run_2": f"VeNRA Core (UFL + {MODEL_TYPE.capitalize()} CoT)",
    "run_3": "PAL + Baseline (Vector Text Only)",
    "run_4": "VeNRA Full (UFL + PAL Hybrid)",
}

DATASETS = {
    "financebench_normalized.jsonl":    "FinanceBench",
    "tatqa_normalized_test_gold.jsonl": "TAT-QA",
    "finqa_normalized.jsonl":           "FinQA",
}


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "N/A"


class FailureAnalyzer:

    def __init__(self):
        if not os.path.exists(FINAL_METRICS_PATH):
            raise FileNotFoundError(
                f"final_metrics.json not found at {FINAL_METRICS_PATH}\n"
                "Run extract_metrics.py first."
            )
        with open(FINAL_METRICS_PATH) as f:
            self.records: List[Dict[str, Any]] = json.load(f)

        assert len(self.records) > 0, "final_metrics.json is empty."
        assert "runs" in self.records[0], "Unexpected schema in final_metrics.json."

    # ── 1. Per-run global failure counts ──────────────────────────────────────

    def per_run_breakdown(self) -> Dict[str, Counter]:
        breakdown: Dict[str, Counter] = {run: Counter() for run in RUNS}
        for rec in self.records:
            for run in RUNS:
                ft = rec["runs"].get(run, {}).get("failure_type", "MISSING")
                breakdown[run][ft] += 1
        return breakdown

    # ── 2. Per-dataset breakdown ───────────────────────────────────────────────

    def per_dataset_breakdown(self) -> Dict[str, Dict[str, Counter]]:
        """
        Failure counts by source dataset for run_1 and run_4 only.
        Shows reviewers WHERE each system fails:
          FinanceBench → expect high T2 for baseline (semantic conflation)
          TAT-QA       → expect high T3 for baseline (arithmetic on hybrid tables)
          FinQA        → expect high T3 for both CoT runs (complex multi-step math)
        """
        ds_stats: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: {run: Counter() for run in ["run_1", "run_4"]}
        )
        for rec in self.records:
            raw_ds = rec.get("source_ds", "unknown")
            ds     = DATASETS.get(raw_ds, raw_ds)
            for run in ["run_1", "run_4"]:
                ft = rec["runs"].get(run, {}).get("failure_type", "MISSING")
                ds_stats[ds][run][ft] += 1
        return dict(ds_stats)

    # ── 3. Cross-run survivor analysis ────────────────────────────────────────

    def survivor_analysis(self) -> Dict[str, Any]:
        """
        Classifies each sample into one of five MUTUALLY EXCLUSIVE groups.
        Groups are evaluated in priority order (first match wins):

          ALL_WIN        – all four runs correct (easy; no discriminative value)
          ALL_FAIL       – all four runs wrong (inherently hard; no system wins)
          ONLY_4_WINS    – ONLY run_4 correct; runs 1, 2, 3 all wrong
                           → VeNRA Full's unique contribution
          BASELINE_FAILS – run_1 fails, at least one of runs 2/3/4 succeeds
                           → any VeNRA component helps
          OTHER          – partial mixed results not covered above

        IMPORTANT: ONLY_4_WINS requires runs 1, 2, AND 3 to all fail.
        If runs 2 and 3 also succeed, the sample belongs in BASELINE_FAILS,
        not ONLY_4_WINS — attributing it there would overstate VeNRA Full's
        unique contribution.
        """
        groups: Counter = Counter()
        only_4_ids: List[str] = []
        baseline_fail_ids: List[str] = []

        for rec in self.records:
            em_vals = {
                run: rec["runs"].get(run, {}).get("em")
                for run in RUNS
            }
            # Exclude samples with any missing run (partially executed benchmark)
            if any(v is None for v in em_vals.values()):
                continue

            r1, r2, r3, r4 = [em_vals[r] for r in RUNS]

            if r1 and r2 and r3 and r4:
                groups["ALL_WIN"] += 1

            elif not r1 and not r2 and not r3 and not r4:
                groups["ALL_FAIL"] += 1

            elif r4 and not r1 and not r2 and not r3:
                # Only run_4 succeeded — VeNRA Full's exclusive win
                groups["ONLY_4_WINS"] += 1
                only_4_ids.append(rec["id"])

            elif not r1 and (r2 or r3 or r4):
                # Baseline fails, but at least one VeNRA variant succeeds
                groups["BASELINE_FAILS"] += 1
                baseline_fail_ids.append(rec["id"])

            else:
                # Mixed results where baseline also succeeds
                groups["OTHER"] += 1

        return {
            "group_counts":          dict(groups),
            "only_4_wins_ids":       only_4_ids[:20],       # sample for inspection
            "baseline_fail_ids":     baseline_fail_ids[:20],
        }

    # ── 4. Hallucination rate summary ─────────────────────────────────────────

    def hallucination_summary(self) -> Dict[str, str]:
        hall_vals: Dict[str, List[float]] = {run: [] for run in RUNS}
        for rec in self.records:
            for run in RUNS:
                h = rec["runs"].get(run, {}).get("hallucination_rate")
                if h is not None:
                    hall_vals[run].append(h)
        return {
            run: f"{sum(v) / len(v) * 100:.2f}%" if v else "N/A"
            for run, v in hall_vals.items()
        }

    # ── Output ─────────────────────────────────────────────────────────────────

    def run(self):
        n = len(self.records)
        run_bd    = self.per_run_breakdown()
        ds_bd     = self.per_dataset_breakdown()
        survivors = self.survivor_analysis()
        hall      = self.hallucination_summary()

        # JSON output
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(FAILURE_ANALYSIS_PATH, "w") as f:
            json.dump({
                "total_samples":         n,
                "per_run_breakdown":     {r: dict(c) for r, c in run_bd.items()},
                "per_dataset_breakdown": {
                    ds: {r: dict(c) for r, c in runs.items()}
                    for ds, runs in ds_bd.items()
                },
                "survivor_analysis":     survivors,
                "hallucination_rates":   hall,
            }, f, indent=2)

        # Markdown output
        lines = [
            "# VeNRA Stage A — Failure Analysis",
            f"Total samples: {n}",
            "",
            "## Per-Run Failure Breakdown",
            "| Run | Configuration | Success | T1 Retrieval | T2 Conflation | T3 Arithmetic |",
            "| :-- | :------------ | :------ | :----------- | :------------ | :------------ |",
        ]
        for run in RUNS:
            c    = run_bd[run]
            succ = c.get("NONE", 0)
            t1   = c.get("TYPE_1_RETRIEVAL_BLINDNESS",     0)
            t2   = c.get("TYPE_2_GENERATIVE_CONFLATION",    0)
            t3   = c.get("TYPE_3_ARITHMETIC_HALLUCINATION", 0)
            lines.append(
                f"| {run} | {RUN_LABEL[run]} "
                f"| {succ} ({pct(succ, n)}) "
                f"| {t1} ({pct(t1, n)}) "
                f"| {t2} ({pct(t2, n)}) "
                f"| {t3} ({pct(t3, n)}) |"
            )

        lines += ["", "## Per-Dataset Breakdown (Baseline vs. VeNRA Full)"]
        for ds, runs in ds_bd.items():
            ds_n = sum(runs["run_1"].values())
            lines += [
                f"### {ds} ({ds_n} samples)",
                "| Run | Success | T1 | T2 | T3 |",
                "| :-- | :------ | :- | :- | :- |",
            ]
            for run in ["run_1", "run_4"]:
                c   = runs[run]
                tot = sum(c.values()) or 1
                lines.append(
                    f"| {RUN_LABEL[run]} "
                    f"| {c.get('NONE',0)} ({pct(c.get('NONE',0), tot)}) "
                    f"| {c.get('TYPE_1_RETRIEVAL_BLINDNESS',0)} ({pct(c.get('TYPE_1_RETRIEVAL_BLINDNESS',0), tot)}) "
                    f"| {c.get('TYPE_2_GENERATIVE_CONFLATION',0)} ({pct(c.get('TYPE_2_GENERATIVE_CONFLATION',0), tot)}) "
                    f"| {c.get('TYPE_3_ARITHMETIC_HALLUCINATION',0)} ({pct(c.get('TYPE_3_ARITHMETIC_HALLUCINATION',0), tot)}) |"
                )

        sg = survivors["group_counts"]
        lines += [
            "",
            "## Cross-Run Survivor Analysis",
            "| Group | Definition | Count | % |",
            "| :---- | :--------- | :---- | :- |",
            f"| ALL_WIN        | All 4 runs correct           | {sg.get('ALL_WIN',0)}        | {pct(sg.get('ALL_WIN',0), n)} |",
            f"| ONLY_4_WINS    | Only VeNRA Full correct      | {sg.get('ONLY_4_WINS',0)}    | {pct(sg.get('ONLY_4_WINS',0), n)} |",
            f"| BASELINE_FAILS | Baseline fails, ≥1 VeNRA wins | {sg.get('BASELINE_FAILS',0)} | {pct(sg.get('BASELINE_FAILS',0), n)} |",
            f"| ALL_FAIL       | All 4 runs wrong             | {sg.get('ALL_FAIL',0)}       | {pct(sg.get('ALL_FAIL',0), n)} |",
            f"| OTHER          | Mixed results                | {sg.get('OTHER',0)}          | {pct(sg.get('OTHER',0), n)} |",
            "",
            "## Hallucination Rates by Run",
            "| Run | Configuration | Hallucination Rate |",
            "| :-- | :------------ | :----------------- |",
        ] + [
            f"| {run} | {RUN_LABEL[run]} | {hall[run]} |"
            for run in RUNS
        ]

        with open(FAILURE_SUMMARY_PATH, "w") as f:
            f.write("\n".join(lines))

        print(
            f"Failure analysis saved:\n"
            f"  {FAILURE_ANALYSIS_PATH}\n"
            f"  {FAILURE_SUMMARY_PATH}"
        )


if __name__ == "__main__":
    FailureAnalyzer().run()