"""
experiments/phase2/run_latency_gpu.py
---------------------------------------
Latency benchmark: SALSA vs Chain-of-Thought on the same model (VeNRA merged).

Both conditions use the VeNRA merged model so the comparison is purely
architectural (single forward pass vs autoregressive generation).

Protocol per prompt, per condition:
  - 3 warmup passes  (discarded)
  - 10 timed passes  (median + P95 reported)
  - torch.cuda.synchronize() wraps each timed call for accurate GPU timing

Sample: 50 short + 50 long prompts (seed=42) from the test manifest.

Writes:
  data/exp/phase2/latency/latency_salsa_gpu.jsonl
  data/exp/phase2/latency/latency_cot_gpu.jsonl

Usage (GPU server, train .env):
  python -m experiments.phase2.run_latency_gpu
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_prompts_venra,
    LATENCY_DIR, PRED_FILES,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_ID       = "pagand/venra"
ADAPTER_REVISION = "r128"
MAX_SEQ_LENGTH   = 4096
MAX_NEW_TOKENS   = 150     # CoT ceiling — matches paper claim
WARMUP_PASSES    = 3
TIMED_PASSES     = 10
N_SHORT          = 50
N_LONG           = 50
SEED             = 42
LENGTH_THRESHOLD = 512

TOKEN_MAP = {"Supported": " Found", "Unfounded": " Fake", "General": " General"}
EXPECTED_IDS = {"Supported": 12315, "Unfounded": 36965, "General": 3251}

SALSA_OUT = LATENCY_DIR / "latency_salsa_gpu.jsonl"
COT_OUT   = LATENCY_DIR / "latency_cot_gpu.jsonl"


def verify_label_ids(tokenizer) -> List[int]:
    ids = []
    for label, token in TOKEN_MAP.items():
        tids = tokenizer.encode(token, add_special_tokens=False)
        if len(tids) != 1:
            raise ValueError(f"Token '{token}' fragments: {tids}")
        ids.append(tids[0])
    return ids


def timed_salsa(model, inputs, device) -> float:
    """Single SALSA forward pass. Returns wall-clock ms."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def timed_cot(model, inputs, tokenizer, device) -> float:
    """CoT generation. Returns wall-clock ms."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = False,
            eos_token_id   = tokenizer.eos_token_id,
            pad_token_id   = tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def select_latency_sample(manifest, prompts_map, rng) -> List[Dict]:
    """Select N_SHORT short + N_LONG long prompts."""
    short_rows = [r for r in manifest
                  if r["meta_token_count"] < LENGTH_THRESHOLD
                  and r["row_id"] in prompts_map]
    long_rows  = [r for r in manifest
                  if r["meta_token_count"] >= LENGTH_THRESHOLD
                  and r["row_id"] in prompts_map]

    rng.shuffle(short_rows)
    rng.shuffle(long_rows)

    selected = short_rows[:N_SHORT] + long_rows[:N_LONG]
    print(f"[sample] Selected {len(short_rows[:N_SHORT])} short + "
          f"{len(long_rows[:N_LONG])} long prompts for latency benchmark")
    return selected


def run_benchmark(
    model, tokenizer, device,
    benchmark_rows: List[Dict],
    prompts_map: Dict,
    condition: str,          # "salsa" or "cot"
    out_file: Path,
) -> Dict:
    """
    Run WARMUP_PASSES warmup + TIMED_PASSES timed passes for each prompt.
    Returns summary statistics dict.
    """
    all_medians = []

    with open(out_file, "w") as f_out:
        for row in benchmark_rows:
            rid    = row["row_id"]
            prompt = prompts_map[rid]

            inputs = tokenizer(
                prompt,
                return_tensors = "pt",
                truncation     = True,
                max_length     = MAX_SEQ_LENGTH,
            ).to(device)

            # Warmup
            for _ in range(WARMUP_PASSES):
                if condition == "salsa":
                    timed_salsa(model, inputs, device)
                else:
                    timed_cot(model, inputs, tokenizer, device)

            # Timed passes
            times = []
            for _ in range(TIMED_PASSES):
                if condition == "salsa":
                    ms = timed_salsa(model, inputs, device)
                else:
                    ms = timed_cot(model, inputs, tokenizer, device)
                times.append(ms)

            median_ms = float(np.median(times))
            p95_ms    = float(np.percentile(times, 95))
            all_medians.append(median_ms)

            record = {
                "row_id":        rid,
                "condition":     condition,
                "token_count":   row["meta_token_count"],
                "pool":          row["pool"],
                "median_ms":     round(median_ms, 3),
                "p95_ms":        round(p95_ms, 3),
                "all_times_ms":  [round(t, 3) for t in times],
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            del inputs

    summary = {
        "condition":        condition,
        "n_prompts":        len(benchmark_rows),
        "overall_median_ms": round(float(np.median(all_medians)), 3),
        "overall_p95_ms":    round(float(np.percentile(all_medians, 95)), 3),
        "overall_mean_ms":   round(float(np.mean(all_medians)), 3),
    }
    return summary


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    rng = random.Random(SEED)

    print(f"\n[model] Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    label_ids = verify_label_ids(tokenizer)

    print(f"\n[model] Loading VeNRA adapter for latency benchmark")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
        use_cache         = True,
        token             = os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(
        model, ADAPTER_ID, revision=ADAPTER_REVISION,
        token=os.environ.get("HF_TOKEN"),
    )
    model = model.merge_and_unload()
    model.eval()
    print(f"[model] Ready on {next(model.parameters()).device}")

    # Data
    manifest    = load_manifest()
    prompts_map = load_prompts_venra()

    benchmark_rows = select_latency_sample(manifest, prompts_map, rng)

    # ── Condition 1: SALSA ────────────────────────────────────────────────────
    print(f"\n[bench] Running SALSA (single forward pass) "
          f"on {len(benchmark_rows)} prompts ×{TIMED_PASSES} timed passes...")
    salsa_summary = run_benchmark(
        model, tokenizer, device,
        benchmark_rows, prompts_map,
        condition="salsa", out_file=SALSA_OUT,
    )

    # ── Condition 2: CoT ──────────────────────────────────────────────────────
    print(f"\n[bench] Running CoT (generate max {MAX_NEW_TOKENS} tokens) "
          f"on {len(benchmark_rows)} prompts ×{TIMED_PASSES} timed passes...")
    cot_summary = run_benchmark(
        model, tokenizer, device,
        benchmark_rows, prompts_map,
        condition="cot", out_file=COT_OUT,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    speedup = cot_summary["overall_median_ms"] / max(salsa_summary["overall_median_ms"], 1e-6)

    print("\n" + "="*60)
    print("LATENCY BENCHMARK RESULTS")
    print("="*60)
    print(f"  Hardware:         {device}  (RTX 3090 / similar)")
    print(f"  Model:            {ADAPTER_ID}@{ADAPTER_REVISION} (merged)")
    print(f"  N prompts:        {len(benchmark_rows)} "
          f"({N_SHORT} short + {N_LONG} long)")
    print(f"  Timed passes:     {TIMED_PASSES} per prompt")
    print()
    print(f"  SALSA   median:   {salsa_summary['overall_median_ms']:.1f} ms")
    print(f"  SALSA   P95:      {salsa_summary['overall_p95_ms']:.1f} ms")
    print(f"  CoT     median:   {cot_summary['overall_median_ms']:.1f} ms  "
          f"(max_new_tokens={MAX_NEW_TOKENS})")
    print(f"  CoT     P95:      {cot_summary['overall_p95_ms']:.1f} ms")
    print(f"  Speedup (median): {speedup:.1f}×")
    print()
    print(f"  CPU reference:    ~12,000 ms  (HF Space, end-to-end, 1 sample)")
    print("="*60)

    # Save summary
    summary_path = LATENCY_DIR / "latency_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "salsa": salsa_summary,
            "cot":   cot_summary,
            "speedup_x": round(speedup, 1),
            "cpu_reference_ms": 12000,
            "config": {
                "model": f"{ADAPTER_ID}@{ADAPTER_REVISION}",
                "device": str(device),
                "max_new_tokens_cot": MAX_NEW_TOKENS,
                "n_short": N_SHORT,
                "n_long": N_LONG,
                "timed_passes": TIMED_PASSES,
            }
        }, f, indent=2)
    print(f"\n[done] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()