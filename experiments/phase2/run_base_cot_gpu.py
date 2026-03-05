"""
experiments/phase2/run_base_cot_gpu.py
----------------------------------------
Base model Chain-of-Thought inference on the 50-pair CoT subsample.

PURPOSE: Quantify the "test-time compute budget" the base model needs to
reach a correct verdict vs VeNRA SALSA's budget of exactly 1 token.
This is the empirical proof of the "1.5 thinking" claim.

Key outputs per row:
  pred          — predicted label (parsed from LAST valid word in generation)
  valid         — whether generation contained a valid label word
  token_budget  — number of tokens generated before stopping
  generated_text — the full reasoning chain (for spot-check, not metrics)

The prompt uses the same content as VeNRA but does NOT end with "Label:".
Instead, the final instruction asks the model to reason then conclude with
exactly one word: Found, Fake, or General.

Usage (GPU server, train .env):
  python -m experiments.phase2.run_base_cot_gpu
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_prompts_frontier,
    get_completed_ids, write_prediction, parse_cot_response,
    PRED_FILES,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_SEQ_LENGTH   = 4096
MAX_NEW_TOKENS   = 300     # budget ceiling; most should finish before this
OUT_FILE         = PRED_FILES["base_qwen_cot"]

# CoT instruction appended in place of "Output your label first..."
COT_CLOSING = (
    "Think step by step through each AUDIT ALGORITHM check above. "
    "After your reasoning, write your final verdict on the last line "
    "as exactly one word: Found, Fake, or General."
)


def build_cot_prompt(system_content: str, user_content: str) -> str:
    """
    Convert a frontier-format prompt into a Qwen CoT prompt.
    Removes the trailing 'Label:' cue and replaces the output-first instruction
    with a reason-then-conclude instruction.
    The model generates freely; we parse the LAST valid label word.
    """
    # Strip trailing "Label:" that conditions first-token output
    user_cot = user_content.rstrip()
    if user_cot.endswith("Label:"):
        user_cot = user_cot[: -len("Label:")].rstrip()

    # Replace "Output your label first" with CoT instruction
    user_cot = user_cot.replace(
        "Output your label (Found, Fake, or General) first, followed by your analysis.",
        COT_CLOSING,
    )
    # Also handle the shorter noinstruct variant just in case
    user_cot = user_cot.replace(
        "Output label (Found, Fake, or General) first, followed by your analysis.",
        COT_CLOSING,
    )

    # Re-wrap in Qwen chat format, assistant turn is open (no "Label:")
    return (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_cot}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print(f"\n[model] Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model — no adapter
    print(f"\n[model] Loading base model (no adapter): {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
        use_cache         = True,
        token             = os.environ.get("HF_TOKEN"),
    )
    model.eval()
    print(f"[model] Ready on {next(model.parameters()).device}")

    # Load data — CoT subsample only
    manifest  = load_manifest()
    frontier  = load_prompts_frontier()
    completed = get_completed_ids(OUT_FILE)

    # Filter: only cot_subsample=True rows that are pairs (parent + child)
    cot_rows = [
        r for r in manifest
        if r.get("cot_subsample", False)
        and r["pool"] in ("pair_short_parent", "pair_short_child",
                          "pair_long_parent",  "pair_long_child")
        and r["row_id"] not in completed
    ]

    print(f"\n[eval] CoT subsample rows: {len(cot_rows)} "
          f"(skipping {len(completed)} already done)")

    if not cot_rows:
        print("[eval] Nothing to do. Exiting.")
        return

    # ── Sequential inference (CoT is slow; batching is tricky with varied lengths) ──
    for row in tqdm(cot_rows, desc="Base CoT"):
        fp = frontier[row["row_id"]]
        prompt = build_cot_prompt(fp["system_content"], fp["user_content"])

        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = MAX_SEQ_LENGTH,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens       = MAX_NEW_TOKENS,
                do_sample            = False,        # greedy — deterministic
                eos_token_id         = tokenizer.eos_token_id,
                pad_token_id         = tokenizer.eos_token_id,
                return_dict_in_generate = True,
                output_scores        = False,        # not needed for budget test
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated_ids   = output.sequences[0][input_len:]
        token_budget    = int(generated_ids.shape[0])
        generated_text  = tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency_ms      = (t1 - t0) * 1000

        pred, valid = parse_cot_response(generated_text)

        write_prediction(OUT_FILE, {
            "row_id":         row["row_id"],
            "pred":           pred,
            "valid":          valid,
            "raw":            generated_text[-300:],  # last 300 chars for debugging
            "token_budget":   token_budget,
            "latency_ms":     round(latency_ms, 3),
            "confidence":     0.5,   # not meaningful for CoT
            "model":          BASE_MODEL + "_cot",
        })

        del inputs, output, generated_ids

    print(f"\n[done] CoT predictions saved to: {OUT_FILE}")

    # ── Budget summary ────────────────────────────────────────────────────────
    import json, numpy as np
    budgets = []
    with open(OUT_FILE) as f:
        for line in f:
            if line.strip():
                budgets.append(json.loads(line)["token_budget"])
    if budgets:
        arr = np.array(budgets)
        print(f"\n[budget] Token budget statistics (n={len(arr)}):")
        print(f"  Median: {np.median(arr):.1f}")
        print(f"  P25:    {np.percentile(arr, 25):.1f}")
        print(f"  P75:    {np.percentile(arr, 75):.1f}")
        print(f"  P95:    {np.percentile(arr, 95):.1f}")
        print(f"  Max:    {arr.max():.1f}")


if __name__ == "__main__":
    main()