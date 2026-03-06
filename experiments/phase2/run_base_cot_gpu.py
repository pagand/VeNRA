"""
experiments/phase2/run_base_cot_gpu.py
----------------------------------------
Base model Chain-of-Thought inference on the FULL test set.

PURPOSE: Quantify the "test-time compute budget" the base model needs to
reach a correct verdict vs VeNRA SALSA's budget of exactly 1 token.
This is the empirical proof of the "1.5 thinking" claim:
  - VeNRA reads 1 token, budget = 1
  - Base model reasons for N tokens to reach the same verdict

The comparison is fair only if both run on the same full test set.
The earlier 50-pair "subsample" idea was dropped because:
  a) All other scripts already run the full set
  b) The 3090 handles CoT × 800 rows comfortably
  c) A subsample can only compute flip rate, not the full composite M

Key outputs per row:
  pred          — label parsed from LAST valid word in generation
  valid         — whether generation produced a valid label word
  token_budget  — tokens generated before the model stopped
  raw           — last 500 chars of generation (debug spot-check)

Usage (GPU server, train .env):
  python -m experiments.phase2.run_base_cot_gpu
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from experiments.phase2.utils import (
    ensure_dirs,
    load_manifest_slim,
    load_prompts_frontier,
    get_completed_ids,
    write_prediction,
    parse_cot_response,
    PRED_FILES,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 300
OUT_FILE       = PRED_FILES["base_qwen_cot"]

COT_CLOSING = (
    "Think step by step through each AUDIT ALGORITHM check above. "
    "After your reasoning, write your final verdict on the last line "
    "as exactly one word: Found, Fake, or General."
)


def build_cot_prompt(system_content: str, user_content: str) -> str:
    """
    Frontier prompt → Qwen CoT prompt.
    Strips trailing 'Label:' and replaces label-first instruction with
    reason-then-label instruction. Model generates freely; we parse the
    LAST valid label word from output.
    """
    user_cot = user_content.rstrip()
    if user_cot.endswith("Label:"):
        user_cot = user_cot[: -len("Label:")].rstrip()

    for phrase in [
        "Output your label (Found, Fake, or General) first, followed by your analysis.",
        "Output label (Found, Fake, or General) first, followed by your analysis.",
    ]:
        user_cot = user_cot.replace(phrase, COT_CLOSING)

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

    print("\n[data] Loading manifest (slim) + frontier prompts...")
    manifest  = load_manifest_slim()
    frontier  = load_prompts_frontier()
    completed = get_completed_ids(OUT_FILE)

    # ALL rows — no subsample filter
    todo = [r for r in manifest if r["row_id"] not in completed]
    print(f"[eval] Total: {len(manifest)} | Completed: {len(completed)} | Remaining: {len(todo)}")

    if not todo:
        print("[eval] All done. Exiting.")
        return

    for row in tqdm(todo, desc="Base CoT", unit="row"):
        rid = row["row_id"]
        fp  = frontier.get(rid)
        if fp is None:
            print(f"  [warn] row {rid} missing from frontier prompts — skipping")
            continue

        prompt    = build_cot_prompt(fp["system_content"], fp["user_content"])
        inputs    = tokenizer(
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
                max_new_tokens          = MAX_NEW_TOKENS,
                do_sample               = False,
                eos_token_id            = tokenizer.eos_token_id,
                pad_token_id            = tokenizer.eos_token_id,
                return_dict_in_generate = True,
                output_scores           = False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated_ids  = output.sequences[0][input_len:]
        token_budget   = int(generated_ids.shape[0])
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred, valid = parse_cot_response(generated_text)

        write_prediction(OUT_FILE, {
            "row_id":       rid,
            "pred":         pred,
            "valid":        valid,
            "raw":          generated_text[-500:],
            "token_budget": token_budget,
            "latency_ms":   round((t1 - t0) * 1000, 3),
            "confidence":   0.5,
            "model":        BASE_MODEL + "_cot",
        })

        del inputs, output, generated_ids

    print(f"\n[done] Predictions → {OUT_FILE}")

    # ── Budget summary ─────────────────────────────────────────────────────────
    budgets = []
    with open(OUT_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    budgets.append(json.loads(line)["token_budget"])
                except (KeyError, json.JSONDecodeError):
                    pass

    if budgets:
        arr = np.array(budgets)
        print(f"\n[budget] Token budget statistics ({len(arr)} rows):")
        print(f"  Median : {np.median(arr):.1f} tokens")
        print(f"  P25    : {np.percentile(arr, 25):.1f}")
        print(f"  P75    : {np.percentile(arr, 75):.1f}")
        print(f"  P95    : {np.percentile(arr, 95):.1f}")
        print(f"  Max    : {arr.max():.1f}")
        print(f"  VeNRA SALSA uses 1 token  (~{np.median(arr):.0f}× less compute)")


if __name__ == "__main__":
    main()