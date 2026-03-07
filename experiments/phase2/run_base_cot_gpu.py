"""
experiments/phase2/run_base_cot_gpu.py
----------------------------------------
Base model Chain-of-Thought inference on the 50-pair CoT subsample (~100 rows).

PURPOSE: Quantify the "test-time compute budget" the base model needs to
reach a correct verdict vs VeNRA SALSA's budget of exactly 1 token.
This is the empirical proof of the "1.5 thinking" claim.

Only runs on rows where cot_subsample=True AND pool is a pair pool.
That is ~100 rows (50 parent + 50 child), built by build_manifest.py.
Running CoT on the full 800+ row test set would take hours sequentially
and adds nothing — the budget distribution is the claim, not the composite M.

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
    load_manifest,
    load_prompts_frontier,
    get_completed_ids,
    write_prediction,
    parse_cot_response,
    ALL_PAIR_TAGS,
    PRED_FILES,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 500
OUT_FILE       = PRED_FILES["base_qwen_cot"]

COT_CLOSING = (
    "\n\nThink step by step through each AUDIT ALGORITHM check above. "
    "After your reasoning, write your final verdict on the very last line "
    "as exactly one word — nothing else on that line: Found, Fake, or General."
)

COT_PREFILL = "Let me analyze each AUDIT ALGORITHM criterion:\n\n"


def build_cot_prompt(system_content: str, user_content: str) -> str:
    """
    Build a CoT prompt by pre-filling the start of the assistant turn.

    WHY PRE-FILL: Qwen2.5-Instruct is heavily aligned to give short, direct
    answers in chat format. Appending reasoning instructions to the user turn
    does not help — the model still snaps to one label word (token_budget=1)
    because that IS the correct behavior for a well-aligned chat model.
    Pre-filling the assistant turn forces the model to CONTINUE the started
    reasoning rather than respond to it, bypassing the short-answer bias.

    The pre-fill tokens are part of the input; token_budget counts only the
    new tokens the model generates after the pre-fill.
    """
    user_cot = user_content.rstrip()

    # Strip trailing 'Label:' conditioning suffix
    if user_cot.endswith("Label:"):
        user_cot = user_cot[: -len("Label:")].rstrip()

    # Strip any label-first instruction
    for phrase in [
        "Output your label (Found, Fake, or General) first, followed by your analysis.",
        "Output label (Found, Fake, or General) first, followed by your analysis.",
    ]:
        user_cot = user_cot.replace(phrase, "").rstrip()

    # Append the task instruction to the user turn
    user_cot = user_cot + COT_CLOSING

    # Pre-fill assistant turn — model continues from here
    return (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_cot}<|im_end|>\n"
        f"<|im_start|>assistant\n{COT_PREFILL}"
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

    # ── Token ID constants (computed once, used in every generate call) ────────
    # For Qwen2.5, tokenizer.eos_token_id IS <|im_end|> (151645).
    # Passing that as eos_token_id makes generate() stop after the first label
    # word + the chat-turn closer — always 2 tokens, never any reasoning.
    # We use <|endoftext|> (151643) as the true generation stop instead, then
    # manually truncate at the first <|im_end|> when decoding.
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    im_end_id    = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"[tokens] endoftext_id={endoftext_id}  im_end_id={im_end_id}  "
          f"(eos_token_id={tokenizer.eos_token_id})")
    if endoftext_id == im_end_id:
        # Some tokenizer configs equate these — fall back to eos and warn.
        print("[warn] endoftext_id == im_end_id — model will stop after first word. "
              "Check tokenizer config.")

    print("\n[data] Loading manifest + frontier prompts...")
    manifest  = load_manifest()
    frontier  = load_prompts_frontier()
    completed = get_completed_ids(OUT_FILE)

    # Only rows flagged as cot_subsample AND in a pair pool.
    # build_manifest.py guarantees cot_subsample=True only on pair rows
    # (not axiom/natural), so the second check is a safety guard.
    cot_rows = [
        r for r in manifest
        if r.get("cot_subsample", False)
        and r["pool"] in ALL_PAIR_TAGS
        and r["row_id"] not in completed
    ]
    total_cot = sum(
        1 for r in manifest
        if r.get("cot_subsample", False) and r["pool"] in ALL_PAIR_TAGS
    )

    print(f"[eval] CoT subsample rows: {total_cot} total | "
          f"Completed: {total_cot - len(cot_rows)} | Remaining: {len(cot_rows)}")

    if not cot_rows:
        print("[eval] All done. Exiting.")
        return

    todo = cot_rows

    # Print the first prompt so we can verify COT_CLOSING is present
    _first = todo[0]
    _fp    = frontier.get(_first["row_id"])
    if _fp:
        _debug_prompt = build_cot_prompt(_fp["system_content"], _fp["user_content"])
        print(f"\n[debug] First CoT prompt tail (last 400 chars, includes pre-fill):\n"
              f"{'─'*60}\n"
              f"{_debug_prompt[-400:]}\n"
              f"{'─'*60}\n"
              f"  Pre-fill: {repr(COT_PREFILL[:80])}\n"
              f"  Model generates FROM the pre-fill onwards.\n")

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
                # No eos_token_id — let the model generate until max_new_tokens.
                # The pre-filled assistant turn means the model reasons freely;
                # we truncate at <|im_end|> in post-processing to get clean text
                # and an accurate token_budget.
                pad_token_id            = tokenizer.eos_token_id,
                return_dict_in_generate = True,
                output_scores           = False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated_ids = output.sequences[0][input_len:]

        # Truncate at first <|im_end|>: the model closes its assistant turn
        # naturally there. Tokens after it would be a new spurious turn.
        im_end_positions = (generated_ids == im_end_id).nonzero(as_tuple=True)[0]
        if len(im_end_positions) > 0:
            generated_ids = generated_ids[:im_end_positions[0].item()]

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