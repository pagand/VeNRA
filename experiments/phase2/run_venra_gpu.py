"""
experiments/phase2/run_venra_gpu.py
-------------------------------------
VeNRA Sentinel SALSA inference on GPU (RTX 3090).

Loads adapter r96, merges weights for speed, runs batched forward pass.
Reads one logit position (last token = "Label:"), applies 3-class softmax.
Crash-safe: skips already-written row_ids on restart.

Usage (GPU server, train .env):
  python -m experiments.phase2.run_venra_gpu
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_prompts_venra,
    get_completed_ids, write_prediction,
    PRED_FILES, GT_SUPPORTED, GT_UNFOUNDED, GT_GENERAL,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_ID       = "pagand/venra"
ADAPTER_REVISION = "r96"
BATCH_SIZE       = 8
MAX_SEQ_LENGTH   = 4096
OUT_FILE         = PRED_FILES["venra_salsa"]

# Orthogonal token mapping — matches train.py TOKEN_MAP
TOKEN_MAP = {
    "Supported": " Found",    # expected ID: 12315
    "Unfounded": " Fake",     # expected ID: 36965
    "General":   " General",  # expected ID: 3251
}
EXPECTED_IDS = {"Supported": 12315, "Unfounded": 36965, "General": 3251}
# pred index: 0=Supported/Found, 1=Unfounded/Fake, 2=General


def verify_label_ids(tokenizer) -> List[int]:
    """Verify orthogonal token IDs match training. Returns ordered [sup, unf, gen]."""
    label_ids = []
    for label, token in TOKEN_MAP.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Token '{token}' for label '{label}' fragments into {len(ids)} "
                f"tokens {ids}. This violates the Orthogonal Label requirement."
            )
        tid = ids[0]
        expected = EXPECTED_IDS[label]
        status = "✅" if tid == expected else "⚠️  MISMATCH"
        print(f"  {status} {label:<12} → '{token}'  (ID: {tid}"
              + (f", expected {expected})" if tid != expected else ")"))
        label_ids.append(tid)
    return label_ids   # [id_for_Found, id_for_Fake, id_for_General]


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Tokenizer
    print(f"\n[model] Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token  = tokenizer.eos_token
    tokenizer.padding_side = "left"   # for batched inference with varied lengths

    # Verify label IDs
    print("\n[tokens] Verifying orthogonal token mapping:")
    label_ids = verify_label_ids(tokenizer)   # [sup_id, unf_id, gen_id]
    label_ids_tensor = torch.tensor(label_ids)

    # Load base model
    print(f"\n[model] Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Load and merge adapter (faster than PEFT forward pass at eval time)
    print(f"[model] Loading adapter {ADAPTER_ID} @ {ADAPTER_REVISION}")
    model = PeftModel.from_pretrained(
        model, ADAPTER_ID, revision=ADAPTER_REVISION,
        token=os.environ.get("HF_TOKEN"),
    )
    print("[model] Merging adapter weights for inference speed...")
    model = model.merge_and_unload()
    model.eval()
    print(f"[model] Ready on {next(model.parameters()).device}")

    # Load data
    manifest = load_manifest()
    prompts  = load_prompts_venra()
    completed = get_completed_ids(OUT_FILE)
    todo      = [r for r in manifest if r["row_id"] not in completed]

    print(f"\n[eval] Total rows: {len(manifest)} | "
          f"Completed: {len(completed)} | Remaining: {len(todo)}")

    if not todo:
        print("[eval] All rows already processed. Exiting.")
        return

    # ── Batched inference ─────────────────────────────────────────────────────
    for batch_start in tqdm(range(0, len(todo), BATCH_SIZE), desc="VeNRA SALSA"):
        batch_rows    = todo[batch_start : batch_start + BATCH_SIZE]
        batch_prompts = [prompts[r["row_id"]] for r in batch_rows]

        inputs = tokenizer(
            batch_prompts,
            return_tensors  = "pt",
            padding         = True,
            truncation      = True,
            max_length      = MAX_SEQ_LENGTH,
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()

        # logits[:, -1, :] = distribution over next token after "Label:"
        logits   = outputs.logits[:, -1, :]             # (B, vocab)
        relevant = logits[:, label_ids_tensor.to(device)]  # (B, 3)
        probs    = F.softmax(relevant, dim=-1)           # (B, 3)
        preds    = torch.argmax(probs, dim=-1).cpu().tolist()
        confs    = torch.max(probs, dim=-1).values.cpu().tolist()

        ms_per_sample = (t1 - t0) * 1000 / len(batch_rows)

        for j, row in enumerate(batch_rows):
            pred_int   = preds[j]
            label_name = ["Found", "Fake", "General"][pred_int]
            write_prediction(OUT_FILE, {
                "row_id":      row["row_id"],
                "pred":        pred_int,
                "valid":       True,   # always valid: we read the logit directly
                "raw":         label_name,
                "confidence":  round(float(confs[j]), 6),
                "latency_ms":  round(ms_per_sample, 3),
                "model":       f"{ADAPTER_ID}@{ADAPTER_REVISION}",
            })

        # Free batch tensors
        del inputs, outputs, logits, relevant, probs

    print(f"\n[done] Predictions saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()