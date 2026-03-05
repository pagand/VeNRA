"""
experiments/phase2/run_base_model_gpu.py
-----------------------------------------
Base Qwen2.5-Coder-3B-Instruct zero-shot inference on GPU.

Identical setup to run_venra_gpu.py EXCEPT no adapter is loaded.
Same prompts (Qwen format, ending with "Label:"), same label ID extraction.
The delta in composite metric vs venra_salsa is 100% attributable to fine-tuning.

Usage (GPU server, train .env):
  python -m experiments.phase2.run_base_model_gpu
"""

import os
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_prompts_venra,
    get_completed_ids, write_prediction,
    PRED_FILES,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen2.5-Coder-3B-Instruct"
BATCH_SIZE     = 8
MAX_SEQ_LENGTH = 4096
OUT_FILE       = PRED_FILES["base_qwen_zeroshot"]

TOKEN_MAP = {
    "Supported": " Found",
    "Unfounded": " Fake",
    "General":   " General",
}
EXPECTED_IDS = {"Supported": 12315, "Unfounded": 36965, "General": 3251}


def verify_label_ids(tokenizer) -> List[int]:
    label_ids = []
    for label, token in TOKEN_MAP.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Token '{token}' for '{label}' fragments into {len(ids)} tokens."
            )
        tid = ids[0]
        status = "✅" if tid == EXPECTED_IDS[label] else "⚠️  MISMATCH"
        print(f"  {status} {label:<12} → '{token}'  (ID: {tid})")
        label_ids.append(tid)
    return label_ids


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

    print("\n[tokens] Verifying orthogonal token mapping:")
    label_ids = verify_label_ids(tokenizer)
    label_ids_tensor = torch.tensor(label_ids)

    # Load base model — NO ADAPTER
    print(f"\n[model] Loading base model (no adapter): {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
        trust_remote_code = True,
        use_cache   = True,
        token       = os.environ.get("HF_TOKEN"),
    )
    model.eval()
    print(f"[model] Ready on {next(model.parameters()).device}")

    manifest  = load_manifest()
    prompts   = load_prompts_venra()          # same prompts as VeNRA
    completed = get_completed_ids(OUT_FILE)
    todo      = [r for r in manifest if r["row_id"] not in completed]

    print(f"\n[eval] Rows: {len(manifest)} | "
          f"Completed: {len(completed)} | Remaining: {len(todo)}")

    if not todo:
        print("[eval] All done. Exiting.")
        return

    for batch_start in tqdm(range(0, len(todo), BATCH_SIZE), desc="Base zero-shot"):
        batch_rows    = todo[batch_start : batch_start + BATCH_SIZE]
        batch_prompts = [prompts[r["row_id"]] for r in batch_rows]

        inputs = tokenizer(
            batch_prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = MAX_SEQ_LENGTH,
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        logits   = outputs.logits[:, -1, :]
        relevant = logits[:, label_ids_tensor.to(device)]
        probs    = F.softmax(relevant, dim=-1)
        preds    = torch.argmax(probs, dim=-1).cpu().tolist()
        confs    = torch.max(probs, dim=-1).values.cpu().tolist()

        ms_per_sample = (t1 - t0) * 1000 / len(batch_rows)

        for j, row in enumerate(batch_rows):
            pred_int   = preds[j]
            label_name = ["Found", "Fake", "General"][pred_int]
            write_prediction(OUT_FILE, {
                "row_id":     row["row_id"],
                "pred":       pred_int,
                "valid":      True,
                "raw":        label_name,
                "confidence": round(float(confs[j]), 6),
                "latency_ms": round(ms_per_sample, 3),
                "model":      BASE_MODEL + "_zeroshot",
            })

        del inputs, outputs, logits, relevant, probs

    print(f"\n[done] Predictions saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()