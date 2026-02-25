import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

# ─── CHANGE THESE FOR EACH RUN ────────────────────────────────────────────────
HF_USERNAME = "pagand"
REPO_NAME   = "venra"
BRANCH      = "r96"           # change to "r128"
LORA_RANK   = 96              # change to 128 
LOCAL_PATH  = Path("./data/output/venra-final-noinstruct-1e-4-r96-w0.10")
PRIVATE     = True

# Tag the current HEAD before overwriting — set to None to skip
# e.g. "r96-v1.0" preserves the old version before you push the new one
SNAPSHOT_TAG = "r96-v1.0"    # change to None if this is a first-time push
# ──────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID  = f"{HF_USERNAME}/{REPO_NAME}"

IGNORE_PATTERNS = [
    "checkpoint-*",
    "training_args.bin",
    "training_complete.flag",
]

MODEL_CARD = f"""---
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
library_name: peft
tags:
  - lora
  - peft
  - hallucination-detection
  - venra
license: apache-2.0
---

# VeNRA — LoRA Adapter (r={LORA_RANK})

Fine-tuned LoRA adapter on `Qwen/Qwen2.5-Coder-3B-Instruct` for
hallucination detection in RAG pipelines.

## Available Adapters

| Branch | Rank | Description |
|--------|------|-------------|
| `r96`  | 96   | Lighter, faster inference |
| `r128` | 128  | Higher capacity |

## Labels
- `Found` — supported by context
- `General`   — common knowledge
- `Fake` — contradicts or unsupported by context

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Always latest on this branch
model = PeftModel.from_pretrained(model, "{REPO_ID}", revision="{BRANCH}")

# Pinned to a specific snapshot tag
model = PeftModel.from_pretrained(model, "{REPO_ID}", revision="r96-v1.0")
```

## Training Details
- Rank: {LORA_RANK}
- Learning rate: 1e-4
- Weight decay: 0.10
- Training regime: WeightedLabelTrainer
"""


def push_main_model_card():
    """Run this once to populate the main branch with a landing model card."""

    MAIN_README = MODEL_CARD

    api = HfApi(token=HF_TOKEN)

    # Write to a temp file and upload directly to main
    tmp = Path("/tmp/venra_main_readme.md")
    tmp.write_text(MAIN_README)

    api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        revision="main",          # explicitly targeting main
        commit_message="Add landing model card to main",
    )
    print(f"✅ Model card pushed to main.")
    print(f"   View: https://huggingface.co/{REPO_ID}")


def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not found in .env file.")
    if not LOCAL_PATH.exists():
        raise FileNotFoundError(f"Local path not found: {LOCAL_PATH}")

    (LOCAL_PATH / "README.md").write_text(MODEL_CARD)
    print("✅ Model card written.")

    api = HfApi(token=HF_TOKEN)

    # Create the shared repo if it doesn't exist
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True,
                token=HF_TOKEN, private=PRIVATE)
    print(f"✅ Repository ready: {REPO_ID}")

    # Create branch if it doesn't exist
    api.create_branch(repo_id=REPO_ID, branch=BRANCH,
                      repo_type="model", exist_ok=True)
    print(f"✅ Branch '{BRANCH}' ready.")

    # Snapshot the current HEAD before overwriting, so old version is recoverable
    if SNAPSHOT_TAG:
        try:
            api.create_tag(
                repo_id=REPO_ID,
                tag=SNAPSHOT_TAG,
                revision=BRANCH,        # tag points at current HEAD of the branch
                repo_type="model",
            )
            print(f"✅ Snapshot tag '{SNAPSHOT_TAG}' created (preserves current HEAD).")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  Tag '{SNAPSHOT_TAG}' already exists — skipping snapshot.")
            else:
                raise

    # Upload new weights — creates a new commit on the branch
    api.upload_folder(
        folder_path=LOCAL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        revision=BRANCH,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Update VeNRA LoRA r={LORA_RANK}" + (f" (prev → {SNAPSHOT_TAG})" if SNAPSHOT_TAG else ""),
    )
    print(f"✅ Branch '{BRANCH}' updated with new weights.")

    print(f"\n🎉 Done → https://huggingface.co/{REPO_ID}/tree/{BRANCH}")
    print(f"\nLoading options:")
    print(f"  Latest:  PeftModel.from_pretrained(model, '{REPO_ID}', revision='{BRANCH}')")
    if SNAPSHOT_TAG:
        print(f"  Pinned:  PeftModel.from_pretrained(model, '{REPO_ID}', revision='{SNAPSHOT_TAG}')")


if __name__ == "__main__":
    # push_main_model_card() # run once for model card
    main()