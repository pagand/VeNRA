#!/usr/bin/env python3
"""
VeNRA Quick Training Test (5 steps)
=====================================
Tests the full training pipeline with just 5 steps.
Takes ~2-3 minutes. Run this before committing to full 3-4 hour training.
"""

import os
import sys
import torch
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

print("="*70)
print("VeNRA Quick Training Test (5 steps)")
print("="*70)
print("\nThis will:")
print("  1. Load the model with 4-bit quantization")
print("  2. Load the dataset")
print("  3. Run 5 training steps")
print("  4. Verify no errors occur")
print("\nEstimated time: 2-3 minutes")
print("="*70)

# Import after dotenv so HF_TOKEN is available
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Configuration
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
TOKEN_MAP = {
    "Supported": " Found",
    "Unfounded": " Fake",
    "General": " General"
}

def format_prompt_func(example):
    """Format training examples (same as main training script)"""
    output_texts = []
    batch_size = len(example['split'])
    
    for i in range(batch_size):
        raw_label = example['label'][i] 
        target_token = TOKEN_MAP.get(raw_label)
        if not target_token: 
            continue

        q = example['input_components'][i]['query']
        c_raw = example['input_components'][i]['context']
        c = "\n".join(c_raw) if isinstance(c_raw, list) else str(c_raw)
        t = example['input_components'][i]['trace']
        s = example['output_components'][i]['target_sentence']
        r = example['output_components'][i]['reasoning']

        prompt = (
            f"<|im_start|>system\nYou are a financial auditor.<|im_end|>\n"
            f"<|im_start|>user\nQuery: {q}\nContext: {c}\nTrace: {t}\nStatement: {s}\n"
            f"Task: Classify [Found, Fake, General].<|im_end|>\n"
            f"<|im_start|>assistant\nLabel:"
        )
        completion = f"{target_token}\nAnalysis: {r}<|im_end|>"
        output_texts.append(prompt + completion)
        
    return output_texts

def main():
    # Check environment
    if "HF_TOKEN" not in os.environ:
        print("❌ ERROR: HF_TOKEN not found in environment")
        print("Add it to .env file: HF_TOKEN=hf_xxxxx")
        return 1
    
    print("\n[1/6] Loading dataset...")
    try:
        dataset = load_dataset("pagand/venra", revision="v2.1")
        # Use only first 100 samples for quick test
        train_subset = dataset["train"].select(range(100))
        val_subset = dataset["validation"].select(range(50))
        print(f"✓ Loaded {len(train_subset)} train samples, {len(val_subset)} val samples")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return 1
    
    print("\n[2/6] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return 1
    
    print("\n[3/6] Loading model with 4-bit quantization...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        model = prepare_model_for_kbit_training(model)
        
        gpu_mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"✓ Model loaded ({gpu_mem:.2f}GB GPU memory)")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return 1
    
    print("\n[4/6] Configuring LoRA...")
    try:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=32,
            lora_dropout=0.05,
            use_rslora=True,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        )
        print("✓ LoRA configured (r=64, alpha=32, rsLoRA=True)")
    except Exception as e:
        print(f"❌ LoRA configuration failed: {e}")
        return 1
    
    print("\n[5/6] Setting up trainer...")
    try:
        training_args = TrainingArguments(
            output_dir="./data/test-output",
            max_steps=5,  # Only 5 steps!
            per_device_train_batch_size=2,  # Small batch for speed
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=1,
            save_strategy="no",  # Don't save checkpoints
            evaluation_strategy="no",  # Skip eval for speed
            bf16=True,
            gradient_checkpointing=True,
            report_to="none",  # No wandb for test
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_subset,
            peft_config=peft_config,
            formatting_func=format_prompt_func,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="\nLabel:",
                tokenizer=tokenizer
            ),
            max_seq_length=2048,  # Shorter for speed
            tokenizer=tokenizer,
            args=training_args,
        )
        print("✓ Trainer configured")
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return 1
    
    print("\n[6/6] Running 5 training steps...")
    print("(This is the critical test - if this works, full training will work)")
    print("-" * 70)
    
    try:
        trainer.train()
        print("-" * 70)
        print("✓ Training test completed successfully!")
    except Exception as e:
        print("-" * 70)
        print(f"❌ Training failed: {e}")
        
        if "CUDA out of memory" in str(e):
            print("\n💡 FIX: Reduce batch size in main training:")
            print("  training_args.per_device_train_batch_size = 2")
            print("  training_args.gradient_accumulation_steps = 16")
        
        return 1
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYour environment is ready for full training.")
    print("\nTo start full training:")
    print("  python src/hal_det/training/train.py --output_dir ./data/output")
    print("\nEstimated time: 3-4 hours on RTX 3090")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())