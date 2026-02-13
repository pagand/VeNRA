"""
VeNRA Hallucination Judge Training Pipeline (v3.0)
===================================================
Phase 1 Implementation: Baseline QLoRA + rsLoRA Training

DEFERRED TO PHASE 2:
- GaLore scaling to 7B (Section 7.1) - Only if 3B fails precision targets
- Hyperparameter grid search (Section 4.2, Table) - Only if baseline underperforms
- LRSL Logit Reranking (Section 7.2) - Only if systematic bias observed
- Temperature Scaling for ECE (Section 6.4) - Post-training calibration step
"""

import os
import sys
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainerCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from dotenv import load_dotenv

# --- Configuration & Defaults ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Orthogonal Token Mapping (Spec Section 3.1)
TOKEN_MAP = {
    "Supported": " Found",   # ID: 1374
    "Unfounded": " Fake",    # ID: 14757
    "General":   " General"  # ID: 15415
}

# Sabotage Type Categories (Spec Section 6.2)
LOGIC_SABOTAGE_TYPES = {"code_lie", "neighbor_trap", "calculation_error"}
NATURAL_FAILURE_TYPE = "natural"

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=DEFAULT_MODEL_ID)
    lora_rank: int = field(default=64, metadata={"help": "LoRA Rank (r). Spec default: 64"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA Alpha. Spec default: 32 (rsLoRA)"})
    learning_rate: float = field(default=2e-4, metadata={"help": "LR. Spec default: 2e-4"})

def check_env_vars():
    """Load and validate environment variables from .env file (Spec requirement)."""
    load_dotenv()  # Load from .env file
    
    if "HF_TOKEN" not in os.environ:
        raise ValueError(
            "CRITICAL: HF_TOKEN not found in environment variables.\n"
            "Please add HF_TOKEN=hf_xxx to your .env file."
        )
    if "WANDB_API_KEY" not in os.environ:
        print("Warning: WANDB_API_KEY not found. Logging will be local only.")

# --- Custom Callback: Stratified Sabotage Audit ---
class SabotageEvalCallback(TrainerCallback):
    """
    Principal Scientist Tool (Spec Section 6.2):
    Calculates Paired Flip Rate, stratified by:
    1. Context Length (Short < 1024 tokens, Long >= 1024 tokens)
    2. Sabotage Type (Logic Sabotage vs Natural Failures)
    """
    def __init__(self, eval_dataset, tokenizer, token_map_ids):
        self.tokenizer = tokenizer
        self.token_map_ids = token_map_ids
        
        print("Initializing Sabotage Callback (Building Paired Test Sets)...")
        
        families = {}
        for i, row in enumerate(eval_dataset):
            # Safe access to metadata
            meta = row.get('meta', {})
            fid = meta.get('family_id')
            if fid:
                if fid not in families: 
                    families[fid] = []
                families[fid].append(row)
            if i > 3000: 
                break # Safety limit
            
        # Build FOUR stratified pair lists (2x2 grid)
        self.pairs_short_natural = []
        self.pairs_short_logic = []
        self.pairs_long_natural = []
        self.pairs_long_logic = []
        
        for fid, members in families.items():
            parent = next((m for m in members if m['label'] == 'Supported'), None)
            child = next((m for m in members if m['label'] == 'Unfounded'), None)
            
            if parent and child:
                # Extract metadata
                t_count = parent.get('meta', {}).get('token_count', 0)
                sabotage_type = child.get('sabotage_type', 'unknown')  # Top-level field!
                
                pair = (parent, child)
                
                # Stratify by BOTH dimensions (Spec Section 6.2)
                is_long = t_count >= 1024  # CORRECTED: Spec says < 1024 for short
                is_logic = sabotage_type in LOGIC_SABOTAGE_TYPES
                
                if is_long and is_logic:
                    self.pairs_long_logic.append(pair)
                elif is_long and not is_logic:
                    self.pairs_long_natural.append(pair)
                elif not is_long and is_logic:
                    self.pairs_short_logic.append(pair)
                else:  # short + natural
                    self.pairs_short_natural.append(pair)
        
        # Limit size for speed (30 pairs per category = 120 total examples)
        self.pairs_short_natural = self.pairs_short_natural[:30]
        self.pairs_short_logic = self.pairs_short_logic[:30]
        self.pairs_long_natural = self.pairs_long_natural[:30]
        self.pairs_long_logic = self.pairs_long_logic[:30]
        
        print(f"Sabotage Callback: Stratified Pair Counts:")
        print(f"  Short + Natural: {len(self.pairs_short_natural)}")
        print(f"  Short + Logic:   {len(self.pairs_short_logic)}")
        print(f"  Long + Natural:  {len(self.pairs_long_natural)}")
        print(f"  Long + Logic:    {len(self.pairs_long_logic)}")

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        
        # Helper to run eval on a specific list of pairs
        def evaluate_subset(pairs, prefix):
            if not pairs: 
                return 0.0, 0.0
            
            prompts = []
            ece_labels = []
            for parent, child in pairs:
                prompts.append(self._make_prompt(parent))
                ece_labels.append(0) # Found (Supported)
                prompts.append(self._make_prompt(child))
                ece_labels.append(1) # Fake (Unfounded)
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=4096
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            next_token_logits = outputs.logits[:, -1, :]
            relevant_ids = [
                self.token_map_ids["Supported"], 
                self.token_map_ids["Unfounded"]
            ]
            probs = torch.softmax(next_token_logits[:, relevant_ids], dim=-1)
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()
            
            # Flip Rate (Paired Sensitivity - Spec Section 6.2, Metric 3)
            correct_flips = 0
            for i in range(len(pairs)):
                parent_correct = predictions[i*2] == 0      # Should predict Found
                child_correct = predictions[i*2 + 1] == 1   # Should predict Fake
                if parent_correct and child_correct:
                    correct_flips += 1
            flip_rate = correct_flips / len(pairs) if len(pairs) > 0 else 0.0
            
            # ECE (Expected Calibration Error - Spec Section 6.4, Metric 7)
            accuracies = (predictions == np.array(ece_labels)).astype(float)
            ece = np.abs(confidences - accuracies).mean()
            
            return flip_rate, ece

        # Run Stratified Audit (2x2 grid)
        fr_short_nat, ece_short_nat = evaluate_subset(self.pairs_short_natural, "short_nat")
        fr_short_log, ece_short_log = evaluate_subset(self.pairs_short_logic, "short_log")
        fr_long_nat, ece_long_nat = evaluate_subset(self.pairs_long_natural, "long_nat")
        fr_long_log, ece_long_log = evaluate_subset(self.pairs_long_logic, "long_log")
        
        # Marginal Aggregates
        total_short = len(self.pairs_short_natural) + len(self.pairs_short_logic)
        total_long = len(self.pairs_long_natural) + len(self.pairs_long_logic)
        total_natural = len(self.pairs_short_natural) + len(self.pairs_long_natural)
        total_logic = len(self.pairs_short_logic) + len(self.pairs_long_logic)
        total_all = total_short + total_long
        
        # Weighted averages for marginals
        if total_short > 0:
            fr_short = (fr_short_nat * len(self.pairs_short_natural) + 
                       fr_short_log * len(self.pairs_short_logic)) / total_short
        else:
            fr_short = 0.0
            
        if total_long > 0:
            fr_long = (fr_long_nat * len(self.pairs_long_natural) + 
                      fr_long_log * len(self.pairs_long_logic)) / total_long
        else:
            fr_long = 0.0
            
        if total_natural > 0:
            fr_natural = (fr_short_nat * len(self.pairs_short_natural) + 
                         fr_long_nat * len(self.pairs_long_natural)) / total_natural
        else:
            fr_natural = 0.0
            
        if total_logic > 0:
            fr_logic = (fr_short_log * len(self.pairs_short_logic) + 
                       fr_long_log * len(self.pairs_long_logic)) / total_logic
        else:
            fr_logic = 0.0
        
        # Global metric (for model selection - Spec Section 4.1)
        if total_all > 0:
            fr_global = (fr_short * total_short + fr_long * total_long) / total_all
        else:
            fr_global = 0.0

        print(f"\n[Sabotage Audit] Step {state.global_step}:")
        print(f"  Short Context (< 1024 tok): Overall={fr_short:.2%}")
        print(f"    ├─ Natural Failures: {fr_short_nat:.2%} (ECE={ece_short_nat:.4f})")
        print(f"    └─ Logic Sabotage:   {fr_short_log:.2%} (ECE={ece_short_log:.4f})")
        print(f"  Long Context (>= 1024 tok): Overall={fr_long:.2%}")
        print(f"    ├─ Natural Failures: {fr_long_nat:.2%} (ECE={ece_long_nat:.4f})")
        print(f"    └─ Logic Sabotage:   {fr_long_log:.2%} (ECE={ece_long_log:.4f})")
        print(f"  By Sabotage Type:")
        print(f"    ├─ Natural: {fr_natural:.2%}")
        print(f"    └─ Logic:   {fr_logic:.2%}")
        print(f"  GLOBAL FLIP RATE: {fr_global:.2%}")
        
        if wandb.run:
            # CORRECTED: Use 'eval_' prefix to match training_args.metric_for_best_model
            wandb.log({
                # Global metric (for checkpoint selection)
                "eval_audit/flip_rate_global": fr_global,
                
                # Marginal metrics (for analysis)
                "eval_audit/flip_rate_short": fr_short,
                "eval_audit/flip_rate_long": fr_long,
                "eval_audit/flip_rate_natural": fr_natural,
                "eval_audit/flip_rate_logic": fr_logic,
                
                # Fine-grained 2x2 grid
                "eval_audit/flip_rate_short_natural": fr_short_nat,
                "eval_audit/flip_rate_short_logic": fr_short_log,
                "eval_audit/flip_rate_long_natural": fr_long_nat,
                "eval_audit/flip_rate_long_logic": fr_long_log,
                
                # Calibration metrics
                "eval_audit/ece_short_natural": ece_short_nat,
                "eval_audit/ece_short_logic": ece_short_log,
                "eval_audit/ece_long_natural": ece_long_nat,
                "eval_audit/ece_long_logic": ece_long_log,
                
                "global_step": state.global_step
            })
        
    def _make_prompt(self, example):
        """Format input for inference (no completion part)."""
        q = example['input_components']['query']
        c_raw = example['input_components']['context']
        c = "\n".join(c_raw) if isinstance(c_raw, list) else str(c_raw)
        t = example['input_components']['trace']
        s = example['output_components']['target_sentence']
        return (
            f"<|im_start|>system\nYou are a financial auditor.<|im_end|>\n"
            f"<|im_start|>user\nQuery: {q}\nContext: {c}\nTrace: {t}\nStatement: {s}\n"
            f"Task: Classify [Found, Fake, General].<|im_end|>\n"
            f"<|im_start|>assistant\nLabel:"
        )

# --- Standard Formatting Function ---
def format_prompt_func(example):
    """
    Format training examples with Label-First structure (SALSA Framework - Spec Section 3.2).
    Loss is calculated ONLY on the Label token via DataCollatorForCompletionOnlyLM.
    """
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

        # Spec Section 3.2: Label-First Template
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
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    check_env_vars()
    
    print(f"Loading dataset pagand/venra (v2.1)...")
    dataset = load_dataset("pagand/venra", revision="v2.1")
    
    print(f"Initializing tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    # Verify Orthogonal Token Mapping (Spec Section 3.1)
    print("Verifying Orthogonal Token Mapping...")
    token_map_ids = {}
    for k, v in TOKEN_MAP.items():
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) != 1: 
            raise ValueError(
                f"CRITICAL: Token '{v}' for label '{k}' is fragmented into {len(ids)} tokens!\n"
                f"Token IDs: {ids}\n"
                f"This violates the Orthogonal Label requirement (Spec Section 3.1)."
            )
        token_map_ids[k] = ids[0]
        print(f"  {k:12s} -> '{v}' (ID: {ids[0]})")
    
    # Spec Section 2.2: 4-bit NF4 Quantization
    print("Configuring 4-bit NF4 Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading base model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Required for gradient checkpointing
    )
    model = prepare_model_for_kbit_training(model)

    # Spec Section 2.2: rsLoRA Configuration
    print(f"Configuring rsLoRA (r={model_args.lora_rank}, alpha={model_args.lora_alpha})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.05,
        use_rslora=True,  # Rank-Stabilized LoRA (Spec Section 2.2)
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"      # MLP (Reasoning)
        ]
    )

    # --- Training Args (Spec Section 4.1: The "Recipe") ---
    print("Configuring Training Arguments...")
    training_args.num_train_epochs = 1.0  # STRICT: Prevent sabotage algorithm overfitting
    training_args.per_device_train_batch_size = 4
    training_args.gradient_accumulation_steps = 8  # Effective batch = 32
    training_args.learning_rate = model_args.learning_rate
    training_args.lr_scheduler_type = "cosine"
    training_args.warmup_ratio = 0.03
    training_args.logging_steps = 10
    
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 50 
    training_args.save_strategy = "steps"
    training_args.save_steps = 50
    training_args.load_best_model_at_end = True
    
    # Model selection based on Paired Flip Rate (Spec Section 6.2, Metric 3)
    training_args.metric_for_best_model = "eval_audit/flip_rate_global"
    training_args.greater_is_better = True
    
    training_args.optim = "paged_adamw_8bit"  # Memory efficient optimizer
    training_args.report_to = ["wandb"]
    training_args.run_name = f"venra-v3-r{model_args.lora_rank}-lr{model_args.learning_rate}"
    training_args.bf16 = True  # BFloat16 for Ampere GPUs (RTX 3090)
    training_args.max_grad_norm = 0.3
    
    # Memory optimizations for long context (10-K filings)
    training_args.gradient_checkpointing = True 
    training_args.group_by_length = True  # Prevent training bias in 1-epoch regime
    
    print("Initializing Sabotage Evaluation Callback...")
    sabotage_cb = SabotageEvalCallback(dataset['val'], tokenizer, token_map_ids)

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        peft_config=peft_config,
        formatting_func=format_prompt_func,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template="\nLabel:",  # Loss calculated only on completion
            tokenizer=tokenizer
        ),
        max_seq_length=4096,  # Support 10-K filing context
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[sabotage_cb]
    )

    print("\n" + "="*80)
    print("PHASE 1 TRAINING: Baseline QLoRA + rsLoRA")
    print("="*80)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"LoRA Config: r={model_args.lora_rank}, alpha={model_args.lora_alpha}, rsLoRA=True")
    print(f"Learning Rate: {model_args.learning_rate}")
    print(f"Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Target Metric: {training_args.metric_for_best_model}")
    print("="*80 + "\n")
    
    trainer.train()
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE: Saving Best Model...")
    print("="*80)
    trainer.save_model(training_args.output_dir)
    
    print("\nPHASE 2 CONTINGENCIES (Only if Phase 1 fails targets):")
    print("  1. GaLore for 7B scaling (if Precision < 90% on code_lie)")
    print("  2. Hyperparameter grid (if Recall < 85% on neighbor_trap)")
    print("  3. LRSL Logit Reranking (if systematic bias observed)")
    print("  4. Temperature Scaling (post-training ECE calibration)")
    print("\nDone.")

if __name__ == "__main__":
    main()