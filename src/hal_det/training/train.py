"""
VeNRA Hallucination Judge Training Pipeline (v3.0)
===================================================
Phase 1 Implementation: Baseline QLoRA + rsLoRA Training

DEFERRED TO PHASE 2:
- GaLore scaling to 7B (Section 7.1) - Only if 3B fails precision targetsd
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
from transformers import EarlyStoppingCallback


load_dotenv()

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                      os.getenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:512'))


# ---------------------------------------------------------------------------
# Configuration & Defaults
# ---------------------------------------------------------------------------
OUTPUT_DIR="./data/output"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MULTIPLIER=1 #based on GPU RAM
LEARNING_RATE=2e-4
LORA_RANK=96 # was 64
LORA_ALPHA=LORA_RANK # was 32
MAX_SEQ_LENGTH=4096
# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
#training config
NUM_TRAIN_EPOCHS=5
TRAIN_BATCH_SIZE=2*MULTIPLIER
EVAL_BATCH_SIZE=3*MULTIPLIER
GRAD_ACCUM_STEP=32//MULTIPLIER #effective 64
EVAL_ACCUM_STEP=32
EVAL_SABOTAGE_BATCH_SIZE=4*MULTIPLIER
LR_SCHEDULER_KWARGS = {"min_lr": 1e-5}
PENALTY=10
RETRAIN=False
DEBUG=False

# Orthogonal Token Mapping (Spec Section 3.1)
TOKEN_MAP = {
    "Supported": " Found",   # ID: 12315
    "Unfounded": " Fake",    # ID: 36965
    "General":   " General"  # ID: 3251
}

# Sabotage Type Categories (Spec Section 6.2)
LOGIC_SABOTAGE_TYPES = {"logic_code_lie", "numeric_neighbor_trap", "irrelevancy_rag", "semantic_drift"}
NATURAL_FAILURE_TYPE = "natural"

def build_prompt(
    query:    str,
    context:  str,
    trace:    str,
    statement: str,
    tokenizer,
    max_seq_length: int = MAX_SEQ_LENGTH,
    # --- training-only fields (None → inference mode, no completion) ---
    label_token: Optional[str] = None,
    reasoning:   Optional[str] = None,
) -> str:
    """
    Build a single prompt with smart context truncation.

    Training mode  : pass label_token + reasoning  → full prompt + completion
    Inference mode : leave label_token=None         → prompt stops at 'Label:'
                     (model generates the next token)

    Truncation strategy
    -------------------
    All fields except `context` are treated as ESSENTIAL and never truncated.
    `context` is truncated from the END (the beginning is usually more
    relevant for financial docs).
    """
    system_msg  = "<|im_start|>system\nYou are a financial auditor.<|im_end|>\n"
    user_prefix = f"<|im_start|>user\nQuery: {query}\n"
    user_suffix = (
        f"Trace: {trace}\nStatement: {statement}\n"
        f"Task: Classify [Found, Fake, General].<|im_end|>\n"
        f"<|im_start|>assistant\nLabel:"
    )

    if label_token is not None and reasoning is not None:
        # Training: completion is part of the sequence
        completion = f"{label_token}\nAnalysis: {reasoning}<|im_end|>"
    else:
        # Inference: no completion
        completion = ""

    # ---------- budget calculation ----------
    essential      = system_msg + user_prefix + user_suffix + completion
    essential_toks = len(tokenizer.encode(essential, add_special_tokens=False))
    context_budget = max_seq_length - essential_toks - 20   # 20-tok safety margin

    # ---------- context truncation ----------
    if context_budget > 50:
        context_text   = f"Context: {context}\n"
        context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
        if len(context_tokens) > context_budget:
            context_tokens = context_tokens[:context_budget]
            context_text   = tokenizer.decode(context_tokens, skip_special_tokens=True) + "\n"
    else:
        context_text = "Context: [Truncated]\n"

    return system_msg + user_prefix + context_text + user_suffix + completion

def format_prompt_func(example, tokenizer, max_seq_length=MAX_SEQ_LENGTH):
    """Format with smart context truncation."""
    output_texts = []
    batch_size = len(example['split'])
    
    for i in range(batch_size):
        raw_label = example['label'][i]
        label_token = TOKEN_MAP.get(raw_label)
        if not label_token:
            continue
        
        query     = example['input_components'][i]['query']
        c_raw     = example['input_components'][i]['context']
        context   = "\n".join(c_raw) if isinstance(c_raw, list) else str(c_raw)
        trace     = example['input_components'][i]['trace']
        statement = example['output_components'][i]['target_sentence']
        reasoning = example['output_components'][i]['reasoning']

        text = build_prompt(
            query=query,
            context=context,
            trace=trace,
            statement=statement,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_token=label_token,
            reasoning=reasoning,
        )
        output_texts.append(text)
        
    
    return output_texts

@dataclass
class ModelArguments:
    """Model configuration arguments (no conflicts with TrainingArguments)."""
    model_name_or_path: str = field(
        default=DEFAULT_MODEL_ID,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_rank: int = field(
        default=LORA_RANK,
        metadata={"help": "LoRA Rank (r). Spec default: 64"}
    )
    lora_alpha: int = field(
        default=LORA_ALPHA,
        metadata={"help": "LoRA Alpha. Spec default: 32 (rsLoRA)"}
    )
    # NOTE: learning_rate moved to TrainingArguments to avoid conflict

def check_env_vars():
    """Load and validate environment variables from .env file."""
    
    if "HF_TOKEN" not in os.environ:
        raise ValueError(
            "CRITICAL: HF_TOKEN not found in environment variables.\n"
            "Please add HF_TOKEN=hf_xxx to your .env file."
        )
    
    # Set HF token for model downloads
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    
    if "WANDB_API_KEY" not in os.environ:
        print("Warning: WANDB_API_KEY not found. Logging will be local only.")
        # Disable wandb if no API key
        os.environ["WANDB_MODE"] = "offline"
    else:
        # Set WandB env vars for trainer integration
        if "WANDB_ENTITY" in os.environ:
            os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
        if "WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
        if "WANDB_DIR" in os.environ:
            os.makedirs(os.getenv("WANDB_DIR"), exist_ok=True)

class WeightedLabelTrainer(SFTTrainer):
    """
    Applies loss weight on verdict tokens using PyTorch's native
    class weighting mechanism.
    """
    def __init__(self, *args, verdict_token_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdict_token_ids = verdict_token_ids
        vocab_size = self.model.config.vocab_size
        self.token_weights = torch.ones(vocab_size)
        for tid in self.verdict_token_ids:
            if tid < vocab_size:
                self.token_weights[tid] = PENALTY 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, self.model.config.vocab_size)
        labels_flat = shift_labels.view(-1)

        if self.token_weights.device != logits_flat.device:
            self.token_weights = self.token_weights.to(logits_flat.device)

        chunk_size = 512
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.token_weights, reduction='sum')

        total_loss = torch.tensor(0.0, device=logits_flat.device)
        total_weight_sum = torch.tensor(0.0, device=logits_flat.device)

        for i in range(0, labels_flat.shape[0], chunk_size):
            chunk_logits = logits_flat[i:i + chunk_size]
            chunk_labels = labels_flat[i:i + chunk_size]

            valid_mask = chunk_labels != -100
            if valid_mask.any():
                # Compute sum of weighted losses for this chunk
                chunk_loss = loss_fct(chunk_logits, chunk_labels)
                total_loss += chunk_loss

                # Track the denominator (sum of weights for valid tokens)
                valid_labels = chunk_labels[valid_mask]
                total_weight_sum += self.token_weights[valid_labels].sum()

        # Final weighted mean
        final_loss = total_loss / total_weight_sum.clamp(min=1e-9)
        return (final_loss, outputs) if return_outputs else final_loss

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
        
        families: Dict[str, list] = {}
        for i, row in enumerate(eval_dataset):
            # Safe access to metadata
            meta = row.get('meta', {})
            fid = meta.get('family_id')
            if fid:
                if fid not in families: 
                    families[fid] = []
                families[fid].append(row)
            if i > 5000: 
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
                sabotage_type = child.get('sabotage_type', 'unknown')
                
                pair = (parent, child)
                
                # Stratify by BOTH dimensions
                is_long = t_count >= 512
                is_logic = sabotage_type in LOGIC_SABOTAGE_TYPES
                
                if is_long and is_logic:
                    self.pairs_long_logic.append(pair)
                elif is_long and not is_logic:
                    self.pairs_long_natural.append(pair)
                elif not is_long and is_logic:
                    self.pairs_short_logic.append(pair)
                else:  # short + natural
                    self.pairs_short_natural.append(pair)
        
        # Limit size for speed (50 pairs cap per category)
        self.pairs_short_natural = self.pairs_short_natural[:150]
        self.pairs_short_logic = self.pairs_short_logic[:150]
        self.pairs_long_natural = self.pairs_long_natural[:150]
        self.pairs_long_logic = self.pairs_long_logic[:150]
        
        print(f"Sabotage Callback: Stratified Pair Counts:")
        print(f"  Short + Natural: {len(self.pairs_short_natural)}")
        print(f"  Short + Logic:   {len(self.pairs_short_logic)}")
        print(f"  Long + Natural:  {len(self.pairs_long_natural)}")
        print(f"  Long + Logic:    {len(self.pairs_long_logic)}")

    def clear_gpu_memory(self):
        """Aggressively clear GPU cache."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _make_prompt(self, example) -> str:
        """
        Inference-mode prompt.
        Calls build_prompt with label_token=None so the sequence ends at 'Label:'
        and uses the SAME truncation logic as training.
        """
        c_raw = example['input_components']['context']
        return build_prompt(
            query     = example['input_components']['query'],
            context   = "\n".join(c_raw) if isinstance(c_raw, list) else str(c_raw),
            trace     = example['input_components']['trace'],
            statement = example['output_components']['target_sentence'],
            tokenizer = self.tokenizer,
            max_seq_length = MAX_SEQ_LENGTH,
            label_token = None,   # ← inference mode: no completion
            reasoning   = None,
        )

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        model.eval()        
        # Helper to run eval on a specific list of pairs
        def evaluate_subset(pairs):
            if not pairs: 
                return 0.0, 0.0, 0.0, 0.0
            
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
                truncation=True, # handled by make_prompt
                max_length=MAX_SEQ_LENGTH
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            next_token_logits = outputs.logits[:, -1, :]

            if DEBUG:
                # Check what the model ACTUALLY wants to predict
                print("********************")
                top5 = torch.topk(next_token_logits[0], 5)
                print("Model's top 5 predictions for first example:")
                for val, idx in zip(top5.values, top5.indices):
                    token_str = self.tokenizer.decode([idx.item()])
                    print(f"  Token: '{token_str}' (ID: {idx.item()}) Score: {val.item():.3f}")
                print(prompts[:100])
                print(f"  'Found' score: {next_token_logits[0, self.token_map_ids['Supported']].item():.3f}")
                print(f"  'Fake'  score: {next_token_logits[0, self.token_map_ids['Unfounded']].item():.3f}")

            relevant_ids = [
                self.token_map_ids["Supported"], 
                self.token_map_ids["Unfounded"]
            ]
            probs = torch.softmax(next_token_logits[:, relevant_ids], dim=-1)
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()
            
            # Flip Rate (Paired Sensitivity)
            correct_flips = 0
            for i in range(len(pairs)):
                parent_correct = predictions[i*2] == 0
                child_correct = predictions[i*2 + 1] == 1
                if parent_correct and child_correct:
                    correct_flips += 1
            flip_rate = correct_flips / len(pairs) if len(pairs) > 0 else 0.0
            
            # ECE (Expected Calibration Error)
            accuracies = (predictions == np.array(ece_labels)).astype(float)
            ece = np.abs(confidences - accuracies).mean()
            # Track per-label accuracy
            parent_acc = (predictions[::2] == 0).mean()  # "Found" accuracy
            child_acc = (predictions[1::2] == 1).mean()  # "Fake" accuracy
            del inputs, outputs, next_token_logits, probs
            return flip_rate, ece, parent_acc, child_acc

        def evaluate_subset_batched(pairs, batch_size=EVAL_SABOTAGE_BATCH_SIZE):
            """Process pairs in mini-batches if list is too large."""
            self.clear_gpu_memory()
            if len(pairs) <= batch_size:
                # Small enough - process all at once
                return evaluate_subset(pairs)
            
            # Too large - split into batches
            all_flip_rates = []
            all_eces = []
            all_parent_accs = [] 
            all_child_accs = []   
            all_pair_counts = []
            
            for batch_start in range(0, len(pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs))
                batch_pairs = pairs[batch_start:batch_end]
                
                fr, ece, parent_acc, child_acc = evaluate_subset(batch_pairs)  # FIX: Unpack 4 values
                
                all_flip_rates.append(fr)
                all_eces.append(ece)
                all_parent_accs.append(parent_acc)
                all_child_accs.append(child_acc)
                all_pair_counts.append(len(batch_pairs))
            
            # Weighted average across batches
            total_pairs = sum(all_pair_counts)
            avg_flip_rate = sum(fr * count for fr, count in zip(all_flip_rates, all_pair_counts)) / total_pairs
            avg_ece = sum(ece * count for ece, count in zip(all_eces, all_pair_counts)) / total_pairs
            avg_parent_acc = sum(pa * count for pa, count in zip(all_parent_accs, all_pair_counts)) / total_pairs  # NEW
            avg_child_acc = sum(ca * count for ca, count in zip(all_child_accs, all_pair_counts)) / total_pairs    # NEW
            
            return avg_flip_rate, avg_ece, avg_parent_acc, avg_child_acc

        # Run Stratified Audit (2x2 grid)
        fr_short_nat, ece_short_nat, pa_short_nat, ca_short_nat = evaluate_subset_batched(self.pairs_short_natural)
        fr_short_log, ece_short_log, pa_short_log, ca_short_log = evaluate_subset_batched(self.pairs_short_logic)
        fr_long_nat, ece_long_nat, pa_long_nat, ca_long_nat = evaluate_subset_batched(self.pairs_long_natural)
        fr_long_log, ece_long_log, pa_long_log, ca_long_log = evaluate_subset_batched(self.pairs_long_logic)

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

        # Calculate global accuracies (weighted average)
        total_all = total_short + total_long
        if total_all > 0:
            parent_acc_global = (
                pa_short_nat * len(self.pairs_short_natural) +
                pa_short_log * len(self.pairs_short_logic) +
                pa_long_nat * len(self.pairs_long_natural) +
                pa_long_log * len(self.pairs_long_logic)
            ) / total_all
            
            child_acc_global = (
                ca_short_nat * len(self.pairs_short_natural) +
                ca_short_log * len(self.pairs_short_logic) +
                ca_long_nat * len(self.pairs_long_natural) +
                ca_long_log * len(self.pairs_long_logic)
            ) / total_all
        else:
            parent_acc_global = 0.0
            child_acc_global = 0.0
        
        # Global metric (for model selection)
        if total_all > 0:
            fr_global = (fr_short * total_short + fr_long * total_long) / total_all
        else:
            fr_global = 0.0


        print(f"\n[Sabotage Audit] Step {state.global_step}:")
        print(f"  Short Context (< 512 tok): Overall={fr_short:.2%}")
        print(f"    ├─ Natural Failures: {fr_short_nat:.2%} (ECE={ece_short_nat:.4f})")
        print(f"    └─ Logic Sabotage:   {fr_short_log:.2%} (ECE={ece_short_log:.4f})")
        print(f"  Long Context (>= 512 tok): Overall={fr_long:.2%}")
        print(f"    ├─ Natural Failures: {fr_long_nat:.2%} (ECE={ece_long_nat:.4f})")
        print(f"    └─ Logic Sabotage:   {fr_long_log:.2%} (ECE={ece_long_log:.4f})")
        print(f"  By Sabotage Type:")
        print(f"    ├─ Natural: {fr_natural:.2%}")
        print(f"    └─ Logic:   {fr_logic:.2%}")
        print(f"  GLOBAL FLIP RATE: {fr_global:.2%}")
        print(f"  Accuracy on FOUND samples: {parent_acc_global:.2%}")
        print(f"  Accuracy on FAKE samples: {child_acc_global:.2%}")

        self.clear_gpu_memory()
        if wandb.run is not None:
            wandb.log({
                    "eval_audit/flip_rate_global": fr_global,
                    "eval_audit/flip_rate_short": fr_short,
                    "eval_audit/flip_rate_long": fr_long,
                    "eval_audit/flip_rate_natural": fr_natural,
                    "eval_audit/flip_rate_logic": fr_logic,
                    "eval_audit/flip_rate_short_natural": fr_short_nat,
                    "eval_audit/flip_rate_short_logic": fr_short_log,
                    "eval_audit/flip_rate_long_natural": fr_long_nat,
                    "eval_audit/flip_rate_long_logic": fr_long_log,
                    "eval_audit/ece_short_natural": ece_short_nat,
                    "eval_audit/ece_short_logic": ece_short_log,
                    "eval_audit/ece_long_natural": ece_long_nat,
                    "eval_audit/ece_long_logic": ece_long_log,
                    "eval_audit/accuracy_found": parent_acc_global,
                    "eval_audit/accuracy_fake": child_acc_global,
                }, step=state.global_step+1)
            print(f"✓ Logged metrics to wandb at step {state.global_step+1}")
        if metrics is not None:
            metrics.update( {
                "eval_audit/flip_rate_global": fr_global,
                "eval_audit/flip_rate_short": fr_short,
                "eval_audit/flip_rate_long": fr_long,
                "eval_audit/flip_rate_natural": fr_natural,
                "eval_audit/flip_rate_logic": fr_logic,
                "eval_audit/flip_rate_short_natural": fr_short_nat,
                "eval_audit/flip_rate_short_logic": fr_short_log,
                "eval_audit/flip_rate_long_natural": fr_long_nat,
                "eval_audit/flip_rate_long_logic": fr_long_log,
                "eval_audit/ece_short_natural": ece_short_nat,
                "eval_audit/ece_short_logic": ece_short_log,
                "eval_audit/ece_long_natural": ece_long_nat,
                "eval_audit/ece_long_logic": ece_long_log,
                "eval_audit/accuracy_found": parent_acc_global,
                "eval_audit/accuracy_fake": child_acc_global,
                "global_step": state.global_step+1
                })


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON config file
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse from command line
        model_args, training_args = parser.parse_args_into_dataclasses()
    
    # Check environment variables
    check_env_vars()
    
    print(f"Loading dataset pagand/venra (v2.2)...")
    dataset = load_dataset("pagand/venra", revision="v2.2")
    
    print(f"Initializing tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    # Verify Orthogonal Token Mapping
    print("Verifying Orthogonal Token Mapping...")
    token_map_ids = {}
    for k, v in TOKEN_MAP.items():
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) != 1: 
            raise ValueError(
                f"CRITICAL: Token '{v}' for label '{k}' is fragmented into {len(ids)} tokens!\n"
                f"Token IDs: {ids}\n"
                f"This violates the Orthogonal Label requirement."
            )
        token_map_ids[k] = ids[0]
        print(f"  {k:12s} -> '{v}' (ID: {ids[0]})")
    
    # 4-bit NF4 Quantization
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
        use_cache=False,  # Required for gradient checkpointing
        token=os.environ.get("HF_TOKEN")
    )
    model = prepare_model_for_kbit_training(model)

    # rsLoRA Configuration
    print(f"Configuring rsLoRA (r={model_args.lora_rank}, alpha={model_args.lora_alpha})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.05,
        use_rslora=True,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"      # MLP (Reasoning)
        ]
    )

    # Set default training arguments if not provided
    if training_args.output_dir in ("tmp", None):  # Default value from TrainingArguments
        training_args.output_dir = OUTPUT_DIR
    
    # Training configuration
    print("Configuring Training Arguments...")
    training_args.num_train_epochs = NUM_TRAIN_EPOCHS
    training_args.per_device_train_batch_size = TRAIN_BATCH_SIZE
    training_args.per_device_eval_batch_size = EVAL_BATCH_SIZE
    training_args.gradient_accumulation_steps = GRAD_ACCUM_STEP
    training_args.eval_accumulation_steps = EVAL_ACCUM_STEP
    if training_args.learning_rate == 5e-5:   # HF default → user didn't set it
        training_args.learning_rate = LEARNING_RATE
    
    training_args.lr_scheduler_type = "cosine_with_min_lr"
    if training_args.warmup_ratio == 0.0:     # HF default → user didn't set it
        training_args.warmup_ratio = 0.1  #was 0.03
    training_args.lr_scheduler_kwargs =  LR_SCHEDULER_KWARGS
    training_args.logging_steps = 10 # Track learning rate
    
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 25 
    training_args.save_strategy = "steps"
    training_args.save_steps = 25
    training_args.load_best_model_at_end = True
    
    # Model selection based on Paired Flip Rate
    training_args.metric_for_best_model = "eval_audit/flip_rate_global"
    training_args.greater_is_better = True
    
    training_args.optim = "paged_adamw_8bit"
    training_args.report_to = ["wandb"]
    if not training_args.run_name:
        training_args.run_name = (
            f"venra-weighted-p{PENALTY}-r{model_args.lora_rank}-a{model_args.lora_rank}"
            f"-lr{training_args.learning_rate}"
            f"-w{training_args.warmup_ratio}"
        )    
    training_args.bf16 = True
    training_args.max_grad_norm = 0.3
    
    # Memory optimizations
    training_args.gradient_checkpointing = True 
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.group_by_length = True
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    print("Initializing Sabotage Evaluation Callback...")
    sabotage_cb = SabotageEvalCallback(dataset["validation"], tokenizer, token_map_ids)

    callbacks=[
    sabotage_cb,
    EarlyStoppingCallback(
        early_stopping_patience=10,  # Stop if no improvement for 10 evals
        early_stopping_threshold=0.005  # Min improvement = 0.5%
    )
    ]
    print("Initializing SFTTrainer...")
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\nLabel:", 
        tokenizer=tokenizer
    )
    trainer = WeightedLabelTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=lambda ex: format_prompt_func(ex, tokenizer, MAX_SEQ_LENGTH),
        data_collator=collator,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        verdict_token_ids=list(token_map_ids.values())  # [12315, 36965, 3251]
    )

    if DEBUG:
        sample = dataset['train'][0]
        formatted = format_prompt_func(
            {k: [v] for k, v in sample.items()}, 
            tokenizer, 
            MAX_SEQ_LENGTH
        )[0]

        tokens = tokenizer(formatted, return_tensors='pt')
        input_ids = tokens['input_ids'][0]

        # Find where "\nLabel:" appears
        label_token_ids = tokenizer.encode("\nLabel:", add_special_tokens=False)
        print(f"Looking for token IDs: {label_token_ids}")

        # Check if collator finds it
        batch = collator([{'input_ids': input_ids.tolist(), 
                        'attention_mask': [1]*len(input_ids)}])
        labels = batch['labels'][0]

        # Count non-masked tokens (-100 = masked)
        active = (labels != -100).sum().item()
        print(f"Tokens with loss computed: {active}")
        # If this is 0 or very small: YOUR COLLATOR IS BROKEN
        # Should be ~115 tokens (the completion part)

    print("\n" + "="*80)
    print("PHASE 1 TRAINING: Baseline QLoRA + rsLoRA")
    print("="*80)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"LoRA Config: r={model_args.lora_rank}, alpha={model_args.lora_alpha}, rsLoRA=True")
    print(f"Learning Rate: {training_args.learning_rate}")
    print(f"Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Output Dir: {training_args.output_dir}")
    print(f"Target Metric: {training_args.metric_for_best_model}")
    print("="*80 + "\n")


    # Run evaluation BEFORE training
    print("\n" + "="*80)
    print("PRE-TRAINING EVALUATION (Baseline)")
    print("="*80)
    eval_results = trainer.evaluate()
    print(f"Baseline metrics: {eval_results}")
    print("="*80 + "\n")

    checkpoint_dir = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints and RETRAIN:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(training_args.output_dir, latest)
            print(f"Resuming from {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            print("Training from beginning")
            trainer.train()
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE: Saving Best Model...")
    print("="*80)
    
    # Save the final model
    trainer.save_model(training_args.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)

    print(model_args) 
    print(training_args)
    
    print(f"\n✅ Model saved to: {training_args.output_dir}")
    print("\nPHASE 2 CONTINGENCIES (Only if Phase 1 fails targets):")
    print("  1. GaLore for 7B scaling (if Precision < 90% on code_lie)")
    print("  2. Hyperparameter grid (if Recall < 85% on neighbor_trap)")
    print("  3. LRSL Logit Reranking (if systematic bias observed)")
    print("  4. Temperature Scaling (post-training ECE calibration)")
    print("\nDone.")

if __name__ == "__main__":
    main()