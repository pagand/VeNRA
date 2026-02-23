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
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from dotenv import load_dotenv


load_dotenv()

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                      os.getenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:512'))


# ---------------------------------------------------------------------------
# Configuration & Defaults
# ---------------------------------------------------------------------------
OUTPUT_DIR="./data/output"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MULTIPLIER=2 #based on GPU RAM
LEARNING_RATE=1e-4
LORA_RANK=128 # was 64
LORA_ALPHA=LORA_RANK # was 32
MAX_SEQ_LENGTH=4096
# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
#training config
NUM_TRAIN_EPOCHS=10
TRAIN_BATCH_SIZE=2*MULTIPLIER
EVAL_BATCH_SIZE=3*MULTIPLIER
GRAD_ACCUM_STEP=32//MULTIPLIER #effective 64
EVAL_ACCUM_STEP=32
EVAL_SABOTAGE_BATCH_SIZE=4*MULTIPLIER
LR_SCHEDULER_KWARGS = {"min_lr": 5e-2*LEARNING_RATE}
PENALTY=50.0
PATIENCE=15
WARMUP_RATIO=0.1
EVAL_STEPS=25 #eval and save freq
MAX_GRAD_NORM=0.3
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
    Build a single prompt with smart context truncation and Selective Repetition.

    Training mode  : pass label_token + reasoning  → full prompt + completion
    Inference mode : leave label_token=None         → prompt stops at 'Label:'

    Truncation strategy & Prompt Repetition:
    -------------------
    To overcome the 'lost-in-the-middle' causal attention bottleneck, we state the 
    Task and Target at the TOP, provide the massive Context, and REPEAT the Target 
    at the BOTTOM. Context is truncated from the END.
    """
    system_msg  = "<|im_start|>system\nYou are a rigorous financial auditor.<|im_end|>\n"
    
    # --- TOP BOOKEND ---
    # Prime the attention heads: Tell it the task and what to look for BEFORE the massive text.
    user_prefix = (
        f"<|im_start|>user\n"
        f"### TASK:\n"
        f"Verify if the CLAIMED_ANSWER to the QUERY given the AGENT_TRACE is [Found, Fake, General] based on the EVIDENCE.\n\n"
        f"### VERIFICATION TARGET:\n"
        f"**Query:**\n {query}\n"
        f"**AGENT TRACE (Methodology):**\n {trace}\n"
        f"**Claimed Answer:**\n {statement}\n\n"
        f"### EVIDENCE (Source Document):\n"
    )
    
    # --- BOTTOM BOOKEND ---
    # The 'Let me repeat' triggers cross-attention between the prompt and the deep KV cache.
    user_suffix = (
        f"\n\n### VERIFICATION TARGET Recap:\n"
        f"**Query:**\n {query}\n"
        f"**AGENT TRACE (Methodology):**\n {trace}\n"
        f"**Claimed Answer:**\n {statement}\n\n"
        f"### TASK:\n"
        f"Is the CLAIMED_ANSWER supported by the EVIDENCE? Answer with one of: Found, Fake, General.\n"
        f"### AUDIT ALGORITHM (CRITICAL):\n"
        f"Do NOT recalculate the math. Assume the arithmetic in the trace evaluates to the Claimed Answer. You must verify the EXTRACTION and LOGIC:\n"
        f"1. EXTRACTION CHECK: Are ALL specific numbers, dates, and entities used in the AGENT TRACE explicitly present in the EVIDENCE? If any number is fabricated or pulled from the wrong row/column -> Fake\n"
        f"2. LOGIC CHECK: Does the AGENT TRACE use the correct operation and the correct metrics to answer the QUERY? If it answers the wrong question or uses the wrong year -> Fake\n"
        f"3. AXIOM CHECK: If the EVIDENCE is irrelevant, but the Claimed Answer is a widely known universal fact -> General\n"
        f"4. If Extraction and Logic are both supported by the EVIDENCE -> Found\n\n"
        f"Output your label (Found, Fake, or General) first, followed by your analysis.<|im_end|>\n"
        f"<|im_start|>assistant\nLabel:"
    )

    if label_token is not None and reasoning is not None:
        # Training: completion is part of the sequence (Note the leading space for label_token)
        completion = f"{label_token}\nAnalysis: {reasoning}<|im_end|>"
    else:
        # Inference: no completion
        completion = ""

    # ---------- budget calculation ----------
    # We must account for the fact that the Target is duplicated in the budget
    essential      = system_msg + user_prefix + user_suffix + completion
    essential_toks = len(tokenizer.encode(essential, add_special_tokens=False))
    context_budget = max_seq_length - essential_toks - 20   # 20-tok safety margin

    # ---------- context truncation ----------
    if context_budget > 50:
        # We only encode the context, not the prefix "Context: " since that's moved to user_prefix
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        if len(context_tokens) > context_budget:
            context_tokens = context_tokens[:context_budget]
            context_text   = tokenizer.decode(context_tokens, skip_special_tokens=True) + "\n"
        else:
            context_text   = context + "\n"
    else:
        context_text = "[Truncated]\n"

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
    Robust Weighted Trainer with Gradient Safety Valve.
    
    Features:
    1. Configurable penalty on verdict tokens.
    2. Optional separate penalty for the General token (verdict_token_ids[2]).
       Defaults to `penalty` for full backward compatibility.
    3. Parameterized loss ceiling to prevent gradient explosions.
    4. Micro-chunking for OOM safety on large vocabs.
    """
    def __init__(self, *args, verdict_token_ids, penalty=50.0, loss_ceiling=None, chunk_size=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdict_token_ids = verdict_token_ids
        self.penalty = penalty
        self.general_penalty = 10.0 # was penalty
        self.loss_ceiling = loss_ceiling if loss_ceiling is not None else 5.0 * penalty
        self.chunk_size = chunk_size

        vocab_size = self.model.config.vocab_size
        self.token_weights = torch.ones(vocab_size)
        for tid in self.verdict_token_ids:
            if tid < vocab_size:
                self.token_weights[tid] = self.penalty

        # Override General token weight if a separate penalty was requested.
        if len(self.verdict_token_ids) >= 3:
            general_tid = self.verdict_token_ids[2]
            if general_tid < vocab_size:
                self.token_weights[general_tid] = self.general_penalty

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop labels BEFORE the forward pass so the model does not compute its own
        # internal cross-entropy allocating a full [batch, seq_len, vocab_size] tensor in one shot
        # The chunked loss below owns the loss computation entirely.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        inputs["labels"] = labels  # Restore so HF internals stay consistent

        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, self.model.config.vocab_size)
        labels_flat = shift_labels.view(-1)

        if self.token_weights.device != logits_flat.device:
            self.token_weights = self.token_weights.to(logits_flat.device)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        total_loss = torch.tensor(0.0, device=logits_flat.device)
        total_weight_sum = torch.tensor(0.0, device=logits_flat.device)

        for i in range(0, labels_flat.shape[0], self.chunk_size):
            chunk_logits = logits_flat[i:i + self.chunk_size]
            chunk_labels = labels_flat[i:i + self.chunk_size]
            valid_mask = chunk_labels != -100

            if valid_mask.any():
                raw_token_losses = loss_fct(chunk_logits, chunk_labels)

                # Safe indexing: replace -100 with 0 before weight lookup
                safe_labels = chunk_labels.clone()
                safe_labels[~valid_mask] = 0
                chunk_weights = self.token_weights[safe_labels]

                weighted_losses = raw_token_losses * chunk_weights

                # Safety valve: ceiling is tied to penalty, not hardcoded
                weighted_losses = torch.clamp(weighted_losses, max=self.loss_ceiling)

                total_loss += weighted_losses[valid_mask].sum()
                total_weight_sum += chunk_weights[valid_mask].sum()

        final_loss = total_loss / total_weight_sum.clamp(min=1e-9)
        return (final_loss, outputs) if return_outputs else final_loss

class SabotageEvalCallback(TrainerCallback):
    """
    Audit Tool (v4.0).

    Stratified evaluation covering:
      1. Paired Flip Rate — stratified by Short / Long context
      2. Natural Failure Recall — orphaned Unfounded rows (real hallucinations)
      3. False Positive Rate — orphaned Supported rows (false alarms)
      4. Axiom Accuracy — General-labelled rows (knowledge retention)

    All subsets are fixed at init time (reproducible seed) so every eval
    checkpoint is comparable.
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        token_map_ids,
        # ── Budget hyper-params ──────────────────────────────────────────────
        max_pairs_per_stratum: int = 100,   # 100 short + 100 long = 200 pair prompts
        max_natural_fake: int = 80,        # natural Unfounded orphans
        max_natural_true: int = 80,        # orphaned Supported
        max_axioms: int = 20,              # General rows
        eval_batch_size: int = EVAL_SABOTAGE_BATCH_SIZE, # batch size
        length_threshold: int = 512,       # token boundary for short vs long
        seed: int = 42,
    ):
        self.tokenizer       = tokenizer
        self.token_map_ids   = token_map_ids
        self.eval_batch_size = eval_batch_size
        self.length_threshold = length_threshold

        print("Initializing Comprehensive Audit Set (v4.0)...")

        rng = random.Random(seed)          # local — does NOT touch global state

        # ── Step 1: Build family dict (all rows have a family_id) ────────────
        families: Dict[str, list] = {}
        for i, row in enumerate(eval_dataset):
            fid = row.get("meta", {}).get("family_id")
            if fid:
                families.setdefault(fid, []).append(row)
            if i >= 5000:
                break

        # ── Step 2: Route into pools ─────────────────────────────────────────
        #
        # Family shapes:
        #   2-member (Supported + Unfounded)  → candidate pair
        #   1-member Supported                → orphaned Supported (natural_true pool)
        #   1-member Unfounded, natural       → natural_fake pool
        #   any member with label=General     → axiom pool
        #
        # sabotage_type is a TOP-LEVEL field, NOT inside meta.

        pairs_short:       List[Tuple[Any, Any]] = []
        pairs_long:        List[Tuple[Any, Any]] = []
        natural_fake_pool: List[Any]             = []
        natural_true_pool: List[Any]             = []
        axiom_pool:        List[Any]             = []

        for fid, members in sorted(families.items()):
            # Separate by label
            supported = [m for m in members if m["label"] == "Supported"]
            unfounded = [m for m in members if m["label"] == "Unfounded"]
            general   = [m for m in members if m["label"] == "General"]

            # General members always go to axiom pool (may co-exist with others)
            axiom_pool.extend(general)

            if supported and unfounded:
                # Valid pair — use first of each (families should be 1+1)
                parent = supported[0]
                child  = unfounded[0]
                t_count = parent.get("meta", {}).get("token_count", 0)
                pair = (parent, child)
                if t_count >= length_threshold:
                    pairs_long.append(pair)
                else:
                    pairs_short.append(pair)

            else:
                # Orphaned members
                for m in supported:
                    natural_true_pool.append(m)
                for m in unfounded:
                    # Only natural Unfounded orphans — top-level sabotage_type field
                    if m.get("sabotage_type", "unknown") == "natural":
                        natural_fake_pool.append(m)

        # ── Step 3: Fixed random subsets ─────────────────────────────────────
        rng.shuffle(pairs_short)
        rng.shuffle(pairs_long)
        rng.shuffle(natural_fake_pool)
        rng.shuffle(natural_true_pool)
        rng.shuffle(axiom_pool)

        self.pairs_short    = pairs_short[:max_pairs_per_stratum]
        self.pairs_long     = pairs_long[:max_pairs_per_stratum]
        nat_fake_subset     = natural_fake_pool[:max_natural_fake]
        nat_true_subset     = natural_true_pool[:max_natural_true]
        axiom_subset        = axiom_pool[:max_axioms]

        # ── Step 4: Flat eval batch (layout fixed, sliceable by meta tag) ────
        #
        # Order: [pairs_short | pairs_long | natural_fake | natural_true | axioms]
        # Child always immediately follows parent → index+1 arithmetic is safe.

        self.eval_examples: List[Any] = []
        self.ground_truth:  List[int] = []   # 0=Found/Supported, 1=Fake/Unfounded, 2=General
        self.meta_labels:   List[str] = []   # for mask-based slicing

        for p, c in self.pairs_short:
            self.eval_examples.extend([p, c])
            self.ground_truth.extend([0, 1])
            self.meta_labels.extend(["pair_short_parent", "pair_short_child"])

        for p, c in self.pairs_long:
            self.eval_examples.extend([p, c])
            self.ground_truth.extend([0, 1])
            self.meta_labels.extend(["pair_long_parent", "pair_long_child"])

        for row in nat_fake_subset:
            self.eval_examples.append(row)
            self.ground_truth.append(1)
            self.meta_labels.append("natural_fake")

        for row in nat_true_subset:
            self.eval_examples.append(row)
            self.ground_truth.append(0)
            self.meta_labels.append("natural_true")

        for row in axiom_subset:
            self.eval_examples.append(row)
            self.ground_truth.append(2)
            self.meta_labels.append("axiom")

        total_calls = len(self.eval_examples)
        print(f"  Pairs (Short  < {length_threshold} tok): {len(self.pairs_short)} pairs → {len(self.pairs_short)*2} prompts")
        print(f"  Pairs (Long  >= {length_threshold} tok): {len(self.pairs_long)} pairs → {len(self.pairs_long)*2} prompts")
        print(f"  Natural Fake (orphaned Unfounded):        {len(nat_fake_subset)} prompts")
        print(f"  Natural True (orphaned Supported):        {len(nat_true_subset)} prompts")
        print(f"  Axioms (General):                         {len(axiom_subset)} prompts")
        print(f"  ─────────────────────────────────────────")
        # Pre-build prompts once — the set is fixed for the whole training run
        print("  Pre-building prompts...")
        self.prompts = [self._make_prompt(x) for x in self.eval_examples]

        # Convert to numpy for fast boolean indexing
        self.meta_np  = np.array(self.meta_labels)
        self.truth_np = np.array(self.ground_truth)
        print("  Done. Audit set is ready.\n")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _make_prompt(self, example) -> str:
        """Inference-mode prompt — ends at 'Label:' with no completion."""
        c_raw = example["input_components"]["context"]
        return build_prompt(
            query          = example["input_components"]["query"],
            context        = "\n".join(c_raw) if isinstance(c_raw, list) else str(c_raw),
            trace          = example["input_components"]["trace"],
            statement      = example["output_components"]["target_sentence"],
            tokenizer      = self.tokenizer,
            max_seq_length = MAX_SEQ_LENGTH,
            label_token    = None,
            reasoning      = None,
        )

    # ── Inference (batched, OOM-safe) ─────────────────────────────────────────

    def _run_batched_inference(self, model) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the full eval_batch in mini-batches of self.eval_batch_size.
        Returns (preds, confs) arrays of shape [N] where preds ∈ {0, 1, 2}.

        Mirrors the original evaluate_subset_batched pattern.
        """
        relevant_ids = [
            self.token_map_ids["Supported"],   # → pred index 0
            self.token_map_ids["Unfounded"],   # → pred index 1
            self.token_map_ids["General"],     # → pred index 2
        ]

        all_preds: List[int]   = []
        all_confs: List[float] = []

        for batch_start in range(0, len(self.prompts), self.eval_batch_size):
            batch_end     = min(batch_start + self.eval_batch_size, len(self.prompts))
            batch_prompts = self.prompts[batch_start:batch_end]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = MAX_SEQ_LENGTH,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Next-token logits → 3-class distribution
            next_token_logits = outputs.logits[:, -1, :]
            relevant_logits   = next_token_logits[:, relevant_ids]
            probs             = torch.softmax(relevant_logits, dim=-1)
            preds             = torch.argmax(probs, dim=-1).cpu().numpy()
            confs             = torch.max(probs, dim=-1).values.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_confs.extend(confs.tolist())

            # Explicit cleanup — mirrors original clear pattern
            del inputs, outputs, next_token_logits, relevant_logits, probs

        return np.array(all_preds), np.array(all_confs)

    # ── Main callback ─────────────────────────────────────────────────────────

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        model.eval()
        self.clear_gpu_memory()

        preds_np, confs_np = self._run_batched_inference(model)

        # ── Helper: paired metrics for one stratum ────────────────────────────
        def compute_pair_metrics(parent_tag: str):
            """
            Returns (flip_rate, ece, parent_acc, child_acc, n_pairs).
            Child tag is inferred by replacing 'parent' → 'child'.
            """
            child_tag  = parent_tag.replace("parent", "child")
            parent_idx = np.where(self.meta_np == parent_tag)[0]
            child_idx  = np.where(self.meta_np == child_tag)[0]
            n = len(parent_idx)
            if n == 0:
                return 0.0, 0.0, 0.0, 0.0, 0

            # Flip Rate (KEPT from original): both must be correct simultaneously
            correct_flips = int(np.sum(
                (preds_np[parent_idx] == 0) & (preds_np[child_idx] == 1)
            ))
            flip_rate  = correct_flips / n
            parent_acc = float(np.mean(preds_np[parent_idx] == 0))
            child_acc  = float(np.mean(preds_np[child_idx]  == 1))

            # ECE (KEPT from original): computed on the binary Found/Fake slice
            all_idx = np.concatenate([parent_idx, child_idx])
            truths  = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
            accs    = (preds_np[all_idx] == truths).astype(float)
            ece     = float(np.abs(confs_np[all_idx] - accs).mean())

            return flip_rate, ece, parent_acc, child_acc, n

        # ── Metric A: Paired Flip Rate — Short + Long (KEPT) ─────────────────
        fr_short, ece_short, pa_short, ca_short, n_short = compute_pair_metrics("pair_short_parent")
        fr_long,  ece_long,  pa_long,  ca_long,  n_long  = compute_pair_metrics("pair_long_parent")

        def weighted(a, na, b, nb):
            denom = na + nb
            return (a * na + b * nb) / denom if denom > 0 else 0.0

        fr_global  = weighted(fr_short, n_short, fr_long, n_long)
        pa_global  = weighted(pa_short, n_short, pa_long, n_long)
        ca_global  = weighted(ca_short, n_short, ca_long, n_long)
        ece_global = weighted(ece_short, n_short, ece_long, n_long)

        # ── Metric B: Recall on Natural Fake orphans (ADDED) ─────────────────
        nat_fake_mask   = (self.meta_np == "natural_fake")
        n_nat_fake      = nat_fake_mask.sum()
        recall_nat_fake = float(np.mean(preds_np[nat_fake_mask] == 1)) if n_nat_fake > 0 else 0.0

        # ── Metric C: FPR / TPR on Natural True orphans (ADDED) ──────────────
        nat_true_mask = (self.meta_np == "natural_true")
        n_nat_true    = nat_true_mask.sum()
        fpr_nat_true  = float(np.mean(preds_np[nat_true_mask] == 1)) if n_nat_true > 0 else 0.0
        tpr_nat_true  = float(np.mean(preds_np[nat_true_mask] == 0)) if n_nat_true > 0 else 0.0

        # ── Metric D: Axiom Accuracy (ADDED) ─────────────────────────────────
        axiom_mask = (self.meta_np == "axiom")
        n_axioms   = axiom_mask.sum()
        acc_axiom  = float(np.mean(preds_np[axiom_mask] == 2)) if n_axioms > 0 else 0.0

        # compute combined counts for FR (you already do weighted mean; recompute successes)
        s_fr = fr_short * n_short + fr_long * n_long
        n_fr = n_short + n_long
        # posterior-mean (Jeffreys prior) using a near-zero prior (alpha tiny, beta = 1).
        def posterior_mean(m_times_n, n, prior_a=1e-6, prior_b=1.0):
            if n <= 0:
                return 0.0
            # allow fractional successes: s = m * n
            s = m_times_n
            return (prior_a + s) / (prior_a + prior_b + n)

        # get posterior means for each pillar
        pm_fr    = posterior_mean(s_fr, n_fr)
        pm_recall= posterior_mean(recall_nat_fake * n_nat_fake, n_nat_fake)
        pm_tpr   = posterior_mean(tpr_nat_true * n_nat_true, n_nat_true)
        pm_axiom = posterior_mean(acc_axiom * n_axioms, n_axioms)

        # final composite (same sqrt-product punishment)
        # Multiplication across four capability pillars.
        # Penalizes FPR directly: high FPR → lower tpr_nat_true → lower score.
        composite = (pm_fr**0.5) * (pm_recall**0.5) * (pm_tpr**0.5) * (pm_axiom**0.5)

        # ── Print ─────────────────────────────────────────────────────────────
        print(f"\n[Comprehensive Audit] Step {state.global_step}")
        print(f"  ── Paired Flip Rate ─────────────────────────────────────────")
        print(f"    Short (<  {self.length_threshold} tok): {fr_short:.2%}  "
              f"(ECE={ece_short:.4f}, Found={pa_short:.2%}, Fake={ca_short:.2%}, n={n_short})")
        print(f"    Long  (>= {self.length_threshold} tok): {fr_long:.2%}  "
              f"(ECE={ece_long:.4f}, Found={pa_long:.2%}, Fake={ca_long:.2%}, n={n_long})")
        print(f"    GLOBAL: {fr_global:.2%}  "
              f"(ECE={ece_global:.4f}, Found acc={pa_global:.2%}, Fake acc={ca_global:.2%})")
        print(f"  ── Natural Distribution ({n_nat_fake} fake / {n_nat_true} true) ──────────")
        print(f"    Recall   (Nat Fake / real hallucinations): {recall_nat_fake:.2%}  ↑ higher is better")
        print(f"    TPR      (Nat True / valid docs correct):  {tpr_nat_true:.2%}  ↑ higher is better")
        print(f"    FPR      (Nat True / false alarms):        {fpr_nat_true:.2%}  ↓ lower is better")
        print(f"  ── General Knowledge (n={n_axioms}) ──────────────────────────────")
        print(f"    Axiom Accuracy: {acc_axiom:.2%}")
        print(f"  ── Composite Score: {composite:.4f}  (range [0, 1]) ─────────────")

        self.clear_gpu_memory()

        # ── Build log dict ────────────────────────────────────────────────────
        log_dict = {
            # ── KEPT from old code ──
            "eval_audit/flip_rate_global":    fr_global,
            "eval_audit/flip_rate_short":     fr_short,
            "eval_audit/flip_rate_long":      fr_long,
            "eval_audit/ece_global":          ece_global,
            "eval_audit/ece_short":           ece_short,
            "eval_audit/ece_long":            ece_long,
            "eval_audit/accuracy_found":      pa_global,    # was parent_acc_global
            "eval_audit/accuracy_fake":       ca_global,    # was child_acc_global
            # ── ADDED ──
            "eval_audit/recall_natural_fake": recall_nat_fake,
            "eval_audit/fpr_natural_true":    fpr_nat_true,
            "eval_audit/tpr_natural_true":    tpr_nat_true,
            "eval_audit/axiom_accuracy":      acc_axiom,
            "eval_audit/composite_score":     composite,
        }

        if wandb.run is not None:
            # Metric E: Confusion Matrix (WandB Native Plot)
            class_names = ["Found", "Fake", "General"]
            confusion_matrix = {
                "eval_audit/conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=self.truth_np,
                    preds=preds_np,
                    class_names=class_names
                )}
            wandb.log(log_dict|confusion_matrix, step=state.global_step + 1)
            print(f"✓ Logged {len(log_dict)} metrics to wandb at step {state.global_step + 1}")

        if metrics is not None:
            metrics.update(log_dict)
            metrics["global_step"] = state.global_step + 1

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
    
    print(f"Loading dataset pagand/venra (v2.3)...")
    dataset = load_dataset("pagand/venra", revision="v2.3")
    
    print(f"Initializing tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    if DEBUG:
        print("\n" + "="*80)
        print("DEBUG MODE: Prompt Preview (Train + Validation)")
        print("="*80)

        # ---- TRAIN SAMPLE ----
        train_sample = dataset["train"][0]
        train_prompt = format_prompt_func(
            {k: [v] for k, v in train_sample.items()},
            tokenizer,
            MAX_SEQ_LENGTH
        )[0]

        print("\n----- TRAIN SAMPLE PROMPT -----\n")
        print(train_prompt)
        print("\nToken count:", len(tokenizer.encode(train_prompt)))
        print("-"*80)

        # ---- VALIDATION SAMPLE ----
        val_sample = dataset["validation"][0]
        val_prompt = format_prompt_func(
            {k: [v] for k, v in val_sample.items()},
            tokenizer,
            MAX_SEQ_LENGTH
        )[0]

        print("\n----- VALIDATION SAMPLE PROMPT -----\n")
        print(val_prompt)
        print("\nToken count:", len(tokenizer.encode(val_prompt)))
        print("="*80 + "\n")
    
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

    if not training_args.run_name:
        training_args.run_name = (
            f"venra-weighted-p{PENALTY}-r{model_args.lora_rank}-a{model_args.lora_rank}"
            f"-lr{training_args.learning_rate}"
            f"-w{training_args.warmup_ratio}"
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
        training_args.warmup_ratio = WARMUP_RATIO 
    training_args.lr_scheduler_kwargs =  LR_SCHEDULER_KWARGS
    training_args.logging_steps = 10 # Track learning rate
    
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = EVAL_STEPS
    training_args.save_strategy = "steps"
    training_args.save_steps = EVAL_STEPS
    training_args.load_best_model_at_end = True
    
    # Model selection based on Paired Flip Rate
    training_args.metric_for_best_model = "eval_audit/composite_score"
    training_args.greater_is_better = True
    
    training_args.optim = "paged_adamw_8bit"
    training_args.report_to = ["wandb"]  
    training_args.bf16 = True
    training_args.max_grad_norm = MAX_GRAD_NORM
    
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
        early_stopping_patience=PATIENCE,  # Stop if no improvement for 10 evals
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
        verdict_token_ids=list(token_map_ids.values()),  # [12315, 36965, 3251]
        penalty=PENALTY
    )

    if DEBUG:
        print("\n" + "="*80)
        print("DEBUG MODE: Collator + Loss Mask Inspection")
        print("="*80)

        sample = dataset["train"][0]
        formatted = format_prompt_func(
            {k: [v] for k, v in sample.items()},
            tokenizer,
            MAX_SEQ_LENGTH
        )[0]

        tokens = tokenizer(formatted, return_tensors="pt")
        batch = collator([{
            "input_ids": tokens["input_ids"][0].tolist(),
            "attention_mask": tokens["attention_mask"][0].tolist(),
        }])

        labels = batch["labels"][0]
        active_tokens = (labels != -100).sum().item()

        print(f"Total tokens: {labels.shape[0]}")
        print(f"Tokens contributing to loss: {active_tokens}")
        print("Expected: identified \"\\nLabel:\" as the cutoff. and masked out rest of prompt tokens.")
        print("="*80 + "\n")

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