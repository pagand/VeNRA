import yaml
from pathlib import Path
from typing import Optional

# Pre-load YAML content
YAML_PATH = Path(__file__).parent / "prompts.yaml"
with open(YAML_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

def build_prompt(
    query:    str,
    context:  str,
    trace:    str,
    statement: str,
    tokenizer,
    max_seq_length: int = 4096,
    # --- training-only fields (None → inference mode, no completion) ---
    label_token: Optional[str] = None,
    reasoning:   Optional[str] = None,
    prompt_type: str = "full"
) -> str:
    """
    Build a single prompt with smart context truncation and Selective Repetition.
    Supports "full" or "noinstruct" prompt types based on YAML configuration.
    """
    system_msg  = PROMPTS["system_msg"]
    
    # Render prefix and suffix with placeholders
    user_prefix = PROMPTS["user_prefix"].format(
        query=query,
        trace=trace,
        statement=statement
    )
    
    if prompt_type not in PROMPTS["user_suffix"]:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Expected one of {list(PROMPTS['user_suffix'].keys())}")
        
    user_suffix = PROMPTS["user_suffix"][prompt_type].format(
        query=query,
        trace=trace,
        statement=statement
    )

    if label_token is not None and reasoning is not None:
        # Training: completion is part of the sequence (Note the leading space for label_token)
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
