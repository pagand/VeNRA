import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F

from prompt_builder import build_prompt

app = FastAPI()

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL          = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_ID          = "pagand/venra"
REVISION            = "r96"         # r96 → noinstruct | r128 → full
PROMPT_TYPE         = "noinstruct"  # must match REVISION
MAX_SEQ_LEN         = 4096
DEBUG_MAX_NEW_TOKENS = 300

# ─── Load at startup ──────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print(f"Loading adapter {ADAPTER_ID} revision={REVISION}...")
model = PeftModel.from_pretrained(model, ADAPTER_ID, revision=REVISION)
model.eval()
print("Model ready.")

# ─── Label token IDs (Orthogonal Token Mapping, Section 3.1) ──────────────────
# Leading space is critical — continuation token after "Label:"
LABEL_TOKEN_MAP = {
    "supported": " Found",    # expected ID: 12315
    "unfounded": " Fake",     # expected ID: 36965
    "general":   " General",  # expected ID: 3251
}

LABEL_IDS = {
    label: tokenizer.encode(token, add_special_tokens=False)[0]
    for label, token in LABEL_TOKEN_MAP.items()
}

EXPECTED_IDS = {"supported": 12315, "unfounded": 36965, "general": 3251}
for label, expected_id in EXPECTED_IDS.items():
    actual_id = LABEL_IDS[label]
    status = "✅" if actual_id == expected_id else "⚠️  MISMATCH"
    print(f"{status} Token ID '{label}': expected {expected_id}, got {actual_id}")

LABEL_ORDER = ["supported", "unfounded", "general"]


# ─── Shared helpers ───────────────────────────────────────────────────────────
def _build_and_tokenize(data: dict):
    """Builds prompt and tokenizes. No caching — statement appears before
    context in the prompt, so there is no shared prefix to cache across
    different sentences on the same (query, context, trace)."""
    prompt = build_prompt(
        query=data.get("query", ""),
        context=data.get("context", ""),
        trace=data.get("trace", ""),
        statement=data.get("sentence", ""),
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        label_token=None,  # inference mode — no completion appended
        reasoning=None,
        prompt_type=PROMPT_TYPE,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    return prompt, inputs


def _extract_label_probs(logits_vec: torch.Tensor) -> tuple:
    """Extracts softmax probabilities over the three label tokens."""
    relevant_logits = torch.tensor([
        logits_vec[LABEL_IDS["supported"]],
        logits_vec[LABEL_IDS["unfounded"]],
        logits_vec[LABEL_IDS["general"]],
    ])
    probs = F.softmax(relevant_logits, dim=0)
    predicted_index = int(torch.argmax(probs))
    return probs, predicted_index


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/verify")
async def verify(data: dict):
    """
    FAST PATH — single forward pass, reads ONE token position, returns immediately.
    The model sees the full prompt but we only inspect logits at the final position.
    No generation loop whatsoever. Latency = one forward pass over prompt length.
    """
    _, inputs = _build_and_tokenize(data)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits[0, -1, :] = distribution over next token after "Label:"
    logits = outputs.logits[0, -1, :]
    probs, predicted_index = _extract_label_probs(logits)

    return {
        "prediction": LABEL_ORDER[predicted_index].upper(),
        "probabilities": {
            "supported": round(float(probs[0]), 6),
            "unfounded":  round(float(probs[1]), 6),
            "general":    round(float(probs[2]), 6),
        },
    }


@app.post("/debug")
async def debug(data: dict):
    """
    SLOW PATH — runs full autoregressive generation so the model writes
    its reasoning. output_scores=True gives us scores[0][0] which is the
    logit distribution at the first generated token (the label) — identical
    position to what /verify reads, so probabilities will agree.
    Latency >> /verify. CPU may take several minutes. Development use only.
    """
    prompt, inputs = _build_and_tokenize(data)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=DEBUG_MAX_NEW_TOKENS,
            do_sample=False,              # greedy — consistent with /verify
            output_scores=True,           # capture per-step logit distributions
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,  # stop early if model finishes
            pad_token_id=tokenizer.eos_token_id,  # suppress pad warning on CPU
        )

    # scores[0][0]: logits at generation step 0, batch item 0 = label token
    # This is the same position /verify reads — probabilities must agree
    first_token_logits = output.scores[0][0]
    probs, predicted_index = _extract_label_probs(first_token_logits)

    # Decode only the newly generated tokens — strip the input prompt
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "prediction": LABEL_ORDER[predicted_index].upper(),
        "probabilities": {
            "supported": round(float(probs[0]), 6),
            "unfounded":  round(float(probs[1]), 6),
            "general":    round(float(probs[2]), 6),
        },
        "analysis": generated_text,  # model's actual written reasoning
    }


@app.get("/")
def home():
    return {
        "status":      "VeNRA Hallucination Detector is Online",
        "device":      "CPU",
        "adapter":     ADAPTER_ID,
        "revision":    REVISION,
        "prompt_type": PROMPT_TYPE,
        "endpoints": {
            "POST /verify": "Fast — single forward pass, one token read, no generation",
            "POST /debug":  "Slow — full generation, returns model reasoning text",
        },
    }