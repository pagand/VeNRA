import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F

app = FastAPI()

# 1. Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_ID = "DreamyPujara/fine-tuned-model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model (this may take a few minutes on CPU)...")
# On CPU, we use float32 or bfloat16. 
# bitsandbytes 4-bit usually requires a GPU.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float32, 
    device_map="cpu" 
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval()

# We need to find the Token IDs for our specific labels.
LABEL_IDS = {
    "GROUNDED": tokenizer.encode("Grounded", add_special_tokens=False)[0],
    "COMMON": tokenizer.encode("Common", add_special_tokens=False)[0],
    "HALLUCINATION": tokenizer.encode("Hallucination", add_special_tokens=False)[0],
}

@app.post("/verify")
async def verify(data: dict):
    # Constructing the white-box prompt
    prompt = f"Context: {data.get('context', '')}\nTrace: {data.get('trace', '')}\nSentence: {data.get('sentence', '')}\nLabel:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits for the last token position
    logits = outputs.logits[0, -1, :]
    
    # Extract only the probabilities for our target labels
    relevant_logits = torch.tensor([
        logits[LABEL_IDS["GROUNDED"]],
        logits[LABEL_IDS["COMMON"]],
        logits[LABEL_IDS["HALLUCINATION"]]
    ])
    
    probs = F.softmax(relevant_logits, dim=0)
    
    return {
        "grounded": float(probs[0]),
        "common": float(probs[1]),
        "hallucination": float(probs[2]),
        "prediction": ["GROUNDED", "COMMON", "HALLUCINATION"][torch.argmax(probs)]
    }

@app.get("/")
def home():
    return {"status": "VeNRA Hallucination Detector is Online", "device": "CPU"}