from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    device_map="auto"
)

# Load your trained adapter
model = PeftModel.from_pretrained(base_model, "./data/output")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./data/output")

# Test
prompt = "<|im_start|>system\nYou are a financial auditor.<|im_end|>\n..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))