import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from collections import Counter

# Load environment variables (HF_TOKEN)
load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_REPO = "pagand/venra"
DATASET_REVISION = "v2.2"
DATA_DIR = Path("data/training_final")

# Prompt template from src/hal_det/training/prompt_template.txt
PROMPT_TEMPLATE = """<|im_start|>system
You are a rigorous financial auditor. Verify if the TARGET_SENTENCE is supported by the EVIDENCE...
<|im_end|>
<|im_start|>user
### USER_QUERY:
{query}

### EVIDENCE (TEXT_CONTEXT):
{context}

### EVIDENCE (TRACE_LOGIC):
{trace}

### TARGET_SENTENCE:
{target_sentence}

### TASK:
Classify the TARGET_SENTENCE.
<|im_end|>
<|im_start|>assistant
Label: {verdict}
Analysis: {reasoning}
<|im_end|>"""

def ensure_data_exists():
    """Downloads data from Hugging Face if not found locally."""
    files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    missing = [f for f in files if not (DATA_DIR / f).exists()]
    
    if not missing:
        print("✅ Local data found in data/training_final.")
        return

    print(f"🚀 Data missing ({missing}). Downloading from {DATASET_REPO} ({DATASET_REVISION})...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset(DATASET_REPO, revision=DATASET_REVISION, use_auth_token=True)
        for split in dataset.keys():
            output_file = DATA_DIR / f"{split}.jsonl"
            print(f"  Saving {split} split to {output_file}...")
            dataset[split].to_json(output_file, orient="records", lines=True)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("Ensure you have HF_TOKEN in your .env and access to the private repo.")

def calculate_stats(file_path, tokenizer):
    token_counts = []
    input_token_counts = []
    output_token_counts = []
    
    if not file_path.exists():
        print(f"File {file_path} not found.")
        return None
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            labels.append(data['label'])
            # Extract fields
            query = data['input_components'].get('query', '')
            context = data['input_components'].get('context', [])
            if isinstance(context, list):
                context = "\n".join(context)
            trace = data['input_components'].get('trace', '')
            target_sentence = data['output_components'].get('target_sentence', '')
            verdict = data['output_components'].get('verdict', '')
            reasoning = data['output_components'].get('reasoning', '')
            
            # Format full prompt
            full_prompt = PROMPT_TEMPLATE.format(
                query=query,
                context=context,
                trace=trace,
                target_sentence=target_sentence,
                verdict=verdict,
                reasoning=reasoning
            )
            
            # Calculate tokens
            tokens = tokenizer.encode(full_prompt)
            token_counts.append(len(tokens))
            
            # Calculate input/output split
            try:
                parts = full_prompt.split("<|im_start|>assistant")
                input_part = parts[0] + "<|im_start|>assistant"
                output_part = parts[1]
                
                input_tokens = tokenizer.encode(input_part)
                output_tokens = tokenizer.encode(output_part)
                
                input_token_counts.append(len(input_tokens))
                output_token_counts.append(len(output_tokens))
            except IndexError:
                # Fallback if assistant tag is missing for some reason
                input_token_counts.append(len(tokens))
                output_token_counts.append(0)
        print("distribution:", Counter(labels))

    if not token_counts:
        return None

    return {
        "count": len(token_counts),
        "total": {
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": sum(token_counts) / len(token_counts)
        },
        "input": {
            "min": min(input_token_counts),
            "max": max(input_token_counts),
            "avg": sum(input_token_counts) / len(input_token_counts)
        },
        "output": {
            "min": min(output_token_counts),
            "max": max(output_token_counts),
            "avg": sum(output_token_counts) / len(output_token_counts)
        }
    }

def main():
    # 1. Ensure data is present
    ensure_data_exists()

    # 2. Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load tokenizer {MODEL_ID}. Error: {e}")
        return

    files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        print(f"\nAnalyzing {file_name}...")
        stats = calculate_stats(file_path, tokenizer)
        
        if stats:
            print(f"  Samples: {stats['count']}")
            print(f"  Total Token Size: Min={stats['total']['min']}, Max={stats['total']['max']}, Avg={stats['total']['avg']:.2f}")
            print(f"  Input Token Size: Min={stats['input']['min']}, Max={stats['input']['max']}, Avg={stats['input']['avg']:.2f}")
            print(f"  Output Token Size: Min={stats['output']['min']}, Max={stats['output']['max']}, Avg={stats['output']['avg']:.2f}")

if __name__ == "__main__":
    main()
