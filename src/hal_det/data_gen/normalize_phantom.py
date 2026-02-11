import json
import uuid
import pandas as pd
from datasets import load_dataset
from typing import Dict, Any, Optional

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_REPO = "seyled/Phantom_Hallucination_Detection"

# Corrected file paths based on 'data_names' list in reference code
# Note: The reference code uses lowercase '10k_seed' and 'def14a_seed'
TARGET_FILES = {
    "10k": "PhantomDataset/Phantom_10k_seed.csv",
    "def14a": "PhantomDataset/Phantom_def14a_seed.csv"
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count (chars / 4) to help downstream filtering
    without loading a heavy tokenizer here.
    """
    if not text: return 0
    return len(str(text)) // 4

# ==========================================
# MAIN CLASS
# ==========================================

class PhantomNormalizer:
    def __init__(self, output_path="data/golden_records/phantom_normalized.jsonl"):
        self.output_path = output_path

    def process(self):
        normalized_data = []
        
        print(f"Dataset Repo: {DATASET_REPO}")

        for doc_type, file_path in TARGET_FILES.items():
            print(f"Loading {doc_type} data from {file_path}...")
            try:
                # Load specific CSV file from the HF repo
                # The reference code confirms usage of data_files={"train": path}
                dataset = load_dataset(DATASET_REPO, data_files=file_path, split='train')
                
                print(f"Processing {len(dataset)} rows from {doc_type}...")
                for row in dataset:
                    entry = self._normalize_row(row, doc_type)
                    if entry:
                        normalized_data.append(entry)
            except Exception as e:
                print(f"⚠️ Error loading {doc_type} ({file_path}): {e}")

        print(f"Total normalized rows: {len(normalized_data)}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for entry in normalized_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved to {self.output_path}")

    def _normalize_row(self, row: Dict, doc_type: str) -> Optional[Dict]:
        """
        Maps Phantom dataset rows to VeNRA Schema.
        Columns based on reference code: 'query', 'context', 'answer', 'label'
        """
        # 1. Extract Fields (Robust retrieval)
        # Reference code uses lowercase: query, context, answer
        query = row.get('query') or row.get('Question')
        context = row.get('context') or row.get('Context')
        answer = row.get('answer') or row.get('Answer')
        
        # Label handling: 'hallucination' vs 'not hallucination'
        raw_label = str(row.get('label') or row.get('Label') or row.get('ground_truth_label') or '').lower().strip()
        
        if not (query and context and answer and raw_label):
            return None

        # 2. Map Labels to VeNRA
        # Phantom Logic: 'not hallucination' -> Supported, 'hallucination' -> Unfounded
        if "not hallucination" in raw_label:
            venra_label = "Supported"
            is_sabotage_ready = True  # These are valid facts we can swap to create attacks
        elif "hallucination" in raw_label:
            venra_label = "Unfounded"
            is_sabotage_ready = False # These are already broken; do not sabotage further
        else:
            # Skip unclear rows
            return None

        # 3. Construct VeNRA Object
        return {
            "id": f"phantom_{doc_type}_{str(uuid.uuid4())[:8]}",
            "dataset_source": "phantom_hallucination",
            "label": venra_label,
            
            # Inputs
            "query": str(query).strip(),
            
            # Phantom contexts are single long strings. We wrap in a list.
            "context_chunks": [str(context).strip()],
            
            # Trace: Explicitly denote text-based analysis (No math trace available)
            "trace_code": "# VERIFICATION_TYPE: TEXT_ANALYSIS\n# Verify via semantic retrieval from long-context documents.",
            
            # Target
            "target_sentence": str(answer).strip(),
            
            # Metadata for Saboteur
            "metadata": {
                "sabotage_ready": is_sabotage_ready,
                "doc_type": doc_type,
                "original_label": raw_label,
                "token_estimate": estimate_token_count(context) # Critical for truncation logic later
            }
        }

if __name__ == "__main__":
    normalizer = PhantomNormalizer()
    normalizer.process()