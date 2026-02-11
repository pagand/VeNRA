import json
import uuid
import pandas as pd
import requests
import io
import os
from typing import Dict, Any, List

# ==========================================
# CONFIGURATION
# ==========================================
# Direct CSV link from the dataset card info provided
TRUTHFULQA_CSV_URL = "https://huggingface.co/datasets/domenicrosati/TruthfulQA/resolve/main/TruthfulQA.csv"
OUTPUT_FILE = "data/golden_records/truthfulqa_normalized.jsonl"

# We target these specific domains to create the "Financial/Regulatory Axiom" dataset
# Note: 'Economics' is not a native TruthfulQA category, but we keep it in the list
# in case the dataset taxonomy changes or includes sub-categories.
TARGET_CATEGORIES = [
    'Finance', 'Law', 'Politics', 'Economics', 
    'History', 'Science', 'Health', 'Sociology', 'Misconceptions'
]

# ==========================================
# MAIN CLASS
# ==========================================

class TruthfulQANormalizer:
    def __init__(self):
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    def process(self):
        print(f"Downloading TruthfulQA from {TRUTHFULQA_CSV_URL}...")
        try:
            # Download directly into Pandas to ensure we get the raw CSV structure
            # bypassing potential HF Config errors.
            r = requests.get(TRUTHFULQA_CSV_URL)
            if r.status_code != 200:
                raise Exception(f"Failed to download: {r.status_code}")
            
            df = pd.read_csv(io.BytesIO(r.content))
            print(f"Loaded {len(df)} total rows.")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading TruthfulQA: {e}")
            return

        normalized_data = []
        
        # Stats counter
        category_counts = {cat: 0 for cat in TARGET_CATEGORIES}
        
        for _, row in df.iterrows():
            entry = self._normalize_row(row, category_counts)
            if entry:
                normalized_data.append(entry)

        # ---------------------------------------------------------
        # Volume Validation
        # ---------------------------------------------------------
        print("-" * 30)
        print("Category Breakdown:")
        for cat, count in category_counts.items():
            print(f"  - {cat}: {count}")
        print("-" * 30)
        
        print(f"Total normalized 'General' rows: {len(normalized_data)}")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in normalized_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved to {OUTPUT_FILE}")

    def _normalize_row(self, row: pd.Series, stats_counter: Dict) -> Dict:
        """
        Maps TruthfulQA Row -> VeNRA General Schema.
        """
        # 1. Category Filter
        raw_category = str(row.get('Category', '')).strip()
        
        # Check if any target category is in the raw category string (case-insensitive)
        matched_category = None
        for target in TARGET_CATEGORIES:
            if target.lower() in raw_category.lower():
                matched_category = target
                stats_counter[target] += 1
                break
        
        if not matched_category:
            return None

        # 2. Extract Data
        question = row.get('Question')
        best_answer = row.get('Best Answer')
        
        if pd.isna(question) or pd.isna(best_answer):
            return None

        return {
            "id": f"truthfulqa_{str(uuid.uuid4())[:8]}",
            "dataset_source": "truthfulqa_axioms",
            "label": "General",
            
            # Inputs
            "query": str(question).strip(),
            
            # CRITICAL SPEC: Empty Context for "General" Class
            # This forces the model to verify based on internal knowledge weights.
            "context_chunks": [], 
            
            # Trace: Explicitly denote no retrieval occurred
            "trace_code": "# VERIFICATION_TYPE: GENERAL_KNOWLEDGE\n# This fact is axiomatic and requires no external evidence context.",
            
            # Target
            "target_sentence": str(best_answer).strip(),
            
            # Metadata
            "metadata": {
                "sabotage_ready": False, # Never sabotage General Knowledge
                "original_category": raw_category,
                "source_ref": str(row.get('Source', 'Unknown'))
            }
        }

if __name__ == "__main__":
    normalizer = TruthfulQANormalizer()
    normalizer.process()