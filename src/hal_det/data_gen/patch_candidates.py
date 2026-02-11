import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NORMALIZED_PATH = PROJECT_ROOT / "data/golden_records/financebench_normalized.jsonl"
CANDIDATES_PATH = PROJECT_ROOT / "data/training_candidates/candidate_train.jsonl"
TEMP_PATH = PROJECT_ROOT / "data/training_candidates/candidate_train.jsonl.tmp"

def format_entry(record):
    """
    Ensure the new rows match the schema of candidate_train.jsonl
    """
    nested = {
        "id": record["id"],
        "dataset_source": record.get("dataset_source", "financebench_test"),
        "label": record["label"],
        "target_sentence": record["target_sentence"],
        "inputs": {
            "query": record.get("query", ""),
            "context_chunks": record.get("context_chunks", []),
            "trace_code": record.get("trace_code", "")
        },
        "metadata": record.get("metadata", {}),
    }
    if "sabotage_info" in record:
        nested["sabotage_info"] = record["sabotage_info"]
    return nested

def run_patch():
    print("Loading normalized FinanceBench data...")
    fb_data = {} # id -> full record
    if NORMALIZED_PATH.exists():
        with open(NORMALIZED_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    # We want ALL financebench test rows, primarily those with reasoning
                    # But actually, the user said "all 1000 samples".
                    fb_data[obj["id"]] = obj
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(fb_data)} FinanceBench rows.")
    
    print("Scanning existing candidate_train.jsonl...")
    existing_ids = set()
    rows_to_keep = []
    
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r") as fin:
            for line in fin:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    row_id = row.get("id")
                    
                    # If this row is already a FinanceBench row, we will replace it with the fresh one 
                    # from fb_data to ensure it has the latest metadata/reasoning.
                    if row_id in fb_data:
                        # We skip adding the OLD version to rows_to_keep
                        # We will add the NEW version later
                        continue
                    
                    rows_to_keep.append(row)
                    existing_ids.add(row_id)
                except json.JSONDecodeError:
                    continue
    
    print(f"Retained {len(rows_to_keep)} non-FinanceBench rows.")
    
    # Now merge: 
    # 1. Old rows (minus old FB rows)
    # 2. All NEW FB rows (formatted correctly)
    
    final_rows = rows_to_keep
    
    fb_added_count = 0
    for fb_id, fb_record in fb_data.items():
        # format it to match candidate schema
        formatted_row = format_entry(fb_record)
        final_rows.append(formatted_row)
        fb_added_count += 1
        
    print(f"Added/Updated {fb_added_count} FinanceBench rows.")
    print(f"Total rows in new file: {len(final_rows)}")
    
    with open(TEMP_PATH, "w") as fout:
        for row in final_rows:
            fout.write(json.dumps(row) + "\n")
            
    # Atomic Swap
    os.replace(TEMP_PATH, CANDIDATES_PATH)
    print("Successfully updated candidate_train.jsonl with ALL FinanceBench rows.")

if __name__ == "__main__":
    run_patch()
