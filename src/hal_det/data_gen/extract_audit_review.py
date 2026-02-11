import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_PATH = PROJECT_ROOT / "data/training_candidates/candidate_train.jsonl"
AUDIT_CACHE_PATH = PROJECT_ROOT / "data/training_ready/audit_cache.jsonl"
EXPORT_PATH = PROJECT_ROOT / "data/training_ready/audit_export_review.jsonl"

def extract_injected_value(diff_text):
    """Helper to extract just the lie from strings like 'Swapped X with Y' or 'X -> Y'"""
    if not diff_text: return "N/A"
    if "with" in diff_text:
        return diff_text.split("with")[-1].strip().rstrip('.')
    if "->" in diff_text:
        return diff_text.split("->")[-1].strip().rstrip('.')
    return diff_text

def run_export():
    print("🚀 Starting Review/Discard Export...")
    
    # 1. Load Candidates into memory (ID lookup)
    candidates = {}
    if not CANDIDATE_PATH.exists():
        print(f"Error: {CANDIDATE_PATH} not found.")
        return
        
    with open(CANDIDATE_PATH, "r") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            candidates[row["id"]] = row
    print(f"Loaded {len(candidates)} candidate rows.")

    # 2. Process Audit Cache
    exported_count = 0
    if not AUDIT_CACHE_PATH.exists():
        print(f"Error: {AUDIT_CACHE_PATH} not found.")
        return

    with open(AUDIT_CACHE_PATH, "r") as fin, open(EXPORT_PATH, "w") as fout:
        for line in fin:
            if not line.strip(): continue
            audit = json.loads(line)
            
            # Filter: We only want Review or Discarded
            status = audit.get("status")
            if status not in ["review"]:
                continue
            
            rid = audit["id"]
            raw = candidates.get(rid, {})
            
            if not raw:
                # Skip if candidate missing (shouldn't happen)
                continue
                
            # Extract Sabotage Info
            sab_info = raw.get("sabotage_info", {})
            
            # Build the Flattened Export Object
            export_row = {
                "id": rid,
                "source_dataset": raw.get("dataset_source", "unknown"),
                "audit_status": status,
                "audit_target": audit.get("audit_target_group", "N/A"),
                "attack_type": sab_info.get("type", "none"),
                "intended_lie": extract_injected_value(sab_info.get("diff", "")),
                "teacher_verdict": audit.get("teacher_label", "N/A"),
                "confidence": audit.get("teacher_confidence", "N/A"),
                "target_sentence": raw.get("target_sentence", ""),
                "logic_trace": raw.get("inputs", {}).get("trace_code", ""),
                "internal_thinking": audit.get("teacher_thinking", ""),
                "final_analysis": audit.get("teacher_analysis", "")
            }
            
            fout.write(json.dumps(export_row) + "\n")
            exported_count += 1

    print(f"✅ Export Complete! {exported_count} rows written to: {EXPORT_PATH}")
    print("You can now open this file to review the problematic cases.")

if __name__ == "__main__":
    run_export()
