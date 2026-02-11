import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_FILE = PROJECT_ROOT / "data" / "training_candidates" / "candidate_train.jsonl"
AUDIT_CACHE = PROJECT_ROOT / "data" / "training_ready" / "audit_cache.jsonl"

def analyze_dataset():
    print("="*75)
    print("🧪 VeNRA COMPREHENSIVE DATASET ANALYSIS")
    print("="*75)
    
    # 1. Load Data
    if not CANDIDATE_FILE.exists():
        print("Error: Missing candidate_train.jsonl")
        return

    candidates = [json.loads(line) for line in open(CANDIDATE_FILE, "r") if line.strip()]
    
    audit_rows = []
    if AUDIT_CACHE.exists():
        audit_rows = [json.loads(line) for line in open(AUDIT_CACHE, "r") if line.strip()]
    
    # Create Audit Map and Valid Sabotage Set
    audit_map = {r["id"]: r for r in audit_rows}
    valid_sabotage_ids = {r["id"] for r in audit_rows if r.get("is_valid_sabotage") == True or (r.get("status") == "verified" and r.get("sabotage_type"))}
    
    # --- COUNTERS ---
    class_stats = Counter()
    sabotage_types = Counter()
    breakdown = defaultdict(int)
    audit_status = Counter()
    source_matrix = defaultdict(lambda: Counter())
    mocked_count = 0

    # 2. Process Records
    for row in candidates:
        rid = row["id"]
        label = row["label"]
        source = row.get("dataset_source", "unknown")
        is_sabotaged = bool(row.get("sabotage_info"))
        
        # Determine Audit Status for Matrix
        if rid in audit_map:
            audit_res = audit_map[rid]
            status = audit_res.get("status", "legacy")
            audit_status[status] += 1
            source_matrix[source][status] += 1
            if "Reasoning derived from original" in audit_res.get("teacher_thinking", ""):
                mocked_count += 1
        else:
            audit_status["pending"] += 1
            source_matrix[source]["pending"] += 1

        # --- CLASS A: GENERAL ---
        if label == "General":
            class_stats["General Axioms"] += 1
            breakdown[f"General ({source})"] += 1
            
        # --- CLASS B: UNFOUNDED ---
        elif label == "Unfounded":
            if is_sabotaged:
                if rid in valid_sabotage_ids:
                    class_stats["Validated Sabotage"] += 1
                    breakdown[f"Sabotaged Unfounded ({source})"] += 1
                    sab_type = row["sabotage_info"].get("type", "unknown")
                    sabotage_types[sab_type] += 1
                elif rid in audit_map:
                    class_stats["Rejected Sabotage"] += 1
                else:
                    class_stats["Pending Sabotage"] += 1
            else:
                class_stats["Natural Unfounded"] += 1
                breakdown[f"Natural Unfounded ({source})"] += 1
                
        # --- CLASS C: SUPPORTED ---
        elif label == "Supported":
            is_parent = any(rid in vid for vid in valid_sabotage_ids)
            if is_parent:
                class_stats["Contrast Parents (Supported)"] += 1
                breakdown[f"Supported Parent ({source})"] += 1
            else:
                class_stats["Other Supported"] += 1
                breakdown[f"Supported Other ({source})"] += 1

    # 3. Print High-Level Report
    print(f"\nTotal Candidates Scanned: {len(candidates)}")
    print(f"Total Audited Rows      : {len(audit_rows)}")
    
    val_sab = class_stats['Validated Sabotage']
    rate = (val_sab / len(audit_rows)) * 100 if len(audit_rows) > 0 else 0
    print(f"Valid Sabotage Rate     : {val_sab}/{len(audit_rows)} ({rate:.1f}%)")

    print(f"\n[AUDIT PIPELINE STATUS]")
    print(f"Verified (Pruned)     : {audit_status['verified']}")
    print(f"Review Required (HITL): {audit_status['review']}")
    print(f"Discarded (Teacher)   : {audit_status['discarded']}")
    print(f"FinanceBench Mocked   : {mocked_count}")
    print(f"Remaining Work        : {audit_status['pending']}")

    print(f"\n[SOURCE-LEVEL COMPLETION MATRIX]")
    print(f"{'Source':<40} | {'Verified':<8} | {'Review':<8} | {'Pending':<8}")
    print("-" * 75)
    for source, s_stats in sorted(source_matrix.items()):
        print(f"{source[:40]:<40} | {s_stats['verified']:<8} | {s_stats['review']:<8} | {s_stats['pending']:<8}")
    
    print("\n--- Category Counts ---")
    category_order = [
        "Contrast Parents (Supported)", "Validated Sabotage", 
        "General Axioms", "Other Supported", "Natural Unfounded",
        "Rejected Sabotage", "Pending Sabotage"
    ]
    for k in category_order:
        print(f"{k:<30}: {class_stats[k]}")
        
    print("\n--- Sabotage Types (Validated) ---")
    for k, v in sabotage_types.items():
        print(f"{k:<30}: {v}")

    print("\n--- Source Breakdown ---")
    for k in sorted(breakdown.keys()):
        print(f"{k:<45}: {breakdown[k]}")

    print("\n" + "="*75)
    if audit_status["pending"] > 0:
        print(f"🚀 ACTION: Run auditor.py to process {audit_status['pending']} pending rows.")
    elif audit_status["review"] > 0:
        print("👉 ACTION: Resolve 'Review' cases in HITL UI (app.py).")
    else:
        print("✅ READY: Dataset is fully processed and verified.")
    print("="*75)

if __name__ == "__main__":
    analyze_dataset()
