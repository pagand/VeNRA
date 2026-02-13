import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_FILE = PROJECT_ROOT / "data" / "training_candidates" / "candidate_train.jsonl"
AUDIT_CACHE = PROJECT_ROOT / "data" / "training_ready" / "audit_cache.jsonl"
HUMAN_DECISIONS = PROJECT_ROOT / "data" / "training_ready" / "human_audit_decisions.jsonl"
AI_DECISIONS = PROJECT_ROOT / "data" / "training_ready" / "ai_audit_decisions.jsonl"

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

    human_decisions = set()
    if HUMAN_DECISIONS.exists():
        human_decisions = {json.loads(line)["id"] for line in open(HUMAN_DECISIONS, "r") if line.strip()}
        
    ai_decisions = set()
    if AI_DECISIONS.exists():
        ai_decisions = {json.loads(line)["id"] for line in open(AI_DECISIONS, "r") if line.strip()}
    
    # Create Audit Map and Valid Sabotage Set
    audit_map = {r["id"]: r for r in audit_rows}
    valid_sabotage_ids = {r["id"] for r in audit_rows if r.get("is_valid_sabotage") == True or (r.get("status") == "verified" and r.get("sabotage_type"))}
    
    # --- COUNTERS ---
    class_stats = Counter()
    sabotage_types = Counter()
    breakdown = defaultdict(int)
    
    # New Granular Counters
    granular_matrix = defaultdict(lambda: defaultdict(int))
    total_pipeline_status = Counter()

    # 2. Process Records
    for row in candidates:
        rid = row["id"]
        label = row["label"]
        raw_source = row.get("dataset_source", "unknown")
        is_sabotaged = bool(row.get("sabotage_info"))
        
        # Split Source into Parent/Sabotage
        suffix = " (Sabotage)" if is_sabotaged else " (Parent)"
        display_source = raw_source + suffix
        
        # --- DERIVE FINAL STATUS ---
        audit_res = audit_map.get(rid, {})
        legacy_status = audit_res.get("status", "pending")
        
        final_status = "Pending"
        
        # Special Logic for TruthfulQA (Axioms are never audited)
        if raw_source == "truthfulqa_axioms":
            final_status = "Skipped"
        elif rid in human_decisions:
            final_status = "Human Ver"
        elif rid in ai_decisions:
            final_status = "AI Ver"
        elif legacy_status == "verified":
            final_status = "Auto Ver"
        elif legacy_status == "discarded":
            final_status = "Discard"
        elif legacy_status == "review":
            final_status = "Pending"
        else:
            final_status = "Pending"
            
        granular_matrix[display_source][final_status] += 1
        total_pipeline_status[final_status] += 1

        # --- CLASS A: GENERAL ---
        if label == "General":
            class_stats["General Axioms"] += 1
            breakdown[f"General ({raw_source})"] += 1
            
        # --- CLASS B: UNFOUNDED ---
        elif label == "Unfounded":
            if is_sabotaged:
                if rid in valid_sabotage_ids:
                    class_stats["Validated Sabotage"] += 1
                    breakdown[f"Sabotaged Unfounded ({raw_source})"] += 1
                    sab_type = row["sabotage_info"].get("type", "unknown")
                    sabotage_types[sab_type] += 1
                elif rid in audit_map:
                    class_stats["Rejected Sabotage"] += 1
                else:
                    class_stats["Pending Sabotage"] += 1
            else:
                class_stats["Natural Unfounded"] += 1
                breakdown[f"Natural Unfounded ({raw_source})"] += 1
                
        # --- CLASS C: SUPPORTED ---
        elif label == "Supported":
            is_parent = any(rid in vid for vid in valid_sabotage_ids)
            if is_parent:
                class_stats["Contrast Parents (Supported)"] += 1
                breakdown[f"Supported Parent ({raw_source})"] += 1
            else:
                class_stats["Other Supported"] += 1
                breakdown[f"Supported Other ({raw_source})"] += 1

    # 3. Print High-Level Report
    print(f"\nTotal Candidates Scanned: {len(candidates)}")
    print(f"Total Audited Rows      : {len(audit_rows)}")
    
    val_sab = class_stats['Validated Sabotage']
    rate = (val_sab / len(audit_rows)) * 100 if len(audit_rows) > 0 else 0
    print(f"Valid Sabotage Rate     : {val_sab}/{len(audit_rows)} ({rate:.1f}%)")

    print(f"\n[GRANULAR PIPELINE STATUS]")
    print(f"Auto Verified          : {total_pipeline_status['Auto Ver']}")
    print(f"AI Verified (Proxy)    : {total_pipeline_status['AI Ver']}")
    print(f"Human Verified         : {total_pipeline_status['Human Ver']}")
    print(f"Discarded              : {total_pipeline_status['Discard']}")
    print(f"Skipped (Axioms)       : {total_pipeline_status['Skipped']}")
    print(f"Action Required        : {total_pipeline_status['Pending']}")

    print(f"\n[DETAILED SOURCE BREAKDOWN]")
    # Header
    print(f"{'Source':<35} | {'Auto':<6} | {'AI':<6} | {'Human':<6} | {'Discard':<8} | {'Skipped':<8} | {'Pending':<8}")
    print("-" * 95)
    
    for source, stats in sorted(granular_matrix.items()):
        print(f"{source[:35]:<35} | {stats['Auto Ver']:<6} | {stats['AI Ver']:<6} | {stats['Human Ver']:<6} | {stats['Discard']:<8} | {stats['Skipped']:<8} | {stats['Pending']:<8}")
    
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

    print("\n" + "="*75)
    if total_pipeline_status["Pending"] > 0:
        print(f"🚀 ACTION: You have {total_pipeline_status['Pending']} items pending review/audit.")
        print("   -> Run `python -m src.hal_det.ui.ai_proxy` to batch process with AI.")
        print("   -> Or run `streamlit run src/hal_det/ui/app.py` for manual review.")
    else:
        print("✅ READY: Dataset is fully processed and verified.")
    print("="*75)

if __name__ == "__main__":
    analyze_dataset()
