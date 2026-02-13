import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_ROOT = Path("data")
CANDIDATES_PATH = DATA_ROOT / "training_candidates" / "candidate_train.jsonl"
READY_PATH = DATA_ROOT / "training_ready"
OUTPUT_DIR = DATA_ROOT / "training_final"

# Audit Files
HUMAN_DECISIONS_PATH = READY_PATH / "human_audit_decisions.jsonl"
AI_DECISIONS_PATH = READY_PATH / "ai_audit_decisions.jsonl"
TEACHER_AUDIT_PATH = READY_PATH / "audit_cache.jsonl"


def ensure_output_dir():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {OUTPUT_DIR}")


def load_jsonl_map(file_path: Path, key_field: str = "id") -> Dict[str, Any]:
    """Loads a JSONL file into a dictionary keyed by ID."""
    data_map = {}
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}. Skipping.")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if key_field in row:
                    data_map[row[key_field]] = row
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(data_map)} records from {file_path.name}")
    return data_map


def load_teacher_audit_map(file_path: Path) -> Dict[str, Any]:
    """
    Loads the teacher audit cache with specific logic to handle race conditions.
    Prioritizes 'verified' status over 'discarded' for the same ID.
    """
    data_map = {}
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}. Skipping.")
        return {}

    total_lines = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                row = json.loads(line)
                row_id = row.get("id")
                if not row_id:
                    continue

                if row_id in data_map:
                    existing = data_map[row_id]
                    if existing.get("status") != "verified" and row.get("status") == "verified":
                        data_map[row_id] = row
                else:
                    data_map[row_id] = row

            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(data_map)} unique audits from {total_lines} lines in {file_path.name}")
    return data_map


def build_distractor_pool(candidates_path: Path, limit: int = 1000) -> List[str]:
    """
    Scans candidates to build a pool of realistic financial text chunks.
    Source: finqa, financebench, tatqa.
    """
    logger.info(f"Building distractor pool from {candidates_path}...")
    pool = []
    if not candidates_path.exists():
        return pool

    target_sources = ["finqa", "financebench", "tatqa"]
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(pool) >= limit:
                break
            try:
                candidate = json.loads(line)
                meta = candidate.get("metadata", {})
                source = (candidate.get("dataset_source") or meta.get("source_dataset") or "").lower()
                
                if any(ts in source for ts in target_sources):
                    inputs = candidate.get("inputs", {})
                    chunks = inputs.get("context_chunks", [])
                    if chunks:
                        pool.extend(chunks)
            except:
                continue
    
    # Shuffle and trim
    random.shuffle(pool)
    pool = pool[:limit]
    logger.info(f"Distractor pool size: {len(pool)}")
    return pool


def format_canonical_record(
    candidate: Dict[str, Any], 
    audit_data: Dict[str, Any], 
    source_type: str,
    distractor_pool: List[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Formats the final record into the schema agnostic structure.
    Implements Noise Injection for TruthfulQA.
    DERIVES FAMILY_ID for Group-Splitting.
    """
    
    # 1. Label
    final_label = (
        audit_data.get("final_label") or 
        audit_data.get("teacher_label") or 
        audit_data.get("label") or 
        audit_data.get("verdict")
    )
    if final_label:
        final_label = final_label.title()
        
    # 2. Target Sentence
    final_sentence = (
        audit_data.get("final_sentence") or 
        audit_data.get("target_sentence") or 
        candidate.get("target_sentence")
    )

    # 3. Reasoning / Analysis
    final_reasoning = (
        audit_data.get("final_analysis") or 
        audit_data.get("final_reasoning") or 
        audit_data.get("teacher_analysis") or 
        audit_data.get("reasoning") or 
        audit_data.get("analysis")
    )
    
    # Metadata extraction
    meta = candidate.get("metadata", {})
    dataset_source = (
        candidate.get("dataset_source") or 
        meta.get("source_dataset") or 
        "unknown"
    ).lower()
    sabotage_type = meta.get("sabotage_type") or "natural"
    
    # --- FAMILY ID DERIVATION (NEW) ---
    # Used to keep Parents and Children together in splits
    # Logic: Explicit metadata > ID Heuristic (strip 'sabotaged_' prefix AND uuid suffix) > ID itself
    candidate_id = candidate["id"]
    if "parent_id" in meta:
        family_id = meta["parent_id"]
    elif "original_id" in meta:
        family_id = meta["original_id"]
    elif candidate_id.startswith("sabotaged_"):
        # Format: sabotaged_{original_id}_{uuid_8_chars}
        # 1. Remove prefix
        temp = candidate_id[len("sabotaged_"):]
        # 2. Remove suffix (last underscore + 8 chars)
        # We use rsplit to be safe against underscores in the original ID
        if "_" in temp:
            family_id = temp.rsplit("_", 1)[0]
        else:
            family_id = temp # Fallback if no suffix found (unlikely)
    else:
        family_id = candidate_id

    if not final_label:
        return None

    # Extract Inputs
    inputs_block = candidate.get("inputs", {})
    query = inputs_block.get("query")
    context_chunks = inputs_block.get("context_chunks", [])
    trace_code = inputs_block.get("trace_code")

    # --- NOISE INJECTION LOGIC ---
    # If TruthfulQA and context is empty, inject distractors
    if "truthfulqa" in dataset_source and not context_chunks and distractor_pool:
        # Select 2-3 random chunks
        num_distractors = random.randint(2, 3)
        context_chunks = random.sample(distractor_pool, min(num_distractors, len(distractor_pool)))

    # Calculate token count (approximate)
    context_text = " ".join(context_chunks) if isinstance(context_chunks, list) else str(context_chunks)
    token_count = len(context_text.split())
    
    return {
        "id": candidate["id"],
        "split": "pending",
        "label": final_label,
        "dataset_source": dataset_source,
        "sabotage_type": sabotage_type,
        "input_components": {
            "query": query,
            "context": context_chunks,
            "trace": trace_code
        },
        "output_components": {
            "verdict": final_label,
            "reasoning": final_reasoning,
            "target_sentence": final_sentence
        },
        "meta": {
            "family_id": family_id,
            "audit_source": source_type,
            "token_count": token_count
        }
    }


def resolve_verdict(
    candidate: Dict[str, Any], 
    human_audit: Optional[Dict[str, Any]], 
    ai_audit: Optional[Dict[str, Any]], 
    teacher_audit: Optional[Dict[str, Any]],
    distractor_pool: List[str] = None
) -> Optional[Dict[str, Any]]:
    """
    The Waterfall Decision Logic (v2.0).
    """
    cid = candidate["id"]
    
    if human_audit:
        return format_canonical_record(candidate, human_audit, "human", distractor_pool)
        
    if ai_audit:
        return format_canonical_record(candidate, ai_audit, "ai", distractor_pool)
        
    if teacher_audit:
        status = teacher_audit.get("status")
        verdict = teacher_audit.get("teacher_label") or teacher_audit.get("verdict")
        is_valid_sabotage = teacher_audit.get("is_valid_sabotage", False)
        
        if status == "discarded":
            return None
            
        meta = candidate.get("metadata", {})
        source = (candidate.get("dataset_source") or meta.get("source_dataset") or "").lower()

        if "truthfulqa" in source:
            if not verdict or verdict.title() != "General":
                return None 
            
        candidate_label = candidate.get("label", "").title()
        teacher_verdict = verdict.title() if verdict else ""
        
        if candidate_label == "Supported" and teacher_verdict == "Unfounded":
            return None
            
        if status == "verified" or is_valid_sabotage:
             return format_canonical_record(candidate, teacher_audit, "teacher", distractor_pool)

    return None


def hybrid_family_split(accepted_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Performs Hybrid-Family Splitting (v2.1).
    Ensures Families (Parents + Sabotage Children) stay together in the same split.
    Buckets families to ensure diverse representation in Val/Test.
    """
    logger.info("Performing Hybrid-Family Splitting...")
    
    # 1. Group by Family
    families = {} # {family_id: [records]}
    for rec in accepted_records:
        fid = rec['meta']['family_id']
        if fid not in families:
            families[fid] = []
        families[fid].append(rec)
    
    # 2. Bucket Families
    buckets = {
        "sabotage_pair": [],  # Contains Supported AND Unfounded
        "natural_fail": [],   # Contains Unfounded (but no Supported sibling found in set)
        "natural_supp": [],   # Contains Supported (no Unfounded sibling)
        "axiom": []           # TruthfulQA
    }
    
    for fid, members in families.items():
        labels = {m['label'] for m in members}
        sources = {m['dataset_source'] for m in members}
        
        if any("truthfulqa" in s for s in sources):
            buckets["axiom"].append(fid)
        elif "Unfounded" in labels and "Supported" in labels:
            buckets["sabotage_pair"].append(fid)
        elif "Unfounded" in labels:
            buckets["natural_fail"].append(fid)
        else:
            buckets["natural_supp"].append(fid)
            
    # 3. Stratified Split (10% Val, 10% Test per Bucket)
    splits = {"train": [], "val": [], "test": []}
    
    for b_name, fids in buckets.items():
        random.shuffle(fids)
        n = len(fids)
        n_val = int(n * 0.10)
        n_test = int(n * 0.10)
        
        # Enforce minimums if possible
        if n > 0 and n_val == 0 and n >= 3: n_val = 1
        if n > 0 and n_test == 0 and n >= 3: n_test = 1
        
        test_fids = fids[:n_test]
        val_fids = fids[n_test : n_test + n_val]
        train_fids = fids[n_test + n_val:]
        
        for fid in test_fids: splits["test"].extend(families[fid])
        for fid in val_fids: splits["val"].extend(families[fid])
        for fid in train_fids: splits["train"].extend(families[fid])
        
        logger.info(f"Bucket '{b_name}': {n} families -> Train:{len(train_fids)} Val:{len(val_fids)} Test:{len(test_fids)}")

    return splits


def main():
    ensure_output_dir()
    random.seed(42)
    
    # 1. Load Audit Maps
    logger.info("Loading audit maps...")
    human_map = load_jsonl_map(HUMAN_DECISIONS_PATH)
    ai_map = load_jsonl_map(AI_DECISIONS_PATH)
    teacher_map = load_teacher_audit_map(TEACHER_AUDIT_PATH)
    
    # 2. Build Distractor Pool
    distractor_pool = build_distractor_pool(CANDIDATES_PATH)
    
    # 3. Stream Candidates
    seen_ids = set()
    accepted_records = []
    
    stats = {
        "total_scanned": 0,
        "duplicates": 0,
        "accepted": 0,
        "rejected": 0,
        "by_source": {"human": 0, "ai": 0, "teacher": 0}
    }
    
    logger.info("Processing candidates...")
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_scanned"] += 1
            try:
                candidate = json.loads(line)
                cid = candidate.get("id")
                
                if cid in seen_ids:
                    stats["duplicates"] += 1
                    continue
                seen_ids.add(cid)
                
                human = human_map.get(cid)
                ai = ai_map.get(cid)
                teacher = teacher_map.get(cid)
                
                final_record = resolve_verdict(candidate, human, ai, teacher, distractor_pool)
                
                if final_record:
                    accepted_records.append(final_record)
                    stats["accepted"] += 1
                    stats["by_source"][final_record["meta"]["audit_source"]] += 1
                else:
                    stats["rejected"] += 1
                    
            except json.JSONDecodeError:
                continue
                
    logger.info(f"Processing complete. Stats: {json.dumps(stats, indent=2)}")
    
    if not accepted_records:
        logger.error("No records accepted! Check input data.")
        return

    # 4. Hybrid-Family Split (Replacing older pandas shuffle)
    final_splits = hybrid_family_split(accepted_records)
    
    # 5. Save Outputs
    logger.info("Saving split files...")
    split_counts = {}
    
    for split_name, records in final_splits.items():
        split_counts[split_name] = len(records)
        output_file = OUTPUT_DIR / f"{split_name}.jsonl"
        
        # Tag the record with its final split
        for rec in records:
            rec['split'] = split_name
            
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {len(records)} records to {output_file.name}")

    # Final Summary (Analysis)
    print("\n" + "="*40)
    print("FINAL DATASET REPORT")
    print("="*40)
    print(f"Total Candidates Processed: {stats['total_scanned']}")
    print(f"Duplicates Removed        : {stats['duplicates']}")
    print(f"Total Accepted            : {stats['accepted']}")
    print(f"Total Rejected            : {stats['rejected']}")
    print("-" * 40)
    print("Source Breakdown:")
    print(f"  Human Verified: {stats['by_source']['human']}")
    print(f"  AI Verified   : {stats['by_source']['ai']}")
    print(f"  Auto Verified : {stats['by_source']['teacher']}")
    print("-" * 40)
    print("Split Distribution (Row Count):")
    for s_name, count in split_counts.items():
        print(f"  {s_name.upper():<6}: {count}")
    print("="*40)


if __name__ == "__main__":
    main()