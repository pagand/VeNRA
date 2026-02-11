import json
import os
import ast
from collections import defaultdict
from typing import Dict, Any

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "training_candidates", "candidate_train.jsonl")

def validate_dataset():
    print(f"Validating dataset: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file does not exist: {INPUT_FILE}")
        return

    stats = defaultdict(int)
    errors = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                record = json.loads(line)
                stats["total_rows"] += 1
                
                # 1. Schema Check
                if "inputs" not in record:
                    errors.append(f"Row {i}: Missing 'inputs' key.")
                    stats["schema_errors"] += 1
                    continue
                    
                inputs = record["inputs"]
                if not inputs.get("query"):
                    errors.append(f"Row {i}: Empty query.")
                    stats["empty_query"] += 1
                    
                if not inputs.get("context_chunks"):
                    # General knowledge rows explicitly have empty context
                    if record.get("label") != "General":
                        errors.append(f"Row {i}: Empty context_chunks for non-General row.")
                        stats["empty_context"] += 1

                # 2. Trace Code Validation
                trace = inputs.get("trace_code")
                if trace:
                    if "# TRACE_GENERATION_ERROR" in trace:
                        errors.append(f"Row {i}: Trace generation error detected.")
                        stats["trace_errors"] += 1
                    elif "# NO_TRACE_AVAILABLE" in trace:
                        stats["no_trace"] += 1
                    else:
                        # Syntax Check
                        try:
                            ast.parse(trace)
                            stats["valid_traces"] += 1
                        except SyntaxError:
                            errors.append(f"Row {i}: Trace syntax error.")
                            stats["trace_syntax_errors"] += 1

                # 3. Label Consistency
                label = record.get("label")
                stats[f"label_{label}"] += 1
                
                if label == "Unfounded":
                    # Check if sabotage info exists (unless it's a natural fail)
                    # Natural fails don't have sabotage_info usually, or minimal.
                    # But Sabotaged rows MUST have it.
                    if "sabotaged" in record.get("id", "") and "sabotage_info" not in record:
                        errors.append(f"Row {i}: Sabotaged ID but missing sabotage_info.")
                        stats["missing_sabotage_info"] += 1

            except json.JSONDecodeError:
                errors.append(f"Row {i}: Invalid JSON.")
                stats["json_errors"] += 1

    print("-" * 30)
    print("Validation Report:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    if errors:
        print("\nErrors Found (showing first 10):")
        for e in errors[:10]:
            print(f"  - {e}")
        print(f"\nTotal Errors: {len(errors)}")
    else:
        print("\n✅ Dataset Validation Passed!")
        
if __name__ == "__main__":
    validate_dataset()