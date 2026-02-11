import json
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

def verify_mock_logic():
    candidate_path = PROJECT_ROOT / "data/training_candidates/candidate_train.jsonl"
    
    print("Searching for a FinanceBench Test row in candidates...")
    target_row = None
    with open(candidate_path, "r") as f:
        for line in f:
            row = json.loads(line)
            # Find one with numbers in reasoning if possible
            if row.get("dataset_source") == "financebench_test" and row.get("metadata", {}).get("original_reasoning"):
                reasoning = row["metadata"]["original_reasoning"]
                if "." in reasoning: # ensure it has periods
                    target_row = row
                    break
    
    if not target_row:
        print("No FinanceBench row with reasoning found!")
        return

    print(f"Found Row ID: {target_row['id']}")
    original_reasoning = target_row["metadata"]["original_reasoning"]
    print(f"Original Reasoning (Raw): {original_reasoning[:100]}...")

    # --- IMPROVED SPLIT LOGIC ---
    # Split by ". " to avoid breaking decimals like 12.5
    sentences = [s.strip() for s in original_reasoning.split('. ') if s.strip()]
    
    # Re-attach periods if they were lost during split
    sentences = [s + "." if not s.endswith('.') else s for s in sentences]

    if len(sentences) > 1:
        simulated_thinking = " ".join(sentences[:-1])
        simulated_analysis = sentences[-1]
    else:
        simulated_thinking = f"The provided text states: {original_reasoning}"
        simulated_analysis = original_reasoning
    
    derived_span = "N/A"
    if target_row["label"] == "Unfounded":
        derived_span = target_row.get("target_sentence", "entire_sentence")

    print("\n--- MOCK RESULT (Improved Split) ---")
    print(f"Label:    {target_row['label']}")
    print(f"Thinking: {simulated_thinking}")
    print("-" * 20)
    print(f"Analysis: {simulated_analysis}")
    print("-" * 20)
    print(f"Span:     {derived_span}")
    print("-------------------")

if __name__ == "__main__":
    verify_mock_logic()