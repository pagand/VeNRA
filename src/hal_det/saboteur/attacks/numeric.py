import random
from typing import Dict, Optional, List
from .base import BaseSaboteur
from saboteur.utils.text_utils import generate_sabotage_id, parse_markdown_table, substitute_token_in_text

class NumericSaboteur(BaseSaboteur):
    """
    Targets: Records with Tables (Raw or Markdown).
    Vector: Neighbor Trap (Row/Col shift).
    """
    def __init__(self):
        super().__init__("numeric_neighbor_trap")

    def attack(self, record: Dict) -> Optional[Dict]:
        meta = record.get("metadata", {})
        inputs = record.get("inputs", {}) # Future proofing, or current flat structure
        
        # 1. Resolve Table Data
        table_data = []
        if "raw_table" in meta and meta["raw_table"]:
            table_data = meta["raw_table"]
        else:
            # Try parsing from Context (Fix for FinQA)
            context = record.get("context_chunks", [])
            for chunk in context:
                if "|" in chunk and "---" in chunk:
                    parsed = parse_markdown_table(chunk)
                    if len(parsed) > 2: # Min size to be useful
                        table_data = parsed
                        break
        
        if not table_data: return None

        # 2. Extract Target Value
        target_sent = record.get("target_sentence", "")
        # Heuristic: Find numbers in the sentence
        # We assume the answer IS a number or contains a prominent one.
        # Ideally, we look for the number that equals the answer.
        # For FinQA/TAT-QA, the answer is often just the number.
        
        # We try to find a value from the table that is present in the target_sentence
        target_val_in_table = None
        target_coords = None
        
        found_candidates = []
        
        for r, row in enumerate(table_data):
            for c, cell in enumerate(row):
                val_str = str(cell).strip()
                if not val_str: continue
                # Check if this table cell value exists in the sentence
                if val_str in target_sent:
                     # Avoid trivial matches like "1" or "0" unless exact match
                    if len(val_str) < 2 and val_str != target_sent: continue
                    found_candidates.append((val_str, r, c))
        
        if not found_candidates: return None
        
        # Pick one to sabotage (e.g. the longest match is likely the specific number)
        found_candidates.sort(key=lambda x: len(x[0]), reverse=True)
        original_val, r, c = found_candidates[0]

        # 3. Pick Neighbor
        neighbors = []
        # Down (Row + 1)
        if r + 1 < len(table_data) and c < len(table_data[r+1]):
            neighbors.append(table_data[r+1][c])
        # Up (Row - 1)
        if r - 1 >= 0 and c < len(table_data[r-1]):
            neighbors.append(table_data[r-1][c])
        # Right (Col + 1)
        if c + 1 < len(table_data[r]):
            neighbors.append(table_data[r][c+1])
            
        # Filter valid neighbors
        valid_neighbors = [n for n in neighbors if n and n != original_val and any(char.isdigit() for char in str(n))]
        if not valid_neighbors: return None
        
        new_val = random.choice(valid_neighbors)
        
        # 4. Apply Substitution
        new_sent = substitute_token_in_text(target_sent, original_val, new_val)
        if not new_sent: return None
        
        # 5. Build Record
        import copy
        new_record = copy.deepcopy(record)
        new_record["id"] = generate_sabotage_id(record["id"])
        new_record["target_sentence"] = new_sent
        new_record["label"] = "Unfounded"
        new_record["sabotage_info"] = {
            "applied": True,
            "type": self.name,
            "subtype": "table_shift",
            "diff": f"Swapped {original_val} (Row {r}, Col {c}) with {new_val}",
            "original_sentence": target_sent
        }
        return new_record