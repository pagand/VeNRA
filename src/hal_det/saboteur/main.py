import json
import random
import os
import glob
import sys
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple

# Import fix: Ensure the local package is path-accessible when running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saboteur.attacks.numeric import NumericSaboteur
from saboteur.attacks.logic import LogicSaboteur
from saboteur.attacks.semantic import SemanticSaboteur
from saboteur.attacks.irrelevancy import IrrelevancySaboteur

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GOLDEN_DIR = os.path.join(PROJECT_ROOT, "data", "golden_records")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "training_candidates")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_train.jsonl")

TARGET_TOTAL_SIZE = 10000
RATIO_UNFOUNDED = 0.40  
RATIO_SUPPORTED = 0.40  
RATIO_GENERAL = 0.20    

class SabotageOrchestrator:
    def __init__(self):
        self.attacks = [LogicSaboteur(), NumericSaboteur(), IrrelevancySaboteur(), SemanticSaboteur()]
        self.pool_general = []      
        self.pool_natural_fail = [] 
        self.pool_supported = []    
        self.successful_sabotages: List[Tuple[Dict, Dict]] = [] 

    def load_data(self):
        print(f"Loading Golden Records from {GOLDEN_DIR}...")
        if not os.path.exists(GOLDEN_DIR):
            print(f"ERROR: GOLDEN_DIR does not exist: {GOLDEN_DIR}")
            return
            
        files = glob.glob(os.path.join(GOLDEN_DIR, "*.jsonl"))
        if not files:
            print(f"ERROR: No .jsonl files found in {GOLDEN_DIR}")
            return
            
        for f in files:
            dataset_name = os.path.basename(f)
            file_count = 0
            with open(f, 'r', encoding='utf-8') as reader:
                for line in reader:
                    if not line.strip(): continue
                    try:
                        r = json.loads(line)
                        label = str(r.get("label", "")).strip().lower()
                        if label == "general": self.pool_general.append(r)
                        elif label == "unfounded": self.pool_natural_fail.append(r)
                        elif label == "supported": self.pool_supported.append(r)
                        file_count += 1
                    except json.JSONDecodeError: continue
            print(f"  - Loaded {file_count} rows from {dataset_name}")

        print("-" * 30)
        print(f"Pool Sizes:")
        print(f"  Supported (Sabotage Candidates): {len(self.pool_supported)}")
        print(f"  Natural Unfounded (Baselines):   {len(self.pool_natural_fail)}")
        print(f"  General (Axioms):                {len(self.pool_general)}")
        print("-" * 30)

    def run_attacks(self):
        if not self.pool_supported:
            print("WARNING: pool_supported is empty. Skipping attacks.")
            return

        distractors = random.sample(self.pool_supported, min(2000, len(self.pool_supported)))
        for atk in self.attacks:
            if isinstance(atk, IrrelevancySaboteur):
                atk.set_pool(distractors)

        random.shuffle(self.pool_supported)
        for record in tqdm(self.pool_supported, desc="Sabotaging"):
            attack_result = None
            trace = record.get("trace_code", "")
            
            if trace and "# VERIFICATION" not in trace and "(" in trace:
                attack_result = self.attacks[0].attack(record) 
            
            if not attack_result:
                attack_result = self.attacks[1].attack(record) 
                
            if not attack_result:
                fallbacks = [self.attacks[2], self.attacks[3]] 
                random.shuffle(fallbacks)
                for atk in fallbacks:
                    res = atk.attack(record)
                    if res:
                        attack_result = res
                        break
            
            if attack_result:
                self.successful_sabotages.append((record, attack_result))

    def balance_and_export(self):
        print("\nBalancing Dataset Distribution...")
        final_dataset = []
        
        # 1. UNFOUNDED Class
        n_unfounded = int(TARGET_TOTAL_SIZE * RATIO_UNFOUNDED)
        
        # FIX: Guarantee that at least 50% of the Unfounded class consists of Sabotaged Contrast Pairs
        sabotaged_clones = [pair[1] for pair in self.successful_sabotages]
        random.shuffle(sabotaged_clones)
        
        guaranteed_sabotage_quota = min(len(sabotaged_clones), int(n_unfounded * 0.5))
        selected_sabotaged = sabotaged_clones[:guaranteed_sabotage_quota]
        
        # Fill remainder with Natural Fails
        remaining_unfounded_slots = n_unfounded - len(selected_sabotaged)
        random.shuffle(self.pool_natural_fail)
        selected_natural = self.pool_natural_fail[:remaining_unfounded_slots]
        
        # If Natural fails are exhausted, try to fill back up with more sabotaged rows
        if len(selected_natural) < remaining_unfounded_slots:
            extra_sabotage_needed = remaining_unfounded_slots - len(selected_natural)
            extra_sabotaged = sabotaged_clones[guaranteed_sabotage_quota:guaranteed_sabotage_quota + extra_sabotage_needed]
            selected_sabotaged.extend(extra_sabotaged)
            
        unfounded_set = selected_natural + selected_sabotaged
        final_dataset.extend(unfounded_set)
        
        # 2. SUPPORTED Class
        n_supported = int(TARGET_TOTAL_SIZE * RATIO_SUPPORTED)
        
        selected_sabotage_ids = {r["id"] for r in selected_sabotaged}
        contrast_parents = []
        for parent, child in self.successful_sabotages:
            if child["id"] in selected_sabotage_ids:
                contrast_parents.append(parent)
                
        parent_ids = {p["id"] for p in contrast_parents}
        
        # Identify priority rows (FinanceBench Test with reasoning)
        priority_supported = []
        other_supported = []
        
        for r in self.pool_supported:
            if r["id"] in parent_ids: continue
            
            # Check source - prioritize FinanceBench Test
            src = r.get("dataset_source", "").lower()
            if "financebench_test" in src or "financebench" in src:
                priority_supported.append(r)
            else:
                other_supported.append(r)
        
        random.shuffle(other_supported)
        
        # Construct Supported Set: Contrast Parents + Priority Rows + Random Fill
        # We allow overflowing n_supported if priority rows demand it
        supported_set = contrast_parents + priority_supported
        
        current_count = len(supported_set)
        remaining_slots = max(0, n_supported - current_count)
        
        if remaining_slots > 0:
            supported_set.extend(other_supported[:remaining_slots])
            
        final_dataset.extend(supported_set)
        
        # 3. GENERAL Class
        n_general = int(TARGET_TOTAL_SIZE * RATIO_GENERAL)
        source_len = len(self.pool_general)
        if source_len > 0:
            if source_len < n_general:
                factor = n_general // source_len
                remainder = n_general % source_len
                general_set = (self.pool_general * factor) + self.pool_general[:remainder]
            else:
                general_set = random.sample(self.pool_general, n_general)
            final_dataset.extend(general_set)
        else:
            general_set = []

        # 4. Export
        formatted_dataset = [self._format_entry(r) for r in final_dataset]
        random.shuffle(formatted_dataset)
        
        print(f"  - Unfounded: {len(unfounded_set)} (Natural: {len(selected_natural)}, Sabotaged: {len(selected_sabotaged)})")
        print(f"  - Supported: {len(supported_set)} (Contrast: {len(contrast_parents)}, Priority FB: {len(priority_supported)})")
        print(f"  - General:   {len(general_set)}")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in formatted_dataset:
                f.write(json.dumps(entry) + '\n')
                
    def _format_entry(self, record: Dict) -> Dict:
        nested = {
            "id": record["id"],
            "dataset_source": record.get("dataset_source", "unknown"),
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

if __name__ == "__main__":
    orchestrator = SabotageOrchestrator()
    orchestrator.load_data()
    orchestrator.run_attacks()
    orchestrator.balance_and_export()
