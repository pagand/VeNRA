import re
import random
import copy
from .base import BaseSaboteur
from typing import Dict, Optional
from saboteur.utils.code_executor import safe_execute_trace
from saboteur.utils.text_utils import generate_sabotage_id

class LogicSaboteur(BaseSaboteur):
    """
    Targets: Records with 'trace_code'.
    Vector: Input Variable Swap (The Code Lie).
    """
    def __init__(self):
        super().__init__("logic_code_lie")

    def attack(self, record: Dict) -> Optional[Dict]:
        trace = record.get("trace_code", "")
        # Must be python, not tokens
        if not trace or "# VERIFICATION" in trace: return None
        
        # 1. Extract Numbers from Trace
        # Ignore indices (0, 1) to avoid crashes
        nums_in_trace = set(re.findall(r'\b\d+\.?\d*\b', trace))
        valid_targets = []
        for n in nums_in_trace:
            try:
                val = float(n)
                # Filter: Don't swap 0, 1, or 2 (common indices/flags)
                # Unless they are explicitly 'const_2' style in source
                if val in [0, 1, 2] and "." not in n: continue
                valid_targets.append(n)
            except: continue
            
        if not valid_targets: return None

        # 2. Extract Distractors from Context
        context_text = " ".join(record.get("context_chunks", []))
        context_nums = set(re.findall(r'\b\d+\.?\d*\b', context_text))
        distractors = [n for n in context_nums if n not in nums_in_trace]
        
        # Filter distractors: Must be somewhat similar magnitude? Or just random?
        # Random is fine for logic errors.
        if not distractors: return None

        # 3. Perform Swap
        target_val = random.choice(valid_targets)
        new_val = random.choice(distractors)
        
        # Replace only the target value in the trace
        # Use simple replace if unambiguous, or regex
        new_trace = re.sub(rf'\b{re.escape(target_val)}\b', new_val, trace)
        
        if new_trace == trace: return None

        # 4. Execute
        new_result = safe_execute_trace(new_trace)
        if not new_result: return None
        
        # 5. Build Record
        # Spec: "Trace uses 150. Text says 100. Sentence matches Trace."
        new_record = copy.deepcopy(record)
        new_record["id"] = generate_sabotage_id(record["id"])
        new_record["trace_code"] = new_trace # The Hallucinated Trace
        new_record["target_sentence"] = new_result # The Hallucinated Result
        new_record["label"] = "Unfounded"
        new_record["sabotage_info"] = {
            "applied": True,
            "type": self.name,
            "diff": f"Code Input {target_val} -> {new_val}. Result -> {new_result}",
            "original_sentence": record.get("target_sentence")
        }
        return new_record