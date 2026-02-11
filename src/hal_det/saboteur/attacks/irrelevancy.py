import random
import re
import copy
from typing import Dict, Optional, List
from .base import BaseSaboteur
from saboteur.utils.text_utils import generate_sabotage_id, substitute_token_in_text

class IrrelevancySaboteur(BaseSaboteur):
    """
    Targets: FinanceBench, Phantom (Supported rows).
    Vectors:
    1. Time Warp: Change year in Query (e.g., 2020 -> 2021) while keeping 2020 context.
    2. Context Swap (Type 3): Keep Query+Answer, replace Context with unrelated chunks.
    """
    def __init__(self):
        super().__init__("irrelevancy_rag")
        self.distractor_pool = []

    def set_pool(self, pool: List[Dict]):
        """
        Populate the pool of records to draw irrelevant contexts/queries from.
        Should typically be other 'Supported' rows from different documents.
        """
        self.distractor_pool = pool

    def attack(self, record: Dict) -> Optional[Dict]:
        meta = record.get("metadata", {})
        
        # Priority 1: Time Warp (Specific to FinanceBench/Annual Reports)
        # Requires 'doc_period' or explicit years in query
        if self._can_time_warp(record):
            res = self._attack_time_warp(record)
            if res: return res
            
        # Priority 2: Context Swap (The "Non-existence" / Type 3 Attack)
        # This is a generic RAG failure mode applicable to almost any QA pair
        if self.distractor_pool:
            return self._attack_context_swap(record)
            
        return None

    def _can_time_warp(self, record: Dict) -> bool:
        query = record.get("query", "")
        # Check for 4-digit years (1990-2029)
        return bool(re.search(r'\b(19|20)\d{2}\b', query))

    def _attack_time_warp(self, record: Dict) -> Optional[Dict]:
        query = record.get("query", "")
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if not years: return None
        
        target_year = years[0]
        # Shift year by +1 or -1
        new_year = str(int(target_year) + random.choice([-1, 1]))
        
        # Use robust substitution from utils to avoid partial string replacements
        new_query = substitute_token_in_text(query, target_year, new_year)
        if not new_query: return None
        
        new_record = copy.deepcopy(record)
        new_record["id"] = generate_sabotage_id(record["id"])
        new_record["query"] = new_query
        new_record["label"] = "Unfounded"
        
        # CRITICAL: Context and Answer remain UNCHANGED.
        # This creates the hallucination: The evidence supports 2020, but the user asked for 2021.
        
        new_record["sabotage_info"] = {
            "applied": True,
            "type": self.name,
            "subtype": "time_warp",
            "diff": f"Query Year {target_year} -> {new_year}",
            "original_query": query
        }
        return new_record

    def _attack_context_swap(self, record: Dict) -> Optional[Dict]:
        """
        Type 3 Attack: Non-existence.
        Keep Query and Answer. Replace Evidence with garbage (irrelevant doc).
        """
        # Pick a distractor from a DIFFERENT document
        current_id = record.get("id")
        current_doc = record.get("metadata", {}).get("doc_name", "")
        
        distractor = None
        for _ in range(10): # Try 10 times to find a valid distractor
            cand = random.choice(self.distractor_pool)
            cand_doc = cand.get("metadata", {}).get("doc_name", "")
            
            # Ensure we don't pick the same document
            if cand["id"] != current_id and (not current_doc or cand_doc != current_doc):
                distractor = cand
                break
        
        if not distractor: return None

        new_record = copy.deepcopy(record)
        new_record["id"] = generate_sabotage_id(record["id"])
        
        # SWAP: Replace context chunks with the distractor's context
        new_record["context_chunks"] = distractor.get("context_chunks", [])
        
        # Update Label
        new_record["label"] = "Unfounded"
        
        new_record["sabotage_info"] = {
            "applied": True,
            "type": self.name,
            "subtype": "context_swap_non_existence",
            "diff": "Replaced evidence with unrelated document chunks.",
            "original_context_source": current_id,
            "distractor_source": distractor.get("id")
        }
        return new_record