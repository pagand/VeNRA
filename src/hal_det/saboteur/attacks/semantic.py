import random
import re
import copy
from typing import Dict, Optional
from .base import BaseSaboteur
from saboteur.utils.text_utils import generate_sabotage_id, substitute_token_in_text

class SemanticSaboteur(BaseSaboteur):
    def __init__(self):
        super().__init__("semantic_drift")
        self.scale_map = {
            "million": ["billion", "thousand"],
            "billion": ["million", "trillion"],
            "percent": ["basis points"],
            "usd": ["EUR", "GBP", "shares"]
        }
        # Expanded Stopwords to prevent grammatically broken entity swaps
        self.blacklist = {
            "The", "A", "An", "In", "On", "At", "By", "For", "To", "With", "As", "But", 
            "Or", "And", "If", "When", "How", "Why", "What", "Which", "Who", "Where",
            "This", "That", "It", "They", "We", "He", "She", "These", "Those", "Some", 
            "Any", "All", "Most", "Many", "January", "February", "March", "April", 
            "May", "June", "July", "August", "September", "October", "November", "December",
            "Table", "Note", "Figure"
        }

    def attack(self, record: Dict) -> Optional[Dict]:
        meta = record.get("metadata", {})
        if "scale" in meta and meta["scale"]:
            return self._attack_scale(record, meta["scale"])
        return self._attack_entity(record)

    def _attack_scale(self, record: Dict, scale: str) -> Optional[Dict]:
        scale = scale.lower()
        if scale not in self.scale_map: return None
        new_scale = random.choice(self.scale_map[scale])
        target_sent = record.get("target_sentence", "")
        
        new_sent = substitute_token_in_text(target_sent, scale, new_scale)
        if not new_sent:
            new_sent = f"{target_sent} {new_scale}"
            
        new_record = copy.deepcopy(record)
        new_record["id"] = generate_sabotage_id(record["id"])
        new_record["target_sentence"] = new_sent
        new_record["label"] = "Unfounded"
        new_record["sabotage_info"] = {
            "applied": True, "type": self.name, "subtype": "unit_mismatch",
            "diff": f"{scale} -> {new_scale}", "original_sentence": target_sent
        }
        return new_record

    def _attack_entity(self, record: Dict) -> Optional[Dict]:
        target_sent = record.get("target_sentence", "")
        meta = record.get("metadata", {})
        target_entity = meta.get("company")
        
        if not target_entity or target_entity not in target_sent:
             caps = re.findall(r'\b[A-Z][a-z]+\b', target_sent)
             # Apply expanded blacklist and minimum length to avoid fake entities
             candidates = [w for w in caps if w not in self.blacklist and len(w) > 2]
             if not candidates: return None
             target_entity = random.choice(candidates)
             
        context = " ".join(record.get("context_chunks", []))
        ctx_caps = set(re.findall(r'\b[A-Z][a-z]+\b', context))
        distractors = [w for w in ctx_caps if w != target_entity and w not in self.blacklist and len(w) > 2]
        
        if not distractors: return None
        new_entity = random.choice(distractors)
        
        new_sent = substitute_token_in_text(target_sent, target_entity, new_entity)
        if not new_sent: return None
        
        new_record = copy.deepcopy(record)
        new_record["id"] = generate_sabotage_id(record["id"])
        new_record["target_sentence"] = new_sent
        new_record["label"] = "Unfounded"
        new_record["sabotage_info"] = {
            "applied": True, "type": self.name, "subtype": "entity_swap",
            "diff": f"{target_entity} -> {new_entity}", "original_sentence": target_sent
        }
        return new_record