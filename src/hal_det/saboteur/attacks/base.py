from typing import Dict, Optional, List
import copy

class BaseSaboteur:
    def __init__(self, name: str):
        self.name = name

    def attack(self, record: Dict) -> Optional[Dict]:
        """Returns a DEEP COPY of the record with sabotage applied, or None."""
        raise NotImplementedError