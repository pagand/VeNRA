import json
import os
from typing import List, Dict, Any
from venra.models import UFLRow, EntityMetadata
from venra.logging_config import logger

class SchemaGenerator:
    """
    Generates a schema summary for the Navigator SLM.
    This helps the SLM map natural language queries to canonical entity IDs and metric names.
    """
    def __init__(self, output_path: str = "data/processed/schema_summary.json"):
        self.output_path = output_path
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, int] = {}

    def add_entity(self, entity: EntityMetadata):
        if entity.canonical_id not in self.entities:
            self.entities[entity.canonical_id] = {
                "id": entity.canonical_id,
                "official_name": entity.official_name,
                "aliases": entity.aliases
            }
            logger.debug(f"Added entity to schema: {entity.canonical_id}")

    def add_rows(self, rows: List[UFLRow]):
        for row in rows:
            # Track metric frequency
            self.metrics[row.metric_name] = self.metrics.get(row.metric_name, 0) + 1

    def save(self):
        # Sort metrics by frequency and take top 500
        sorted_metrics = sorted(self.metrics.items(), key=lambda x: x[1], reverse=True)
        top_metrics = [m[0] for m in sorted_metrics[:500]]

        schema_data = {
            "entities": list(self.entities.values()),
            "metrics": top_metrics
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(schema_data, f, indent=2)
        
        logger.info(f"Schema summary saved to {self.output_path} ({len(top_metrics)} metrics, {len(self.entities)} entities)")

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)
