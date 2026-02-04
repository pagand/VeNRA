import os
import json
from typing import Optional, Dict, Any
import instructor
from openai import OpenAI
from venra.models import RetrievalPlan
from venra.config import settings
from venra.prompt_loader import load_prompt
from venra.logging_config import logger

class Navigator:
    """
    Translates natural language queries into structured RetrievalPlans.
    Uses schema_summary.json to map user intent to canonical metrics and entities.
    """
    def __init__(self, api_key: Optional[str] = None, file_prefix: Optional[str] = None, schema_path: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        
        if schema_path:
            self.schema_path = schema_path
        elif file_prefix:
            self.schema_path = os.path.join(settings.DATA_DIR, "processed", f"{file_prefix}_schema_summary.json")
        else:
            self.schema_path = os.path.join(settings.DATA_DIR, "processed/schema_summary.json")
        
        self.client = instructor.from_openai(
            OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key or "dummy_key"
            ),
            mode=instructor.Mode.JSON
        )
        self.model = settings.SLM_MODEL_FAST
        self.schema_context = self._load_schema()
        self.system_prompt_template = load_prompt("navigator_system_prompt")

    def _load_schema(self) -> str:
        if not os.path.exists(self.schema_path):
            logger.warning(f"Schema summary not found at {self.schema_path}. Navigator will run without schema context.")
            return "No schema available."
        
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            self.entities = schema.get("entities", [])
            return json.dumps(schema, indent=2)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return "Error loading schema."

    async def navigate(self, query: str) -> RetrievalPlan:
        """
        Generates a RetrievalPlan for a given user query.
        """
        logger.info(f"Navigating query: {query}")
        
        # Identify current document year from schema or entities
        current_year = "2025" # Default if not found
        if hasattr(self, 'entities') and self.entities:
            # Simple heuristic: most recent year found in metric names or document metadata
            pass 

        last_year = str(int(current_year) - 1)
        system_prompt = self.system_prompt_template.replace("{{schema_context}}", self.schema_context)
        system_prompt = system_prompt.replace("{{current_year}}", current_year)
        system_prompt = system_prompt.replace("{{last_year}}", last_year)

        try:
            plan = self.client.chat.completions.create(
                model=self.model,
                response_model=RetrievalPlan,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            logger.info(f"Plan generated. Reasoning: {plan.reasoning}")
            return plan
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            # Fallback plan
            return RetrievalPlan(
                ufl_query=None,
                vector_hypothesis=query,
                vector_keywords=query.split()[:5],
                reasoning=f"Navigation error: {str(e)}. Falling back to direct query search."
            )
