import os
import json
import itertools
from typing import Optional, Dict, Any, Tuple
import instructor
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from venra.models import RetrievalPlan
from venra.config import settings
from venra.prompt_loader import load_prompt
from venra.logging_config import logger

class Navigator:
    """
    Translates natural language queries into structured RetrievalPlans.
    Uses schema_summary.json to map user intent to canonical metrics and entities.
    Now resilient with API key pooling and tenacity retries to avoid rate limits.
    """
    def __init__(self, api_key: Optional[str] = None, file_prefix: Optional[str] = None, schema_path: Optional[str] = None):
        if schema_path:
            self.schema_path = schema_path
        elif file_prefix:
            self.schema_path = os.path.join(settings.DATA_DIR, "processed", f"{file_prefix}_schema_summary.json")
        else:
            self.schema_path = os.path.join(settings.DATA_DIR, "processed/schema_summary.json")
        
        # Key pooling for Groq from centralized config
        keys = settings.GROQ_KEYS
        if api_key and api_key not in keys:
            keys = [api_key] + keys
            
        if not keys:
            raise ValueError("No GROQ API keys found in settings.")

        instructor_clients = []
        for k in keys:
            raw = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=k,
                timeout=30.0,
            )
            instructor_clients.append(instructor.from_openai(raw, mode=instructor.Mode.JSON))

        self._client_cycle = itertools.cycle(instructor_clients)
        
        self.model = settings.SLM_MODEL_FAST
        self.full_schema = self._load_schema()
        self.system_prompt_template = load_prompt("navigator_system_prompt")

    def _next_client(self) -> instructor.AsyncInstructor:
        return next(self._client_cycle)

    def _load_schema(self) -> Dict[str, Any]:
        if not os.path.exists(self.schema_path):
            logger.warning(f"Schema summary not found at {self.schema_path}. Navigator will run without schema context.")
            return {}
        
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            self.entities = schema.get("entities", [])
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _generate_plan(self, system_prompt: str, query: str) -> RetrievalPlan:
        client = self._next_client()
        return await client.chat.completions.create(
            model=self.model,
            response_model=RetrievalPlan,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )

    async def navigate(self, query: str, doc_id: Optional[str] = None) -> RetrievalPlan:
        """
        Generates a RetrievalPlan for a given user query.
        """
        logger.info(f"Navigating query: {query} (doc_id={doc_id})")
        
        # Build filtered schema context to avoid prompt bloat
        filtered_schema = {
            "entity_ids": self.full_schema.get("entity_ids", []),
            "entities": self.full_schema.get("entities", []),
        }
        
        # FIX (Bug D): Correct Temporal Anchor Resolution per document
        doc_periods = []
        if doc_id and "periods_by_record" in self.full_schema:
            doc_periods = self.full_schema["periods_by_record"].get(doc_id, [])
        
        if not doc_periods:
            # Fallback to global if record-level fails (backward compatibility)
            doc_periods = self.full_schema.get("period_ends", [])

        filtered_schema["period_ends"] = doc_periods
        
        if doc_id and "metrics_by_record" in self.full_schema:
            filtered_schema["metrics"] = self.full_schema["metrics_by_record"].get(doc_id, [])
        else:
            # Fallback to global metrics if available
            filtered_schema["metrics"] = self.full_schema.get("metrics", [])[:100]

        schema_context_str = json.dumps(filtered_schema, indent=2)
        
        # Identify current document year from schema
        current_year = "2025" # Default if not found
        if doc_periods:
            years_found = [y for y in doc_periods if y.isdigit() and len(y) == 4]
            if years_found:
                current_year = sorted(years_found)[-1]

        last_year = str(int(current_year) - 1)
        system_prompt = self.system_prompt_template.replace("{{schema_context}}", schema_context_str)
        system_prompt = system_prompt.replace("{{current_year}}", current_year)
        system_prompt = system_prompt.replace("{{last_year}}", last_year)

        # STRICT TEMPORAL RULE: Navigator must not guess a year if none exists in query.
        # Emit years: [] if no year is mentioned.
        plan = await self._generate_plan(system_prompt, query)
        
        # BUG D GUARD (Dataset-Agnostic): 
        # Post-process to ensure any year in ufl_query was actually present 
        # or implied by the original query.
        if plan.ufl_query and plan.ufl_query.years:
            import re
            # FIX: Remove strict word boundary at start to allow 'FY2023'
            query_years = set(re.findall(r"20\d{2}\b", query))
            
            # Identify implied years (if query mentions 'current' or 'last/prior')
            # But only allow them if the anchor years were used.
            implied_years = set()
            if any(w in query.lower() for w in ["current", "recent", "latest"]):
                implied_years.add(current_year)
            if any(w in query.lower() for w in ["last year", "prior year", "previous year"]):
                implied_years.add(last_year)
                
            allowed_years = query_years | implied_years
            
            # If the user specified NO years and NO relative terms, 
            # we force years to [] regardless of SLM output.
            if not allowed_years:
                logger.warning(f"Navigator hallucinated years {plan.ufl_query.years} for query '{query}'. Clearing.")
                plan.ufl_query.years = []
            else:
                # Filter to only allowed years
                plan.ufl_query.years = [y for y in plan.ufl_query.years if y in allowed_years]

        logger.info(f"Plan generated. Reasoning: {plan.reasoning}")
        return plan
