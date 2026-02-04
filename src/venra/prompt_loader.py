import os
import re
from typing import Optional
from venra.config import settings
from venra.logging_config import logger

def load_prompt(prompt_id: str) -> str:
    """
    Loads a specific prompt from assets/PROMPTS.md by its heading or ID.
    """
    try:
        if not os.path.exists(settings.PROMPTS_PATH):
            logger.error(f"Prompts file not found at {settings.PROMPTS_PATH}")
            return ""

        with open(settings.PROMPTS_PATH, "r") as f:
            content = f.read()

        # Try to find the prompt by the ID tag first: **ID:** `prompt_id`
        pattern = rf"\*\*ID:\*\* `{prompt_id}`(.*?)(?=\n##|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback to heading match if ID not found (though we added IDs)
        heading_map = {
            "extract_financial_facts": "## Text Extraction (System Prompt)",
            "navigator_system_prompt": "## Query Navigation (System Prompt)",
            "agent_pass_1_reasoning": "## Reasoning Agent: Pass 1 (Logic & Code)",
            "agent_pass_2_synthesis": "## Reasoning Agent: Pass 2 (Synthesis)",
            "assembler_instructions": "## Reasoning Instructions (Assembler Context)"
        }
        
        heading = heading_map.get(prompt_id)
        if heading:
            heading_escaped = re.escape(heading)
            pattern = rf"{heading_escaped}(.*?)(?=\n##|$)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        logger.warning(f"Prompt with ID '{prompt_id}' not found in {settings.PROMPTS_PATH}")
        return ""

    except Exception as e:
        logger.error(f"Error loading prompt '{prompt_id}': {e}")
        return ""
