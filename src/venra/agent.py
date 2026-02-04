import os
import json
import requests
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from venra.config import settings
from venra.executor import PythonExecutor
from venra.prompt_loader import load_prompt
from venra.logging_config import logger

class AgentReasoning(BaseModel):
    """The internal thought process of the agent."""
    plan: str = Field(..., description="Step-by-step logic of how to answer the query.")
    requires_math: bool = Field(..., description="Whether Python code is needed for calculation.")
    python_code: Optional[str] = Field(None, description="The code to run. Use print() to output final numbers.")
    missing_info: List[str] = Field(default_factory=list, description="Any data points expected but not found in context.")

class FinalResponse(BaseModel):
    """The final answer delivered to the user."""
    answer: str = Field(..., description="The definitive answer string. If grounded, mention chunk/row IDs in the text.")
    nuances: Optional[str] = Field(None, description="Important context found in text chunks (e.g. 'adjusted for inflation').")
    data_source_type: Literal["GROUNDED", "INTERNAL_KNOWLEDGE", "MIXED", "NOT_FOUND"] = Field(
        ..., description="The primary source of the information provided."
    )
    citations: List[str] = Field(..., description="Specific IDs of rows or chunks used.")
    groundedness_score: float = Field(..., description="0.0 to 1.0. High for document context, low for internal knowledge.")
    is_self_aware_warning: bool = Field(..., description="True if the agent is guessing, lacks data, or used internal knowledge.")

class ReasoningAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.nvidia_api_key = api_key or settings.NVIDIA_API_KEY
        self.nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.executor = PythonExecutor()
        self.kimi_model = "moonshotai/kimi-k2.5"
        
        # Fast client for second pass (Synthesis)
        self.fast_client = instructor.from_openai(
            OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY or "dummy_key"
            ),
            mode=instructor.Mode.JSON
        )
        self.pass_1_prompt = load_prompt("agent_pass_1_reasoning")
        self.pass_2_prompt = load_prompt("agent_pass_2_synthesis")

    def _call_kimi_reasoning(self, messages: List[Dict[str, str]]) -> AgentReasoning:
        """Calls Kimi for deep reasoning and code generation."""
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Accept": "application/json"
        }
        
        schema_json = json.dumps(AgentReasoning.model_json_schema(), indent=2)
        messages[0]["content"] += f"\n\nOUTPUT_SCHEMA:\n{schema_json}\n\nStrictly return valid JSON matching this schema."
        
        payload = {
            "model": self.kimi_model,
            "messages": messages,
            "temperature": 0.1,
            "chat_template_kwargs": {"thinking": True}
        }

        response = requests.post(self.nvidia_url, headers=headers, json=payload)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return AgentReasoning.model_validate_json(content)

    async def answer(self, query: str, context: str) -> Dict[str, Any]:
        logger.info(f"Agent processing query (Pass 1: Kimi Reasoning): {query}")
        
        # Pass 1: Kimi for Logic/Code
        messages_1 = [
            {"role": "system", "content": self.pass_1_prompt}, 
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY: {query}"}
        ]
        
        reasoning = self._call_kimi_reasoning(messages_1)
        logger.info(f"Reasoning Plan: {reasoning.plan}")

        code_result = None
        if reasoning.requires_math and reasoning.python_code:
            logger.info(f"Executing Python code:\n{reasoning.python_code}")
            code_result = self.executor.execute(reasoning.python_code)
            if code_result["error"]:
                logger.error(f"Code execution failed: {code_result['error']}")
            else:
                logger.info(f"Code output: {code_result['output']}")

        # Pass 2: Groq for fast Synthesis
        logger.info("Pass 2: Synthesis (Fast)")
        
        user_prompt_2 = f"""
QUERY: {query}
CONTEXT: {context}
REASONING: {reasoning.plan}
CODE_OUTPUT: {code_result['output'] if code_result else 'No code run'}
CODE_ERROR: {code_result['error'] if code_result else 'None'}
"""
        
        final = self.fast_client.chat.completions.create(
            model=settings.SLM_MODEL_PRECISION, # llama-3.3-70b
            response_model=FinalResponse,
            messages=[
                {"role": "system", "content": self.pass_2_prompt},
                {"role": "user", "content": user_prompt_2}
            ],
            temperature=0.0
        )
        
        return {
            "final_response": final,
            "reasoning": reasoning,
            "code_result": code_result
        }