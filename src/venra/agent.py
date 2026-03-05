import os
import json
import requests
import itertools
import asyncio
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
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
        self.nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.executor = PythonExecutor()
        self.kimi_model = "moonshotai/kimi-k2.5"
        
        # NVIDIA API Key Pooling from centralized config
        nvidia_keys = settings.NVIDIA_KEYS
        if api_key and api_key not in nvidia_keys:
            nvidia_keys = [api_key] + nvidia_keys
            
        if not nvidia_keys:
            raise ValueError("No NVIDIA API keys found in settings.")
        self._nvidia_key_cycle = itertools.cycle(nvidia_keys)
        
        # Groq API Key Pooling for Fast Synthesis from centralized config
        groq_keys = settings.GROQ_KEYS
        if not groq_keys:
            raise ValueError("No GROQ API keys found in settings.")
            
        instructor_clients = []
        for k in groq_keys:
            raw = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=k,
                timeout=30.0,
            )
            instructor_clients.append(instructor.from_openai(raw, mode=instructor.Mode.JSON))
            
        self._groq_client_cycle = itertools.cycle(instructor_clients)

        self.pass_1_prompt = load_prompt("agent_pass_1_reasoning")
        self.pass_2_prompt = load_prompt("agent_pass_2_synthesis")

    def _next_nvidia_key(self) -> str:
        return next(self._nvidia_key_cycle)
        
    def _next_groq_client(self) -> instructor.AsyncInstructor:
        return next(self._groq_client_cycle)

    @retry(
        stop=stop_after_attempt(3), # Reduced from 5 to 3 to fail faster if truly broken
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _call_kimi_reasoning(self, messages: List[Dict[str, str]]) -> AgentReasoning:
        """Calls Kimi for deep reasoning and code generation."""
        key = self._next_nvidia_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json"
        }
        
        # Make a copy of messages to avoid appending multiple times on retry
        messages_copy = [dict(m) for m in messages]
        
        schema_json = json.dumps(AgentReasoning.model_json_schema(), indent=2)
        messages_copy[0]["content"] += f"\n\nOUTPUT_SCHEMA:\n{schema_json}\n\nStrictly return valid JSON matching this schema."
        
        payload = {
            "model": self.kimi_model,
            "messages": messages_copy,
            "temperature": 0.1,
            "chat_template_kwargs": {"thinking": True}
        }

        logger.info(f"Pass 1: Requesting deep reasoning from {self.kimi_model} (Timeout: 300s)...")
        try:
            response = requests.post(self.nvidia_url, headers=headers, json=payload, timeout=300.0)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"NVIDIA API Error: {str(e)}")
            raise

        resp_json = response.json()
        message = resp_json["choices"][0]["message"]
        content = message.get("content", "")
        
        # Capture NIM-specific reasoning if available
        thinking = message.get("reasoning") or message.get("reasoning_content")
        if thinking:
            # We log it at debug level to avoid cluttering but keep it for traceability
            logger.debug(f"Kimi Thinking Trace: {thinking}")
            
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return AgentReasoning.model_validate_json(content)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_fast_synthesis(self, user_prompt_2: str) -> FinalResponse:
        client = self._next_groq_client()
        return await client.chat.completions.create(
            model=settings.SLM_MODEL_PRECISION, # llama-3.3-70b
            response_model=FinalResponse,
            messages=[
                {"role": "system", "content": self.pass_2_prompt},
                {"role": "user", "content": user_prompt_2}
            ],
            temperature=0.0
        )

    async def answer(
        self, 
        query: str, 
        context: str, 
        reasoning_mode: Literal["CODE", "COT"] = "CODE"
    ) -> Dict[str, Any]:
        logger.info(f"Agent processing query (Pass 1: Kimi Reasoning, Mode: {reasoning_mode}): {query}")
        
        # Select prompt based on mode
        prompt_key = "agent_pass_1_reasoning" if reasoning_mode == "CODE" else "agent_pass_1_cot"
        pass_1_prompt = load_prompt(prompt_key) or self.pass_1_prompt

        # Pass 1: Kimi for Logic/Code or Pure CoT
        messages_1 = [
            {"role": "system", "content": pass_1_prompt}, 
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY: {query}"}
        ]
        
        # Call Kimi in a thread to not block the event loop
        reasoning = await asyncio.to_thread(self._call_kimi_reasoning, messages_1)
        logger.info(f"Reasoning Plan: {reasoning.plan}")

        code_result = None
        # Only execute math if in CODE mode and the model requested it
        if reasoning_mode == "CODE" and reasoning.requires_math and reasoning.python_code:
            logger.info(f"Executing Python code:\n{reasoning.python_code}")
            code_result = self.executor.execute(reasoning.python_code)
            if code_result["error"]:
                logger.error(f"Code execution failed: {code_result['error']}")
            else:
                logger.info(f"Code output: {code_result['output']}")

        # Pass 2: Groq for fast Synthesis
        if settings.ENABLE_AGENT_SYNTHESIS:
            logger.info("Pass 2: Synthesis (Fast)")
            
            # Context for synthesis
            code_output = code_result['output'] if code_result else 'No code run'
            if reasoning_mode == "COT":
                code_output = "N/A (Chain-of-Thought Mode)"

            user_prompt_2 = f"""
QUERY: {query}
CONTEXT: {context}
REASONING: {reasoning.plan}
CODE_OUTPUT: {code_output}
CODE_ERROR: {code_result['error'] if code_result else 'None'}
"""
            final = await self._call_fast_synthesis(user_prompt_2)
        else:
            logger.info("Pass 2: Synthesis Skipped (Config)")
            
            # Determine best available answer
            if code_result and code_result.get("output"):
                # If code ran, the output is the most precise answer
                raw_answer = str(code_result["output"])
            else:
                # Fallback to the plan if no code was executed
                raw_answer = f"[Synthesis Skipped] Reasoning Plan: {reasoning.plan}"

            final = FinalResponse(
                answer=raw_answer,
                nuances=None,
                data_source_type="MIXED", # Default as we skipped verification
                citations=[],
                groundedness_score=-1.0, # Indicator for skipped scoring
                is_self_aware_warning=False
            )
        
        return {
            "final_response": final,
            "reasoning": reasoning,
            "code_result": code_result
        }
