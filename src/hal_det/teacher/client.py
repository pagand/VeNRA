import os
import instructor
import openai
import itertools
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from .schema import AuditVerdict
from .prompts import AUDITOR_SYSTEM_PROMPT, AUDITOR_USER_PROMPT_TEMPLATE

class TeacherAuditor:
    def __init__(self):
        # --- POOL 1: OpenRouter (DeepSeek R1 Chimera) ---
        self.primary_model = "tngtech/deepseek-r1t2-chimera:free"
        self.primary_pool = self._init_pool(
            ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
            "https://openrouter.ai/api/v1",
            {"HTTP-Referer": "https://github.com/pedram/VeNRA", "X-Title": "VeNRA Hallucination Detector"}
        )

        # --- POOL 2: Groq (GPT OSS 120B) ---
        self.secondary_model = "openai/gpt-oss-120b"
        self.secondary_pool = self._init_pool(
            ["GROQ_API_KEY", "GROQ_API_KEY_2"],
            "https://api.groq.com/openai/v1"
        )

        # --- POOL 3: NVIDIA (Kimi k2.5) ---
        self.tertiary_model = "moonshotai/kimi-k2.5"
        self.tertiary_pool = self._init_pool(
            ["NVIDIA_API_KEY", "NVIDIA_API_KEY_2"],
            "https://integrate.api.nvidia.com/v1"
        )

    def _init_pool(self, env_vars: list, base_url: str, headers: dict = None):
        """Creates a rotating pool of clients for available keys."""
        clients = []
        for var in env_vars:
            key = os.getenv(var)
            if key:
                client = instructor.from_openai(
                    AsyncOpenAI(
                        base_url=base_url, 
                        api_key=key, 
                        default_headers=headers,
                        timeout=240.0  # 4 minutes max patience per request
                    ),
                    mode=instructor.Mode.JSON
                )
                clients.append(client)
        
        if not clients: return None
        return itertools.cycle(clients)

    def _get_next_client(self, pool):
        """Round-robin fetch of the next client."""
        if not pool: return None
        return next(pool)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(min=1, max=3),
        retry=retry_if_exception_type((
            openai.RateLimitError, 
            openai.APIConnectionError, 
            openai.InternalServerError
        ))
    )
    async def _call_api(self, client, model_id, messages, extra_body=None) -> tuple[AuditVerdict, str]:
        kwargs = {
            "model": model_id,
            "response_model": AuditVerdict,
            "messages": messages
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        verdict, completion = await client.chat.completions.create_with_completion(**kwargs)
        
        raw_choice = completion.choices[0]
        thinking = ""
        for attr in ['reasoning', 'reasoning_content', 'thinking']:
            if hasattr(raw_choice.message, attr):
                val = getattr(raw_choice.message, attr)
                if val:
                    thinking = val
                    break
            
        return verdict, thinking

    async def audit_sample(self, sample: dict) -> tuple[AuditVerdict, str]:
        messages = [
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": AUDITOR_USER_PROMPT_TEMPLATE.format(
                context="\n".join(sample["inputs"]["context_chunks"]),
                trace=sample["inputs"]["trace_code"],
                query=sample["inputs"]["query"],
                target=sample["target_sentence"]
            )}
        ]

        # 1. Try Primary Pool (OpenRouter)
        try:
            client = self._get_next_client(self.primary_pool)
            if not client: raise Exception("No Primary Keys")
            return await self._call_api(client, self.primary_model, messages)
        except Exception:
            # 2. Try Secondary Pool (Groq)
            try:
                client = self._get_next_client(self.secondary_pool)
                if not client: raise Exception("No Secondary Keys")
                return await self._call_api(client, self.secondary_model, messages)
            except Exception:
                # 3. Try Tertiary Pool (NVIDIA)
                client = self._get_next_client(self.tertiary_pool)
                if not client: raise Exception("No Tertiary Keys available")
                
                return await self._call_api(
                    client, 
                    self.tertiary_model, 
                    messages,
                    extra_body={"chat_template_kwargs": {"thinking": True}}
                )