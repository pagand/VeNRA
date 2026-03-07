"""
experiments/phase2/run_frontier_api.py
----------------------------------------
Frontier model inference via API. Runs on LAPTOP (serving .env).

DEFAULT: runs on the 92-row CoT subsample (cot_subsample=True, pair pools only).
  - Same rows as run_base_cot_gpu.py → flip rate is directly comparable.
  - Natural/axiom pools absent on subsample, so composite M is not computed.
  - Cost: ~92 API calls per model (cheap / within free-tier limits).

Use --full to run all 812 test rows (needed for full composite M).

Supported models (--model flag):
  Strict (1 token, zero-shot):
    gemini25      → gemini-2.5-flash            via google-genai
    kimi          → moonshotai/kimi-k2.5        via NVIDIA NIM
    llama70b      → llama-3.3-70b-versatile     via Groq
  
  Flexible (Up to MAX_EXTRA_TOKENS, allows mandatory logic/markdown):
    gemini3       → gemini-3-flash-preview      via google-genai
    qwen3         → qwen/qwen3-32b              via Groq
    gpt_oss_120b  → openai/gpt-oss-20b          via Groq
"""

import argparse
import asyncio
import itertools
import json
import math
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from venra.config import settings

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_prompts_frontier,
    get_completed_ids, write_prediction, parse_response,
    ALL_PAIR_TAGS, PRED_FILES,
)

# ── Global Inference Configuration ────────────────────────────────────────────
# The maximum allowed tokens for flexible models that enforce invisible thought 
# blocks or markdown formatting. 100 is enough to bypass limits but small 
# enough to prevent full Chain-of-Thought cheating.
MAX_EXTRA_TOKENS = 100

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    # Original Strict Models
    "gemini25":     {"pred_key": "gemini_25_flash",   "display": "gemini-2.5-flash"},
    "kimi":         {"pred_key": "kimi_k25_nvidia",   "display": "moonshotai/kimi-k2.5"},
    "llama70b":     {"pred_key": "llama33_70b_groq",  "display": "llama-3.3-70b-versatile"},
    
    # New Flexible Models
    "gemini3":      {"pred_key": "gemini_3_flash",    "display": "gemini-3-flash-preview"},
    "qwen3":        {"pred_key": "qwen3_32b_groq",    "display": "qwen/qwen3-32b"},
    "gpt_oss_120b": {"pred_key": "gpt_oss_120b_groq", "display": "openai/gpt-oss-20b"},
}

SEMAPHORE_LIMITS = {
    "gemini25":       2,
    "kimi":           2,
    "llama70b":       2,
    "gemini3":        2,
    "qwen3":          2,
    "gpt_oss_120b":   2,
}

# ── Clean & Extract Helper ────────────────────────────────────────────────────
def clean_and_count(raw: str) -> Tuple[str, int]:
    """
    Strips <think> blocks and Markdown. Finds the verdict keyword.
    Counts 'extra' tokens (words/punctuation) generated BEFORE the verdict.
    """
    if not raw:
        return "", 0
        
    # 1. Strip <think> blocks safely
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    if "</think>" not in cleaned and "<think>" in cleaned:
        # Cut off during thinking, no verdict given
        return "", len(re.findall(r'\w+|[^\w\s]', raw))

    # 2. Strip Markdown bolding/italics
    cleaned = cleaned.replace('**', '').replace('*', '')

    # 3. Find verdict to isolate tokens generated before it
    # Maps directly to the labels parse_response looks for (including general)
    match = re.search(r'(?i)(found|supported|fake|unfounded|general)', cleaned)
    
    if match:
        # Extract text up to and including the verdict (e.g., "Label: Fake")
        text_up_to_verdict = cleaned[:match.end()]
        
        # Simple token approximation: count words and punctuation characters
        total_tokens = len(re.findall(r'\w+|[^\w\s]', text_up_to_verdict))
        
        # Extra tokens = Total tokens MINUS the 1 token used for the verdict itself
        extra_tokens = max(0, total_tokens - 1)
        
        # Final cleanup to assist the parser (remove common prefaces)
        final_clean = re.sub(r'(?i)^(label\s*:?\s*)', '', cleaned.strip()).strip()
        return final_clean, extra_tokens
    else:
        # Model failed to output a valid verdict keyword.
        final_clean = re.sub(r'(?i)^(label\s*:?\s*)', '', cleaned.strip()).strip()
        all_tokens = len(re.findall(r'\w+|[^\w\s]', cleaned))
        return final_clean, all_tokens

# ── Gemini ────────────────────────────────────────────────────────────────────

def build_gemini_caller(model_name: str, max_t: int, strict: bool):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=api_key)

    async def call(system_content: str, user_content: str) -> Tuple[str, str, float, int]:
        try:
            cfg_kwargs = {
                "system_instruction": system_content,
                "max_output_tokens": max_t,
                "temperature": 0.0,
            }
            if strict:
                try:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                except Exception:
                    pass
            
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            raw = response.text or ""
            
            if strict:
                return raw, raw, 0.5, 0
            else:
                cleaned, extra_tokens = clean_and_count(raw)
                return raw, cleaned, 0.5, extra_tokens
            
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}") from e

    return call


# ── Kimi K2.5 via NVIDIA NIM ──────────────────────────────────────────────────

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

def build_kimi_caller():
    nvidia_keys = settings.NVIDIA_KEYS
    if not nvidia_keys:
        raise ValueError("No NVIDIA_API_KEY found in .env")
    key_cycle = itertools.cycle(nvidia_keys)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=5, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _sync_call(system_content: str, user_content: str) -> str:
        key = next(key_cycle)
        resp = requests.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            json={
                "model": "moonshotai/kimi-k2.5",
                "messages":[
                    {"role": "system", "content": system_content},
                    {"role": "user",   "content": user_content},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "chat_template_kwargs": {"thinking": False},
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def call(system_content: str, user_content: str) -> Tuple[str, str, float, int]:
        raw = await asyncio.to_thread(_sync_call, system_content, user_content)
        # Kimi natively obeys strict 1-token settings. No cleaning overhead.
        return raw, raw, 0.5, 0

    return call


# ── Groq models ───────────────────────────────────────────────────────────────

def build_groq_caller(model_id: str, max_t: int, strict: bool, res_effort: str = None):
    from openai import AsyncOpenAI

    groq_keys = settings.GROQ_KEYS
    if not groq_keys:
        raise ValueError("No GROQ_API_KEY found in .env")

    clients =[
        AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=k, timeout=60.0)
        for k in groq_keys
    ]
    client_cycle = itertools.cycle(clients)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def call(system_content: str, user_content: str) -> Tuple[str, str, float, int]:
        client = next(client_cycle)

        kwargs = {
            "model": model_id,
            "messages":[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens": max_t,
            "temperature": 0.0,
        }
        
        # Apply reasoning effort if required by API constraints
        if res_effort:
            kwargs["extra_body"] = {"reasoning_effort": res_effort}

        # Logprobs only requested on strict models to prevent API crashes on reasoning wrappers
        if strict and max_t == 1 and not model_id.startswith("llama"):
            kwargs["logprobs"] = True

        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""

        conf = 0.5
        if strict and max_t == 1:
            try:
                lp = response.choices[0].logprobs.content
                if lp:
                    conf = math.exp(lp[0].logprob)
            except Exception:
                pass

        if strict:
            return raw, raw, conf, 0
        else:
            cleaned, extra_tokens = clean_and_count(raw)
            return raw, cleaned, conf, extra_tokens

    return call


# ── Async runner ──────────────────────────────────────────────────────────────

async def run_model(
    model_key:  str,
    call_fn,
    rows:       List[Dict],
    frontier:   Dict[int, Dict],
    out_file:   Path,
    sem_limit:  int,
    full_run:   bool,
) -> None:
    error_file = out_file.parent / (out_file.stem + "_errors.jsonl")

    completed = get_completed_ids(out_file)
    todo = [r for r in rows if r["row_id"] not in completed]

    scope = "full test set" if full_run else "subsample"
    print(f"[{model_key}] Scope: {scope} | "
          f"Total: {len(rows)} | Completed: {len(completed)} | Remaining: {len(todo)}")

    if not todo:
        print(f"[{model_key}] Nothing to do. Exiting.")
        return

    sem      = asyncio.Semaphore(sem_limit)
    errors: List[int] =[]
    lock     = asyncio.Lock()
    done_count = 0

    async def process_row(row: Dict) -> None:
        nonlocal done_count
        rid = row["row_id"]
        fp  = frontier.get(rid)
        if fp is None:
            async with lock:
                done_count += 1
                print(f"  [warn] row {rid} missing from frontier prompts — skipping")
            return

        async with sem:
            t0 = time.perf_counter()
            try:
                # Retrieve raw string (for saving), cleaned string (for parsing), and extra token count
                raw, cleaned, conf, extra_tokens = await call_fn(fp["system_content"], fp["user_content"])
                t1 = time.perf_counter()
                
                # Pass ONLY the cleaned text to the parser
                pred, valid = parse_response(cleaned)
                
                write_prediction(out_file, {
                    "row_id":       rid,
                    "pred":         pred,
                    "valid":        valid,
                    "raw":          raw, # Preserves exactly what the API generated
                    "confidence":   round(conf, 6),
                    "latency_ms":   round((t1 - t0) * 1000, 3),
                    "extra_tokens": extra_tokens, # Tokens used before reaching the verdict
                    "model":        model_key,
                })
                
                async with lock:
                    done_count += 1
                    pct = 100 * done_count / len(todo)
                    status = f"valid={valid} pred={pred} extra_toks={extra_tokens}"
                    print(f"[{done_count:>3}/{len(todo)}  {pct:5.1f}%]  "
                          f"row {rid:>4}  {t1-t0:5.1f}s  {status}")
            except Exception as e:
                t1 = time.perf_counter()
                async with lock:
                    done_count += 1
                    pct = 100 * done_count / len(todo)
                    print(f"[{done_count:>3}/{len(todo)}  {pct:5.1f}%]  "
                          f"row {rid:>4}  ERROR: {e}")
                    errors.append(rid)
                write_prediction(error_file, {
                    "row_id":     rid,
                    "error":      str(e),
                    "latency_ms": round((t1 - t0) * 1000, 3),
                })

    await asyncio.gather(*[process_row(r) for r in todo])

    print(f"[{model_key}] Done. Errors this run: {len(errors)}")
    if errors:
        print(f"  Failed row_ids (retried on restart): "
              f"{errors[:20]}{'...' if len(errors) > 20 else ''}")
        print(f"  Error log: {error_file}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run frontier model inference")
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_CONFIGS.keys()),
        help="Which frontier model to run",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run on all 812 test rows instead of the 92-row subsample. "
             "Required for full composite M; costs ~9x more API calls.",
    )
    args      = parser.parse_args()
    model_key = args.model
    cfg       = MODEL_CONFIGS[model_key]
    full_run  = args.full

    ensure_dirs()
    print(f"\n[run] Model  : {cfg['display']}")
    print(f"[run] Scope  : {'FULL test set (812 rows)' if full_run else 'CoT subsample (92 rows) — add --full for all 812'}")

    # Set up specialized routes based on strict vs flexible parsing constraints
    if model_key == "gemini25":
        # Strict 1-token model
        call_fn = build_gemini_caller("gemini-2.5-flash", max_t=1, strict=True)
        
    elif model_key == "kimi":
        # Native strict 1-token model
        call_fn = build_kimi_caller()
        
    elif model_key == "llama70b":
        # Strict 1-token model
        call_fn = build_groq_caller("llama-3.3-70b-versatile", max_t=1, strict=True)
        
    elif model_key == "gemini3":
        # Flexible model (needs tokens to bypass invisible thought blocks)
        call_fn = build_gemini_caller("gemini-3-flash-preview", max_t=MAX_EXTRA_TOKENS, strict=False)
        
    elif model_key == "qwen3":
        # Flexible model (needs tokens for markdown/thinking format)
        call_fn = build_groq_caller("qwen/qwen3-32b", max_t=MAX_EXTRA_TOKENS, strict=False, res_effort="none")
        
    elif model_key == "gpt_oss_120b":
        # Flexible model (needs tokens for mandatory "low" reasoning effort wrapper)
        call_fn = build_groq_caller("openai/gpt-oss-20b", max_t=MAX_EXTRA_TOKENS, strict=False, res_effort="low")

    manifest = load_manifest()
    frontier = load_prompts_frontier()
    
    # Automatically maps to exactly what compute_metrics.py expects
    out_file = PRED_FILES[cfg["pred_key"]]

    if full_run:
        rows = manifest
    else:
        rows =[
            r for r in manifest
            if r.get("cot_subsample", False) and r["pool"] in ALL_PAIR_TAGS
        ]
        if not rows:
            print("[error] No cot_subsample rows found. Run build_manifest.py first.")
            return

    print(f"[run] Output    : {out_file}")
    print(f"[run] Rows      : {len(rows)}")
    print(f"[run] Semaphore : {SEMAPHORE_LIMITS[model_key]} concurrent requests\n")

    asyncio.run(run_model(
        model_key = cfg["display"],
        call_fn   = call_fn,
        rows      = rows,
        frontier  = frontier,
        out_file  = out_file,
        sem_limit = SEMAPHORE_LIMITS[model_key],
        full_run  = full_run,
    ))

    print(f"\n[done] → {out_file}")


if __name__ == "__main__":
    main()