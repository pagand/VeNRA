"""
experiments/phase2/run_frontier_api.py
----------------------------------------
Frontier model inference via API. Runs on LAPTOP (serving .env).
Follows agent.py patterns exactly for NVIDIA NIM and Groq.

Supported models (--model flag):
  gemini    → gemini-3-flash-preview      via google-genai
  kimi      → moonshotai/kimi-k2.5        via NVIDIA NIM (requests + asyncio.to_thread)
  qwen3     → qwen/qwen3-32b              via Groq (AsyncOpenAI)
  llama70b  → llama-3.3-70b-versatile     via Groq (AsyncOpenAI)

All write to data/exp/phase2/predictions/{model}.jsonl.
Crash-safe: completed row_ids are skipped on restart.

Usage (laptop, serving .env):
  python -m experiments.phase2.run_frontier_api --model gemini
  python -m experiments.phase2.run_frontier_api --model kimi
  python -m experiments.phase2.run_frontier_api --model qwen3
  python -m experiments.phase2.run_frontier_api --model llama70b
"""

import argparse
import asyncio
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

# Settings for key pools (follows agent.py / config.py)
from venra.config import settings

from experiments.phase2.utils import (
    ensure_dirs, load_manifest, load_prompts_frontier,
    get_completed_ids, write_prediction, parse_response,
    PRED_FILES,
)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemini":   {"pred_key": "gemini_3_flash",   "display": "gemini-3-flash-preview"},
    "kimi":     {"pred_key": "kimi_k25_nvidia",  "display": "moonshotai/kimi-k2.5"},
    "qwen3":    {"pred_key": "qwen3_32b_groq",   "display": "qwen/qwen3-32b"},
    "llama70b": {"pred_key": "llama33_70b_groq", "display": "llama-3.3-70b-versatile"},
}

# Concurrency limits per provider
SEMAPHORE_LIMITS = {
    "gemini":   20,
    "kimi":      4,
    "qwen3":     8,
    "llama70b":  8,
}


# ── Gemini ────────────────────────────────────────────────────────────────────

def build_gemini_caller():
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in .env")
    client = genai.Client(api_key=api_key)

    async def call(system_content: str, user_content: str) -> Tuple[str, float]:
        try:
            cfg_kwargs = {
                "system_instruction": system_content,
                "max_output_tokens": 1,
                "temperature": 0.0,
            }
            # Disable thinking budget if available in this SDK version
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass

            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3-flash-preview",
                contents=user_content,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            raw = response.text or ""
            return raw, 0.5  # Gemini doesn't reliably expose logprobs here
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}") from e

    return call


# ── Kimi K2.5 via NVIDIA NIM ──────────────────────────────────────────────────
# Follows agent.py _call_kimi_reasoning pattern exactly.

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

def build_kimi_caller():
    nvidia_keys = settings.NVIDIA_KEYS
    if not nvidia_keys:
        raise ValueError("No NVIDIA_API_KEY found in .env")
    key_cycle = itertools.cycle(nvidia_keys)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _sync_call(system_content: str, user_content: str) -> str:
        key = next(key_cycle)
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }
        payload = {
            "model": "moonshotai/kimi-k2.5",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False},
        }
        resp = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def call(system_content: str, user_content: str) -> Tuple[str, float]:
        raw = await asyncio.to_thread(_sync_call, system_content, user_content)
        return raw, 0.5   # NVIDIA NIM doesn't expose logprobs easily

    return call


# ── Groq models (Qwen3-32B, Llama 3.3 70B) ───────────────────────────────────

def build_groq_caller(model_id: str, prepend_no_think: bool = False):
    from openai import AsyncOpenAI

    groq_keys = settings.GROQ_KEYS
    if not groq_keys:
        raise ValueError("No GROQ_API_KEY found in .env")

    clients = [
        AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=k,
            timeout=30.0,
        )
        for k in groq_keys
    ]
    client_cycle = itertools.cycle(clients)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def call(system_content: str, user_content: str) -> Tuple[str, float]:
        client = next(client_cycle)

        # Qwen3: disable thinking with /no_think prefix (model honors this directive)
        sys_msg = ("/no_think\n" + system_content) if prepend_no_think else system_content

        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_content},
            ],
            max_tokens  = 1,
            temperature = 0.0,
            logprobs    = True,
        )
        raw = response.choices[0].message.content or ""

        # Extract confidence from logprob of first generated token
        conf = 0.5
        try:
            lp_content = response.choices[0].logprobs.content
            if lp_content:
                conf = math.exp(lp_content[0].logprob)
        except Exception:
            pass

        return raw, conf

    return call


# ── Async runner ──────────────────────────────────────────────────────────────

async def run_model(
    model_key:   str,
    call_fn,
    manifest:    List[Dict],
    frontier:    Dict[int, Dict],
    out_file:    Path,
    sem_limit:   int,
) -> None:
    # Only successful predictions go into out_file — errors go to a separate
    # error log so they are retried on restart (get_completed_ids reads out_file only).
    error_file = out_file.parent / (out_file.stem + "_errors.jsonl")

    completed = get_completed_ids(out_file)
    todo = [r for r in manifest if r["row_id"] not in completed]
    print(f"[{model_key}] Total: {len(manifest)} | "
          f"Completed: {len(completed)} | Remaining: {len(todo)}")

    if not todo:
        print(f"[{model_key}] Nothing to do. Exiting.")
        return

    sem     = asyncio.Semaphore(sem_limit)
    errors  = []
    lock    = asyncio.Lock()   # protect counter prints

    async def process_row(row: Dict) -> None:
        rid = row["row_id"]
        fp  = frontier.get(rid)
        if fp is None:
            async with lock:
                print(f"  [warn] row {rid} missing from frontier prompts — skipping")
            return

        async with sem:
            t0 = time.perf_counter()
            try:
                raw, conf = await call_fn(fp["system_content"], fp["user_content"])
                t1 = time.perf_counter()
                pred, valid = parse_response(raw)
                # Only successful calls are written to the main prediction cache
                write_prediction(out_file, {
                    "row_id":     rid,
                    "pred":       pred,
                    "valid":      valid,
                    "raw":        raw,
                    "confidence": round(conf, 6),
                    "latency_ms": round((t1 - t0) * 1000, 3),
                    "model":      model_key,
                })
            except Exception as e:
                t1 = time.perf_counter()
                err_msg = str(e)
                async with lock:
                    print(f"  [warn] row {rid} failed: {err_msg}")
                    errors.append(rid)
                # Errors go to a SEPARATE file — NOT the main cache.
                # On restart the main cache won't contain this row_id,
                # so it will be retried automatically.
                write_prediction(error_file, {
                    "row_id":     rid,
                    "error":      err_msg,
                    "latency_ms": round((t1 - t0) * 1000, 3),
                })

    tasks = [process_row(r) for r in todo]

    # Process in chunks for readable progress output
    chunk = 50
    for i in range(0, len(tasks), chunk):
        await asyncio.gather(*tasks[i : i + chunk])
        done = min(i + chunk, len(tasks))
        print(f"  [{model_key}] Progress: {done}/{len(tasks)} "
              f"({100 * done / len(tasks):.1f}%)")

    print(f"[{model_key}] Done. Errors this run: {len(errors)}")
    if errors:
        print(f"  Error row_ids (will be retried on restart): "
              f"{errors[:20]}{'...' if len(errors) > 20 else ''}")
        print(f"  Error log: {error_file}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run frontier model inference")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Which frontier model to run",
    )
    args = parser.parse_args()
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    ensure_dirs()
    print(f"\n[run] Model: {cfg['display']}  (key: {model_key})")

    # Build the appropriate caller
    if model_key == "gemini":
        call_fn = build_gemini_caller()
    elif model_key == "kimi":
        call_fn = build_kimi_caller()
    elif model_key == "qwen3":
        call_fn = build_groq_caller("qwen/qwen3-32b", prepend_no_think=True)
    elif model_key == "llama70b":
        call_fn = build_groq_caller("llama-3.3-70b-versatile", prepend_no_think=False)

    # Load data
    manifest = load_manifest()
    frontier = load_prompts_frontier()
    out_file = PRED_FILES[cfg["pred_key"]]

    print(f"[run] Output: {out_file}")
    print(f"[run] Semaphore: {SEMAPHORE_LIMITS[model_key]} concurrent requests")

    asyncio.run(run_model(
        model_key = cfg["display"],
        call_fn   = call_fn,
        manifest  = manifest,
        frontier  = frontier,
        out_file  = out_file,
        sem_limit = SEMAPHORE_LIMITS[model_key],
    ))

    print(f"\n[done] Predictions saved to: {out_file}")


if __name__ == "__main__":
    main()