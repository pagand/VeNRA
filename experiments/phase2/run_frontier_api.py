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
  gemini        → gemini-3-flash-preview      via google-genai
  kimi          → moonshotai/kimi-k2.5        via NVIDIA NIM
  qwen3         → qwen/qwen3-32b              via Groq
  llama70b      → llama-3.3-70b-versatile     via Groq
  gpt_oss_120b  → openai/gpt-oss-120b         via Groq

If a model is skipped entirely, compute_metrics.py silently skips it —
all other models still produce valid results.

Usage (laptop, serving .env):
  python -m experiments.phase2.run_frontier_api --model gemini
  python -m experiments.phase2.run_frontier_api --model kimi
  python -m experiments.phase2.run_frontier_api --model qwen3
  python -m experiments.phase2.run_frontier_api --model llama70b
  python -m experiments.phase2.run_frontier_api --model gpt_oss_120b
  python -m experiments.phase2.run_frontier_api --model gemini --full
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

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemini":       {"pred_key": "gemini_3_flash",    "display": "gemini-3-flash-preview"},
    "kimi":         {"pred_key": "kimi_k25_nvidia",   "display": "moonshotai/kimi-k2.5"},
    "qwen3":        {"pred_key": "qwen3_32b_groq",    "display": "qwen/qwen3-32b"},
    "llama70b":     {"pred_key": "llama33_70b_groq",  "display": "llama-3.3-70b-versatile"},
    "gpt_oss_120b": {"pred_key": "gpt_oss_120b_groq", "display": "openai/gpt-oss-120b"},
}

SEMAPHORE_LIMITS = {
    "gemini":         2,
    "kimi":           2,   # NVIDIA NIM free tier is strict; 4 causes 429s
    "qwen3":          2,
    "llama70b":       2,
    "gpt_oss_120b":   2,
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
            return raw, 0.5
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
        wait=wait_random_exponential(min=5, max=60),  # longer backoff for 429s
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
                "messages": [
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

    async def call(system_content: str, user_content: str) -> Tuple[str, float]:
        raw = await asyncio.to_thread(_sync_call, system_content, user_content)
        return raw, 0.5

    return call


# ── Groq models ───────────────────────────────────────────────────────────────

def build_groq_caller(model_id: str, prepend_no_think: bool = False):
    from openai import AsyncOpenAI

    groq_keys = settings.GROQ_KEYS
    if not groq_keys:
        raise ValueError("No GROQ_API_KEY found in .env")

    clients = [
        AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=k, timeout=30.0)
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

        conf = 0.5
        try:
            lp = response.choices[0].logprobs.content
            if lp:
                conf = math.exp(lp[0].logprob)
        except Exception:
            pass

        return raw, conf

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
    errors: List[int] = []
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
                raw, conf = await call_fn(fp["system_content"], fp["user_content"])
                t1 = time.perf_counter()
                pred, valid = parse_response(raw)
                write_prediction(out_file, {
                    "row_id":     rid,
                    "pred":       pred,
                    "valid":      valid,
                    "raw":        raw,
                    "confidence": round(conf, 6),
                    "latency_ms": round((t1 - t0) * 1000, 3),
                    "model":      model_key,
                })
                async with lock:
                    done_count += 1
                    pct = 100 * done_count / len(todo)
                    status = f"valid={valid} pred={pred}"
                    print(f"  [{done_count:>3}/{len(todo)}  {pct:5.1f}%]  "
                          f"row {rid:>4}  {t1-t0:5.1f}s  {status}")
            except Exception as e:
                t1 = time.perf_counter()
                async with lock:
                    done_count += 1
                    pct = 100 * done_count / len(todo)
                    print(f"  [{done_count:>3}/{len(todo)}  {pct:5.1f}%]  "
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

    if model_key == "gemini":
        call_fn = build_gemini_caller()
    elif model_key == "kimi":
        call_fn = build_kimi_caller()
    elif model_key == "qwen3":
        call_fn = build_groq_caller("qwen/qwen3-32b", prepend_no_think=True)
    elif model_key == "llama70b":
        call_fn = build_groq_caller("llama-3.3-70b-versatile", prepend_no_think=False)
    elif model_key == "gpt_oss_120b":
        call_fn = build_groq_caller("openai/gpt-oss-120b", prepend_no_think=False)

    manifest = load_manifest()
    frontier = load_prompts_frontier()
    out_file = PRED_FILES[cfg["pred_key"]]

    if full_run:
        rows = manifest
    else:
        rows = [
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