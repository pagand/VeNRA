import json
import asyncio
import re
import sys
import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
from rapidfuzz import fuzz
from dotenv import load_dotenv

# Absolute Path Resolution & Path Patching
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

# Load .env file from project root
load_dotenv(PROJECT_ROOT / ".env")

# Now we can import from siblings and project modules
from hal_det.teacher.client import TeacherAuditor

class AuditOrchestrator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.input_path = PROJECT_ROOT / self.config["input_file"]
        self.cache_path = PROJECT_ROOT / self.config["cache_file"]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.teacher = TeacherAuditor()
        self.cache = self._load_cache_ids()
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(self.config.get("concurrency", 5))
        
        self.stats = {
            "total_processed": 0,
            "skipped_cache": 0,
            "skipped_config_filter": 0,
            "audited": 0,
            "skipped_has_reasoning": 0,
            "status_verified": 0,
            "status_review": 0,
            "status_discarded": 0,
            "timeout_dropped": 0
        }

    def _load_config(self, path_str: str) -> dict:
        path = PROJECT_ROOT / path_str
        if not path.exists(): raise FileNotFoundError(f"Config not found at {path}")
        with open(path, "r") as f: return yaml.safe_load(f)

    def _load_cache_ids(self) -> set:
        if not self.cache_path.exists(): return set()
        with open(self.cache_path, "r") as f:
            ids = set()
            for line in f:
                if line.strip():
                    try: ids.add(json.loads(line)["id"])
                    except: continue
            return ids

    def _extract_sabotage_injection(self, diff_text: str) -> str:
        if "Swapped" in diff_text and "with" in diff_text:
            return diff_text.split("with")[-1].strip().rstrip('.')
        match = re.search(r'->\s*([^ ]+)', diff_text)
        if match:
            return match.group(1).strip().rstrip('.')
        return diff_text.split("->")[-1].strip().rstrip('.')

    def _should_audit(self, row: dict) -> tuple[bool, str, dict]:
        targets = self.config.get("audit_targets", {})
        for target_name, rules in targets.items():
            if not rules.get("enabled", False): continue
            filters = rules.get("filters", {})
            match = True
            if "label" in filters and row.get("label") != filters["label"]:
                match = False
            if "has_sabotage_info" in filters:
                has_info = bool(row.get("sabotage_info"))
                if filters["has_sabotage_info"] != has_info:
                    match = False
            if match:
                return True, target_name, rules
        return False, None, None

    def _validate_result(self, row: dict, verdict, thinking: str, rules: dict) -> tuple[bool, str]:
        mode = rules.get("validation_mode", "label_consistency")
        if mode == "strict_injection_match":
            if verdict.label != "Unfounded":
                return False, f"Teacher said {verdict.label}"
            sab_info = row.get("sabotage_info", {})
            injection_value = self._extract_sabotage_injection(sab_info.get("diff", ""))
            sab_type = sab_info.get("type", "")
            if "irrelevancy" in sab_type:
                return True, "Irrelevancy Accepted"
            span_score = fuzz.token_set_ratio(injection_value.lower(), verdict.detected_error_span.lower())
            reasoning_text = (verdict.forensic_analysis + " " + thinking).lower()
            found_in_reasoning = injection_value.lower() in reasoning_text
            if span_score > 60 or found_in_reasoning:
                return True, "Span Match"
            return False, f"Span Mismatch ({injection_value} vs {verdict.detected_error_span})"
        elif mode == "label_consistency":
            expected = row.get("label")
            if verdict.label == expected:
                return True, "Label Matched"
            return False, f"Label Mismatch (Expected {expected}, Got {verdict.label})"
        elif mode == "trust_teacher":
            if verdict.label == "Unfounded":
                return True, "Teacher Confirmed Unfounded"
            return False, "Teacher refutes hallucination (Dirty Fail)"
        elif mode == "generate_only":
            return True, "Reasoning Generated"
        return False, "Unknown Validation Mode"

    def _apply_action(self, is_valid: bool, rules: dict) -> str:
        action_on_mismatch = rules.get("action_on_mismatch", "discard")
        if is_valid:
            return "verified"
        else:
            if action_on_mismatch == "review": return "review"
            elif action_on_mismatch == "keep": return "verified" 
            else: return "discarded"

    async def writer_task(self):
        while True:
            result = await self.queue.get()
            if result is None: break
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            self.queue.task_done()

    async def process_row(self, row: dict, target_name: str, rules: dict, pbar: tqdm):
        await asyncio.sleep(0.5)
        
        # --- PRE-CHECK: Skip API if original reasoning exists (FinanceBench Test) ---
        original_reasoning = row.get("metadata", {}).get("original_reasoning")
        
        if original_reasoning and row.get("dataset_source") == "financebench_test":
            self.stats["skipped_has_reasoning"] += 1
            self.stats["audited"] += 1
            
            # --- SCIENTIFIC DATA AUGMENTATION (LIST-BASED SPLIT) ---
            # We expect original_reasoning to be a list of strings (bullet points)
            if isinstance(original_reasoning, list) and len(original_reasoning) > 0:
                # Ensure all elements are strings
                original_reasoning = [str(item) for item in original_reasoning]
                
                if len(original_reasoning) > 1:
                    # Multi-step: First N-1 are Thinking, last is Analysis
                    simulated_thinking = " ".join(original_reasoning[:-1])
                    simulated_analysis = original_reasoning[-1]
                else:
                    # Single-step: Synthesize generic process thinking to avoid redundancy
                    simulated_thinking = "I will compare the target sentence against the provided documentary evidence to verify its accuracy."
                    simulated_analysis = original_reasoning[0]
            else:
                # Fallback for old/malformed entries
                simulated_thinking = "Analyzing the evidence grounded in the context provided."
                simulated_analysis = str(original_reasoning)
            
            # Ensure period termination for clean training data
            # Cast to string first to handle edge case where simulated_analysis might be a number 
            # (though we handled list/str above, robustness is key)
            simulated_thinking = str(simulated_thinking)
            simulated_analysis = str(simulated_analysis)
            
            if not simulated_thinking.endswith('.'): simulated_thinking += "."
            if not simulated_analysis.endswith('.'): simulated_analysis += "."

            # For Unfounded cases, treat target sentence as error.
            derived_span = "N/A"
            if row["label"] == "Unfounded":
                derived_span = row.get("target_sentence", "entire_sentence")
            
            # Mock Verdict
            verdict = SimpleNamespace(
                label=row["label"],
                forensic_analysis=simulated_analysis,
                detected_error_span=derived_span,
                confidence=1.0
            )
            
            # Validate using the simulated data
            is_valid, reason = self._validate_result(row, verdict, simulated_thinking, rules)
            status = self._apply_action(is_valid, rules)
            self.stats[f"status_{status}"] += 1

            await self.queue.put({
                "id": row["id"],
                "audit_target_group": target_name,
                "status": status,
                "is_valid_logic": is_valid,
                "validation_reason": reason, 
                "is_valid_sabotage": is_valid if "sabotage" in target_name else None,
                "sabotage_type": row.get("sabotage_info", {}).get("type"),
                "teacher_label": verdict.label,
                "teacher_analysis": verdict.forensic_analysis,
                "teacher_thinking": simulated_thinking,
                "teacher_span": verdict.detected_error_span,
                "teacher_confidence": verdict.confidence
            })
            pbar.update(1)
            return

        async with self.semaphore:
            try:
                verdict, thinking = await asyncio.wait_for(
                    self.teacher.audit_sample(row), 
                    timeout=self.config.get("timeout_seconds", 300)
                )
                self.stats["audited"] += 1
                is_valid, reason = self._validate_result(row, verdict, thinking, rules)
                status = self._apply_action(is_valid, rules)
                self.stats[f"status_{status}"] += 1

                await self.queue.put({
                    "id": row["id"],
                    "audit_target_group": target_name,
                    "status": status,
                    "is_valid_logic": is_valid,
                    "validation_reason": reason,
                    "is_valid_sabotage": is_valid if "sabotage" in target_name else None,
                    "sabotage_type": row.get("sabotage_info", {}).get("type"),
                    "teacher_label": verdict.label,
                    "teacher_analysis": verdict.forensic_analysis,
                    "teacher_thinking": thinking,
                    "teacher_span": verdict.detected_error_span,
                    "teacher_confidence": verdict.confidence
                })
            except asyncio.TimeoutError:
                self.stats["timeout_dropped"] += 1
            except Exception as e:
                print(f"\n[Error] {row['id']}: {str(e)[:100]}")
            finally:
                pbar.update(1)

    async def run(self):
        writer = asyncio.create_task(self.writer_task())
        with open(self.input_path, "r") as f:
            rows = [json.loads(line) for line in f]
        
        eligible_rows = []
        cached_count = 0
        
        print(f"Analyzing {len(rows)} rows against config...")
        for r in rows:
            self.stats["total_processed"] += 1
            should_run, target_name, rules = self._should_audit(r)
            
            if not should_run:
                self.stats["skipped_config_filter"] += 1
                continue
            
            if r["id"] in self.cache:
                self.stats["skipped_cache"] += 1
                cached_count += 1
            else:
                eligible_rows.append((r, target_name, rules))
        
        total_goal = len(eligible_rows) + cached_count
        print(f"Audit Plan: {total_goal} total eligible. {cached_count} already in cache. {len(eligible_rows)} remaining.")
        
        with tqdm(total=total_goal, desc="Overall Progress") as pbar:
            pbar.update(cached_count)
            if eligible_rows:
                tasks = [self.process_row(r, t, ru, pbar) for r, t, ru in eligible_rows]
                await asyncio.gather(*tasks)

        await self.queue.join()
        await self.queue.put(None)
        await writer
        
        print("\n" + "="*30)
        print("AUDIT SUMMARY")
        print("="*30)
        for k, v in self.stats.items():
            print(f"{k.replace('_', ' ').title():<25}: {v}")

if __name__ == "__main__":
    orchestrator = AuditOrchestrator(
        config_path="src/hal_det/teacher/audit_config.yaml"
    )
    asyncio.run(orchestrator.run())