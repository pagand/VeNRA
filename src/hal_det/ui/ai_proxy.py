import os
import json
from google import genai
from google.genai import types
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- CONFIG & PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_PATH = PROJECT_ROOT / "data/training_candidates/candidate_train.jsonl"
AUDIT_CACHE_PATH = PROJECT_ROOT / "data/training_ready/audit_cache.jsonl"
AI_DECISIONS_PATH = PROJECT_ROOT / "data/training_ready/ai_audit_decisions.jsonl"
ENV_PATH = PROJECT_ROOT / ".env"
META_PROMPT_PATH = Path(__file__).parent / "meta_prompt.md"

load_dotenv(ENV_PATH)
# Model Selection
MODEL_NAME = "gemini-3-flash-preview"

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def load_meta_prompt() -> str:
    if not META_PROMPT_PATH.exists():
        return "You are a forensic financial auditor. Audit the following samples."
    return META_PROMPT_PATH.read_text(encoding="utf-8")

def get_or_create_cache(client):
    """Manages context caching for the meta-prompt using the new google.genai SDK."""
    display_name = "venra_forensic_meta_prompt"
    
    # Check for existing cache
    try:
        for c in client.caches.list():
            if c.display_name == display_name:
                print(f"🔄 Using existing cache: {c.name}")
                return c
    except Exception as e:
        print(f"⚠️ Cache list failed: {e}")

    # Create new cache
    print("✨ Creating new context cache...")
    meta_content = load_meta_prompt()
    try:
        # The new SDK uses a different config structure for caches
        cache = client.caches.create(
            model=MODEL_NAME,
            config=types.CreateCachedContentConfig(
                display_name=display_name,
                system_instruction=meta_content,
                ttl="3600s", # 60 minutes
            )
        )
        return cache
    except Exception as e:
        print(f"❌ Cache creation failed: {e}")
        return None

def prepare_payloads(target_ids: List[str]) -> List[Dict[str, Any]]:
    """Loads and filters data for the AI auditor."""
    candidates = {}
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                candidates[obj["id"]] = obj
                
    audits = {}
    if AUDIT_CACHE_PATH.exists():
        with open(AUDIT_CACHE_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                audits[obj["id"]] = obj

    payloads = []
    for rid in target_ids:
        raw = candidates.get(rid)
        audit = audits.get(rid, {})
        if not raw: continue
        
        payloads.append({
            "id": rid,
            "user_query": raw.get("inputs", {}).get("query", ""),
            "target_sentence": raw.get("target_sentence", ""),
            "trace_code": raw.get("inputs", {}).get("trace_code", ""),
            "sabotage_info": raw.get("sabotage_info", {}),
            "teacher_thinking": audit.get("teacher_thinking", ""),
            "original_label": raw.get("label", "Unknown"),
            "audit_target_group": audit.get("audit_target_group", "N/A"),
            "teacher_label": audit.get("teacher_label", "N/A")
        })
    return payloads

def run_ai_audit_batch(ids: List[str]):
    """Processes a list of IDs in batches of 5 using the new Gemini SDK."""
    client = get_client()
    if not client:
        yield {"error": "API Key missing"}
        return

    cache = get_or_create_cache(client)
    all_payloads = prepare_payloads(ids)
    
    # Process in chunks of 5
    for i in range(0, len(all_payloads), 5):
        batch = all_payloads[i:i+5]
        batch_ids = [b["id"] for b in batch]
        
        yield {"status": f"Auditing batch {i//5 + 1} ({', '.join(batch_ids[:2])}...)"}
        
        user_prompt = f"### SAMPLES FOR AUDIT:\n{json.dumps(batch, indent=2)}\n\n### TASK:\nReturn exactly {len(batch)} JSON objects in a list following the schema."
        
        try:
            # Setup config for generation
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            
            # Link cache if it exists
            if cache:
                gen_config.cached_content = cache.name
            
            # Generate content using the new SDK
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=gen_config
            )
            
            ai_results = json.loads(response.text)
            if isinstance(ai_results, dict) and "decisions" in ai_results:
                 ai_results = ai_results["decisions"]
            if not isinstance(ai_results, list):
                ai_results = [ai_results]
                
            processed_decisions = []
            for res in ai_results:
                rid = res.get("id")
                # Find original batch item for metadata
                orig = next((b for b in batch if b["id"] == rid), {})
                
                decision = {
                    "id": rid,
                    "final_label": res.get("label"),
                    "final_sentence": res.get("sentence"),
                    "final_trace": res.get("trace"),
                    "final_thinking": res.get("thinking"),
                    "final_analysis": res.get("analysis"),
                    "metadata": {
                        "original_teacher_label": orig.get("teacher_label"),
                        "sabotage_type": orig.get("sabotage_info", {}).get("type", "unknown"),
                        "is_manual_fix": False,
                        "ai_refined": True,
                        "auditor_model": MODEL_NAME,
                        "audit_target_group": orig.get("audit_target_group")
                    }
                }
                processed_decisions.append(decision)
                
            # Save to file
            with open(AI_DECISIONS_PATH, "a") as f:
                for d in processed_decisions:
                    f.write(json.dumps(d) + "\n")
            
            yield {"success": len(processed_decisions), "ids": batch_ids}
            
        except Exception as e:
            yield {"error": f"Batch failed: {str(e)}", "ids": batch_ids}

if __name__ == "__main__":
    # Test block
    test_ids = [
  "finqa_C/2010/page_229.pdf-1", "finqa_ORLY/2006/page_40.pdf-4","tatqa_c36a2e33-021f-4b18-a306-6792e0ab0c60", "tatqa_a9cb8a81-e7d8-40ca-98c4-dd486c1cf603", "finqa_ETR/2008/page_212.pdf-2",  "phantom_10k_9c24d43a", "tatqa_d78d3cc9-a160-43c6-a90b-2f718a5be567", "finqa_AWK/2018/page_172.pdf-3",  "finqa_AAPL/2013/page_78.pdf-4","finqa_STT/2014/page_69.pdf-1", "finqa_ETR/2016/page_267.pdf-4", "finqa_PNC/2011/page_74.pdf-3", "tatqa_8153e1b3-5478-4bac-8dd9-ff362d9a2385", "finqa_ANET/2015/page_155.pdf-3","finqa_JPM/2018/page_150.pdf-2",  "tatqa_9907aed5-a213-48ff-930f-c214d5930957",  "phantom_10k_98c57694", "tatqa_218932fe-fb9b-427e-8126-9e376cc1e9a6", "finqa_ADBE/2012/page_102.pdf-3", "finqa_CDW/2015/page_70.pdf-1","finqa_LMT/2012/page_44.pdf-1", "tatqa_b29f1fb0-a644-4f3d-9968-8977e5e37d4c", "finqa_PPG/2005/page_24.pdf-2", "tatqa_69a4977f-afb4-47ba-8d30-e4717132c290","finqa_LMT/2016/page_85.pdf-4", "finqa_APTV/2013/page_48.pdf-3",  "finqa_ETR/2016/page_315.pdf-4", "finqa_IP/2014/page_65.pdf-4",  "finqa_CME/2010/page_109.pdf-1", "finqa_ADBE/2018/page_66.pdf-2",  "finqa_UNP/2006/page_62.pdf-1", "finqa_IPG/2016/page_37.pdf-3",  "tatqa_e94e3b01-1ce3-4dba-8aa8-4aaff5af3381", "finqa_AON/2011/page_63.pdf-2", "finqa_AMT/2014/page_149.pdf-2", "tatqa_86329f59-3c72-4956-a7c7-20062ebeac8a","finqa_UA/2011/page_69.pdf-2", "tatqa_22ea63c7-2e41-48d8-8133-1fd9acebbb5b",  "tatqa_7eba92be-ec70-4319-b14f-fcc4bc4b7841","tatqa_335c8f44-c6f7-4276-a20b-388ce7a21bc8", "tatqa_88f850e1-3cfe-48d2-8e0c-d283f781fbbb", "tatqa_5a8662ed-26f9-4731-a127-1222ec91d79f", "finqa_LMT/2015/page_55.pdf-1",  "finqa_NCLH/2018/page_64.pdf-2","finqa_IPG/2017/page_92.pdf-1", "tatqa_91c87f4f-db59-41f6-912d-2e7bb3d2559a", "tatqa_75f49dc2-e741-4453-9900-ec12e10739fe",  "tatqa_1c54669e-2f75-4c5b-85fa-df37ad085170",  "tatqa_0de98696-bcb2-40d7-8270-5ba1ead88b7f", "tatqa_a0559d0d-da55-4f48-aa29-9f5f21bfcb7d",  "tatqa_8826b261-e835-4663-980d-c3279711f106", "finqa_CE/2014/page_32.pdf-4", "finqa_ETR/2011/page_316.pdf-2", "phantom_10k_c7a9bd80", "tatqa_f49f0e74-329a-4aa7-ad51-75e4e1639b31", "tatqa_aa380825-a5ae-4801-b954-f5d836d8aa20",  "tatqa_1ce45158-34b1-4e9b-979e-ca82f7235b20", "tatqa_a00ed9bc-a380-4753-a94b-b9359e09a87b",  "tatqa_1da5ca73-dae2-42d2-ab4c-05acc7c9e5b0",  "finqa_RE/2006/page_39.pdf-3","finqa_LMT/2012/page_72.pdf-3", "tatqa_de5f9a65-3593-4ba5-aabe-1b73301fc322","finqa_AMAT/2015/page_33.pdf-2",  "finqa_AWK/2018/page_162.pdf-4", "finqa_UNP/2016/page_52.pdf-2",  "finqa_UNP/2009/page_35.pdf-2", "finqa_DRE/2007/page_39.pdf-2", "finqa_FIS/2006/page_31.pdf-1", "finqa_AWK/2018/page_178.pdf-1",  "tatqa_3d8556f7-147f-47b7-a800-66d72add3264","phantom_10k_7663f3b1",  "tatqa_dbe4ebb6-d7d2-4a83-9258-63c36c07ac61", "finqa_ETR/2011/page_281.pdf-2",  "tatqa_6f673e7e-a8c7-4e58-80dc-ade898ba8230", "finqa_DRE/2012/page_34.pdf-2", "tatqa_e8f20aab-623a-48d1-a5e5-f794f045edb8",  "finqa_CMCSA/2015/page_64.pdf-3", "finqa_SNA/2013/page_84.pdf-2", "tatqa_e74feea3-645a-462a-bf9c-512e94fd9238",  "tatqa_bd9d571d-adc0-47a9-9dd3-d12ed5a1c1a6", "tatqa_0896ef8f-ae3a-4455-9ae0-da5084453e63",  "finqa_CME/2010/page_123.pdf-5",  "finqa_ECL/2016/page_52.pdf-1",  "tatqa_f2dec8da-0005-4583-b005-fb87ed60f0b1",  "finqa_ETR/2017/page_441.pdf-1", "finqa_IP/2005/page_27.pdf-1", "tatqa_73d0f45a-6e24-4b18-9a1b-aff1f0f075f1", "finqa_MMM/2007/page_66.pdf-2",  "finqa_AMT/2007/page_111.pdf-1", "tatqa_d3e285f4-5973-46a7-8076-b4d81e7ed476"
    ]
    
    print(f"🚀 Starting Test Audit for {len(test_ids)} IDs...")
    for result in run_ai_audit_batch(test_ids):
        if "status" in result:
            print(f"ℹ️ {result['status']}")
        elif "success" in result:
            print(f"✅ Batch complete! Processed {result['success']} items.")
        elif "error" in result:
            print(f"❌ Error: {result['error']}")

    print("🎉 Test Run Complete.")
    # To run: python -m src.hal_det.ui.ai_proxy
    # (Requires GEMINI_API_KEY in env)
