import json
import uuid
import re
import requests
import pandas as pd
import io
from typing import List, Dict, Any, Optional

# ==========================================
# CONFIGURATION
# ==========================================
# We use the direct URLs you provided to ensure we get the exact schema matching your description.
URL_GOLD_JSONL = "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl"
URL_TEST_PARQUET = "https://huggingface.co/datasets/PatronusAI/financebench-test/resolve/main/data/test-00000-of-00001.parquet"

OUTPUT_PATH = "data/golden_records/financebench_normalized.jsonl"

# ==========================================
# HELPER: PROMPT PARSING (Specific to FinanceBench-Test)
# ==========================================
def parse_financebench_prompt(prompt_text: str) -> Dict[str, str]:
    """
    Parses the specific prompt format used in PatronusAI/financebench-test.
    
    Format:
    QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
    <question>
    
    --
    DOCUMENT:
    <context>
    
    --
    ANSWER:
    <answer>
    
    --
    """
    result = {"question": "", "context": "", "answer": ""}
    
    # 1. Extract Question
    # Matches content between the specific header and the first divider "--"
    q_match = re.search(
        r"QUESTION \(THIS DOES NOT COUNT AS BACKGROUND INFORMATION\):\s*\n(.*?)\n\s*--", 
        prompt_text, 
        re.DOTALL
    )
    if q_match:
        result['question'] = q_match.group(1).strip()
        
    # 2. Extract Document
    # Matches content between "DOCUMENT:" and the next divider
    d_match = re.search(
        r"DOCUMENT:\s*\n(.*?)\n\s*--", 
        prompt_text, 
        re.DOTALL
    )
    if d_match:
        result['context'] = d_match.group(1).strip()
        
    # 3. Extract Answer
    # Matches content between "ANSWER:" and the next divider
    a_match = re.search(
        r"ANSWER:\s*\n(.*?)\n\s*--", 
        prompt_text, 
        re.DOTALL
    )
    if a_match:
        result['answer'] = a_match.group(1).strip()
        
    return result

# ==========================================
# MAIN CLASS
# ==========================================

class FinanceBenchNormalizer:
    def __init__(self):
        self.output_path = OUTPUT_PATH

    def process(self):
        normalized_data = []

        # ---------------------------------------------------------
        # 1. Process GOLD Set (financebench_merged.jsonl)
        # ---------------------------------------------------------
        print(f"Downloading Gold Set from {URL_GOLD_JSONL}...")
        try:
            r = requests.get(URL_GOLD_JSONL)
            # The file is a JSONL (one JSON object per line)
            lines = r.text.strip().split('\n')
            print(f"Processing {len(lines)} Gold rows...")
            
            for line in lines:
                if not line.strip(): continue
                row = json.loads(line)
                entry = self._normalize_gold_row(row)
                if entry: normalized_data.append(entry)
                
        except Exception as e:
            print(f"CRITICAL ERROR processing Gold Set: {e}")

        # ---------------------------------------------------------
        # 2. Process TEST Set (test-00000-of-00001.parquet)
        # ---------------------------------------------------------
        print(f"Downloading Test Set from {URL_TEST_PARQUET}...")
        try:
            r = requests.get(URL_TEST_PARQUET)
            # Load parquet bytes into Pandas
            df_test = pd.read_parquet(io.BytesIO(r.content))
            print(f"Processing {len(df_test)} Test rows...")
            
            for _, row in df_test.iterrows():
                # Pandas rows can be accessed as dicts
                entry = self._normalize_test_row(row.to_dict())
                if entry: normalized_data.append(entry)
                
        except Exception as e:
            print(f"CRITICAL ERROR processing Test Set: {e}")

        # ---------------------------------------------------------
        # Save Output
        # ---------------------------------------------------------
        print(f"Total normalized rows: {len(normalized_data)}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for entry in normalized_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Saved to {self.output_path}")

    def _normalize_gold_row(self, row: Dict) -> Dict:
        """
        Maps PatronusAI/financebench (Gold) to VeNRA Schema.
        Schema Source: financebench_merged.jsonl
        """
        # 1. Extract Evidence
        # Input format: "evidence": [ { "evidence_text": "..." } ]
        context_chunks = []
        evidence_list = row.get('evidence', [])
        if isinstance(evidence_list, list):
            for item in evidence_list:
                if isinstance(item, dict) and 'evidence_text' in item:
                    context_chunks.append(item['evidence_text'])
        
        if not context_chunks:
            # Fallback (unlikely given schema, but safe)
            return None

        # 2. Metadata Extraction
        # We need doc_period (Year) and doc_name for Sabotage Stage 2 (Irrelevancy Attack)
        doc_period = row.get('doc_period')
        doc_name = row.get('doc_name', '')
        company = row.get('company', '')

        return {
            "id": f"finbench_gold_{row.get('financebench_id', str(uuid.uuid4())[:8])}",
            "dataset_source": "financebench_gold",
            "label": "Supported", # Gold set is always supported
            
            # Inputs
            "query": row.get('question', ''),
            "context_chunks": context_chunks,
            # SPEC: Use Text Analysis Token
            "trace_code": "# VERIFICATION_TYPE: TEXT_ANALYSIS\n# No calculation required. Verify based on semantic matching.",
            
            # Target
            "target_sentence": row.get('answer', ''),
            
            # Metadata for Saboteur
            "metadata": {
                "sabotage_ready": True,       # Eligible for Irrelevancy/Entity attacks
                "original_split": "gold",
                "doc_name": doc_name,
                "doc_period": str(doc_period), # Ensure string for consistency
                "company": company,
                "doc_link": row.get('doc_link', '')
            }
        }

    def _normalize_test_row(self, row: Dict) -> Dict:
        """
        Maps PatronusAI/financebench-test to VeNRA Schema.
        Schema Source: Parquet file with 'messages' and 'LABEL'.
        """
        # 1. Determine Label
        # Schema: LABEL column (string) -> "PASS" or "FAIL"
        raw_label = str(row.get('LABEL', '')).upper()
        
        if 'FAIL' in raw_label:
            venra_label = "Unfounded"
            sabotage_ready = False # Don't touch natural hallucinations
        elif 'PASS' in raw_label:
            venra_label = "Supported"
            sabotage_ready = True  # Can be used for attacks
        else:
            return None # Skip unclear labels

        # 2. Extract Data from Prompt
        # Schema: 'messages' is a list. messages[0]['content'] contains the prompt.
        messages = row.get('messages', [])
        
        # Parquet loading might make lists into numpy arrays or strings depending on engine
        # Safety check:
        if hasattr(messages, 'tolist'): messages = messages.tolist()
        
        if not messages or not isinstance(messages, list) or len(messages) < 1:
            return None
            
        # The user prompt is the first message
        user_content = messages[0].get('content', '')
        extracted = parse_financebench_prompt(user_content)

        # 3. Extract Reasoning (New Feature)
        # The assistant response (messages[1]) often contains JSON with REASONING
        original_reasoning = None
        if len(messages) > 1:
            assistant_content = messages[1].get('content', '')
            match = re.search(r'\"REASONING\":\s*(\[.*?\]),\s*\"SCORE\":', assistant_content, re.DOTALL)
            if match:
                try:
                    reasoning_raw = match.group(1)
                    # We keep it as a list to preserve the bullet points for CoT training
                    reasoning_list = eval(reasoning_raw)
                    if isinstance(reasoning_list, list):
                        original_reasoning = reasoning_list
                except:
                    # Fallback to string if eval fails
                    original_reasoning = [match.group(1)]
        
        # Validation: Ensure we actually found the parts
        if not extracted['question'] or not extracted['context'] or not extracted['answer']:
            # Log failure to parse?
            # print(f"Failed to parse prompt for row {row.get('_id')}")
            return None

        return {
            "id": f"finbench_test_{row.get('_id', str(uuid.uuid4())[:8])}",
            "dataset_source": "financebench_test",
            "label": venra_label,
            
            # Inputs
            "query": extracted['question'],
            # The context here is usually a single extracted snippet
            "context_chunks": [extracted['context']], 
            "trace_code": "# VERIFICATION_TYPE: TEXT_ANALYSIS",
            
            # Target (This is the candidate answer the model generated)
            "target_sentence": extracted['answer'],
            
            # Metadata
            "metadata": {
                "sabotage_ready": sabotage_ready,
                "original_split": "test_failure" if venra_label == "Unfounded" else "test_success",
                "original_label": raw_label,
                "original_reasoning": original_reasoning
            }
        }

if __name__ == "__main__":
    normalizer = FinanceBenchNormalizer()
    normalizer.process()