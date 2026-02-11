import json
import uuid
import re
import os
import requests
from typing import List, Dict

# ==========================================
# CONFIGURATION
# ==========================================
# TAT-QA official raw dataset URL
TATQA_URL = "https://github.com/NExTplusplus/TAT-QA/raw/master/dataset_raw/tatqa_dataset_test_gold.json"
OUTPUT_DIR = "data/golden_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tatqa_normalized_test_gold.jsonl")

# ==========================================
# HELPER: TEXT & TABLE PROCESSING
# ==========================================

def table_to_markdown(header: List[str], rows: List[List[str]]) -> str:
    """Converts 2D list table to Markdown."""
    if not header and not rows: return ""
    
    # TAT-QA tables usually come as a single 2D list where row[0] is header
    # But sometimes the extraction splits them. We handle both.
    
    # Sanitize header
    clean_header = [str(h).replace('\n', ' ').strip() for h in header]
    
    md = "| " + " | ".join(clean_header) + " |\n"
    md += "| " + " | ".join(["---"] * len(clean_header)) + " |\n"
    
    for row in rows:
        clean_row = [str(c).replace('\n', ' ').strip() for c in row]
        # Handle mismatched row lengths
        if len(clean_row) < len(clean_header):
            clean_row += [""] * (len(clean_header) - len(clean_row))
        md += "| " + " | ".join(clean_row[:len(clean_header)]) + " |\n"
        
    return md

def clean_for_python(val: str) -> str:
    """
    Cleans a specific number string for Python math.
    $1,000 -> 1000
    10% -> 0.10
    """
    val = str(val).strip()
    
    # Handle Percentage
    if val.endswith('%'):
        try:
            num = float(val.replace('%', '').replace(',', ''))
            return str(num / 100)
        except:
            pass
            
    # Remove currency and commas
    val = re.sub(r'[,$£€]', '', val)
    return val

def transpile_derivation(derivation: str) -> str:
    """
    Converts TAT-QA derivation strings into valid Python.
    Input: "1,204.5 - 10%"
    Output: print(1204.5 - 0.10)
    """
    if not derivation:
        return "# NO_DERIVATION"

    # Split by operators to clean individual numbers
    # Operators: +, -, *, /, (, )
    tokens = re.split(r'(\+|\-|\*|\/|\(|\))', derivation)
    
    clean_tokens = []
    for t in tokens:
        t = t.strip()
        if not t: continue
        
        if t in ['+', '-', '*', '/', '(', ')']:
            clean_tokens.append(t)
        else:
            # It's a number (or should be)
            clean_tokens.append(clean_for_python(t))
            
    expression = " ".join(clean_tokens)
    
    # Wrap in simple Python script
    return f"result = {expression}\nprint(result)"

# ==========================================
# MAIN NORMALIZER CLASS
# ==========================================

class TATQANormalizer:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def download_data(self) -> List[Dict]:
        """Downloads the raw JSON directly from GitHub."""
        print(f"Downloading TAT-QA from {TATQA_URL}...")
        try:
            r = requests.get(TATQA_URL)
            if r.status_code != 200:
                raise Exception(f"Failed to download: {r.status_code}")
            return r.json()
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please ensure you have internet access or place 'tat_qa_dataset.json' manually.")
            return []

    def process(self):
        raw_data = self.download_data()
        normalized_data = []

        print(f"Processing {len(raw_data)} TAT-QA pages...")
        
        for page in raw_data:
            # TAT-QA structure: 1 Page = Table + Paragraphs + List of Questions
            extracted_rows = self._normalize_page(page)
            normalized_data.extend(extracted_rows)

        print(f"Total normalized rows generated: {len(normalized_data)}")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in normalized_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved to {OUTPUT_FILE}")

    def _normalize_page(self, page: Dict) -> List[Dict]:
        """
        Takes one TAT-QA page object and extracts multiple VeNRA rows (one per question).
        """
        venra_rows = []
        
        # 1. Context Assembly (Table + Paragraphs)
        context_chunks = []
        raw_table_data = [] # 2D List for Metadata
        
        # Process Table
        # TAT-QA table is key 'table' -> 'table': [[...]]
        if 'table' in page and page['table']:
            t_data = page['table']['table'] # Access nested table list
            raw_table_data = t_data
            
            if t_data and len(t_data) > 0:
                header = t_data[0]
                rows = t_data[1:]
                md_table = table_to_markdown(header, rows)
                context_chunks.append(md_table)
                
        # Process Paragraphs
        if 'paragraphs' in page and page['paragraphs']:
            for p in page['paragraphs']:
                # paragraphs can be objects or strings
                if isinstance(p, dict):
                    context_chunks.append(p.get('text', ''))
                else:
                    context_chunks.append(str(p))

        # 2. Question Iteration
        questions = page.get('questions', [])
        
        for q in questions:
            # Filter: We only want rows that have an answer
            if not q.get('answer'): continue

            # --- Trace Generation ---
            answer_type = q.get('answer_type', 'span')
            derivation = q.get('derivation', '')
            
            trace_code = ""
            if answer_type == 'arithmetic' and derivation:
                trace_code = transpile_derivation(derivation)
            elif answer_type in ['span', 'multi-span']:
                trace_code = "# VERIFICATION_TYPE: LOOKUP\n# The answer is explicitly stated in the text/table."
            elif answer_type == 'count':
                # e.g., counting items in a list
                trace_code = "# VERIFICATION_TYPE: COUNTING\n# result = len(matches)"
            else:
                trace_code = "# VERIFICATION_TYPE: TEXT_ANALYSIS"

            # --- Target Sentence ---
            # Answers are lists in TAT-QA (e.g., ["10", "20"]). Join them.
            ans_list = q.get('answer')
            if isinstance(ans_list, list):
                target_sentence = ", ".join([str(x) for x in ans_list])
            else:
                target_sentence = str(ans_list)

            # --- Construct VeNRA Object ---
            venra_entry = {
                "id": f"tatqa_{q.get('uid', str(uuid.uuid4()))}",
                "dataset_source": "tatqa",
                "label": "Supported",
                
                # Core Inputs
                "query": q.get('question'),
                "context_chunks": context_chunks,
                "trace_code": trace_code,
                "target_sentence": target_sentence,
                
                # METADATA (Critical for Sabotage)
                "metadata": {
                    # Unit Mismatch Sabotage needs 'scale' (e.g., "million")
                    "scale": q.get('scale', ''), 
                    
                    # Neighbor Trap Sabotage needs the raw table structure
                    "raw_table": raw_table_data,
                    
                    # Filter helpers
                    "answer_type": answer_type,
                    "sabotage_ready": True
                }
            }
            venra_rows.append(venra_entry)
            
        return venra_rows

if __name__ == "__main__":
    normalizer = TATQANormalizer()
    normalizer.process()