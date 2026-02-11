import json
import uuid
import re
import os
import io
import zipfile
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

# ==========================================
# CONFIGURATION
# ==========================================
FINQA_ARCHIVE_URL = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
OUTPUT_DIR = "data/golden_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finqa_normalized.jsonl")

# ==========================================
# HELPER: DATA DOWNLOADER
# ==========================================
def download_and_extract_finqa():
    """
    Downloads the official FinQA repo to ensure we get the 'program' field 
    which is often missing in simplified HF dataset versions.
    """
    print(f"Downloading raw data from {FINQA_ARCHIVE_URL}...")
    r = requests.get(FINQA_ARCHIVE_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # We only care about the JSON files in the dataset folder
    data = []
    for split in ['train', 'dev', 'test']:
        filename = f"FinQA-main/dataset/{split}.json"
        try:
            with z.open(filename) as f:
                content = json.load(f)
                print(f"Loaded {split}: {len(content)} rows")
                data.extend(content)
        except KeyError:
            print(f"Warning: Could not find {filename} in zip.")
            
    return data

# ==========================================
# HELPER: TEXT PROCESSING
# ==========================================

def clean_number(val: Any) -> str:
    """Cleans financial number strings for Python math."""
    val = str(val).strip()
    if not val: return "0"
    
    # Handle percentages (10% -> 0.10)
    if val.endswith('%'):
        try:
            return str(float(val.replace('%', '').replace(',', '')) / 100)
        except:
            pass
            
    # Handle negative in parenthesis (500) -> -500
    if val.startswith('(') and val.endswith(')'):
        val = '-' + val[1:-1]
        
    return re.sub(r'[,$€£]', '', val)

def table_to_markdown(table_rows: List[List[str]]) -> str:
    """
    Converts FinQA list-of-lists table to Markdown.
    Assumes row[0] is the header.
    """
    if not table_rows or len(table_rows) < 1:
        return ""
    
    header = table_rows[0]
    data = table_rows[1:]
    
    # Sanitize header
    header = [str(h).replace('\n', ' ').strip() for h in header]
    
    # Build MD
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    for row in data:
        clean_row = [str(c).replace('\n', ' ').strip() for c in row]
        # Handle mismatched row lengths
        if len(clean_row) < len(header):
            clean_row += [""] * (len(header) - len(clean_row))
        md += "| " + " | ".join(clean_row[:len(header)]) + " |\n"
        
    return md

# ==========================================
# HELPER: TRACE TRANSPILER (The "Logic Core")
# ==========================================

def parse_finqa_program(program_input: Any) -> List[Dict]:
    """
    Parses FinQA program which might be a list of strings or a single string 
    into a structured list of dicts: [{'op': 'add', 'arg1': '1', 'arg2': '2'}]
    """
    steps = []
    
    # Normalize input to list of strings
    if isinstance(program_input, str):
        # "add(1, 2), divide(3, 4)" -> ["add(1, 2)", "divide(3, 4)"]
        # Warning: This naive split might fail if args contain commas inside quotes (not common in FinQA)
        # Using a regex split to handle function calls better is safer if needed, but FinQA structure is simple.
        # But wait, arguments CAN contain commas? e.g. large numbers?
        # The debug output showed: 'divide(100, 100), divide(3.8, #0)'
        # 'multiply(607, 18.13), multiply(#0, const_1000)'
        # Splitting by '), ' is safer.
        program_input = program_input.replace('), ', ')|') # Temporary delimiter
        raw_steps = program_input.split('|')
    elif isinstance(program_input, list):
        raw_steps = program_input
    else:
        return []

    for step_str in raw_steps:
        if isinstance(step_str, dict):
            steps.append(step_str) # Already structured
            continue
            
        # Parse string: "op(arg1, arg2)"
        step_str = str(step_str).strip()
        match = re.match(r'([a-zA-Z0-9_]+)\((.*)\)', step_str)
        if match:
            op = match.group(1)
            args_str = match.group(2)
            # Split args. Handle "table_sum(header, row)" vs "add(1, 2)"
            # FinQA args are simple usually.
            args = [a.strip() for a in args_str.split(',')]
            
            step_dict = {'op': op}
            if len(args) > 0: step_dict['arg1'] = args[0]
            if len(args) > 1: step_dict['arg2'] = args[1]
            if len(args) > 2: step_dict['arg3'] = args[2] # Rare
            
            steps.append(step_dict)
            
    return steps

def transpile_program(program_raw: Any) -> str:
    """
    Converts FinQA LISP steps to Python Trace.
    Input: [{'op': 'subtract', 'arg1': 'const_100', 'arg2': '#0'}] 
           OR "subtract(const_100, #0)"
    Output: step_1 = 100 - step_0
    """
    program_steps = parse_finqa_program(program_raw)
    
    if not program_steps:
        return "# NO_TRACE_AVAILABLE"

    code_lines = []
    
    # Mapping FinQA ops to Python
    op_map = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "exp": "**",
        "greater": ">",
        "less": "<"
    }

    try:
        for i, step in enumerate(program_steps):
            op = step.get('op') or step.get('function') # Handle key variations
            arg1 = step.get('arg1')
            arg2 = step.get('arg2')

            # --- Argument Resolver ---
            def resolve(arg):
                if arg is None: return "0"
                s = str(arg).strip()
                # Reference to previous step (#0)
                if s.startswith('#') and s[1:].isdigit():
                    return f"step_{s[1:]}"
                # Constants
                if "const_" in s:
                    if "const_100" in s: return "100"
                    if "const_1" in s: return "1"
                    if "const_0" in s: return "0"
                    return clean_number(s.replace("const_", ""))
                # Table lookups (FinQA specific, usually just indices)
                # If it's just a number string, return it
                return clean_number(s)

            val1 = resolve(arg1)
            val2 = resolve(arg2)
            
            # --- Logic Generation ---
            line = ""
            if op in op_map:
                line = f"step_{i} = {val1} {op_map[op]} {val2}"
            elif op in ["table_sum", "table_average", "table_max", "table_min"]:
                # These ops in FinQA take table indices. 
                # We map them to a pseudo-code representation for the Judge.
                # e.g., table_sum(header, row_index)
                func_name = op.replace("table_", "") # sum, average
                if func_name == "average": func_name = "mean" # pythonic
                line = f"step_{i} = {func_name}([table_value({val1}), table_value({val2})])" 
            else:
                line = f"step_{i} = {op}({val1}, {val2})"
                
            code_lines.append(line)
        
        # Add final result print
        if code_lines:
            last_var = f"step_{len(program_steps)-1}"
            code_lines.append(f"print({last_var})")
            
        return "\n".join(code_lines)

    except Exception as e:
        return f"# TRACE_GENERATION_ERROR: {str(e)}"

# ==========================================
# MAIN CLASS
# ==========================================

class FinQANormalizer:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def process(self):
        # 1. Load Raw Data (Bypassing incomplete HF datasets)
        raw_data = download_and_extract_finqa()
        
        normalized_data = []
        
        print("Normalizing data...")
        for row in raw_data:
            norm_entry = self._normalize_row(row)
            if norm_entry:
                normalized_data.append(norm_entry)
                
        # 2. Save to JSONL
        print(f"Saving {len(normalized_data)} rows to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in normalized_data:
                f.write(json.dumps(entry) + '\n')
        print("Done.")

    def _normalize_row(self, row: Dict) -> Optional[Dict]:
        """
        Maps raw FinQA JSON to VeNRA Schema.
        """
        try:
            # 1. Basic Extraction
            # FinQA structure: 'qa' dict contains the question/program
            qa = row.get('qa', {})
            question = qa.get('question', '')
            program = qa.get('program', []) # Raw program (string or list)
            answer = qa.get('exe_ans', '') # The calculated answer
            
            # ID Handling
            finqa_id = row.get('id', str(uuid.uuid4()))
            
            # 2. Context Flattening (Pre + Table + Post)
            context_chunks = []
            
            # Add Pre-text
            if row.get('pre_text'):
                context_chunks.extend(row['pre_text'])
                
            # Add Table (Markdown)
            if row.get('table'):
                md_table = table_to_markdown(row['table'])
                if md_table:
                    context_chunks.append(md_table)
            
            # Add Post-text
            if row.get('post_text'):
                context_chunks.extend(row['post_text'])

            # 3. Trace Generation (CRITICAL for Sabotage)
            trace_code = transpile_program(program)
            
            # Filter: If we failed to generate a trace, this row is less useful for Logic Sabotage
            # But still useful for generic training. We keep it but flag metadata.
            has_trace = "# TRACE_GENERATION_ERROR" not in trace_code and "# NO_TRACE_AVAILABLE" not in trace_code

            # 4. Metadata for Saboteur
            # Extract numbers from answer to help Saboteur find replacements
            original_vals = re.findall(r'-?\d*\.?\d+', str(answer))

            return {
                "id": f"finqa_{finqa_id}",
                "dataset_source": "finqa",
                "label": "Supported",
                
                # The Core VeNRA Inputs
                "query": question,
                "context_chunks": context_chunks,
                "trace_code": trace_code,
                "target_sentence": str(answer),
                
                # Metadata for Stage 2 (Sabotage)
                "metadata": {
                    "original_values": original_vals,
                    "sabotage_ready": True,       # Eligible for sabotage?
                    "has_logic_trace": has_trace, # Eligible for Logic Sabotage?
                    "raw_program": program        # Keep raw just in case
                }
            }
        except Exception as e:
            # print(f"Skipping row {row.get('id')}: {e}")
            return None

if __name__ == "__main__":
    normalizer = FinQANormalizer()
    normalizer.process()