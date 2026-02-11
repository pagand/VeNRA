import re
import uuid
from typing import List, Optional

def generate_sabotage_id(original_id: str) -> str:
    return f"sabotaged_{original_id}_{str(uuid.uuid4())[:8]}"

def parse_markdown_table(markdown_text: str) -> List[List[str]]:
    """
    Parses Markdown tables robustly, handling missing outer pipes.
    """
    rows = []
    lines = markdown_text.strip().split('\n')
    for line in lines:
        if '|' not in line: 
            continue
        if '---' in line: 
            continue
            
        cells = [c.strip() for c in line.split('|')]
        
        # Cleanup leading/trailing empty strings caused by outer pipes
        if cells and cells[0] == '': cells.pop(0)
        if cells and cells[-1] == '': cells.pop()
        
        if any(cells): 
            rows.append(cells)
    return rows

def substitute_token_in_text(text: str, old_val: str, new_val: str) -> Optional[str]:
    """Safe token replacement within a sentence."""
    if not old_val or not new_val: return None
    
    pattern = re.escape(old_val)
    if old_val[0].isalnum(): pattern = r'\b' + pattern
    if old_val[-1].isalnum(): pattern = pattern + r'\b'
        
    if not re.search(pattern, text, flags=re.IGNORECASE):
        return None
        
    new_text = re.sub(pattern, new_val, text, count=1, flags=re.IGNORECASE)
    if new_text == text: return None
    return new_text