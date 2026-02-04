import pandas as pd
from typing import List, Dict, Any
from venra.models import UFLRow, DocBlock
from venra.prompt_loader import load_prompt
from venra.logging_config import logger

class ContextAssembler:
    """
    Step 2: Prepares the retrieved data for the Reasoning Agent.
    Handles deduplication, cross-referencing, and formatting.
    """
    def __init__(self):
        self.instructions = load_prompt("assembler_instructions")
    
    def assemble(self, retrieval_results: Dict[str, Any]) -> str:
        """
        Transforms raw retrieval results into a structured prompt context.
        """
        ufl_rows = retrieval_results.get("ufl_rows", [])
        text_chunks = retrieval_results.get("text_chunks", [])
        keywords = retrieval_results.get("meta", {}).get("vector_keywords", [])
        
        # 1. Deduplicate
        unique_rows = self._deduplicate_rows(ufl_rows)
        unique_chunks = self._deduplicate_chunks(text_chunks)
        
        # 2. Smart Filter (Limit to top 5 relevant chunks to prevent context rot)
        final_chunks = self._rank_and_filter_chunks(unique_chunks, keywords, unique_rows, limit=5)
        
        # 3. Format UFL Section
        ufl_context = self._format_ufl_table(unique_rows)
        
        # 4. Format Text Section
        text_context = self._format_text_blocks(final_chunks)
        
        # 5. Final Assembly
        full_context = f"""
# VERIFIABLE FACT LEDGER (UFL) ROWS
{ufl_context}

# SOURCE TEXT CHUNKS
{text_context}

# INSTRUCTIONS FOR REASONING:
{self.instructions}
"""
        return full_context

    def _rank_and_filter_chunks(self, chunks: List[DocBlock], keywords: List[str], ufl_rows: List[UFLRow], limit: int = 5) -> List[DocBlock]:
        if len(chunks) <= limit:
            return chunks
            
        scored_chunks = []
        ufl_source_ids = {r.source_chunk_id for r in ufl_rows if r.source_chunk_id}
        
        for chunk in chunks:
            score = 0
            # Priority 1: Linked to a retrieved UFL row (Verification Context)
            if chunk.id in ufl_source_ids:
                score += 5
            
            # Priority 2: Contains keywords (Relevance)
            content_lower = chunk.content.lower()
            for kw in keywords:
                if kw.lower() in content_lower:
                    score += 1
            
            scored_chunks.append((score, chunk))
            
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        top_chunks = [c for s, c in scored_chunks[:limit]]
        log_msg = f"Smart filtered {len(chunks)} -> {len(top_chunks)}. "
        log_msg += f"Top Scores: {[s for s, c in scored_chunks[:limit]]}. "
        log_msg += f"Selected IDs: {[c.id for c in top_chunks]}"
        logger.info(log_msg)
        
        return top_chunks

    def _deduplicate_rows(self, rows: List[UFLRow]) -> List[UFLRow]:
        seen_ids = set()
        unique = []
        for r in rows:
            if r.row_id not in seen_ids:
                unique.append(r)
                seen_ids.add(r.row_id)
        return unique

    def _deduplicate_chunks(self, chunks: List[DocBlock]) -> List[DocBlock]:
        seen_ids = set()
        unique = []
        for c in chunks:
            if c.id not in seen_ids:
                unique.append(c)
                seen_ids.add(c.id)
        return unique

    def _format_ufl_table(self, rows: List[UFLRow]) -> str:
        if not rows:
            return "No structured facts found."
            
        # Convert to DataFrame for pretty markdown printing
        df = pd.DataFrame([r.model_dump() for r in rows])
        
        # Select and reorder columns for clarity in prompt
        cols = ['row_id', 'metric_name', 'value', 'unit', 'period', 'nuance_note', 'source_chunk_id']
        # Filter for only existing columns to avoid errors
        cols = [c for c in cols if c in df.columns]
        
        table_md = df[cols].to_markdown(index=False)
        return table_md

    def _format_text_blocks(self, chunks: List[DocBlock]) -> str:
        if not chunks:
            return "No source text available."
            
        formatted = []
        for c in chunks:
            section = " > ".join(c.section_path) if c.section_path else "Unknown"
            block_str = f"""--- CHUNK_ID: {c.id} ---
SECTION: {section}
CONTENT:
{c.content}
"""
            formatted.append(block_str)
            
        return "\n".join(formatted)
