import os
import json
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
from collections import Counter
import chromadb
from venra.models import RetrievalPlan, UFLRow, DocBlock, BlockType
from venra.config import settings
from venra.logging_config import logger

class DualRetriever:
    """
    Implements Step 1: Parallel retrieval from UFL and Vector Store 
    with refined Relational Expansion.
    """
    def __init__(self, file_prefix: Optional[str] = None, ufl_path: Optional[str] = None, db_path: Optional[str] = None):
        if ufl_path:
            self.ufl_path = os.path.abspath(ufl_path)
        elif file_prefix:
            self.ufl_path = os.path.join(settings.DATA_DIR, "processed", f"{file_prefix}_ufl.parquet")
        else:
            self.ufl_path = os.path.join(settings.DATA_DIR, "processed/ufl.parquet")
            
        self.db_path = db_path or settings.CHROMA_DB_PATH
        
        if os.path.exists(self.ufl_path):
            self.df = pd.read_parquet(self.ufl_path)
            logger.info(f"Retriever loaded UFL with {len(self.df)} rows.")
        else:
            self.df = pd.DataFrame()
            logger.warning(f"UFL file not found at {self.ufl_path}.")

        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.text_collection = self.chroma_client.get_or_create_collection("venra_text_chunks")

    async def retrieve(self, 
                       plan: RetrievalPlan, 
                       k: int = 4,
                       include_all_chunks_for_ufl: bool = True,
                       include_all_ufl_for_chunks: bool = True) -> Dict[str, Any]:
        """
        Executes dual retrieval with specific expansion logic.
        """
        logger.info(f"Starting retrieval for query: {plan.vector_hypothesis[:50]}... (k={k})")
        
        # 1. CORE SIMILARITY (The Foundation)
        # Always start with the chunks most similar to the Navigator's hypothesis
        selected_chunks = self._query_vector(plan.vector_hypothesis, k=k)
        
        # 1b. KEYWORD BOOST (New)
        # Also search using the extracted keywords to catch exact matches missed by semantic search
        # FORCE RECALL: Use at least k=5 for keywords to cast a wider net
        effective_k_keywords = max(k, 5)
        if plan.vector_keywords:
            keyword_query = " ".join(plan.vector_keywords)
            logger.info(f"Keyword Boost Search: '{keyword_query}' (k={effective_k_keywords})")
            keyword_chunks = self._query_vector(keyword_query, k=effective_k_keywords)
            selected_chunks.extend(keyword_chunks)

        chunk_id_map = {c.id: c for c in selected_chunks}
        
        # 2. DIRECT UFL QUERY
        # Look for candidates in the ledger based on clues
        selected_ufl_rows = self._query_ufl(plan.ufl_query) if plan.ufl_query else []
        row_id_map = {r.row_id: r for r in selected_ufl_rows}

        # 3. EXPANSION LOGIC
        
        # Expansion A: Related Entity Pivoting (Always on)
        # If UFL rows mention a related entity (e.g. "VPoC"), fetch chunks about it
        related_entities = list(set([r.related_entity_id for r in selected_ufl_rows if r.related_entity_id]))
        for entity in related_entities:
            entity_chunks = self._query_vector(f"Information about {entity}", k=2)
            for ec in entity_chunks:
                if ec.id not in chunk_id_map:
                    chunk_id_map[ec.id] = ec

        # Expansion B: UFL -> Chunk (Frequency Based)
        # If enabled, add the most common source chunks from the UFL matches
        if include_all_chunks_for_ufl and selected_ufl_rows:
            source_counts = Counter([r.source_chunk_id for r in selected_ufl_rows])
            # Filter out chunks already found via similarity
            new_candidate_ids = [cid for cid, count in source_counts.most_common() if cid not in chunk_id_map]
            # Add top 3 most frequent sources
            for cid in new_candidate_ids[:3]:
                expanded_chunk = self._fetch_chunks_by_ids([cid])
                if expanded_chunk:
                    chunk_id_map[cid] = expanded_chunk[0]

        # Expansion C: Chunk -> UFL (Completeness)
        # If enabled, fetch ALL UFL rows that were extracted from the current set of chunks
        if include_all_ufl_for_chunks and chunk_id_map:
            current_chunk_ids = list(chunk_id_map.keys())
            expanded_rows = self.df[self.df['source_chunk_id'].isin(current_chunk_ids)]
            for _, er in expanded_rows.iterrows():
                row_obj = UFLRow(**er.to_dict())
                if row_obj.row_id not in row_id_map:
                    row_id_map[row_obj.row_id] = row_obj

        final_rows = list(row_id_map.values())
        final_chunks = list(chunk_id_map.values())

        logger.info(f"Retrieval complete: {len(final_rows)} UFL rows, {len(final_chunks)} text chunks.")
        
        return {
            "ufl_rows": final_rows,
            "text_chunks": final_chunks,
            "meta": {
                "ufl_count": len(final_rows),
                "text_count": len(final_chunks),
                "vector_keywords": plan.vector_keywords
            }
        }

    def _query_ufl(self, filter_spec: Any) -> List[UFLRow]:
        if self.df.empty: return []
        mask = pd.Series(True, index=self.df.index)
        if filter_spec.entity_ids:
            mask &= self.df['entity_id'].isin(filter_spec.entity_ids)
        if filter_spec.years:
            mask &= self.df['period'].str.contains('|'.join(filter_spec.years), na=False)
            
        if filter_spec.metric_keywords:
            metric_mask = self.df['metric_name'].isin(filter_spec.metric_keywords)
            if not metric_mask.any():
                pattern = '|'.join([f".*{m}.*" for m in filter_spec.metric_keywords])
                metric_mask = self.df['metric_name'].str.contains(pattern, case=False, na=False)
            mask &= metric_mask
            
        results = self.df[mask]
        return [UFLRow(**r.to_dict()) for _, r in results.iterrows()]

    def _query_vector(self, hypothesis: str, k: int = 3) -> List[DocBlock]:
        results = self.text_collection.query(query_texts=[hypothesis], n_results=k)
        blocks = []
        if not results['ids'] or not results['ids'][0]: return []
        for i in range(len(results['ids'][0])):
            blocks.append(DocBlock(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                block_type=BlockType(results['metadatas'][0][i]['block_type']),
                section_path=json.loads(results['metadatas'][0][i]['section_path']),
                page_num=results['metadatas'][0][i].get('page_num')
            ))
        return blocks

    def _fetch_chunks_by_ids(self, ids: List[str]) -> List[DocBlock]:
        if not ids: return []
        results = self.text_collection.get(ids=ids)
        blocks = []
        for i in range(len(results['ids'])):
            blocks.append(DocBlock(
                id=results['ids'][i],
                content=results['documents'][i],
                block_type=BlockType(results['metadatas'][i]['block_type']),
                section_path=json.loads(results['metadatas'][i]['section_path']),
                page_num=results['metadatas'][i].get('page_num')
            ))
        return blocks