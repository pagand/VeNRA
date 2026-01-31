import os
import json
import re
import hashlib
import io
from typing import List, Optional
import pandas as pd
import instructor
import chromadb
from openai import OpenAI
from venra.models import DocBlock, TableBlock, TextBlock, UFLRow, EntityMetadata, FactExtractionResponse
from venra.config import settings
from venra.logging_config import logger

class EntityResolver:
    # ... (existing code preserved)
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.groq.com/openai/v1"):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
             logger.warning("GROQ_API_KEY not found. EntityResolver might fail against real API.")
        
        self.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key=self.api_key or "dummy_key"
            ),
            mode=instructor.Mode.JSON
        )
        self.model = settings.SLM_MODEL_PRECISION

    async def resolve_entity(self, blocks: List[DocBlock]) -> EntityMetadata:
        context_text = ""
        for block in blocks[:20]:
            context_text += f"[{block.block_type.value.upper()}] Path: {block.section_path}\nContent: {block.content}\n---\n"
            
        logger.info("Resolving Entity from Cover Page context...")
        
        resp = self.client.chat.completions.create(
            model=self.model,
            response_model=EntityMetadata,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a financial data extraction engine. You will be given the raw text of a 10-K cover page. Your job is to extract the exact legal name, CIK (if present), and create a Canonical ID (e.g. 'ID_AAPL') and list of common aliases (e.g. 'The Company')."
                },
                {
                    "role": "user", 
                    "content": f"Extract Entity Metadata from this cover page content:\n\n{context_text}"
                }
            ],
            temperature=0.0
        )
        
        logger.info(f"Resolved Entity: {resp.canonical_id} ({resp.official_name})")
        return resp

class TableMelter:
    def __init__(self, entity_id: str, entity_name_raw: str = "Unknown Entity"):
        self.entity_id = entity_id
        self.entity_name_raw = entity_name_raw

    def melt(self, block: TableBlock) -> List[UFLRow]:
        logger.debug(f"Melting table from section: {block.section_path}")
        
        content = block.content.strip()
        lines = content.split("\n")
        
        clean_lines = [l for l in lines if not re.match(r"^\|?\s*:?-+:?\s*\|", l.strip())]
        
        csv_content = "\n".join([l.strip("|") for l in clean_lines])
        df = pd.read_csv(io.StringIO(csv_content), sep=r"\s*\|\s*", engine="python")
        
        df.columns = [c.strip() for c in df.columns]
        
        scale_factor = self._detect_scale(block)
        
        id_col = df.columns[0]
        period_cols = [c for c in df.columns[1:] if self._is_period_col(c)]
        
        if not period_cols:
            period_cols = [df.columns[1]] if len(df.columns) > 1 else []

        ufl_rows = []
        
        for _, row in df.iterrows():
            metric_raw = str(row[id_col]).strip()
            if not metric_raw or metric_raw.lower() == "nan":
                continue
                
            for period in period_cols:
                raw_val = str(row[period]).strip()
                val, nuance = self._parse_numeric(raw_val)
                
                # Logic Update: Handle Placeholders
                if val is None:
                    confidence = 0.0
                    scaled_val = None
                else:
                    confidence = settings.CONFIDENCE_TABLE
                    scaled_val = val * scale_factor
                
                row_id_seed = f"{self.entity_id}_{metric_raw}_{period}_{block.id}"
                row_id = hashlib.md5(row_id_seed.encode()).hexdigest()
                
                ufl_rows.append(UFLRow(
                    row_id=row_id,
                    entity_id=self.entity_id,
                    entity_name_raw=self.entity_name_raw,
                    metric_name=metric_raw,
                    value=scaled_val,
                    scale_factor=scale_factor,
                    period=period,
                    doc_section=" > ".join(block.section_path),
                    source_chunk_id=block.id,
                    nuance_note=nuance,
                    confidence=confidence
                ))
                
        return ufl_rows

    def _detect_scale(self, block: TableBlock) -> float:
        context = " ".join(block.section_path).lower() + " " + block.content.split("\n")[0].lower()
        if "millions" in context:
            return 1_000_000.0
        if "thousands" in context or "000s" in context:
            return 1_000.0
        return 1.0

    def _is_period_col(self, col_name: str) -> bool:
        return bool(re.search(r"20\d{2}", str(col_name)))

    def _parse_numeric(self, val_str: str):
        s = val_str.replace(",", "").strip()
        if not s or s in ["—", "-", "N/A", "nan"]:
            return None, None
            
        nuance = None
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
            nuance = "Negative (parentheses)"
            
        try:
            return float(s), nuance
        except ValueError:
            return None, None

class TextSynthesizer:
    def __init__(self, entity_id: str, entity_name_raw: str = "Unknown Entity", api_key: Optional[str] = None, base_url: str = "https://api.groq.com/openai/v1"):
        self.entity_id = entity_id
        self.entity_name_raw = entity_name_raw
        self.api_key = api_key or settings.GROQ_API_KEY
        
        self.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key=self.api_key or "dummy_key"
            ),
            mode=instructor.Mode.JSON
        )
        self.model = settings.SLM_MODEL_FAST

    async def extract_facts(self, block: TextBlock) -> List[UFLRow]:
        if len(block.content.strip()) < 50:
            return []
            
        logger.debug(f"Extracting facts from text chunk: {block.id}")
        
        system_prompt = (
            "You are a financial analyst. Extract atomic financial facts from the text. "
            "Return JSON matching the schema. "
            "If value is qualitative, set value=null and use nuance_note. "
            "Normalize units (e.g. $1.2B -> 1,200,000,000). "
            "Assign confidence."
        )
        
        user_content = f"Section: {block.section_path}\nText: {block.content}"
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_model=FactExtractionResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"Failed to extract facts from block {block.id}: {e}")
            return []
        
        ufl_rows = []
        for fact in resp.facts:
            if fact.confidence < settings.CONFIDENCE_TEXT_LOW:
                continue
                
            period = fact.period or "UNKNOWN"
            
            # Handle potential string values (like IDs or failed parsing)
            final_value = None
            final_nuance = fact.nuance_note
            
            if isinstance(fact.value, (int, float)):
                final_value = float(fact.value)
            elif isinstance(fact.value, str):
                try:
                    # Try to clean and parse "1,200.50"
                    clean_val = fact.value.replace(",", "").strip()
                    final_value = float(clean_val)
                except ValueError:
                    # It's an ID or qualitative string
                    final_value = None
                    if final_nuance:
                        final_nuance += f" (Raw Value: {fact.value})"
                    else:
                        final_nuance = str(fact.value)

            val_str = str(final_value) if final_value is not None else "None"
            row_id_seed = f"{self.entity_id}_{fact.metric_name}_{period}_{block.id}_{val_str}"
            row_id = hashlib.md5(row_id_seed.encode()).hexdigest()
            
            ufl_rows.append(UFLRow(
                row_id=row_id,
                entity_id=self.entity_id,
                entity_name_raw=self.entity_name_raw,
                metric_name=fact.metric_name,
                value=final_value,
                unit=fact.unit,
                scale_factor=1.0,
                period=period,
                doc_section=" > ".join(block.section_path),
                source_chunk_id=block.id,
                nuance_note=final_nuance,
                confidence=fact.confidence
            ))
            
        return ufl_rows

class ContextIndexer:
    def __init__(self, db_path: str = settings.CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=db_path)
        self.text_collection = self.client.get_or_create_collection(
            name="venra_text_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.schema_collection = self.client.get_or_create_collection(
            name="venra_metric_schema",
            metadata={"hnsw:space": "cosine"}
        )

    def index_blocks(self, blocks: List[DocBlock]):
        if not blocks:
            return
            
        documents = [b.content for b in blocks]
        ids = [b.id for b in blocks]
        metadatas = [
            {
                "block_type": b.block_type.value,
                "section_path": json.dumps(b.section_path),
                "page_num": b.page_num or 0
            } 
            for b in blocks
        ]
        
        self.text_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Indexed {len(blocks)} blocks in ChromaDB.")

    def index_ufl_schema(self, rows: List[UFLRow]):
        if not rows:
            return
            
        unique_metrics = {}
        for r in rows:
            key = f"{r.entity_id}_{r.metric_name}"
            if key not in unique_metrics:
                unique_metrics[key] = {
                    "id": hashlib.md5(key.encode()).hexdigest(),
                    "metric_name": r.metric_name,
                    "entity_id": r.entity_id
                }
        
        ids = [m['id'] for m in unique_metrics.values()]
        documents = [m['metric_name'] for m in unique_metrics.values()]
        metadatas = [
            {"entity_id": m['entity_id'], "metric_name": m['metric_name']} 
            for m in unique_metrics.values()
        ]
        
        self.schema_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Indexed {len(unique_metrics)} unique metrics for schema mapping.")

    def update_chunk_linkage(self, chunk_id: str, row_ids: List[str]):
        """
        Updates a text chunk's metadata to include the list of UFL row IDs derived from it.
        This enables the 'Bi-Directional Hook'.
        """
        if not row_ids:
            return
            
        # Chroma metadata only supports str/int/float/bool
        row_ids_json = json.dumps(row_ids)
        
        self.text_collection.update(
            ids=[chunk_id],
            metadatas=[{"contains_rows": row_ids_json}]
        )
