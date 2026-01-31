import os
import json
import re
import hashlib
import io
from typing import List, Optional, Dict, Any
import pandas as pd
import instructor
import chromadb
from openai import OpenAI
from pydantic import BaseModel
from venra.models import DocBlock, TableBlock, TextBlock, UFLRow, EntityMetadata, FactExtractionResponse, ScrapedFact
from venra.config import settings
from venra.logging_config import logger

class EntityResolver:
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
        """
        Analyzes the first few blocks (Cover Page) to extract canonical entity info.
        """
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
    def __init__(self, entity_id: str, entity_name_raw: str = "Unknown Entity", api_key: Optional[str] = None):
        self.entity_id = entity_id
        self.entity_name_raw = entity_name_raw
        self.api_key = api_key or settings.GROQ_API_KEY
        
        # Initialize SLM for header normalization
        self.client = instructor.from_openai(
            OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key or "dummy_key"
            ),
            mode=instructor.Mode.JSON
        )

    def melt(self, block: TableBlock) -> List[UFLRow]:
        content = block.content.strip()
        lines = content.split("\n")
        
        # 1. Hierarchical Disambiguation (Track parents by indentation)
        hierarchy_lines = []
        parent_stack = [] # List of names
        
        for line in lines:
            if not line.strip() or re.match(r"^\|?\s*:?-+:?\s*\|", line.strip()):
                hierarchy_lines.append(line)
                continue
                
            # Detect indentation: count leading &nbsp; or spaces after the first optional |
            indent_match = re.match(r"\|?\s*((?:&nbsp;|\s)*)([^|]+)", line)
            if indent_match:
                indent_str = indent_match.group(1)
                metric_text = indent_match.group(2).strip()
                
                if not metric_text or metric_text.lower() in ["item", "value", "metric"]:
                    hierarchy_lines.append(line)
                    continue

                # Depth calculation: 1 &nbsp; or 2 spaces = 1 depth unit
                normalized_indent = indent_str.replace("&nbsp;", "  ")
                depth = len(normalized_indent) // 2
                
                # Pop stack to correct depth
                parent_stack = parent_stack[:depth]
                
                # Construct full name
                if parent_stack:
                    full_name = " > ".join(parent_stack + [metric_text])
                else:
                    full_name = metric_text
                
                # Robust replacement: Split by pipe, replace the first content segment
                # Strip leading and trailing pipes for splitting
                inner_content = line.strip().strip("|")
                parts = [p.strip() for p in inner_content.split("|")]
                
                # Heuristic: If all subsequent columns are empty, it's a parent
                is_parent = len(parts) > 1 and all(not p for p in parts[1:])
                if is_parent or len(parts) == 1:
                    parent_stack.append(metric_text)
                
                parts[0] = full_name
                new_line = "| " + " | ".join(parts) + " |"
                hierarchy_lines.append(new_line)
            else:
                hierarchy_lines.append(line)

        # 2. Cleanup and DataFrame Conversion
        clean_lines = [l for l in hierarchy_lines if not re.match(r"^\|?\s*:?-+:?\s*\|", l.strip())]
        csv_content = "\n".join([l.strip("|") for l in clean_lines])
        
        try:
            df = pd.read_csv(io.StringIO(csv_content), sep=r"\s*\|\s*", engine="python")
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            logger.error(f"Pandas parsing failed: {e}")
            return []
        
        table_scale_factor = self._detect_scale(block)
        id_col = df.columns[0]
        period_cols = [c for c in df.columns[1:] if self._is_period_col(c)]
        if not period_cols:
            period_cols = [df.columns[1]] if len(df.columns) > 1 else []

        ufl_rows = []
        for _, row in df.iterrows():
            metric_raw = str(row[id_col]).strip()
            if not metric_raw or metric_raw.lower() == "nan":
                continue
            
            metric_clean = re.sub(r"\s*\([\d\w]+\)", "", metric_raw).strip()
            row_scale_factor = table_scale_factor
            unit = "USD"
            if any(kw in metric_clean.lower() for kw in ["per share", "eps"]):
                row_scale_factor = 1.0
                unit = "USD/Share"
            elif any(kw in metric_clean.lower() for kw in ["ratio", "percentage", "margin"]):
                row_scale_factor = 1.0
                unit = "Ratio"
                
            for period in period_cols:
                raw_val = str(row[period]).strip()
                val, nuance = self._parse_numeric(raw_val)
                scaled_val = val * row_scale_factor if val is not None else None
                
                if val is None:
                    confidence = 0.0
                else:
                    confidence = settings.CONFIDENCE_TABLE
                
                if "restated" in period.lower():
                    nuance = (nuance + " (Restated)") if nuance else "Restated"

                row_id_seed = f"{self.entity_id}_{metric_clean}_{period}_{block.id}_{scaled_val}"
                row_id = hashlib.md5(row_id_seed.encode()).hexdigest()
                
                ufl_rows.append(UFLRow(
                    row_id=row_id,
                    entity_id=self.entity_id,
                    entity_name_raw=self.entity_name_raw,
                    metric_name=metric_clean,
                    value=scaled_val,
                    unit=unit,
                    scale_factor=row_scale_factor,
                    period=period,
                    doc_section=" > ".join(block.section_path),
                    source_chunk_id=block.id,
                    nuance_note=nuance,
                    confidence=confidence
                ))
                
        return ufl_rows

    async def normalize_headers_with_slm(self, headers: List[str]) -> Dict[str, str]:
        """
        Uses SLM to normalize messy column headers to ISO 8601 dates.
        """
        prompt = f"Convert these financial table column headers into ISO 8601 dates (YYYY-MM-DD) or standardized period names. Headers: {headers}"
        
        class HeaderMap(BaseModel):
            mapping: Dict[str, str]

        resp = self.client.chat.completions.create(
            model=settings.SLM_MODEL_FAST,
            response_model=HeaderMap,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.mapping

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
        s = val_str.strip().replace("&nbsp;", " ")
        s = s.strip()
        if s in ["—", "-", "–"]:
            return 0.0, "Dash treated as zero"
        
        # Check for negative in parens BEFORE footnote stripping
        is_neg_parens = False
        if s.startswith("(") and s.endswith(")"):
            is_neg_parens = True
            s = s[1:-1]

        s = re.sub(r"\s*\([\d\w]+\)", "", s)
        s = re.sub(r"([0-9])[a-z]$", r"\1", s)
        s = s.replace(",", "").replace("$", "").strip()
        
        if not s or s.lower() in ["n/a", "nan"]:
            return None, None
        
        if is_neg_parens:
            s = "-" + s
            nuance = "Negative (parentheses)"
        else:
            nuance = None
            
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
        
        try:
            with open(settings.PROMPTS_PATH, "r") as f:
                content = f.read()
                match = re.search(r"## Text Extraction \(System Prompt\)(.*?)(##|$)", content, re.DOTALL)
                if match:
                    self.prompt_template = match.group(1).strip()
                else:
                    self.prompt_template = content
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {settings.PROMPTS_PATH}. Using default.")
            self.prompt_template = "You are a financial analyst. Extract facts from: {{text_content}}"

    async def extract_facts(self, block: TextBlock, context_str: str = "", model_name: Optional[str] = None) -> List[UFLRow]:
        if len(block.content.strip()) < 10:
            return []
            
        target_model = model_name or self.model
        filled_prompt = self.prompt_template.replace("{{section_path}}", str(block.section_path))
        filled_prompt = filled_prompt.replace("{{context_str}}", context_str)
        filled_prompt = filled_prompt.replace("{{text_content}}", block.content)
        
        try:
            resp = self.client.chat.completions.create(
                model=target_model,
                response_model=FactExtractionResponse,
                messages=[
                    {"role": "system", "content": filled_prompt},
                    {"role": "user", "content": "Extract facts."}
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
            final_value = None
            final_nuance = fact.nuance_note
            if isinstance(fact.value, (int, float)):
                final_value = float(fact.value)
            elif isinstance(fact.value, str):
                try:
                    clean_val = fact.value.replace(",", "").strip()
                    final_value = float(clean_val)
                except ValueError:
                    final_value = None
                    final_nuance = (final_nuance + f" (Raw Value: {fact.value})") if final_nuance else str(fact.value)

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
                confidence=fact.confidence,
                related_entity_id=getattr(fact, "related_entity", None)
            ))
            
        return ufl_rows

class ContextIndexer:
    def __init__(self, db_path: str = settings.CHROMA_DB_PATH, embedding_fn: Optional[Any] = None):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_fn
        self.text_collection = self.client.get_or_create_collection(
            name="venra_text_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.schema_collection = self.client.get_or_create_collection(
            name="venra_metric_schema",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )

    def index_blocks(self, blocks: List[DocBlock]):
        if not blocks: return
        documents = [b.content for b in blocks]
        ids = [b.id for b in blocks]
        metadatas = [{"block_type": b.block_type.value, "section_path": json.dumps(b.section_path), "page_num": b.page_num or 0} for b in blocks]
        self.text_collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(blocks)} blocks in ChromaDB.")

    def index_ufl_schema(self, rows: List[UFLRow]):
        if not rows: return
        unique_metrics = {}
        for r in rows:
            key = f"{r.entity_id}_{r.metric_name}"
            if key not in unique_metrics:
                unique_metrics[key] = {"id": hashlib.md5(key.encode()).hexdigest(), "metric_name": r.metric_name, "entity_id": r.entity_id}
        ids = [m['id'] for m in unique_metrics.values()]
        documents = [m['metric_name'] for m in unique_metrics.values()]
        metadatas = [{"entity_id": m['entity_id'], "metric_name": m['metric_name']} for m in unique_metrics.values()]
        self.schema_collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(unique_metrics)} unique metrics for schema mapping.")

    def update_chunk_linkage(self, chunk_id: str, row_ids: List[str]):
        if not row_ids: return
        self.text_collection.update(ids=[chunk_id], metadatas=[{"contains_rows": json.dumps(row_ids)}])
