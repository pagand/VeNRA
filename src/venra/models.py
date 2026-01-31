import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Union
from pydantic import BaseModel, Field

class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADER = "header"

class DocBlock(BaseModel):
    """Base class for a chunk of document content."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    block_type: BlockType
    section_path: List[str] = Field(default_factory=list, description="Hierarchy stack e.g. ['MD&A', 'Liquidity']")
    page_num: Optional[int] = Field(default=None)
    content: str = Field(..., description="The raw text or markdown table")

class TextBlock(DocBlock):
    block_type: BlockType = BlockType.TEXT

class TableBlock(DocBlock):
    block_type: BlockType = BlockType.TABLE
    
class HeaderBlock(DocBlock):
    block_type: BlockType = BlockType.HEADER
    level: int

class EntityMetadata(BaseModel):
    """
    Taxonomy info extracted from the cover page.
    Ensures we have a canonical ID for the entity.
    """
    canonical_id: str = Field(..., description="Unique ID, e.g. 'ID_TDG' or 'ID_AAPL'")
    official_name: str = Field(..., description="Exact name from charter, e.g. 'TransDigm Group Incorporated'")
    cik: Optional[str] = Field(None, description="Central Index Key")
    aliases: List[str] = Field(default_factory=list, description="List of other names used in doc, e.g. ['The Company', 'TransDigm']")

class UFLRow(BaseModel):
    row_id: str = Field(..., description="Unique Hash: md5(entity + metric + period)")
    
    # --- Search & Graph Keys ---
    entity_id: str = Field(..., description="Canonical ID from Alias Map (e.g., 'ID_AAPL')")
    entity_name_raw: str = Field(..., description="The raw name as it appeared in source (e.g., 'The Company')")
    metric_name: str = Field(..., description="The raw row header (e.g., 'Net sales')")
    metric_embedding_id: Optional[str] = Field(None, description="ID for vector lookup of this metric name")
    related_entity_id: Optional[str] = Field(None, description="Target of edge. If metric is 'Supplier', this is 'FOXCONN'.")
    
    # --- Computation Values ---
    value: Optional[float] = Field(None, description="Normalized float. None/NaN if qualitative.")
    unit: str = Field(default="USD", description="Currency or Unit")
    scale_factor: float = Field(default=1.0, description="Multiplier found in header (e.g., 1e6)")
    
    # --- Context & Filtering ---
    period: str = Field(..., description="Time period (e.g., '2023-FY', '2023-Q3')")
    doc_section: str = Field(..., description="Breadcrumb path (e.g., 'MD&A > Liquidity > Table 4')")
    
    # --- Audit & Linkage ---
    source_chunk_id: str = Field(..., description="Foreign Key to ChromaDB Text Chunk")
    source_bbox: Optional[dict] = Field(None, description="Bounding box {page, x, y, w, h} for UI highlight.")
    nuance_note: Optional[str] = Field(None, description="Footnotes or conditions (e.g., 'Unaudited')")
    confidence: float = Field(..., description="0.95 for Table Melts, 0.7 for Text Extraction")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# --- Instructor Models ---

class ScrapedFact(BaseModel):
    """A single fact extracted from text."""
    metric_name: str = Field(..., description="The name of the metric or fact, e.g. 'Backlog', 'Litigation Risk'")
    value: Optional[Union[float, str]] = Field(None, description="The numerical value. Can be string if parsing fails or ID.")
    unit: str = Field(default="USD", description="Unit of the value, e.g. 'USD', 'Employees', 'Percent'")
    period: Optional[str] = Field(None, description="The time period mentioned, e.g. '2023', 'December 31, 2023'. If implicit, leave None.")
    related_entity: Optional[str] = Field(None, description="Target of relationship, e.g. 'Boeing' or 'CEO-owned VIE'")
    nuance_note: Optional[str] = Field(None, description="Context, conditions, exclusions, or the full qualitative statement.")
    confidence: float = Field(..., description="0.0 to 1.0 confidence in this extraction.")

class FactExtractionResponse(BaseModel):
    """Container for multiple facts extracted from a single text block."""
    facts: List[ScrapedFact]