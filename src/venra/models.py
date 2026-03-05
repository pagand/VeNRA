"""
venra/models.py
Pydantic data contracts for the VeNRA pipeline.

Schema versioning note:
  UFLRow     — v2.0  (UFL_spec.md flattened schema + alignment metadata)
  ScrapedFact — v2.0  (flat, alignment-aware)
  All other models unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Document Object Model (DOM)
# ---------------------------------------------------------------------------

class BlockType(str, Enum):
    TEXT  = "text"
    TABLE = "table"


class DocBlock(BaseModel):
    id: str = Field(default_factory=lambda: "")   # set by hashing content
    block_type: BlockType
    content: str
    section_path: List[str]
    page_num: Optional[int] = None


class TextBlock(DocBlock):
    block_type: BlockType = BlockType.TEXT


class TableBlock(DocBlock):
    block_type: BlockType = BlockType.TABLE


# ---------------------------------------------------------------------------
# UFL Row  (v2.0 — matches UFL_spec.md)
# ---------------------------------------------------------------------------

class UFLRow(BaseModel):
    """
    Universal Fact Ledger row. Every numeric and qualitative fact extracted
    from a document is stored in this flattened, alignment-tracked schema.

    Design constraints (UFL_spec §1):
    • Flat key-value pairs only — no nested Pydantic sub-models.
    • char_interval + alignment_status guarantee character-level grounding.
    • confidence_score drives the Runtime Fallback Protocol (§5).
    """

    # --- Identity ---
    row_id: str = Field(
        ...,
        description="Unique hash: md5(canonical_entity_id + metric_name + period_start).",
    )

    # --- Search & Graph Keys ---
    canonical_entity_id: str = Field(
        ...,
        description="Normalised entity ID (e.g. 'ID_AAPL'). Enables deterministic Pandas graph traversal.",
    )
    entity_name_raw: str = Field(
        ...,
        description="Verbatim name as it appeared in the source document (e.g. 'Apple Inc.').",
    )
    metric_name: str = Field(
        ...,
        description="Semantic key used by the Code Agent (e.g. 'Revenue', 'Gross Margin').",
    )
    related_entity_id: Optional[str] = Field(
        None,
        description="Target of a relational edge (e.g. 'ID_FOXCONN' when metric_name='Supplier').",
    )

    # --- Computation Values ---
    num_value: Optional[float] = Field(
        None,
        description="Pure float after scale applied. None/NaN for qualitative facts.",
    )
    grounding_quote: Optional[str] = Field(
        default="",
        description=(
            "Constructed by PostHocAligner. Verbatim substring from the text that justifies this fact. "
            "Can be None if alignment fails or if the SLM provides no quote."
        ),
    )
    unit_normalized: str = Field(
        default="USD",
        description="Standardised unit — no '$' or '%' symbols (e.g. 'USD', 'USD/Share', 'Ratio', 'Percent').",
    )
    scale: float = Field(
        default=1.0,
        description="Multiplier from table header context (e.g. 1e6 for 'in millions').",
    )

    # --- Temporal Context ---
    period_start: Optional[str] = Field(
        None,
        description="ISO-8601 start date (e.g. '2023-01-01'). Essential for time-series filtering.",
    )
    period_end: Optional[str] = Field(
        None,
        description="ISO-8601 end date (e.g. '2023-12-31').",
    )
    period_type: Optional[str] = Field(
        None,
        description="Period granularity tag: FY | Q1 | Q2 | Q3 | Q4 | TTM | YTD.",
    )

    # --- Provenance ---
    doc_section: str = Field(
        ...,
        description="Breadcrumb path to the source section (e.g. 'MD&A > Liquidity > Table 4').",
    )
    source_chunk_id: str = Field(
        ...,
        description="Foreign key → ChromaDB text chunk containing this fact.",
    )
    source_record_id: Optional[str] = Field(
        default=None,
        description="Original source record ID (e.g. 'finqa_CME/2012/page_73.pdf') for doc-id scoping fallback.",
    )

    # --- Nuance & Qualitative Fallback ---
    text_nuance: Optional[str] = Field(
        None,
        description=(
            "Footnotes, restatement notes, conditions, or qualitative summaries. "
            "Used by the LLM agent when num_value is None."
        ),
    )

    # --- Alignment Metadata (Post-Hoc Aligner output) ---
    char_interval: Optional[Tuple[int, int]] = Field(
        None,
        description="[start, end] character positions in the source chunk text.",
    )
    alignment_status: Literal["EXACT", "PARTIAL", "FUZZY", "UNALIGNED"] = Field(
        default="UNALIGNED",
        description=(
            "Grounding tier set by PostHocAligner: "
            "EXACT → difflib exact match; PARTIAL → substring; FUZZY → token overlap; UNALIGNED → failed."
        ),
    )
    confidence_score: float = Field(
        default=0.0,
        description=(
            "Reliability score: 0.95 for table melts, 0.70 for text extractions, "
            "0.0 if alignment_status=UNALIGNED."
        ),
    )


# ---------------------------------------------------------------------------
# Entity Registry
# ---------------------------------------------------------------------------

class EntityMetadata(BaseModel):
    canonical_id: str
    official_name: str
    cik: Optional[str] = None
    aliases: List[str] = []


# ---------------------------------------------------------------------------
# SLM Extraction Contracts  (v2.0 — flat, alignment-aware)
# ---------------------------------------------------------------------------

class ScrapedFact(BaseModel):
    """
    Intermediate extraction unit produced by TextSynthesizer.
    Flat schema reduces SLM JSON syntax errors (UFL_spec §2 Step C).

    Fields map 1-to-1 onto UFLRow so that conversion is a single
    attribute-copy without any nesting or renaming surprises.
    """

    # Core fact
    metric_name: str = Field(
        ...,
        description="The semantic name of the metric (e.g. 'Revenue', 'Operating Income').",
    )
    num_value: Optional[float] = Field(
        None,
        description=(
            "Parsed float value. Set to null for qualitative facts. "
            "Do NOT include commas, currency symbols, or scale words."
        ),
    )
    grounding_quote: Optional[str] = Field(
        default="",
        description=(
            "The exact verbatim text snippet from the source as it appeared "
            "(e.g. '$2.4 billion', '(345)', 'substantially all'). Used for alignment verification. "
            "Can be None if the fact is purely qualitative or if the model fails to extract a quote."
        ),
    )

    # Units & scale
    unit_normalized: str = Field(
        default="USD",
        description="Standardised unit string (USD | USD/Share | Ratio | Percent | Units | Other).",
    )
    scale: float = Field(
        default=1.0,
        description="Scale factor if stated in the text (e.g. 1e6 for 'in millions').",
    )

    # Temporal
    period_start: Optional[str] = Field(
        None,
        description="ISO-8601 start date if determinable, otherwise null.",
    )
    period_end: Optional[str] = Field(
        None,
        description="ISO-8601 end date if determinable, otherwise null.",
    )
    period_type: Optional[str] = Field(
        None,
        description="FY | Q1 | Q2 | Q3 | Q4 | TTM | YTD — null if ambiguous.",
    )

    # Qualitative nuance
    text_nuance: Optional[str] = Field(
        None,
        description="Footnotes, restatement conditions, or qualitative context.",
    )

    # Relational edge
    related_entity: Optional[str] = Field(
        None,
        description="Raw name of a related entity if the fact describes a relationship.",
    )

    # Confidence (LLM self-assessment; will be overridden by PostHocAligner)
    confidence: float = Field(
        ...,
        description=(
            "Self-reported confidence [0.0, 1.0]. "
            "Will be overridden by PostHocAligner based on alignment_status."
        ),
    )

    # ------------------------------------------------------------------
    # Alignment fields — populated by PostHocAligner, NOT by the SLM.
    # Defined here directly so PostHocAligner can set them via normal
    # Pydantic attribute assignment without resorting to object.__setattr__
    # hacks that silently break under Pydantic v2.
    # ------------------------------------------------------------------
    char_interval: Optional[Tuple[int, int]] = Field(
        default=None,
        description="[start, end] char positions in the source chunk. Set by PostHocAligner.",
        exclude=True,   # excluded from SLM prompt serialisation
    )
    alignment_status: Literal["EXACT", "PARTIAL", "FUZZY", "UNALIGNED"] = Field(
        default="UNALIGNED",
        description="Grounding tier assigned by PostHocAligner.",
        exclude=True,
    )
    alignment_confidence: float = Field(
        default=0.0,
        description="Final confidence after alignment. Overrides the SLM's self-reported confidence.",
        exclude=True,
    )


class FactExtractionResponse(BaseModel):
    facts: List[ScrapedFact]


# ---------------------------------------------------------------------------
# Navigator Models
# ---------------------------------------------------------------------------

class UFLFilter(BaseModel):
    """Configuration for the structured data lookup."""

    entity_ids: List[str] = Field(
        ...,
        description="Canonical IDs (e.g. 'ID_AAPL').",
    )
    metric_keywords: List[str] = Field(
        ...,
        description="Potential column headers to search for (e.g. ['Net Sales', 'Revenue']).",
    )
    years: List[str] = Field(
        ...,
        description="Specific years mentioned in the query (e.g. ['2019', '2020']). Set to an empty list [] if no specific year is requested or if the query is a trend.",
    )
    nuance_focus: Optional[str] = Field(
        None,
        description="Keywords for nuance filtering (e.g. 'Restated', 'Adjusted').",
    )


class RetrievalPlan(BaseModel):
    """Master retrieval plan generated by the Navigator SLM."""

    ufl_query: Optional[UFLFilter] = Field(
        None,
        description="Clues for the structured database lookup.",
    )
    vector_hypothesis: str = Field(
        ...,
        description=(
            "A hypothetical sentence or table header that would appear in the document "
            "containing the answer."
        ),
    )
    vector_keywords: List[str] = Field(
        ...,
        description="3-5 key search terms for BM25/keyword search.",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the translation logic.",
    )


# ---------------------------------------------------------------------------
# Sentinel / Judge Models
# ---------------------------------------------------------------------------

class VerificationRequest(BaseModel):
    query: str
    answer_text: str
    context: str = Field(
        ...,
        description="Consolidated text chunks used by the agent.",
    )
    trace: str = Field(
        ...,
        description="Computational trace (Python code + results) generated by the executor.",
    )


class SentenceVerification(BaseModel):
    sentence: str
    label: Literal["GROUNDED", "COMMON", "HALLUCINATION"]
    grounded_prob: float
    common_prob: float
    hallucination_prob: float
    explanation: Optional[str] = None


class VerificationResponse(BaseModel):
    overall_groundedness_score: float = Field(
        ...,
        description="Ratio of GROUNDED sentences to total sentences.",
    )
    sentence_results: List[SentenceVerification]