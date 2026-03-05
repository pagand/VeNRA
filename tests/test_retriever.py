import pytest
import pandas as pd
import json
from unittest.mock import MagicMock, patch
from venra.retriever import DualRetriever
from venra.models import RetrievalPlan, UFLRow, DocBlock, BlockType, UFLFilter

@pytest.fixture
def mock_ufl_df():
    data = [
        {
            "row_id": "r1", "canonical_entity_id": "ID_TDG", "entity_name_raw": "TransDigm",
            "metric_name": "Net Sales", "num_value": 100.0, "unit_normalized": "USD", "period_end": "2023",
            "doc_section": "Financials", "source_chunk_id": "c1", "confidence_score": 1.0,
            "related_entity_id": None
        },
        {
            "row_id": "r2", "canonical_entity_id": "ID_TDG", "entity_name_raw": "TransDigm",
            "metric_name": "Acquisition", "num_value": 50.0, "unit_normalized": "USD", "period_end": "2023",
            "doc_section": "Notes", "source_chunk_id": "c2", "confidence_score": 1.0,
            "related_entity_id": "Boeing"
        }
    ]
    return pd.DataFrame(data)

@pytest.mark.asyncio
async def test_retriever_direct_and_expansion(mock_ufl_df):
    """
    Tests that Retriever fetches both UFL and Vector data, and handles Expansion.
    We use a realistic mock document that satisfies the Lexical Gate recall requirement.
    """
    plan = RetrievalPlan(
        strategy="HYBRID",
        ufl_query=UFLFilter(
            entity_ids=["ID_TDG"],
            metric_keywords=["Net Sales"],
            years=["2023"]
        ),
        vector_hypothesis="The company reported net sales of $100 million for the fiscal year 2023.",
        vector_keywords=["net", "sales"],
        reasoning="Test"
    )

    # Mock ChromaDB and OS path check
    with patch("venra.retriever.chromadb.PersistentClient") as mock_chroma, \
         patch("os.path.exists") as mock_exists, \
         patch("pandas.read_parquet") as mock_read_parquet:
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ufl_df
        
        mock_collection = MagicMock()
        # Use get_or_create_collection to match DualRetriever.__init__
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock vector search result (Chunk c3)
        # Content updated to realistically match the hypothesis and pass the 0.30 recall threshold.
        mock_collection.query.return_value = {
            "ids": [["c3"]],
            "documents": [["The company reported net sales of $100 million for the fiscal year 2023."]],
            "metadatas": [[{
                "block_type": "text",
                "section_path": json.dumps(["MD&A"]),
                "page_num": 10
            }]]
        }
        
        # Mock ID fetch for expansion (Chunk c1 for row r1)
        mock_collection.get.return_value = {
            "ids": ["c1"],
            "documents": ["The raw table text for sales"],
            "metadatas": [{
                "block_type": "table",
                "section_path": json.dumps(["Financials"]),
                "page_num": 5
            }]
        }

        retriever = DualRetriever(ufl_path="fake.parquet")
        
        # Scenario: include_all_chunks_for_ufl=True
        results = await retriever.retrieve(plan, include_all_chunks_for_ufl=True)
        
        # Assertions
        assert len(results["ufl_rows"]) == 1
        assert results["ufl_rows"][0].metric_name == "Net Sales"
        
        # Should have 2 chunks: one from vector search (c3), one from row expansion (c1)
        assert len(results["text_chunks"]) == 2
        chunk_ids = [c.id for c in results["text_chunks"]]
        assert "c3" in chunk_ids
        assert "c1" in chunk_ids

@pytest.mark.asyncio
async def test_retriever_subchunk_expansion_fix(mock_ufl_df):
    """
    Regression Test: Ensures that UFL rows with sub-chunk suffixes (e.g. _f820)
    are correctly retrieved by Expansion C when the parent ID is in ChromaDB.
    """
    # Create a mock DF where the row is linked to a sub-chunk
    suffixed_df = mock_ufl_df.copy()
    suffixed_df.loc[0, "source_chunk_id"] = "c3_f820"
    
    plan = RetrievalPlan(
        strategy="TEXT_ONLY",
        ufl_query=None,
        vector_hypothesis="The company reported net sales of $100 million.",
        vector_keywords=["sales"],
        reasoning="Test"
    )

    with patch("venra.retriever.chromadb.PersistentClient") as mock_chroma, \
         patch("os.path.exists") as mock_exists, \
         patch("pandas.read_parquet") as mock_read_parquet:
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = suffixed_df
        
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        # Vector search returns parent ID 'c3'
        mock_collection.query.return_value = {
            "ids": [["c3"]],
            "documents": [["The company reported net sales of $100 million."]],
            "metadatas": [[{
                "block_type": "text",
                "section_path": json.dumps(["MD&A"]),
                "page_num": 10
            }]]
        }

        retriever = DualRetriever(ufl_path="fake.parquet")
        
        # Expansion C should now strip '_f820' and match 'c3'
        results = await retriever.retrieve(plan, include_all_ufl_for_chunks=True)
        
        # Assertions
        assert len(results["ufl_rows"]) == 1
        assert results["ufl_rows"][0].source_chunk_id == "c3_f820"

@pytest.mark.asyncio
async def test_retriever_entity_fallback_fix(mock_ufl_df):
    """
    Regression Test: Ensures that if a ticker mismatch occurs (e.g. ID_VZ vs ID_VERIZON),
    the retriever falls back to record-scoping (source_record_id) if doc_id is provided.
    """
    # Add record ID to mock data
    scoped_df = mock_ufl_df.copy()
    scoped_df["source_record_id"] = "doc_123"
    scoped_df["canonical_entity_id"] = "ID_VERIZON" # Registry has full name
    
    # Navigator asks for Ticker ID
    plan = RetrievalPlan(
        strategy="UFL_ONLY",
        ufl_query=UFLFilter(
            entity_ids=["ID_VZ"], # Ticker ID
            metric_keywords=["Net Sales"],
            years=["2023"]
        ),
        vector_hypothesis="Test",
        vector_keywords=["test"],
        reasoning="Test"
    )

    with patch("venra.retriever.chromadb.PersistentClient"), \
         patch("os.path.exists") as mock_exists, \
         patch("pandas.read_parquet") as mock_read_parquet:
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = scoped_df
        retriever = DualRetriever(ufl_path="fake.parquet")
        
        # Should return rows because even though ID_VZ != ID_VERIZON, 
        # the record ID 'doc_123' matches.
        results = await retriever.retrieve(plan, doc_id="doc_123")
        
        assert len(results["ufl_rows"]) == 1
        assert results["ufl_rows"][0].canonical_entity_id == "ID_VERIZON"

@pytest.mark.asyncio
async def test_retriever_fuzzy_metric_fallback(mock_ufl_df):
    """
    Test that if exact metric fails, fuzzy substring search works.
    Fuzzy matches are gated by the tokens of the metric keywords themselves.
    """
    plan = RetrievalPlan(
        strategy="UFL_ONLY",
        ufl_query=UFLFilter(
            entity_ids=["ID_TDG"],
            metric_keywords=["Sales"], # Matches "Net Sales" via substring
            years=["2023"]
        ),
        vector_hypothesis="Looking for sales figures in 2023.",
        vector_keywords=["sales"],
        reasoning="Test"
    )

    with patch("venra.retriever.chromadb.PersistentClient"), \
         patch("os.path.exists") as mock_exists, \
         patch("pandas.read_parquet") as mock_read_parquet:
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ufl_df
        retriever = DualRetriever(ufl_path="fake.parquet")
        
        results = await retriever.retrieve(plan)
        # Should match "Net Sales" because it contains "Sales"
        # The lexical gate for metrics now uses metric_keywords tokens (["sales"]),
        # so "Net Sales" (tokens: ["sales"]) will have recall 1.0 and pass.
        assert len(results["ufl_rows"]) == 1
        assert results["ufl_rows"][0].metric_name == "Net Sales"
