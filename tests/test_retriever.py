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
            "row_id": "r1", "entity_id": "ID_TDG", "entity_name_raw": "TransDigm",
            "metric_name": "Net Sales", "value": 100.0, "unit": "USD", "period": "2023",
            "doc_section": "Financials", "source_chunk_id": "c1", "confidence": 1.0,
            "related_entity_id": None
        },
        {
            "row_id": "r2", "entity_id": "ID_TDG", "entity_name_raw": "TransDigm",
            "metric_name": "Acquisition", "value": 50.0, "unit": "USD", "period": "2023",
            "doc_section": "Notes", "source_chunk_id": "c2", "confidence": 1.0,
            "related_entity_id": "Boeing"
        }
    ]
    return pd.DataFrame(data)

@pytest.mark.asyncio
async def test_retriever_direct_and_expansion(mock_ufl_df):
    """
    Tests that Retriever fetches both UFL and Vector data, and handles Expansion.
    """
    plan = RetrievalPlan(
        strategy="HYBRID",
        ufl_query=UFLFilter(
            entity_ids=["ID_TDG"],
            metric_keywords=["Net Sales"],
            years=["2023"]
        ),
        vector_hypothesis="Net sales for 2023 were...",
        vector_keywords=["sales"],
        reasoning="Test"
    )

    # Mock ChromaDB and OS path check
    with patch("venra.retriever.chromadb.PersistentClient") as mock_chroma, \
         patch("os.path.exists") as mock_exists, \
         patch("pandas.read_parquet") as mock_read_parquet:
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ufl_df
        
        mock_collection = MagicMock()
        mock_chroma.return_value.get_collection.return_value = mock_collection
        
        # Mock vector search result (Chunk c3)
        mock_collection.query.return_value = {
            "ids": [["c3"]],
            "documents": [["Relevant text about sales"]],
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
async def test_retriever_fuzzy_metric_fallback(mock_ufl_df):
    """
    Test that if exact metric fails, fuzzy substring search works.
    """
    plan = RetrievalPlan(
        strategy="UFL_ONLY",
        ufl_query=UFLFilter(
            entity_ids=["ID_TDG"],
            metric_keywords=["Sales"], # Matches "Net Sales" via substring
            years=["2023"]
        ),
        vector_hypothesis="None",
        vector_keywords=[],
        reasoning="Test"
    )

    with patch("venra.retriever.chromadb.PersistentClient"), \
         patch("os.path.exists") as mock_exists, \
         patch("pandas.read_parquet") as mock_read_parquet:
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ufl_df
        retriever = DualRetriever(ufl_path="fake.parquet")
        
        results = await retriever.retrieve(plan)
        assert len(results["ufl_rows"]) == 1
        assert results["ufl_rows"][0].metric_name == "Net Sales"
