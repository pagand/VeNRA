import os
import pytest
import json
from unittest.mock import MagicMock, patch
from venra.pipeline import IngestionPipeline
from venra.schema import SchemaGenerator
from venra.models import DocBlock, BlockType, EntityMetadata, UFLRow, FactExtractionResponse

@pytest.fixture
def mock_blocks():
    return [
        DocBlock(block_type=BlockType.TEXT, content="TransDigm Group Incorporated", section_path=["Cover"]),
        DocBlock(block_type=BlockType.TABLE, content="""| Metric | 2023 |
|---|---|
| Net Sales | 100 |""", section_path=["Financials"]),
    ]

@pytest.mark.asyncio
async def test_pipeline_generates_schema_mocked(tmp_path, mock_blocks):
    """
    Test the pipeline with mocked LLM calls and file IO.
    """
    # 1. Setup Mocks
    mock_entity = EntityMetadata(
        canonical_id="ID_TDG",
        official_name="TransDigm Group Incorporated",
        aliases=["TransDigm"]
    )
    
    mock_facts = FactExtractionResponse(facts=[]) # Empty for simplicity

    # Mock the entire LLM client flow
    with patch("venra.synthesis.instructor.from_openai") as mock_instructor_init, \
         patch("venra.ingestion.StructuralParser.parse_pdf") as mock_parse, \
         patch("venra.synthesis.ContextIndexer.index_blocks") as mock_index_blocks, \
         patch("venra.synthesis.ContextIndexer.index_ufl_schema") as mock_index_schema:
        
        mock_client = MagicMock()
        mock_instructor_init.return_value = mock_client
        
        # Configure the mock client to return entity metadata then empty facts
        # It's called by EntityResolver then TextSynthesizer
        mock_client.chat.completions.create.side_effect = [mock_entity, mock_facts]
        
        # Mock structural parser to return our dummy blocks
        mock_parse.return_value = mock_blocks
        
        # 2. Configure Pipeline
        test_schema_path = os.path.join(tmp_path, "schema_summary.json")
        pipeline = IngestionPipeline()
        pipeline.schema_gen.output_path = test_schema_path
        
        # 3. Run Pipeline
        # pdf_path doesn't need to exist because parse_pdf is mocked
        await pipeline.run("dummy.pdf", skip_parsing=False)
        
        # 4. Assertions
        assert os.path.exists(test_schema_path)
        
        with open(test_schema_path, "r") as f:
            schema = json.load(f)
            
        assert "entities" in schema
        assert "metrics" in schema
        
        # Verify Entity Resolution worked
        entity_ids = [e["id"] for e in schema["entities"]]
        assert "ID_TDG" in entity_ids
        
        # Verify Metric collection (Net Sales from the mock table)
        metrics = schema["metrics"]
        assert "Net Sales" in metrics

def test_schema_generator_logic():
    """
    Pure unit test for SchemaGenerator without any mocks.
    """
    gen = SchemaGenerator(output_path="dummy.json")
    from venra.models import EntityMetadata, UFLRow
    
    # Add dummy entity
    entity = EntityMetadata(
        canonical_id="ID_TEST",
        official_name="Test Corp",
        aliases=["Test", "TC"]
    )
    gen.add_entity(entity)
    
    # Add dummy rows
    rows = [
        UFLRow(row_id="1", entity_id="ID_TEST", metric_name="Revenue", value=100.0, period="2023", source_chunk_id="c1", entity_name_raw="Test", doc_section="Financials", confidence=1.0),
        UFLRow(row_id="2", entity_id="ID_TEST", metric_name="Revenue", value=110.0, period="2022", source_chunk_id="c1", entity_name_raw="Test", doc_section="Financials", confidence=1.0),
        UFLRow(row_id="3", entity_id="ID_TEST", metric_name="EBITDA", value=20.0, period="2023", source_chunk_id="c1", entity_name_raw="Test", doc_section="Financials", confidence=1.0),
    ]
    gen.add_rows(rows)
    
    # Test internal logic
    sorted_metrics = sorted(gen.metrics.items(), key=lambda x: x[1], reverse=True)
    top_metrics = [m[0] for m in sorted_metrics[:500]]
    
    assert "Revenue" in top_metrics
    assert "EBITDA" in top_metrics
    assert gen.metrics["Revenue"] == 2
    assert gen.metrics["EBITDA"] == 1