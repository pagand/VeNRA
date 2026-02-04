import pytest
from venra.assembler import ContextAssembler
from venra.models import UFLRow, DocBlock, BlockType

def test_assembler_deduplication():
    assembler = ContextAssembler()
    
    # Duplicate Rows
    row = UFLRow(
        row_id="row1", entity_id="ID_T", entity_name_raw="Test",
        metric_name="Revenue", value=100.0, unit="USD", period="2023",
        doc_section="S1", source_chunk_id="chunk1",
        confidence=1.0
    )
    
    # Duplicate Chunks
    chunk = DocBlock(
        id="chunk1", content="Revenue was $100", 
        block_type=BlockType.TEXT, section_path=["S1"]
    )
    
    results = {
        "ufl_rows": [row, row],
        "text_chunks": [chunk, chunk]
    }
    
    context = assembler.assemble(results)
    
    # Verify counts in text (not perfect but simple)
    assert context.count("row1") == 1
    assert context.count("chunk1") == 2 # Once in UFL table row, once in Chunk ID header
    assert "Revenue" in context
    assert "--- CHUNK_ID: chunk1 ---" in context

def test_assembler_empty():
    assembler = ContextAssembler()
    context = assembler.assemble({})
    assert "No structured facts found." in context
    assert "No source text available." in context
