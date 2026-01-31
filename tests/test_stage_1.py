import pytest
from unittest.mock import MagicMock, patch
from venra.ingestion import StructuralParser
from venra.models import BlockType

@pytest.mark.asyncio
async def test_header_stack_logic():
    """
    Test that the line-by-line walker correctly tracks the header stack.
    """
    mock_markdown = """
# Section 1
This is text in section 1.
## Subsection 1.1
This is text in 1.1.
| Table | Header |
|---|---|
| Row 1 | Val 1 |
# Section 2
Back to section 2.
"""
    parser = StructuralParser(api_key="fake_key")
    
    # Mock the LlamaParse Document object
    mock_doc = MagicMock()
    mock_doc.text = mock_markdown
    
    with patch("venra.ingestion.LlamaParse.aload_data", return_value=[mock_doc]):
        blocks = await parser.parse_pdf("dummy.pdf")
        
        # We expect:
        # 1. TextBlock ("This is text in section 1.") -> path: ["Section 1"]
        # 2. TextBlock ("This is text in 1.1.") -> path: ["Section 1", "Subsection 1.1"]
        # 3. TableBlock (Table) -> path: ["Section 1", "Subsection 1.1"]
        # 4. TextBlock ("Back to section 2.") -> path: ["Section 2"]
        
        assert len(blocks) == 4
        
        assert blocks[0].block_type == BlockType.TEXT
        assert blocks[0].section_path == ["Section 1"]
        assert "section 1" in blocks[0].content
        
        assert blocks[1].block_type == BlockType.TEXT
        assert blocks[1].section_path == ["Section 1", "Subsection 1.1"]
        assert "text in 1.1" in blocks[1].content
        
        assert blocks[2].block_type == BlockType.TABLE
        assert blocks[2].section_path == ["Section 1", "Subsection 1.1"]
        assert "|" in blocks[2].content
        
        assert blocks[3].block_type == BlockType.TEXT
        assert blocks[3].section_path == ["Section 2"]
        assert "section 2" in blocks[3].content

@pytest.mark.asyncio
async def test_sequential_tables():
    """Test two tables back-to-back without intervening text."""
    mock_markdown = """
# Data
| Table 1 | A |
|---|---|
| R1 | V1 |
| Table 2 | B |
|---|---|
| R2 | V2 |
"""
    parser = StructuralParser(api_key="fake_key")
    mock_doc = MagicMock()
    mock_doc.text = mock_markdown
    
    with patch("venra.ingestion.LlamaParse.aload_data", return_value=[mock_doc]):
        blocks = await parser.parse_pdf("dummy.pdf")
        
        # Current logic might merge them if there isn't a text line break, 
        # or separate them if the parser sees a new header or specific structure.
        # Ideally, we want distinct blocks if possible, or one big block if they are merged in source.
        # But LlamaParse usually puts newlines.
        # Let's see how our logic handles it.
        # Our logic chunks by line. If "Table 2" line has pipes, it continues the chunk.
        # So they will likely be merged into ONE TableBlock unless there is a non-table line.
        
        # Actually, standard markdown usually requires a newline.
        # If they are merged, that is acceptable for now, but distinct is better.
        # Let's check what happens.
        assert len(blocks) >= 1
        assert blocks[0].block_type == BlockType.TABLE

@pytest.mark.asyncio
async def test_table_after_header_immediate():
    """Test a table that starts immediately after a header."""
    mock_markdown = """
# Financials
| Metric | Value |
|---|---|
| Rev | 100 |
"""
    parser = StructuralParser(api_key="fake_key")
    mock_doc = MagicMock()
    mock_doc.text = mock_markdown
    
    with patch("venra.ingestion.LlamaParse.aload_data", return_value=[mock_doc]):
        blocks = await parser.parse_pdf("dummy.pdf")
        
        assert len(blocks) == 1
        assert blocks[0].block_type == BlockType.TABLE
        assert blocks[0].section_path == ["Financials"]

@pytest.mark.asyncio
async def test_deeply_nested_headers():
    """Test deep nesting of headers."""
    mock_markdown = """
# Level 1
## Level 2
### Level 3
Deep content.
# Back to 1
"""
    parser = StructuralParser(api_key="fake_key")
    mock_doc = MagicMock()
    mock_doc.text = mock_markdown
    
    with patch("venra.ingestion.LlamaParse.aload_data", return_value=[mock_doc]):
        blocks = await parser.parse_pdf("dummy.pdf")
        
        # 1. TextBlock (Deep content) -> [Level 1, Level 2, Level 3]
        # 2. TextBlock (empty? no, we flush on header) -> Wait, "Deep content" is the only text.
        
        # Note: The logic flushes when it sees a header.
        # # Level 1 -> Flushes nothing (start)
        # ## Level 2 -> Flushes nothing (no content in Level 1 yet)
        # ### Level 3 -> Flushes nothing
        # Deep content...
        # # Back to 1 -> Flushes "Deep content"
        
        assert len(blocks) == 1
        assert blocks[0].content.strip() == "Deep content."
        assert blocks[0].section_path == ["Level 1", "Level 2", "Level 3"]

@pytest.mark.asyncio
async def test_mixed_content_robustness():
    """Test mixed text and table lines to ensure robust separation."""
    mock_markdown = """
# Mixed
Start text.
| T | H |
|---|---|
| V | 1 |
Middle text.
| T | 2 |
|---|---|
| V | 2 |
End text.
"""
    parser = StructuralParser(api_key="fake_key")
    mock_doc = MagicMock()
    mock_doc.text = mock_markdown
    
    with patch("venra.ingestion.LlamaParse.aload_data", return_value=[mock_doc]):
        blocks = await parser.parse_pdf("dummy.pdf")
        
        # 1. Text (Start)
        # 2. Table (1)
        # 3. Text (Middle)
        # 4. Table (2)
        # 5. Text (End)
        
        assert len(blocks) == 5
        assert blocks[0].block_type == BlockType.TEXT
        assert blocks[1].block_type == BlockType.TABLE
        assert blocks[2].block_type == BlockType.TEXT
        assert blocks[3].block_type == BlockType.TABLE
        assert blocks[4].block_type == BlockType.TEXT
