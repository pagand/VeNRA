import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from venra.models import DocBlock, BlockType, EntityMetadata, TableBlock, UFLRow, TextBlock, FactExtractionResponse, ScrapedFact
from venra.synthesis import EntityResolver, TableMelter, TextSynthesizer, ContextIndexer

# ==========================================
# Feature: Entity Resolution
# ==========================================

@pytest.fixture
def mock_cover_blocks():
    return [
        DocBlock(block_type=BlockType.TEXT, content="UNITED STATES SECURITIES AND EXCHANGE COMMISSION", section_path=[]),
        DocBlock(block_type=BlockType.TEXT, content="FORM 10-K", section_path=[]),
        DocBlock(block_type=BlockType.TEXT, content="TransDigm Group Incorporated", section_path=["Exact name of registrant as specified in its charter"]),
        DocBlock(block_type=BlockType.TEXT, content="Delaware", section_path=["State or other jurisdiction"]),
        DocBlock(block_type=BlockType.TEXT, content="1350 Euclid Avenue, Suite 1600, Cleveland, Ohio 44115", section_path=["Address of principal executive offices"]),
    ]

@pytest.mark.asyncio
async def test_entity_resolution_flow(mock_cover_blocks):
    """
    Test that EntityResolver correctly constructs the prompt context and parses the SLM response.
    """
    # 1. Setup Mock for Instructor/LLM
    mock_metadata = EntityMetadata(
        canonical_id="ID_TDG",
        official_name="TransDigm Group Incorporated",
        cik="0001260221",
        aliases=["TransDigm", "The Company", "TD Group"]
    )
    
    # We mock the `create` method of the instructor client
    with patch("venra.synthesis.instructor.from_openai") as mock_instructor_init:
        mock_client = MagicMock()
        mock_instructor_init.return_value = mock_client
        
        # When client.chat.completions.create is called, return our mock object
        mock_client.chat.completions.create.return_value = mock_metadata
        
        resolver = EntityResolver(api_key="fake_key")
        result = await resolver.resolve_entity(mock_cover_blocks)
        
        # 2. Assertions
        assert isinstance(result, EntityMetadata)
        assert result.canonical_id == "ID_TDG"
        assert result.official_name == "TransDigm Group Incorporated"
        assert "The Company" in result.aliases
        
        # Verify the context sent to LLM contained the registrant name
        call_args = mock_client.chat.completions.create.call_args
        assert call_args is not None
        messages = call_args[1]['messages']
        user_content = messages[1]['content']
        assert "TransDigm Group Incorporated" in user_content

# ==========================================
# Feature: Table Melting & Extraction
# ==========================================

def test_table_melter_basic_scaling():
    """
    Test that the melter correctly identifies 'millions' scaling and flattens the table.
    """
    markdown = """
| Item | 2023 | 2022 |
|---|---|---|
| Net Sales | 100 | 90 |
| Net Income | (10.5) | 5 |
"""
    # Context indicates scaling
    block = TableBlock(
        content=markdown,
        section_path=["Financial Statements", "Consolidated", "(In millions)"]
    )
    
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    # We expect 4 rows (2 items * 2 years)
    assert len(rows) == 4
    
    # Check Net Sales 2023
    sales_2023 = next(r for r in rows if r.metric_name == "Net Sales" and r.period == "2023")
    assert sales_2023.value == 100_000_000.0
    assert sales_2023.scale_factor == 1_000_000.0
    
    # Check Net Income 2023 (Negative handling)
    income_2023 = next(r for r in rows if r.metric_name == "Net Income" and r.period == "2023")
    assert income_2023.value == -10_500_000.0
    assert income_2023.nuance_note == "Negative (parentheses)"

def test_table_melter_thousands_scaling():
    """
    Test that the melter identifies 'thousands' scaling.
    """
    markdown = """
| Asset | Value |
|---|---|
| Cash | 500 |
"""
    # Context indicates scaling in title
    block = TableBlock(
        content=markdown,
        section_path=["Balance Sheet", "in thousands"]
    )
    
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    assert rows[0].value == 500_000.0
    assert rows[0].scale_factor == 1_000.0

def test_table_melter_no_scaling():
    """
    Test default behavior when no scaling keywords are found.
    """
    markdown = """
| Item | Count |
|---|---|
| Employees | 1200 |
"""
    block = TableBlock(
        content=markdown,
        section_path=["General Info"]
    )
    
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    assert rows[0].value == 1200.0
    assert rows[0].scale_factor == 1.0

def test_melter_placeholder_rows():
    """
    Test that 'I Don't Know' scenarios create placeholder rows instead of skipping.
    """
    markdown = """
| Metric | Value |
|---|---|
| Ambiguous Item | N/A |
"""
    block = TableBlock(content=markdown, section_path=["Notes"])
    
    # We pass entity_name_raw to constructor now (Requirement)
    melter = TableMelter(entity_id="ID_TEST", entity_name_raw="Test Corp")
    rows = melter.melt(block)
    
    assert len(rows) == 1
    row = rows[0]
    assert row.metric_name == "Ambiguous Item"
    assert row.value is None
    assert row.confidence == 0.0 # Should be 0.0 for placeholders
    assert row.entity_name_raw == "Test Corp" # Check raw name population

def test_table_melter_deterministic_cleaning():
    """
    Test that footnotes are stripped and dashes are treated as zero.
    """
    markdown = """
| Metric (1) | 2023 | 2022 |
|---|---|---|
| Sales (a) | 125(2) | — |
| Profit | 10.5b | (5.0) |
"""
    block = TableBlock(content=markdown, section_path=["Financials"])
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    # Check Sales 2023 (Footnote stripped from both metric and value)
    sales_23 = next(r for r in rows if r.metric_name == "Sales" and r.period == "2023")
    assert sales_23.value == 125.0
    
    # Check Sales 2022 (Dash to zero)
    sales_22 = next(r for r in rows if r.metric_name == "Sales" and r.period == "2022")
    assert sales_22.value == 0.0
    assert sales_22.nuance_note == "Dash treated as zero"
    
    # Check Profit 2023 (Trailing letter footnote stripped)
    profit_23 = next(r for r in rows if r.metric_name == "Profit" and r.period == "2023")
    assert profit_23.value == 10.5

# ==========================================
# Feature: Text-to-Fact Extraction
# ==========================================

@pytest.mark.asyncio
async def test_text_synthesizer_numerical():
    """
    Test extraction of a standard numerical fact.
    """
    block = TextBlock(
        content="As of December 31, 2023, our total backlog was approximately $1.2 billion.",
        section_path=["MD&A", "Backlog"]
    )
    
    # Mock Response
    mock_resp = FactExtractionResponse(facts=[
        ScrapedFact(
            metric_name="Total Backlog",
            value=1_200_000_000.0,
            unit="USD",
            period="2023-12-31",
            confidence=0.9
        )
    ])
    
    with patch("venra.synthesis.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp
        
        synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
        rows = await synthesizer.extract_facts(block)
        
        assert len(rows) == 1
        row = rows[0]
        assert row.metric_name == "Total Backlog"
        assert row.value == 1_200_000_000.0
        assert row.period == "2023-12-31"
        assert row.source_chunk_id == block.id
        assert row.doc_section == "MD&A > Backlog"

@pytest.mark.asyncio
async def test_text_synthesizer_qualitative():
    """
    Test extraction of a qualitative fact (NaN value).
    """
    block = TextBlock(
        content="We are subject to a tax audit which may result in material liability.",
        section_path=["Risk Factors"]
    )
    
    mock_resp = FactExtractionResponse(facts=[
        ScrapedFact(
            metric_name="Tax Audit Risk",
            value=None,
            unit="N/A",
            nuance_note="Potential material liability from ongoing tax audit.",
            confidence=0.8
        )
    ])
    
    with patch("venra.synthesis.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp
        
        synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
        rows = await synthesizer.extract_facts(block)
        
        assert len(rows) == 1
        row = rows[0]
        assert row.metric_name == "Tax Audit Risk"
        assert row.value is None
        assert "tax audit" in row.nuance_note.lower()

@pytest.mark.asyncio
async def test_text_synthesizer_implicit_period():
    """
    Test fallback when period is not explicit in the sentence.
    """
    block = TextBlock(content="The backlog remains solid due to strong commercial aftermarket demand across all segments.", section_path=[])
    
    # Mock Response
    mock_resp = FactExtractionResponse(facts=[
        ScrapedFact(
            metric_name="Backlog Strength",
            value=None,
            confidence=0.7, # Higher than 0.60 threshold
            period=None # Missing
        )
    ])
    
    with patch("venra.synthesis.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp
        
        synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
        # Should default period to "UNKNOWN"
        rows = await synthesizer.extract_facts(block)
        
        assert len(rows) == 1
        assert rows[0].period == "UNKNOWN"

@pytest.mark.asyncio
async def test_text_synthesizer_string_id_handling():
    """
    Test extraction where value is a string ID (e.g. IRS ID).
    Should move string to nuance_note and set value to None.
    """
    block = TextBlock(
        content="Our I.R.S. Employer Identification No. is 41-2101738.",
        section_path=["Cover"]
    )
    
    # Mock Response with string value
    mock_resp = FactExtractionResponse(facts=[
        ScrapedFact(
            metric_name="IRS ID",
            value="41-2101738", # String ID
            unit="ID",
            confidence=1.0
        )
    ])
    
    with patch("venra.synthesis.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp
        
        synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
        rows = await synthesizer.extract_facts(block)
        
        assert len(rows) == 1
        assert rows[0].value is None
        assert "41-2101738" in rows[0].nuance_note

# ==========================================
# Feature: Multi-Fact Extraction (Single Block)
# ==========================================

@pytest.mark.asyncio
async def test_text_synthesizer_multiple_facts_single_block():
    """
    Test extraction of multiple distinct facts from a single text block.
    A single sentence can contain multiple metrics - we must extract all of them.
    """
    block = TextBlock(
        content="In fiscal 2023, revenue increased 12% to $500 million, while operating income grew 8% to $75 million and our workforce expanded to 2,500 employees.",
        section_path=["MD&A", "Operating Results"]
    )

    # Mock Response with multiple facts extracted from one sentence
    mock_resp = FactExtractionResponse(facts=[
        ScrapedFact(
            metric_name="Revenue",
            value=500_000_000.0,
            unit="USD",
            period="2023",
            nuance_note="12% increase",
            confidence=0.95
        ),
        ScrapedFact(
            metric_name="Operating Income",
            value=75_000_000.0,
            unit="USD",
            period="2023",
            nuance_note="8% growth",
            confidence=0.92
        ),
        ScrapedFact(
            metric_name="Employee Count",
            value=2500.0,
            unit="count",
            period="2023",
            confidence=0.98
        )
    ])

    with patch("venra.synthesis.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp

        synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
        rows = await synthesizer.extract_facts(block)

        # Should extract all 3 facts
        assert len(rows) == 3

        # Verify revenue row
        revenue_row = next(r for r in rows if r.metric_name == "Revenue")
        assert revenue_row.value == 500_000_000.0
        assert revenue_row.period == "2023"
        assert "12%" in revenue_row.nuance_note

        # Verify operating income row
        oi_row = next(r for r in rows if r.metric_name == "Operating Income")
        assert oi_row.value == 75_000_000.0
        assert "8%" in oi_row.nuance_note

        # Verify all rows link back to the same source chunk
        for row in rows:
            assert row.source_chunk_id == block.id
            assert row.entity_id == "ID_TEST"

# ==========================================
# Feature: Context Indexing (ChromaDB)
# ==========================================

@pytest.fixture
def mock_chroma():
    with patch("venra.synthesis.chromadb.PersistentClient") as mock_client:
        yield mock_client

def test_context_indexer_blocks(mock_chroma):
    """
    Test that blocks are correctly indexed with metadata.
    """
    mock_collection = MagicMock()
    mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
    
    # Initialize indexer AFTER mock setup
    indexer = ContextIndexer()
    
    block = TextBlock(
        content="Revenue was $100M.",
        section_path=["Financials", "Income Statement"],
        page_num=10
    )
    
    indexer.index_blocks([block])
    
    # Verify add was called
    mock_collection.add.assert_called()
    
    # Verify arguments
    call_args = mock_collection.add.call_args[1]
    assert call_args['documents'] == ["Revenue was $100M."]
    assert call_args['ids'] == [block.id]
    assert call_args['metadatas'][0]['page_num'] == 10

def test_context_indexer_ufl_schema(mock_chroma):
    """
    Test that UFLRow metric names are indexed for semantic schema mapping.
    """
    mock_collection = MagicMock()
    mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
    
    indexer = ContextIndexer()
    
    row = UFLRow(
        row_id="hash1",
        entity_id="ID_AAPL",
        entity_name_raw="Apple Inc.",
        metric_name="Senior Notes Payable",
        value=1000.0,
        period="2023",
        doc_section="Note 5",
        source_chunk_id="chunk1",
        confidence=0.9
    )
    
    indexer.index_ufl_schema([row])
    
    # Verify add was called for the metric
    mock_collection.add.assert_called()
    kwargs = mock_collection.add.call_args[1]
    
    assert kwargs['documents'] == ["Senior Notes Payable"]
    assert kwargs['metadatas'][0]['metric_name'] == "Senior Notes Payable"
    assert kwargs['metadatas'][0]['entity_id'] == "ID_AAPL"

def test_indexer_back_population(mock_chroma):
    """
    Test that we can update chunk metadata with extracted row IDs.
    """
    mock_collection = MagicMock()
    mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
    
    indexer = ContextIndexer()
    
    chunk_id = "chunk_123"
    row_ids = ["row_a", "row_b"]
    
    indexer.update_chunk_linkage(chunk_id, row_ids)
    
    # Verify collection.update was called on the text_collection (first call to get_or_create)
    mock_collection.update.assert_called_once()
    kwargs = mock_collection.update.call_args[1]
    
    assert kwargs['ids'] == [chunk_id]
    assert "contains_rows" in kwargs['metadatas'][0]
    assert "row_a" in kwargs['metadatas'][0]["contains_rows"]