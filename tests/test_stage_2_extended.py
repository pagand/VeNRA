import pytest
import pandas as pd
from typing import Any
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
# Feature: Advanced Table Melting (The "Production Killers")
# ==========================================

def test_table_melter_mixed_scale_exception():
    """
    CRITICAL: Test that 'Per Share' data ignores the table-wide 'Millions' scaling.
    If this fails, we report $5M earnings per share instead of $5.
    """
    markdown = """
| Metric | 2023 |
|---|---|
| Net Income | 500 |
| Earnings Per Share | 5.25 |
"""
    # Table header says "In Millions"
    block = TableBlock(
        content=markdown,
        section_path=["Income Statement", "(In Millions, except per share data)"]
    )
    
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    # 1. Check Net Income (Should be scaled)
    net_income = next(r for r in rows if r.metric_name == "Net Income")
    assert net_income.num_value == 500_000_000.0
    
    # 2. Check EPS (Should NOT be scaled)
    eps = next(r for r in rows if "Earnings Per Share" in r.metric_name)
    assert eps.num_value == 5.25 # MUST equal raw value
    assert eps.scale == 1.0
    assert eps.unit_normalized == "USD/Share" # Optional: Check if unit inference works

def test_table_melter_hierarchical_indentation():
    """
    Test that identical metric names (Cash) are disambiguated by their parents.
    Markdown represents indentation with non-breaking spaces (&nbsp;) or spaces.
    """
    markdown = """
| Item | Value |
|---|---|
| Assets | |
| &nbsp;&nbsp;Current Assets | |
| &nbsp;&nbsp;&nbsp;&nbsp;Cash | 100 |
| Liabilities | |
| &nbsp;&nbsp;Current Liabilities | |
| &nbsp;&nbsp;&nbsp;&nbsp;Cash | (50) |
"""
    block = TableBlock(content=markdown, section_path=["Balance Sheet"])
    
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    # We expect flattened names
    asset_cash = next(r for r in rows if r.num_value == 100.0)
    liability_cash = next(r for r in rows if r.num_value == -50.0)
    
    # Flexible assertion depending on your implementation style
    # Ideally: "Assets > Current Assets > Cash"
    assert "Assets" in asset_cash.metric_name
    assert "Liabilities" in liability_cash.metric_name
    assert asset_cash.metric_name != liability_cash.metric_name

# ==========================================
# Feature: SLM Header Normalization (Sub-Stage 2.2)
# ==========================================

@pytest.mark.asyncio
async def test_header_date_normalization():
    """
    Test the SLM Helper that converts messy table headers into ISO dates.
    """
    messy_headers = ["Year Ended Sept 30, 2023", "Three Months Ended June 30, 2022"]
    
    # Mock the SLM response for header cleaning
    # It must be an object with a 'mapping' attribute to match synthesis.py
    class MockHeaderMap:
        def __init__(self, mapping):
            self.mapping = mapping
            
    mock_resp = MockHeaderMap(mapping={
        "Year Ended Sept 30, 2023": "2023-09-30",
        "Three Months Ended June 30, 2022": "2022-06-30"
    })
    
    with patch("venra.synthesis.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        
        # Setup the mock to return the object
        mock_client.chat.completions.create.return_value = mock_resp
        
        melter = TableMelter(entity_id="ID_TEST")
        
        # Assume melter has a method `normalize_headers` or similar
        clean_headers = await melter.normalize_headers_with_slm(messy_headers)
        
        assert clean_headers["Year Ended Sept 30, 2023"] == "2023-09-30"
        
        # Verify prompt instructions
        call_args = mock_client.chat.completions.create.call_args
        assert "ISO 8601" in str(call_args) # We must instruct SLM to use ISO format

# ==========================================
# Feature: The "Restated" Logic (Data Collision)
# ==========================================

def test_table_melter_restated_columns():
    """
    Test that 'Restated' columns don't overwrite original data and generate unique IDs.
    """
    markdown = """
| Metric | 2022 | 2022 (Restated) |
|---|---|---|
| Revenue | 100 | 95 |
"""
    block = TableBlock(content=markdown, section_path=["Corrections"])
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    assert len(rows) == 2
    
    orig = next(r for r in rows if r.num_value == 100.0)
    restated = next(r for r in rows if r.num_value == 95.0)
    
    # 1. Ensure IDs are different
    assert orig.row_id != restated.row_id
    
    # 2. Check nuance note or period modification
    # Implementation choice: Either modify period "2022-RESTATED" or add note
    assert "Restated" in restated.text_nuance

# ==========================================
# Feature: Semantic Schema & Embedding
# ==========================================

class MockEmbeddingFunction:
    """Mock that satisfies ChromaDB signature validation."""
    def __call__(self, input: Any) -> Any:
        # Return a dummy vector for each input string
        return [[0.1] * 384] * len(input)
    def name(self) -> str:
        return "mock"

def test_semantic_schema_embedding_call():
    """
    Test that we actually generate embeddings for the metrics.
    If this isn't mocked, the test suite might try to call OpenAI/HuggingFace API.
    """
    mock_embedding_fn = MockEmbeddingFunction()
    
    indexer = ContextIndexer(embedding_fn=mock_embedding_fn)
    
    row = UFLRow(
        row_id="1", canonical_entity_id="A", entity_name_raw="Alpha Corp", metric_name="Revenue", grounding_quote="dummy", 
        num_value=10, period_end="2023", doc_section="A", source_chunk_id="C", confidence_score=1
    )
    
    # Verify method runs without error using the mock
    indexer.index_ufl_schema([row])

# ==========================================
# Feature: Multi-Entity/Relationship Extraction
# ==========================================

@pytest.mark.asyncio
async def test_text_synthesizer_relationships():
    """
    Test extraction of facts involving a Related Entity (Graph Edge).
    "Boeing accounted for 15% of our Net Sales."
    """
    block = TextBlock(
        content="Boeing accounted for 15% of our Net Sales in 2023.",
        section_path=["Concentration Risk"]
    )
    
    mock_fact = ScrapedFact(
        metric_name="Customer Concentration", grounding_quote="dummy",
        num_value=0.15,
        unit_normalized="Percent",
        period_start="2023",
        related_entity="Boeing", # This is the key field
        confidence=1.0
    )
    
    # Mock OpenAI client used in TextSynthesizer._single_pass
    with patch("venra.synthesis.OpenAI") as mock_openai_init:
        mock_client = MagicMock()
        mock_openai_init.return_value = mock_client
        
        # Mock completion response
        mock_choice = MagicMock()
        mock_choice.message.content = '{"facts": [' + mock_fact.model_dump_json() + ']}'
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        
        # Also mock the aligner to return the same facts
        with patch("venra.synthesis.PostHocAligner.align", side_effect=lambda f, **kw: f):
            synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
            rows = await synthesizer.extract_facts(block)
            
            row = rows[0]
            # related_entity name is now preserved in text_nuance until resolved
            assert "Boeing" in row.text_nuance
            assert row.related_entity_id is None


# ==========================================
# SME Case 1: The "Footnote Bomb" & "Dash vs Blank"
# ==========================================

def test_table_melter_cleaning_heuristics():
    """
    SME REQUIREMENT:
    1. '1,234(1)' must be parsed as 1234.0, NOT 12341.0.
    2. '—' (em-dash) must be parsed as 0.0.
    3. Empty string must be parsed as None (NaN).
    """
    markdown = """
| Metric | 2023 | 2022 |
|---|---|---|
| Debt | 1,234(1) | 1,000 |
| Derivative Gain | — | - |
| Obscure Item | | N/A |
"""
    block = TableBlock(content=markdown, section_path=["Balance Sheet"])
    
    melter = TableMelter(entity_id="ID_TEST")
    rows = melter.melt(block)
    
    # 1. Check Footnote Stripping
    # Current TableMelter puts the year in period_end for point-in-time metrics.
    debt_2023 = next(r for r in rows if r.metric_name == "Debt" and r.period_end == "2023")
    assert debt_2023.num_value == 1234.0, f"Failed to strip footnote, got {debt_2023.num_value}"
    
    # 2. Check Dash -> Zero
    gain_2023 = next(r for r in rows if r.metric_name == "Derivative Gain" and r.period_end == "2023")
    assert gain_2023.num_value == 0.0, "Em-dash should be 0.0"
    
    # 3. Check Hyphen -> Zero
    gain_2022 = next(r for r in rows if r.metric_name == "Derivative Gain" and r.period_end == "2022")
    assert gain_2022.num_value == 0.0, "Hyphen should be 0.0"

    # 4. Check Blank -> None
    obscure_2023 = next(r for r in rows if r.metric_name == "Obscure Item" and r.period_end == "2023")
    assert obscure_2023.num_value is None, "Blank cell should be None/NaN"

# ==========================================
# SME Case 2: Inequalities & Thresholds
# ==========================================

@pytest.mark.asyncio
async def test_text_synthesizer_inequalities():
    """
    SME REQUIREMENT:
    If text says "less than 0.5 million", we capture 500,000 but MUST flag the inequality.
    """
    block = TextBlock(
        content="We expect capital expenditures to be less than $50 million.",
        section_path=["Outlook"]
    )
    
    mock_fact = ScrapedFact(
        metric_name="Projected Capex", grounding_quote="dummy",
        num_value=50_000_000.0,
        unit_normalized="USD",
        text_nuance="operator: < (less than)", # The extractor prompt must be trained for this
        confidence=0.9
    )
    
    with patch("venra.synthesis.OpenAI") as mock_openai_init:
        mock_client = MagicMock()
        mock_openai_init.return_value = mock_client
        
        mock_choice = MagicMock()
        mock_choice.message.content = '{"facts": [' + mock_fact.model_dump_json() + ']}'
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        
        with patch("venra.synthesis.PostHocAligner.align", side_effect=lambda f, **kw: f):
            synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
            rows = await synthesizer.extract_facts(block)
            
            row = rows[0]
            assert row.num_value == 50_000_000.0
            assert "<" in row.text_nuance or "less than" in row.text_nuance

# ==========================================
# SME Case 3: The "Basis Point" Unit
# ==========================================

@pytest.mark.asyncio
async def test_text_synthesizer_basis_points():
    """
    SME REQUIREMENT:
    '50 basis points' is a distinct unit. Do not convert to 0.005 implicitly unless standardized.
    Better to keep unit_normalized='bps' so the Agent knows the math rules.
    """
    block = TextBlock(content="Gross margin improved by 120 basis points.", section_path=[])
    
    mock_fact = ScrapedFact(
        metric_name="Gross Margin Change", grounding_quote="dummy",
        num_value=120.0,
        unit_normalized="bps", # The extractor should identify this unit
        confidence=0.95
    )
    
    with patch("venra.synthesis.OpenAI") as mock_openai_init:
        mock_client = MagicMock()
        mock_openai_init.return_value = mock_client
        
        mock_choice = MagicMock()
        mock_choice.message.content = '{"facts": [' + mock_fact.model_dump_json() + ']}'
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        
        with patch("venra.synthesis.PostHocAligner.align", side_effect=lambda f, **kw: f):
            synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
            rows = await synthesizer.extract_facts(block)
            
            assert rows[0].num_value == 120.0
            assert rows[0].unit_normalized == "bps"

# ==========================================
# SME Case 4: Non-Numeric Entity Graph
# ==========================================

@pytest.mark.asyncio
async def test_graph_relationship_extraction():
    """
    SME REQUIREMENT:
    Extract knowledge graph edges where no money is involved.
    'Company A owns Company B' -> metric='Subsidiary', related_entity='Company B', num_value=NaN.
    """
    block = TextBlock(
        content="Our primary operating subsidiaries include Champion Aerospace and Avionic Instruments.",
        section_path=["Business"]
    )
    
    # We expect the LLM to return TWO facts here
    fact1 = ScrapedFact(
        metric_name="Subsidiary", grounding_quote="dummy",
        num_value=None,
        related_entity="Champion Aerospace",
        confidence=1.0
    )
    fact2 = ScrapedFact(
        metric_name="Subsidiary", grounding_quote="dummy",
        num_value=None,
        related_entity="Avionic Instruments",
        confidence=1.0
    )
    
    with patch("venra.synthesis.OpenAI") as mock_openai_init:
        mock_client = MagicMock()
        mock_openai_init.return_value = mock_client
        
        mock_choice = MagicMock()
        mock_choice.message.content = '{"facts": [' + fact1.model_dump_json() + ', ' + fact2.model_dump_json() + ']}'
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        
        with patch("venra.synthesis.PostHocAligner.align", side_effect=lambda f, **kw: f):
            synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
            rows = await synthesizer.extract_facts(block)
            
            assert len(rows) == 2
            assert rows[0].metric_name == "Subsidiary"
            # related_entity name is now preserved in text_nuance until resolved
            assert "Champion Aerospace" in rows[0].text_nuance
            assert rows[0].related_entity_id is None
            assert rows[0].num_value is None
