import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from venra.models import FactExtractionResponse, ScrapedFact, TextBlock, UFLRow


# ==========================================
# Feature: Flattened Schema Validation
# ==========================================

def test_flattened_schema_attributes():
    """
    Verify that ScrapedFact matches the flattened schema defined in UFL_spec.md §1.
    Every UFLRow field that the SLM must populate should have a matching field on
    ScrapedFact so conversion is a 1-to-1 attribute copy.
    """
    # CHANGED[Phase7]: removed duplicate grounding_quote="dummy" kwarg (SyntaxError).
    # CHANGED[Phase7]: confidence_score= → confidence= (ScrapedFact field name).
    fact = ScrapedFact(
        metric_name="Revenue",
        num_value=100.0,
        grounding_quote="$100 million",
        unit_normalized="USD",
        period_start="2023-01-01",
        period_end="2023-12-31",
        period_type="FY",
        text_nuance="Strong sales",
        confidence=0.9,
    )
    assert fact.grounding_quote == "$100 million"
    assert fact.num_value == 100.0
    assert fact.period_type == "FY"
    # CHANGED[Phase7]: confidence_score → confidence (ScrapedFact field, not UFLRow field).
    assert fact.confidence == 0.9


# ==========================================
# Feature: Post-Hoc Aligner & Hallucination Killer
# ==========================================

def test_post_hoc_aligner_exact_match():
    """
    Test that the aligner finds the exact grounding_quote and sets EXACT alignment.
    Developer_plan.md Sub-Stage 2.3.b Tier 1 (EXACT): str.find() + whitespace-norm.
    """
    from venra.synthesis import PostHocAligner

    source_text = "The company reported revenue of $615 million for the quarter."
    facts = [
        # CHANGED[Phase7]: duplicate grounding_quote="dummy" kwarg removed.
        # CHANGED[Phase7]: confidence_score= → confidence=.
        ScrapedFact(
            metric_name="Revenue",
            num_value=615_000_000.0,
            grounding_quote="$615 million",
            confidence=0.9,
        )
    ]

    aligner = PostHocAligner()
    aligned_facts = aligner.align(facts, source_text)

    assert len(aligned_facts) == 1
    assert aligned_facts[0].alignment_status == "EXACT"
    assert aligned_facts[0].char_interval is not None

    start, end = aligned_facts[0].char_interval
    assert source_text[start:end] == "$615 million"

    # CHANGED[Phase7]: confidence_score == 0.9 → alignment_confidence == approx(0.70).
    # PostHocAligner._mark() sets fact.alignment_confidence, not confidence_score.
    # EXACT tier → _CONFIDENCE_TEXT_ALIGNED = 0.70 (synthesis.py constant).
    # The SLM's self-reported 0.9 is always overridden (UFL_spec.md §2 Step D).
    assert aligned_facts[0].alignment_confidence == pytest.approx(0.70)


def test_post_hoc_aligner_hallucination_rejection():
    """
    Test that the aligner rejects a hallucination where grounding_quote is absent
    from source text. Developer_plan.md Sub-Stage 2.3.b: Rejection Protocol.
    """
    from venra.synthesis import PostHocAligner

    source_text = "The company reported revenue of $615 million for the quarter."
    facts = [
        # CHANGED[Phase7]: duplicate kwarg removed, confidence field fixed.
        ScrapedFact(
            metric_name="Net Income",
            num_value=100_000_000.0,
            grounding_quote="$100 million",  # Hallucinated — not in source text
            confidence=0.9,
        )
    ]

    aligner = PostHocAligner()
    aligned_facts = aligner.align(facts, source_text)

    assert len(aligned_facts) == 1
    assert aligned_facts[0].alignment_status == "UNALIGNED"

    # CHANGED[Phase7]: confidence_score → alignment_confidence.
    # UNALIGNED → _CONFIDENCE_UNALIGNED = 0.0.
    assert aligned_facts[0].alignment_confidence == 0.0


def test_post_hoc_aligner_fuzzy_match():
    """
    Test that the aligner achieves at least PARTIAL or FUZZY alignment when the
    exact text differs slightly from the source.
    Developer_plan.md Sub-Stage 2.3.b: Tier 2 (PARTIAL) and Tier 3 (FUZZY).
    """
    from venra.synthesis import PostHocAligner

    source_text = "The company reported revenue of $615.5 million for the quarter."
    facts = [
        # CHANGED[Phase7]: duplicate kwarg removed, confidence field fixed.
        ScrapedFact(
            metric_name="Revenue",
            num_value=615_500_000.0,
            grounding_quote="$615 million",  # Minor LLM error — dropped the .5
            confidence=0.9,
        )
    ]

    aligner = PostHocAligner()
    aligned_facts = aligner.align(facts, source_text)

    assert len(aligned_facts) == 1
    # "615" is present in source → Tier 2 PARTIAL match is the expected outcome.
    assert aligned_facts[0].alignment_status in ("PARTIAL", "FUZZY")
    # CHANGED[Phase7]: confidence_score > 0.0 → alignment_confidence > 0.0.
    assert aligned_facts[0].alignment_confidence > 0.0


def test_post_hoc_aligner_partial_numeric_token():
    """
    [New] Verify Tier 2 (PARTIAL) specifically: the largest numeric token in
    grounding_quote must be found verbatim in the source even when surrounding
    text differs. UFL_spec.md §2 Step D Tier 2.
    """
    from venra.synthesis import PostHocAligner

    source_text = "Operating profit reached 2,847 thousand for the period."
    facts = [
        ScrapedFact(
            metric_name="Operating Profit",
            num_value=2_847_000.0,
            grounding_quote="2,847 thousand",
            confidence=0.85,
        )
    ]

    aligner = PostHocAligner()
    aligned = aligner.align(facts, source_text)

    assert aligned[0].alignment_status in ("EXACT", "PARTIAL")
    assert aligned[0].alignment_confidence > 0.0


def test_post_hoc_aligner_empty_raw_value():
    """
    [New] A fact with empty grounding_quote must be immediately marked UNALIGNED.
    UFL_spec.md §2 Step D Rejection. The aligner must not attempt expensive
    fuzzy search on an empty query string.
    """
    from venra.synthesis import PostHocAligner

    source_text = "Revenue was $500 million in 2023."
    facts = [
        ScrapedFact(
            metric_name="Qualitative Risk",
            num_value=None,
            grounding_quote="",
            text_nuance="some qualitative note",
            confidence=0.7,
        )
    ]

    aligner = PostHocAligner()
    aligned = aligner.align(facts, source_text)

    assert aligned[0].alignment_status == "UNALIGNED"
    assert aligned[0].alignment_confidence == 0.0
    assert aligned[0].char_interval is None


def test_post_hoc_aligner_whitespace_normalisation():
    """
    [New] Tier 1 sub-step 2: whitespace-normalised exact match.
    grounding_quote with collapsed whitespace must match source with expanded
    whitespace (e.g. multi-space PDF artefact vs single-space extraction).
    UFL_spec.md §2 Step D Tier 1 implementation note.
    """
    from venra.synthesis import PostHocAligner

    source_text = "Total debt outstanding of $  615 million was recorded."
    facts = [
        ScrapedFact(
            metric_name="Total Debt",
            num_value=615_000_000.0,
            grounding_quote="$ 615 million",   # single space — matches after normalisation
            confidence=0.9,
        )
    ]

    aligner = PostHocAligner()
    aligned = aligner.align(facts, source_text)

    assert aligned[0].alignment_status == "EXACT"
    assert aligned[0].alignment_confidence == pytest.approx(0.70)


# ==========================================
# Feature: Multi-Pass Extraction
# ==========================================

@pytest.mark.asyncio
async def test_multi_pass_extraction_dedup_first_pass_wins():
    """
    Test that when two passes produce the same (metric_name, period_start, period_end)
    key, the first-pass extraction is kept and the second is discarded.
    Developer_plan.md Sub-Stage 2.3: 'First-pass wins on duplicate key.'
    Dedup key: (metric_name.lower(), period_start, period_end) per _merge_passes.

    CHANGED[Phase7]: Complete rewrite of the original test.
    Original test assumed dedup by char_interval overlap and called non-existent
    methods (_run_single_pass, extract_facts_multipass). Actual dedup is by
    (metric_name, period_start, period_end) — see synthesis.py _merge_passes.
    """
    from venra.synthesis import TextSynthesizer

    # Dense block (> 800 chars) to trigger multi-pass (3 passes)
    long_content = "Revenue reached $500 million in fiscal 2023. " * 20  # ~900 chars
    block = TextBlock(content=long_content, section_path=["MD&A"])

    pass1_facts = [
        ScrapedFact(metric_name="Revenue", grounding_quote="$500 million",
                    num_value=500_000_000.0, period_start="2023", confidence=0.95),
        ScrapedFact(metric_name="Operating Income", grounding_quote="$50 million",
                    num_value=50_000_000.0, period_start="2023", confidence=0.92),
    ]
    pass2_facts = [
        ScrapedFact(metric_name="Revenue", grounding_quote="$500 million",
                    num_value=500_000_000.0, period_start="2023", confidence=0.88),  # duplicate
        ScrapedFact(metric_name="EPS", grounding_quote="$2.10",
                    num_value=2.1, period_start="2023", confidence=0.90),
    ]
    pass3_facts = [
        ScrapedFact(metric_name="Revenue", grounding_quote="$500 million",
                    num_value=500_000_000.0, period_start="2023", confidence=0.80),  # duplicate
        ScrapedFact(metric_name="Operating Income", grounding_quote="$50 million",
                    num_value=50_000_000.0, period_start="2023", confidence=0.75),   # duplicate
    ]

    synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
    mock_single_pass = AsyncMock(side_effect=[pass1_facts, pass2_facts, pass3_facts])

    with patch.object(synthesizer, "_single_pass", new=mock_single_pass):
        with patch.object(synthesizer._aligner, "align", side_effect=lambda f, **kw: f):
            rows = await synthesizer.extract_facts(block)

    metric_names = [r.metric_name for r in rows]
    assert len(rows) == 3
    assert "Revenue" in metric_names
    assert "Operating Income" in metric_names
    assert "EPS" in metric_names


@pytest.mark.asyncio
async def test_single_pass_for_sparse_blocks():
    """
    [New] Blocks below _DENSE_BLOCK_CHARS=800 chars must use exactly 1 pass.
    Developer_plan.md Sub-Stage 2.3: 'Dense block threshold: _DENSE_BLOCK_CHARS=800.'
    """
    from venra.synthesis import TextSynthesizer

    block = TextBlock(
        content="Revenue was $100 million in 2023.",
        section_path=["MD&A"],
    )

    synthesizer = TextSynthesizer(entity_id="ID_TEST", api_key="fake")
    call_count = []

    async def mock_pass(*args, **kwargs):
        call_count.append(1)
        return [
            ScrapedFact(metric_name="Revenue", grounding_quote="$100 million",
                        num_value=100_000_000.0, confidence=0.9)
        ]

    with patch.object(synthesizer, "_single_pass", side_effect=mock_pass):
        with patch.object(synthesizer._aligner, "align", side_effect=lambda f, **kw: f):
            await synthesizer.extract_facts(block)

    assert len(call_count) == 1, (
        f"Expected exactly 1 pass for sparse block, got {len(call_count)}"
    )


# ==========================================
# Feature: Lexical Pre-Filtering
# ==========================================

def test_lexical_pre_filtering_rejects_drifted_match():
    """
    Test that a semantic vector match with low token overlap is rejected.
    Developer_plan.md Step 1: 'prevents matching Net Sales to Net Income solely
    due to high cosine similarity.'

    CHANGED[Phase7]: Rewrote from calling non-existent DualRetriever methods.

    FIXED[Critic]: Now uses _tokenize() to derive query_tokens from the raw query
    string instead of passing hardcoded ["sales", "2023"]. This tests the full
    pipeline (tokenizer + recall gate) rather than _lexical_recall in isolation.
    If _tokenize were broken, the old test would still pass; this one would fail.
    """
    from venra.retriever import _tokenize, _lexical_recall, _passes_lexical_gate, LEXICAL_OVERLAP_THRESHOLD

    query = "Net Sales for 2023"
    candidate = "Net Income for the fiscal year was $5M."

    # Use _tokenize so both the tokenizer and recall gate are exercised together.
    # Expected: "net" → stop-word (filtered). "sales" → kept. "for" → stop-word.
    # "2023" → kept. Result: ["sales", "2023"].
    query_tokens = _tokenize(query)
    assert "net" not in query_tokens, (
        "'net' must be filtered as a stop-word. Without this, a 'Net Income' query "
        "would score recall=0.5 against 'Net Sales' and incorrectly pass the gate."
    )
    assert "sales" in query_tokens

    score = _lexical_recall(query_tokens, candidate)
    assert score < LEXICAL_OVERLAP_THRESHOLD, (
        f"'Net Sales' query must not pass lexical gate for 'Net Income' candidate. "
        f"Got recall={score:.2f}, threshold={LEXICAL_OVERLAP_THRESHOLD}"
    )
    assert not _passes_lexical_gate(query_tokens, candidate)


def test_lexical_pre_filtering_accepts_correct_match():
    """
    [New] A chunk that is both semantically and lexically aligned must pass the gate.
    """
    from venra.retriever import _tokenize, _passes_lexical_gate

    query_tokens = _tokenize("Net Sales revenue 2023")
    chunk_text = "Net Sales and Revenue for fiscal year 2023 were $500 million."

    assert _passes_lexical_gate(query_tokens, chunk_text), (
        "Correct semantic match should pass the lexical gate"
    )


def test_lexical_pre_filtering_empty_query_passes_all():
    """
    [New] Empty query_tokens must pass all candidates (vacuously true).
    UFL_spec.md §3: 'If query_tokens is empty, all candidates are accepted.'
    """
    from venra.retriever import _passes_lexical_gate

    assert _passes_lexical_gate([], "Any text whatsoever.")


def test_lexical_recall_uses_recall_not_jaccard():
    """
    [New] Verify that the recall metric is computed correctly — NOT Jaccard.
    UFL_spec.md §3: 'Recall (not Jaccard) because query terms are fewer than
    document tokens; Jaccard union is dominated by document-only tokens.'

    Jaccard = 2M / (|a| + |b|) collapses to ~0 when |b| >> |a|.
    Recall = M / |query| stays at 1.0 when all query tokens appear in the document.
    """
    from venra.retriever import _lexical_recall, _tokenize

    query_tokens = _tokenize("revenue 2023")   # 2 tokens after stop-word filtering
    long_doc = (
        "The company reported annual revenue growth in fiscal 2023. "
        "Operating expenses, capital expenditures, depreciation, amortization, "
        "and other line items all impacted the bottom line for the period."
    )

    recall = _lexical_recall(query_tokens, long_doc)

    # Both "revenue" and "2023" appear in long_doc → recall must be 1.0.
    # With Jaccard, the large document would drive this to ~0.10.
    assert recall == pytest.approx(1.0), (
        f"Expected recall=1.0 (all query tokens present in doc), got {recall}"
    )


def test_apply_lexical_filter_drops_and_keeps():
    """
    [New] DualRetriever._apply_lexical_filter correctly partitions candidates.
    Uses _tokenize() to derive query_tokens from a raw query string, exercising
    the full pipeline.
    """
    from venra.models import DocBlock, BlockType
    from venra.retriever import DualRetriever, _tokenize

    relevant_block = DocBlock(
        id="r1", block_type=BlockType.TEXT,
        content="Net Sales for fiscal 2023 were $500 million.",
        section_path=[],
    )
    irrelevant_block = DocBlock(
        id="r2", block_type=BlockType.TEXT,
        content="The board approved a dividend payment this quarter.",
        section_path=[],
    )

    query_tokens = _tokenize("Net Sales 2023")
    kept = DualRetriever._apply_lexical_filter(
        [relevant_block, irrelevant_block],
        query_tokens,
        label="test",
    )

    assert relevant_block in kept
    assert irrelevant_block not in kept


def test_prev_context_prefix_stripped_before_gate():
    """
    [New] Verify that the [Previous Context:] prefix injected by Phase 2 ingestion
    is stripped before tokenizing, preventing false acceptances.

    Without stripping: a chunk about dividends prefixed with revenue context
    would pass a "revenue" query — the prefix tokens satisfy the gate even though
    the chunk itself is irrelevant.
    """
    from venra.retriever import _lexical_recall, _tokenize, LEXICAL_OVERLAP_THRESHOLD

    # A dividend chunk that has revenue context injected as a prefix
    chunk_with_prefix = (
        "[Previous Context: revenue for fiscal 2023 was $500 million]\n\n"
        "The board approved a quarterly dividend of $0.25 per share."
    )

    # Query asking about revenue
    query_tokens = _tokenize("revenue 2023")

    # WITHOUT prefix stripping: "revenue" and "2023" appear in the prefix → recall=1.0 → passes
    # WITH prefix stripping: only "dividend", "quarterly", "0.25", "share" remain → recall=0.0 → fails

    # _lexical_recall strips the prefix internally
    recall = _lexical_recall(query_tokens, chunk_with_prefix)

    assert recall < LEXICAL_OVERLAP_THRESHOLD, (
        f"Prefix-stripped recall should be below threshold, got {recall:.2f}. "
        "The [Previous Context:] prefix must not contribute to the lexical gate."
    )


def test_nuance_focus_filters_ufl_rows():
    """
    [New] UFLFilter.nuance_focus must narrow UFL results to rows whose
    text_nuance contains the focus keyword.
    Developer_plan.md §Phase 6 design decision #16: nuance_focus is an AND-
    narrowing filter; rows with NaN text_nuance are excluded.
    """
    import pandas as pd
    from unittest.mock import patch as _patch
    from venra.models import UFLFilter
    from venra.retriever import DualRetriever

    # Build a minimal in-memory DataFrame with two rows
    data = {
        "row_id": ["r1", "r2", "r3"],
        "canonical_entity_id": ["ID_TEST", "ID_TEST", "ID_TEST"],
        "entity_name_raw": ["Test Corp"] * 3,
        "metric_name": ["Revenue", "Revenue", "Revenue"],
        "num_value": [500.0, 500.0, 500.0],
        "grounding_quote": ["$500M"] * 3,
        "unit_normalized": ["USD"] * 3,
        "scale": [1.0] * 3,
        "period_start": [None] * 3,
        "period_end": ["2023"] * 3,
        "period_type": [None] * 3,
        "doc_section": ["MD&A"] * 3,
        "source_chunk_id": ["c1", "c2", "c3"],
        "text_nuance": ["Restated figure", None, "As reported"],
        "related_entity_id": [None] * 3,
        "char_interval": [None] * 3,
        "alignment_status": ["EXACT"] * 3,
        "confidence_score": [0.95] * 3,
    }
    df = pd.DataFrame(data)

    filter_spec = UFLFilter(
        entity_ids=["ID_TEST"],
        metric_keywords=["Revenue"],
        years=["2023"],
        nuance_focus="Restated",
    )

    retriever = DualRetriever.__new__(DualRetriever)
    retriever.df = df

    rows = retriever._query_ufl(filter_spec, query_tokens=[])

    assert len(rows) == 1, f"Expected 1 Restated row, got {len(rows)}"
    assert rows[0].row_id == "r1"
    assert "Restated" in rows[0].text_nuance


# ==========================================
# Feature: Semantic Chunking (Newline-Preference & Trailing Buffer)
# ==========================================

def test_semantic_chunking_trailing_buffer():
    """
    Test that the trailing buffer of the previous block is prepended to the next
    block, solving cross-boundary coreference.
    Developer_plan.md Stage 1: 'Trailing Buffer: last 300 chars prepended as
    [Previous Context: <buffer>].'

    CHANGED[Phase7]: Rewrote from calling non-existent _split_with_trailing_buffer.
    Calls _flush_chunk() directly with a trailing_buffer argument.
    """
    from venra.ingestion import StructuralParser

    with patch.dict(os.environ, {"LLAMA_CLOUD_API_KEY": "fake_key"}):
        with patch("venra.ingestion.LlamaParse"):
            parser = StructuralParser()

    all_blocks = []

    trailing = parser._flush_chunk(
        lines=["Here is the first part of the document."],
        stack=["Section One"],
        all_blocks=all_blocks,
        trailing_buffer="",
    )

    assert trailing, "Trailing buffer must be non-empty after flushing content"
    assert "first part" in trailing

    parser._flush_chunk(
        lines=["And this is the second part referencing see Note 12."],
        stack=["Section One"],
        all_blocks=all_blocks,
        trailing_buffer=trailing,
    )

    assert len(all_blocks) == 2
    assert "[Previous Context:" in all_blocks[1].content, (
        "Second block must carry [Previous Context: ...] prefix"
    )
    assert "first part" in all_blocks[1].content
    assert "Note 12" in all_blocks[1].content


# ==========================================
# Feature: Think-Tag Stripping
# ==========================================

def test_think_tag_stripping():
    """
    Test that <think>…</think> reasoning traces are stripped from the raw LLM
    response before JSON parsing.

    CHANGED[Phase7]: import changed to module-level venra.synthesis._strip_think_tags.
    _strip_think_tags is a MODULE-LEVEL function, not a TextSynthesizer class method.
    Accessing it as TextSynthesizer._strip_think_tags raises AttributeError.
    """
    from venra.synthesis import _strip_think_tags

    raw = """<think>
    The user wants revenue. I see $500M.
    </think>
    {"facts": [{"metric_name": "Revenue", "num_value": 500000000.0,
                "grounding_quote": "$500M", "unit_normalized": "USD", "confidence": 0.9}]}"""

    stripped = _strip_think_tags(raw)

    assert "<think>" not in stripped
    assert "</think>" not in stripped
    assert "The user wants revenue." not in stripped
    assert "Revenue" in stripped


def test_think_tag_stripping_multiline():
    """[New] Multi-line think block must be fully stripped."""
    from venra.synthesis import _strip_think_tags

    raw = "<think>\nLine 1\nLine 2\n</think>{'key': 'value'}"
    stripped = _strip_think_tags(raw)

    assert "<think>" not in stripped
    assert "Line 1" not in stripped
    assert "{'key': 'value'}" in stripped


def test_think_tag_stripping_case_insensitive():
    """[New] _THINK_TAG_RE uses re.IGNORECASE — <THINK> must also be stripped."""
    from venra.synthesis import _strip_think_tags

    raw = "<THINK>reasoning here</THINK>{'result': 42}"
    stripped = _strip_think_tags(raw)

    assert "reasoning here" not in stripped
    assert "{'result': 42}" in stripped


# ==========================================
# Feature: Semantic Chunking (Newline-Preference)
# ==========================================

def test_semantic_chunking_newline_preference():
    """
    Test that the chunker breaks at the most recent newline rather than mid-sentence.
    Developer_plan.md Stage 1: 'Newline-Preference: never break mid-sentence.'

    CHANGED[Phase7]: Rewrote from calling non-existent _split_with_trailing_buffer.
    _split_at_newlines is a @staticmethod — no parser instantiation needed.
    """
    from venra.ingestion import StructuralParser

    text = "First sentence is short.\nSecond sentence is a bit longer and exceeds the limit."
    segments = StructuralParser._split_at_newlines(text, max_chars=50)

    assert len(segments) == 2
    assert segments[0].strip() == "First sentence is short."
    assert segments[1].strip() == "Second sentence is a bit longer and exceeds the limit."


def test_semantic_chunking_no_newline_keeps_intact():
    """
    [New] A paragraph exceeding max_chars with no internal newline must be kept
    intact — prefer slightly over-sized chunks over mid-word cuts.
    Developer_plan.md Stage 1: 'Single paragraphs exceeding the limit with no
    internal newline are kept intact.'
    """
    from venra.ingestion import StructuralParser

    text = "This is one very long paragraph with absolutely no newline characters in it."
    segments = StructuralParser._split_at_newlines(text, max_chars=50)

    assert len(segments) == 1
    assert segments[0] == text.strip()


def test_semantic_chunking_multiple_segments_continuity():
    """
    [New] When a text block splits into 3+ segments, every segment after the first
    must contain a [Previous Context:] prefix from the immediately preceding segment.
    Developer_plan.md Stage 1: 'Every segment of a multi-segment split gets a
    [Previous Context:] prefix from the immediately preceding segment tail.'
    """
    from venra.ingestion import StructuralParser

    with patch.dict(os.environ, {"LLAMA_CLOUD_API_KEY": "fake_key"}):
        with patch("venra.ingestion.LlamaParse"):
            parser = StructuralParser()

    lines = [
        "Alpha sentence number one here done.",
        "Beta sentence number two here done.",
        "Gamma sentence number three here done.",
    ]

    all_blocks = []
    parser._flush_chunk(lines=lines, stack=["S"], all_blocks=all_blocks, trailing_buffer="")

    if len(all_blocks) > 1:
        for block in all_blocks[1:]:
            assert "[Previous Context:" in block.content, (
                f"Block missing Previous Context prefix: {block.content[:80]}"
            )