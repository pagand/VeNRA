import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from venra.synthesis import TextSynthesizer
from venra.models import TextBlock

@pytest.mark.asyncio
async def test_extract_facts_with_proposals_backward_compatibility():
    synth = TextSynthesizer("ID_TEST")
    
    # Mock _single_pass to return an empty list so it doesn't call an API
    synth._single_pass = AsyncMock(return_value=[])
    
    block = TextBlock(content="This is a test block with enough characters.", section_path=["A"])
    
    # 1. Test new method
    accepted, proposed, failed = await synth.extract_facts_with_proposals(block)
    assert isinstance(accepted, list)
    assert isinstance(proposed, list)
    assert isinstance(failed, bool)
    
    # 2. Test old method (should just return the first element)
    accepted_only = await synth.extract_facts(block)
    assert isinstance(accepted_only, list)
    assert accepted_only == accepted

if __name__ == "__main__":
    asyncio.run(test_extract_facts_with_proposals_backward_compatibility())
