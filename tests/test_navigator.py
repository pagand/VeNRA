import pytest
import os
import json
from unittest.mock import MagicMock, patch
from venra.navigator import Navigator
from venra.models import RetrievalPlan

@pytest.fixture
def mock_schema_file(tmp_path):
    schema_data = {
        "entities": [
            {"id": "ID_TDG", "official_name": "TransDigm Group Incorporated", "aliases": ["TransDigm"]}
        ],
        "metrics": ["Net Sales", "Net Income", "Total Debt", "EBITDA"]
    }
    schema_path = os.path.join(tmp_path, "schema_summary.json")
    with open(schema_path, "w") as f:
        json.dump(schema_data, f)
    return schema_path

@pytest.mark.asyncio
async def test_navigator_generates_plan(mock_schema_file):
    """
    Test that Navigator correctly parses a query into a RetrievalPlan using mocked LLM.
    """
    mock_plan = RetrievalPlan(
        strategy="HYBRID",
        ufl_query={
            "entity_ids": ["ID_TDG"],
            "metric_keywords": ["Net Sales", "Revenue"],
            "years": ["2023"],
            "nuance_focus": None
        },
        vector_hypothesis="Net sales for the fiscal year ended September 30, 2023 were...",
        vector_keywords=["Net sales", "2023", "Revenue"],
        reasoning="User is asking for 2023 sales for TransDigm."
    )
    
    with patch("venra.navigator.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_plan
        
        nav = Navigator(api_key="fake", schema_path=mock_schema_file)
        plan = await nav.navigate("What were TransDigm's sales in 2023?")
        
        assert plan.strategy == "HYBRID"
        assert "ID_TDG" in plan.ufl_query.entity_ids
        assert "Net Sales" in plan.ufl_query.metric_keywords
        
        # Verify schema was loaded into context
        call_args = mock_client.chat.completions.create.call_args
        system_msg = call_args[1]['messages'][0]['content']
        assert "ID_TDG" in system_msg
        assert "Net Sales" in system_msg

@pytest.mark.asyncio
async def test_navigator_fallback_on_error(mock_schema_file):
    """
    Test that Navigator returns a fallback plan if the LLM call fails.
    """
    with patch("venra.navigator.instructor.from_openai") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        nav = Navigator(api_key="fake", schema_path=mock_schema_file)
        plan = await nav.navigate("Some query")
        
        assert plan.strategy == "TEXT_ONLY"
        assert "Navigation error" in plan.reasoning
