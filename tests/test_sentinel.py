import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.venra.main import app

client = TestClient(app)

@pytest.fixture
def mock_judge_api():
    with patch("src.venra.sentinel.requests.post") as mock_post:
        yield mock_post

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_verify_endpoint_success(mock_judge_api):
    # Mock a successful response from the external Judge API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "grounded": 0.95,
        "common": 0.03,
        "hallucination": 0.02,
        "top_class": "GROUNDED"
    }
    mock_judge_api.return_value = mock_response

    payload = {
        "query": "What is the revenue?",
        "answer_text": "Revenue was $50M. This is a good growth.",
        "context": "The 10K shows revenue of $50 million.",
        "trace": "calc: 50"
    }

    response = client.post("/verify", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "overall_groundedness_score" in data
    assert len(data["sentence_results"]) == 2
    assert data["sentence_results"][0]["label"] == "GROUNDED"
    assert data["overall_groundedness_score"] == 1.0 # Both sentences mocked as grounded

def test_verify_endpoint_mixed_results(mock_judge_api):
    # Mock different responses for different sentences
    # We can use side_effect to return different values for each call
    mock_grounded = MagicMock()
    mock_grounded.status_code = 200
    mock_grounded.json.return_value = {
        "grounded": 0.9, "common": 0.05, "hallucination": 0.05, "top_class": "GROUNDED"
    }

    mock_hallucination = MagicMock()
    mock_hallucination.status_code = 200
    mock_hallucination.json.return_value = {
        "grounded": 0.1, "common": 0.1, "hallucination": 0.8, "top_class": "HALLUCINATION"
    }

    mock_judge_api.side_effect = [mock_grounded, mock_hallucination]

    payload = {
        "query": "test query",
        "answer_text": "Sentence 1. Sentence 2.",
        "context": "context",
        "trace": "trace"
    }

    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall_groundedness_score"] == 0.5
    assert data["sentence_results"][0]["label"] == "GROUNDED"
    assert data["sentence_results"][1]["label"] == "HALLUCINATION"
