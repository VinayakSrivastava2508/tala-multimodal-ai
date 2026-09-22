"""Tests for src/rag/gemini_generator.py -- the ONLY generation path. Mocks
the Gemini client; no real API calls in the unit test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.gemini_generator import (
    GeminiConfigError, GeminiGenerationError, generate_answer, load_gemini_config, _parse_structured_output,
)

VALID_JSON = json.dumps({
    "concise_answer": "x", "claim_assessment": "y", "supporting_evidence_ids": [],
    "challenging_evidence_ids": [], "contextual_evidence_ids": [], "evidence_gaps": [],
    "confidence_explanation": "z", "caveats": [],
})

EVIDENCE = [{"id": "text::a::1", "document": "doc", "metadata": {"modality": "text", "source_type": "official_claims", "evidence_strength": "strong"}}]


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr("src.rag.gemini_generator.load_dotenv", lambda *a, **k: None)
    with pytest.raises(GeminiConfigError):
        load_gemini_config()


def test_missing_model_raises_config_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr("src.rag.gemini_generator.load_dotenv", lambda *a, **k: None)
    with pytest.raises(GeminiConfigError):
        load_gemini_config()


def test_generate_answer_raises_config_error_without_calling_api(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.rag.gemini_generator.load_dotenv", lambda *a, **k: None)
    with pytest.raises(GeminiConfigError):
        generate_answer("question", EVIDENCE)


def test_generate_answer_success_with_mocked_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr("src.rag.gemini_generator.load_dotenv", lambda *a, **k: None)

    mock_response = MagicMock()
    mock_response.text = VALID_JSON
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = generate_answer("question", EVIDENCE)

    assert result["concise_answer"] == "x"
    assert result["_model_used"] == "gemini-3.8-flash"
    mock_client.models.generate_content.assert_called_once()


def test_generate_answer_propagates_api_error_without_fallback(monkeypatch):
    """Regression: a Gemini API failure must raise, never silently substitute
    a deterministic or template answer."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr("src.rag.gemini_generator.load_dotenv", lambda *a, **k: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("503 UNAVAILABLE")

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(GeminiGenerationError):
            generate_answer("question", EVIDENCE)


def test_parse_structured_output_rejects_invalid_json():
    with pytest.raises(GeminiGenerationError):
        _parse_structured_output("not json at all")


def test_parse_structured_output_rejects_missing_fields():
    with pytest.raises(GeminiGenerationError):
        _parse_structured_output(json.dumps({"concise_answer": "x"}))


def test_parse_structured_output_strips_markdown_fences():
    fenced = f"```json\n{VALID_JSON}\n```"
    parsed = _parse_structured_output(fenced)
    assert parsed["concise_answer"] == "x"


def test_no_engagement_prediction_fields_in_required_output():
    from src.rag.gemini_generator import REQUIRED_OUTPUT_FIELDS
    forbidden = {"predicted_engagement", "engagement_score", "predicted_likes", "predicted_views"}
    assert forbidden.isdisjoint(REQUIRED_OUTPUT_FIELDS)
