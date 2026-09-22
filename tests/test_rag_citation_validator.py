"""Tests for src/rag/citation_validator.py -- automated post-hoc validation of
a Gemini response against the evidence it was actually given."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.citation_validator import extract_cited_ids, validate_response

EVIDENCE = [
    {"id": "text::customer_experience::ce_rev_0001", "document": "Great fit.", "metadata": {
        "evidence_id": "ce_rev_0001", "evidence_strength": "strong", "self_reported": False, "independent_source": False,
        "source_type": "customer_experience", "local_asset_path": "", "source_url": "https://example.com/review",
    }},
    {"id": "text::official_claims::oc_0001", "document": "We use recycled materials.", "metadata": {
        "evidence_id": "oc_0001", "evidence_strength": "strong", "self_reported": True, "independent_source": False,
        "source_type": "official_claims", "local_asset_path": "", "source_url": "https://www.wearetala.com/",
    }},
]


def _response(**overrides):
    base = {
        "concise_answer": "The claim is supported by evidence [text::customer_experience::ce_rev_0001].",
        "claim_assessment": "Aligned based on [text::official_claims::oc_0001].",
        "supporting_evidence_ids": ["text::customer_experience::ce_rev_0001", "text::official_claims::oc_0001"],
        "challenging_evidence_ids": [], "contextual_evidence_ids": [],
        "evidence_gaps": [], "confidence_explanation": "High confidence.", "caveats": [],
    }
    base.update(overrides)
    return base


def test_extract_cited_ids_finds_bracketed_ids():
    ids = extract_cited_ids("See [text::a::1] and [text::b::2].")
    assert ids == ["text::a::1", "text::b::2"]


def test_valid_response_passes_all_checks():
    result = validate_response(_response(), EVIDENCE, true_fusion_label="aligned")
    assert result["passed"] is True
    assert result["failures"] == []


def test_hallucinated_citation_is_rejected():
    resp = _response(concise_answer="Supported by [text::fabricated::9999].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["passed"] is False
    assert result["checks"]["citations_exist_in_context"] is False


def test_hallucinated_id_in_supporting_list_is_rejected():
    resp = _response(supporting_evidence_ids=["text::customer_experience::ce_rev_0001", "text::nonexistent::123"])
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["passed"] is False


def test_production_label_preserved_when_stated_correctly():
    resp = _response(claim_assessment="This claim remains aligned, supported by [text::official_claims::oc_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["production_label_preserved"] is True


def test_production_label_overwritten_is_rejected():
    resp = _response(claim_assessment="This claim is actually divergent based on [text::official_claims::oc_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["production_label_preserved"] is False
    assert result["passed"] is False


def test_self_reported_evidence_called_independent_is_rejected():
    resp = _response(claim_assessment="This is independently verified by [text::official_claims::oc_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["self_reported_not_called_independent"] is False


def test_self_reported_evidence_correctly_described_as_lacking_independence_passes():
    resp = _response(confidence_explanation="No independent verification exists for [text::official_claims::oc_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["self_reported_not_called_independent"] is True


def test_overclaimed_video_inspection_is_rejected():
    resp = _response(concise_answer="I watched the video and confirmed [text::customer_experience::ce_rev_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["no_overclaimed_inspection"] is False


def test_extract_cited_ids_handles_multiple_ids_in_one_bracket():
    """Gemini sometimes writes [id1, id2] instead of [id1][id2] despite the
    prompt asking for one citation per bracket -- both IDs must still be
    extracted, not silently dropped."""
    ids = extract_cited_ids("See [text::a::1, text::b::2] for details.")
    assert ids == ["text::a::1", "text::b::2"]


def test_self_reported_check_handles_multiple_ids_in_one_bracket():
    resp = _response(caveats=["The official brand chunks [text::official_claims::oc_0001] are self-reported statements."])
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["self_reported_not_called_independent"] is True


def test_raw_id_citation_without_namespace_prefix_is_not_hallucinated():
    """Regression: a smaller/lighter Gemini model sometimes drops the
    text::<source_type>:: / visual::<modality>:: namespace prefix and cites
    the raw underlying evidence_id directly. That still refers to genuinely
    retrieved evidence and must not be flagged as fabricated."""
    resp = _response(concise_answer="Supported by review evidence [ce_rev_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["citations_exist_in_context"] is True
    assert result["passed"] is True


def test_raw_id_that_matches_nothing_retrieved_is_still_hallucinated():
    """A raw-looking ID that doesn't correspond to ANY retrieved evidence item
    is a genuine fabrication, not a formatting quirk, and must still fail."""
    resp = _response(concise_answer="Supported by [oc_9999], a document never retrieved.")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["citations_exist_in_context"] is False
    assert result["passed"] is False


def test_raw_id_self_reported_check_still_catches_violation():
    """The raw-ID normalisation must not accidentally suppress the
    self-reported/independent check -- a raw-ID citation of self-reported
    evidence wrongly called independent must still fail."""
    resp = _response(claim_assessment="This is independently verified by [oc_0001].")
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["self_reported_not_called_independent"] is False


def test_no_citation_bleed_across_response_fields():
    """Regression: a citation at the end of one field must never be matched
    against an unrelated 'independent' mention in the START of a different
    field just because they were naively string-joined."""
    resp = _response(
        claim_assessment="Supported by official brand copy [text::official_claims::oc_0001].",
        confidence_explanation="Independent verification of this specific point is not available.",
    )
    result = validate_response(resp, EVIDENCE, true_fusion_label="aligned")
    assert result["checks"]["self_reported_not_called_independent"] is True
