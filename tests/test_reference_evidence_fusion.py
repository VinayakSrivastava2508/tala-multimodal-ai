"""Tests for src/fusion/reference_evidence.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.reference_evidence import classify_reference_evidence, unsupported_reference_result

CONFIG = {"relevance_thresholds": {"reference_relevance_min": 0.35}}


def test_below_threshold_is_irrelevant_reference():
    result = classify_reference_evidence("claim", "chunk", "", "sustainability_page", "", "www.wearetala.com", 0.1, CONFIG)
    assert result["reference_category"] == "irrelevant_reference"
    assert result["stance"] == "not_applicable"


def test_official_disclosure_never_labelled_independent():
    """Part 6 rule 2: official brand references establish consistency with
    disclosure, never independent real-world proof."""
    result = classify_reference_evidence("claim", "chunk", "", "sustainability_page", "", "www.wearetala.com", 0.5, CONFIG)
    assert result["reference_category"] == "supported_by_official_disclosure"
    assert result["source_independence"] == "self_reported"
    assert result["permitted_evidence_role"] == "disclosure_only"


def test_certification_registry_gives_independent_support():
    result = classify_reference_evidence("claim", "chunk", "", "certification", "goodonyou.eco", "goodonyou.eco", 0.5, CONFIG)
    assert result["reference_category"] == "independently_supported"
    assert result["source_independence"] == "independent"


def test_certification_type_without_authority_is_not_independent():
    """A row merely classified 'certification' by keyword, with no actual
    registry authority attached, must not be granted independent status."""
    result = classify_reference_evidence("claim", "chunk", "", "certification", "", "www.wearetala.com", 0.5, CONFIG)
    assert result["reference_category"] != "independently_supported"


def test_conflicting_percentages_flagged_as_conflict_not_challenge_by_default():
    result = classify_reference_evidence(
        "Made from 80% recycled polyester", "chunk text", "50%",
        "material_specification", "", "www.wearetala.com", 0.5, CONFIG,
    )
    assert result["reference_category"] == "conflicting_reference_evidence"
    assert result["stance"] == "challenges"


def test_absent_number_in_reference_is_not_a_conflict():
    """Part 6 rule 8/9: absence from a document is not contradiction."""
    result = classify_reference_evidence(
        "Made from 80% recycled polyester", "General sustainability text with no numbers.", "",
        "material_specification", "", "www.wearetala.com", 0.5, CONFIG,
    )
    assert result["reference_category"] != "conflicting_reference_evidence"


def test_matching_percentages_do_not_conflict():
    result = classify_reference_evidence(
        "Made from 80% recycled polyester", "chunk mentioning 80%", "80%",
        "material_specification", "", "www.wearetala.com", 0.5, CONFIG,
    )
    assert result["reference_category"] != "conflicting_reference_evidence"


def test_unsupported_reference_result_is_not_a_contradiction():
    result = unsupported_reference_result()
    assert result["reference_category"] == "unsupported_in_available_references"
    assert result["stance"] == "not_applicable"
