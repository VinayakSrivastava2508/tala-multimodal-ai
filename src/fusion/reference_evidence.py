"""Day 3A Part 11: reference evidence pipeline.

Applies Part 6 rules 2-3 to reference_document_chunks.csv matches already
computed in scripts/build_claim_multimodal_candidates.py::match_reference_evidence:
official brand references (TALA's own site) may only ever support
"consistency with brand disclosure" -- never "independent, real-world-verified
performance" -- while a recognised third-party certification registry may
independently confirm the specific certification status it states.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

REFERENCE_RELEVANCE_CATEGORIES = (
    "supported_by_official_disclosure", "independently_supported",
    "unsupported_in_available_references", "conflicting_reference_evidence",
    "irrelevant_reference",
)

_PERCENT_RX = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")


def _numbers_conflict(claim_text: str, chunk_numerical_claims: str) -> bool:
    """Best-effort numeric-consistency check (Part 11). Only flags a conflict
    when BOTH the claim and the chunk state an explicit percentage AND they
    disagree -- absence of a matching number is never treated as conflict
    (Part 6 rule 8/9: absence is not contradiction)."""
    claim_pcts = set(_PERCENT_RX.findall(str(claim_text)))
    chunk_pcts = set(str(chunk_numerical_claims).split(";")) & set(_PERCENT_RX.findall(str(chunk_numerical_claims)))
    if not claim_pcts or not chunk_pcts:
        return False
    return claim_pcts.isdisjoint(chunk_pcts) and len(claim_pcts) > 0 and len(chunk_pcts) > 0


def classify_reference_evidence(
    claim_text: str, chunk_text: str, chunk_numerical_claims: str,
    reference_type: str, certification_authority: str, source_domain: str,
    relevance_score: float, config: dict,
) -> Dict:
    """Returns reference-specific fields: category, stance, source_independence,
    permitted_evidence_role."""
    threshold = config["relevance_thresholds"]["reference_relevance_min"]
    if relevance_score < threshold:
        return {
            "reference_category": "irrelevant_reference", "stance": "not_applicable",
            "stance_score": 0.0, "source_independence": "unclear",
            "permitted_evidence_role": "context_only",
        }

    is_certification_registry = bool(certification_authority) and reference_type == "certification"
    conflict = _numbers_conflict(claim_text, chunk_numerical_claims)

    if conflict:
        return {
            "reference_category": "conflicting_reference_evidence", "stance": "challenges",
            "stance_score": -0.5, "source_independence": "independent" if is_certification_registry else "self_reported",
            "permitted_evidence_role": "material_challenge",
        }

    if is_certification_registry:
        # Part 6 rule 3: independent support ONLY for the certification status
        # explicitly confirmed -- never generalised to unrelated claim content.
        return {
            "reference_category": "independently_supported", "stance": "supports",
            "stance_score": 0.7, "source_independence": "independent",
            "permitted_evidence_role": "independent_support",
        }

    # Part 6 rule 2: official brand disclosure establishes consistency with a
    # documented policy/stated target -- never independent real-world proof.
    return {
        "reference_category": "supported_by_official_disclosure", "stance": "supports",
        "stance_score": 0.4, "source_independence": "self_reported",
        "permitted_evidence_role": "disclosure_only",
    }


def unsupported_reference_result() -> Dict:
    """Used when a claim's category is reference-eligible but no chunk cleared
    the relevance threshold -- this is 'unsupported in available references',
    never converted into a challenge/contradiction (Part 11 rule)."""
    return {
        "reference_category": "unsupported_in_available_references", "stance": "not_applicable",
        "stance_score": 0.0, "source_independence": "unclear", "permitted_evidence_role": "context_only",
    }
