"""Day 3A governance correction: final presentation-restriction logic.

This is a pure, post-fusion classification layer. It never changes evidence
units, fusion labels (automated_label), confidence, thresholds, NLI outputs,
or sensitivity results -- it only decides how a claim's ALREADY-COMPUTED
result may be presented, using signals gathered from fusion, sensitivity, and
provenance evaluation.
"""

from __future__ import annotations

from typing import Tuple

DO_NOT_PRESENT = "do_not_present_as_conclusion"
STRONG_CAVEAT = "present_with_strong_caveat"
EVIDENCE_GAP_ONLY = "present_as_evidence_gap_finding_only"
MAY_PRESENT = "may_present_as_automated_finding"

GOVERNANCE_RESTRICTIONS = (DO_NOT_PRESENT, STRONG_CAVEAT, EVIDENCE_GAP_ONLY, MAY_PRESENT)


def classify_presentation_restriction(
    automated_label: str,
    self_reported_only: bool,
    single_evidence_dependency: bool,
    sensitive_to_weights: bool,
    provenance_failure: bool,
    groundability_failure: bool,
) -> Tuple[str, str, bool]:
    """Returns (presentation_restriction, presentation_restriction_reason,
    evidence_gap_finding) applying this fixed precedence:

    1. provenance or eligibility (visual/video groundability) failure
       -> do_not_present_as_conclusion
    2. mixed/divergent, sensitivity-unstable, self-reported-only, or
       single-evidence-dependent -> present_with_strong_caveat
    3. stable, eligible, multi-source evidence -> may_present_as_automated_finding
    4. insufficient_evidence -> present_as_evidence_gap_finding_only (never a
       substantive claim conclusion)
    """
    # Tier 1: hard failures always win, regardless of label.
    reasons = []
    if provenance_failure:
        reasons.append("provenance_failure")
    if groundability_failure:
        reasons.append("visual_video_groundability_failure")
    if reasons:
        return DO_NOT_PRESENT, ";".join(reasons), False

    # Tier 2: caveat-triggering conditions.
    reasons = []
    if automated_label in ("mixed", "divergent"):
        reasons.append(f"automated_label={automated_label}")
    if sensitive_to_weights:
        reasons.append("sensitive_to_weights")
    if self_reported_only:
        reasons.append("self_reported_only")
    if single_evidence_dependency:
        reasons.append("single_evidence_dependency")
    if reasons:
        return STRONG_CAVEAT, ";".join(reasons), False

    # Tier 4: insufficient evidence is never a substantive conclusion.
    if automated_label == "insufficient_evidence":
        return EVIDENCE_GAP_ONLY, "insufficient_evidence", True

    # Tier 3: everything else -- stable, eligible, multi-source, no red flags.
    return MAY_PRESENT, "stable_multi_source_eligible_evidence", False
