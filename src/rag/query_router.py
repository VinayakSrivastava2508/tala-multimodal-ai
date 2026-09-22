"""Day 3B: deterministic, inspectable intent classification and modality
routing. No LLM decides which corpus is searched -- every rule here is a
plain keyword/category lookup that can be read and unit-tested directly."""

from __future__ import annotations

from typing import Dict, List

from src.rag.schemas import claim_category_groundability, is_visually_groundable_category, load_fusion_rules

INTENTS = (
    "claim_validation", "customer_experience", "quality", "fit_sizing", "durability_care",
    "materials", "sustainability_responsibility", "returns", "creator_strategy",
    "competitor_comparison", "evidence_gap_analysis",
)

# Ordered so more specific intents are checked before broad ones; every intent
# whose keywords match is returned (a question can carry multiple intents).
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "fit_sizing": ["fit", "sizing", "size guide", "size chart", "true to size", "runs small", "runs large"],
    "durability_care": ["durability", "durable", "wash", "care", "pilling", "tear", "rip", "wear and tear", "last", "longevity"],
    "materials": ["material", "fabric", "recycled", "polyester", "nylon", "composition"],
    "sustainability_responsibility": ["sustainab", "responsib", "emission", "certif", "ethical", "labour", "labor", "supply chain", "circular"],
    "returns": ["return", "refund", "exchange"],
    "customer_experience": ["customer", "review", "reviewer", "complaint", "experience says", "what do customers"],
    "creator_strategy": ["creator", "influencer", "partnership", "sponsor", "platform strategy"],
    "competitor_comparison": ["adanola", "girlfriend collective", "oner active", "competitor", "compare", "versus", "vs "],
    "evidence_gap_analysis": ["lack", "gap", "no evidence", "missing evidence", "insufficient evidence", "unvalidated", "no independent"],
    "quality": ["quality", "well made", "well-made", "construction", "craftsmanship"],
    "claim_validation": ["claim", "classified as", "why is", "mixed", "divergent", "aligned", "verify", "validated", "supported by"],
}


def classify_intents(question: str) -> List[str]:
    """Deterministic keyword-based intent classification. Returns at least
    one intent (['claim_validation'] as the conservative default) so routing
    never falls through with zero modalities searched."""
    lowered = question.lower()
    matched = [intent for intent, keywords in INTENT_KEYWORDS.items() if any(kw in lowered for kw in keywords)]
    return matched or ["claim_validation"]


# ── Modality routing ────────────────────────────────────────────────────────────

# Intents that can ever justify visual (image/video) retrieval -- reuses the
# Day 3A visually-groundable-category philosophy applied at the intent level
# for open questions (no specific claim_category is known yet).
VISUALLY_ELIGIBLE_INTENTS = {"quality", "fit_sizing", "materials", "durability_care", "claim_validation"}
VIDEO_ELIGIBLE_INTENTS = {"quality", "fit_sizing", "durability_care", "claim_validation"}
REFERENCE_ELIGIBLE_INTENTS = {
    "claim_validation", "materials", "sustainability_responsibility", "returns", "durability_care", "fit_sizing", "quality",
}


def route_modalities_for_intents(intents: List[str]) -> List[str]:
    """Text is always searched. Image/video/reference are added only when an
    eligible intent is present -- never indiscriminately."""
    modalities = ["text"]
    if any(i in VISUALLY_ELIGIBLE_INTENTS for i in intents):
        modalities.append("image")
    if any(i in VIDEO_ELIGIBLE_INTENTS for i in intents):
        modalities.append("video")
    if any(i in REFERENCE_ELIGIBLE_INTENTS for i in intents):
        modalities.append("reference")
    return modalities


def route_modalities_for_claim(claim_category: str, fusion_rules: dict | None = None) -> List[str]:
    """For claim-investigation mode: text and reference are always searched;
    image/video are added ONLY when the claim's category is visually
    groundable per the authoritative Day 3A rule
    (configs/fusion_rules.yaml::visual_groundability.claim_category_groundability).
    A labour/emissions/manufacturing/circularity/other claim never gets
    image/video evidence, regardless of any visual similarity score."""
    rules = fusion_rules or load_fusion_rules()
    modalities = ["text", "reference"]
    if is_visually_groundable_category(claim_category, rules):
        modalities.append("image")
        modalities.append("video")
    return modalities


def groundability_tier_for_claim(claim_category: str, fusion_rules: dict | None = None) -> str:
    return claim_category_groundability(claim_category, fusion_rules or load_fusion_rules())
