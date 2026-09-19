"""Day 3A NLI disagreement diagnostic (Parts A-E).

Everything here is read/derived from the model's own runtime configuration
and from transparent, code-driven rules -- never from assumed label
positions and never by manually forcing agreement on real claims.
"""

from __future__ import annotations

import platform
from typing import Dict, List

from src.fusion.text_evidence import (
    NLI_DECISION_THRESHOLD, NLI_HYPOTHESIS_FIELD, NLI_MODEL_NAME, NLI_PREMISE_FIELD,
    _load_nli_model, nli_label_from_probs, nli_label_indices, nli_predict_probs,
)

# ── Part A: implementation audit ───────────────────────────────────────────────


def implementation_audit_row() -> Dict:
    model = _load_nli_model()
    label2id = nli_label_indices()
    id2label = {v: k for k, v in label2id.items()}
    return {
        "model_name": NLI_MODEL_NAME,
        "model_revision": getattr(model.config, "_name_or_path", NLI_MODEL_NAME),
        "transformers_version": getattr(model.config, "transformers_version", ""),
        "tokenizer_class": type(model.tokenizer).__name__,
        "framework": f"pytorch (sentence-transformers CrossEncoder, {platform.python_version()})",
        "config_id2label": str(id2label),
        "config_label2id": str(label2id),
        "entailment_label_id": label2id["entailment"],
        "contradiction_label_id": label2id["contradiction"],
        "neutral_label_id": label2id["neutral"],
        "premise_field": NLI_PREMISE_FIELD,
        "hypothesis_field": NLI_HYPOTHESIS_FIELD,
        "max_seq_length": getattr(model, "max_seq_length", None),
        "truncation_behaviour": "truncation=True, strategy=longest_first (HuggingFace tokenizer default via sentence-transformers CrossEncoder.preprocess)",
        "inference_device": str(getattr(model.model, "device", "cpu")),
        "decision_threshold_entailment_min": NLI_DECISION_THRESHOLD,
        "decision_threshold_contradiction_min": NLI_DECISION_THRESHOLD,
        "decision_rule": "entailment>=threshold -> entailment; elif contradiction>=threshold -> contradiction; else neutral",
        "label_mapping_verified_against_config": True,
    }


# ── Part B: synthetic calibration set ──────────────────────────────────────────
# 18 cases (6 entailment / 6 contradiction / 6 neutral) spanning the project's
# claim-category taxonomy. Never copied from real claim/evidence text.

SYNTHETIC_CALIBRATION_CASES: List[Dict] = [
    # -- entailment (evidence genuinely supports the claim) --
    {"case_id": "syn_ent_fit", "category": "fit", "expected_label": "entailment",
     "claim_text": "Our leggings offer a supportive, flattering fit for every body shape.",
     "evidence_text": "I found these leggings incredibly flattering and they fit my body perfectly, very supportive too."},
    {"case_id": "syn_ent_sizing", "category": "sizing", "expected_label": "entailment",
     "claim_text": "Our size guide is accurate and true to size.",
     "evidence_text": "I ordered my usual size and it fit true to size exactly as the size guide said."},
    {"case_id": "syn_ent_durability", "category": "durability", "expected_label": "entailment",
     "claim_text": "These leggings are built to last through years of intense workouts.",
     "evidence_text": "I've worn these leggings for two years of daily gym sessions and they still look brand new."},
    {"case_id": "syn_ent_returns", "category": "returns", "expected_label": "entailment",
     "claim_text": "We offer a hassle-free 30-day return policy.",
     "evidence_text": "I returned my order within 30 days and got a full refund with no hassle at all."},
    {"case_id": "syn_ent_materials", "category": "materials", "expected_label": "entailment",
     "claim_text": "Our fabric is made from recycled, low-impact materials.",
     "evidence_text": "The tag confirms this fabric is made from recycled polyester, a low-impact material."},
    {"case_id": "syn_ent_certification", "category": "certification", "expected_label": "entailment",
     "claim_text": "Our products are OEKO-TEX certified for safety.",
     "evidence_text": "This garment carries the OEKO-TEX Standard 100 certification label confirming its safety testing."},

    # -- contradiction (evidence directly negates the claim) --
    {"case_id": "syn_con_fit", "category": "fit", "expected_label": "contradiction",
     "claim_text": "Our leggings offer a supportive, flattering fit for every body shape.",
     "evidence_text": "These leggings were extremely unflattering and did not fit my body shape at all."},
    {"case_id": "syn_con_sizing", "category": "sizing", "expected_label": "contradiction",
     "claim_text": "Our size guide is accurate and true to size.",
     "evidence_text": "The size guide was completely wrong; I ordered my usual size and it was two sizes too small."},
    {"case_id": "syn_con_durability", "category": "durability", "expected_label": "contradiction",
     "claim_text": "These leggings are built to last through years of intense workouts.",
     "evidence_text": "After just two workouts, the fabric ripped and the seams completely fell apart."},
    {"case_id": "syn_con_returns", "category": "returns", "expected_label": "contradiction",
     "claim_text": "We offer a hassle-free 30-day return policy.",
     "evidence_text": "They refused to accept my return within the 30-day window and denied my refund entirely."},
    {"case_id": "syn_con_materials", "category": "materials", "expected_label": "contradiction",
     "claim_text": "Our fabric is made from recycled, low-impact materials.",
     "evidence_text": "The tag actually lists 100% virgin polyester with no recycled content whatsoever."},
    {"case_id": "syn_con_certification", "category": "certification", "expected_label": "contradiction",
     "claim_text": "Our products are OEKO-TEX certified for safety.",
     "evidence_text": "An independent lab test found no OEKO-TEX certification and flagged banned chemical residues."},

    # -- neutral / unrelated --
    {"case_id": "syn_neu_fit", "category": "fit", "expected_label": "neutral",
     "claim_text": "Our leggings offer a supportive, flattering fit for every body shape.",
     "evidence_text": "The delivery arrived two days late but the packaging was nice."},
    {"case_id": "syn_neu_sizing", "category": "sizing", "expected_label": "neutral",
     "claim_text": "Our size guide is accurate and true to size.",
     "evidence_text": "I love the color options available in this collection."},
    {"case_id": "syn_neu_durability", "category": "durability", "expected_label": "neutral",
     "claim_text": "These leggings are built to last through years of intense workouts.",
     "evidence_text": "Customer service replied to my email within an hour, which was great."},
    {"case_id": "syn_neu_returns", "category": "returns", "expected_label": "neutral",
     "claim_text": "We offer a hassle-free 30-day return policy.",
     "evidence_text": "The website has a clean, easy to navigate design."},
    {"case_id": "syn_neu_materials", "category": "materials", "expected_label": "neutral",
     "claim_text": "Our fabric is made from recycled, low-impact materials.",
     "evidence_text": "My order arrived in a nice branded box with a thank-you card."},
    {"case_id": "syn_neu_certification", "category": "certification", "expected_label": "neutral",
     "claim_text": "Our products are OEKO-TEX certified for safety.",
     "evidence_text": "The influencer posted a workout video wearing the leggings at the gym."},
]

ORIENTATIONS = {
    "evidence_premise_claim_hypothesis": ("evidence_text", "claim_text"),  # current production orientation
    "claim_premise_evidence_hypothesis": ("claim_text", "evidence_text"),
}


def run_calibration(orientation_name: str) -> List[Dict]:
    premise_field, hypothesis_field = ORIENTATIONS[orientation_name]
    rows = []
    for case in SYNTHETIC_CALIBRATION_CASES:
        premise = case[premise_field]
        hypothesis = case[hypothesis_field]
        probs = nli_predict_probs(premise, hypothesis)
        predicted = nli_label_from_probs(probs)
        rows.append({
            "orientation": orientation_name, "case_id": case["case_id"], "category": case["category"],
            "expected_label": case["expected_label"], "predicted_label": predicted,
            "correct": predicted == case["expected_label"],
            "entailment_prob": probs["entailment"], "contradiction_prob": probs["contradiction"],
            "neutral_prob": probs["neutral"], "confidence": probs["confidence"],
            "claim_text": case["claim_text"], "evidence_text": case["evidence_text"],
        })
    return rows


def summarize_orientation(rows: List[Dict]) -> Dict:
    labels = ("entailment", "contradiction", "neutral")
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    per_class_acc = {}
    per_class_conf = {}
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for lbl in labels:
        class_rows = [r for r in rows if r["expected_label"] == lbl]
        per_class_acc[lbl] = round(sum(r["correct"] for r in class_rows) / len(class_rows), 4) if class_rows else None
        per_class_conf[lbl] = round(sum(r["confidence"] for r in class_rows) / len(class_rows), 4) if class_rows else None
    for r in rows:
        confusion[r["expected_label"]][r["predicted_label"]] += 1
    return {
        "orientation": rows[0]["orientation"] if rows else "",
        "n": n, "overall_accuracy": round(correct / n, 4) if n else None,
        "entailment_accuracy": per_class_acc["entailment"], "contradiction_accuracy": per_class_acc["contradiction"],
        "neutral_accuracy": per_class_acc["neutral"],
        "mean_confidence_entailment": per_class_conf["entailment"], "mean_confidence_contradiction": per_class_conf["contradiction"],
        "mean_confidence_neutral": per_class_conf["neutral"],
        "confusion_matrix": str(confusion),
    }


# ── Part C: real disagreement decomposition ────────────────────────────────────

_STANCE_TO_NLI_LABEL = {"supports": "entailment", "challenges": "contradiction", "neutral_context": "neutral", "unclear": "neutral"}

NEGATION_WORDS = ("not ", "n't", "never", "no longer", "without", "none", "neither", "nor ", "denied", "refused")
COMPOUND_MARKERS = (" but ", " however ", ";", " although ", " despite ", " while ", " yet ")


def rule_trigger_description(stance: str, polarity: float, subjectivity: float, thresholds: dict) -> str:
    if stance == "supports":
        return f"polarity={polarity:.4f} >= supports_polarity_min={thresholds['supports_polarity_min']}"
    if stance == "challenges":
        return f"polarity={polarity:.4f} <= challenges_polarity_max={thresholds['challenges_polarity_max']}"
    if stance == "unclear":
        return f"subjectivity={subjectivity:.4f} >= {thresholds['unclear_subjectivity_min']} and |polarity|={abs(polarity):.4f} < {thresholds['supports_polarity_min']}"
    return f"polarity={polarity:.4f} between thresholds (neutral_context)"


def categorize_disagreement(
    rule_label: str, nli_label: str, nli_confidence: float, confidence_threshold: float,
    category_gate_passed: bool, truncation_occurred: bool, negation_present: bool,
    compound_claim: bool, orientation_agrees_alt: bool, nli_result_missing: bool,
) -> str:
    """Automatic, transparent disagreement categorisation -- first matching
    rule wins. Never manually overridden per claim."""
    mapped_rule_label = _STANCE_TO_NLI_LABEL.get(rule_label, "neutral")
    if mapped_rule_label == nli_label:
        return ""  # agreement, not a disagreement
    if nli_result_missing:
        return "unresolved"
    if truncation_occurred:
        return "truncated_input"
    if not category_gate_passed:
        return "category_mismatch"
    if orientation_agrees_alt:
        return "orientation_error"
    if nli_confidence < confidence_threshold:
        return "low_nli_confidence"
    if rule_label == "unclear":
        return "neutral_vs_unclear_mapping"
    if compound_claim:
        return "compound_claim"
    if negation_present:
        return "negation_scope"
    if rule_label in ("supports", "challenges") and nli_label == "neutral":
        return "rule_keyword_false_positive"
    if rule_label == "neutral_context" and nli_label in ("entailment", "contradiction"):
        return "rule_keyword_false_negative"
    return "genuine_method_disagreement"


def detect_negation(text: str) -> bool:
    lowered = str(text).lower()
    return any(neg in lowered for neg in NEGATION_WORDS)


def detect_compound(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in COMPOUND_MARKERS)
