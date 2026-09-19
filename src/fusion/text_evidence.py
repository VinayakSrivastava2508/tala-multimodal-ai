"""Day 3A Part 8: text evidence pipeline (customer, critic, press, reference text).

Deterministic rule-based stance classification is the primary, always-available
method (Part 8 requires a rule-based fallback regardless of any model). An
optional zero-shot NLI cross-check is attempted separately (see
`nli_stance_if_available`) purely for the automated cross-method comparison in
Part 15.D -- it is never used as the fusion pipeline's actual stance, and the
final fusion label is never fed back into stance classification (no circularity).
"""

from __future__ import annotations

from typing import Dict, Optional

RULE_BASED_METHOD = "rule_based_sentiment_v1"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"  # small, local-cacheable, deterministic at inference

_NLI_CACHE: dict = {}


def source_independence_for_corpus(corpus_source: str, brand_owned: bool = False) -> str:
    """Classify evidence-source independence. The claim's own corpus is always
    self_reported (Part 6 rule 1: a claim cannot validate itself)."""
    if corpus_source == "official_claims":
        return "self_reported"
    if brand_owned:
        return "self_reported"
    if corpus_source == "customer_experience":
        return "customer_reported"
    if corpus_source == "creator_strategy":
        return "unclear"  # creator content may be paid/sponsored -- provenance is ambiguous
    if corpus_source in ("certification_registry",):
        return "independent"
    return "unclear"


def extract_passage(text: str, max_chars: int = 400) -> str:
    """Return the evidence passage actually used -- truncated, never fabricated
    beyond what the source text contains."""
    text = str(text).strip()
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "..."


def rule_based_stance(evidence_text: str, relevance_score: float, relevance_threshold: float, config: dict) -> Dict:
    """Deterministic sentiment-based stance classification. Returns
    {"stance", "stance_score", "method"}. Below the relevance threshold, stance
    is never computed (not_applicable) -- relevance is a precondition for
    stance, never treated as stance itself (Part 6 rule: relevance != support)."""
    if relevance_score < relevance_threshold:
        return {"stance": "not_applicable", "stance_score": 0.0, "method": RULE_BASED_METHOD}

    from textblob import TextBlob  # lazy import, already a project dependency
    blob = TextBlob(str(evidence_text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    thresholds = config["stance_thresholds"]
    if subjectivity >= thresholds["unclear_subjectivity_min"] and abs(polarity) < thresholds["supports_polarity_min"]:
        return {"stance": "unclear", "stance_score": round(polarity, 4), "method": RULE_BASED_METHOD}
    if polarity >= thresholds["supports_polarity_min"]:
        return {"stance": "supports", "stance_score": round(polarity, 4), "method": RULE_BASED_METHOD}
    if polarity <= thresholds["challenges_polarity_max"]:
        return {"stance": "challenges", "stance_score": round(polarity, 4), "method": RULE_BASED_METHOD}
    return {"stance": "neutral_context", "stance_score": round(polarity, 4), "method": RULE_BASED_METHOD}


def nli_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# Default orientation for the production auxiliary cross-check. Day 3A Part 15D
# diagnostic (scripts/run_nli_diagnostics.py, docs/automated_fusion_evaluation.md)
# selected this orientation from a synthetic calibration set, not from the real
# claim-evidence agreement rate.
NLI_PREMISE_FIELD = "evidence_text"
NLI_HYPOTHESIS_FIELD = "claim_text"
NLI_DECISION_THRESHOLD = 0.5  # entailment/contradiction probability threshold


def _load_nli_model():
    if "model" not in _NLI_CACHE:
        from sentence_transformers import CrossEncoder
        _NLI_CACHE["model"] = CrossEncoder(NLI_MODEL_NAME)
    return _NLI_CACHE["model"]


def nli_label_indices() -> Dict[str, int]:
    """Read entailment/contradiction/neutral label IDs from the model's OWN
    config.id2label -- never assume a fixed LABEL_0/LABEL_1/LABEL_2 ordering.
    Day 3A NLI diagnostic (Part A) verified this checkpoint's ordering is
    {0: contradiction, 1: entailment, 2: neutral}, but that mapping is read
    fresh from config here rather than hardcoded, so a checkpoint/revision
    change can never silently invert supports/challenges."""
    model = _load_nli_model()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    missing = [lbl for lbl in ("entailment", "contradiction", "neutral") if lbl not in label2id]
    if missing:
        raise ValueError(f"NLI model {NLI_MODEL_NAME} config.id2label is missing {missing}: {id2label}")
    return label2id


def nli_predict_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    """Runs the NLI cross-encoder on one (premise, hypothesis) pair. Returns
    {"entailment":.., "contradiction":.., "neutral":.., "confidence":..} where
    class probabilities are read via the model's own config.id2label mapping,
    never via an assumed logit position."""
    model = _load_nli_model()
    label2id = nli_label_indices()
    import numpy as np
    scores = model.predict([(str(premise), str(hypothesis))])[0]
    probs = np.exp(scores) / np.exp(scores).sum()
    result = {label: round(float(probs[idx]), 4) for label, idx in label2id.items()}
    result["confidence"] = round(max(result["entailment"], result["contradiction"], result["neutral"]), 4)
    return result


def nli_label_from_probs(probs: Dict[str, float], threshold: float = NLI_DECISION_THRESHOLD) -> str:
    """entailment >= threshold -> 'entailment'; else contradiction >= threshold
    -> 'contradiction'; else 'neutral'. Matches the production decision rule."""
    if probs["entailment"] >= threshold:
        return "entailment"
    if probs["contradiction"] >= threshold:
        return "contradiction"
    return "neutral"


_NLI_LABEL_TO_STANCE = {"entailment": "supports", "contradiction": "challenges", "neutral": "neutral_context"}


def nli_stance_if_available(claim_text: str, evidence_text: str) -> Optional[Dict]:
    """Optional zero-shot-style NLI cross-check via a small, local, deterministic
    cross-encoder (premise=evidence, hypothesis=claim -- see NLI_PREMISE_FIELD/
    NLI_HYPOTHESIS_FIELD). Returns None if the model cannot be loaded --
    callers must report 'not run', never fabricate agreement. Model outputs
    are cached per (claim_text, evidence_text) pair for determinism and to
    avoid redundant inference.

    The premise is bounded to the same evidence_passage length (400 chars)
    used everywhere else in this pipeline for citation -- some evidence_text
    values run to ~6000 chars (full scraped documents), which silently
    exceeds the NLI model's 512-token window. Without this bound, the
    tokenizer truncates the premise mid-document while the rule-based
    stance classifier (TextBlob) scores the FULL untruncated text, making
    the two methods compare against different effective inputs (Day 3A NLI
    diagnostic, Part C: `truncated_input` was the single largest
    disagreement category on real evidence)."""
    key = (claim_text, evidence_text)
    if key in _NLI_CACHE:
        return _NLI_CACHE[key]
    if not nli_available():
        return None
    try:
        bounded_evidence = extract_passage(evidence_text)
        probs = nli_predict_probs(premise=bounded_evidence, hypothesis=claim_text)
        nli_label = nli_label_from_probs(probs)
        result = {
            "stance": _NLI_LABEL_TO_STANCE[nli_label],
            "stance_score": round(probs["entailment"] - probs["contradiction"], 4),
            "method": f"nli_{NLI_MODEL_NAME}", "entailment": probs["entailment"],
            "contradiction": probs["contradiction"], "neutral": probs["neutral"],
            "confidence": probs["confidence"],
        }
    except Exception:
        result = None
    _NLI_CACHE[key] = result
    return result


def reliability_score(evidence_strength: str, source_independence: str, config: dict) -> float:
    """Deterministic reliability = evidence_strength_weight * source_reliability_tier."""
    strength_w = config["evidence_strength_weight"].get(evidence_strength, 0.0)
    source_w = config["source_reliability_tiers"].get(source_independence, 0.4)
    return round(strength_w * source_w, 4)
