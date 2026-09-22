"""Day 3B: automated, post-hoc validation of a Gemini response against the
evidence it was actually given. A failed validation surfaces a structured
error -- it is never silently patched or replaced with a fallback answer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_LABELS = ("aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence")
INSPECTION_OVERCLAIM_PHRASES = (
    "i watched the video", "i reviewed the entire video", "having watched the full video",
    "after watching the complete video", "reviewing the whole video",
)

# Matches [EVIDENCE_ID] as instructed, but also tolerates a model writing
# several IDs in one bracket ([id1, id2]) despite the prompt asking for one
# citation per bracket -- the bracket content is comma/semicolon-split so
# neither form silently escapes citation validation.
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_:.\-,;\s]+)\]")


def extract_cited_ids(text: str) -> List[str]:
    ids = []
    for bracket_content in CITATION_PATTERN.findall(text or ""):
        for token in re.split(r"[,;]\s*", bracket_content):
            token = token.strip()
            if token:
                ids.append(token)
    return ids


_RAW_ID_METADATA_FIELDS = ("evidence_id", "asset_id", "frame_id", "video_id", "document_id")


def _build_raw_id_resolver(evidence_items: List[dict]) -> Dict[str, str]:
    """Maps a raw underlying ID (evidence_id/asset_id/frame_id/video_id/
    document_id, e.g. 'ce_press_0016' or 'tala_005') to its full namespaced
    Chroma ID (e.g. 'text::customer_experience::ce_press_0016'). Smaller/
    lighter Gemini models sometimes cite the raw ID and drop the namespace
    prefix despite the prompt's exact-format instruction -- this recognises
    that as the SAME real, retrieved evidence rather than flagging genuine
    evidence as hallucinated purely over ID formatting. An ID that resolves
    to nothing here is a true, un-retrieved fabrication and still fails."""
    resolver = {}
    for item in evidence_items:
        for field in _RAW_ID_METADATA_FIELDS:
            raw = item["metadata"].get(field)
            if raw:
                resolver.setdefault(str(raw), item["id"])
    return resolver


def _normalize_cited_ids(cited_ids: set, context_ids: set, raw_id_resolver: Dict[str, str]) -> set:
    """Returns cited_ids with any raw-ID citation rewritten to its full
    context ID where that resolution is unambiguous and real."""
    normalized = set()
    for cid in cited_ids:
        if cid in context_ids:
            normalized.add(cid)
        elif cid in raw_id_resolver:
            normalized.add(raw_id_resolver[cid])
        else:
            normalized.add(cid)  # left as-is; will correctly fail the hallucination check
    return normalized


def validate_response(response: dict, evidence_items: List[dict], true_fusion_label: str | None = None) -> Dict:
    """Returns {"passed": bool, "failures": [str, ...], "checks": {name: bool}}."""
    context_ids = {item["id"] for item in evidence_items}
    metadata_by_id = {item["id"]: item["metadata"] for item in evidence_items}
    raw_id_resolver = _build_raw_id_resolver(evidence_items)

    failures: List[str] = []
    checks: Dict[str, bool] = {}

    # Discrete text units, each validated independently -- never concatenated
    # with plain string joins, which can bleed a citation in one field into a
    # sentence-boundary check that actually belongs to an unrelated field.
    text_fields = [
        str(response.get("concise_answer", "")), str(response.get("claim_assessment", "")),
        str(response.get("confidence_explanation", "")),
    ] + list(response.get("caveats", []) or []) + list(response.get("evidence_gaps", []) or [])
    full_text = " ".join(text_fields)
    inline_cited_ids = set(extract_cited_ids(full_text))
    listed_ids = set(response.get("supporting_evidence_ids", []) or []) | \
        set(response.get("challenging_evidence_ids", []) or []) | \
        set(response.get("contextual_evidence_ids", []) or [])
    all_cited_ids = _normalize_cited_ids(inline_cited_ids | listed_ids, context_ids, raw_id_resolver)

    # 1 & 2 & 3: every cited ID exists in the retrieved context; no ID is invented.
    # (After raw-ID normalisation above -- a citation that drops the namespace
    # prefix but names a genuinely retrieved item is not a hallucination.)
    hallucinated = all_cited_ids - context_ids
    checks["citations_exist_in_context"] = not hallucinated
    if hallucinated:
        failures.append(f"cited evidence ID(s) not present in retrieved context: {sorted(hallucinated)}")

    # 4: source URLs / local asset paths referenced by cited evidence must resolve.
    broken_assets = []
    for eid in all_cited_ids & context_ids:
        meta = metadata_by_id[eid]
        local_path = meta.get("local_asset_path")
        if local_path and not (PROJECT_ROOT / local_path).exists():
            broken_assets.append(eid)
        source_url = meta.get("source_url")
        if source_url and not (source_url.startswith("http://") or source_url.startswith("https://")):
            broken_assets.append(eid)
    checks["referenced_assets_resolve"] = not broken_assets
    if broken_assets:
        failures.append(f"cited evidence ID(s) with unresolved local asset path or malformed source_url: {broken_assets}")

    # 5: production fusion classification preserved.
    label_preserved = True
    if true_fusion_label:
        other_labels_present = [lbl for lbl in CANONICAL_LABELS if lbl != true_fusion_label and re.search(rf"\b{re.escape(lbl)}\b", full_text, re.IGNORECASE)]
        true_label_present = re.search(rf"\b{re.escape(true_fusion_label)}\b", full_text, re.IGNORECASE) is not None
        if other_labels_present and not true_label_present:
            label_preserved = False
            failures.append(f"response appears to assert a different fusion label than the production classification '{true_fusion_label}': found {other_labels_present}")
    checks["production_label_preserved"] = label_preserved

    # 6: self-reported evidence never called independent.
    self_reported_called_independent = _check_self_reported_not_called_independent(text_fields, metadata_by_id, all_cited_ids & context_ids, raw_id_resolver)
    checks["self_reported_not_called_independent"] = not self_reported_called_independent
    if self_reported_called_independent:
        failures.append(f"self-reported evidence ID(s) described as independent: {self_reported_called_independent}")

    # 7: no claim to have inspected evidence beyond what was actually supplied
    # (e.g. claiming to have watched a full video when only sampled frames were given).
    overclaim = any(phrase in full_text.lower() for phrase in INSPECTION_OVERCLAIM_PHRASES)
    checks["no_overclaimed_inspection"] = not overclaim
    if overclaim:
        failures.append("response claims to have inspected evidence beyond what was supplied (e.g. 'watched the full video')")

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks, "cited_ids": sorted(all_cited_ids)}


def _check_self_reported_not_called_independent(
    text_fields: List[str], metadata_by_id: Dict[str, dict], cited_ids, raw_id_resolver: Dict[str, str],
) -> List[str]:
    """Splits sentences WITHIN each discrete response field only -- never
    across field boundaries, so a citation at the end of one field can never
    be matched against an unrelated "independent" mention that starts a
    different field."""
    sentences: List[str] = []
    for field_text in text_fields:
        sentences.extend(re.split(r"(?<=[.!?])\s+", field_text))

    violations = []
    for eid in cited_ids:
        meta = metadata_by_id.get(eid, {})
        is_self_reported_only = str(meta.get("self_reported", False)) in ("True", "true") and str(meta.get("independent_source", False)) not in ("True", "true")
        if not is_self_reported_only:
            continue
        for sentence in sentences:
            sentence_ids = {raw_id_resolver.get(cid, cid) for cid in extract_cited_ids(sentence)}
            if eid not in sentence_ids:
                continue
            # A sentence that says independent verification is ABSENT/MISSING for
            # this evidence is correct, not a violation -- only flag sentences that
            # positively assert the evidence itself IS independent/third-party.
            has_independence_claim = re.search(r"\bindependent(ly)?\b|\bthird[- ]party\b", sentence, re.IGNORECASE)
            has_negation_nearby = re.search(r"\b(no|not|without|absence|lack(s|ing)?|none|constrained by)\b", sentence, re.IGNORECASE)
            if has_independence_claim and not has_negation_nearby:
                violations.append(eid)
                break
    return violations
