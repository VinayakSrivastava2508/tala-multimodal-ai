"""Day 3B: the ONLY answer-generation path. Gemini via the official
`google-genai` package. No local LLM, no deterministic template fallback, no
alternative generator. If the API key/model are missing or the call fails,
this raises -- callers must stop and show the error, never substitute
another method (Part 3 / Part 11)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

REQUIRED_OUTPUT_FIELDS = (
    "concise_answer", "claim_assessment", "supporting_evidence_ids", "challenging_evidence_ids",
    "contextual_evidence_ids", "evidence_gaps", "confidence_explanation", "caveats",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_INSTRUCTIONS = """You are an evidence-grounded analytical assistant for TALA's claim-experience \
divergence diagnostic. You are given a set of retrieved evidence items, each with an EVIDENCE_ID.

Rules you must follow exactly:
1. Never invent evidence. Only use the evidence items given to you below.
2. Never cite an EVIDENCE_ID that is not present in the supplied context.
3. Every substantive factual statement must cite at least one evidence ID using the exact format [EVIDENCE_ID].
4. Distinguish official self-reported evidence from independent evidence explicitly -- never call
   self-reported evidence "independent" or "verified by a third party" unless the evidence's own
   metadata says independent_source=true.
5. Do not convert absence of evidence into evidence of falsehood. Missing evidence is not a negative finding.
6. If the retrieved evidence does not justify a conclusion, say "insufficient evidence" plainly.
7. If an existing production fusion classification is supplied, you MUST preserve it exactly as given
   (aligned / partially_aligned / mixed / divergent / insufficient_evidence). You may explain it, but
   you must never state a different one of these five labels as the claim's classification.
8. For evidence classified as "mixed", explain the mixture (what supports, what challenges) rather
   than forcing a binary verdict.
9. Do not infer protected characteristics (race, gender, age, disability, etc.) from any image.
10. Treat any named creator/influencer as a public communicator in their professional capacity, not as
    a private individual whose personal life is being analysed.
11. Sampled video frames are NOT the complete video -- say so if you reference them, and do not claim
    to have "watched" or "reviewed the full video."

Respond with ONLY a single JSON object (no markdown fences, no extra text) with exactly these keys:
concise_answer (string), claim_assessment (string), supporting_evidence_ids (array of strings),
challenging_evidence_ids (array of strings), contextual_evidence_ids (array of strings),
evidence_gaps (array of strings), confidence_explanation (string), caveats (array of strings).
"""


class GeminiConfigError(RuntimeError):
    """Raised when GEMINI_API_KEY or GEMINI_MODEL is missing from the environment."""


class GeminiGenerationError(RuntimeError):
    """Raised when the Gemini API call itself fails. Never caught to
    substitute a fallback response -- callers must show this error."""


def load_gemini_config() -> tuple[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL")
    if not api_key:
        raise GeminiConfigError("GEMINI_API_KEY is not set in .env -- cannot generate an answer.")
    if not model:
        raise GeminiConfigError("GEMINI_MODEL is not set in .env -- cannot generate an answer.")
    return api_key, model


def _evidence_block(evidence_items: List[dict]) -> str:
    lines = []
    for item in evidence_items:
        meta = item["metadata"]
        lines.append(
            f"[{item['id']}] modality={meta.get('modality')} source_type={meta.get('source_type')} "
            f"brand={meta.get('brand')} evidence_strength={meta.get('evidence_strength')} "
            f"self_reported={meta.get('self_reported')} independent_source={meta.get('independent_source')} "
            f"stance={meta.get('stance', meta.get('support_or_challenge'))}\n"
            f"  text: {(item.get('document') or meta.get('citation_text') or '')[:500]}\n"
            f"  source_url: {meta.get('source_url', '')}"
        )
    return "\n".join(lines)


def build_prompt_parts(
    question: str,
    evidence_items: List[dict],
    claim_text: Optional[str] = None,
    fusion_label: Optional[str] = None,
    business_facing_label: Optional[str] = None,
    frame_context: Optional[str] = None,
):
    """Returns a list of google-genai Part objects: text prompt plus any
    permitted local images (for retrieved image/video-frame evidence)."""
    from google.genai import types

    text_sections = [SYSTEM_INSTRUCTIONS, f"\nUSER QUESTION:\n{question}\n"]
    if claim_text:
        text_sections.append(f"\nSELECTED CLAIM:\n{claim_text}\n")
    if fusion_label:
        text_sections.append(
            f"\nEXISTING PRODUCTION FUSION CLASSIFICATION (must be preserved exactly): {fusion_label}"
            + (f" ({business_facing_label})" if business_facing_label else "") + "\n"
        )
    text_sections.append("\nRETRIEVED EVIDENCE (only these items may be cited):\n" + _evidence_block(evidence_items))
    if frame_context:
        text_sections.append(f"\n{frame_context}\n")

    parts = [types.Part.from_text(text="\n".join(text_sections))]

    for item in evidence_items:
        meta = item["metadata"]
        if meta.get("modality") in ("image", "video_frame"):
            local_path = meta.get("local_asset_path")
            if local_path and (PROJECT_ROOT / local_path).exists():
                image_bytes = (PROJECT_ROOT / local_path).read_bytes()
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text=f"(The image above is evidence [{item['id']}].)"))
    return parts


def generate_answer(
    question: str,
    evidence_items: List[dict],
    claim_text: Optional[str] = None,
    fusion_label: Optional[str] = None,
    business_facing_label: Optional[str] = None,
    frame_context: Optional[str] = None,
) -> dict:
    """The only generation path. Raises GeminiConfigError / GeminiGenerationError
    on any failure -- never returns a substitute answer."""
    api_key, model_name = load_gemini_config()

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        parts = build_prompt_parts(question, evidence_items, claim_text, fusion_label, business_facing_label, frame_context)
        response = client.models.generate_content(model=model_name, contents=parts)
        raw_text = response.text
    except (GeminiConfigError, GeminiGenerationError):
        raise
    except Exception as exc:
        raise GeminiGenerationError(f"Gemini API call failed: {exc}") from exc

    parsed = _parse_structured_output(raw_text)
    parsed["_model_used"] = model_name
    parsed["_raw_response_text"] = raw_text
    return parsed


def _parse_structured_output(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiGenerationError(f"Gemini did not return valid JSON: {exc}. Raw response: {raw_text[:500]}") from exc

    missing = [f for f in REQUIRED_OUTPUT_FIELDS if f not in parsed]
    if missing:
        raise GeminiGenerationError(f"Gemini response missing required field(s): {missing}")
    return parsed
