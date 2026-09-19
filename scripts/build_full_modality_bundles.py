"""Day 2.6B Part H: full-modality prototype bundle construction.

Identifies claims/issues that already have official/claim text, customer text,
image, video, and reference evidence simultaneously, and packages them as
prototype bundles for later fusion work. This does NOT assign an
aligned/divergent verdict -- that is a later scoring task.

Usage: python scripts/build_full_modality_bundles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "claim_multimodal_evidence_candidates.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "full_modality_prototype_bundles.csv"

REQUIRED_FIELD_ORDER = [
    "text_evidence_ids", "image_asset_ids", "video_asset_ids", "reference_document_ids",
]


def _split(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [v for v in str(value).split(";") if v]


def main() -> int:
    if not CANDIDATES_PATH.exists():
        print(f"{CANDIDATES_PATH} not found -- run scripts/build_claim_multimodal_candidates.py first.")
        return 1

    df = pd.read_csv(CANDIDATES_PATH)
    for col in REQUIRED_FIELD_ORDER:
        df[col] = df[col].fillna("")

    bundles = []
    for _, r in df.iterrows():
        modalities_present = [c for c in REQUIRED_FIELD_ORDER if r[c] != ""]
        modality_count = len(modalities_present)
        # A full-modality prototype bundle requires text AND (image OR video) AND reference --
        # the strongest evidence combination this project currently has, never forced.
        has_visual = r["image_asset_ids"] != "" or r["video_asset_ids"] != ""
        if not (r["text_evidence_ids"] != "" and has_visual and r["reference_document_ids"] != ""):
            continue

        missing = [m.replace("_evidence_ids", "").replace("_document_ids", "").replace("_asset_ids", "")
                   for m in REQUIRED_FIELD_ORDER if r[m] == ""]

        proposed_use = (
            "product fit/appearance divergence check (does customer experience match the visual/reference "
            "presentation of this materials claim?)" if r["claim_category"] == "materials"
            else "descriptive claim-evidence cross-reference (not yet scored)"
        )
        limitations = []
        if r["video_asset_ids"] != "":
            limitations.append("video evidence is presentation-only -- cannot by itself establish durability, "
                                "wash performance, emissions, labour, ethics, or certification status")
        if r["reference_document_ids"] != "":
            limitations.append("reference match is a category-gated semantic candidate, not yet human-validated")

        bundles.append({
            "bundle_id": f"bundle_full_{r['claim_id']}",
            "claim_or_issue_id": r["claim_id"],
            "issue_statement": r["claim_text"],
            "official_text_ids": r["claim_id"],  # the claim itself is the official-text anchor
            "customer_text_ids": r["text_evidence_ids"],
            "image_asset_ids": r["image_asset_ids"],
            "video_asset_ids": r["video_asset_ids"],
            "reference_chunk_ids": r["reference_document_ids"],
            "modality_count": modality_count,
            "evidence_eligibility": "candidate_prototype_not_scored",
            "missing_modalities": ";".join(missing) if missing else "none",
            "proposed_analytical_use": proposed_use,
            "limitations": "; ".join(limitations) if limitations else "none noted",
            "requires_human_validation": True,
            "provenance_note": f"assembled from claim_multimodal_evidence_candidates.csv row for {r['claim_id']}",
        })

    out = pd.DataFrame(bundles)
    out.to_csv(OUT_PATH, index=False)
    try:
        out_display = OUT_PATH.relative_to(PROJECT_ROOT)
    except ValueError:
        out_display = OUT_PATH
    print(f"Saved: {out_display} ({len(out)} full-modality prototype bundle(s))")
    if not out.empty:
        print(out[["bundle_id", "claim_or_issue_id", "modality_count"]].to_string(index=False))
    else:
        print("No claim currently has text + (image or video) + reference evidence simultaneously.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
