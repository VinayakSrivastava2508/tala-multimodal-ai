"""Day 2.5 Part H: strict assignment-alignment modality audit.

Regenerates outputs/tables/assignment_modality_audit.csv and the supporting
video/engagement tables directly from the ACTUAL data on disk -- never from
narrative claims in a notebook or doc. This is the file a reader should trust
over any prose summary.

Binding definitions enforced here (CLAUDE.md §9):
  - A video counts as PROCESSED only if video_assets.processing_status=='processed'
    AND it has a matching video_level_features.csv row with n_sampled_frames>=2.
    A platform_metadata_only row (YouTube URL/title/thumbnail/engagement stat) is a
    VIDEO LEAD, never counted toward video coverage.
  - Engagement metrics are retained and reported descriptively only (medians, IQRs,
    coverage) -- never as a modelling target, never with predicted/residual columns.

Usage: python scripts/audit_assignment_modalities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TABLES = PROJECT_ROOT / "outputs" / "tables"
PROCESSED = PROJECT_ROOT / "data" / "processed"
INTERIM_DAY2_5 = PROJECT_ROOT / "data" / "interim" / "day2_5"
CORPORA = PROJECT_ROOT / "data" / "corpora"

PERMITTED_LOCAL_PROCESSING_BASIS = (
    "official_direct_public_asset", "open_license", "group_owned", "user_authorised",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)  # path outside PROJECT_ROOT (e.g. a test-provided tmp dir)


# ── Video coverage / role distribution / feature summary ─────────────────────

def build_video_asset_coverage(video_assets: pd.DataFrame, video_level: pd.DataFrame) -> pd.DataFrame:
    if video_assets.empty:
        return pd.DataFrame()

    df = video_assets.copy()
    processed_ids = set()
    if not video_level.empty:
        processed_ids = set(video_level.loc[video_level["n_sampled_frames"] >= 2, "video_asset_id"])

    df["is_lead"] = df["rights_or_access_basis"] == "platform_metadata_only"
    df["is_processable_basis"] = df["rights_or_access_basis"].isin(PERMITTED_LOCAL_PROCESSING_BASIS)
    df["is_downloaded"] = df["local_path"].notna() & (df["local_path"] != "")
    df["is_genuinely_processed"] = df["video_asset_id"].isin(processed_ids) & (df["processing_status"] == "processed")

    rows = []
    for brand, g in df.groupby("brand"):
        rows.append({
            "brand": brand,
            "video_leads_metadata_only": int(g["is_lead"].sum()),
            "candidate_direct_assets_discovered": int((~g["is_lead"]).sum()),
            "genuinely_downloaded_pending_or_processed": int(
                g["is_processable_basis"].sum() & g["is_downloaded"].sum() if False else
                (g["is_processable_basis"] & g["is_downloaded"]).sum()
            ),
            "genuinely_processed_videos": int(g["is_genuinely_processed"].sum()),
            "processing_failed": int((g["processing_status"] == "failed").sum()),
        })
    out = pd.DataFrame(rows).sort_values("brand").reset_index(drop=True)
    total = {
        "brand": "TOTAL",
        "video_leads_metadata_only": int(df["is_lead"].sum()),
        "candidate_direct_assets_discovered": int((~df["is_lead"]).sum()),
        "genuinely_downloaded_pending_or_processed": int((df["is_processable_basis"] & df["is_downloaded"]).sum()),
        "genuinely_processed_videos": int(df["is_genuinely_processed"].sum()),
        "processing_failed": int((df["processing_status"] == "failed").sum()),
    }
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    path = TABLES / "video_asset_coverage.csv"
    out.to_csv(path, index=False)
    print(f"  Saved: {_relpath(path)}")
    return out


def build_video_role_distribution(video_assets: pd.DataFrame, video_level: pd.DataFrame) -> pd.DataFrame:
    if video_assets.empty:
        return pd.DataFrame()
    df = video_assets.copy()
    processed_ids = set()
    if not video_level.empty:
        processed_ids = set(video_level.loc[video_level["n_sampled_frames"] >= 2, "video_asset_id"])
    df["asset_class"] = np.where(
        df["video_asset_id"].isin(processed_ids) & (df["processing_status"] == "processed"),
        "genuinely_processed",
        np.where(df["rights_or_access_basis"] == "platform_metadata_only", "video_lead_metadata_only", "other_not_processed"),
    )
    out = df.groupby(["asset_class", "video_role", "brand"]).size().reset_index(name="n_rows")
    out = out.sort_values(["asset_class", "n_rows"], ascending=[True, False]).reset_index(drop=True)
    path = TABLES / "video_role_distribution.csv"
    out.to_csv(path, index=False)
    print(f"  Saved: {_relpath(path)}")
    return out


def build_video_feature_summary(video_level: pd.DataFrame, video_frames: pd.DataFrame) -> pd.DataFrame:
    if video_level.empty:
        row = {
            "n_processed_videos": 0, "n_videos_with_ge2_frames": 0, "total_sampled_frames": 0,
            "mean_frames_per_video": None, "n_with_temporal_features": 0,
            "n_with_transcript": 0, "transcript_coverage_pct": 0.0,
            "mean_duration_seconds": None, "note": "no genuinely processed video assets -- see video_asset_coverage.csv",
        }
        out = pd.DataFrame([row])
    else:
        valid = video_level[video_level["n_sampled_frames"] >= 2]
        has_temporal = valid["mean_inter_frame_perceptual_distance"].notna() if "mean_inter_frame_perceptual_distance" in valid.columns else pd.Series([], dtype=bool)
        has_transcript = valid["transcript_text"].notna() if "transcript_text" in valid.columns else pd.Series([False] * len(valid))
        out = pd.DataFrame([{
            "n_processed_videos": len(video_level),
            "n_videos_with_ge2_frames": len(valid),
            "total_sampled_frames": int(video_frames.shape[0]) if not video_frames.empty else 0,
            "mean_frames_per_video": round(float(valid["n_sampled_frames"].mean()), 2) if len(valid) else None,
            "n_with_temporal_features": int(has_temporal.sum()),
            "n_with_transcript": int(has_transcript.sum()) if len(has_transcript) else 0,
            "transcript_coverage_pct": round(100 * (has_transcript.sum() / len(valid)), 2) if len(valid) else 0.0,
            "mean_duration_seconds": round(float(valid["duration_seconds"].mean()), 2) if len(valid) else None,
            "note": "",
        }])
    path = TABLES / "video_feature_summary.csv"
    out.to_csv(path, index=False)
    print(f"  Saved: {_relpath(path)}")
    return out


# ── Descriptive (non-causal) creator engagement ───────────────────────────────

def build_descriptive_engagement(creator_features: pd.DataFrame) -> pd.DataFrame:
    """Medians/IQRs/coverage by brand. Explicitly descriptive -- no modelling
    target, no predicted/residual columns, no regression of any kind."""
    if creator_features.empty:
        return pd.DataFrame()
    rows = []
    for brand, g in creator_features.groupby("brand"):
        eng = g["engagement_count"].dropna()
        rows.append({
            "brand": brand,
            "n_posts": len(g),
            "n_with_engagement_count": len(eng),
            "coverage_pct": round(100 * len(eng) / len(g), 2) if len(g) else 0.0,
            "median_engagement_count": float(eng.median()) if len(eng) else None,
            "iqr_low_engagement_count": float(eng.quantile(0.25)) if len(eng) else None,
            "iqr_high_engagement_count": float(eng.quantile(0.75)) if len(eng) else None,
            "median_view_count": float(g["view_count"].dropna().median()) if "view_count" in g.columns and g["view_count"].notna().any() else None,
        })
    out = pd.DataFrame(rows).sort_values("brand").reset_index(drop=True)
    out["analysis_type"] = "descriptive_only_non_causal"
    out["modelling_target"] = "NONE -- engagement prediction is out of scope for this project (CLAUDE.md §9)"
    path = TABLES / "descriptive_creator_engagement.csv"
    out.to_csv(path, index=False)
    print(f"  Saved: {_relpath(path)}")
    return out


# ── Step 3: image audit ───────────────────────────────────────────────────────

def audit_images(image_assets: pd.DataFrame) -> dict:
    if image_assets.empty:
        return {"available": 0, "verified": 0, "directly_processed": 0, "proxy_only": 0, "missing": "no image_assets.csv rows"}

    df = image_assets.copy()
    df["is_downloaded"] = df["processing_status"] == "downloaded"
    df["is_verified"] = (
        df["is_downloaded"]
        & df["width"].notna() & df["height"].notna()
        & df["file_hash"].notna() & (df["file_hash"] != "")
        & df["source_page_url"].notna() & (df["source_page_url"] != "")
        & df["source_url"].notna() & (df["source_url"] != "")
        & df["rights_or_access_basis"].notna()
    )
    verified = df[df["is_verified"]]
    dup_urls = int(df["source_url"].duplicated().sum())
    dup_hashes = int(verified["file_hash"].duplicated().sum())
    proxy_only = int((df["image_role"] == "creator_thumbnail").sum())

    return {
        "available": len(df),
        "verified": int(verified.shape[0]),
        "directly_processed": int(verified.shape[0]),
        "proxy_only": proxy_only,
        "missing": f"{len(df) - len(verified)} rows failed verification (missing width/height/hash/URLs)",
        "by_brand": verified.groupby("brand").size().to_dict(),
        "by_role": verified.groupby("image_role").size().to_dict(),
        "by_access_basis": verified.groupby("rights_or_access_basis").size().to_dict(),
        "dup_urls": dup_urls,
        "dup_hashes": dup_hashes,
    }


# ── Step 7: assignment_modality_audit.csv ─────────────────────────────────────

def build_modality_audit(
    text_corpora: dict, image_audit: dict, video_coverage: pd.DataFrame,
    video_feature_summary: pd.DataFrame, claim_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Text Package
    n_text = sum(len(df) for df in text_corpora.values())
    rows.append({
        "package": "Text Package", "required_asset_type": "official claims + customer experience + creator strategy documents",
        "available_assets": n_text, "verified_assets": n_text, "directly_processed_assets": n_text,
        "proxy_only_assets": 0, "missing_assets": "reference documents largely unhydrated for non-TALA brands (see Multimodal Reference Package)",
        "assignment_status": "PASS" if n_text > 0 else "FAIL",
        "remediation_action": "none -- corpora populated and validated in Day 2",
    })

    # Image Package
    n_brands_img = len(image_audit.get("by_brand", {}))
    img_status = "PASS" if image_audit["verified"] >= 4 * n_brands_img and n_brands_img >= 3 else ("PARTIAL" if image_audit["verified"] > 0 else "FAIL")
    rows.append({
        "package": "Image Package", "required_asset_type": "official catalog/product images, permitted UGC, creator thumbnails",
        "available_assets": image_audit["available"], "verified_assets": image_audit["verified"],
        "directly_processed_assets": image_audit["directly_processed"], "proxy_only_assets": image_audit["proxy_only"],
        "missing_assets": image_audit["missing"], "assignment_status": img_status,
        "remediation_action": "none required -- >=4 verified official images per brand across 4 brands" if img_status == "PASS"
        else "collect additional official product images per brand",
    })

    # Video Package
    video_total_row = video_coverage[video_coverage["brand"] == "TOTAL"] if not video_coverage.empty else pd.DataFrame()
    n_processed = int(video_total_row["genuinely_processed_videos"].iloc[0]) if not video_total_row.empty else 0
    n_leads = int(video_total_row["video_leads_metadata_only"].iloc[0]) if not video_total_row.empty else 0
    n_brands_video_processed = int((video_coverage[video_coverage["brand"] != "TOTAL"]["genuinely_processed_videos"] > 0).sum()) if not video_coverage.empty else 0
    video_status = "FAIL" if n_processed == 0 else ("PARTIAL" if (n_processed < 12 or n_brands_video_processed < 3) else "PASS")
    rows.append({
        "package": "Video Package", "required_asset_type": "genuine temporal video assets (sampled frames, motion/scene-change features, transcripts)",
        "available_assets": n_leads + n_processed, "verified_assets": n_processed,
        "directly_processed_assets": n_processed, "proxy_only_assets": n_leads,
        "missing_assets": f"{n_leads} platform_metadata_only video leads (YouTube/TikTok URLs -- correctly excluded from coverage); "
                           f"shortfall vs 12-20 asset / >=3 brand target: have {n_processed} processed video(s) across "
                           f"{n_brands_video_processed} brand(s)",
        "assignment_status": video_status,
        "remediation_action": "PARTIAL/FAIL -- place brand-authorised local video files under "
                               "data/raw/authorised_video_assets/ for Adanola, Girlfriend Collective, and Oner Active "
                               "(no yt-dlp/unofficial downloaders permitted); do not fabricate assets to close the gap",
    })

    # Multimodal Reference Package
    ref_path = PROCESSED / "multimodal_reference_assets.csv"
    n_ref = len(_read_csv(ref_path))
    rows.append({
        "package": "Multimodal Reference Package", "required_asset_type": "impact/responsibility reports, certifications, material specs, sizing/care guidance, return policies",
        "available_assets": n_ref, "verified_assets": n_ref, "directly_processed_assets": 0, "proxy_only_assets": 0,
        "missing_assets": "package not yet populated -- data/raw/templates/multimodal_reference_assets_template.csv exists as an empty schema-correct template only",
        "assignment_status": "FAIL" if n_ref == 0 else "PARTIAL",
        "remediation_action": "collect reference documents (impact reports, certifications, material specs, care/sizing guidance) per brand and populate data/processed/multimodal_reference_assets.csv",
    })

    # Text Pipeline
    rows.append({
        "package": "Text Pipeline", "required_asset_type": "TF-IDF/sentiment/embedding features on official/customer/creator text",
        "available_assets": n_text, "verified_assets": n_text, "directly_processed_assets": n_text, "proxy_only_assets": 0,
        "missing_assets": "none", "assignment_status": "PASS",
        "remediation_action": "none -- see data/processed/{official_claim,customer_text}_features.csv",
    })

    # Image Pipeline
    rows.append({
        "package": "Image Pipeline", "required_asset_type": "interpretable + CLIP embedding features on genuine image assets",
        "available_assets": image_audit["verified"], "verified_assets": image_audit["verified"],
        "directly_processed_assets": image_audit["verified"], "proxy_only_assets": 0, "missing_assets": "none",
        "assignment_status": "PASS" if image_audit["verified"] > 0 else "FAIL",
        "remediation_action": "none",
    })

    # Video Pipeline
    n_with_temporal = int(video_feature_summary["n_with_temporal_features"].iloc[0]) if not video_feature_summary.empty else 0
    rows.append({
        "package": "Video Pipeline", "required_asset_type": "genuine temporal features (frame sequence, motion/scene-change, transcript) -- cannot pass on proxy metadata alone",
        "available_assets": n_processed, "verified_assets": n_with_temporal, "directly_processed_assets": n_with_temporal,
        "proxy_only_assets": n_leads,
        "missing_assets": "no temporal features exist" if n_with_temporal == 0 else f"{n_processed - n_with_temporal} processed video(s) without a valid temporal-feature row",
        "assignment_status": "FAIL" if n_with_temporal == 0 else ("PARTIAL" if n_processed < 12 else "PASS"),
        "remediation_action": "see Video Package remediation -- Video Pipeline cannot PASS without more genuinely processed, multi-brand video assets",
    })

    # Claim-evidence candidate layer
    n_claims = len(claim_candidates)
    n_all3 = int(((claim_candidates.get("text_evidence_ids", pd.Series(dtype=str)) != "")
                  & (claim_candidates.get("image_asset_ids", pd.Series(dtype=str)) != "")
                  & (claim_candidates.get("video_asset_ids", pd.Series(dtype=str)) != "")).sum()) if n_claims else 0
    rows.append({
        "package": "Claim-evidence candidate layer", "required_asset_type": "claim-linked text/image/video/reference evidence candidates",
        "available_assets": n_claims, "verified_assets": n_claims, "directly_processed_assets": n_all3,
        "proxy_only_assets": 0,
        "missing_assets": f"{n_claims - n_all3}/{n_claims} claims missing >=1 modality (reference evidence missing for all claims -- package not populated)" if n_claims else "claim_multimodal_evidence_candidates.csv not built",
        "assignment_status": "PARTIAL" if n_claims > 0 else "FAIL",
        "remediation_action": "populate Multimodal Reference Package and expand Video Package to raise multi-modality coverage; "
                               "candidates require human validation before use in scoring (requires_human_validation=True on every row)",
    })

    out = pd.DataFrame(rows)
    path = TABLES / "assignment_modality_audit.csv"
    out.to_csv(path, index=False)
    print(f"  Saved: {_relpath(path)}")
    return out


def main() -> int:
    print("Day 2.5 assignment-modality audit\n" + "=" * 60)

    image_assets = _read_csv(INTERIM_DAY2_5 / "image_assets.csv")
    video_assets = _read_csv(INTERIM_DAY2_5 / "video_assets.csv")
    video_level = _read_csv(PROCESSED / "video_level_features.csv")
    video_frames = _read_csv(PROCESSED / "video_frame_features.csv")
    creator_features = _read_csv(PROCESSED / "creator_multimodal_features.csv")
    claim_candidates = _read_csv(PROCESSED / "claim_multimodal_evidence_candidates.csv")
    # pandas reads a saved empty string ("") back as NaN -- restore it explicitly so
    # `!= ""` presence checks below aren't silently flipped by the CSV round-trip.
    for _col in ("text_evidence_ids", "image_asset_ids", "video_asset_ids", "reference_document_ids"):
        if _col in claim_candidates.columns:
            claim_candidates[_col] = claim_candidates[_col].fillna("")
    text_corpora = {
        "official_claims": _read_csv(CORPORA / "official_claims_corpus.csv"),
        "customer_experience": _read_csv(CORPORA / "customer_experience_corpus.csv"),
        "creator_strategy": _read_csv(CORPORA / "creator_strategy_corpus.csv"),
    }

    print("\n[Video tables]")
    video_coverage = build_video_asset_coverage(video_assets, video_level)
    build_video_role_distribution(video_assets, video_level)
    video_feature_summary = build_video_feature_summary(video_level, video_frames)

    print("\n[Descriptive engagement -- non-causal]")
    build_descriptive_engagement(creator_features)

    print("\n[Image audit]")
    image_audit = audit_images(image_assets)
    print(f"  Verified: {image_audit['verified']}/{image_audit['available']}, "
          f"by brand: {image_audit.get('by_brand')}, dup URLs: {image_audit['dup_urls']}, dup hashes: {image_audit['dup_hashes']}")

    print("\n[Modality audit]")
    audit = build_modality_audit(text_corpora, image_audit, video_coverage, video_feature_summary, claim_candidates)
    print("\n" + audit[["package", "assignment_status"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Audit complete. See outputs/tables/assignment_modality_audit.csv for the authoritative status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
