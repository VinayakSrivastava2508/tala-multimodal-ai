"""Day 3B: build the three RAG corpora (claims, text evidence, visual evidence)
from the existing, authoritative Day 1-3A processed datasets only. Applies the
existing evidence-strength/verification gates -- never invents a new gate and
never fabricates a row that doesn't exist in the source data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.rag.schemas import (
    ELIGIBLE_EVIDENCE_STRENGTHS, PROJECT_ROOT, claim_id_for, classify_evidence_provenance,
    sha256_file, sha256_text, text_evidence_id_for, visual_evidence_id_for,
)

DATA = PROJECT_ROOT / "data"
CORPORA = DATA / "corpora"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FUSION_DIR = PROCESSED / "fusion"


def _relpath(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


# ── Claim-evidence link lookups (from the authoritative Day 3A fusion output) ──

def _claim_links(evidence_units: pd.DataFrame, modality: str) -> Dict[str, dict]:
    """Returns {evidence_id: {"claim_ids": [...], "stances": [...],
    "source_independences": [...], "evidence_strengths": [...]}} for every
    real (non self-referencing) evidence unit of the given modality."""
    sub = evidence_units[(evidence_units["modality"] == modality) & (evidence_units["source_type"] != "official_claim")]
    links: Dict[str, dict] = {}
    for evidence_id, group in sub.groupby("evidence_id"):
        links[str(evidence_id)] = {
            "claim_ids": group["claim_id"].astype(str).tolist(),
            "stances": group["stance"].astype(str).tolist(),
            "source_independences": group["source_independence"].astype(str).tolist(),
            "evidence_strengths": group["evidence_strength"].astype(str).tolist(),
        }
    return links


def _join_field(values: List[str]) -> str:
    return ";".join(v for v in values if v and v != "nan")


# ── 1. tala_claims ──────────────────────────────────────────────────────────────

def build_claims_records() -> List[dict]:
    fusion_results = _read_csv(FUSION_DIR / "claim_fusion_results.csv")
    claims_expanded = _read_csv(INTERIM / "day2" / "claims_expanded.csv")
    evidence_units = _read_csv(FUSION_DIR / "claim_evidence_units.csv")

    if fusion_results.empty:
        return []

    meta_cols = ["claim_id", "source_url", "source_platform", "evidence_strength"]
    merged = fusion_results.merge(
        claims_expanded[meta_cols] if not claims_expanded.empty else pd.DataFrame(columns=meta_cols),
        on="claim_id", how="left",
    )

    official_units = evidence_units[evidence_units["source_type"] == "official_claim"].set_index("claim_id") if not evidence_units.empty else pd.DataFrame()

    records = []
    for _, r in merged.iterrows():
        claim_id = str(r["claim_id"])
        source_strength = r.get("evidence_strength")
        if pd.isna(source_strength) or not source_strength:
            source_strength = official_units.loc[claim_id, "evidence_strength"] if claim_id in getattr(official_units, "index", []) else "strong"
        metadata = {
            "claim_id": claim_id,
            "claim_text": str(r["claim_text"]),
            "claim_category": str(r["claim_category"]),
            "source_id": claim_id,
            "source_url": "" if pd.isna(r.get("source_url")) else str(r.get("source_url")),
            "source_type": "official_claim",
            "source_platform": "" if pd.isna(r.get("source_platform")) else str(r.get("source_platform")),
            "brand": "TALA",
            "evidence_strength": str(source_strength),
            "fusion_label": str(r["automated_label"]),
            "business_facing_label": str(r["business_facing_label"]),
            "confidence": float(r["confidence"]),
            "conflicting_evidence": bool(r["evidence_conflict"]),
            "self_reported_only": bool(r["self_reported_only"]),
            "external_validation_recommended": bool(r["external_validation_recommended"]),
            "presentation_restriction": str(r["presentation_restriction"]),
            "presentation_restriction_reason": "" if pd.isna(r.get("presentation_restriction_reason")) else str(r.get("presentation_restriction_reason")),
            "evidence_gap_finding": bool(r.get("evidence_gap_finding", False)),
        }
        records.append({
            "id": claim_id_for(claim_id), "document": str(r["claim_text"]),
            "embedding_text": str(r["claim_text"]), "metadata": metadata,
        })
    return records


# ── 2. tala_text_evidence ────────────────────────────────────────────────────────

def _corpus_text_records(path: Path, corpus_source_type: str, evidence_units: pd.DataFrame, join_modality: Optional[str] = None) -> List[dict]:
    """Builds text-evidence records for one of the three RAG corpus CSVs
    (official_claims / customer_experience / creator_strategy), applying the
    rag_usable and evidence-strength gates and attaching any real claim links."""
    df = _read_csv(path)
    if df.empty:
        return []
    df = df[df["rag_usable"] == True]  # noqa: E712
    df = df[df["evidence_strength"].isin(ELIGIBLE_EVIDENCE_STRENGTHS)]
    df = df[df["extracted_text"].fillna("").astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["document_id"])

    links = _claim_links(evidence_units, join_modality) if join_modality else {}

    records = []
    for _, r in df.iterrows():
        doc_id = str(r["document_id"])
        link = links.get(doc_id, {})
        claim_ids = link.get("claim_ids", [])
        stances = link.get("stances", [])
        source_independences = link.get("source_independences", [])
        text = str(r["extracted_text"])
        # official_claims is brand-published by construction: it is always
        # self-reported, independent of whether a claim-evidence-unit join
        # produced any source_independence value (join_modality=None here, so
        # it never does) -- see Part D of the Executive Cockpit Acceptance
        # Patch. customer_experience/creator_strategy keep using the real,
        # verified join-derived flags rather than any inferred default.
        if corpus_source_type == "official_claims":
            self_reported, independent_source = True, False
        else:
            self_reported = "self_reported" in source_independences
            independent_source = "independent" in source_independences
        source_origin, independence_status = classify_evidence_provenance(
            corpus_source_type, self_reported, independent_source
        )
        metadata = {
            "evidence_id": doc_id, "document_id": doc_id, "claim_id": _join_field(claim_ids),
            "brand": str(r["brand"]), "modality": "text", "source_type": corpus_source_type,
            "source_platform": "" if pd.isna(r.get("source_platform")) else str(r["source_platform"]),
            "source_url": "" if pd.isna(r.get("source_url")) else str(r["source_url"]),
            "source_title": "" if pd.isna(r.get("document_title")) else str(r["document_title"]),
            "publication_date": "" if pd.isna(r.get("publication_date")) else str(r["publication_date"]),
            "creator_handle": "", "evidence_strength": str(r["evidence_strength"]),
            "stance": _join_field(stances) or "unlinked", "support_or_challenge": _join_field(stances) or "unlinked",
            "source_independence": _join_field(source_independences),
            "self_reported": self_reported,
            "independent_source": independent_source,
            "source_origin": source_origin,
            "independence_status": independence_status,
            "verified": True, "rights_basis": "official_direct_public_asset" if corpus_source_type == "official_claims" else "user_authorised",
            "citation_text": text[:400], "product": "", "claim_category": "",
            "timestamp": "" if pd.isna(r.get("retrieved_at")) else str(r["retrieved_at"]), "temporal_role": "",
        }
        records.append({
            "id": text_evidence_id_for(corpus_source_type, doc_id), "document": text,
            "embedding_text": text, "metadata": metadata,
        })
    return records


def _reference_chunk_records(evidence_units: pd.DataFrame) -> List[dict]:
    df = _read_csv(PROCESSED / "reference_document_chunks.csv")
    if df.empty:
        return []
    df = df[df["evidence_strength"].isin(ELIGIBLE_EVIDENCE_STRENGTHS)]
    df = df[df["chunk_text"].fillna("").astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["chunk_id"])

    links = _claim_links(evidence_units, "reference")

    records = []
    for _, r in df.iterrows():
        chunk_id = str(r["chunk_id"])
        link = links.get(chunk_id, {})
        claim_ids = link.get("claim_ids", [])
        stances = link.get("stances", [])
        source_independences = link.get("source_independences", []) or ["self_reported"]
        text = str(r["chunk_text"])
        ref_self_reported = "self_reported" in source_independences
        ref_independent = "independent" in source_independences
        source_origin, independence_status = classify_evidence_provenance(
            "official_reference", ref_self_reported, ref_independent
        )
        metadata = {
            "evidence_id": chunk_id, "document_id": str(r["reference_id"]), "claim_id": _join_field(claim_ids),
            "brand": str(r["brand"]), "modality": "text", "source_type": "official_reference",
            "source_platform": "", "source_url": "" if pd.isna(r.get("source_url")) else str(r["source_url"]),
            "source_title": "" if pd.isna(r.get("document_title")) else str(r["document_title"]),
            "publication_date": "", "creator_handle": "", "evidence_strength": str(r["evidence_strength"]),
            "stance": _join_field(stances) or "unlinked", "support_or_challenge": _join_field(stances) or "unlinked",
            "source_independence": _join_field(source_independences),
            "self_reported": ref_self_reported,
            "independent_source": ref_independent,
            "source_origin": source_origin,
            "independence_status": independence_status,
            "verified": True, "rights_basis": "official_direct_public_asset",
            "citation_text": text[:400], "product": "" if pd.isna(r.get("product_category")) else str(r["product_category"]),
            "claim_category": str(r.get("reference_type", "")), "timestamp": "" if pd.isna(r.get("retrieved_at")) else str(r["retrieved_at"]),
            "temporal_role": "",
        }
        records.append({
            "id": text_evidence_id_for("reference_chunk", chunk_id), "document": text,
            "embedding_text": text, "metadata": metadata,
        })
    return records


def _video_summary_text(row: pd.Series, video_role: str) -> str:
    return (
        f"Processed video {row['video_asset_id']} (role: {video_role}): "
        f"{int(row['n_sampled_frames'])} frames sampled across a {row['duration_seconds']:.1f}s clip. "
        f"Motion intensity proxy {row['motion_intensity_proxy']:.3f}; "
        f"scene-change count proxy {row['scene_change_count_proxy']:.0f}; "
        f"visual consistency score {row['visual_consistency_score']:.3f}; "
        f"opening-to-closing visual change {row['opening_to_closing_visual_change']:.3f}; "
        f"{row['pct_frames_product_similar'] * 100:.0f}% of sampled frames are product-semantically similar. "
        "This is a deterministic summary of genuinely processed temporal/visual features, not a transcript."
    )


def _video_summary_records(evidence_units: pd.DataFrame) -> List[dict]:
    video_level = _read_csv(PROCESSED / "video_level_features.csv")
    video_assets = _read_csv(INTERIM / "day2_6" / "video_assets_expanded.csv")
    if video_level.empty:
        return []
    roles = dict(zip(video_assets["video_asset_id"], video_assets.get("video_role", "")))
    links = _claim_links(evidence_units, "video")

    records = []
    for _, r in video_level.iterrows():
        video_id = str(r["video_asset_id"])
        role = roles.get(video_id, "")
        link = links.get(video_id, {})
        text = _video_summary_text(r, role)
        source_origin, independence_status = classify_evidence_provenance("official_video", True, False)
        metadata = {
            "evidence_id": video_id, "document_id": video_id, "claim_id": _join_field(link.get("claim_ids", [])),
            "brand": "TALA", "modality": "video_summary", "source_type": "official_video",
            "source_platform": "", "source_url": "", "source_title": f"Processed video {video_id}",
            "publication_date": "", "creator_handle": "", "evidence_strength": "strong",
            "stance": _join_field(link.get("stances", [])) or "unlinked",
            "support_or_challenge": _join_field(link.get("stances", [])) or "unlinked",
            "source_independence": _join_field(link.get("source_independences", [])) or "self_reported",
            "self_reported": True, "independent_source": False,
            "source_origin": source_origin, "independence_status": independence_status,
            "verified": True,
            "rights_basis": "official_direct_public_asset", "citation_text": text[:400],
            "product": "", "claim_category": "", "timestamp": "", "temporal_role": str(role),
        }
        records.append({
            "id": text_evidence_id_for("video_summary", video_id), "document": text,
            "embedding_text": text, "metadata": metadata,
        })
    return records


def build_text_evidence_records() -> List[dict]:
    evidence_units = _read_csv(FUSION_DIR / "claim_evidence_units.csv")
    records: List[dict] = []
    records += _corpus_text_records(CORPORA / "official_claims_corpus.csv", "official_claims", evidence_units, join_modality=None)
    records += _corpus_text_records(CORPORA / "customer_experience_corpus.csv", "customer_experience", evidence_units, join_modality="text")
    records += _corpus_text_records(CORPORA / "creator_strategy_corpus.csv", "creator_strategy", evidence_units, join_modality="text")
    records += _reference_chunk_records(evidence_units)
    records += _video_summary_records(evidence_units)
    return records


# ── 3. tala_visual_evidence ──────────────────────────────────────────────────────

def build_visual_evidence_records() -> List[dict]:
    evidence_units = _read_csv(FUSION_DIR / "claim_evidence_units.csv")
    image_links = _claim_links(evidence_units, "image")
    video_links = _claim_links(evidence_units, "video")

    records: List[dict] = []

    # -- official product images (only genuinely downloaded assets) --
    images = _read_csv(INTERIM / "day2_5" / "image_assets.csv")
    if not images.empty:
        images = images[images["processing_status"] == "downloaded"]
        images = images[images["local_path"].fillna("").astype(str).str.strip() != ""]
        images = images.drop_duplicates(subset=["file_hash"])
        for _, r in images.iterrows():
            local_path = str(r["local_path"])
            if not (PROJECT_ROOT / local_path).exists():
                continue  # Part 6 gate: never index a missing asset
            asset_id = str(r["asset_id"])
            link = image_links.get(asset_id, {})
            img_origin, img_independence = classify_evidence_provenance("official_product_image", True, False)
            records.append({
                "id": visual_evidence_id_for("image", asset_id), "document": None,
                "local_path": str(PROJECT_ROOT / local_path), "modality": "image",
                "metadata": {
                    "evidence_id": asset_id, "asset_id": asset_id, "frame_id": "", "video_id": "",
                    "claim_id": _join_field(link.get("claim_ids", [])), "brand": str(r["brand"]),
                    "modality": "image", "source_type": "official_product_image",
                    "source_url": "" if pd.isna(r.get("source_url")) else str(r["source_url"]),
                    "local_asset_path": _relpath(PROJECT_ROOT / local_path),
                    "rights_basis": str(r["rights_or_access_basis"]), "evidence_strength": str(r["evidence_strength"]),
                    "visual_role": str(r["image_role"]), "video_role": "", "frame_index": -1,
                    "timestamp_seconds": -1.0, "opening_middle_closing_position": "",
                    "perceptual_hash": "", "width": int(r["width"]) if pd.notna(r.get("width")) else -1,
                    "height": int(r["height"]) if pd.notna(r.get("height")) else -1,
                    "claim_category": "", "visually_groundable": True,
                    "support_or_challenge": _join_field(link.get("stances", [])) or "unlinked",
                    "self_reported": True, "independent_source": False,
                    "source_origin": img_origin, "independence_status": img_independence,
                },
            })

    # -- genuinely processed video frames only (never metadata-only leads) --
    video_assets = _read_csv(INTERIM / "day2_6" / "video_assets_expanded.csv")
    processed_video_ids = set(video_assets.loc[video_assets["processing_status"] == "processed", "video_asset_id"].astype(str)) if not video_assets.empty else set()
    video_role_map = dict(zip(video_assets.get("video_asset_id", []), video_assets.get("video_role", [])))
    video_brand_map = dict(zip(video_assets.get("video_asset_id", []), video_assets.get("brand", [])))
    video_rights_map = dict(zip(video_assets.get("video_asset_id", []), video_assets.get("rights_or_access_basis", [])))

    POSITION_MAP = {"opening": "opening", "closing": "closing", "half": "middle", "quarter": "middle", "three_quarter": "middle", "interval": "middle"}

    frames = _read_csv(PROCESSED / "video_frame_features.csv")
    if not frames.empty:
        frames = frames[frames["video_asset_id"].astype(str).isin(processed_video_ids)]
        frames = frames.drop_duplicates(subset=["frame_id"])
        for _, r in frames.iterrows():
            local_path = str(r["local_frame_path"])
            if not (PROJECT_ROOT / local_path).exists():
                continue
            video_id = str(r["video_asset_id"])
            frame_id = str(r["frame_id"])
            link = video_links.get(video_id, {})
            frame_origin, frame_independence = classify_evidence_provenance("official_video", True, False)
            records.append({
                "id": visual_evidence_id_for("video_frame", frame_id), "document": None,
                "local_path": str(PROJECT_ROOT / local_path), "modality": "video_frame",
                "metadata": {
                    "evidence_id": frame_id, "asset_id": "", "frame_id": frame_id, "video_id": video_id,
                    "claim_id": _join_field(link.get("claim_ids", [])), "brand": str(video_brand_map.get(video_id, "TALA")),
                    "modality": "video_frame", "source_type": "official_video",
                    "source_url": "", "local_asset_path": _relpath(PROJECT_ROOT / local_path),
                    "rights_basis": str(video_rights_map.get(video_id, "official_direct_public_asset")),
                    "evidence_strength": "strong", "visual_role": "", "video_role": str(video_role_map.get(video_id, "")),
                    "frame_index": int(r["frame_index"]), "timestamp_seconds": float(r["timestamp_seconds"]),
                    "opening_middle_closing_position": POSITION_MAP.get(str(r["sample_position"]), str(r["sample_position"])),
                    "perceptual_hash": str(r.get("perceptual_hash", "")),
                    "width": -1, "height": -1, "claim_category": "", "visually_groundable": True,
                    "support_or_challenge": _join_field(link.get("stances", [])) or "unlinked",
                    "self_reported": True, "independent_source": False,
                    "source_origin": frame_origin, "independence_status": frame_independence,
                },
            })

    return records
