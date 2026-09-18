"""Day 2.5 Part G: claim-level multimodal evidence candidate construction.

For each verified official TALA claim, discovers CANDIDATE (not yet scored)
supporting/related evidence across the other modalities:
  - text:  customer-experience + creator-strategy corpus documents, same brand,
           ranked by sentence-embedding cosine similarity to the claim text.
  - image: TALA official product images, ranked by CLIP image-text similarity
           between the claim text and each image's CLIP embedding.
  - video: TALA genuinely PROCESSED videos only (video_level_features.csv rows
           with n_sampled_frames >= 2) -- never a video lead/thumbnail/title.
  - reference: multimodal_reference_assets.csv, if populated (currently an
           empty template -- see docs/source_log_template.md; no reference
           evidence exists yet, so this modality is honestly reported missing
           for every claim rather than fabricated).

This is a CANDIDATE layer only: it does not classify alignment/divergence
between the claim and the evidence (that is a later fusion/scoring task). A
claim may legitimately end up with missing modalities -- this is reported,
never papered over.

Usage: python scripts/build_claim_multimodal_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_features import (  # noqa: E402
    clip_is_available,
    cosine_similarity,
    get_clip_text_embedding,
    get_or_compute_clip_embedding,
)
from src.text_features import compute_semantic_similarity  # noqa: E402

CLAIMS_PATH = PROJECT_ROOT / "data" / "interim" / "day2" / "claims_expanded.csv"
CUSTOMER_CORPUS_PATH = PROJECT_ROOT / "data" / "corpora" / "customer_experience_corpus.csv"
CREATOR_CORPUS_PATH = PROJECT_ROOT / "data" / "corpora" / "creator_strategy_corpus.csv"
IMAGE_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "image_assets.csv"
VIDEO_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"
VIDEO_LEVEL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "video_level_features.csv"
REFERENCE_ASSETS_PATH = PROJECT_ROOT / "data" / "processed" / "multimodal_reference_assets.csv"
VIDEO_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings" / "video"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "claim_multimodal_evidence_candidates.csv"

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


TEXT_SIMILARITY_THRESHOLD = 0.35
TEXT_TOP_K = 3
# CLIP text-image cosine similarity between UNRELATED pairs on this small (24-image)
# pool sits at ~0.21-0.28 baseline (measured empirically during development -- see
# docs/image_video_codebook.md) -- close to src/video_features.py's own calibrated
# PRODUCT_SIMILARITY_THRESHOLD (0.22) for the much easier "is this a product photo at
# all" question. Matching a SPECIFIC claim to a SPECIFIC image/video is a harder,
# more discriminating task, so thresholds here are set above that baseline and only
# the single best (top-1) candidate is kept -- never every image/video above a loose
# bar, which would fabricate breadth rather than reflect genuine alignment.
IMAGE_SIMILARITY_THRESHOLD = 0.27
VIDEO_SIMILARITY_THRESHOLD = 0.24
IMAGE_VIDEO_TOP_K = 1
# Official product-catalog photography (plain garment-on-model/flat-lay shots) can
# plausibly corroborate a MATERIALS claim (fabric/construction visible in the photo).
# It cannot plausibly corroborate a labour, manufacturing-process, packaging,
# emissions, or circularity claim -- there is nothing in a catalog photo that shows
# a supply-chain practice. Gating by category avoids manufacturing spurious
# image/video "evidence" for claims no photograph could actually support.
VISUALLY_GROUNDABLE_CLAIM_CATEGORIES = {"materials"}
CLAIM_EVIDENCE_STRENGTH_FILTER = ("strong", "medium")


def load_claims() -> pd.DataFrame:
    df = pd.read_csv(CLAIMS_PATH)
    usable = df[(df["rag_usable"] == True) & (df["evidence_strength"].isin(CLAIM_EVIDENCE_STRENGTH_FILTER))].copy()
    return usable.reset_index(drop=True)


def load_text_corpora() -> pd.DataFrame:
    frames = []
    for path, corpus_name in ((CUSTOMER_CORPUS_PATH, "customer_experience"), (CREATOR_CORPUS_PATH, "creator_strategy")):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["corpus_source"] = corpus_name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[out["extracted_text"].notna() & (out["extracted_text"].str.strip() != "")]


def match_text_evidence(claims: pd.DataFrame, corpus: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Returns {claim_id: [(document_id, similarity), ...]} above threshold, top-k."""
    matches: dict[str, list[tuple[str, float]]] = {cid: [] for cid in claims["claim_id"]}
    if corpus.empty:
        return matches

    for brand, claim_group in claims.groupby("brand"):
        brand_corpus = corpus[corpus["brand"] == brand]
        if brand_corpus.empty:
            continue
        sims = compute_semantic_similarity(
            claim_group["claim_text"].tolist(), brand_corpus["extracted_text"].tolist()
        )
        doc_ids = brand_corpus["document_id"].tolist()
        for row_i, (_, claim_row) in enumerate(claim_group.iterrows()):
            scored = sorted(zip(doc_ids, sims[row_i]), key=lambda t: -t[1])
            top = [(doc_id, round(float(score), 4)) for doc_id, score in scored[:TEXT_TOP_K] if score >= TEXT_SIMILARITY_THRESHOLD]
            matches[claim_row["claim_id"]] = top
    return matches


def load_image_embeddings(image_assets: pd.DataFrame) -> dict[str, np.ndarray]:
    embeddings = {}
    if not clip_is_available():
        return embeddings
    for _, r in image_assets.iterrows():
        if r.get("processing_status") != "downloaded" or not r.get("local_path") or not r.get("file_hash"):
            continue
        local_path = PROJECT_ROOT / r["local_path"]
        if not local_path.exists():
            continue
        emb = get_or_compute_clip_embedding(local_path, r["file_hash"])
        if emb is not None:
            embeddings[r["asset_id"]] = emb
    return embeddings


def load_video_embeddings(video_level: pd.DataFrame) -> dict[str, np.ndarray]:
    embeddings = {}
    if not clip_is_available() or video_level.empty:
        return embeddings
    for _, r in video_level.iterrows():
        file_hash = r.get("mean_frame_embedding_file_hash")
        if not isinstance(file_hash, str) or not file_hash:
            continue
        npy_path = VIDEO_EMBEDDINGS_DIR / f"mean_{file_hash}.npy"
        if npy_path.exists():
            embeddings[r["video_asset_id"]] = np.load(npy_path)
    return embeddings


def match_visual_evidence(
    claims: pd.DataFrame, brand: str, embeddings: dict[str, np.ndarray], threshold: float
) -> dict[str, list[tuple[str, float]]]:
    """Top-1 best match only, above threshold, and only for claim categories a
    product photo could plausibly corroborate (see VISUALLY_GROUNDABLE_CLAIM_CATEGORIES)."""
    matches: dict[str, list[tuple[str, float]]] = {cid: [] for cid in claims["claim_id"]}
    if not embeddings or not clip_is_available():
        return matches
    ids = list(embeddings.keys())
    for _, claim_row in claims[claims["brand"] == brand].iterrows():
        if claim_row.get("claim_category") not in VISUALLY_GROUNDABLE_CLAIM_CATEGORIES:
            continue
        text_emb = get_clip_text_embedding(claim_row["claim_text"])
        if text_emb is None:
            continue
        scored = [(aid, round(cosine_similarity(text_emb, embeddings[aid]), 4)) for aid in ids]
        scored.sort(key=lambda t: -t[1])
        best = scored[:IMAGE_VIDEO_TOP_K]
        matches[claim_row["claim_id"]] = [(aid, s) for aid, s in best if s >= threshold]
    return matches


def main() -> int:
    claims = load_claims()
    print(f"Loaded {len(claims)} verified official claims (rag_usable, strong/medium) "
          f"across brands: {claims['brand'].value_counts().to_dict()}")

    corpus = load_text_corpora()
    print(f"Text-evidence corpus pool: {len(corpus)} documents "
          f"(customer_experience + creator_strategy)")
    text_matches = match_text_evidence(claims, corpus)

    image_assets = _safe_read_csv(IMAGE_ASSETS_PATH)
    image_embeddings = load_image_embeddings(image_assets) if not image_assets.empty else {}
    print(f"Image evidence pool: {len(image_embeddings)} embedded official images")

    video_level = _safe_read_csv(VIDEO_LEVEL_FEATURES_PATH)
    video_assets = _safe_read_csv(VIDEO_ASSETS_PATH)
    if not video_level.empty and not video_assets.empty:
        processed_ids = set(video_assets.loc[video_assets["processing_status"] == "processed", "video_asset_id"])
        video_level = video_level[
            video_level["video_asset_id"].isin(processed_ids) & (video_level["n_sampled_frames"] >= 2)
        ]
    video_embeddings = load_video_embeddings(video_level)
    print(f"Video evidence pool: {len(video_embeddings)} genuinely processed video(s) with mean embedding "
          f"(video leads/thumbnails/titles excluded by construction)")

    reference_assets = _safe_read_csv(REFERENCE_ASSETS_PATH)
    print(f"Reference evidence pool: {len(reference_assets)} rows "
          f"({'none -- Multimodal Reference Package not yet populated, see docs/source_log_template.md' if reference_assets.empty else 'populated'})")

    image_id_to_brand = dict(zip(image_assets.get("asset_id", []), image_assets.get("brand", [])))
    video_id_to_brand = dict(zip(video_assets.get("video_asset_id", []), video_assets.get("brand", [])))

    rows = []
    for _, claim in claims.iterrows():
        brand = claim["brand"]
        brand_image_embeddings = {k: v for k, v in image_embeddings.items() if image_id_to_brand.get(k) == brand}
        brand_video_embeddings = {k: v for k, v in video_embeddings.items() if video_id_to_brand.get(k) == brand}

        t_matches = text_matches.get(claim["claim_id"], [])
        i_matches = match_visual_evidence(
            claims[claims["claim_id"] == claim["claim_id"]], brand, brand_image_embeddings, IMAGE_SIMILARITY_THRESHOLD
        ).get(claim["claim_id"], [])
        v_matches = match_visual_evidence(
            claims[claims["claim_id"] == claim["claim_id"]], brand, brand_video_embeddings, VIDEO_SIMILARITY_THRESHOLD
        ).get(claim["claim_id"], [])

        missing = []
        if not t_matches:
            missing.append("text")
        if not i_matches:
            missing.append("image")
        if not v_matches:
            missing.append("video")
        if reference_assets.empty:
            missing.append("reference")

        n_present = sum([bool(t_matches), bool(i_matches), bool(v_matches)])
        if n_present == 3:
            strength = "strong"
        elif n_present == 2:
            strength = "medium"
        elif n_present == 1:
            strength = "weak"
        else:
            strength = "unusable"

        rows.append({
            "evidence_bundle_id": f"bundle_{claim['claim_id']}",
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "claim_category": claim.get("claim_category", ""),
            "product_name": "",
            "product_category": "",
            "text_evidence_ids": ";".join(d for d, _ in t_matches),
            "image_asset_ids": ";".join(d for d, _ in i_matches),
            "video_asset_ids": ";".join(d for d, _ in v_matches),
            "reference_document_ids": "",
            "text_similarity": round(float(np.mean([s for _, s in t_matches])), 4) if t_matches else None,
            "image_similarity": round(float(np.mean([s for _, s in i_matches])), 4) if i_matches else None,
            "video_similarity": round(float(np.mean([s for _, s in v_matches])), 4) if v_matches else None,
            "evidence_strength_summary": strength,
            "missing_modalities": ";".join(missing),
            "candidate_generation_method": (
                "text: sentence-transformers all-MiniLM-L6-v2 cosine similarity >= "
                f"{TEXT_SIMILARITY_THRESHOLD} (top {TEXT_TOP_K}), same brand only; "
                f"image/video: CLIP ViT-B/32 text-image cosine similarity >= {IMAGE_SIMILARITY_THRESHOLD}/"
                f"{VIDEO_SIMILARITY_THRESHOLD}; video restricted to genuinely processed assets "
                "(>=2 sampled frames) -- leads/thumbnails/titles never counted"
            ),
            "requires_human_validation": True,
            "provenance_note": f"generated by scripts/build_claim_multimodal_candidates.py",
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n_text = int((out["text_evidence_ids"] != "").sum())
    n_image = int((out["image_asset_ids"] != "").sum())
    n_video = int((out["video_asset_ids"] != "").sum())
    n_reference = int((out["reference_document_ids"] != "").sum())
    n_all_three = int(((out["text_evidence_ids"] != "") & (out["image_asset_ids"] != "") & (out["video_asset_ids"] != "")).sum())

    print(f"\nSaved: {OUT_PATH.relative_to(PROJECT_ROOT)} ({len(out)} claim bundles)")
    print(f"  Claims with text evidence:      {n_text}/{len(out)}")
    print(f"  Claims with image evidence:     {n_image}/{len(out)}")
    print(f"  Claims with video evidence:     {n_video}/{len(out)}")
    print(f"  Claims with reference evidence: {n_reference}/{len(out)}")
    print(f"  Claims with text+image+video:   {n_all_three}/{len(out)}")
    print(f"  requires_human_validation:      {len(out)}/{len(out)} (candidate layer only, never auto-scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
