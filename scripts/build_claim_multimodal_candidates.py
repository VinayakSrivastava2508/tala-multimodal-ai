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
VIDEO_ASSETS_PATH_DAY2_5 = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"
VIDEO_ASSETS_PATH_EXPANDED = PROJECT_ROOT / "data" / "interim" / "day2_6" / "video_assets_expanded.csv"
VIDEO_ASSETS_PATH = VIDEO_ASSETS_PATH_EXPANDED if VIDEO_ASSETS_PATH_EXPANDED.exists() else VIDEO_ASSETS_PATH_DAY2_5
VIDEO_LEVEL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "video_level_features.csv"
REFERENCE_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "multimodal_reference_assets.csv"
REFERENCE_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "reference_document_chunks.csv"
CLAIM_REFERENCE_MATCHES_OUT = PROJECT_ROOT / "data" / "processed" / "claim_reference_matches.csv"
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


# Day 2.6A Part I: claim_category -> compatible reference_type(s). A generic
# return policy cannot support a materials claim; a size guide cannot support
# an emissions claim -- this gate is checked BEFORE any semantic similarity is
# computed, so an irrelevant reference_type is never even in the candidate pool.
CLAIM_CATEGORY_TO_REFERENCE_TYPES: dict[str, list[str]] = {
    "materials": ["material_specification", "product_specification"],
    "labour": ["labour_or_ethics_policy", "supplier_disclosure"],
    "manufacturing": ["supplier_disclosure", "material_specification"],
    "packaging": ["packaging_policy"],
    "emissions": ["emissions_disclosure"],
    "circularity": ["sustainability_page", "material_specification"],
    # "other" and any unmapped category: no compatible reference_type is known
    # to plausibly support it, so no semantic reference match is attempted --
    # not gated open by default.
}
REFERENCE_SEMANTIC_THRESHOLD = 0.35


def match_reference_evidence(
    claims: pd.DataFrame, chunks: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Day 2.6A Part I hierarchical claim-reference matching. Returns
    ({claim_id: [reference_id, ...]}, [match_detail_row, ...]).

    Hierarchy, in order (first tier that produces a hit wins for a given claim):
      1. exact product match       -- claim.product_name == chunk.product_category
                                       (not populated on current claims/chunks; kept
                                       for forward-compatibility, never fabricated).
      2. exact claim category      -- claim_category has an unambiguous 1:1 reference_type
                                       mapping AND a chunk of that type exists.
      3. material/certification    -- a material/certification keyword extracted from the
                                       claim text also appears in a chunk's materials/
                                       certifications entity columns.
      4. product-category match    -- as (1) but at category granularity (not populated;
                                       kept for forward-compatibility).
      5. category-gated semantic   -- best cosine-similarity chunk within the
                                       CLAIM_CATEGORY_TO_REFERENCE_TYPES-gated pool, only if
                                       similarity clears REFERENCE_SEMANTIC_THRESHOLD.
    A claim whose category has no compatible reference_type (e.g. 'other') is never
    matched -- there is no tier this claim can legitimately clear.
    """
    matches: dict[str, list[str]] = {cid: [] for cid in claims["claim_id"]}
    detail_rows: list[dict] = []
    if chunks.empty:
        return matches, detail_rows

    from src.reference_features import extract_entities

    for _, claim in claims.iterrows():
        claim_id, brand, category = claim["claim_id"], claim["brand"], claim.get("claim_category", "")
        brand_chunks = chunks[chunks["brand"] == brand]
        if brand_chunks.empty:
            continue

        compatible_types = CLAIM_CATEGORY_TO_REFERENCE_TYPES.get(category, [])
        if not compatible_types:
            continue  # no plausible reference_type for this category -- never gated open

        gated_pool = brand_chunks[brand_chunks["reference_type"].isin(compatible_types)]
        if gated_pool.empty:
            continue

        # Tier 3: material/certification keyword overlap
        claim_entities = extract_entities(str(claim["claim_text"]))
        claim_keywords = set(claim_entities["materials"]) | set(claim_entities["certifications"])
        best_row, best_method, best_score, best_explanation = None, "", 0.0, ""
        if claim_keywords:
            for _, chunk in gated_pool.iterrows():
                chunk_keywords = set(str(chunk.get("materials", "")).split(";")) | set(str(chunk.get("certifications", "")).split(";"))
                overlap = claim_keywords & chunk_keywords
                if overlap:
                    best_row, best_method, best_score = chunk, "material_or_certification_match", 1.0
                    best_explanation = f"keyword overlap: {', '.join(sorted(overlap))}"
                    break

        # Tier 5: category-gated semantic match (only if tier 3 found nothing)
        if best_row is None:
            sims = compute_semantic_similarity([str(claim["claim_text"])], gated_pool["chunk_text"].tolist())[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= REFERENCE_SEMANTIC_THRESHOLD:
                best_row = gated_pool.iloc[best_idx]
                best_method = "category_gated_semantic_match"
                best_score = round(float(sims[best_idx]), 4)
                best_explanation = (
                    f"claim_category '{category}' compatible with reference_type "
                    f"'{best_row['reference_type']}'; semantic similarity {best_score}"
                )

        if best_row is not None:
            ref_id = best_row["reference_id"]
            matches[claim_id] = [ref_id]
            detail_rows.append({
                "claim_id": claim_id, "reference_id": ref_id, "chunk_id": best_row.get("chunk_id", ""),
                "match_method": best_method, "match_score": best_score, "category_gate_passed": True,
                "match_explanation": best_explanation, "requires_human_validation": True,
            })

    return matches, detail_rows


# Day 2.6B Part G: a product video/image may support appearance, presentation,
# movement, styling, visible design/construction, or expressly demonstrated
# functional behaviour. It may NEVER, by itself, establish durability,
# long-term quality, wash performance, emissions, labour conditions, ethical
# sourcing, or certification status -- so visual evidence is only ever offered
# for claim categories where a photo/video is a plausible witness at all.
# (Today that is 'materials' only -- none of TALA's other claim categories
# describe visible appearance/movement, so the gate correctly excludes them.)
VIDEO_INELIGIBLE_CLAIM_CATEGORIES = {"labour", "manufacturing", "packaging", "emissions", "circularity"}


def match_visual_evidence(
    claims: pd.DataFrame, brand: str, embeddings: dict[str, np.ndarray], threshold: float,
    asset_metadata: dict[str, dict] | None = None, modality: str = "image",
) -> tuple[dict[str, list[tuple[str, float]]], list[dict]]:
    """Part G hierarchical matching: (1) exact product, (2) exact product
    handle, (3) verified product category, (4) exact material, (5) claim
    category with semantic confirmation (CLIP text-image similarity, top-1,
    above threshold). Tiers 1-4 require product-linked claim/asset metadata
    (`product_name`/`product_category`/`material`) -- when claims carry no
    product linkage (the current TALA claim set is brand-level, not
    product-linked), those tiers structurally never fire and matching falls
    through to tier 5, which is never fabricated to compensate.

    Returns ({claim_id: [(asset_id, score), ...]}, [match_detail_row, ...]).
    """
    asset_metadata = asset_metadata or {}
    matches: dict[str, list[tuple[str, float]]] = {cid: [] for cid in claims["claim_id"]}
    details: list[dict] = []
    if not embeddings or not clip_is_available():
        return matches, details
    ids = list(embeddings.keys())

    for _, claim_row in claims[claims["brand"] == brand].iterrows():
        category = claim_row.get("claim_category")
        claim_id = claim_row["claim_id"]

        if modality == "video" and category in VIDEO_INELIGIBLE_CLAIM_CATEGORIES:
            continue  # a video can never establish this category of claim by itself
        if category not in VISUALLY_GROUNDABLE_CLAIM_CATEGORIES:
            continue

        claim_product = str(claim_row.get("product_name") or "")
        claim_category_val = str(claim_row.get("product_category") or "")
        claim_material = str(claim_row.get("material") or "")

        # Tier 1: exact product name
        tier1 = [aid for aid in ids if claim_product and asset_metadata.get(aid, {}).get("product_name") == claim_product]
        # Tier 2: exact product handle
        tier2 = [aid for aid in ids if claim_product and asset_metadata.get(aid, {}).get("product_handle") == claim_product]
        # Tier 3: verified product category
        tier3 = [aid for aid in ids if claim_category_val and asset_metadata.get(aid, {}).get("product_category") == claim_category_val]
        # Tier 4: exact material
        tier4 = [aid for aid in ids if claim_material and asset_metadata.get(aid, {}).get("material") == claim_material]

        for tier_name, tier_ids in (("exact_product", tier1), ("exact_product_handle", tier2),
                                     ("verified_product_category", tier3), ("exact_material", tier4)):
            if tier_ids:
                aid = tier_ids[0]
                matches[claim_id] = [(aid, 1.0)]
                details.append({
                    "claim_id": claim_id, "asset_id": aid, "modality": modality, "match_method": tier_name,
                    "match_score": 1.0, "category_gate_passed": True,
                    "match_explanation": f"{tier_name} match on product-linked metadata",
                    "evidence_eligibility": "presentation_only" if modality == "video" else "visual_evidence",
                    "requires_human_validation": True,
                })
                break
        else:
            # Tier 5: category-gated semantic match
            text_emb = get_clip_text_embedding(claim_row["claim_text"])
            if text_emb is None:
                continue
            scored = [(aid, round(cosine_similarity(text_emb, embeddings[aid]), 4)) for aid in ids]
            scored.sort(key=lambda t: -t[1])
            best = scored[:IMAGE_VIDEO_TOP_K]
            top = [(aid, s) for aid, s in best if s >= threshold]
            if top:
                matches[claim_id] = top
                aid, score = top[0]
                details.append({
                    "claim_id": claim_id, "asset_id": aid, "modality": modality, "match_method": "category_gated_semantic_match",
                    "match_score": score, "category_gate_passed": True,
                    "match_explanation": f"claim_category '{category}' compatible with {modality} evidence; semantic similarity {score}",
                    "evidence_eligibility": "presentation_only" if modality == "video" else "visual_evidence",
                    "requires_human_validation": True,
                })
    return matches, details


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
    reference_chunks = _safe_read_csv(REFERENCE_CHUNKS_PATH)
    n_verified_ref = int(reference_assets["reference_status"].isin(["collected", "already_available"]).sum()) if not reference_assets.empty and "reference_status" in reference_assets.columns else 0
    print(f"Reference evidence pool: {n_verified_ref} verified document(s), {len(reference_chunks)} chunk(s) "
          f"({'none -- Multimodal Reference Package not yet populated' if n_verified_ref == 0 else 'populated'})")
    reference_matches, reference_match_details = match_reference_evidence(claims, reference_chunks)
    if reference_match_details:
        pd.DataFrame(reference_match_details).to_csv(CLAIM_REFERENCE_MATCHES_OUT, index=False)
        print(f"Saved: {CLAIM_REFERENCE_MATCHES_OUT.relative_to(PROJECT_ROOT)} ({len(reference_match_details)} claim-reference matches)")

    image_id_to_brand = dict(zip(image_assets.get("asset_id", []), image_assets.get("brand", [])))
    video_id_to_brand = dict(zip(video_assets.get("video_asset_id", []), video_assets.get("brand", [])))
    image_metadata = {
        r["asset_id"]: {"product_name": r.get("product_name", ""), "product_category": r.get("product_category", ""), "material": ""}
        for _, r in image_assets.iterrows()
    } if not image_assets.empty else {}
    video_metadata = {
        r["video_asset_id"]: {"product_name": r.get("product_name", ""), "product_category": r.get("product_category", ""), "material": ""}
        for _, r in video_assets.iterrows()
    } if not video_assets.empty else {}

    visual_match_details: list[dict] = []
    image_matches_all, video_matches_all = {}, {}
    for brand in claims["brand"].unique():
        brand_image_embeddings = {k: v for k, v in image_embeddings.items() if image_id_to_brand.get(k) == brand}
        brand_video_embeddings = {k: v for k, v in video_embeddings.items() if video_id_to_brand.get(k) == brand}
        m, d = match_visual_evidence(claims, brand, brand_image_embeddings, IMAGE_SIMILARITY_THRESHOLD, image_metadata, modality="image")
        image_matches_all.update({k: v for k, v in m.items() if v})
        visual_match_details.extend(d)
        m, d = match_visual_evidence(claims, brand, brand_video_embeddings, VIDEO_SIMILARITY_THRESHOLD, video_metadata, modality="video")
        video_matches_all.update({k: v for k, v in m.items() if v})
        visual_match_details.extend(d)

    if visual_match_details:
        VISUAL_MATCHES_OUT = PROJECT_ROOT / "data" / "processed" / "claim_visual_evidence_matches.csv"
        pd.DataFrame(visual_match_details).to_csv(VISUAL_MATCHES_OUT, index=False)
        print(f"Saved: {VISUAL_MATCHES_OUT.relative_to(PROJECT_ROOT)} ({len(visual_match_details)} claim-visual matches)")

    rows = []
    for _, claim in claims.iterrows():
        brand = claim["brand"]
        t_matches = text_matches.get(claim["claim_id"], [])
        i_matches = image_matches_all.get(claim["claim_id"], [])
        v_matches = video_matches_all.get(claim["claim_id"], [])
        r_matches = reference_matches.get(claim["claim_id"], [])

        missing = []
        if not t_matches:
            missing.append("text")
        if not i_matches:
            missing.append("image")
        if not v_matches:
            missing.append("video")
        if not r_matches:
            missing.append("reference")

        # Strength reflects how many of the four modalities (text/image/video/reference)
        # have candidate evidence, not just text+image+video -- a bundle with verified
        # text + reference evidence is genuinely processed, not "unusable", even though
        # it is not yet fully (text+image+video) multimodal. See
        # docs/assignment_alignment_audit.md "Claim bundle evidence" for the full
        # breakdown by combination (scripts/audit_assignment_modalities.py).
        n_present = sum([bool(t_matches), bool(i_matches), bool(v_matches), bool(r_matches)])
        if t_matches and i_matches and v_matches:
            strength = "strong"
        elif n_present >= 2:
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
            "reference_document_ids": ";".join(r_matches),
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
