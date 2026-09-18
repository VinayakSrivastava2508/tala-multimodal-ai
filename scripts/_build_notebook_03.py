"""One-off generator for notebooks/03_video_pipeline_and_multimodal_evidence.ipynb.
Not part of the Day 2.5 pipeline -- run once to (re)build the notebook skeleton,
then execute it with nbconvert. Safe to delete after the notebook exists; kept
for reproducibility of how the notebook was authored."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# 03 — Video Pipeline and Multimodal Evidence (Day 2.5)

This notebook is the **authoritative evidence** that the genuine Video Package/Pipeline and the
claim-level multimodal evidence candidate layer exist and were actually run — Notebook 02 covers
the Text Package and descriptive creator/engagement analysis only and is **not** relied on here.

Per `CLAUDE.md` §9 / `docs/project_charter.md`: a YouTube URL, oEmbed thumbnail, title,
description, or engagement statistic is a **video lead**, never a processed video asset.
Everything below distinguishes the two explicitly, and reports shortfalls honestly rather than
fabricating coverage.

Pipeline entry point: `python scripts/run_day2_5_modality_alignment.py`
Authoritative status: `outputs/tables/assignment_modality_audit.csv`
Full narrative: `docs/assignment_alignment_audit.md`""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from IPython.display import Image as IPyImage, display as ipy_display

pd.set_option("display.max_colwidth", 100)
RANDOM_SEED = 42

PROCESSED = PROJECT_ROOT / "data" / "processed"
INTERIM_DAY2_5 = PROJECT_ROOT / "data" / "interim" / "day2_5"
TABLES = PROJECT_ROOT / "outputs" / "tables"

image_assets = pd.read_csv(INTERIM_DAY2_5 / "image_assets.csv")
video_assets = pd.read_csv(INTERIM_DAY2_5 / "video_assets.csv")
video_frames = pd.read_csv(PROCESSED / "video_frame_features.csv")
video_level = pd.read_csv(PROCESSED / "video_level_features.csv")
claim_candidates = pd.read_csv(PROCESSED / "claim_multimodal_evidence_candidates.csv")
modality_audit = pd.read_csv(TABLES / "assignment_modality_audit.csv")
print("Loaded Day 2.5 outputs.")""")

md("## 1. Image Package — official catalog images\n\nCollected directly from each brand's own public Shopify `products.json` endpoint (`scripts/collect_official_product_media.py`). No search-result thumbnails, no login/anti-bot bypass.")

code("""display(image_assets.groupby(["brand", "image_role", "rights_or_access_basis", "processing_status"]).size().reset_index(name="n_rows"))
print(f"\\nTotal: {len(image_assets)} rows, {int((image_assets['processing_status']=='downloaded').sum())} successfully downloaded")
print(f"Duplicate source URLs: {int(image_assets['source_url'].duplicated().sum())}")
print(f"Duplicate file hashes: {int(image_assets.loc[image_assets['file_hash'].notna(),'file_hash'].duplicated().sum())}")""")

code("""sample_row = image_assets[image_assets["processing_status"] == "downloaded"].iloc[0]
sample_path = PROJECT_ROOT / sample_row["local_path"]
print(f"Example verified official image: {sample_row['asset_id']} ({sample_row['brand']}) -- {sample_row['product_name']}")
if sample_path.exists():
    ipy_display(IPyImage(filename=str(sample_path), width=220))""")

md("""## 2. Video Package — leads vs. genuinely processed assets

**Binding rule:** a row counts as a processed video asset only if `rights_or_access_basis` permits
local processing AND `processing_status == 'processed'` AND a matching `video_level_features.csv`
row exists with `n_sampled_frames >= 2`. Everything else — including all 61
`platform_metadata_only` YouTube/TikTok rows here — is a **video lead**, reported separately and
never counted toward coverage.""")

code("""video_coverage = pd.read_csv(TABLES / "video_asset_coverage.csv")
display(video_coverage)

n_leads = int((video_assets["rights_or_access_basis"] == "platform_metadata_only").sum())
n_processed = int((video_assets["processing_status"] == "processed").sum())
print(f"\\nVideo leads (metadata only, NOT processed): {n_leads}")
print(f"Genuinely processed video assets: {n_processed}")
print(f"Genuinely downloaded (any status): {int((video_assets['local_path'].notna() & (video_assets['local_path']!='')).sum())}")""")

code("""video_role_dist = pd.read_csv(TABLES / "video_role_distribution.csv")
display(video_role_dist)""")

md("""### 2a. Shortfall vs. target — reported honestly

Target: 12–20 processable video assets across ≥3 brands, ≥4 TALA, ≥2 distinct video roles.
**Actual: 1 processed video asset, 1 brand (TALA), 1 role (`product_demonstration`).**

Route 1 (official product-page embedded video) found a direct `<source src=.mp4>` only on TALA's
storefront theme — Adanola, Girlfriend Collective, and Oner Active product pages (25 scanned each)
had none. Route 2 (open-licensed video) found none. Route 3 (brand-authorised local video files
under `data/raw/authorised_video_assets/`) has 0 files supplied. No yt-dlp, unofficial downloader,
or stream extraction was used against YouTube/TikTok/Instagram to close this gap — per the charter,
that would be a prohibited access method, not a legitimate remediation.

**This is a genuine, documented shortfall.** The Video Package is marked PARTIAL in
`outputs/tables/assignment_modality_audit.csv`, not silently reported as met.""")

md("## 3. Genuine temporal video processing — frame-level and video-level features")

code("""display(video_level)
print(f"\\nSampled frames for {video_level['video_asset_id'].iloc[0]}: {int(video_level['n_sampled_frames'].iloc[0])}")
print(f"Duration: {video_level['duration_seconds'].iloc[0]:.1f}s, fps: {video_level['fps'].iloc[0]:.1f}")
print(f"has_audio_stream: {video_level['has_audio_stream'].iloc[0]} (None = unknown -- no system ffprobe binary in this environment, a documented limitation, not a fabricated False)")""")

code("""display(video_frames[["frame_id", "sample_position", "timestamp_seconds", "brightness", "contrast",
                        "saturation", "dominant_color_group", "edge_density", "clip_embedding_available",
                        "product_semantic_similarity"]])""")

md("""### 3a. Genuine temporal features (not proxy/thumbnail-only)

These require ≥2 distinct sampled frames and are computed across the frame sequence — a single
thumbnail could never produce them.""")

code("""temporal_cols = ["video_asset_id", "n_sampled_frames", "mean_inter_frame_perceptual_distance",
                  "clip_embedding_change_mean", "scene_change_count_proxy", "motion_intensity_proxy",
                  "visual_consistency_score", "opening_to_closing_visual_change", "pct_frames_product_similar"]
display(video_level[temporal_cols])
video_feature_summary = pd.read_csv(TABLES / "video_feature_summary.csv")
display(video_feature_summary)""")

md("""## 4. Claim-level multimodal evidence candidates

Links each verified official TALA claim to candidate text/image/video/reference evidence
(`scripts/build_claim_multimodal_candidates.py`). This is a **candidate layer**, not a scored
divergence assessment — every row requires human validation before use downstream.""")

code("""n_claims = len(claim_candidates)
n_text = int((claim_candidates["text_evidence_ids"].fillna("") != "").sum())
n_image = int((claim_candidates["image_asset_ids"].fillna("") != "").sum())
n_video = int((claim_candidates["video_asset_ids"].fillna("") != "").sum())
n_reference = int((claim_candidates["reference_document_ids"].fillna("") != "").sum())
n_all3 = int(((claim_candidates["text_evidence_ids"].fillna("") != "")
              & (claim_candidates["image_asset_ids"].fillna("") != "")
              & (claim_candidates["video_asset_ids"].fillna("") != "")).sum())

print(f"Claims represented: {n_claims}")
print(f"  with text evidence:      {n_text}/{n_claims}")
print(f"  with image evidence:     {n_image}/{n_claims}")
print(f"  with video evidence:     {n_video}/{n_claims}")
print(f"  with reference evidence: {n_reference}/{n_claims}")
print(f"  with text+image+video:   {n_all3}/{n_claims}")
print(f"  requires_human_validation: {int(claim_candidates['requires_human_validation'].sum())}/{n_claims}")""")

code("""display(claim_candidates[["claim_id", "claim_category", "claim_text", "text_evidence_ids",
                           "image_asset_ids", "video_asset_ids", "evidence_strength_summary",
                           "missing_modalities"]].head(10))""")

md("""**Note on image/video matching:** candidate image/video links are gated to claims in the
`materials` category — a catalog product photo can plausibly corroborate a materials claim, but
nothing in a garment photo can corroborate a labour, manufacturing-process, packaging, emissions,
or circularity claim. No image/video match is fabricated for those categories regardless of CLIP
similarity score. Video matching additionally draws from a pool of exactly one genuinely processed
video, so a 0/37 video-match result reflects real evidence scarcity, not a broken matcher.""")

md("## 5. Modality-gap findings — assignment modality audit")

code("""display(modality_audit)""")

md("""## 6. Descriptive engagement — retained, never a modelling target

Engagement prediction is out of scope for the entire project (`CLAUDE.md` §9). The metrics below
are retained strictly for descriptive, non-causal comparison (medians, IQRs, coverage) — see
Notebook 02 §9c for the full modelling-feasibility reclassification (tracks 4–10 marked
`OUT_OF_SCOPE`).""")

code("""descriptive_engagement = pd.read_csv(TABLES / "descriptive_creator_engagement.csv")
display(descriptive_engagement)""")

md("""## 7. Transition to primary claim-evidence fusion and RAG

**Video Package: PARTIAL.** **Multimodal Reference Package: FAIL.** **Claim-evidence candidate
layer: PARTIAL** (0/37 claims have all three of text+image+video evidence; reference evidence is
missing for all 37 claims because the Reference Package is not yet populated).

### GO / NO-GO

| Downstream stage | Verdict |
|---|---|
| Primary claim-evidence fusion | **NO-GO** — Video and Reference packages are not yet adequate for a representative fusion input. |
| Multimodal RAG | **NO-GO** — full multimodal RAG (text + image + video + reference) is premature until Video and Reference packages close their gaps. |

**Remediation before re-attempting fusion/RAG:**
1. Place brand-authorised local video files under `data/raw/authorised_video_assets/` for
   Adanola, Girlfriend Collective, and Oner Active (no yt-dlp/unofficial downloaders).
2. Collect Multimodal Reference Package documents (impact reports, certifications, material
   specs, sizing/care guidance, return policies) per brand.
3. Re-run `python scripts/run_day2_5_modality_alignment.py --refresh` and re-check
   `outputs/tables/assignment_modality_audit.csv`.

Full narrative: `docs/assignment_alignment_audit.md`.""")

code("""print("=" * 60)
print("NOTEBOOK 03 COMPLETE -- Day 2.5 Video Pipeline and Multimodal Evidence")
print("=" * 60)
print(f"  Verified official images:        {int((image_assets['processing_status']=='downloaded').sum())}")
print(f"  Video leads (not processed):     {n_leads}")
print(f"  Genuinely processed videos:      {n_processed}")
print(f"  Claim bundles:                   {n_claims}")
print(f"  Claims with all 3 modalities:    {n_all3}/{n_claims}")
print(f"  Primary fusion:                  NO-GO (see §7)")
print(f"  Multimodal RAG:                  NO-GO (see §7)")""")

nb["cells"] = cells
with open("notebooks/03_video_pipeline_and_multimodal_evidence.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook written.")
