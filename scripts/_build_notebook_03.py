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

md("""### 2a. Video recovery priority and shortfall — reported honestly, not competitor-penalised

Competitor video is strategically useful for secondary brand comparison but is **not mandatory**
for the primary TALA claim-experience objective. Recovery priority order:
1. Permitted TALA official/product videos. 2. Other permitted TALA-relevant videos.
3. Official competitor videos, secondary comparison only. 4. Open-license generic apparel video,
method demonstration only.

Judged against the **TALA-depth target** (ideal ≥4 TALA videos), not competitor brand count:
**1/4 target TALA videos processed, 0 competitor videos (not required).**

Route 1 (official product-page embedded video) found a direct `<source src=.mp4>` only on TALA's
storefront theme — Adanola, Girlfriend Collective, and Oner Active product pages (25 scanned each)
had none. Route 2 (open-licensed video) found none. Route 3 (brand-authorised local video files
under `data/raw/authorised_video_assets/`) has 0 files supplied. No yt-dlp, unofficial downloader,
or stream extraction was used against YouTube/TikTok/Instagram to close this gap.

**This is a genuine, documented shortfall against the TALA depth target** — but the Video
*Pipeline* itself is PASS (100% of genuinely processed videos produced valid temporal features);
only the Video *Package*'s TALA depth is PARTIAL. Competitor absence alone never fails either.""")

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

md("""**A bundle counts as "processed" as soon as it has ≥1 verified modality** — collapsing every
partial bundle (e.g. text + reference, no image/video) down to zero would misreport real evidence
as absent. The breakdown below reports every combination separately
(`outputs/tables/claim_evidence_bundle_summary.csv`, `scripts/audit_assignment_modalities.py::build_claim_evidence_bundle_summary`).""")

code("""bundle_summary = pd.read_csv(TABLES / "claim_evidence_bundle_summary.csv")
display(bundle_summary.T.rename(columns={0: "count"}))

n_claims = int(bundle_summary["n_claims"].iloc[0])
n_ge1 = int(bundle_summary["bundles_with_ge1_verified_modality"].iloc[0])
n_ge2 = int(bundle_summary["bundles_with_ge2_eligible_modalities"].iloc[0])
n_all3 = int(bundle_summary["bundles_with_text_image_video"].iloc[0])
n_reference = int(bundle_summary["bundles_with_reference_evidence"].iloc[0])

print(f"Claims represented: {n_claims}")
print(f"  processed (>=1 verified modality): {n_ge1}/{n_claims}")
print(f"  >=2 eligible modalities:           {n_ge2}/{n_claims}")
print(f"  with reference evidence:           {n_reference}/{n_claims}")
print(f"  with text+image+video:             {n_all3}/{n_claims}")
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

md("""## 7. Primary claim-evidence fusion readiness (minimum defensible bar)

Primary fusion does **not** require all 37 claims to carry every modality. Readiness is judged
against a 7-criterion minimum bar
(`scripts/audit_assignment_modalities.py::evaluate_primary_fusion_readiness`,
`outputs/tables/primary_fusion_readiness.csv`):""")

code("""readiness = pd.read_csv(TABLES / "primary_fusion_readiness.csv")
display(readiness)""")

md("""### Multimodal RAG — modality-routing design principle

RAG is not implemented yet. The binding rule for when it is: **route each claim's retrieval to
the modalities that are actually relevant to it.** A responsibility/labour/sustainability claim
should retrieve text, tables, certificates, and document images — not an irrelevant product
video. A fit/product-presentation/movement claim may legitimately retrieve text, catalog images,
and product/try-on video. This is already implemented at the candidate-generation layer above
(`VISUALLY_GROUNDABLE_CLAIM_CATEGORIES` gating image/video matches to `materials`-category
claims only) — the future RAG retriever must reuse this same category→modality routing table
rather than retrieving all modalities uniformly for every query.

### GO / NO-GO

| Downstream stage | Verdict | Basis |
|---|---|---|
| Primary claim-evidence fusion | **NO-GO** | 3/7 minimum-bar criteria unmet: Reference Package empty, <5 claims with >=2 modalities, 0 claims with a defensible text+image+video bundle. All three pipelines genuinely work -- these are depth gaps, not capability gaps. |
| Multimodal RAG | **NO-GO** | Same underlying gaps -- building RAG now would silently over-rely on text, the only consistently available modality. |

**Remediation, in priority order:**
1. Recover more permitted TALA official/product videos (place any not directly downloadable
   under `data/raw/authorised_video_assets/`); competitor video is secondary and not required.
2. Collect the first Multimodal Reference Package documents (impact reports, certifications,
   material specs, sizing/care guidance, return policies).
3. Re-run `python scripts/run_day2_5_modality_alignment.py --refresh` and re-check
   `outputs/tables/primary_fusion_readiness.csv`.

Full narrative: `docs/assignment_alignment_audit.md`.""")

code("""print("=" * 60)
print("NOTEBOOK 03 COMPLETE -- Day 2.5 Video Pipeline and Multimodal Evidence")
print("=" * 60)
print(f"  Verified official images:        {int((image_assets['processing_status']=='downloaded').sum())}")
print(f"  Video leads (not processed):     {n_leads}")
print(f"  Genuinely processed videos:      {n_processed}")
print(f"  Claim bundles:                   {n_claims}")
print(f"  Claims processed (>=1 modality): {n_ge1}/{n_claims}")
print(f"  Claims with text+image+video:    {n_all3}/{n_claims}")
overall = readiness[readiness["criterion"] == "OVERALL"].iloc[0]
print(f"  Primary fusion:                  {overall['status']} (see \\u00a77)")
print(f"  Multimodal RAG:                  NO-GO (see \\u00a77)")""")

nb["cells"] = cells
with open("notebooks/03_video_pipeline_and_multimodal_evidence.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook written.")
