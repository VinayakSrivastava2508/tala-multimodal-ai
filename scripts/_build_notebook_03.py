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

INTERIM_DAY2_6 = PROJECT_ROOT / "data" / "interim" / "day2_6"

image_assets = pd.read_csv(INTERIM_DAY2_5 / "image_assets.csv")
video_assets = pd.read_csv(INTERIM_DAY2_5 / "video_assets.csv")
video_frames = pd.read_csv(PROCESSED / "video_frame_features.csv")
video_level = pd.read_csv(PROCESSED / "video_level_features.csv")
claim_candidates = pd.read_csv(PROCESSED / "claim_multimodal_evidence_candidates.csv")
modality_audit = pd.read_csv(TABLES / "assignment_modality_audit.csv")

reference_assets = pd.read_csv(INTERIM_DAY2_6 / "multimodal_reference_assets.csv")
reference_chunks = pd.read_csv(PROCESSED / "reference_document_chunks.csv")
reference_tables_df = pd.read_csv(PROCESSED / "reference_tables.csv")
reference_images_df = pd.read_csv(PROCESSED / "reference_images.csv")
tala_requirement_status = pd.read_csv(TABLES / "day2_6_tala_requirement_status.csv")
reference_package_readiness = pd.read_csv(TABLES / "day2_6_reference_package_readiness.csv")
claim_reference_matches = pd.read_csv(PROCESSED / "claim_reference_matches.csv") if (PROCESSED / "claim_reference_matches.csv").exists() else pd.DataFrame()
print("Loaded Day 2.5 + Day 2.6A outputs.")""")

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

md("""## 5. Multimodal Reference Package (Day 2.6A)

Sustainability/impact pages, certifications, material/product specs, size/care guides, policies,
supplier disclosures, and brand style references for TALA (primary) and, for secondary
benchmarking only, the three competitor brands. Pipeline entry point:
`python scripts/run_day2_6a_reference_package.py`. Full narrative: `docs/assignment_alignment_audit.md`.""")

md("### 5.1 Reference discovery funnel")

code("""funnel = reference_assets.groupby(["discovery_method", "reference_status"]).size().reset_index(name="n_rows")
display(funnel)
print(f"\\nTotal rows attempted: {len(reference_assets)}")
print(f"Verified (collected/already_available): {int(reference_assets['reference_status'].isin(['collected','already_available']).sum())}")
print(f"Not found: {int((reference_assets['reference_status']=='not_found').sum())}, "
      f"Blocked: {int((reference_assets['reference_status']=='blocked').sum())}, "
      f"Duplicate: {int((reference_assets['reference_status']=='duplicate').sum())}, "
      f"Extraction failed: {int((reference_assets['reference_status']=='extraction_failed').sum())}")""")

md("### 5.2 TALA reference requirement coverage (13-item checklist)")

code("""display(tala_requirement_status[["requirement", "status", "reference_type", "pages_checked", "discovery_method"]])
n_found = int((tala_requirement_status["status"] == "found").sum())
print(f"\\n{n_found}/{len(tala_requirement_status)} TALA requirements found. Not-found items are documented, never fabricated.")""")

md("### 5.3 Reference type distribution")

code("""reference_type_dist = pd.read_csv(TABLES / "day2_6_reference_type_distribution.csv")
display(reference_type_dist)
reference_coverage = pd.read_csv(TABLES / "day2_6_reference_coverage.csv")
display(reference_coverage)""")

md("### 5.4 Extracted document examples")

code("""verified_refs = reference_assets[reference_assets["reference_status"].isin(["collected", "already_available"])]
if not verified_refs.empty:
    sample = verified_refs.sort_values("text_length", ascending=False).iloc[0]
    print(f"{sample['reference_id']} ({sample['brand']}, {sample['reference_type']}) -- {sample['document_title']}")
    print(f"Source: {sample['source_url']}")
    print(f"Evidence strength: {sample['evidence_strength']}, text_length: {sample['text_length']}")
    print(f"\\nExtract (first 600 chars):\\n{str(sample['extracted_text'])[:600]}")
else:
    print("No verified reference documents available.")""")

md("### 5.5 Table extraction examples")

code("""if not reference_tables_df.empty:
    display(reference_tables_df[["table_id", "reference_id", "table_type", "headers", "source_url", "evidence_strength"]].head(5))
else:
    print("No structured tables were extracted from the currently collected reference documents.")""")

md("### 5.6 Document-image examples")

code("""if not reference_images_df.empty:
    display(reference_images_df[["reference_image_id", "reference_id", "image_role", "caption", "source_url", "evidence_strength"]].head(5))
else:
    print("No qualifying (non-decorative) document images were extracted from the currently collected reference documents.")""")

md("### 5.7 Claim-reference matching (Part I hierarchy)")

code("""if not claim_reference_matches.empty:
    display(claim_reference_matches[["claim_id", "reference_id", "match_method", "match_score",
                                       "category_gate_passed", "match_explanation"]].head(10))
    print(f"\\nTotal claim-reference matches: {len(claim_reference_matches)}")
    print(claim_reference_matches["match_method"].value_counts().to_string())
else:
    print("No claim-reference matches yet -- see day2_6_tala_requirement_status.csv for what was searched.")""")

md("""**Matching hierarchy** (first tier that clears wins): (1) exact product match, (2) exact claim
category, (3) material/certification keyword overlap, (4) product-category match, (5)
category-gated semantic match. A generic return policy can never support a materials claim, and a
size guide can never support an emissions claim -- the category gate is checked BEFORE any
similarity score is computed, so an incompatible reference_type is never even in the candidate
pool. Every match is a **candidate**, not a supported/contradicted classification.""")

md("### 5.8 Updated claim modality coverage")

code("""claim_reference_coverage = pd.read_csv(TABLES / "day2_6_claim_reference_coverage.csv")
display(claim_reference_coverage)

bundle_summary_after = pd.read_csv(TABLES / "claim_evidence_bundle_summary.csv")
display(bundle_summary_after.T.rename(columns={0: "count_after_reference_package"}))""")

md("### 5.9 Reference Package GO/NO-GO")

code("""display(reference_package_readiness)
ref_overall = reference_package_readiness[reference_package_readiness["criterion"] == "OVERALL"].iloc[0]
print(f"\\nReference Package: {ref_overall['status']}")
print("A private internal TALA style guide is NOT required for GO -- its absence is documented as a limitation, never a blocker.")""")

md("""### 5.10 Remaining video blocker (untouched by this task)

Per the Day 2.6A scope, video collection/fusion/RAG are explicitly out of scope for this task --
the Video Package/Pipeline verdicts from Notebook 03 §2/§3 are carried forward unchanged into the
primary-fusion readiness check below.""")

md("## 6. Modality-gap findings — assignment modality audit")

code("""display(modality_audit)""")

md("""## 7. Descriptive engagement — retained, never a modelling target

Engagement prediction is out of scope for the entire project (`CLAUDE.md` §9). The metrics below
are retained strictly for descriptive, non-causal comparison (medians, IQRs, coverage) — see
Notebook 02 §9c for the full modelling-feasibility reclassification (tracks 4–10 marked
`OUT_OF_SCOPE`).""")

code("""descriptive_engagement = pd.read_csv(TABLES / "descriptive_creator_engagement.csv")
display(descriptive_engagement)""")

md("""## 8. Primary claim-evidence fusion readiness (minimum defensible bar)

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
print("NOTEBOOK 03 COMPLETE -- Video Pipeline + Multimodal Reference Package (Day 2.5 + Day 2.6A)")
print("=" * 60)
print(f"  Verified official images:        {int((image_assets['processing_status']=='downloaded').sum())}")
print(f"  Video leads (not processed):     {n_leads}")
print(f"  Genuinely processed videos:      {n_processed}")
print(f"  Verified reference documents:    {int(reference_assets['reference_status'].isin(['collected','already_available']).sum())}")
print(f"  TALA requirements found:         {int((tala_requirement_status['status']=='found').sum())}/{len(tala_requirement_status)}")
print(f"  Reference chunks/tables/images:  {len(reference_chunks)}/{len(reference_tables_df)}/{len(reference_images_df)}")
print(f"  Claim bundles:                   {n_claims}")
print(f"  Claims processed (>=1 modality): {n_ge1}/{n_claims}")
print(f"  Claims with text+image+video:    {n_all3}/{n_claims}")
print(f"  Reference Package:               {ref_overall['status']} (see \\u00a75.9)")
overall = readiness[readiness["criterion"] == "OVERALL"].iloc[0]
print(f"  Primary fusion:                  {overall['status']} (see \\u00a78)")
print(f"  Multimodal RAG:                  NO-GO (see \\u00a78)")""")

nb["cells"] = cells
with open("notebooks/03_video_pipeline_and_multimodal_evidence.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook written.")
