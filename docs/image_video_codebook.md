# Image / Video Codebook (Day 2.5)

Defines the controlled vocabularies used in `data/interim/day2_5/image_assets.csv`,
`data/interim/day2_5/video_assets.csv`, and the human-validation sample at
`data/manual_labels/image_video_validation_sample.csv`. Schemas are enforced in
`configs/schema.yaml` (`image_assets`, `video_assets`); this document explains what each
value *means* so a human reviewer can label consistently.

## `image_role`

| Value | Meaning |
|---|---|
| `official_catalog` | Primary product photo from the brand's own storefront (first image in a Shopify product's `images` array). |
| `product_detail` | Additional (non-primary) product photo from the same storefront listing. |
| `permitted_ugc` | User-generated content the brand has explicit rights to reuse (e.g. a brand repost with a stated licence). Not currently populated. |
| `creator_thumbnail` | A creator/influencer post thumbnail. **Never counted as an official catalog/product image** — evidence for creator strategy only, per CLAUDE.md §2. |
| `website_ui` | Non-product brand website imagery (banners, layout assets). |
| `packaging` | Product packaging photography. |
| `material_or_label` | Close-up of fabric, care label, or material swatch. |
| `other` | Anything not covered above — must have a `provenance_note` explaining what it is. |

## `video_role`

| Value | Meaning |
|---|---|
| `official_campaign` | Brand-produced marketing/campaign video. |
| `product_demonstration` | Shows a product being worn/used/demonstrated. |
| `try_on` | Try-on/fit content. |
| `unboxing` | Unboxing content. |
| `review` | Review-style content. |
| `durability_test` | Wear-test or durability demonstration. |
| `wash_test` | Wash/care test. |
| `styling` | Styling/outfit content. |
| `responsibility_communication` | Sustainability/responsibility messaging. |
| `other` | Used for video **leads** whose actual content type cannot be determined without watching the platform-hosted video (which this project does not download) — e.g. every `platform_metadata_only` YouTube/TikTok row in `video_assets.csv` is `other` because only a URL/handle is known, not the content. |

## `rights_or_access_basis`

| Value | Meaning | May be locally processed (frames/temporal features)? |
|---|---|---|
| `official_direct_public_asset` | Fetched directly from the brand's own public storefront/CDN (e.g. Shopify `products.json`, an embedded `<source>` URL on a product page). | Yes |
| `open_license` | Explicitly, verifiably open-licensed (e.g. Creative Commons with stated terms). | Yes |
| `group_owned` | Media the project team created or owns outright. | Yes |
| `user_authorised` | Third-party media with explicit authorisation on file, placed under `data/raw/authorised_video_assets/`. | Yes |
| `platform_metadata_only` | A URL, title, thumbnail, or engagement statistic read via an official oEmbed/API endpoint — **not** the underlying media file. This is a **video/image lead**, not a processed asset. | **No** — never processed locally; no yt-dlp/unofficial downloader/stream extraction is used against YouTube/Instagram/TikTok. |
| `unknown` | Access basis not yet determined — flag for human review, do not process. | No |

## `processing_status` (video_assets)

- `processed` — genuinely processed: opened with OpenCV, validated (readable, non-zero duration, ≥2 distinct sampled frames), frame-level and temporal features computed. Only rows with a permitted `rights_or_access_basis` can reach this status.
- `metadata_only` — a lead; never processed, and never eligible to become `processed`.
- `downloaded_pending_processing` — file downloaded, `scripts/process_video_assets.py` not yet run on it (or run and result pending).
- `failed` — file downloaded but failed technical validation (see `video_level_features.csv::invalid_reason` for why).
- `skipped` — candidate discovered but not downloaded (e.g. duplicate file hash, shared brand-wide video already captured once).

## What counts as a "processed video asset" (binding rule)

A row counts toward Video Package / Video Pipeline coverage **only if all of the following hold**:
1. `rights_or_access_basis` is one of the four locally-processable values above.
2. `processing_status == 'processed'`.
3. A matching row exists in `data/processed/video_level_features.csv` with `n_sampled_frames >= 2`.
4. That row has at least one non-null temporal feature (e.g. `mean_inter_frame_perceptual_distance`) — a row with only a single thumbnail's CLIP features does not qualify, because temporal features genuinely require a frame sequence.

A YouTube URL, oEmbed thumbnail, title, description, or engagement statistic never satisfies this — see `outputs/tables/video_asset_coverage.csv` and `outputs/tables/assignment_modality_audit.csv` for the current, code-verified counts.

## Human validation sample

`data/manual_labels/image_video_validation_sample.csv` contains every row from both manifests (86 total: 24 image, 62 video) for human spot-check — small enough at this stage to review in full rather than sample. A reviewer fills in `human_label_correct_role`, `human_label_correct_rights_basis`, and `human_notes`; disagreements should be corrected in the source manifest and re-run through the relevant collection/processing script rather than hand-edited only in the sample file.
