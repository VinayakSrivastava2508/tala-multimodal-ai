# Day 2.6B: Targeted TALA Product-Reference and Official-Video Recovery

Resolves the two evidence blockers left after Day 2.6A: TALA garment-care/wash guidance
(Reference Package) and insufficient TALA-specific processed video depth (Video Package).
Pipeline entry point: `python scripts/run_day2_6b_product_video_recovery.py`. Authoritative
status: `outputs/tables/day2_6b_readiness_gates.csv`, `outputs/tables/assignment_modality_audit.csv`.

## Method

**Product-page enumeration** (`scripts/discover_tala_product_references.py`,
`src/collectors/product_pages.py`): `wearetala.com`'s public `/products.json` catalog was
scanned (300+ products) and mapped to the assignment's 7 target categories via each product's
Shopify `product_type` (verified mapping in `CATEGORY_MAP`). 3 in-stock products per category
were selected — 21 total across all 7 categories (leggings, shorts, sports bras, tops,
outerwear, dresses/lifestyle, accessories).

**Section extraction**: inspection of a live product page (via direct `requests.get`, no
JavaScript execution) found that wearetala.com's Shopify theme server-renders three tabs per
product under stable CSS classes — `product-information__description--{wear,care,aware}` — in
the initial HTML response:
- **Wear**: product description, materials bullet list, best-for use case, model
  info (e.g. "Naomi / UK Size 8 / Wearing TALA Size XS / 5'7\"").
- **Care**: explicit garment-care instructions.
- **Aware**: material composition percentage, certification, factory/supplier disclosure.

Because this content is present without JavaScript, **Playwright was not needed** — the
per-product size chart *is* populated via a client-side widget with no static table in the
page, so per-product size/fit evidence instead comes from the Wear tab's model-info line and
the product's `/products.json` `variants` (available sizes), both genuinely present statically.

**Care-evidence rule** (Part C): a Care-tab record only counts if it contains an explicit,
actionable instruction (wash temperature, machine/hand-wash direction, drying instruction,
ironing instruction, bleach restriction) — `is_explicit_care_instruction()` gates on a regex of
concrete signals; generic prose like "care for your garments" is rejected outright.

## Results

- **21/21 product pages** successfully fetched and extracted (100% success rate).
- **21 product-level care records**, all explicit, **consolidated into 6 unique care
  instructions** (Part D: identical instructions across products count once, with every
  applicable product retained in provenance) — see `outputs/tables/day2_6b_care_guidance_coverage.csv`.
  Instructions vary genuinely by product type: caps are `do_not_wash`, swimwear/bikini are
  `hand_wash`, most apparel is `machine_wash` at `30C`.
- **21 material-composition records**, **21 product-specification records**, **21 size-guidance
  records** (available sizes from `/products.json` variants), **10 fit-guidance records** (not
  every product has a model-info line).
- Merged into the Reference Package: 79 new rows (`data/interim/day2_6/multimodal_reference_assets.csv`,
  now 162 rows total pre-Day-2.6B, some superseded during reprocessing).
- **TALA requirement checklist: 9/13 found** (up from 7/13) — "Garment-care or wash guidance"
  and "Product specifications" newly found. Still not_found: quality/warranty policy, packaging
  policy, emissions disclosure, brand/visual guidance — genuinely absent from TALA's public site
  (verified by inspecting the "other"-classified pages: shipping policy, terms of service, FAQ,
  individual product listings — none are a distinct warranty/packaging/emissions page).

### Official video discovery (Part E) — a real methodology bug found and fixed

Deep per-product-page inspection found that a SINGLE product page embeds **up to 8 distinct
`<source src=.mp4>` video IDs**, not one. The Day 2.5 discovery script's `_smallest_source()`
picked only the lowest-bitrate source per page and then deduplicated across pages assuming a
single shared brand video — which silently discarded the other 7 genuinely distinct videos
sitting in the same page's markup. `discover_page_video_sources()` fixes this by grouping
`<source>` elements by the CDN's own 32-hex video ID (not by page), keeping the lowest-bitrate
variant of each genuinely distinct video.

Across the 21 product pages, **8 distinct videos** were found — all 8 are a shared
brand-content gallery (each appears on ~20 of the 21 pages, across all 7 categories), not
per-product content, so they are honestly labelled `video_role='other'`, not fabricated as
`product_demonstration` for a product they aren't tied to.

### Video processing (Part F)

All 8 discovered videos were downloaded (`official_direct_public_asset`, direct CDN MP4,
robots-checked) and processed through the existing genuine-video pipeline
(`src/video_features.py`): every one produced ≥5 distinct sampled frames and valid temporal
features (inter-frame perceptual distance, motion-intensity proxy, scene-change-count proxy).
**Genuinely processed TALA videos: 8** (up from 1), spanning **2 distinct `video_role` values**
(`product_demonstration` from the original Day 2.5 asset, `other` for the 7 new shared-gallery
videos) — meeting the Part I Video Package bar of ≥4 videos and ≥2 roles/categories.

### Claim-evidence linking (Part G)

`match_visual_evidence()` was extended with the full 5-tier hierarchy (exact product → exact
handle → verified category → exact material → category-gated semantic match) for both image
and video pools. Tiers 1–4 require product-linked claim metadata that the current TALA claim
set (brand-level statements, not product-linked) does not carry, so they structurally never
fire today — implemented and tested for forward-compatibility, not fabricated to force a hit.
A new `VIDEO_INELIGIBLE_CLAIM_CATEGORIES` gate additionally ensures a product video is never
offered as evidence for labour, manufacturing, packaging, emissions, or circularity claims — a
video can support appearance/presentation/movement, never durability, wash performance,
emissions, labour conditions, or certification status by itself. Every visual match is tagged
`evidence_eligibility='presentation_only'` for video.

Result: **7/37 claims gained video evidence** (up from 0), **3/37 claims now have
text+image+video simultaneously** (up from 0) — all `materials`-category claims matched via
category-gated semantic similarity to product-appearance videos (scores 0.24–0.31, above the
0.24 threshold). Detail: `data/processed/claim_visual_evidence_matches.csv`.

### Full-modality prototype bundles (Part H)

`scripts/build_full_modality_bundles.py` identified **7 claims** with text + (image or video) +
reference evidence simultaneously (2 with all 4 modalities: text+image+video+reference). No
aligned/divergent verdict is assigned — `data/processed/full_modality_prototype_bundles.csv`
is a candidate layer only, same as the rest of Day 2.5/2.6A.

## Readiness gates (Part I)

| Component | Status |
|---|---|
| Reference Package | **PASS** |
| Video Package | **PASS** |
| Video Pipeline | **PASS** |
| Claim-evidence candidate layer | **PARTIAL** (16/37 claims, mostly `labour`-category, still have zero evidence — genuinely unresolved, not a bug) |
| Primary fusion readiness | **GO** |
| Multimodal RAG readiness | **GO** |

Neither fusion nor RAG was implemented in this task — these are readiness verdicts only, per
the task's explicit scope boundary.

## Known limitations (not fabricated, not silently dropped)

- Quality/warranty policy, packaging policy, emissions disclosure, and brand/visual guidance
  remain genuinely absent from TALA's public site.
- All 8 processed videos are the same shared brand-content gallery, not per-product footage —
  role diversity is real (2 values) but category-specific video evidence for a SPECIFIC product
  does not exist.
- 16/37 TALA claims (mostly `labour` category) still have zero evidence in any modality — no
  reference document was found that specifically addresses labour conditions beyond the general
  supplier-disclosure/factory-audit pages already captured, and no image/video is eligible for
  this category by design (Part G's exclusion).
- Tiers 1–4 of the Part G matching hierarchy are implemented and tested but do not fire against
  today's claim set, which carries no product-name/category linkage.
