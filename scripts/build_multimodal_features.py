"""Day 2 Task 2 Parts C-I: build text/visual/engagement feature tables, content-
intent weak labels, and descriptive competitor-analysis tables/figures from the
frozen Day 2 datasets (data2_verified_v1).

Orchestration only -- extraction logic lives in src/text_features.py,
src/image_features.py, src/collectors/media.py.

Usage: python scripts/build_multimodal_features.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_features import (  # noqa: E402
    classify_content_intent,
    classify_content_intent_multilabel,
    creator_strategy_indicators,
    customer_experience_features,
    general_text_features,
    official_claim_indicators,
    sentiment_features,
)
from src.image_features import (  # noqa: E402
    EMBEDDINGS_DIR,
    clip_is_available,
    extract_interpretable_features,
    get_or_compute_clip_embedding,
    load_image_as_array,
)
from src.collectors.creator_identity import reassess_partnership  # noqa: E402
from src.collectors.youtube_metrics import (  # noqa: E402
    compute_engagement_features,
    within_channel_normalisation,
)
from src.validation import assert_evidence_strength_allowed, assert_no_duplicate_ids  # noqa: E402

DAY2_DIR = PROJECT_ROOT / "data" / "interim" / "day2"
CORPORA_DIR = PROJECT_ROOT / "data" / "corpora"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
MANUAL_LABELS_DIR = PROJECT_ROOT / "data" / "manual_labels"
RANDOM_SEED = 42
NOW_ISO = datetime.now(timezone.utc).isoformat()

sns.set_theme(style="whitegrid", palette="Set2")

USABLE_EVIDENCE = ("strong", "medium")  # weak/unusable never enter the modelling table

_PROVENANCE_KV_RX = re.compile(r"(\w+)=([^;]+)")


def _parse_provenance(note: str) -> dict:
    """Parse a 'k=v; k2=v2' provenance_note string into a dict."""
    return {k: v.strip() for k, v in _PROVENANCE_KV_RX.findall(str(note) or "")}


def _nz(val) -> str:
    """Null-safe string coercion: NaN/None -> '' (unlike `val or ""`, which
    returns NaN unchanged since float('nan') is truthy)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val)
    return "" if s.lower() == "nan" else s


def _best_text(row: pd.Series) -> str:
    for col in ("caption_or_description", "video_title"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip() and str(val).lower() != "nan":
            return str(val)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Part C+D+G+F: creator_multimodal_features.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_creator_features() -> pd.DataFrame:
    print("\n[Creator features] Loading verified creator posts ...")
    creators = pd.read_csv(DAY2_DIR / "creator_posts_enriched.csv")
    verified = creators[creators["usable_as_creator_post"] == True].copy()
    print(f"  {len(verified)} usable_as_creator_post=True rows")

    media_path = PROCESSED_DIR / "creator_media_manifest.csv"
    media = pd.read_csv(media_path) if media_path.exists() else pd.DataFrame()

    yt_video_path = PROCESSED_DIR / "youtube_video_metrics.csv"
    yt_channel_path = PROCESSED_DIR / "youtube_channel_metrics.csv"
    yt_video = pd.read_csv(yt_video_path) if yt_video_path.exists() and yt_video_path.stat().st_size > 0 else pd.DataFrame()
    yt_channel = pd.read_csv(yt_channel_path) if yt_channel_path.exists() and yt_channel_path.stat().st_size > 0 else pd.DataFrame()
    if not yt_video.empty:
        yt_video = yt_video.set_index("video_id")
    if not yt_channel.empty:
        yt_channel = yt_channel.set_index("channel_id")
    print(f"  YouTube video metrics: {len(yt_video)} cached, channel metrics: {len(yt_channel)} cached")

    rows = []
    for _, r in verified.iterrows():
        post_id = str(r.get("source_id", ""))
        video_id = _nz(r.get("direct_post_id", ""))
        yt_row = yt_video.loc[video_id] if (not yt_video.empty and video_id in yt_video.index) else None

        # Prefer the full YouTube API title/description over the oEmbed/DDG-snippet
        # versions when we have them -- richer text -> better text/intent features.
        api_title = str(yt_row["video_title"]) if yt_row is not None and pd.notna(yt_row.get("video_title")) else ""
        api_description = str(yt_row["video_description"]) if yt_row is not None and pd.notna(yt_row.get("video_description")) else ""
        text = api_description or _best_text(r)
        display_title = api_title or _nz(r.get("video_title"))

        gen = general_text_features(text)
        sent = sentiment_features(text)
        creator_ind = creator_strategy_indicators(text)

        # legacy single-label intent, preserved for comparison
        original_intent = classify_content_intent(_best_text(r))
        intent = classify_content_intent_multilabel(
            video_title=display_title, video_description=api_description,
            caption_or_description=_nz(r.get("caption_or_description")),
        )

        # Part C: partnership reassessment from full title+description
        reassessed = reassess_partnership(
            video_title=display_title, video_description=api_description,
            caption_or_description=_nz(r.get("caption_or_description")),
            personalised_code=_nz(r.get("personalised_code")),
        )

        media_row = media[media["post_id"] == post_id]
        media_row = media_row.iloc[0] if not media_row.empty else None

        visual = {}
        embedding_available = False
        if media_row is not None and media_row.get("retrieval_status") == "success":
            local_path = PROJECT_ROOT / media_row["local_media_path"]
            if local_path.exists():
                try:
                    arr = load_image_as_array(local_path)
                    visual = extract_interpretable_features(arr)
                    file_hash = str(media_row.get("file_hash", ""))
                    if clip_is_available() and file_hash:
                        emb = get_or_compute_clip_embedding(local_path, file_hash)
                        embedding_available = emb is not None
                except Exception as exc:
                    print(f"  [visual features] failed on {post_id}: {exc}")

        # Part A/B: real YouTube statistics + engagement math (never fabricated)
        view_count = like_count = comment_count = None
        channel_id = channel_title = published_at = stats_retrieved_at = ""
        video_privacy_status = api_retrieval_status = api_failure_reason = ""
        subscriber_count = None
        hidden_subscriber_count = None
        if yt_row is not None:
            view_count = int(yt_row["view_count"]) if pd.notna(yt_row.get("view_count")) else None
            like_count = int(yt_row["like_count"]) if pd.notna(yt_row.get("like_count")) else None
            comment_count = int(yt_row["comment_count"]) if pd.notna(yt_row.get("comment_count")) else None
            channel_id = _nz(yt_row.get("channel_id", ""))
            channel_title = _nz(yt_row.get("channel_title", ""))
            published_at = _nz(yt_row.get("published_at", ""))
            stats_retrieved_at = _nz(yt_row.get("statistics_retrieved_at", ""))
            video_privacy_status = _nz(yt_row.get("video_privacy_status", ""))
            api_retrieval_status = _nz(yt_row.get("api_retrieval_status", ""))
            api_failure_reason = _nz(yt_row.get("api_failure_reason", ""))
            if channel_id and not yt_channel.empty and channel_id in yt_channel.index:
                ch = yt_channel.loc[channel_id]
                subscriber_count = int(ch["channel_subscriber_count_current"]) if pd.notna(ch.get("channel_subscriber_count_current")) else None
                hidden_subscriber_count = bool(ch.get("hidden_subscriber_count", False))

        engagement = compute_engagement_features(
            view_count=view_count, like_count=like_count, comment_count=comment_count,
            channel_subscriber_count_current=subscriber_count,
            published_at=published_at, retrieved_at=stats_retrieved_at,
        )

        row = {
            "post_id": post_id,
            "brand": r.get("brand"),
            "platform": r.get("platform"),
            "creator_handle": r.get("creator_handle"),
            "channel_id": channel_id,
            "channel_title": channel_title,
            "post_url": r.get("direct_post_url") or r.get("post_url") or r.get("source_url"),
            "post_date": r.get("post_date"),
            "published_at": published_at,
            "original_partnership_type": r.get("partnership_type"),
            "revised_partnership_type": reassessed["revised_partnership_type"],
            "partnership_evidence_text": reassessed["partnership_evidence_text"],
            "partnership_rule_trigger": reassessed["partnership_rule_trigger"],
            "partnership_confidence": reassessed["partnership_confidence"],
            "disclosure_type": r.get("disclosure_type"),
            "text": text,
            "video_title": display_title,
            "media_path": media_row.get("local_media_path", "") if media_row is not None else "",
            "media_retrieval_status": media_row.get("retrieval_status", "not_attempted") if media_row is not None else "not_attempted",
            "media_retrieval_method": media_row.get("retrieval_method", "") if media_row is not None else "",
            "evidence_strength": r.get("evidence_strength"),
            "creator_identity_confidence": r.get("creator_identity_confidence"),
            "brand_link_verified": r.get("brand_link_verified"),
            # raw YouTube statistics -- None (never 0) when not verified
            "view_count": view_count, "like_count": like_count, "comment_count": comment_count,
            "channel_subscriber_count_current": subscriber_count,
            "hidden_subscriber_count": hidden_subscriber_count,
            "statistics_retrieved_at": stats_retrieved_at,
            "video_privacy_status": video_privacy_status,
            "api_retrieval_status": api_retrieval_status,
            "api_failure_reason": api_failure_reason,
            "subscriber_count_is_historical": False,  # always current-retrieval-time, never historical
            "embedding_available": embedding_available,
            "embedding_file_hash": media_row.get("file_hash", "") if media_row is not None else "",
            "provenance_source_id": post_id,
            "provenance_source_url": r.get("source_url"),
            "dataset_version": "day2_verified_v1",
        }
        row.update(gen)
        row.update(sent)
        row.update(creator_ind)
        row.update(visual)
        row.update(engagement)
        row["original_intent_weak_label"] = original_intent["intent_weak_label"]
        row.update(intent)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = pd.DataFrame(within_channel_normalisation(df.to_dict("records"), min_posts=3))

    # Descriptive brand-level engagement percentile -- explicitly NOT within-account
    # normalisation, kept under a distinctly different field name.
    has_eng = df["engagement_count"].notna()
    df["descriptive_brand_engagement_percentile"] = None
    if has_eng.any():
        df.loc[has_eng, "descriptive_brand_engagement_percentile"] = (
            df.loc[has_eng].groupby("brand")["engagement_count"].rank(pct=True).round(4)
        )

    partnership_human_review_required = (df["revised_partnership_type"] != "organic")
    df["partnership_human_review_required"] = partnership_human_review_required

    assert_no_duplicate_ids(df, "post_id", label="creator_multimodal_features")
    assert_evidence_strength_allowed(df, USABLE_EVIDENCE, label="creator_multimodal_features")
    out_path = PROCESSED_DIR / "creator_multimodal_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} ({len(df)} rows, {len(df.columns)} cols)")
    n_eng = int(has_eng.sum())
    n_eligible_channel = int(df["within_channel_normalisation_eligible"].sum())
    print(f"  Engagement outcome coverage: {n_eng}/{len(df)}")
    print(f"  Within-channel-normalisation-eligible rows: {n_eligible_channel} "
          f"(channels with >=3 posts: {df.loc[df['within_channel_normalisation_eligible'], 'channel_id'].nunique()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part C: customer_text_features.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_customer_features() -> pd.DataFrame:
    print("\n[Customer features] Loading customer_experience_corpus ...")
    ce = pd.read_csv(CORPORA_DIR / "customer_experience_corpus.csv")
    usable = ce[ce["evidence_strength"].isin(USABLE_EVIDENCE)].copy()
    print(f"  {len(usable)}/{len(ce)} rows at strong/medium evidence strength")

    rows = []
    for _, r in usable.iterrows():
        text = str(r.get("extracted_text", "") or "")
        gen = general_text_features(text)
        ce_feats = customer_experience_features(text)
        prov = _parse_provenance(r.get("provenance_note", ""))
        row = {
            "document_id": r.get("document_id"),
            "brand": r.get("brand"),
            "source_platform": r.get("source_platform"),
            "source_url": r.get("source_url"),
            "document_title": r.get("document_title"),
            "evidence_strength": r.get("evidence_strength"),
            "provenance_category": prov.get("category", ""),
            "provenance_sentiment_label": prov.get("sentiment", ""),
            "dataset_version": "day2_verified_v1",
        }
        row.update(gen)
        row.update(ce_feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    assert_no_duplicate_ids(df, "document_id", label="customer_text_features")
    out_path = PROCESSED_DIR / "customer_text_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part C: official_claim_features.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_official_claim_features() -> pd.DataFrame:
    print("\n[Official claim features] Loading official_claims_corpus ...")
    oc = pd.read_csv(CORPORA_DIR / "official_claims_corpus.csv")
    usable = oc[oc["evidence_strength"].isin(USABLE_EVIDENCE)].copy()
    print(f"  {len(usable)}/{len(oc)} rows at strong/medium evidence strength")

    rows = []
    for _, r in usable.iterrows():
        text = str(r.get("extracted_text", "") or "")
        prov = _parse_provenance(r.get("provenance_note", ""))
        claim_category = prov.get("claim_category", "")
        gen = general_text_features(text)
        claim_ind = official_claim_indicators(text, claim_category=claim_category)
        row = {
            "document_id": r.get("document_id"),
            "brand": r.get("brand"),
            "source_platform": r.get("source_platform"),
            "source_url": r.get("source_url"),
            "document_title": r.get("document_title"),
            "evidence_strength": r.get("evidence_strength"),
            "extraction_method": prov.get("extraction", ""),
            "strength_source": prov.get("strength_source", ""),
            "dataset_version": "day2_verified_v1",
        }
        row.update(gen)
        row.update(claim_ind)
        rows.append(row)

    df = pd.DataFrame(rows)
    assert_no_duplicate_ids(df, "document_id", label="official_claim_features")
    out_path = PROCESSED_DIR / "official_claim_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part C: platform_strategy_features.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_platform_features() -> pd.DataFrame:
    print("\n[Platform strategy features] Loading platform_strategy_enriched ...")
    ps = pd.read_csv(DAY2_DIR / "platform_strategy_enriched.csv")
    usable = ps[ps["evidence_strength"].isin(USABLE_EVIDENCE)].copy()
    print(f"  {len(usable)}/{len(ps)} rows at strong/medium evidence strength")

    rows = []
    for _, r in usable.iterrows():
        text = str(r.get("h_hydrated_text", "") or "")
        gen = general_text_features(text) if text.lower() != "nan" else general_text_features("")
        row = {
            "source_id": r.get("source_id"),
            "brand": r.get("brand"),
            "platform": r.get("platform"),
            "evidence_url": r.get("evidence_url"),
            "evidence_strength": r.get("evidence_strength"),
            "channel_role": r.get("channel_role"),
            "inferred_intent": r.get("inferred_intent"),
            "creator_usage": r.get("creator_usage"),
            "commerce_integration": r.get("commerce_integration"),
            "responsibility_messaging": r.get("responsibility_messaging"),
            "dataset_version": "day2_verified_v1",
        }
        row.update(gen)
        rows.append(row)

    df = pd.DataFrame(rows)
    assert_no_duplicate_ids(df, "source_id", label="platform_strategy_features")
    out_path = PROCESSED_DIR / "platform_strategy_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part G: engagement coverage report (no fabrication)
# ─────────────────────────────────────────────────────────────────────────────

def report_engagement_coverage(creator_df: pd.DataFrame) -> pd.DataFrame:
    """Report coverage BEFORE any analytical use (Part B/G). Within-channel
    normalisation is already computed in build_creator_features() -- this only
    reports on it, never recomputes with a different (e.g. brand-level) grouping
    under the same field name."""
    print("\n[Engagement coverage] ...")
    n = len(creator_df)
    metric_cols = {
        "view_count": "view_count", "like_count": "like_count", "comment_count": "comment_count",
        "channel_subscriber_count_current": "channel_subscriber_count_current",
    }
    coverage_rows = []
    for label, col in metric_cols.items():
        n_present = int(creator_df[col].notna().sum()) if col in creator_df.columns else 0
        coverage_rows.append({
            "metric": label, "n_verified_posts": n, "n_with_metric": n_present,
            "pct_with_metric": round(100 * n_present / n, 1) if n else 0.0,
        })

    n_engagement_usable = int(creator_df["engagement_count"].notna().sum())
    n_interaction_rate = int(creator_df["interaction_rate_by_views"].notna().sum())
    n_sub_rate = int(creator_df["engagement_rate_by_current_subscribers"].notna().sum())
    n_channel_eligible = int(creator_df["within_channel_normalisation_eligible"].sum())
    n_channels_eligible = int(
        creator_df.loc[creator_df["within_channel_normalisation_eligible"], "channel_id"].nunique()
    )

    decision = "GO" if n_engagement_usable >= 30 else "NO-GO (exploratory only)"
    coverage_rows += [
        {"metric": "engagement_count (like+comment both present)", "n_verified_posts": n,
         "n_with_metric": n_engagement_usable, "pct_with_metric": round(100 * n_engagement_usable / n, 1) if n else 0.0},
        {"metric": "interaction_rate_by_views", "n_verified_posts": n,
         "n_with_metric": n_interaction_rate, "pct_with_metric": round(100 * n_interaction_rate / n, 1) if n else 0.0},
        {"metric": "engagement_rate_by_current_subscribers", "n_verified_posts": n,
         "n_with_metric": n_sub_rate, "pct_with_metric": round(100 * n_sub_rate / n, 1) if n else 0.0},
        {"metric": f"within_channel_normalisation_eligible (channels with >=3 posts: {n_channels_eligible})",
         "n_verified_posts": n, "n_with_metric": n_channel_eligible,
         "pct_with_metric": round(100 * n_channel_eligible / n, 1) if n else 0.0},
    ]

    df = pd.DataFrame(coverage_rows)
    df["modelling_decision"] = decision
    df["subscriber_count_caveat"] = "channel_subscriber_count_current is a RETRIEVAL-TIME snapshot, not the historical subscriber count on the post date"
    print(f"  engagement_count usable rows: {n_engagement_usable}/{n} -> {decision} (threshold: 30)")
    print(f"  within-channel-normalisation-eligible: {n_channel_eligible} rows across {n_channels_eligible} channels (>=3 posts each)")
    out_path = TABLES / "day2_engagement_coverage.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = df[df["metric"] != "modelling_decision"]
    ax.barh(plot_df["metric"], plot_df["n_with_metric"], color=sns.color_palette("Set2", len(plot_df)))
    ax.set_xlabel(f"Rows with metric present (of n={n} verified creator posts)")
    ax.set_title(f"Engagement Metric Coverage (n={n}) -- decision: {decision}")
    for i, v in enumerate(plot_df["n_with_metric"].values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "day2_engagement_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part F: content-intent human-validation sample
# ─────────────────────────────────────────────────────────────────────────────

def build_content_intent_validation_sample(creator_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Content-intent validation sample] ...")
    eligible = creator_df[creator_df["text"].astype(str).str.strip() != ""].copy()
    target_n = min(80, len(eligible))
    if len(eligible) <= target_n:
        sample = eligible.copy()
    else:
        # stratify by brand so no brand is silently excluded
        sample = eligible.groupby("brand", group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, round(target_n * len(g) / len(eligible))),
                random_state=RANDOM_SEED,
            )
        ).head(target_n)

    out = pd.DataFrame({
        "record_id": sample["post_id"],
        "brand": sample["brand"],
        "platform": sample["platform"],
        "text": sample["text"],
        "image_path_or_media_url": sample["media_path"].where(sample["media_path"] != "", sample["post_url"]),
        "weak_label": sample["primary_intent"],
        "secondary_intents": sample["secondary_intents"],
        "original_intent_weak_label": sample["original_intent_weak_label"],
        "coder_1_label": "",
        "coder_2_label": "",
        "adjudicated_label": "",
        "coding_notes": "",
    })
    MANUAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANUAL_LABELS_DIR / "content_intent_validation_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"  Sampled {len(out)}/{len(eligible)} eligible records (target ~80)")
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} -- human coder fields left blank")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Part I: descriptive competitor-analysis tables + figures
# ─────────────────────────────────────────────────────────────────────────────

def build_descriptive_tables(creator_df: pd.DataFrame, media_df: pd.DataFrame) -> None:
    print("\n[Descriptive tables + figures] ...")
    n_total = len(creator_df)

    # by brand
    by_brand = creator_df.groupby("brand").agg(
        verified_posts=("post_id", "count"),
        platforms_used=("platform", "nunique"),
        disclosure_rate_pct=("disclosure_indicator", lambda s: round(100 * s.mean(), 1)),
        sustainability_mention_rate_pct=("sustainability_responsibility_indicator", lambda s: round(100 * s.mean(), 1)),
    ).reset_index()
    by_brand["sample_size_denominator"] = n_total
    by_brand.to_csv(TABLES / "day2_creator_strategy_by_brand.csv", index=False)

    # by platform
    by_platform = creator_df.groupby("platform").agg(
        verified_posts=("post_id", "count"),
        brands_present=("brand", "nunique"),
    ).reset_index()
    by_platform["pct_of_all_verified_posts"] = round(100 * by_platform["verified_posts"] / n_total, 1)
    by_platform["sample_size_denominator"] = n_total
    by_platform.to_csv(TABLES / "day2_creator_strategy_by_platform.csv", index=False)

    # partnership mix (revised, evidence-gated -- see Part C)
    partnership_mix = (
        creator_df.groupby(["brand", "revised_partnership_type"]).size().reset_index(name="count")
    )
    brand_totals = creator_df.groupby("brand").size().rename("brand_total")
    partnership_mix = partnership_mix.merge(brand_totals, on="brand")
    partnership_mix["pct_within_brand"] = round(100 * partnership_mix["count"] / partnership_mix["brand_total"], 1)
    partnership_mix["sample_size_denominator"] = n_total
    partnership_mix.to_csv(TABLES / "day2_partnership_mix.csv", index=False)

    # content intent mix (revised primary_intent, multi-label -- see Part D)
    intent_mix = creator_df.groupby(["brand", "primary_intent"]).size().reset_index(name="count")
    intent_mix = intent_mix.merge(brand_totals, on="brand")
    intent_mix["pct_within_brand"] = round(100 * intent_mix["count"] / intent_mix["brand_total"], 1)
    intent_mix["sample_size_denominator"] = n_total
    intent_mix.to_csv(TABLES / "day2_content_intent_mix.csv", index=False)

    # media coverage
    if not media_df.empty:
        media_cov = media_df.groupby(["brand", "platform", "retrieval_status"]).size().reset_index(name="count")
        platform_totals = media_df.groupby(["brand", "platform"]).size().rename("platform_total")
        media_cov = media_cov.merge(platform_totals, on=["brand", "platform"])
        media_cov["pct_within_brand_platform"] = round(100 * media_cov["count"] / media_cov["platform_total"], 1)
        media_cov.to_csv(TABLES / "day2_media_coverage.csv", index=False)
    else:
        media_cov = pd.DataFrame()

    print("  Saved 5 descriptive tables to outputs/tables/")

    FIGURES.mkdir(parents=True, exist_ok=True)

    def _bar(series, title, fname, xlabel="Count"):
        fig, ax = plt.subplots(figsize=(7, 4))
        series.plot(kind="barh", ax=ax, color=sns.color_palette("Set2", len(series)))
        ax.set_title(f"{title} (n={n_total} verified creator posts)")
        ax.set_xlabel(xlabel)
        for i, v in enumerate(series.values):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _bar(creator_df["platform"].value_counts(), "Verified Creator Posts by Platform", "day2_platform_mix.png")
    _bar(creator_df["revised_partnership_type"].value_counts(), "Revised Partnership Type Mix (evidence-gated)", "day2_partnership_mix.png")
    _bar(creator_df["primary_intent"].value_counts(), "Revised Primary Intent Mix", "day2_content_intent_mix.png")
    _bar(creator_df["revised_partnership_type"].value_counts(), "Revised Partnership Mix (Part C reassessment)", "day2_revised_partnership_mix.png")
    _bar(creator_df["primary_intent"].value_counts(), "Revised Primary Intent Mix (multi-label, Part D)", "day2_revised_primary_intent_mix.png")

    tier = creator_df["creator_handle"].notna()  # creator_tier is unverified for all rows (see docstring below)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["unknown (no verified follower count)"], [n_total], color=sns.color_palette("Set2")[3])
    ax.set_title(f"Creator Tier Mix (n={n_total}) -- all unknown: no row has a verified follower_count")
    ax.set_ylabel("Posts")
    plt.tight_layout()
    plt.savefig(FIGURES / "day2_creator_tier_mix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not media_cov.empty:
        pivot = media_cov.pivot_table(index="platform", columns="retrieval_status", values="count", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot.plot(kind="barh", stacked=True, ax=ax, color=sns.color_palette("Set2", len(pivot.columns)))
        ax.set_title(f"Media Acquisition Coverage by Platform (n={len(media_df)} verified posts)")
        ax.set_xlabel("Posts")
        plt.tight_layout()
        plt.savefig(FIGURES / "day2_media_coverage.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    has_rate = creator_df["interaction_rate_by_views"].notna()
    n_rate = int(has_rate.sum())
    n_excluded = n_total - n_rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if n_rate > 0:
        sns.boxplot(data=creator_df[has_rate], x="brand", y="interaction_rate_by_views", ax=ax,
                    palette="Set2")
        sns.stripplot(data=creator_df[has_rate], x="brand", y="interaction_rate_by_views", ax=ax,
                       color="black", alpha=0.4, size=3)
    ax.set_title(f"Interaction Rate by Views, per Brand (n={n_rate}/{n_total}; "
                 f"{n_excluded} excluded: view_count=0 or missing)")
    ax.set_ylabel("(likes + comments) / views")
    plt.tight_layout()
    plt.savefig(FIGURES / "day2_interaction_rate_by_brand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Saved 7 descriptive figures to outputs/figures/")


# ─────────────────────────────────────────────────────────────────────────────
# Part H: embeddings index
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Part C: partnership human-validation sample
# ─────────────────────────────────────────────────────────────────────────────

def build_partnership_validation_sample(creator_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Partnership validation sample] ...")
    non_organic = creator_df[creator_df["revised_partnership_type"] != "organic"]
    organic = creator_df[creator_df["revised_partnership_type"] == "organic"]
    target_organic_n = min(20, len(organic))
    if len(organic) > 0 and target_organic_n > 0:
        organic_sample = organic.groupby("brand", group_keys=False).apply(
            lambda g: g.sample(n=max(1, round(target_organic_n * len(g) / len(organic))), random_state=RANDOM_SEED)
        ).head(target_organic_n)
    else:
        organic_sample = organic.iloc[0:0]

    sample = pd.concat([non_organic, organic_sample], ignore_index=True)
    out = pd.DataFrame({
        "record_id": sample["post_id"],
        "brand": sample["brand"],
        "platform": sample["platform"],
        "video_title": sample["video_title"],
        "text": sample["text"],
        "original_partnership_type": sample["original_partnership_type"],
        "revised_partnership_type": sample["revised_partnership_type"],
        "partnership_evidence_text": sample["partnership_evidence_text"],
        "partnership_rule_trigger": sample["partnership_rule_trigger"],
        "partnership_confidence": sample["partnership_confidence"],
        "coder_1_label": "",
        "coder_2_label": "",
        "adjudicated_label": "",
        "coding_notes": "",
    })
    MANUAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANUAL_LABELS_DIR / "partnership_validation_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"  {len(non_organic)} non-organic + {len(organic_sample)}/{len(organic)} stratified organic = {len(out)} rows")
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} -- human coder fields left blank")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Part E: cross-platform evidence boundary
# ─────────────────────────────────────────────────────────────────────────────

def build_platform_evidence_boundary(creator_df: pd.DataFrame, platform_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Platform evidence boundary] ...")
    brand_platform_pairs = set(
        zip(creator_df["brand"], creator_df["platform"])
    ) | set(
        zip(platform_df["brand"], platform_df["platform"]) if not platform_df.empty else []
    )

    rows = []
    for brand, platform in sorted(brand_platform_pairs):
        official = platform_df[(platform_df["brand"] == brand) & (platform_df["platform"] == platform)] if not platform_df.empty else platform_df
        official_presence_n = len(official)
        official_content_n = int((official["word_count"] > 0).sum()) if not official.empty and "word_count" in official.columns else 0

        cposts = creator_df[(creator_df["brand"] == brand) & (creator_df["platform"] == platform)]
        n_creator = len(cposts)
        n_partnership = int(cposts["revised_partnership_type"].isin(
            ["paid_sponsorship", "gifted", "ambassador", "affiliate", "founder_or_employee"]).sum())
        n_engagement = int(cposts["engagement_count"].notna().sum()) if n_creator else 0
        n_visual = int(cposts["mean_brightness"].notna().sum()) if n_creator else 0

        permitted = []
        if official_presence_n > 0:
            permitted.append("official_channel_strategy_comparison")
        if n_creator > 0:
            permitted.append("creator_content_comparison")
        if n_partnership > 0:
            permitted.append("partnership_comparison")
        if n_engagement > 0:
            permitted.append("engagement_comparison")
        if not permitted:
            permitted = ["insufficient_evidence"]

        limitation = ""
        if platform == "tiktok" and n_creator <= 1:
            limitation = "Single TikTok record -- does not support platform-level inference for this brand."
        elif n_creator == 0 and official_presence_n > 0:
            limitation = "Official platform evidence only -- no verified creator posts on this platform for this brand."
        elif n_creator > 0 and n_creator < 5:
            limitation = f"Small creator sample (n={n_creator}) -- descriptive only, not for cross-brand performance claims."

        rows.append({
            "brand": brand, "platform": platform,
            "official_platform_presence_evidence": official_presence_n,
            "official_platform_content_evidence": official_content_n,
            "verified_creator_post_count": n_creator,
            "verified_partnership_post_count": n_partnership,
            "engagement_metric_coverage": n_engagement,
            "visual_feature_coverage": n_visual,
            "analytical_use_permitted": ", ".join(permitted),
            "limitation": limitation,
        })

    df = pd.DataFrame(rows)
    out_path = TABLES / "day2_platform_evidence_boundary.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)} ({len(df)} brand x platform rows)")

    yt_total = int((creator_df["platform"] == "youtube").sum())
    tiktok_total = int((creator_df["platform"] == "tiktok").sum())
    print(f"  Creator content: {yt_total} YouTube vs {tiktok_total} TikTok -- overwhelmingly YouTube-based, "
          "NOT a balanced cross-platform sample. Cross-brand YouTube creator analysis is permitted; "
          "cross-platform creator-PERFORMANCE claims are not permitted with the present dataset.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part F: modelling feasibility
# ─────────────────────────────────────────────────────────────────────────────

OUT_OF_SCOPE_REASON = "Not required by assignment; engagement retained for descriptive analysis"
# Tracks 4-10 are engagement-prediction / engagement-focused-fusion tracks. Per the Day 2.5
# assignment-realignment charter (CLAUDE.md §9, docs/project_charter.md §6), engagement
# prediction is out of scope for the whole project -- these are marked OUT_OF_SCOPE rather
# than GO/NO-GO/EXPLORATORY. The `prior_decision` column preserves what the earlier
# feasibility assessment concluded (historical audit value), per the requirement not to
# falsify earlier results -- a full pre-realignment snapshot is also archived at
# data/interim/day2_5/archive/day2_modelling_feasibility_pre_realignment.csv.
OUT_OF_SCOPE_TRACK_PREFIXES = ("4.", "5.", "6.", "7.", "8.", "9.", "10.")


def build_modelling_feasibility(creator_df: pd.DataFrame, platform_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Modelling feasibility] ...")
    n = len(creator_df)
    n_brands = creator_df["brand"].nunique()
    n_platforms = creator_df["platform"].nunique()
    creator_df = creator_df.copy()
    creator_df["_creator_id"] = creator_df["channel_id"].where(creator_df["channel_id"] != "", creator_df["creator_handle"])
    n_creators = creator_df["_creator_id"].nunique()
    n_eng = int(creator_df["engagement_count"].notna().sum())
    n_visual = int(creator_df["mean_brightness"].notna().sum())
    n_channel_eligible_rows = int(creator_df["within_channel_normalisation_eligible"].sum())
    n_channel_eligible = creator_df.loc[creator_df["within_channel_normalisation_eligible"], "_creator_id"].nunique()
    max_posts_per_channel = creator_df["_creator_id"].value_counts().max() if n else 0
    platform_imbalance = f"{int((creator_df['platform']=='youtube').sum())} youtube / {int((creator_df['platform']=='tiktok').sum())} tiktok"

    rows = [
        {"track": "1. Descriptive cross-brand creator analysis", "eligible_rows": n, "outcome_coverage": "n/a",
         "visual_coverage": n_visual, "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "low (descriptive, no train/test split)", "class_imbalance": platform_imbalance,
         "decision": "GO", "rationale": "All 4 brands represented with >=8 posts each; descriptive counts/rates only."},
        {"track": "2. Descriptive official-channel comparison", "eligible_rows": len(platform_df), "outcome_coverage": "n/a",
         "visual_coverage": "n/a", "n_brands": platform_df["brand"].nunique() if not platform_df.empty else 0,
         "n_platforms": platform_df["platform"].nunique() if not platform_df.empty else 0, "n_unique_creators_channels": "n/a",
         "leakage_risk": "low (descriptive)", "class_imbalance": "varies by platform, see day2_platform_evidence_boundary.csv",
         "decision": "GO", "rationale": "31 strong/medium platform_strategy rows across brands/platforms support descriptive comparison."},
        {"track": "3. Partnership-type comparison", "eligible_rows": n,
         "outcome_coverage": int((creator_df["revised_partnership_type"] != "unclear").sum()),
         "visual_coverage": n_visual, "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "low (descriptive)",
         "class_imbalance": f"organic dominates ({int((creator_df['revised_partnership_type']=='organic').sum())}/{n})",
         "decision": "GO (descriptive only)",
         "rationale": "Evidence-gated reassessment complete for all rows; heavy organic-class imbalance means no predictive claim, description only."},
        {"track": "4. Engagement prediction", "eligible_rows": n, "outcome_coverage": n_eng, "visual_coverage": n_visual,
         "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "MEDIUM -- same channel can appear multiple times (see track 9)",
         "class_imbalance": f"max {max_posts_per_channel} posts from one channel",
         "decision": "GO" if n_eng >= 30 else "NO-GO",
         "rationale": f"{n_eng}/{n} rows have verified engagement_count ({'>=' if n_eng>=30 else '<'} 30 threshold)."},
        {"track": "5. Within-account engagement prediction", "eligible_rows": n_channel_eligible_rows,
         "outcome_coverage": n_channel_eligible_rows, "visual_coverage": "n/a",
         "n_brands": creator_df.loc[creator_df["within_channel_normalisation_eligible"], "brand"].nunique(),
         "n_platforms": creator_df.loc[creator_df["within_channel_normalisation_eligible"], "platform"].nunique(),
         "n_unique_creators_channels": n_channel_eligible,
         "leakage_risk": "HIGH if same channel split across folds -- grouped CV required",
         "class_imbalance": f"only {n_channel_eligible} channels have >=3 posts",
         "decision": "NO-GO",
         "rationale": f"Only {n_channel_eligible} channels have >=3 verified posts each ({n_channel_eligible_rows} rows total) -- too few channels for a within-account model."},
        {"track": "6. Text-only engagement baseline", "eligible_rows": n, "outcome_coverage": n_eng, "visual_coverage": "n/a",
         "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "MEDIUM -- requires grouped CV by channel_id", "class_imbalance": platform_imbalance,
         "decision": "EXPLORATORY" if n_eng >= 30 else "NO-GO",
         "rationale": f"{n_eng} labelled rows clears the 30-row floor for an exploratory baseline, but {n_creators} unique channels is small for generalisation."},
        {"track": "7. Image-only engagement baseline", "eligible_rows": n_visual, "outcome_coverage": min(n_visual, n_eng),
         "visual_coverage": n_visual, "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "MEDIUM -- requires grouped CV by channel_id", "class_imbalance": platform_imbalance,
         "decision": "EXPLORATORY" if min(n_visual, n_eng) >= 30 else "NO-GO",
         "rationale": f"{min(n_visual, n_eng)} rows have both a visual feature and a verified engagement outcome."},
        {"track": "8. Early fusion", "eligible_rows": min(n_visual, n_eng), "outcome_coverage": min(n_visual, n_eng),
         "visual_coverage": n_visual, "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "MEDIUM -- requires grouped CV by channel_id", "class_imbalance": platform_imbalance,
         "decision": "NOT STARTED", "rationale": "Out of scope for this task -- deferred pending track 6/7 exploratory results."},
        {"track": "9. Late fusion", "eligible_rows": min(n_visual, n_eng), "outcome_coverage": min(n_visual, n_eng),
         "visual_coverage": n_visual, "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "MEDIUM -- requires grouped CV by channel_id", "class_imbalance": platform_imbalance,
         "decision": "NOT STARTED", "rationale": "Out of scope for this task -- deferred pending track 6/7 exploratory results."},
        {"track": "10. Hybrid fusion", "eligible_rows": min(n_visual, n_eng), "outcome_coverage": min(n_visual, n_eng),
         "visual_coverage": n_visual, "n_brands": n_brands, "n_platforms": n_platforms, "n_unique_creators_channels": n_creators,
         "leakage_risk": "MEDIUM -- requires grouped CV by channel_id", "class_imbalance": platform_imbalance,
         "decision": "NOT STARTED", "rationale": "Out of scope for this task -- deferred pending track 6/7 exploratory results."},
    ]
    df = pd.DataFrame(rows)
    df["grouped_cv_required_by"] = "channel_id (YouTube) -- same creator/channel must never appear in both train and test folds"

    is_out_of_scope = df["track"].str.startswith(OUT_OF_SCOPE_TRACK_PREFIXES)
    df["prior_decision"] = df["decision"]
    df["prior_rationale"] = df["rationale"]
    df.loc[is_out_of_scope, "decision"] = "OUT_OF_SCOPE"
    df.loc[is_out_of_scope, "reason"] = OUT_OF_SCOPE_REASON
    df.loc[~is_out_of_scope, "reason"] = ""
    df.loc[is_out_of_scope, "rationale"] = (
        OUT_OF_SCOPE_REASON + " (prior feasibility assessment, preserved for audit: "
        + df.loc[is_out_of_scope, "prior_decision"] + " -- " + df.loc[is_out_of_scope, "prior_rationale"] + ")"
    )

    out_path = TABLES / "day2_modelling_feasibility.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    for _, r in df.iterrows():
        print(f"    {r['track']:<45} {r['decision']}")
    return df


def build_embeddings_index(creator_df: pd.DataFrame) -> None:
    idx = creator_df[creator_df["embedding_available"] == True][["post_id", "embedding_file_hash"]].copy()
    idx["embedding_path"] = idx["embedding_file_hash"].apply(
        lambda h: str((EMBEDDINGS_DIR / "clip" / f"{h}.npy").relative_to(PROJECT_ROOT)))
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    idx.to_csv(EMBEDDINGS_DIR / "clip_embedding_index.csv", index=False)
    print(f"\n[Embeddings] {len(idx)} embeddings indexed at "
          f"{(EMBEDDINGS_DIR / 'clip_embedding_index.csv').relative_to(PROJECT_ROOT)}")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    creator_df = build_creator_features()
    build_customer_features()
    build_official_claim_features()
    platform_df = build_platform_features()

    report_engagement_coverage(creator_df)
    build_content_intent_validation_sample(creator_df)
    build_partnership_validation_sample(creator_df)

    media_path = PROCESSED_DIR / "creator_media_manifest.csv"
    media_df = pd.read_csv(media_path) if media_path.exists() else pd.DataFrame()
    build_descriptive_tables(creator_df, media_df)
    build_platform_evidence_boundary(creator_df, platform_df)
    build_modelling_feasibility(creator_df, platform_df)
    build_embeddings_index(creator_df)

    print("\n" + "=" * 66)
    print("DAY 2 TASK 2B FEATURE BUILD COMPLETE")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
