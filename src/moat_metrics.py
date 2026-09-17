"""Brand moat and creator strategy metrics for cross-brand benchmarking."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_creator_tier_distribution(
    creator_posts_df: pd.DataFrame,
    brand: str,
) -> pd.Series:
    """Return the proportion of creator posts by tier for a given brand.

    Expects creator_posts_df to have columns: brand, creator_tier.
    Returns a Series indexed by tier names with fractional values summing to 1.
    """
    subset = creator_posts_df[creator_posts_df["brand"] == brand]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset["creator_tier"].value_counts(normalize=True)


def compute_platform_diversity_score(
    creator_posts_df: pd.DataFrame,
    brand: str,
) -> float:
    """Compute normalised platform diversity (Shannon entropy / log(n_platforms)).

    Returns a float in [0, 1] where 1 = perfectly even distribution across platforms.
    """
    subset = creator_posts_df[creator_posts_df["brand"] == brand]
    if subset.empty or "platform" not in subset.columns:
        return 0.0
    counts = subset["platform"].value_counts(normalize=True)
    entropy = -np.sum(counts * np.log(counts + 1e-9))
    max_entropy = np.log(len(counts) + 1e-9)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_avg_engagement_rate(
    creator_posts_df: pd.DataFrame,
    brand: str,
) -> float:
    """Compute mean engagement rate across creator posts for a brand.

    Engagement rate = (likes + comments) / followers.
    Returns mean across rows where followers > 0, or NaN if insufficient data.
    """
    subset = creator_posts_df[creator_posts_df["brand"] == brand].copy()
    if subset.empty:
        return float("nan")
    mask = subset["creator_followers"] > 0
    subset = subset[mask]
    if subset.empty:
        return float("nan")
    subset["engagement_rate"] = (
        subset.get("likes", 0).fillna(0) + subset.get("comments", 0).fillna(0)
    ) / subset["creator_followers"]
    return float(subset["engagement_rate"].mean())


def compute_sentiment_distribution(
    reviews_df: pd.DataFrame,
    brand: str,
) -> pd.Series:
    """Return proportion of reviews by sentiment label for a given brand.

    Returns a Series indexed by sentiment label (positive / neutral / negative / mixed).
    """
    subset = reviews_df[reviews_df["brand"] == brand]
    if subset.empty or "sentiment_label" not in subset.columns:
        return pd.Series(dtype=float)
    return subset["sentiment_label"].value_counts(normalize=True)


def compute_claim_divergence_score(
    claim_assessments_df: pd.DataFrame,
    brand: str,
    category: Optional[str] = None,
) -> float:
    """Return mean divergence score for a brand (and optional category).

    divergence_score in [0, 1]: 0 = fully supported, 1 = fully contradicted.
    Returns NaN if no assessments available.
    """
    subset = claim_assessments_df[claim_assessments_df["brand"] == brand]
    if category:
        subset = subset[subset["claim_category"] == category]
    if subset.empty or "divergence_score" not in subset.columns:
        return float("nan")
    return float(subset["divergence_score"].mean())


def build_brand_scorecard(
    brand: str,
    reviews_df: pd.DataFrame,
    creator_posts_df: pd.DataFrame,
    claim_assessments_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Aggregate key moat metrics for a brand into a single scorecard dict.

    Returns a dict with keys: avg_star_rating, pct_positive_sentiment,
    platform_diversity, avg_engagement_rate, mean_divergence_score.
    """
    scorecard: Dict[str, float] = {}

    brand_reviews = reviews_df[reviews_df["brand"] == brand]
    if not brand_reviews.empty and "star_rating" in brand_reviews.columns:
        scorecard["avg_star_rating"] = float(brand_reviews["star_rating"].mean())
    else:
        scorecard["avg_star_rating"] = float("nan")

    sentiment = compute_sentiment_distribution(reviews_df, brand)
    scorecard["pct_positive_sentiment"] = float(sentiment.get("positive", 0.0))

    scorecard["platform_diversity"] = compute_platform_diversity_score(creator_posts_df, brand)
    scorecard["avg_engagement_rate"] = compute_avg_engagement_rate(creator_posts_df, brand)

    if claim_assessments_df is not None:
        scorecard["mean_divergence_score"] = compute_claim_divergence_score(claim_assessments_df, brand)
    else:
        scorecard["mean_divergence_score"] = float("nan")

    return scorecard


def compare_brands_scorecard(
    brands: List[str],
    reviews_df: pd.DataFrame,
    creator_posts_df: pd.DataFrame,
    claim_assessments_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build scorecards for multiple brands and return as a comparison DataFrame.

    Returns a DataFrame indexed by brand with metric columns.
    """
    rows = {}
    for brand in brands:
        rows[brand] = build_brand_scorecard(brand, reviews_df, creator_posts_df, claim_assessments_df)
    return pd.DataFrame(rows).T
