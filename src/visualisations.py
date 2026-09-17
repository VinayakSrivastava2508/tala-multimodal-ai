"""Plot generation utilities. All figures saved to outputs/figures/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = sns.color_palette("Set2")


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to outputs/figures/{filename} and return the path."""
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[vis] Saved → {out_path}")
    return out_path


def plot_rating_distribution(
    reviews_df: pd.DataFrame,
    brand: str,
    save: bool = True,
) -> plt.Figure:
    """Bar chart of star-rating distribution for a brand's reviews.

    Returns the matplotlib Figure object.
    """
    subset = reviews_df[reviews_df["brand"] == brand]
    fig, ax = plt.subplots(figsize=(7, 4))
    subset["star_rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color=PALETTE[0])
    ax.set_title(f"Star Rating Distribution — {brand}", fontsize=13)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    if save:
        save_figure(fig, f"rating_distribution_{brand.lower()}.png")
    return fig


def plot_sentiment_comparison(
    reviews_df: pd.DataFrame,
    brands: List[str],
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart comparing sentiment label distributions across brands.

    Returns the matplotlib Figure object.
    """
    records = []
    for brand in brands:
        subset = reviews_df[reviews_df["brand"] == brand]
        counts = subset["sentiment_label"].value_counts(normalize=True)
        for label, pct in counts.items():
            records.append({"brand": brand, "sentiment": label, "proportion": pct})

    df_plot = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_plot, x="sentiment", y="proportion", hue="brand", ax=ax, palette=PALETTE)
    ax.set_title("Sentiment Distribution by Brand", fontsize=13)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Proportion of Reviews")
    ax.legend(title="Brand")
    if save:
        save_figure(fig, "sentiment_comparison.png")
    return fig


def plot_creator_tier_heatmap(
    creator_posts_df: pd.DataFrame,
    brands: List[str],
    save: bool = True,
) -> plt.Figure:
    """Heatmap of creator tier × brand normalised frequency.

    Returns the matplotlib Figure object.
    """
    tier_order = ["nano", "micro", "mid", "macro", "mega"]
    records = {}
    for brand in brands:
        subset = creator_posts_df[creator_posts_df["brand_mentioned"] == brand]
        counts = subset["creator_tier"].value_counts(normalize=True)
        records[brand] = {t: counts.get(t, 0.0) for t in tier_order}

    heat_df = pd.DataFrame(records, index=tier_order).T
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heat_df, annot=True, fmt=".0%", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Creator Tier Mix by Brand", fontsize=13)
    ax.set_xlabel("Creator Tier")
    ax.set_ylabel("Brand")
    if save:
        save_figure(fig, "creator_tier_heatmap.png")
    return fig


def plot_divergence_scores(
    claim_assessments_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Box plot of divergence scores by brand and claim category.

    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=claim_assessments_df,
        x="claim_category",
        y="divergence_score",
        hue="brand",
        ax=ax,
        palette=PALETTE,
    )
    ax.set_title("Claim–Experience Divergence Scores by Category and Brand", fontsize=13)
    ax.set_xlabel("Claim Category")
    ax.set_ylabel("Divergence Score (0=supported, 1=contradicted)")
    ax.tick_params(axis="x", rotation=30)
    if save:
        save_figure(fig, "divergence_scores.png")
    return fig


def plot_embedding_scatter(
    embeddings_2d: "np.ndarray",
    labels: List[str],
    title: str = "Embedding Space",
    color_by: Optional[List[str]] = None,
    save: bool = True,
    filename: str = "embedding_scatter.png",
) -> plt.Figure:
    """2D scatter plot of reduced embeddings, coloured by label.

    embeddings_2d: ndarray of shape (n, 2) from UMAP or PCA.
    Returns the matplotlib Figure object.
    """
    import numpy as np  # lazy import
    fig, ax = plt.subplots(figsize=(9, 7))
    if color_by is not None:
        unique_labels = sorted(set(color_by))
        color_map = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique_labels)}
        colors = [color_map[lbl] for lbl in color_by]
        for lbl in unique_labels:
            mask = [c == lbl for c in color_by]
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=lbl, s=20, alpha=0.7)
        ax.legend(title="Label", markerscale=2)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=20, alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if save:
        save_figure(fig, filename)
    return fig
