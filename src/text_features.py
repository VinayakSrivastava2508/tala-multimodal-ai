"""Text feature extraction: embeddings, TF-IDF, sentiment, and topic modelling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42


def get_sentence_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of strings with a Sentence Transformers model.

    Returns an ndarray of shape (n_texts, embedding_dim).
    Downloads model from HuggingFace on first use (free, no API key needed).
    """
    from sentence_transformers import SentenceTransformer  # lazy import
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


def get_tfidf_matrix(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """Fit TF-IDF on texts and return (sparse_matrix, vectorizer).

    Returns (tfidf_matrix, fitted TfidfVectorizer) for use in downstream models.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def get_sentiment_scores(
    texts: List[str],
    method: str = "textblob",
) -> pd.DataFrame:
    """Compute sentiment polarity and subjectivity for a list of texts.

    method: 'textblob' (default, free) or 'vader' (requires nltk vader_lexicon).
    Returns a DataFrame with columns: text, polarity, subjectivity, label.
    """
    if method == "textblob":
        from textblob import TextBlob  # lazy import
        records = []
        for text in texts:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            label = "positive" if polarity > 0.05 else ("negative" if polarity < -0.05 else "neutral")
            records.append({"text": text, "polarity": polarity, "subjectivity": subjectivity, "label": label})
        return pd.DataFrame(records)

    elif method == "vader":
        import nltk  # lazy import
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        records = []
        for text in texts:
            scores = sia.polarity_scores(str(text))
            label = "positive" if scores["compound"] >= 0.05 else ("negative" if scores["compound"] <= -0.05 else "neutral")
            records.append({"text": text, **scores, "label": label})
        return pd.DataFrame(records)

    else:
        raise ValueError(f"Unknown sentiment method: {method}. Use 'textblob' or 'vader'.")


def get_topic_keywords(
    texts: List[str],
    n_topics: int = 5,
    n_top_words: int = 10,
) -> list[dict]:
    """Run NMF topic modelling on texts and return top keywords per topic.

    Returns a list of dicts: [{'topic': 0, 'keywords': [...]}, ...].
    """
    from sklearn.decomposition import NMF  # lazy import
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    model = NMF(n_components=n_topics, random_state=RANDOM_SEED, max_iter=200)
    model.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for idx, component in enumerate(model.components_):
        top_indices = component.argsort()[: -n_top_words - 1 : -1]
        keywords = [feature_names[i] for i in top_indices]
        topics.append({"topic": idx, "keywords": keywords})
    return topics


def compute_semantic_similarity(
    query_texts: List[str],
    corpus_texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Compute cosine similarity between query embeddings and corpus embeddings.

    Returns ndarray of shape (n_queries, n_corpus).
    """
    from sentence_transformers import SentenceTransformer, util  # lazy import
    model = SentenceTransformer(model_name)
    query_emb = model.encode(query_texts, normalize_embeddings=True)
    corpus_emb = model.encode(corpus_texts, normalize_embeddings=True)
    return util.cos_sim(query_emb, corpus_emb).numpy()
