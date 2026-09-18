"""Text feature extraction: embeddings, TF-IDF, sentiment, topic modelling, and
deterministic rule-based indicators (keywords defined in configs/feature_rules.yaml)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42

_FEATURE_RULES_CACHE: Optional[dict] = None

_EMOJI_RX = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_URL_RX = re.compile(r"https?://\S+")
_HASHTAG_RX = re.compile(r"#\w+")
_MENTION_RX = re.compile(r"@\w+")
_SENTENCE_SPLIT_RX = re.compile(r"[.!?]+")
_PERCENT_RX = re.compile(r"\d+(\.\d+)?\s*%|\bpercent\b", re.I)
_NUMERICAL_RX = re.compile(r"\b\d+(\.\d+)?\b")
_YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")


def load_feature_rules() -> dict:
    """Return parsed configs/feature_rules.yaml as a dict (cached after first load)."""
    global _FEATURE_RULES_CACHE
    if _FEATURE_RULES_CACHE is None:
        with open(PROJECT_ROOT / "configs" / "feature_rules.yaml", "r", encoding="utf-8") as f:
            _FEATURE_RULES_CACHE = yaml.safe_load(f)
    return _FEATURE_RULES_CACHE


def _nz(text) -> str:
    """Null-safe string coercion for feature functions."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    return "" if s.lower() == "nan" else s


def _keyword_hit(text_l: str, keywords: List[str]) -> bool:
    return any(kw.lower() in text_l for kw in keywords)


def _keyword_count(text_l: str, keywords: List[str]) -> int:
    return sum(text_l.count(kw.lower()) for kw in keywords)


# ── General text features (Part C) ────────────────────────────────────────────

def general_text_features(text) -> Dict:
    """Compute interpretable, deterministic general text statistics for one string.

    Returns character_count, word_count, sentence_count, hashtag_count,
    mention_count, emoji_count, url_count, question_mark_count,
    exclamation_mark_count, lexical_diversity.
    """
    t = _nz(text)
    words = t.split()
    unique_words = {w.lower().strip(".,!?\"'") for w in words}
    sentences = [s for s in _SENTENCE_SPLIT_RX.split(t) if s.strip()]
    return {
        "character_count": len(t),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "hashtag_count": len(_HASHTAG_RX.findall(t)),
        "mention_count": len(_MENTION_RX.findall(t)),
        "emoji_count": len(_EMOJI_RX.findall(t)),
        "url_count": len(_URL_RX.findall(t)),
        "question_mark_count": t.count("?"),
        "exclamation_mark_count": t.count("!"),
        "lexical_diversity": round(len(unique_words) / len(words), 4) if words else 0.0,
    }


def general_text_features_batch(texts: List[str]) -> pd.DataFrame:
    """Vectorised general_text_features over a list of strings."""
    return pd.DataFrame([general_text_features(t) for t in texts])


def sentiment_features(text) -> Dict:
    """Return sentiment_score (polarity, -1..1) and subjectivity_score (0..1) via
    TextBlob. subjectivity_score is TextBlob's own estimate -- documented as
    approximate, not a validated psychometric measure."""
    from textblob import TextBlob  # lazy import
    t = _nz(text)
    if not t.strip():
        return {"sentiment_score": 0.0, "subjectivity_score": 0.0}
    blob = TextBlob(t)
    return {
        "sentiment_score": round(blob.sentiment.polarity, 4),
        "subjectivity_score": round(blob.sentiment.subjectivity, 4),
    }


# ── Creator-strategy indicators (Part C) ──────────────────────────────────────

def creator_strategy_indicators(text) -> Dict:
    """Deterministic keyword-rule indicators for creator-post text (rules in
    configs/feature_rules.yaml::creator_indicators). Each *_indicator is boolean."""
    rules = load_feature_rules()["creator_indicators"]
    text_l = _nz(text).lower()
    return {
        "disclosure_indicator": _keyword_hit(text_l, rules["disclosure"]),
        "discount_code_indicator": _keyword_hit(text_l, rules["discount_code"]),
        "product_mention_indicator": _keyword_hit(text_l, rules["product_mention"]),
        "call_to_action_indicator": _keyword_hit(text_l, rules["call_to_action"]),
        "launch_indicator": _keyword_hit(text_l, rules["launch"]),
        "community_indicator": _keyword_hit(text_l, rules["community"]),
        "sustainability_responsibility_indicator": _keyword_hit(text_l, rules["sustainability_responsibility"]),
        "quality_fit_indicator": _keyword_hit(text_l, rules["quality_fit"]),
    }


# ── Customer-experience term frequencies (Part C) ─────────────────────────────

def customer_experience_features(text) -> Dict:
    """Term-frequency counts per experience category (rules in
    configs/feature_rules.yaml::customer_experience_terms), plus sentiment and a
    simple review_polarity label derived from sentiment_score."""
    rules = load_feature_rules()["customer_experience_terms"]
    text_l = _nz(text).lower()
    out = {f"{cat}_term_frequency": _keyword_count(text_l, kws) for cat, kws in rules.items()}
    sent = sentiment_features(text)
    out.update(sent)
    score = sent["sentiment_score"]
    out["review_polarity"] = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
    return out


# ── Official-claim indicators (Part C) ────────────────────────────────────────

def official_claim_indicators(text, claim_category: Optional[str] = None) -> Dict:
    """Deterministic indicators for an official-claim sentence (rules in
    configs/feature_rules.yaml::official_claim_indicators)."""
    rules = load_feature_rules()["official_claim_indicators"]
    t = _nz(text)
    text_l = t.lower()
    return {
        "numerical_claim_indicator": bool(_NUMERICAL_RX.search(t)),
        "percentage_indicator": bool(_PERCENT_RX.search(t)),
        "target_year_indicator": bool(_YEAR_RX.search(t)),
        "certification_indicator": _keyword_hit(text_l, rules["certification"]),
        "material_indicator": _keyword_hit(text_l, rules["material"]),
        "geography_indicator": _keyword_hit(text_l, rules["geography"]),
        "claim_category": claim_category or "",
    }


# ── Content-intent weak labels (Part F) ───────────────────────────────────────

CONTENT_INTENT_LABELS = (
    "awareness", "product_demonstration", "product_launch", "conversion",
    "social_proof", "community", "education", "responsibility", "review",
    "mixed", "unclear",
)


def classify_content_intent(text) -> Dict:
    """Rule-based weak label over CONTENT_INTENT_LABELS (see
    docs/content_intent_codebook.md for full category definitions). Multiple
    keyword-category hits -> 'mixed'; no hits -> 'unclear'. Always
    requires_human_validation=True -- this is a weak label, not a verified one."""
    rules = load_feature_rules()["content_intent_keywords"]
    text_l = _nz(text).lower()
    hits: Dict[str, List[str]] = {}
    for label, kws in rules.items():
        matched = [kw for kw in kws if kw.lower() in text_l]
        if matched:
            hits[label] = matched

    if not hits:
        return {
            "intent_weak_label": "unclear", "intent_rule_evidence": "",
            "intent_confidence": "low", "requires_human_validation": True,
        }
    if len(hits) == 1:
        label = next(iter(hits))
        evidence = ", ".join(hits[label])
        return {
            "intent_weak_label": label, "intent_rule_evidence": evidence,
            "intent_confidence": "medium", "requires_human_validation": True,
        }
    label = max(hits, key=lambda k: len(hits[k]))
    evidence = "; ".join(f"{lbl}: {', '.join(kws)}" for lbl, kws in hits.items())
    return {
        "intent_weak_label": "mixed", "intent_rule_evidence": evidence,
        "intent_confidence": "low", "requires_human_validation": True,
    }


# ── Multi-label content-intent (Day 2 Task 2B Part D) ─────────────────────────
# Replaces the single forced-choice classifier above for the creator feature
# table. Scores each category from title (weight 2) + description/caption
# (weight 1) matches, so a title-level signal dominates a passing mention deeper
# in a description. 'mixed' is reserved for a genuine top-score TIE with score>0
# -- it is not the default whenever more than one category matches at all.

def classify_content_intent_multilabel(
    video_title: str = "", video_description: str = "", caption_or_description: str = "",
) -> Dict:
    """Multi-label content-intent classification over the 9
    CONTENT_INTENT_CATEGORIES. Returns per-category intent_<category> booleans,
    primary_intent (+evidence, +confidence), and secondary_intents (comma-joined,
    excludes primary). label_source='automated_weak_label',
    requires_human_validation=True always."""
    rules = load_feature_rules()["content_intent_keywords"]
    title_l = _nz(video_title).lower()
    desc_l = _nz(video_description).lower()
    caption_l = _nz(caption_or_description).lower()
    # avoid double-counting when caption_or_description duplicates the title (as
    # it does for DDG-sourced rows where caption = "title + snippet")
    body_l = desc_l if desc_l.strip() else caption_l

    scores: Dict[str, float] = {}
    title_matches: Dict[str, List[str]] = {}
    body_matches: Dict[str, List[str]] = {}
    for category, kws in rules.items():
        t_hits = [kw for kw in kws if kw.lower() in title_l]
        b_hits = [kw for kw in kws if kw.lower() in body_l]
        score = 2 * len(t_hits) + 1 * len(b_hits)
        if score > 0:
            scores[category] = score
            title_matches[category] = t_hits
            body_matches[category] = b_hits

    out: Dict[str, object] = {f"intent_{cat}": (cat in scores) for cat in CONTENT_INTENT_CATEGORIES}
    out["label_source"] = "automated_weak_label"
    out["requires_human_validation"] = True

    if not scores:
        out.update(primary_intent="unclear", primary_intent_evidence="",
                    primary_intent_confidence="low", secondary_intents="")
        return out

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranked[0][1]
    co_dominant = [cat for cat, s in ranked if s == top_score]

    if len(co_dominant) >= 2:
        evidence = "; ".join(
            f"{cat}: title={title_matches[cat]}, body={body_matches[cat]}" for cat in co_dominant
        )
        out.update(
            primary_intent="mixed", primary_intent_evidence=evidence,
            primary_intent_confidence="low",
            secondary_intents=", ".join(cat for cat, _ in ranked[len(co_dominant):]) or "",
        )
        return out

    primary = ranked[0][0]
    secondary = [cat for cat, _ in ranked[1:]]
    confidence = "high" if title_matches[primary] else ("medium" if body_matches[primary] else "low")
    evidence = f"title={title_matches[primary]}, body={body_matches[primary]}"
    out.update(
        primary_intent=primary, primary_intent_evidence=evidence,
        primary_intent_confidence=confidence,
        secondary_intents=", ".join(secondary),
    )
    return out


CONTENT_INTENT_CATEGORIES = (
    "awareness", "product_demonstration", "product_launch", "conversion",
    "social_proof", "community", "education", "responsibility", "review",
)


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
