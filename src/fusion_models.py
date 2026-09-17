"""Early, late, and hybrid multimodal fusion models."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def early_fusion_concat(
    feature_arrays: List[np.ndarray],
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate multiple feature arrays column-wise (early fusion baseline).

    All arrays must have the same number of rows.
    Returns a single ndarray of shape (n_samples, sum_of_features).
    """
    for i, arr in enumerate(feature_arrays):
        if arr.shape[0] != feature_arrays[0].shape[0]:
            raise ValueError(f"Array {i} has {arr.shape[0]} rows; expected {feature_arrays[0].shape[0]}.")
    return np.hstack(feature_arrays)


def late_fusion_average(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Average predictions from multiple modality-specific models (late fusion).

    predictions: list of probability/score arrays, each shape (n_samples,) or (n_samples, n_classes).
    weights: optional per-modality weights (must sum to 1 if provided).
    Returns averaged prediction array of the same shape as each input.
    """
    stacked = np.stack(predictions, axis=0)
    if weights is not None:
        if len(weights) != len(predictions):
            raise ValueError("weights length must match number of prediction arrays.")
        w = np.array(weights, dtype=float)
        w /= w.sum()
        return np.average(stacked, axis=0, weights=w)
    return stacked.mean(axis=0)


def late_fusion_vote(
    class_predictions: List[np.ndarray],
) -> np.ndarray:
    """Majority-vote late fusion over hard class predictions.

    class_predictions: list of 1-D integer arrays of class labels.
    Returns 1-D array of majority-voted class labels.
    """
    from scipy import stats  # lazy import
    stacked = np.stack(class_predictions, axis=1)
    voted, _ = stats.mode(stacked, axis=1, keepdims=False)
    return voted.flatten()


def hybrid_fusion(
    early_features: np.ndarray,
    late_predictions: List[np.ndarray],
    late_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Combine early-fusion features with late-fusion prediction signals.

    Appends late-fusion averaged predictions as additional columns to early features.
    Returns an ndarray of shape (n_samples, early_dim + n_late_modalities).
    """
    late_avg = late_fusion_average(late_predictions, weights=late_weights)
    if late_avg.ndim == 1:
        late_avg = late_avg.reshape(-1, 1)
    return np.hstack([early_features, late_avg])


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "logistic",
) -> object:
    """Fit a simple scikit-learn classifier on multimodal features.

    model_type: 'logistic' | 'random_forest' | 'gradient_boosting'.
    Returns the fitted model object.
    """
    from sklearn.linear_model import LogisticRegression  # lazy import
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    models = {
        "logistic": LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(models.keys())}.")
    clf = models[model_type]
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Return accuracy, macro F1, and classification report for a fitted classifier.

    Returns a dict with keys: accuracy, macro_f1, report_str.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report  # lazy import
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "report_str": classification_report(y_test, y_pred, zero_division=0),
    }
