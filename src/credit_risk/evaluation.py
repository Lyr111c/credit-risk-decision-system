"""Consistent evaluation metrics for binary default-probability models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_CALIBRATION_BINS = 10


def evaluate_binary_classifier(
    target: Sequence[int],
    probability: Sequence[float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, object]:
    """Return discrimination, calibration, and fixed-threshold metrics."""
    y_true = np.asarray(target)
    y_probability = np.asarray(probability, dtype=float)
    if len(y_true) != len(y_probability):
        raise ValueError("target and probability must have the same length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("target must contain both binary classes 0 and 1")
    if not np.isfinite(y_probability).all() or (
        (y_probability < 0) | (y_probability > 1)
    ).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)
    predictions = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    observed_rate, mean_probability = calibration_curve(
        y_true,
        y_probability,
        n_bins=calibration_bins,
        strategy="uniform",
    )
    interior_edges = np.linspace(0, 1, calibration_bins + 1)[1:-1]
    bin_indices = np.searchsorted(interior_edges, y_probability)
    nonempty_counts = np.bincount(bin_indices, minlength=calibration_bins)
    nonempty_counts = nonempty_counts[nonempty_counts > 0]
    calibration = [
        {
            "mean_predicted_probability": float(predicted),
            "observed_default_rate": float(observed),
            "count": int(count),
        }
        for predicted, observed, count in zip(
            mean_probability, observed_rate, nonempty_counts, strict=True
        )
    ]
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "ks": float(np.max(true_positive_rate - false_positive_rate)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "calibration": calibration,
    }
