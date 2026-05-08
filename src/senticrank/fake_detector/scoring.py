"""Mismatch and suspicion score computation."""

from __future__ import annotations

import numpy as np
import pandas as pd


FAKE_CATEGORIES = ("clean", "suspicious", "very_suspicious", "uncertain")


def compute_mismatch(predicted_star: np.ndarray, actual_star: np.ndarray) -> np.ndarray:
    """Absolute difference between predicted and actual star ratings.

    Args:
        predicted_star: Array of predicted star values (1–5).
        actual_star: Array of actual star values (1–5).

    Returns:
        Integer array of absolute mismatches.
    """
    return np.abs(predicted_star.astype(int) - actual_star.astype(int))


def compute_suspicion(mismatch: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Confidence-weighted suspicion score.

    Args:
        mismatch: Absolute mismatch array.
        confidence: Model confidence (max predicted_proba) per row.

    Returns:
        Float array of suspicion scores in [0, 4].
    """
    return mismatch.astype(float) * confidence


def assign_fake_category(
    mismatch: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.6,
) -> np.ndarray:
    """Assign fake_category label based on mismatch and confidence.

    Rules:
      - uncertain:       confidence <= threshold
      - clean:           mismatch <= 1
      - suspicious:      mismatch == 2 and confidence > threshold
      - very_suspicious: mismatch >= 3 and confidence > threshold

    Args:
        mismatch: Absolute mismatch per review.
        confidence: Max predicted_proba per review.
        confidence_threshold: Min confidence to flag as suspicious.

    Returns:
        Object array of category strings.
    """
    cats = np.full(len(mismatch), "uncertain", dtype=object)
    cats[mismatch <= 1] = "clean"
    cats[(mismatch == 2) & (confidence > 0.7)] = "suspicious"
    cats[(mismatch >= 3) & (confidence > confidence_threshold)] = "very_suspicious"
    return cats


def score_dataframe(
    df: pd.DataFrame,
    predicted_star: np.ndarray,
    predicted_proba: np.ndarray,
    confidence_threshold: float = 0.6,
) -> pd.DataFrame:
    """Attach scoring columns to a copy of df.

    Args:
        df: Source DataFrame with star_rating column.
        predicted_star: Model predictions (1–5) for each row.
        predicted_proba: Probability matrix (N × 5).
        confidence_threshold: Threshold for uncertain vs suspicious.

    Returns:
        DataFrame with added columns: predicted_star, predicted_confidence,
        mismatch, suspicion, fake_category.
    """
    out = df.copy()
    confidence = predicted_proba.max(axis=1)
    mismatch = compute_mismatch(predicted_star, df["star_rating"].values)

    out["predicted_star"] = predicted_star
    out["predicted_confidence"] = confidence
    out["mismatch"] = mismatch
    out["suspicion"] = compute_suspicion(mismatch, confidence)
    out["fake_category"] = assign_fake_category(mismatch, confidence, confidence_threshold)
    return out
