"""Bayesian shrinkage discount for products with few reviews."""

from __future__ import annotations

import pandas as pd


def apply_confidence_discount(
    scores: pd.Series,
    n_reviews: pd.Series,
    prior_score: float = 50.0,
    smoothing_k: int = 5,
) -> pd.Series:
    """Bayesian shrinkage: pull low-review products toward prior_score.

    Formula: discounted = (n × score + k × prior) / (n + k)

    At n=0: returns prior_score
    At n=k: equal weight between data and prior
    At n>>k: returns score unchanged

    Args:
        scores: Product scores (e.g. scaled to [0, 100]).
        n_reviews: Number of clean reviews per product.
        prior_score: Global prior (neutral default = 50.0).
        smoothing_k: Equivalent prior observation count.

    Returns:
        Shrunk scores as Series.
    """
    return (n_reviews * scores + smoothing_k * prior_score) / (n_reviews + smoothing_k)
