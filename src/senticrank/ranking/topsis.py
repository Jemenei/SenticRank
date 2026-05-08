"""TOPSIS multi-criteria ranking (Hwang & Yoon 1981)."""

from __future__ import annotations

import numpy as np


class TOPSISRanker:
    """Standard TOPSIS with vector normalization.

    All criteria are treated as benefit criteria (higher = better).

    Args:
        weights: Criterion weights (will be L1-normalized internally).
    """

    def __init__(self, weights: list[float] | np.ndarray) -> None:
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def rank(self, X: np.ndarray) -> np.ndarray:
        """Compute TOPSIS closeness coefficients.

        Args:
            X: Decision matrix of shape (n_alternatives, n_criteria).
               All criteria assumed benefit (higher is better).

        Returns:
            Closeness coefficient array C in [0, 1], shape (n_alternatives,).
            Higher C → closer to ideal solution.
        """
        X = np.asarray(X, dtype=float)
        col_norms = np.sqrt((X ** 2).sum(axis=0))
        col_norms = np.where(col_norms == 0, 1.0, col_norms)
        R = X / col_norms

        V = R * self.weights

        A_pos = V.max(axis=0)
        A_neg = V.min(axis=0)

        d_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
        d_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

        denom = d_pos + d_neg
        with np.errstate(invalid="ignore"):
            C = np.where(denom == 0, 0.5, d_neg / denom)
        return C
