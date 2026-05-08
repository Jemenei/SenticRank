"""AHP priority-vector and consistency-ratio calculator."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AHPCalculator:
    """Analytic Hierarchy Process weight derivation (Saaty 1980).

    Args:
        pairwise_matrix: Square positive reciprocal matrix (list of lists or ndarray).
    """

    RI_TABLE: dict[int, float] = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90,
        5: 1.12, 6: 1.24, 7: 1.32,
    }

    def __init__(self, pairwise_matrix: list[list[float]] | np.ndarray) -> None:
        self.A = np.array(pairwise_matrix, dtype=float)
        self.n = self.A.shape[0]
        if self.A.shape != (self.n, self.n):
            raise ValueError("Pairwise matrix must be square.")

    def priority_vector(self) -> np.ndarray:
        """Compute normalized eigenvector via column-normalization method.

        Returns:
            Priority weights array of length n, summing to 1.
        """
        col_sums = self.A.sum(axis=0)
        normalized = self.A / col_sums
        return normalized.mean(axis=1)

    def consistency_ratio(self) -> float:
        """Compute Consistency Ratio (CR = CI / RI).

        Returns:
            CR value. Should be < 0.10 for an acceptable matrix.
        """
        w = self.priority_vector()
        lambda_max = float(((self.A @ w) / w).mean())
        ci = (lambda_max - self.n) / (self.n - 1)
        ri = self.RI_TABLE.get(self.n, 1.49)
        cr = ci / ri if ri > 0 else 0.0
        logger.info("AHP: n=%d, λmax=%.4f, CI=%.4f, RI=%.2f, CR=%.4f", self.n, lambda_max, ci, ri, cr)
        if cr > 0.10:
            logger.warning("AHP CR=%.4f > 0.10 — matrix is inconsistent, weights unreliable.", cr)
        return cr
