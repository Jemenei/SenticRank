"""Fuzzy membership functions and transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from senticrank.config import FuzzyConfig


def trapezoidal(x: float | np.ndarray, a: float, b: float, c: float, d: float) -> float | np.ndarray:
    """Trapezoidal membership function.

    Args:
        x: Input value(s) in [0, 1].
        a, b: Left foot and left shoulder.
        c, d: Right shoulder and right foot.

    Returns:
        Membership degree(s) in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    result = np.where((x >= a) & (x < b), (x - a) / (b - a + 1e-12), result)
    result = np.where((x >= b) & (x <= c), 1.0, result)
    result = np.where((x > c) & (x <= d), (d - x) / (d - c + 1e-12), result)
    return float(result) if result.ndim == 0 else result


def s_curve(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """S-shaped membership function via linear interpolation.

    Args:
        x: Input value(s).
        a: Crossover point (output ≈ 0).
        b: Saturation point (output ≈ 1).

    Returns:
        Membership degree(s) in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    result = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def _defuzz_height(x: float | np.ndarray, traps: list[tuple[float, float, float, float]]) -> float | np.ndarray:
    """Height-method defuzzification for a set of trapezoids.

    Centroids are midpoints of each trapezoid's flat region [b, c].

    Args:
        x: Input value(s) in [0, 1].
        traps: List of (a, b, c, d) trapezoid params.

    Returns:
        Defuzzified output in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    heights = np.stack([trapezoidal(x, *t) for t in traps], axis=1)
    centroids = np.array([(t[1] + t[2]) / 2.0 for t in traps])
    h_sum = heights.sum(axis=1)
    output = np.where(h_sum == 0, 0.5, (heights * centroids).sum(axis=1) / h_sum)
    return float(output[0]) if scalar else output


class FuzzyTransformer:
    """Applies fuzzy membership functions to product-level aggregates.

    Args:
        config: FuzzyConfig with trapezoid and S-curve parameters.
    """

    def __init__(self, config: FuzzyConfig) -> None:
        self._traps = [
            tuple(config.quality_trapezoids.low),
            tuple(config.quality_trapezoids.medium),
            tuple(config.quality_trapezoids.high),
        ]
        self._cons_a, self._cons_b = config.consensus_s_curve
        self._disp_a, self._disp_b = config.dispute_s_curve

    def fuzzy_quality(self, predicted_star_norm: np.ndarray) -> np.ndarray:
        """Transform normalized predicted star into fuzzy quality score.

        Args:
            predicted_star_norm: Values in [0, 1].

        Returns:
            Fuzzy quality scores in [0, 1].
        """
        return _defuzz_height(predicted_star_norm, self._traps)

    def fuzzy_consensus(self, confidence: np.ndarray) -> np.ndarray:
        """High confidence → high consensus.

        Args:
            confidence: Model confidence values in [0, 1].

        Returns:
            Consensus scores in [0, 1].
        """
        return s_curve(confidence, self._cons_a, self._cons_b)

    def fuzzy_dispute(self, mean_mismatch_norm: np.ndarray) -> np.ndarray:
        """Low mismatch → high anti-dispute score (benefit criterion).

        Args:
            mean_mismatch_norm: Mean mismatch normalized to [0, 1] (divide raw by 4).

        Returns:
            Anti-dispute scores in [0, 1].
        """
        return 1.0 - s_curve(mean_mismatch_norm, self._disp_a, self._disp_b)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fuzzy_quality, fuzzy_consensus, fuzzy_dispute columns.

        Args:
            df: Product-level DataFrame with agg_predicted_star_norm,
                agg_predicted_confidence, agg_mean_mismatch columns.

        Returns:
            Copy of df with three fuzzy columns added.
        """
        out = df.copy()
        out["fuzzy_quality"] = self.fuzzy_quality(out["agg_predicted_star_norm"].to_numpy(float))
        out["fuzzy_consensus"] = self.fuzzy_consensus(out["agg_predicted_confidence"].to_numpy(float))
        mismatch_norm = out["agg_mean_mismatch"].to_numpy(float) / 4.0
        out["fuzzy_dispute"] = self.fuzzy_dispute(mismatch_norm)
        return out
