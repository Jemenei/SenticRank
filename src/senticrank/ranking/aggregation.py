"""Trust-weighted aggregation from review-level to product-level."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FAKE_CATS = frozenset(["suspicious", "very_suspicious"])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = weights.sum()
    if w_sum == 0:
        return float(values.mean())
    return float((values * weights).sum() / w_sum)


class TrustWeightedAggregator:
    """Aggregates review-level features to product-level via trust weights.

    Args:
        fake_categories_to_remove: Categories that are excluded before aggregation.
    """

    def __init__(self, fake_categories_to_remove: list[str] | None = None) -> None:
        self._remove = set(fake_categories_to_remove or _FAKE_CATS)

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-product aggregates.

        Args:
            df: Scored dataset with fake_category, trust_weight, norm_rating,
                predicted_star, predicted_confidence, mismatch columns.

        Returns:
            Product-level DataFrame with aggregated features.
        """
        total_by_product = df.groupby("product_id").size().rename("num_reviews_total")

        clean = df[~df["fake_category"].isin(self._remove)].copy()
        clean["predicted_star_norm"] = (clean["predicted_star"] - 1) / 4.0

        records: list[dict] = []
        for pid, grp in clean.groupby("product_id"):
            w = grp["trust_weight"].to_numpy(dtype=float)
            records.append({
                "product_id": pid,
                "product_name": grp["product_name"].iloc[0],
                "category": grp["category"].iloc[0],
                "num_reviews_clean": len(grp),
                "avg_star_rating": float(grp["star_rating"].mean()),
                "avg_predicted_star": float(grp["predicted_star"].mean()),
                "avg_predicted_confidence": float(grp["predicted_confidence"].mean()),
                "agg_norm_rating": _weighted_mean(grp["norm_rating"].to_numpy(float), w),
                "agg_predicted_star_norm": _weighted_mean(grp["predicted_star_norm"].to_numpy(float), w),
                "agg_predicted_confidence": _weighted_mean(grp["predicted_confidence"].to_numpy(float), w),
                "agg_mean_mismatch": _weighted_mean(grp["mismatch"].to_numpy(float), w),
                "agg_helpful_votes_total": float(grp["helpful_votes"].fillna(0).sum()),
            })

        agg = pd.DataFrame(records)
        agg = agg.merge(total_by_product.reset_index(), on="product_id", how="left")
        agg["num_fakes_removed"] = agg["num_reviews_total"] - agg["num_reviews_clean"]

        logger.info(
            "Aggregated %d products (total reviews=%d, fakes removed=%d)",
            len(agg), agg["num_reviews_total"].sum(), agg["num_fakes_removed"].sum(),
        )
        return agg
