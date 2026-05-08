"""SenticRankSystem: orchestrates the full ranking pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from senticrank.config import RankingConfig
from .aggregation import TrustWeightedAggregator
from .fuzzy import FuzzyTransformer
from .ahp import AHPCalculator
from .topsis import TOPSISRanker
from .confidence import apply_confidence_discount

logger = logging.getLogger(__name__)


def _scale_to_100(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.full_like(scores, 50.0)
    return (scores - mn) / (mx - mn) * 100.0


class SenticRankSystem:
    """Full SenticRank pipeline: aggregate → fuzzy → AHP → TOPSIS → discount.

    Args:
        config: RankingConfig with all hyper-parameters.
    """

    def __init__(self, config: RankingConfig) -> None:
        self.cfg = config
        self._aggregator = TrustWeightedAggregator(config.fake_categories_to_remove)
        self._fuzzy = FuzzyTransformer(config.fuzzy)
        self._ahp = AHPCalculator(config.ahp_matrix)

    def run(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full ranking pipeline.

        Args:
            scored_df: dataset_with_fake_scores.csv as DataFrame.

        Returns:
            Product-level rankings DataFrame.
        """
        fakes_total = scored_df["fake_category"].isin(self.cfg.fake_categories_to_remove).sum()
        logger.info("Fake reviews excluded from ranking: %d", fakes_total)

        products = self._aggregator.aggregate(scored_df)
        products = self._fuzzy.transform(products)

        helpful_log = np.log1p(products["agg_helpful_votes_total"].to_numpy(float))
        rng = helpful_log.max() - helpful_log.min()
        products["fuzzy_engagement"] = (helpful_log - helpful_log.min()) / (rng + 1e-9)

        cr = self._ahp.consistency_ratio()
        ahp_weights = self._ahp.priority_vector()
        logger.info("AHP weights: %s  (CR=%.4f)", np.round(ahp_weights, 4).tolist(), cr)

        criteria_cols = self.cfg.criteria
        X = products[criteria_cols].to_numpy(dtype=float)
        ranker = TOPSISRanker(ahp_weights)
        topsis_scores = ranker.rank(X)
        products["topsis_score"] = topsis_scores

        topsis_100 = _scale_to_100(topsis_scores)
        disc_cfg = self.cfg.confidence_discount
        if disc_cfg.method == "bayesian_shrinkage":
            n = products["num_reviews_clean"]
            final_100 = apply_confidence_discount(
                pd.Series(topsis_100, index=products.index),
                n,
                prior_score=disc_cfg.prior_score,
                smoothing_k=disc_cfg.smoothing_k,
            ).to_numpy()
            products["confidence_discount"] = (n / (n + disc_cfg.smoothing_k)).to_numpy()
        else:
            final_100 = topsis_100
            products["confidence_discount"] = 1.0

        products["senticrank_score_100"] = final_100
        products["senticrank_score"] = final_100 / 100.0

        products["baseline_star_score"] = products["avg_star_rating"] / 5.0 * 100.0
        products["sentic_rank_overall"] = (
            products["senticrank_score"].rank(ascending=False, method="min").astype(int)
        )
        products["baseline_rank_overall"] = (
            products["baseline_star_score"].rank(ascending=False, method="min").astype(int)
        )
        products["rank_delta"] = products["baseline_rank_overall"] - products["sentic_rank_overall"]
        products["sentic_rank_in_category"] = (
            products.groupby("category")["senticrank_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        rho, _ = spearmanr(products["sentic_rank_overall"], products["baseline_rank_overall"])
        logger.info("Spearman ρ (SenticRank vs baseline): %.4f", rho)
        products.attrs["spearman_rho"] = float(rho)
        products.attrs["ahp_cr"] = float(cr)
        products.attrs["ahp_weights"] = ahp_weights.tolist()

        self._log_top5_per_category(products)
        return products

    def _log_top5_per_category(self, df: pd.DataFrame) -> None:
        for cat, grp in df.groupby("category"):
            top5 = grp.nsmallest(5, "sentic_rank_in_category")[
                ["product_name", "senticrank_score_100", "avg_star_rating", "num_reviews_clean"]
            ]
            logger.info("Top-5 [%s]:\n%s", cat, top5.to_string(index=False))

    def save(self, products: pd.DataFrame, output_dir: Path, suffix: str = "") -> Path:
        """Save full rankings CSV and per-category CSVs.

        Args:
            products: Output of run().
            output_dir: Base output directory.
            suffix: Optional suffix for directory name (e.g. '_with_fakes').

        Returns:
            Path to main rankings CSV.
        """
        out = Path(output_dir) if not suffix else Path(str(output_dir) + suffix)
        out.mkdir(parents=True, exist_ok=True)
        cat_dir = out / "category_rankings"
        cat_dir.mkdir(exist_ok=True)

        main_path = out / "senticrank_product_rankings.csv"
        _output_cols = [
            "product_id", "product_name", "category",
            "num_reviews_total", "num_reviews_clean", "num_fakes_removed",
            "avg_star_rating", "avg_predicted_star", "avg_predicted_confidence",
            "agg_norm_rating", "agg_predicted_star_norm", "agg_predicted_confidence",
            "agg_mean_mismatch", "agg_helpful_votes_total",
            "fuzzy_quality", "fuzzy_consensus", "fuzzy_dispute", "fuzzy_engagement",
            "topsis_score", "confidence_discount", "senticrank_score", "senticrank_score_100",
            "baseline_star_score", "sentic_rank_overall", "sentic_rank_in_category",
            "baseline_rank_overall", "rank_delta",
        ]
        save_cols = [c for c in _output_cols if c in products.columns]
        products[save_cols].to_csv(main_path, index=False)

        for cat, grp in products.groupby("category"):
            cat_path = cat_dir / f"{cat.lower()}_rankings.csv"
            grp[save_cols].sort_values("sentic_rank_in_category").to_csv(cat_path, index=False)

        logger.info("Rankings saved to %s (%d products)", main_path, len(products))
        return main_path
