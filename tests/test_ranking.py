"""Tests for ranking module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from senticrank.config import load_config
from senticrank.ranking import (
    AHPCalculator,
    TOPSISRanker,
    TrustWeightedAggregator,
    FuzzyTransformer,
    apply_confidence_discount,
    SenticRankSystem,
)
from senticrank.ranking.fuzzy import trapezoidal, s_curve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


def _make_review_df(n_products: int = 5, reviews_per: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_products):
        for _ in range(reviews_per):
            star = int(rng.integers(1, 6))
            pred = int(rng.integers(1, 6))
            rows.append({
                "product_id": f"P{pid:03d}",
                "product_name": f"Product {pid}",
                "category": ["Electronics", "Beauty", "Sports"][pid % 3],
                "star_rating": star,
                "norm_rating": (star - 1) / 4.0,
                "predicted_star": pred,
                "predicted_confidence": float(rng.uniform(0.4, 0.99)),
                "mismatch": abs(pred - star),
                "trust_weight": float(rng.uniform(0.2, 1.0)),
                "fake_category": rng.choice(
                    ["clean", "clean", "clean", "uncertain", "suspicious"],
                    p=[0.7, 0.1, 0.1, 0.06, 0.04],
                ),
                "helpful_votes": int(rng.integers(0, 20)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------

def test_topsis_all_same_scores():
    X = np.ones((4, 3))
    ranker = TOPSISRanker([1, 1, 1])
    C = ranker.rank(X)
    assert C.shape == (4,)
    np.testing.assert_allclose(C, 0.5, atol=1e-6)


def test_topsis_weights_normalized():
    ranker = TOPSISRanker([2.0, 2.0, 1.0])
    np.testing.assert_allclose(ranker.weights.sum(), 1.0)


def test_topsis_best_dominates():
    X = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
    ])
    ranker = TOPSISRanker([0.25, 0.25, 0.25, 0.25])
    C = ranker.rank(X)
    assert C[0] > C[2] > C[1], f"Expected C[0]>C[2]>C[1], got {C}"


def test_topsis_textbook_example():
    """3 alternatives × 4 criteria, equal weights — best must outscore worst."""
    X = np.array([
        [0.9, 0.8, 0.7, 0.9],
        [0.3, 0.4, 0.3, 0.4],
        [0.6, 0.5, 0.6, 0.6],
    ])
    weights = [0.25, 0.25, 0.30, 0.20]
    ranker = TOPSISRanker(weights)
    C = ranker.rank(X)
    assert C[0] > C[2] > C[1], f"Unexpected order: {C}"
    assert C[0] == pytest.approx(1.0, abs=1e-4)
    assert C[1] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------

def test_ahp_equal_weights_matrix():
    # All-ones matrix → equal weights, λmax=n, CR=0
    A = np.ones((4, 4))
    ahp = AHPCalculator(A)
    w = ahp.priority_vector()
    np.testing.assert_allclose(w, 0.25, atol=1e-6)
    cr = ahp.consistency_ratio()
    assert cr == pytest.approx(0.0, abs=1e-6)


def test_ahp_known_matrix(cfg):
    ahp = AHPCalculator(cfg.ranking.ahp_matrix)
    cr = ahp.consistency_ratio()
    assert cr < 0.10, f"CR={cr:.4f} should be < 0.10"
    w = ahp.priority_vector()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert len(w) == 5
    assert w[0] > w[1] > w[2] > w[3] > w[4], f"Expected descending weights, got {w}"


def test_ahp_cr_consistent_matrix():
    A = np.array([
        [1,   3,   7],
        [1/3, 1,   3],
        [1/7, 1/3, 1],
    ])
    ahp = AHPCalculator(A)
    cr = ahp.consistency_ratio()
    assert cr < 0.10, f"Known consistent matrix has CR={cr:.4f}"


# ---------------------------------------------------------------------------
# Fuzzy functions
# ---------------------------------------------------------------------------

def test_trapezoidal_plateau():
    assert trapezoidal(0.5, 0.25, 0.375, 0.625, 0.75) == pytest.approx(1.0)


def test_trapezoidal_outside():
    assert trapezoidal(0.0, 0.5, 0.75, 1.0, 1.0) == pytest.approx(0.0)
    assert trapezoidal(1.0, 0.0, 0.0, 0.25, 0.5) == pytest.approx(0.0)


def test_s_curve_bounds():
    assert s_curve(0.0, 0.3, 0.8) == pytest.approx(0.0)
    assert s_curve(1.0, 0.3, 0.8) == pytest.approx(1.0)


def test_fuzzy_quality_normalized_to_unit_interval(cfg):
    ft = FuzzyTransformer(cfg.ranking.fuzzy)
    xs = np.linspace(0, 1, 21)
    qs = ft.fuzzy_quality(xs)
    assert np.all(qs >= 0.0) and np.all(qs <= 1.0), f"Out of [0,1]: {qs}"


def test_fuzzy_dispute_inverted(cfg):
    ft = FuzzyTransformer(cfg.ranking.fuzzy)
    low = float(ft.fuzzy_dispute(np.array([0.0]))[0])
    high = float(ft.fuzzy_dispute(np.array([1.0]))[0])
    assert low > high, "Low mismatch should yield higher anti-dispute score"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregation_returns_product_rows():
    df = _make_review_df(5, 10)
    agg = TrustWeightedAggregator()
    result = agg.aggregate(df)
    assert len(result) == 5
    for col in ("num_reviews_total", "num_reviews_clean", "num_fakes_removed",
                "agg_norm_rating", "agg_predicted_star_norm", "agg_predicted_confidence",
                "agg_helpful_votes_total"):
        assert col in result.columns, f"Missing column: {col}"


def test_aggregation_zero_weights_fallback():
    df = _make_review_df(2, 5)
    df["trust_weight"] = 0.0
    df["fake_category"] = "clean"
    agg = TrustWeightedAggregator()
    result = agg.aggregate(df)
    assert result["agg_norm_rating"].notna().all()


def test_aggregation_counts_fakes_correctly():
    df = _make_review_df(3, 10)
    # Mark P001 reviews: half clean, half very_suspicious
    mask = (df["product_id"] == "P001")
    df.loc[mask & (df.index % 2 == 0), "fake_category"] = "very_suspicious"
    df.loc[mask & (df.index % 2 != 0), "fake_category"] = "clean"
    agg = TrustWeightedAggregator()
    result = agg.aggregate(df)
    p1 = result[result["product_id"] == "P001"].iloc[0]
    assert p1["num_fakes_removed"] == 5
    assert p1["num_reviews_clean"] == 5


# ---------------------------------------------------------------------------
# Confidence discount
# ---------------------------------------------------------------------------

def test_confidence_discount_single_review():
    # n=1, k=5, prior=50, score=100 → (1×100 + 5×50)/6 = 58.33
    scores = pd.Series([100.0])
    n = pd.Series([1])
    discounted = apply_confidence_discount(scores, n, prior_score=50.0, smoothing_k=5)
    assert 40.0 <= float(discounted.iloc[0]) <= 70.0, f"Expected ~58.3, got {discounted.iloc[0]}"


def test_confidence_discount_large_n():
    # n=1000, k=5 → score barely changes
    scores = pd.Series([80.0])
    n = pd.Series([1000])
    discounted = apply_confidence_discount(scores, n, prior_score=50.0, smoothing_k=5)
    assert discounted.iloc[0] == pytest.approx(80.0, abs=0.5)


def test_confidence_discount_zero_reviews_returns_prior():
    scores = pd.Series([100.0])
    n = pd.Series([0])
    discounted = apply_confidence_discount(scores, n, prior_score=50.0, smoothing_k=5)
    assert discounted.iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_runs(cfg):
    df = _make_review_df(n_products=5, reviews_per=15)
    df["suspicion"] = 0.0
    system = SenticRankSystem(cfg.ranking)
    result = system.run(df)

    assert len(result) == 5
    for col in ("senticrank_score", "senticrank_score_100", "sentic_rank_overall",
                "sentic_rank_in_category", "baseline_rank_overall", "rank_delta",
                "fuzzy_engagement", "agg_helpful_votes_total"):
        assert col in result.columns

    assert result["senticrank_score_100"].between(0, 100).all()
    assert result["sentic_rank_overall"].min() == 1
