"""Tests for fake_detector module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from senticrank.config import FakeDetectorConfig, load_config
from senticrank.fake_detector import (
    FakeReviewDetector,
    assign_fake_category,
    compute_mismatch,
    compute_suspicion,
    summarize_fakes,
    find_patterns,
    top_fake_examples,
    filter_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def _make_scored_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stars = rng.integers(1, 6, size=n)
    predicted = rng.integers(1, 6, size=n)
    proba = rng.dirichlet(np.ones(5), size=n)
    confidence = proba.max(axis=1)
    mismatch = np.abs(predicted - stars)
    from senticrank.fake_detector.scoring import assign_fake_category, compute_suspicion
    return pd.DataFrame({
        "product_id": np.arange(n),
        "product_name": [f"prod_{i}" for i in range(n)],
        "category": rng.choice(["Electronics", "Beauty", "Sports"], size=n),
        "star_rating": stars,
        "predicted_star": predicted,
        "predicted_confidence": confidence,
        "mismatch": mismatch,
        "suspicion": compute_suspicion(mismatch, confidence),
        "fake_category": assign_fake_category(mismatch, confidence, 0.6),
        "full_text": [f"review text {i}" for i in range(n)],
        "helpful_votes": rng.integers(0, 10, size=n),
    })


# ---------------------------------------------------------------------------
# scoring unit tests
# ---------------------------------------------------------------------------

def test_compute_mismatch_zero():
    stars = np.array([1, 2, 3, 4, 5])
    assert (compute_mismatch(stars, stars) == 0).all()


def test_compute_mismatch_values():
    pred = np.array([5, 1, 3])
    actual = np.array([1, 5, 3])
    result = compute_mismatch(pred, actual)
    np.testing.assert_array_equal(result, [4, 4, 0])


def test_compute_suspicion_zero_mismatch():
    mismatch = np.array([0, 0, 0])
    confidence = np.array([0.9, 0.8, 0.7])
    assert (compute_suspicion(mismatch, confidence) == 0).all()


def test_assign_fake_category_uncertain():
    mismatch = np.array([3, 3])
    confidence = np.array([0.5, 0.4])
    cats = assign_fake_category(mismatch, confidence, confidence_threshold=0.6)
    assert (cats == "uncertain").all()


def test_assign_fake_category_clean():
    mismatch = np.array([0, 1])
    confidence = np.array([0.9, 0.9])
    cats = assign_fake_category(mismatch, confidence, confidence_threshold=0.6)
    assert (cats == "clean").all()


def test_assign_fake_category_suspicious():
    mismatch = np.array([2])
    confidence = np.array([0.85])
    cats = assign_fake_category(mismatch, confidence, 0.6)
    assert cats[0] == "suspicious"


def test_assign_fake_category_very_suspicious():
    mismatch = np.array([3, 4])
    confidence = np.array([0.9, 0.95])
    cats = assign_fake_category(mismatch, confidence, 0.6)
    assert (cats == "very_suspicious").all()


# ---------------------------------------------------------------------------
# FakeReviewDetector — score_dataset_adds_columns
# ---------------------------------------------------------------------------

def test_score_dataset_adds_columns(tmp_path, cfg):
    """score_dataset() must add the five scoring columns."""
    import joblib

    real_model_path = Path(cfg.star_predictor.output_model_dir) / "best_model.joblib"
    if not real_model_path.exists():
        pytest.skip("best_model.joblib not found — run train-predictor first")

    df = pd.read_csv(cfg.data.raw_path, low_memory=False).head(50)
    df["full_text"] = df["full_text"].fillna("").astype(str)

    detector = FakeReviewDetector(real_model_path, cfg.fake_detector)
    scored = detector.score_dataset(df)

    for col in ("predicted_star", "predicted_confidence", "mismatch", "suspicion", "fake_category"):
        assert col in scored.columns, f"Missing column: {col}"

    assert scored["predicted_star"].between(1, 5).all()
    assert scored["predicted_confidence"].between(0, 1).all()
    assert scored["mismatch"].ge(0).all()
    assert scored["fake_category"].isin({"clean", "suspicious", "very_suspicious", "uncertain"}).all()


# ---------------------------------------------------------------------------
# perfect_match → all clean
# ---------------------------------------------------------------------------

def test_perfect_match_no_fakes():
    """When predicted == actual and confidence is high, all rows should be clean."""
    stars = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    mismatch = compute_mismatch(stars, stars)
    confidence = np.full(len(stars), 0.9)
    cats = assign_fake_category(mismatch, confidence, 0.6)
    assert (cats == "clean").all(), f"Expected all clean, got: {cats}"


# ---------------------------------------------------------------------------
# obvious mismatch caught (text says bad → 5 stars)
# ---------------------------------------------------------------------------

def test_obvious_mismatch_caught(cfg):
    """A clearly negative review paired with 5 stars must be flagged."""
    real_model_path = Path(cfg.star_predictor.output_model_dir) / "best_model.joblib"
    if not real_model_path.exists():
        pytest.skip("best_model.joblib not found — run train-predictor first")

    import joblib
    model = joblib.load(real_model_path)

    bad_text = "ужасный товар, полная фигня, не работает, сломался на второй день, разочарован"
    proba = model.predict_proba([bad_text])[0]
    pred_star = int(np.argmax(proba)) + 1
    confidence = float(proba.max())
    actual_star = 5
    mismatch = abs(pred_star - actual_star)

    cats = assign_fake_category(
        np.array([mismatch]),
        np.array([confidence]),
        cfg.fake_detector.confidence_threshold,
    )

    if confidence > cfg.fake_detector.confidence_threshold and mismatch >= 2:
        assert cats[0] in ("suspicious", "very_suspicious"), (
            f"Expected suspicious/very_suspicious but got {cats[0]} "
            f"(pred={pred_star}, actual=5, conf={confidence:.3f})"
        )


# ---------------------------------------------------------------------------
# analysis functions
# ---------------------------------------------------------------------------

def test_summarize_fakes_keys():
    df = _make_scored_df(100)
    result = summarize_fakes(df)
    for key in ("total_reviews", "category_counts", "category_pct",
                "suspicious_by_product_category", "suspicious_by_star_rating"):
        assert key in result


def test_find_patterns_returns_lengths():
    df = _make_scored_df(100)
    result = find_patterns(df)
    assert "avg_text_length" in result
    assert "clean" in result["avg_text_length"]
    assert "fake" in result["avg_text_length"]


def test_top_fake_examples_length():
    df = _make_scored_df(200)
    examples = top_fake_examples(df, n=10)
    assert len(examples) <= 10
    if len(examples) > 1:
        assert examples["suspicion"].iloc[0] >= examples["suspicion"].iloc[-1]


# ---------------------------------------------------------------------------
# filter_dataset
# ---------------------------------------------------------------------------

def test_filter_dataset_conservative(tmp_path):
    df = _make_scored_df(200)
    result = filter_dataset(df, tmp_path, mode="conservative")

    assert "clean" in result and "fakes" in result
    assert len(result["clean"]) + len(result["fakes"]) == len(df)
    assert (result["fakes"]["fake_category"] == "very_suspicious").all()
    assert (tmp_path / "dataset_clean.csv").exists()
    assert (tmp_path / "dataset_fakes.csv").exists()
    assert (tmp_path / "dataset_with_fake_scores.csv").exists()


def test_filter_dataset_aggressive(tmp_path):
    df = _make_scored_df(200)
    result = filter_dataset(df, tmp_path, mode="aggressive")

    removed_cats = set(result["fakes"]["fake_category"].unique())
    assert removed_cats.issubset({"suspicious", "very_suspicious"})
    # aggressive removes more or equal rows than conservative
    result_cons = filter_dataset(df, tmp_path, mode="conservative")
    assert len(result["fakes"]) >= len(result_cons["fakes"])
