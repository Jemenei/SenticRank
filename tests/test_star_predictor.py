"""Tests for star_predictor module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from senticrank.star_predictor import (
    DummyPredictor,
    TfidfLogregPredictor,
    compute_all_metrics,
)
from senticrank.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def sample_texts_labels():
    df = pd.read_csv("data/raw/senticrank_master_dataset.csv", low_memory=False)
    df = df.dropna(subset=["star_rating", "full_text"])
    df["star_rating"] = df["star_rating"].astype(int)
    df = df[df["full_text"].str.len() >= 5]
    df = df.sample(n=min(300, len(df)), random_state=42).reset_index(drop=True)
    texts = df["full_text"].tolist()
    labels = df["star_rating"].values
    return texts, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dummy_baseline_works(sample_texts_labels):
    texts, labels = sample_texts_labels
    pred = DummyPredictor()
    pred.fit(texts, labels)
    preds = pred.predict(texts)
    majority = np.bincount(labels).argmax()
    assert set(preds) == {majority}


def test_tfidf_logreg_beats_dummy(sample_texts_labels, cfg):
    texts, labels = sample_texts_labels
    dummy = DummyPredictor()
    dummy.fit(texts, labels)
    dummy_f1 = compute_all_metrics(labels, dummy.predict(texts))["f1_macro"]

    logreg = TfidfLogregPredictor(cfg.star_predictor)
    logreg.fit(texts, labels)
    logreg_f1 = compute_all_metrics(labels, logreg.predict(texts))["f1_macro"]

    assert logreg_f1 > dummy_f1, f"logreg f1={logreg_f1:.3f} not > dummy f1={dummy_f1:.3f}"


def test_predictions_in_valid_range(sample_texts_labels, cfg):
    texts, labels = sample_texts_labels
    logreg = TfidfLogregPredictor(cfg.star_predictor)
    logreg.fit(texts, labels)
    preds = logreg.predict(texts)
    assert set(preds).issubset({1, 2, 3, 4, 5}), f"Invalid predictions: {set(preds) - {1,2,3,4,5}}"


def test_save_load_roundtrip(sample_texts_labels, cfg, tmp_path):
    texts, labels = sample_texts_labels
    logreg = TfidfLogregPredictor(cfg.star_predictor)
    logreg.fit(texts, labels)
    preds_before = logreg.predict(texts)

    save_path = tmp_path / "logreg.joblib"
    logreg.save(save_path)
    loaded = TfidfLogregPredictor.load(save_path)
    preds_after = loaded.predict(texts)

    np.testing.assert_array_equal(preds_before, preds_after)


def test_metrics_perfect_prediction():
    y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    m = compute_all_metrics(y, y)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["qwk"] == pytest.approx(1.0)


def test_metrics_handle_class_imbalance():
    # 90% class 5, 10% class 1 — accuracy will be high but f1_macro low
    y_true = np.array([5] * 90 + [1] * 10)
    y_pred = np.array([5] * 100)  # always predict 5
    m = compute_all_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.90)
    assert m["f1_macro"] < m["accuracy"], "f1_macro should be lower than accuracy for skewed data"
    assert m["f1_macro"] < 0.5
