from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from .base import BasePredictor


class DummyPredictor(BasePredictor):
    name = "dummy"

    def __init__(self) -> None:
        self.clf = DummyClassifier(strategy="most_frequent")

    def fit(self, texts: list[str], labels: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        self.clf.fit(texts, labels, sample_weight=sample_weight)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(texts)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "DummyPredictor":
        return joblib.load(path)
