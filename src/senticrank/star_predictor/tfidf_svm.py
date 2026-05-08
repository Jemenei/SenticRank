from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .base import BasePredictor


class TfidfLinearSVMPredictor(BasePredictor):
    name = "tfidf_svm"

    def __init__(self, config) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=config.tfidf.max_features,
                ngram_range=tuple(config.tfidf.ngram_range),
                min_df=config.tfidf.min_df,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(
                    C=config.svm.C,
                    class_weight="balanced",
                    max_iter=config.svm.max_iter,
                ),
                method="sigmoid",
                cv=3,
            )),
        ])

    def fit(self, texts: list[str], labels: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        fit_params = {}
        if sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight
        self.pipeline.fit(texts, labels, **fit_params)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfLinearSVMPredictor":
        return joblib.load(path)
