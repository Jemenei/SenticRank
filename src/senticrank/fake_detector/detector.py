"""FakeReviewDetector: scores and flags suspicious reviews."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from senticrank.config import FakeDetectorConfig
from .scoring import score_dataframe

logger = logging.getLogger(__name__)


class FakeReviewDetector:
    """Wraps a trained star predictor to detect review-star mismatches.

    Args:
        model_path: Path to best_model.joblib.
        config: FakeDetectorConfig with thresholds.
    """

    def __init__(self, model_path: Path, config: FakeDetectorConfig) -> None:
        self._model_path = Path(model_path)
        self.config = config
        logger.info("Loading model from %s", self._model_path)
        self._model = joblib.load(self._model_path)

    def score_dataset(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        """Run model inference and attach scoring columns.

        Args:
            df: DataFrame containing text_col and star_rating.
            text_col: Column name with review text.

        Returns:
            Copy of df with predicted_star, predicted_confidence,
            mismatch, suspicion, fake_category columns added.
        """
        texts = df[text_col].fillna("").astype(str).tolist()
        logger.info("Scoring %d reviews…", len(texts))
        predicted_star = self._model.predict(texts)
        predicted_proba = self._model.predict_proba(texts)
        scored = score_dataframe(
            df,
            predicted_star,
            predicted_proba,
            self.config.confidence_threshold,
        )
        logger.info("Scoring done. Category counts:\n%s", scored["fake_category"].value_counts().to_string())
        return scored

    def find_fakes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return suspicious and very_suspicious rows sorted by suspicion desc.

        Args:
            df: DataFrame already scored by score_dataset().

        Returns:
            Filtered DataFrame with fake_category in {suspicious, very_suspicious}.
        """
        mask = df["fake_category"].isin({"suspicious", "very_suspicious"})
        return df[mask].sort_values("suspicion", ascending=False).reset_index(drop=True)
