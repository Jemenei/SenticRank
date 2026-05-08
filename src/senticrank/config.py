"""Pydantic-based config loader for default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TfidfConfig(BaseModel):
    max_features: int = 50000
    ngram_range: list[int] = [1, 2]
    min_df: int = 3
    sublinear_tf: bool = True


class LogRegConfig(BaseModel):
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str = "balanced"


class SvmConfig(BaseModel):
    C: float = 1.0
    class_weight: str = "balanced"
    max_iter: int = 2000


class FastTextConfig(BaseModel):
    epoch: int = 25
    lr: float = 0.5
    word_ngrams: int = 2
    dim: int = 100


class StarPredictorConfig(BaseModel):
    models_to_train: list[str] = ["dummy", "tfidf_logreg", "tfidf_svm"]
    primary_metric: str = "f1_macro"
    use_trust_weight_as_sample_weight: bool = True
    tfidf: TfidfConfig = Field(default_factory=TfidfConfig)
    logreg: LogRegConfig = Field(default_factory=LogRegConfig)
    svm: SvmConfig = Field(default_factory=SvmConfig)
    fasttext: FastTextConfig = Field(default_factory=FastTextConfig)
    output_model_dir: str = "data/outputs/star_predictor"


class FakeDetectorConfig(BaseModel):
    mismatch_threshold: float = 2.0
    use_confidence: bool = True
    confidence_threshold: float = 0.6
    output_dir: str = "data/outputs/fake_detector"


class QualityTrapezoids(BaseModel):
    low: list[float] = [0.0, 0.0, 0.25, 0.50]
    medium: list[float] = [0.25, 0.375, 0.625, 0.75]
    high: list[float] = [0.50, 0.75, 1.0, 1.0]


class FuzzyConfig(BaseModel):
    quality_trapezoids: QualityTrapezoids = Field(default_factory=QualityTrapezoids)
    consensus_s_curve: list[float] = [0.3, 0.8]
    dispute_s_curve: list[float] = [0.1, 0.5]


class ConfidenceDiscountConfig(BaseModel):
    method: str = "bayesian_shrinkage"
    prior_score: float = 50.0
    smoothing_k: int = 5


class RankingConfig(BaseModel):
    fake_categories_to_remove: list[str] = ["suspicious", "very_suspicious"]
    criteria: list[str] = ["agg_norm_rating", "fuzzy_quality", "fuzzy_consensus", "fuzzy_dispute", "fuzzy_engagement"]
    fuzzy: FuzzyConfig = Field(default_factory=FuzzyConfig)
    ahp_matrix: list[list[float]] = Field(default_factory=lambda: [
        [1.0,  2.0,  3.0,  4.0,  5.0],
        [0.5,  1.0,  2.0,  3.0,  4.0],
        [0.333, 0.5, 1.0,  2.0,  3.0],
        [0.25, 0.333, 0.5, 1.0,  2.0],
        [0.2,  0.25, 0.333, 0.5, 1.0],
    ])
    confidence_discount: ConfidenceDiscountConfig = Field(default_factory=ConfidenceDiscountConfig)
    output_dir: str = "data/outputs/ranking"


class DataConfig(BaseModel):
    raw_path: str = "data/raw/senticrank_master_dataset.csv"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    outputs_dir: str = "data/outputs"
    text_column: str = "full_text"
    label_column: str = "star_rating"
    group_column: str = "category"


class SplitConfig(BaseModel):
    test_size: float = 0.15
    val_size: float = 0.15
    stratify_by: list[str] = ["star_rating", "category"]


class AppConfig(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    star_predictor: StarPredictorConfig = Field(default_factory=StarPredictorConfig)
    fake_detector: FakeDetectorConfig = Field(default_factory=FakeDetectorConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)


def load_config(path: Path | str = "configs/default.yaml") -> AppConfig:
    """Load YAML config and parse into AppConfig.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated AppConfig instance.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
