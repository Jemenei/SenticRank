from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .dummy import DummyPredictor
from .metrics import compute_all_metrics
from .tfidf_logreg import TfidfLogregPredictor
from .tfidf_svm import TfidfLinearSVMPredictor

logger = logging.getLogger(__name__)

_PREDICTOR_REGISTRY = {
    "dummy": DummyPredictor,
    "tfidf_logreg": TfidfLogregPredictor,
    "tfidf_svm": TfidfLinearSVMPredictor,
}


def _resolve_texts(df: pd.DataFrame) -> tuple[list[str], int]:
    texts = []
    fallbacks = 0
    for _, row in df.iterrows():
        ft = str(row.get("full_text", "") or "")
        if len(ft) >= 5:
            texts.append(ft)
            continue
        rt = str(row.get("review_text", "") or "")
        if len(rt) >= 5:
            texts.append(rt)
            fallbacks += 1
            continue
        pros = str(row.get("pros", "") or "")
        cons = str(row.get("cons", "") or "")
        texts.append((pros + " " + cons).strip())
        fallbacks += 1
    return texts, fallbacks


def _load_split(path: str) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["star_rating"])
    df["star_rating"] = df["star_rating"].astype(int)
    texts, fallbacks = _resolve_texts(df)
    logger.info("Loaded %s: %d rows, %d text fallbacks", path, len(df), fallbacks)
    labels = df["star_rating"].values
    weights = df["trust_weight"].values if "trust_weight" in df.columns else None
    return texts, labels, weights


class Trainer:
    def __init__(self, config) -> None:
        self.cfg = config
        self.sp_cfg = config.star_predictor
        self.out_dir = Path(self.sp_cfg.output_model_dir)

    def train_all(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        interim = Path(self.cfg.data.interim_dir)

        train_texts, train_labels, train_weights = _load_split(str(interim / "train.csv"))
        val_texts, val_labels, _ = _load_split(str(interim / "val.csv"))

        use_weights = getattr(self.sp_cfg, "use_trust_weight_as_sample_weight", True)
        if use_weights and train_weights is not None:
            mean_w = train_weights.mean()
            sample_weight = train_weights / mean_w if mean_w > 0 else None
        else:
            sample_weight = None

        results = {}
        for name in self.sp_cfg.models_to_train:
            cls = _PREDICTOR_REGISTRY.get(name)
            if cls is None:
                logger.warning("Unknown model '%s', skipping", name)
                continue

            logger.info("Training %s …", name)
            predictor = cls() if name == "dummy" else cls(self.sp_cfg)
            predictor.fit(train_texts, train_labels, sample_weight=sample_weight)

            val_pred = predictor.predict(val_texts)
            metrics = compute_all_metrics(val_labels, val_pred)
            results[name] = metrics

            save_path = self.out_dir / f"{name}.joblib"
            predictor.save(save_path)
            logger.info(
                "%s val — f1_macro=%.4f  mae=%.4f  qwk=%.4f  acc=%.4f",
                name, metrics["f1_macro"], metrics["mae"], metrics["qwk"], metrics["accuracy"],
            )

        best_name = max(results, key=lambda n: results[n]["f1_macro"])
        best_f1 = results[best_name]["f1_macro"]
        import shutil
        shutil.copy(self.out_dir / f"{best_name}.joblib", self.out_dir / "best_model.joblib")
        (self.out_dir / "best_model.txt").write_text(best_name)
        logger.info("Best model: %s (val f1_macro=%.4f)", best_name, best_f1)

        return results
