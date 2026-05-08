from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .metrics import compute_all_metrics
from .trainer import _load_split

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config) -> None:
        self.cfg = config
        self.out_dir = Path(config.star_predictor.output_model_dir)

    def evaluate_on_test(self) -> pd.DataFrame:
        test_path = str(Path(self.cfg.data.interim_dir) / "test.csv")
        test_texts, test_labels, _ = _load_split(test_path)

        all_metrics: dict = {}
        model_files = sorted(self.out_dir.glob("*.joblib"))
        model_files = [f for f in model_files if f.stem != "best_model"]

        for model_path in model_files:
            name = model_path.stem
            predictor = joblib.load(model_path)
            preds = predictor.predict(test_texts)
            metrics = compute_all_metrics(test_labels, preds)
            all_metrics[name] = metrics
            logger.info(
                "%s test — f1_macro=%.4f  mae=%.4f  qwk=%.4f  acc=%.4f",
                name, metrics["f1_macro"], metrics["mae"], metrics["qwk"], metrics["accuracy"],
            )

        best_model_txt = self.out_dir / "best_model.txt"
        best_name = best_model_txt.read_text().strip() if best_model_txt.exists() else ""
        best_f1 = all_metrics.get(best_name, {}).get("f1_macro", 0.0)

        output = dict(all_metrics)
        output["best_model"] = best_name
        output["best_f1_macro"] = best_f1

        (self.out_dir / "test_metrics.json").write_text(json.dumps(output, indent=2))

        rows = []
        for name, m in all_metrics.items():
            rows.append({
                "model": name,
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"],
                "mae": m["mae"],
                "qwk": m["qwk"],
            })
        df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
        df.to_csv(self.out_dir / "test_metrics.csv", index=False)

        for name, m in all_metrics.items():
            cm_path = self.out_dir / f"{name}_confusion_matrix.json"
            cm_path.write_text(json.dumps(m["confusion_matrix"], indent=2))

        return df
