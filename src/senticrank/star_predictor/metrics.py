from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    mean_absolute_error,
)


def compute_all_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Primary: f1_macro. Secondary: mae. Tertiary: qwk."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true) | set(y_pred))

    f1_per = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    cm = sk_confusion_matrix(y_true, y_pred, labels=classes)
    cm_dict = {
        int(c): {int(classes[j]): int(cm[i, j]) for j in range(len(classes))}
        for i, c in enumerate(classes)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_per_class": {int(c): float(v) for c, v in zip(classes, f1_per)},
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix": cm_dict,
    }
