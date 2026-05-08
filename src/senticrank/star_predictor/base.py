from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BasePredictor(ABC):
    name: str

    @abstractmethod
    def fit(self, texts: list[str], labels: np.ndarray, sample_weight: np.ndarray | None = None) -> None: ...

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BasePredictor": ...
