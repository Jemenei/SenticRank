"""Pytest fixtures for fast unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

RAW_PATH = Path("data/raw/senticrank_master_dataset.csv")


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """200-row sample of the master dataset for fast tests."""
    df = pd.read_csv(RAW_PATH, low_memory=False)
    return df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
