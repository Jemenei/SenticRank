"""Dataset loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "review_text",
    "full_text",
    "star_rating",
}


def load_master_dataset(path: Path | str) -> pd.DataFrame:
    """Load and validate master dataset, return only needed columns.

    Args:
        path: Path to senticrank_master_dataset.csv.

    Returns:
        DataFrame with validated schema.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["star_rating"] = df["star_rating"].astype(int)
    df["full_text"] = df["full_text"].fillna("").astype(str)

    logger.debug("Schema OK. Sample:\n%s", df.sample(min(3, len(df))).to_string())
    return df
