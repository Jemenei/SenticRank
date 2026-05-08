"""Stratified train/val/test splitter."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _make_strat_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].astype(str).agg("_".join, axis=1)


def split_dataset(
    df: pd.DataFrame,
    interim_dir: Path | str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_columns: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split and persist to parquet.

    Args:
        df: Full dataset.
        interim_dir: Output directory for split files.
        test_size: Fraction held out for test.
        val_size: Fraction of remaining data held out for val.
        stratify_columns: Columns to stratify on.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    strat_cols = stratify_columns or ["star_rating"]
    strat_key = _make_strat_key(df, strat_cols)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=strat_key,
        random_state=seed,
    )
    strat_key_tv = _make_strat_key(train_val, strat_cols)
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=strat_key_tv,
        random_state=seed,
    )

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = interim_dir / f"{split_name}.csv"
        split_df.to_csv(out, index=False)
        logger.info("Saved %s → %d rows → %s", split_name, len(split_df), out)

    logger.info(
        "Split complete: train=%d val=%d test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test
