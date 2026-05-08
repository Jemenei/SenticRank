"""Dataset filtering: remove suspicious reviews to produce a clean dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REMOVE_CONSERVATIVE = {"very_suspicious"}
_REMOVE_AGGRESSIVE = {"suspicious", "very_suspicious"}


def filter_dataset(
    df: pd.DataFrame,
    processed_dir: str | Path,
    mode: str = "conservative",
) -> dict[str, pd.DataFrame]:
    """Split scored df into clean and fake subsets and persist to disk.

    Args:
        df: DataFrame with fake_category column (output of score_dataset).
        processed_dir: Directory for output CSV files.
        mode: 'conservative' removes very_suspicious only;
              'aggressive' removes suspicious + very_suspicious.

    Returns:
        Dict with keys 'all', 'clean', 'fakes' mapping to DataFrames.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if mode == "conservative":
        remove_cats = _REMOVE_CONSERVATIVE
    elif mode == "aggressive":
        remove_cats = _REMOVE_AGGRESSIVE
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'conservative' or 'aggressive'.")

    mask_fake = df["fake_category"].isin(remove_cats)
    df_clean = df[~mask_fake].reset_index(drop=True)
    df_fakes = df[mask_fake].sort_values("suspicion", ascending=False).reset_index(drop=True)

    all_path = processed_dir / "dataset_with_fake_scores.csv"
    clean_path = processed_dir / "dataset_clean.csv"
    fakes_path = processed_dir / "dataset_fakes.csv"

    df.to_csv(all_path, index=False)
    df_clean.to_csv(clean_path, index=False)
    df_fakes.to_csv(fakes_path, index=False)

    logger.info(
        "Filtered (mode=%s): %d clean, %d removed → saved to %s",
        mode, len(df_clean), len(df_fakes), processed_dir,
    )
    return {"all": df, "clean": df_clean, "fakes": df_fakes}
