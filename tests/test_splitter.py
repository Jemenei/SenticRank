"""Smoke tests for data loader and splitter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from senticrank.data.splitter import split_dataset


def test_split_sizes(sample_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train, val, test = split_dataset(
            sample_df,
            interim_dir=tmp,
            test_size=0.15,
            val_size=0.15,
            stratify_columns=["star_rating"],
            seed=42,
        )
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)


def test_split_writes_parquet(sample_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        split_dataset(
            sample_df,
            interim_dir=tmp,
            stratify_columns=["star_rating"],
        )
        for name in ("train", "val", "test"):
            assert (Path(tmp) / f"{name}.csv").exists()


def test_no_data_leakage(sample_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train, val, test = split_dataset(
            sample_df,
            interim_dir=tmp,
            stratify_columns=["star_rating"],
        )
        idx_sets = [set(train.index), set(val.index), set(test.index)]
        assert idx_sets[0].isdisjoint(idx_sets[1])
        assert idx_sets[0].isdisjoint(idx_sets[2])
        assert idx_sets[1].isdisjoint(idx_sets[2])
