"""Standalone script: load master dataset and produce stratified splits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senticrank.config import load_config
from senticrank.data.loader import load_master_dataset
from senticrank.data.splitter import split_dataset
from senticrank.logging_setup import setup_logging


def main() -> None:
    setup_logging()
    cfg = load_config("configs/default.yaml")
    df = load_master_dataset(cfg.data.raw_path)
    split_dataset(
        df,
        interim_dir=cfg.data.interim_dir,
        test_size=cfg.split.test_size,
        val_size=cfg.split.val_size,
        stratify_columns=cfg.split.stratify_by,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    main()
