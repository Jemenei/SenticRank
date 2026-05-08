"""Statistical analysis of detected fake reviews."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_FAKE_CATS = ("suspicious", "very_suspicious")


def summarize_fakes(df: pd.DataFrame) -> dict:
    """High-level count/percentage breakdown of fake categories.

    Args:
        df: Scored DataFrame with fake_category column.

    Returns:
        Dict with total, per-category counts/percentages, and
        suspicious_by_category / suspicious_by_star breakdowns.
    """
    total = len(df)
    counts = df["fake_category"].value_counts().to_dict()
    pct = {k: round(v / total * 100, 2) for k, v in counts.items()}

    fakes = df[df["fake_category"].isin(_FAKE_CATS)]

    by_category: dict = {}
    if "category" in df.columns and not fakes.empty:
        by_category = fakes["category"].value_counts().to_dict()

    by_star: dict = {}
    if not fakes.empty:
        by_star = fakes["star_rating"].value_counts().sort_index().to_dict()

    return {
        "total_reviews": total,
        "category_counts": counts,
        "category_pct": pct,
        "suspicious_by_product_category": by_category,
        "suspicious_by_star_rating": {int(k): int(v) for k, v in by_star.items()},
    }


def find_patterns(df: pd.DataFrame, text_col: str = "full_text") -> dict:
    """Compare text length and helpful_votes between clean and fake reviews.

    Args:
        df: Scored DataFrame.
        text_col: Column with review text.

    Returns:
        Dict with avg_length comparison, helpful_votes comparison,
        and authors_with_multiple_fakes (if author_id column present).
    """
    df = df.copy()
    df["_text_len"] = df[text_col].fillna("").str.len()

    clean = df[df["fake_category"] == "clean"]
    fakes = df[df["fake_category"].isin(_FAKE_CATS)]

    avg_length = {
        "clean": round(clean["_text_len"].mean(), 1) if not clean.empty else 0.0,
        "fake": round(fakes["_text_len"].mean(), 1) if not fakes.empty else 0.0,
    }

    helpful_votes: dict = {}
    if "helpful_votes" in df.columns:
        helpful_votes = {
            "clean_mean": round(clean["helpful_votes"].mean(), 2) if not clean.empty else 0.0,
            "fake_mean": round(fakes["helpful_votes"].mean(), 2) if not fakes.empty else 0.0,
            "fake_zero_pct": round(
                (fakes["helpful_votes"] == 0).sum() / max(len(fakes), 1) * 100, 1
            ),
        }

    multi_fake_authors: dict = {}
    if "author_id" in df.columns and not fakes.empty:
        author_counts = fakes["author_id"].value_counts()
        multi = author_counts[author_counts >= 3].to_dict()
        multi_fake_authors = {str(k): int(v) for k, v in multi.items()}

    return {
        "avg_text_length": avg_length,
        "helpful_votes": helpful_votes,
        "authors_with_3plus_fakes": multi_fake_authors,
    }


def top_fake_examples(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top-n most suspicious rows for manual inspection.

    Args:
        df: Scored DataFrame.
        n: Number of rows to return.

    Returns:
        DataFrame with key columns sorted by suspicion desc.
    """
    fakes = df[df["fake_category"].isin(_FAKE_CATS)].sort_values("suspicion", ascending=False)
    cols = [
        c for c in [
            "product_id", "product_name", "category", "star_rating",
            "predicted_star", "predicted_confidence", "mismatch", "suspicion",
            "fake_category", "full_text",
        ]
        if c in df.columns
    ]
    return fakes[cols].head(n).reset_index(drop=True)
