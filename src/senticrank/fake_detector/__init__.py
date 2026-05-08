"""Fake review detection module."""

from .detector import FakeReviewDetector
from .scoring import score_dataframe, assign_fake_category, compute_mismatch, compute_suspicion
from .analysis import summarize_fakes, find_patterns, top_fake_examples
from .filter import filter_dataset

__all__ = [
    "FakeReviewDetector",
    "score_dataframe",
    "assign_fake_category",
    "compute_mismatch",
    "compute_suspicion",
    "summarize_fakes",
    "find_patterns",
    "top_fake_examples",
    "filter_dataset",
]
