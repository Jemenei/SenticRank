"""SenticRank ranking engine."""

from .aggregation import TrustWeightedAggregator
from .fuzzy import FuzzyTransformer, trapezoidal, s_curve
from .ahp import AHPCalculator
from .topsis import TOPSISRanker
from .confidence import apply_confidence_discount
from .system import SenticRankSystem

__all__ = [
    "TrustWeightedAggregator",
    "FuzzyTransformer",
    "trapezoidal",
    "s_curve",
    "AHPCalculator",
    "TOPSISRanker",
    "apply_confidence_discount",
    "SenticRankSystem",
]
