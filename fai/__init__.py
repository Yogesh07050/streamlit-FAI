"""Reconstructed FAI (Flourishing AI) scoring, runnable locally."""

from .dimensions import DIMENSIONS, BY_KEY, KEYS, ROBUSTNESS_THRESHOLD, PUBLISHED_GC
from .scoring import (
    objective_score,
    subjective_score,
    tangential_score,
    composite_dimension,
    fai_score,
    arithmetic_mean,
    spread_penalty,
    gap_to_threshold,
    marginal_returns,
    uplift_if_raised,
    DimensionBreakdown,
    FaiResult,
    DEFAULT_WEIGHTS,
)

__all__ = [
    "DIMENSIONS", "BY_KEY", "KEYS", "ROBUSTNESS_THRESHOLD", "PUBLISHED_GC",
    "objective_score", "subjective_score", "tangential_score",
    "composite_dimension", "fai_score", "arithmetic_mean", "spread_penalty",
    "gap_to_threshold", "marginal_returns", "uplift_if_raised",
    "DimensionBreakdown", "FaiResult", "DEFAULT_WEIGHTS",
]
