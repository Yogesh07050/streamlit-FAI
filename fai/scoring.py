"""Reconstructed FAI scoring mathematics.

PROVENANCE -- what is published vs. what is reconstructed here
=============================================================
Published by Gloo (verbatim, load-bearing):
  * The seven dimensions and their symbols.
  * The top-level aggregation is a GEOMETRIC MEAN, chosen explicitly so that
        FAI = 7-th root of (chi * rho * eta * mu * psi * phi * sigma)
    is non-compensatory: strength in one dimension cannot offset weakness in
    another.
  * Each dimension score is built from three components -- Objective (accuracy
    against an answer key), Subjective (judge agreement on scenario responses)
    and Tangential (cross-dimensional relevance).
  * The tangential score is defined as  TS = sum(R) / sum(I).
  * Subjective alignment "receives greatest weight", because worldview
    coherence shows up in interpretive framing rather than factual recall.
  * Responses are judged by ALL seven dimension personas simultaneously; the
    non-primary personas are what produce tangential credit.
  * Scores are normalised to 0-100. The robustness threshold is 90.

Reconstructed (our choice, made explicit so it can be argued with):
  * The exact O/S/T -> dimension composite is not published. Gloo's developer
    docs describe a geometric mean across the three; the framework page leaves
    the formula unstated. We implement BOTH a weighted arithmetic and a
    weighted geometric composite and default to weights (0.30, 0.50, 0.20),
    honouring "subjective weighted most".
  * The meaning of R and I in TS = sum(R)/sum(I). We read I as the
    *opportunity* indicator (dimension d was a legitimate consideration for an
    item whose primary dimension was not d) and R as the *credit* awarded
    (0..1) for actually engaging it. So the tangential score is the fraction of
    available cross-dimensional opportunity that the model spontaneously took.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .dimensions import DIMENSIONS, KEYS, ROBUSTNESS_THRESHOLD

# Default composite weights: subjective carries the most weight, per the paper.
DEFAULT_WEIGHTS: dict[str, float] = {"objective": 0.30, "subjective": 0.50, "tangential": 0.20}


# --------------------------------------------------------------------------
# Component scores (all return 0-100)
# --------------------------------------------------------------------------

def objective_score(correct_flags: list[bool]) -> float:
    """Mean accuracy over objective items with a known answer key."""
    if not correct_flags:
        return float("nan")
    return 100.0 * sum(1 for c in correct_flags if c) / len(correct_flags)


def subjective_score(judgements: list[float]) -> float:
    """Mean of judge ratings in [0, 1] over subjective items.

    `judgements` is flattened over (item, judge): the published method takes the
    mean across evaluators AND across judgement-based questions, and a mean of
    means over equal-sized groups is the mean of the flattened pool.
    """
    if not judgements:
        return float("nan")
    return 100.0 * sum(judgements) / len(judgements)


def tangential_score(credits: list[float], opportunities: list[float]) -> float:
    """TS = sum(R) / sum(I), expressed 0-100.

    `opportunities[i]` (I) is 1.0 when dimension d was a legitimate but
    non-primary consideration for item i; `credits[i]` (R) in [0, 1] is how well
    the response actually engaged d there.
    """
    total_i = sum(opportunities)
    if total_i <= 0:
        return float("nan")
    return 100.0 * sum(credits) / total_i


# --------------------------------------------------------------------------
# Dimension composite
# --------------------------------------------------------------------------

def composite_dimension(
    objective: float,
    subjective: float,
    tangential: float,
    weights: dict[str, float] | None = None,
    mode: str = "arithmetic",
) -> float:
    """Combine the three component scores into one dimension score (0-100).

    Components that are NaN (e.g. a dimension with no objective items) are
    dropped and the remaining weights renormalised, so a missing component never
    silently scores zero.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    parts = {"objective": objective, "subjective": subjective, "tangential": tangential}
    live = {k: v for k, v in parts.items() if v is not None and not math.isnan(v)}
    if not live:
        return float("nan")
    wsum = sum(w[k] for k in live)
    if wsum <= 0:
        return float("nan")

    if mode == "geometric":
        # Weighted geometric mean. A true zero in any live component zeroes the
        # dimension -- the same non-compensatory logic as the top-level formula.
        if any(v <= 0 for v in live.values()):
            return 0.0
        log_sum = sum(w[k] * math.log(v) for k, v in live.items())
        return math.exp(log_sum / wsum)

    return sum(w[k] * v for k, v in live.items()) / wsum


# --------------------------------------------------------------------------
# Top-level FAI score -- the published geometric mean
# --------------------------------------------------------------------------

def fai_score(dimension_scores: dict[str, float], floor: float = 0.0) -> float:
    """FAI = 7th root of the product of the seven dimension scores.

    `floor` clamps each dimension up to at least `floor` before multiplying.
    With floor=0 (the default, and the faithful reading) a single zeroed
    dimension collapses the whole score to zero.
    """
    vals = []
    for k in KEYS:
        v = dimension_scores[k]
        if math.isnan(v):
            raise ValueError(f"dimension {k!r} has no score; FAI needs all seven")
        vals.append(max(v, floor))
    if any(v <= 0 for v in vals):
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def arithmetic_mean(dimension_scores: dict[str, float]) -> float:
    """The compensatory average FAI deliberately rejects -- shown for contrast."""
    vals = [dimension_scores[k] for k in KEYS]
    return sum(vals) / len(vals)


def gap_to_threshold(score: float) -> float:
    """Points short of the published 90-point robustness threshold."""
    return ROBUSTNESS_THRESHOLD - score


# --------------------------------------------------------------------------
# Why the geometric mean changes incentives -- the provable part
# --------------------------------------------------------------------------

def marginal_returns(dimension_scores: dict[str, float]) -> dict[str, float]:
    """d(FAI)/d(F_d) for each dimension.

    Because FAI = (prod F_d)^(1/7), differentiating gives

        d FAI / d F_d = FAI / (7 * F_d)

    FAI is common to all seven, so the derivative is inversely proportional to
    the dimension's own score: **the weakest dimension always has the largest
    marginal absolute return.** That is the whole point of the geometric mean.
    Note the *relative* elasticity is a flat 1/7 for every dimension
    (d ln FAI / d ln F_d = 1/7) -- it is only in absolute points that fixing the
    weak dimension wins.
    """
    fai = fai_score(dimension_scores)
    n = len(KEYS)
    out: dict[str, float] = {}
    for k in KEYS:
        f = dimension_scores[k]
        out[k] = fai / (n * f) if f > 0 else float("inf")
    return out


def uplift_if_raised(dimension_scores: dict[str, float], key: str, delta: float) -> float:
    """Exact FAI gain from adding `delta` points to one dimension."""
    before = fai_score(dimension_scores)
    bumped = dict(dimension_scores)
    bumped[key] = min(100.0, bumped[key] + delta)
    return fai_score(bumped) - before


def spread_penalty(dimension_scores: dict[str, float]) -> float:
    """Arithmetic mean minus geometric mean: the points FAI withholds for imbalance.

    Zero only when all seven dimensions are equal; grows with their variance
    (AM >= GM, by the AM-GM inequality).
    """
    return arithmetic_mean(dimension_scores) - fai_score(dimension_scores)


# --------------------------------------------------------------------------
# A recorded, inspectable run -- this is what the UI renders as "the working"
# --------------------------------------------------------------------------

@dataclass
class DimensionBreakdown:
    key: str
    objective: float
    subjective: float
    tangential: float
    composite: float
    n_objective: int = 0
    n_subjective: int = 0
    n_opportunities: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class FaiResult:
    dimensions: dict[str, DimensionBreakdown]
    composite_mode: str
    weights: dict[str, float]

    @property
    def dimension_scores(self) -> dict[str, float]:
        return {k: b.composite for k, b in self.dimensions.items()}

    @property
    def score(self) -> float:
        return fai_score(self.dimension_scores)

    @property
    def naive_mean(self) -> float:
        return arithmetic_mean(self.dimension_scores)

    @property
    def penalty(self) -> float:
        return spread_penalty(self.dimension_scores)

    @property
    def gap(self) -> float:
        return gap_to_threshold(self.score)

    def weakest(self) -> str:
        return min(KEYS, key=lambda k: self.dimension_scores[k])

    def formula_string(self) -> str:
        """The published formula with this run's numbers substituted in."""
        syms = " x ".join(
            f"{d.symbol}={self.dimension_scores[d.key]:.1f}" for d in DIMENSIONS
        )
        return f"FAI = 7-th root of ( {syms} ) = {self.score:.2f}"
