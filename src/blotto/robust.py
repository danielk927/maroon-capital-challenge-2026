"""Robust selection: rate candidates per group and keep the best worst case over plausible field mixes."""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from blotto.pool import Field, Group, random_splits, sample_field
from blotto.scoring import MAX_SCORE
from blotto.search import metrics, rating, search

# Chatbot answers make up 70-95% of every field: nearly all applicants are expected to ask an LLM.
# The other maximums total less than 1, so the pipeline requires chatbot answers.
BOUNDS: dict[str, tuple[float, float]] = {
    "riddler": (0.00, 0.20),
    "heuristics": (0.00, 0.15),
    "chatbots": (0.70, 0.95),
    "random": (0.00, 0.05),
    "counters": (0.05, 0.15),
}
LAM_RANGE = (0.25, 0.75)
LAMS = (0.25, 0.5, 0.75)


def group_names(groups: Mapping[str, Group]) -> list[str]:
    """Names of the groups present, in BOUNDS order."""
    return [g for g in BOUNDS if g in groups]


def bounds_arrays(
    names: Sequence[str], bounds: Mapping[str, tuple[float, float]] = BOUNDS
) -> tuple[np.ndarray, np.ndarray]:
    """Min and max weight per group; raises ValueError for a minimum above its maximum or when no mix sums to 1."""
    lo = np.array([bounds[g][0] for g in names])
    hi = np.array([bounds[g][1] for g in names])
    inverted = [g for g, low, high in zip(names, lo, hi) if low > high]
    if inverted:
        raise ValueError(f"weight bounds have a minimum above the maximum for {inverted}")
    if lo.sum() > 1 + 1e-9 or hi.sum() < 1 - 1e-9:
        raise ValueError(
            f"weight bounds for {list(names)} cannot sum to 1 (minimums total {lo.sum():.2f}, maximums total {hi.sum():.2f})"
        )
    return lo, hi


def center_mix(names: Sequence[str]) -> dict[str, float]:
    """The midpoint of each group's bounds, normalized to sum to 1."""
    lo, hi = bounds_arrays(names)
    mid = (lo + hi) / 2
    return dict(zip(names, (mid / mid.sum()).tolist()))


def vertex_mixes(names: Sequence[str]) -> list[dict[str, float]]:
    """Extreme points of {lo <= w <= hi, sum(w) = 1}: every group at a bound except at most one."""
    lo, hi = bounds_arrays(names)
    found = set()
    for free in range(len(names)):
        others = [g for g in range(len(names)) if g != free]
        for at_max in itertools.product([False, True], repeat=len(others)):
            w = lo.copy()
            for g, up in zip(others, at_max):
                w[g] = hi[g] if up else lo[g]
            w[free] = 1 - w[others].sum()
            if lo[free] - 1e-9 <= w[free] <= hi[free] + 1e-9:
                found.add(tuple(float(v) for v in np.round(w, 9)))
    return [dict(zip(names, w)) for w in sorted(found)]


def worst_case_value(ratings: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per row, the minimum of ratings @ w over weights lo <= w <= hi summing to 1 (greedy fill)."""
    order = np.argsort(ratings, axis=1)
    w = np.tile(lo, (len(ratings), 1))
    remaining = np.full(len(ratings), 1 - lo.sum())
    rows = np.arange(len(ratings))
    for position in range(ratings.shape[1]):
        g = order[:, position]
        add = np.minimum(hi[g] - lo[g], remaining)
        w[rows, g] += add
        remaining -= add
    return (ratings * w).sum(axis=1)


def worst_case_rating(
    win: np.ndarray, pts: np.ndarray, lo: np.ndarray, hi: np.ndarray, lam_range: tuple[float, float] = LAM_RANGE
) -> np.ndarray:
    """Worst rating over mixes within the bounds and lam within lam_range; win and pts are (K, G)."""
    return np.minimum.reduce(
        [worst_case_value(lam * win + (1 - lam) * pts / MAX_SCORE, lo, hi) for lam in lam_range]
    )


def group_metrics(X: np.ndarray, groups: Mapping[str, Group], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Win rate and average points of each row of X against each named group, as (K, G) arrays."""
    results = [metrics(X, Field(groups[g].entries, groups[g].weights)) for g in names]
    return np.column_stack([r[0] for r in results]), np.column_stack([r[1] for r in results])


def generate_candidates(
    groups: Mapping[str, Group],
    names: Sequence[str],
    rng: np.random.Generator,
    chains: int = 32,
    steps: int = 3000,
    sample_size: int = 2000,
    seeds_per_setting: int = 1,
) -> np.ndarray:
    """Distinct local optima from searches under the center mix and every extreme mix, for each lam in LAMS."""
    found = []
    for mix in [center_mix(names), *vertex_mixes(names)]:
        for lam in LAMS:
            for _ in range(seeds_per_setting):
                field = sample_field(groups, mix, sample_size, rng)
                X, _ = search(lambda Z, f=field, l=lam: rating(Z, f, l), random_splits(chains, rng), rng, steps)
                found.append(X)
    return np.unique(np.vstack(found), axis=0)


def robust_search(
    groups: Mapping[str, Group],
    names: Sequence[str],
    starts: np.ndarray,
    rng: np.random.Generator,
    steps: int = 3000,
    per_group_sample: int = 400,
) -> np.ndarray:
    """Distinct local optima of the worst-case rating, with each group represented by a sample."""
    lo, hi = bounds_arrays(names)
    samples = {g: sample_field({g: groups[g]}, {g: 1.0}, per_group_sample, rng) for g in names}

    def objective(Z: np.ndarray) -> np.ndarray:
        results = [metrics(Z, samples[g]) for g in names]
        win = np.column_stack([r[0] for r in results])
        pts = np.column_stack([r[1] for r in results])
        return worst_case_rating(win, pts, lo, hi)

    X, _ = search(objective, starts, rng, steps)
    return X


@dataclass(frozen=True)
class Ranked:
    """Candidates sorted by worst-case rating (best first), with per-group metrics on the full groups."""

    candidates: np.ndarray  # (K, 10)
    worst: np.ndarray  # (K,)
    win: np.ndarray  # (K, G)
    pts: np.ndarray  # (K, G)


def rank_candidates(
    candidates: np.ndarray, groups: Mapping[str, Group], names: Sequence[str], keep: int = 300
) -> Ranked:
    """The top `keep` distinct candidates by worst-case rating against the full groups."""
    X = np.unique(np.asarray(candidates, dtype=np.int64), axis=0)
    lo, hi = bounds_arrays(names)
    win, pts = group_metrics(X, groups, names)
    worst = worst_case_rating(win, pts, lo, hi)
    order = np.argsort(-worst, kind="stable")[:keep]
    return Ranked(X[order], worst[order], win[order], pts[order])
