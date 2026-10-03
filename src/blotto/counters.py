"""Counter loop: simulated applicants who best-respond to the rest of the field (fictitious play)."""

from collections.abc import Mapping

import numpy as np

from blotto.pool import Group, make_group, random_splits, sample_field
from blotto.scoring import N_FIELDS
from blotto.search import rating, search


def pick_distinct(X: np.ndarray, k: int, min_l1: int) -> np.ndarray:
    """Up to k rows of X (sorted best first), each at least min_l1 troops (L1) from those already picked."""
    chosen: list[np.ndarray] = []
    for x in X:
        if all(np.abs(x - c).sum() >= min_l1 for c in chosen):
            chosen.append(x)
            if len(chosen) == k:
                break
    return np.array(chosen, dtype=np.int64).reshape(-1, N_FIELDS)


def counter_loop(
    groups: Mapping[str, Group],
    mix: Mapping[str, float],
    rng: np.random.Generator,
    rounds: int = 6,
    per_round: int = 5,
    min_l1: int = 6,
    chains: int = 32,
    steps: int = 3000,
    sample_size: int = 2000,
    lam: float = 0.5,
) -> np.ndarray:
    """Allocations found by repeatedly best-responding to the pool plus every counter found so far."""
    counters = np.empty((0, N_FIELDS), dtype=np.int64)
    for _ in range(rounds):
        current = dict(groups)
        if len(counters):
            current["counters"] = make_group(counters, rng)
        field = sample_field(current, mix, sample_size, rng)
        X, _ = search(lambda Z, f=field: rating(Z, f, lam), random_splits(chains, rng), rng, steps)
        counters = np.vstack([counters, pick_distinct(X, per_round, min_l1)])
    return counters
