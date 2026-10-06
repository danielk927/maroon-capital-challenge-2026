"""Validation: mock tournaments, the held-out comparison, and the counter check."""

from collections.abc import Mapping, Sequence

import numpy as np

from blotto.pool import Field, Group, jitter, random_splits
from blotto.robust import BOUNDS, bounds_arrays
from blotto.scoring import N_FIELDS, play
from blotto.search import metrics, rating, search

TOP_QUARTILE = 0.25


def sample_mix(
    names: Sequence[str],
    rng: np.random.Generator,
    bounds: Mapping[str, tuple[float, float]] = BOUNDS,
    max_tries: int = 100_000,
) -> np.ndarray:
    """A field mix drawn uniformly from {lo <= w <= hi, sum(w) = 1}, by rejection sampling."""
    lo, hi = bounds_arrays(names, bounds)
    if lo.sum() >= 1 - 1e-9:
        return lo  # the minimums already total 1, so they are the only feasible mix
    if hi.sum() <= 1 + 1e-9:
        return hi  # the maximums total exactly 1, so they are the only feasible mix
    for _ in range(max_tries):
        w = lo + (1 - lo.sum()) * rng.dirichlet(np.ones(len(names)))
        if (w <= hi + 1e-12).all():
            return w
    raise ValueError(f"weight bounds for {list(names)} leave too little room to sample mixes; widen them")


def _draw_field(
    groups: Mapping[str, Group], names: Sequence[str], w: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    counts = rng.multinomial(size, w)
    return np.vstack(
        [
            groups[g].entries[rng.choice(len(groups[g].entries), size=c, p=groups[g].weights)]
            for g, c in zip(names, counts)
            if c > 0
        ]
    )


def mock_tournaments(
    finalists: np.ndarray,
    groups: Mapping[str, Group],
    names: Sequence[str],
    rng: np.random.Generator,
    n: int = 1000,
    size_range: tuple[int, int] = (100, 300),
) -> dict[str, np.ndarray]:
    """Play each finalist into n random realistic fields and record where it ranks on wins and on points."""
    finalists = np.atleast_2d(np.asarray(finalists, dtype=np.int64))
    above_wins = np.empty((n, len(finalists)))
    above_points = np.empty((n, len(finalists)))
    for t in range(n):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        field = _draw_field(groups, names, sample_mix(names, rng), size, rng)
        mine, theirs = play(field, field)
        field_wins = (mine > theirs).sum(axis=1)
        field_points = mine.sum(axis=1)
        f_mine, f_theirs = play(finalists, field)
        for f in range(len(finalists)):
            wins = (f_mine[f] > f_theirs[f]).sum()
            points = f_mine[f].sum()
            above_wins[t, f] = ((field_wins + (f_theirs[f] > f_mine[f])) > wins).sum() / size
            above_points[t, f] = ((field_points + f_theirs[f]) > points).sum() / size
    return {
        "top_quartile_wins": (above_wins < TOP_QUARTILE).mean(axis=0),
        "top_quartile_points": (above_points < TOP_QUARTILE).mean(axis=0),
        "top_quartile_both": ((above_wins < TOP_QUARTILE) & (above_points < TOP_QUARTILE)).mean(axis=0),
        "share_above_wins": above_wins,
        "share_above_points": above_points,
    }


def heldout_report(
    pick: np.ndarray, seen: Field, heldout: Mapping[str, Field], lam: float = 0.5
) -> dict[str, dict[str, float]]:
    """Win rate, average points, and rating at lam of the pick on the seen field and each held-out field."""
    x = np.asarray(pick, dtype=np.int64)[None]
    out = {}
    for name, field in {"seen": seen, **heldout}.items():
        win_rate, avg_points = metrics(x, field)
        out[name] = {
            "win_rate": float(win_rate[0]),
            "avg_points": float(avg_points[0]),
            "rating": float(rating(x, field, lam)[0]),
        }
    return out


def one_troop_moves(pick: np.ndarray, field: Field) -> list[dict]:
    """For each battlefield with troops, the best win rate against the field after moving one of them elsewhere."""
    pick = np.asarray(pick, dtype=np.int64)
    out = []
    for i in np.flatnonzero(pick):
        moved = np.repeat(pick[None], N_FIELDS, axis=0)
        moved[:, i] -= 1
        moved[np.arange(N_FIELDS), np.arange(N_FIELDS)] += 1  # row j moves the troop to battlefield j + 1
        targets = np.delete(np.arange(N_FIELDS), i)
        win_rate, avg_points = metrics(moved[targets], field)
        best = int(np.argmax(win_rate))
        out.append(
            {
                "from": int(i) + 1,
                "to": int(targets[best]) + 1,
                "win_rate": float(win_rate[best]),
                "avg_points": float(avg_points[best]),
            }
        )
    return out


def counter_check(
    pick: np.ndarray,
    pool_field: Field,
    rng: np.random.Generator,
    chains: int = 32,
    steps: int = 3000,
    copies: int = 50,
) -> dict:
    """The strongest counter to the pick, how much of the pool sits near it, and how the pick fares against it."""
    x = np.asarray(pick, dtype=np.int64)[None]
    target = Field(x, np.ones(1))
    X, _ = search(lambda Z: rating(Z, target, 0.5), random_splits(chains, rng), rng, steps)
    counter = X[0]
    near = np.abs(pool_field.entries - counter).sum(axis=1) <= 10
    family = np.vstack([counter[None], jitter(np.repeat(counter[None], copies, axis=0), rng)])
    win_rate, avg_points = metrics(x, Field(family, np.full(len(family), 1 / len(family))))
    return {
        "counter": counter.tolist(),
        "pool_share_within_10": float(pool_field.weights[near].sum()),
        "pick_win_rate_vs_counter_family": float(win_rate[0]),
        "pick_avg_points_vs_counter_family": float(avg_points[0]),
    }
