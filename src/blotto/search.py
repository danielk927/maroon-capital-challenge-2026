"""Rating allocations against a weighted field, and searching for the best ones."""

from collections.abc import Callable

import numpy as np

from blotto.pool import Field
from blotto.scoring import MAX_SCORE, N_FIELDS, play

Objective = Callable[[np.ndarray], np.ndarray]


def metrics(X: np.ndarray, field: Field) -> tuple[np.ndarray, np.ndarray]:
    """Weighted win rate and average points (true score units) of each row of X against the field."""
    mine, theirs = play(X, field.entries)
    return (mine > theirs) @ field.weights, (mine @ field.weights) / 100.0


def rating(X: np.ndarray, field: Field, lam: float) -> np.ndarray:
    """lam * win rate + (1 - lam) * average points / 127.05, for each row of X."""
    win_rate, avg_points = metrics(X, field)
    return lam * win_rate + (1 - lam) * avg_points / MAX_SCORE


def _moves(max_shift: int) -> np.ndarray:
    moves = []
    for i in range(N_FIELDS):
        for j in range(N_FIELDS):
            if i != j:
                for k in range(1, max_shift + 1):
                    m = np.zeros(N_FIELDS, dtype=np.int64)
                    m[i], m[j] = -k, k
                    moves.append(m)
    return np.array(moves)


MOVES = _moves(3)


def neighbors(x: np.ndarray) -> np.ndarray:
    """Every allocation reachable by moving 1 to 3 troops from one battlefield to another."""
    nb = np.asarray(x, dtype=np.int64)[None, :] + MOVES
    return nb[(nb >= 0).all(axis=1)]


def polish(x: np.ndarray, objective: Objective) -> tuple[np.ndarray, float]:
    """Steepest ascent over neighbors until none improves; returns the local optimum and its value."""
    x = np.asarray(x, dtype=np.int64)
    current = float(objective(x[None])[0])
    while True:
        nb = neighbors(x)
        values = objective(nb)
        best = int(np.argmax(values))
        if values[best] <= current:
            return x, current
        x, current = nb[best], float(values[best])


def anneal(
    starts: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    steps: int = 3000,
    t_start: float = 0.05,
    t_end: float = 0.001,
    max_move: int = 5,
) -> np.ndarray:
    """Simulated annealing on every start in parallel; returns the best allocation each chain visited."""
    X = np.array(starts, dtype=np.int64, copy=True)
    current = objective(X)
    best_X, best = X.copy(), current.copy()
    chains = np.arange(len(X))
    for t in np.geomspace(t_start, t_end, steps):
        i = rng.integers(0, N_FIELDS, len(X))
        j = (i + rng.integers(1, N_FIELDS, len(X))) % N_FIELDS
        k = np.minimum(rng.integers(1, max_move + 1, len(X)), X[chains, i])
        proposal = X.copy()
        proposal[chains, i] -= k
        proposal[chains, j] += k
        values = objective(proposal)
        delta = values - current
        accept = (k > 0) & ((delta >= 0) | (rng.random(len(X)) < np.exp(np.minimum(delta, 0) / t)))
        X[accept] = proposal[accept]
        current[accept] = values[accept]
        improved = current > best
        best_X[improved] = X[improved]
        best[improved] = current[improved]
    return best_X


def search(
    objective: Objective, starts: np.ndarray, rng: np.random.Generator, steps: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Anneal then polish every start; returns distinct local optima sorted best first, with their values."""
    found: dict[tuple[int, ...], float] = {}
    for x in anneal(starts, objective, rng, steps):
        optimum, value = polish(x, objective)
        found[tuple(optimum.tolist())] = value
    ordered = sorted(found.items(), key=lambda kv: -kv[1])
    return np.array([k for k, _ in ordered], dtype=np.int64), np.array([v for _, v in ordered])
