"""Exact scoring for the Maroon Capital 2026 Blotto variant.

Battlefield v (1 to 10) is array index v - 1 and bit v - 1 of a win mask.
Scores are integers equal to the true score times 100, so ties compare exactly.
"""

from collections.abc import Iterable

import numpy as np

N_FIELDS = 10
BUDGET = 100
MAX_SCORE = 127.05
BITS = (1 << np.arange(N_FIELDS)).astype(np.int64)


def score_x100(won: Iterable[int]) -> int:
    """Total score times 100 for winning the battlefields with the given values."""
    won = set(won)
    if not won:
        return 0
    base = sum(won)
    adjacent = sum(v for v in won if v - 1 in won or v + 1 in won)
    total = 2 * (base - adjacent) + 3 * adjacent  # Rule 1, scaled by 2
    total *= 7 if base % 5 == 0 else 5  # Rule 2, scaled by 5 more
    total *= 11 if 6 in won else 10  # Rule 3, scaled by 10 more
    return total


def mask_of(won: Iterable[int]) -> int:
    """Win mask for a set of battlefield values."""
    return sum(1 << (v - 1) for v in set(won))


SCORE_TABLE = np.array(
    [score_x100(v + 1 for v in range(N_FIELDS) if mask >> v & 1) for mask in range(1 << N_FIELDS)],
    dtype=np.int64,
)


def play(X: np.ndarray, Y: np.ndarray, chunk_pairs: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Scores times 100 for every allocation in X against every allocation in Y.

    X is (K, 10) and Y is (N, 10). Returns (mine, theirs), each (K, N) int64.
    Rows of X are processed in chunks of about chunk_pairs matchups to bound memory.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.int64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.int64))
    mine = np.empty((len(X), len(Y)), dtype=np.int64)
    theirs = np.empty_like(mine)
    step = max(1, chunk_pairs // max(1, len(Y)))
    for start in range(0, len(X), step):
        x = X[start : start + step, None, :]
        mine[start : start + step] = SCORE_TABLE[(x > Y[None]) @ BITS]
        theirs[start : start + step] = SCORE_TABLE[(Y[None] > x) @ BITS]
    return mine, theirs
