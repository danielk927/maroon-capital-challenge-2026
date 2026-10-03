from fractions import Fraction

import numpy as np

from blotto.scoring import MAX_SCORE, SCORE_TABLE, mask_of, play, score_x100


def reference_score(won):
    """Independent exact-fraction implementation of the three rules."""
    won = set(won)
    if not won:
        return Fraction(0)
    score = sum(Fraction(3, 2) * v if (v - 1 in won or v + 1 in won) else Fraction(v) for v in won)
    if sum(won) % 5 == 0:
        score *= Fraction(7, 5)
    if 6 in won:
        score *= Fraction(11, 10)
    return score


def test_problem_example():
    assert score_x100({3, 5, 6, 7}) == 3300


def test_known_values():
    assert score_x100(range(1, 11)) == 12705
    assert MAX_SCORE == 127.05
    assert score_x100(set()) == 0
    assert score_x100({10}) == 1400
    assert score_x100(range(6, 11)) == 9240
    assert score_x100({1, 6, 7, 8, 9, 10}) == 6710


def test_table_matches_reference_on_all_sets():
    for mask in range(1024):
        won = {v + 1 for v in range(10) if mask >> v & 1}
        assert int(SCORE_TABLE[mask]) == reference_score(won) * 100


def test_mask_of():
    assert mask_of({1}) == 1
    assert mask_of({10}) == 512
    assert SCORE_TABLE[mask_of({3, 5, 6, 7})] == 3300


def test_play_hand_checked_matchups():
    x = np.array([[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]])
    Y = np.array(
        [
            [2, 4, 5, 7, 9, 11, 13, 15, 16, 18],
            [0, 6, 7, 12, 12, 21, 3, 31, 3, 5],
            [10, 0, 0, 0, 0, 0, 0, 30, 30, 30],
        ]
    )
    mine, theirs = play(x, Y)
    assert mine.tolist() == [[9240, 3550, 2145]]
    assert theirs.tolist() == [[3150, 4180, 4150]]


def test_play_identical_allocations_tie_at_zero():
    x = np.array([[10] * 10])
    mine, theirs = play(x, x)
    assert mine[0, 0] == 0 and theirs[0, 0] == 0


def test_play_swapping_sides_swaps_scores():
    rng = np.random.default_rng(0)
    X = rng.multinomial(100, np.full(10, 0.1), size=7)
    Y = rng.multinomial(100, np.full(10, 0.1), size=5)
    mine, theirs = play(X, Y)
    mine2, theirs2 = play(Y, X)
    assert (mine == theirs2.T).all() and (theirs == mine2.T).all()


def test_play_chunking_gives_same_result():
    rng = np.random.default_rng(1)
    X = rng.multinomial(100, np.full(10, 0.1), size=50)
    Y = rng.multinomial(100, np.full(10, 0.1), size=40)
    a = play(X, Y)
    b = play(X, Y, chunk_pairs=7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
