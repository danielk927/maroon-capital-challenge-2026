import numpy as np

from blotto.pool import Field, random_splits
from blotto.scoring import play
from blotto.search import MOVES, anneal, metrics, neighbors, polish, rating, search


def uniform(entries):
    entries = np.asarray(entries)
    return Field(entries, np.full(len(entries), 1 / len(entries)))


def test_metrics_and_rating_hand_checked():
    field = uniform([[2, 4, 5, 7, 9, 11, 13, 15, 16, 18], [10, 0, 0, 0, 0, 0, 0, 30, 30, 30]])
    x = np.array([[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]])
    win_rate, avg_points = metrics(x, field)
    assert np.isclose(win_rate[0], 0.5)
    assert np.isclose(avg_points[0], (92.40 + 21.45) / 2)
    assert np.isclose(rating(x, field, 1.0)[0], 0.5)
    assert np.isclose(rating(x, field, 0.0)[0], (92.40 + 21.45) / 2 / 127.05)


def test_neighbors_are_valid_and_complete():
    assert MOVES.shape == (270, 10)
    nb = neighbors(np.array([10] * 10))
    assert len(nb) == 270 and (nb.sum(axis=1) == 100).all()
    edge = neighbors(np.array([100] + [0] * 9))
    assert len(edge) == 27 and (edge >= 0).all()  # only battlefield 1 can give: 9 targets x 3 sizes


def test_polish_reaches_a_local_optimum():
    field = uniform(random_splits(200, np.random.default_rng(0)))
    objective = lambda X: rating(X, field, 0.5)
    x, value = polish(np.array([10] * 10), objective)
    assert np.isclose(value, objective(x[None])[0])
    assert (objective(neighbors(x)) <= value).all()
    assert value >= objective(np.array([[10] * 10]))[0]


def test_search_beats_a_single_opponent_and_is_reproducible():
    opponent = np.array([10] * 10)
    objective = lambda X: rating(X, uniform([opponent]), 0.5)
    starts = random_splits(8, np.random.default_rng(1))
    X1, v1 = search(objective, starts, np.random.default_rng(2), steps=300)
    X2, v2 = search(objective, starts, np.random.default_rng(2), steps=300)
    assert (X1 == X2).all() and np.allclose(v1, v2)
    mine, theirs = play(X1[:1], opponent[None])
    assert mine[0, 0] > theirs[0, 0]
    assert (np.diff(v1) <= 0).all()
    assert len({tuple(r) for r in X1.tolist()}) == len(X1)


def test_anneal_never_returns_worse_than_start():
    field = uniform(random_splits(100, np.random.default_rng(3)))
    objective = lambda X: rating(X, field, 0.5)
    starts = np.vstack([random_splits(3, np.random.default_rng(4)), [[100] + [0] * 9]])
    out = anneal(starts, objective, np.random.default_rng(5), steps=200)
    assert (out.sum(axis=1) == 100).all() and (out >= 0).all()
    assert (objective(out) >= objective(starts) - 1e-12).all()
