import numpy as np
import pytest

from blotto.pool import Group, random_splits
from blotto.robust import (
    BOUNDS,
    LAM_RANGE,
    bounds_arrays,
    center_mix,
    generate_candidates,
    group_metrics,
    group_names,
    rank_candidates,
    robust_search,
    vertex_mixes,
    worst_case_rating,
    worst_case_value,
)

ALL = ["riddler", "heuristics", "chatbots", "random", "counters"]


def test_bounds_and_center_mix():
    lo, hi = bounds_arrays(ALL)
    assert lo.tolist() == [0.05, 0.05, 0.50, 0.00, 0.05]
    assert hi.tolist() == [0.40, 0.30, 0.80, 0.05, 0.30]
    mix = center_mix(ALL)
    assert np.isclose(sum(mix.values()), 1)
    assert all(BOUNDS[g][0] <= w <= BOUNDS[g][1] for g, w in mix.items())


def test_infeasible_bounds_raise():
    with pytest.raises(ValueError, match="random"):
        bounds_arrays(["random", "counters"])  # at most 0.35 in total


def test_vertex_mixes_are_feasible_extreme_points():
    V = vertex_mixes(ALL)
    assert len(V) == 18
    lo, hi = bounds_arrays(ALL)
    for mix in V:
        w = np.array([mix[g] for g in ALL])
        assert np.isclose(w.sum(), 1) and (w >= lo - 1e-9).all() and (w <= hi + 1e-9).all()
        assert np.sum((w > lo + 1e-9) & (w < hi - 1e-9)) <= 1


def test_greedy_worst_case_matches_brute_force_over_vertices():
    lo, hi = bounds_arrays(ALL)
    V = np.array([[m[g] for g in ALL] for m in vertex_mixes(ALL)])
    r = np.random.default_rng(0).random((100, 5))
    assert np.allclose(worst_case_value(r, lo, hi), (r @ V.T).min(axis=1))


def test_worst_case_rating_takes_the_worse_lambda_endpoint():
    lo, hi = bounds_arrays(ALL)
    win = np.full((1, 5), 0.9)
    pts = np.full((1, 5), 0.1 * 127.05)  # a points rating of 0.1 against every group
    expected = min(lam * 0.9 + (1 - lam) * 0.1 for lam in LAM_RANGE)
    assert np.isclose(worst_case_rating(win, pts, lo, hi)[0], expected)


def test_group_names_follow_bounds_order():
    g = Group(np.array([[10] * 10]), np.array([1.0]))
    assert group_names({"random": g, "riddler": g, "counters": g}) == ["riddler", "random", "counters"]


def test_rank_candidates_sorts_by_worst_case():
    rng = np.random.default_rng(0)
    names = ["riddler", "heuristics", "random", "counters"]  # no chatbots group: bounds stay feasible
    groups = {g: Group(random_splits(40, rng), np.full(40, 1 / 40)) for g in names}
    cands = random_splits(20, rng)
    ranked = rank_candidates(np.vstack([cands, cands[:3]]), groups, names, keep=10)
    assert len(ranked.candidates) == 10
    assert len({tuple(r) for r in ranked.candidates.tolist()}) == 10
    assert (np.diff(ranked.worst) <= 1e-12).all()
    lo, hi = bounds_arrays(names)
    assert np.allclose(ranked.worst, worst_case_rating(ranked.win, ranked.pts, lo, hi))
    win, pts = group_metrics(ranked.candidates, groups, names)
    assert np.allclose(win, ranked.win) and np.allclose(pts, ranked.pts)


def test_bounds_with_a_minimum_above_its_maximum_raise():
    with pytest.raises(ValueError, match="riddler"):
        bounds_arrays(ALL, {**BOUNDS, "riddler": (0.60, 0.50)})


def _small_groups(rng, names):
    return {g: Group(random_splits(30, rng), np.full(30, 1 / 30)) for g in names}


def test_generate_candidates_returns_distinct_valid_allocations():
    rng = np.random.default_rng(0)
    names = ["riddler", "heuristics", "random", "counters"]
    X = generate_candidates(_small_groups(rng, names), names, rng, chains=3, steps=50, sample_size=60)
    assert X.ndim == 2 and X.shape[1] == 10 and len(X) >= 1
    assert (X.sum(axis=1) == 100).all() and (X >= 0).all()
    assert len(np.unique(X, axis=0)) == len(X)


def test_robust_search_never_ends_below_its_best_start():
    rng = np.random.default_rng(1)
    names = ["riddler", "heuristics", "random", "counters"]
    groups = _small_groups(rng, names)
    starts = random_splits(4, rng)
    X = robust_search(groups, names, starts, rng, steps=100, per_group_sample=30)  # samples cover each whole group
    lo, hi = bounds_arrays(names)

    def worst(Z):
        win, pts = group_metrics(Z, groups, names)
        return worst_case_rating(win, pts, lo, hi)

    assert (X.sum(axis=1) == 100).all() and (X >= 0).all()
    assert worst(X[:1])[0] >= worst(starts).max() - 1e-12


def test_chatbot_answers_are_at_least_half_of_every_allowed_field():
    assert all(mix["chatbots"] >= 0.5 - 1e-9 for mix in vertex_mixes(ALL))
    assert center_mix(ALL)["chatbots"] >= 0.5
