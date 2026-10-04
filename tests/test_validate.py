import numpy as np
import pytest

from blotto.pool import Field, Group, random_splits
from blotto.robust import bounds_arrays
from blotto.scoring import play
from blotto.validate import counter_check, heldout_report, mock_tournaments, sample_mix

NAMES = ["riddler", "heuristics", "random", "counters"]
WEAK = np.array([[100] + [0] * 9])
STRONG = np.array([[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]])


def crowd_of(entry):
    return {g: Group(entry, np.array([1.0])) for g in NAMES}


def test_sample_mix_stays_within_bounds():
    rng = np.random.default_rng(0)
    lo, hi = bounds_arrays(NAMES)
    for _ in range(200):
        w = sample_mix(NAMES, rng)
        assert np.isclose(w.sum(), 1) and (w >= lo - 1e-12).all() and (w <= hi + 1e-12).all()


def test_round_robin_invariants():
    field = random_splits(60, np.random.default_rng(1))
    mine, theirs = play(field, field)
    wins, losses = mine > theirs, mine < theirs
    ties = ~wins & ~losses
    assert ((wins.sum(axis=1) + losses.sum(axis=1) + ties.sum(axis=1)) == 60).all()
    assert (wins == losses.T).all()
    assert (np.diag(mine) == 0).all()


def test_mock_tournaments_dominant_finalist_is_always_top():
    out = mock_tournaments(STRONG, crowd_of(WEAK), NAMES, np.random.default_rng(0), n=5, size_range=(10, 12))
    assert out["top_quartile_both"].tolist() == [1.0]
    assert out["share_above_wins"].shape == (5, 1) and (out["share_above_wins"] == 0).all()


def test_mock_tournaments_dominated_finalist_is_never_top():
    out = mock_tournaments(WEAK, crowd_of(STRONG), NAMES, np.random.default_rng(0), n=5, size_range=(10, 12))
    assert out["top_quartile_wins"].tolist() == [0.0]
    assert (out["share_above_wins"] == 1).all() and (out["share_above_points"] == 1).all()


def test_heldout_report_structure_and_values():
    pick = STRONG[0]
    seen = Field(np.array([[10] * 10]), np.array([1.0]))
    held = {"riddler_round_4": Field(np.array([[10, 0, 0, 0, 0, 0, 0, 30, 30, 30]]), np.array([1.0]))}
    report = heldout_report(pick, seen, held)
    assert set(report) == {"seen", "riddler_round_4"}
    assert report["seen"]["win_rate"] == 1.0 and np.isclose(report["seen"]["avg_points"], 92.40)
    assert report["riddler_round_4"]["win_rate"] == 0.0
    assert np.isclose(report["riddler_round_4"]["avg_points"], 21.45)


def test_counter_check_finds_a_counter_that_beats_the_pick():
    pick = np.array([10] * 10)
    pool = Field(np.array([[10] * 10, [0] * 9 + [100]]), np.array([0.5, 0.5]))
    result = counter_check(pick, pool, np.random.default_rng(0), chains=6, steps=300, copies=10)
    counter = np.array(result["counter"])
    mine, theirs = play(counter[None], pick[None])
    assert mine[0, 0] > theirs[0, 0]
    assert 0 <= result["pool_share_within_10"] <= 1
    assert 0 <= result["pick_win_rate_vs_counter_family"] <= 1


def test_sample_mix_returns_the_only_mix_when_maximums_total_exactly_one():
    bounds = {"riddler": (0.2, 0.6), "heuristics": (0.1, 0.4)}
    assert np.allclose(sample_mix(["riddler", "heuristics"], np.random.default_rng(0), bounds), [0.6, 0.4])


def test_sample_mix_gives_up_with_a_clear_error_when_the_room_is_too_thin():
    bounds = {"riddler": (0.0, 0.5), "heuristics": (0.0, 0.5), "chatbots": (0.0, 1e-7)}
    with pytest.raises(ValueError, match="widen"):
        sample_mix(["riddler", "heuristics", "chatbots"], np.random.default_rng(0), bounds, max_tries=1000)
