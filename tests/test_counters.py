import numpy as np

from blotto.counters import counter_loop, pick_distinct
from blotto.pool import Field, Group
from blotto.search import metrics


def test_pick_distinct_respects_distance_and_order():
    X = np.array(
        [
            [10] * 10,
            [11, 9] + [10] * 8,
            [13, 7] + [10] * 8,
            [20, 0] + [10] * 8,
        ]
    )
    assert pick_distinct(X, k=3, min_l1=6).tolist() == [[10] * 10, [13, 7] + [10] * 8, [20, 0] + [10] * 8]
    assert pick_distinct(X, k=2, min_l1=6).tolist() == [[10] * 10, [13, 7] + [10] * 8]
    assert pick_distinct(X, k=5, min_l1=100).tolist() == [[10] * 10]


def test_counter_loop_produces_valid_counters_that_beat_the_crowd():
    rng = np.random.default_rng(0)
    crowd = Group(np.array([[10] * 10, [0, 0, 0, 0, 0, 20, 20, 20, 20, 20]]), np.array([0.5, 0.5]))
    counters = counter_loop(
        {"crowd": crowd}, {"crowd": 0.85, "counters": 0.15}, rng,
        rounds=2, per_round=2, chains=6, steps=200, sample_size=50,
    )
    assert counters.shape[1] == 10 and 2 <= len(counters) <= 4
    assert (counters.sum(axis=1) == 100).all() and (counters >= 0).all()
    win_rate, _ = metrics(counters[:1], Field(crowd.entries, crowd.weights))
    assert win_rate[0] >= 0.5
