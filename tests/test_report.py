import numpy as np

from blotto.pool import Field, Group, field_from, random_splits
from blotto.report import (
    pareto_mask,
    plot_allocation,
    plot_candidates,
    plot_finalists,
    plot_pool,
    plot_price_lists,
    plot_tournaments,
    price_list,
)
from blotto.robust import center_mix


def test_pareto_mask():
    win = np.array([0.5, 0.6, 0.4, 0.45, 0.6])
    pts = np.array([50, 40, 60, 45, 40])
    assert pareto_mask(win, pts).tolist() == [True, True, True, False, True]


def test_price_list():
    field = Field(np.array([[10] * 10, [20] * 5 + [0] * 5]), np.array([0.25, 0.75]))
    assert price_list(field, 0, np.array([0, 10, 11, 20, 21])).tolist() == [0.0, 0.0, 0.25, 0.25, 1.0]


def test_all_figures_are_written(tmp_path):
    rng = np.random.default_rng(0)
    names = ["riddler", "heuristics", "random", "counters"]
    groups = {g: Group(random_splits(30, rng), np.full(30, 1 / 30)) for g in names}
    field = field_from(groups, center_mix(names))
    pick = np.array([1, 4, 7, 9, 13, 23, 26, 4, 6, 7])
    plot_pool(groups, names, center_mix(names), tmp_path / "pool.png")
    plot_price_lists(field, pick, tmp_path / "price_lists.png")
    plot_candidates(rng.random(40), rng.random(40) * 80, 3, tmp_path / "candidates.png")
    plot_finalists(random_splits(3, rng), rng.random((3, 4)), names, tmp_path / "finalists.png")
    plot_tournaments(rng.random(50) * 0.4, rng.random(50) * 0.4, tmp_path / "tournaments.png")
    plot_allocation(pick, rng.random(10), tmp_path / "allocation.png")
    for name in ["pool", "price_lists", "candidates", "finalists", "tournaments", "allocation"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
