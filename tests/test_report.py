import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

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


def _pick_label(fig):
    """The axes, the pick's label, and a renderer at the figure's own dpi (closing a figure drops its canvas)."""
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    ax = fig.axes[0]
    return ax, next(t for t in ax.texts if t.get_text() == "final pick"), fig.canvas.get_renderer()


def test_price_list_dots_at_zero_troops_are_fully_visible(tmp_path):
    field = Field(np.array([[10] * 10]), np.array([1.0]))
    fig = plot_price_lists(field, np.array([0, 0, 0, 0, 15, 28, 1, 1, 33, 22]), tmp_path / "p.png")
    assert all(ax.get_ylim()[0] < 0 for ax in fig.axes)


def test_pick_label_does_not_cover_other_candidates(tmp_path):
    # A dense block of candidates below-left of the pick, and only low candidates to its right.
    gx, gy = np.meshgrid(np.arange(0.700, 0.748, 0.002), np.arange(50.0, 57.9, 0.1))
    rng = np.random.default_rng(0)
    win = np.r_[0.75, gx.ravel(), rng.uniform(0.755, 0.80, 30)]
    pts = np.r_[58.0, gy.ravel(), rng.uniform(50.0, 54.0, 30)]
    ax, label, renderer = _pick_label(plot_candidates(win, pts, 0, tmp_path / "c.png"))
    box = label.get_window_extent(renderer)
    shown = ax.transData.transform(np.column_stack([win[1:], pts[1:]]))
    inside = (shown[:, 0] >= box.x0) & (shown[:, 0] <= box.x1) & (shown[:, 1] >= box.y0) & (shown[:, 1] <= box.y1)
    assert not inside.any()


def test_pick_label_background_does_not_touch_a_marker_just_past_the_text(tmp_path):
    # Far-away anchors fix the axis limits, so both renders share one data-to-pixel mapping.
    win, pts = np.array([0.75, 0.70, 0.80]), np.array([58.0, 50.0, 60.0])
    ax, label, renderer = _pick_label(plot_candidates(win, pts, 0, tmp_path / "first.png"))
    text, patch = label.get_window_extent(renderer), label.get_bbox_patch().get_window_extent(renderer)
    radius = np.sqrt(ax.collections[0].get_sizes()[0]) / 2 * ax.figure.dpi / 72
    # A marker whose center is outside the text but whose edge overlaps the label background by half a pixel.
    gap = (patch.x1 - text.x1) + radius - 0.5
    dot = ax.transData.inverted().transform([(text.x1 + gap, (text.y0 + text.y1) / 2)])[0]
    win2, pts2 = np.r_[win, dot[0]], np.r_[pts, dot[1]]
    ax2, label2, renderer2 = _pick_label(plot_candidates(win2, pts2, 0, tmp_path / "second.png"))
    box = label2.get_bbox_patch().get_window_extent(renderer2)
    shown = ax2.transData.transform(np.column_stack([win2[1:], pts2[1:]]))
    touches = (
        (shown[:, 0] >= box.x0 - radius) & (shown[:, 0] <= box.x1 + radius)
        & (shown[:, 1] >= box.y0 - radius) & (shown[:, 1] <= box.y1 + radius)
    )
    assert not touches.any()


def test_pick_label_finds_the_clear_side_when_every_corner_is_blocked(tmp_path):
    # The real layout: the pick is the top point (spots above leave the axes), the frontier runs down-right
    # from it (blocking below-right), and one candidate sits just below-left; only the pick's left side is clear.
    frontier = [(0.792, 58.05), (0.796, 57.40), (0.800, 56.60), (0.815, 55.30), (0.835, 53.30), (0.855, 50.00)]
    win = np.r_[0.778, [w for w, _ in frontier], 0.7755, 0.730]
    pts = np.r_[58.70, [p for _, p in frontier], 57.65, 49.60]
    ax, label, renderer = _pick_label(plot_candidates(win, pts, 0, tmp_path / "c.png"))
    box = label.get_bbox_patch().get_window_extent(renderer)
    radius = np.sqrt(ax.collections[0].get_sizes()[0]) / 2 * ax.figure.dpi / 72
    shown = ax.transData.transform(np.column_stack([win[1:], pts[1:]]))
    touches = (
        (shown[:, 0] >= box.x0 - radius) & (shown[:, 0] <= box.x1 + radius)
        & (shown[:, 1] >= box.y0 - radius) & (shown[:, 1] <= box.y1 + radius)
    )
    assert not touches.any()
