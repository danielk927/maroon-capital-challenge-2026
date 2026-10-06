import csv

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

from blotto.pool import Field, Group, field_from, random_splits
from blotto.report import (
    bonus_rates,
    dot_offsets,
    outcome_shares,
    pareto_mask,
    plot_allocation,
    plot_candidates,
    plot_finalists,
    plot_pool,
    plot_price_lists,
    plot_tournaments,
    price_list,
    write_figure_data,
)
from blotto.robust import center_mix

# The pick wins battlefields 1-5 against the first opponent, all ten against the second, and 6-10 against the third.
PICK = np.array([11] * 5 + [9] * 5)
THREE = Field(np.array([[10] * 10, [0] * 10, [20] * 5 + [0] * 5]), np.array([0.5, 0.3, 0.2]))


def test_pareto_mask():
    win = np.array([0.5, 0.6, 0.4, 0.45, 0.6])
    pts = np.array([50, 40, 60, 45, 40])
    assert pareto_mask(win, pts).tolist() == [True, True, True, False, True]


def test_price_list():
    field = Field(np.array([[10] * 10, [20] * 5 + [0] * 5]), np.array([0.25, 0.75]))
    assert price_list(field, 0, np.array([0, 10, 11, 20, 21])).tolist() == [0.0, 0.0, 0.25, 0.25, 1.0]


def test_dot_offsets_put_equal_values_side_by_side():
    offsets = dot_offsets(np.array([5, 1, 5, 5, 1, 9]), spacing=0.1, max_width=0.6)
    assert offsets[5] == 0
    assert sorted(offsets[[0, 2, 3]].round(9).tolist()) == [-0.1, 0.0, 0.1]
    assert sorted(offsets[[1, 4]].round(9).tolist()) == [-0.05, 0.05]


def test_dot_offsets_squeeze_long_runs_into_max_width():
    offsets = dot_offsets(np.zeros(13), spacing=0.1, max_width=0.6)
    assert np.isclose(offsets.min(), -0.3) and np.isclose(offsets.max(), 0.3)


def test_outcome_shares():
    assert outcome_shares(PICK, THREE, top=2) == [
        {"won": [1, 2, 3, 4, 5], "score": 31.5, "share": 0.5},
        {"won": list(range(1, 11)), "score": 127.05, "share": 0.3},
    ]


def test_bonus_rates():
    all_on_one = np.array([100] + [0] * 9)  # wins only battlefield 1 against all three
    mod5, six = bonus_rates(np.vstack([PICK, all_on_one]), THREE)
    assert np.allclose(mod5, [1.0, 0.0]) and np.allclose(six, [0.5, 0.0])


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_figure_data_is_written(tmp_path):
    answers = {"claude": np.array([[0] * 5 + [20] * 5, [10] * 10]), "chatgpt": np.array([[10] * 10])}
    win, pts = np.array([0.8, 0.9, 0.7, 0.85]), np.array([60.0, 50.0, 55.0, 40.0])
    write_figure_data(
        tmp_path, PICK, np.linspace(0, 0.9, 10), answers, win, pts, 0, np.array([0.08, 0.081, 0.12]), np.zeros(3)
    )
    allocation = _read(tmp_path / "allocation.csv")
    assert [int(r["troops"]) for r in allocation] == PICK.tolist()
    assert [int(r["battlefield"]) for r in allocation] == list(range(1, 11))
    bids = _read(tmp_path / "chatbot_bids.csv")
    assert len(bids) == 30 and {r["model"] for r in bids} == {"claude", "chatgpt"}
    # The two answers with 10 troops on battlefield 1 sit side by side; the lone 0 sits on the center line.
    first = sorted(float(r["offset"]) for r in bids if r["battlefield"] == "1" and r["troops"] == "10")
    assert first[0] == -first[1] < 0 and next(r["offset"] for r in bids if r["troops"] == "0") == "0.0"
    candidates = _read(tmp_path / "candidates.csv")
    assert [float(r["win_rate"]) for r in candidates] == [0.7, 0.8, 0.85, 0.9]
    assert [r["frontier"] for r in candidates] == ["0", "1", "0", "1"]
    assert [r["pick"] for r in candidates] == ["0", "1", "0", "0"]
    tournaments = _read(tmp_path / "tournaments.csv")
    assert sum(int(r["wins"]) for r in tournaments) == 3 and sum(int(r["points"]) for r in tournaments) == 3
    by_bin = {round(float(r["share_above"]), 2): int(r["wins"]) for r in tournaments}
    assert by_bin[0.08] == 2 and by_bin[0.12] == 1


def test_all_figures_are_written(tmp_path):
    rng = np.random.default_rng(0)
    names = ["riddler", "heuristics", "chatbots", "random", "counters"]
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


def test_tournament_panels_scale_independently_and_zoom_to_the_data(tmp_path):
    rng = np.random.default_rng(0)
    fig = plot_tournaments(rng.random(200) * 0.3, np.zeros(200), tmp_path / "t.png")
    left, right = fig.axes
    assert not left.get_shared_y_axes().joined(left, right)
    assert left.get_xlim()[1] <= 0.5 and right.get_xlim() == left.get_xlim()


def test_finalists_title_lines_up_with_the_row_labels(tmp_path):
    finalists = np.array([[1, 4, 7, 9, 13, 23, 26, 4, 6, 7], [0, 0, 0, 0, 15, 28, 1, 1, 33, 22]])
    ratings = np.array([[0.60, 0.50], [0.55, 0.62]])
    fig = plot_finalists(finalists, ratings, ["riddler", "heuristics"], tmp_path / "f.png")
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    title = next(t for t in ax.get_children() if isinstance(t, Text) and t.get_text().startswith("Finalists"))
    labels_left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
    assert abs(title.get_window_extent(renderer).x0 - labels_left) <= 2


def test_finalists_colorbar_spans_the_whole_heatmap(tmp_path):
    rng = np.random.default_rng(0)
    fig = plot_finalists(random_splits(7, rng), rng.random((7, 4)), ["riddler", "heuristics", "random", "counters"], tmp_path / "f.png")
    heatmap, colorbar = fig.axes
    assert colorbar.get_position().height >= 0.95 * heatmap.get_position().height
