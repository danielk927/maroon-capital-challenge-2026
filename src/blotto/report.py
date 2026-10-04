"""Static figures for the writeup: light surface, validated palette, one axis per chart."""

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

from blotto.pool import Field, Group
from blotto.robust import bounds_arrays

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 8,
}
PERCENT = PercentFormatter(1.0, decimals=0)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


LABEL_SPOTS = [  # offset in points from the labeled mark, then horizontal and vertical alignment
    (-10, -14, "right", "top"),
    (10, -14, "left", "top"),
    (-10, 14, "right", "bottom"),
    (10, 14, "left", "bottom"),
    (-12, 0, "right", "center"),
    (12, 0, "left", "center"),
    (0, -16, "center", "top"),
    (0, 16, "center", "bottom"),
]


def _place_label(
    ax: plt.Axes, text: str, xy: tuple[float, float], obstacles: np.ndarray, margin_pt: float = 3.0
) -> None:
    """Label xy at the spot that stays inside the axes, avoids the legend, and covers the fewest obstacles.

    An obstacle counts as covered when its center is within margin_pt (a marker radius) of the label's background.
    """
    renderer = ax.figure.canvas.get_renderer()
    shown = ax.transData.transform(obstacles)
    frame = ax.get_window_extent(renderer)
    legend = ax.get_legend()
    legend_box = legend.get_window_extent(renderer) if legend else None
    margin = margin_pt * ax.figure.dpi / 72
    best = None
    for dx, dy, ha, va in LABEL_SPOTS:
        label = ax.annotate(
            text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=INK, fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"},
        )
        label.draw(renderer)  # positions the rounded background so its extent is current
        box = label.get_bbox_patch().get_window_extent(renderer)
        covered = int(
            (
                (shown[:, 0] >= box.x0 - margin) & (shown[:, 0] <= box.x1 + margin)
                & (shown[:, 1] >= box.y0 - margin) & (shown[:, 1] <= box.y1 + margin)
            ).sum()
        )
        outside = not (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1)
        score = (outside, bool(legend_box and box.overlaps(legend_box)), covered)
        if best is None or score < best[0]:
            if best is not None:
                best[1].remove()
            best = (score, label)
        else:
            label.remove()


def price_list(field: Field, index: int, troops: np.ndarray) -> np.ndarray:
    """Weighted share of the field beaten on battlefield index + 1 at each troop count."""
    column = field.entries[:, index]
    return np.array([field.weights[column < t].sum() for t in troops])


def pareto_mask(win: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """True where no other point is at least as good on both measures and better on one."""
    keep = np.ones(len(win), dtype=bool)
    for i in range(len(win)):
        dominated = (win >= win[i]) & (pts >= pts[i]) & ((win > win[i]) | (pts > pts[i]))
        keep[i] = not dominated.any()
    return keep


def plot_pool(groups: Mapping[str, Group], names: Sequence[str], mix: Mapping[str, float], path: Path) -> plt.Figure:
    """Center-mix share of each group, with the range the pessimist may choose."""
    lo, hi = bounds_arrays(names)
    y = np.arange(len(names))[::-1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        ax.barh(y, [mix[g] for g in names], height=0.55, color=BLUE)
        ax.hlines(y, lo, hi, color=INK_2, linewidth=1)
        ax.scatter(np.r_[lo, hi], np.r_[y, y], marker="|", s=80, color=INK_2, linewidths=1)
        ax.set_yticks(y, [f"{g}  ({len(groups[g].entries):,})" for g in names])
        ax.xaxis.set_major_formatter(PERCENT)
        ax.set_xlim(0, hi.max() + 0.05)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of the field (bar: center mix, line: range the pessimist may choose)")
        ax.set_title("Simulated applicant pool (entries per group, including jittered copies)")
        _save(fig, path)
    return fig


def plot_price_lists(field: Field, pick: np.ndarray, path: Path) -> plt.Figure:
    """Share of the field beaten at each troop count, one small panel per battlefield."""
    max_troops = max(40, int(np.max(pick)) + 5)
    troops = np.arange(max_troops + 1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flat):
            ax.step(troops, price_list(field, i, troops), where="post", color=BLUE, linewidth=2, solid_joinstyle="miter")
            beaten = price_list(field, i, np.array([pick[i]]))[0]
            ax.scatter([pick[i]], [beaten], s=36, color=ORANGE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
            label = f"{pick[i]} troop{'' if pick[i] == 1 else 's'}: beats {beaten:.0%}"
            ax.text(0.97, 0.05, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK_2)
            ax.set_title(f"Battlefield {i + 1}", fontsize=9)
            ax.yaxis.set_major_formatter(PERCENT)
            ax.set_ylim(-0.04, 1.02)  # room below 0% so a dot at 0 troops is not cut by the axis
        fig.supxlabel("Troops placed on the battlefield", fontsize=9, color=INK_2)
        fig.supylabel("Share of field beaten", fontsize=9, color=INK_2)
        _save(fig, path)
    return fig


def plot_candidates(win: np.ndarray, pts: np.ndarray, pick_index: int, path: Path) -> plt.Figure:
    """Every candidate's win rate and average points, with the Pareto frontier and the pick."""
    front = pareto_mask(win, pts)
    order = np.argsort(win[front])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.scatter(win, pts, s=14, color=AXIS, label="candidate", zorder=2)
        ax.plot(win[front][order], pts[front][order], color=BLUE, linewidth=2, marker="o", markersize=5, label="Pareto frontier", zorder=3)
        ax.scatter([win[pick_index]], [pts[pick_index]], s=90, color=ORANGE, edgecolors=SURFACE, linewidths=2, label="final pick", zorder=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(PERCENT)
        ax.set_xlabel("Win rate against the center-mix field")
        ax.set_ylabel("Average points per match")
        ax.set_title("Candidates: win rate versus points")
        ax.legend(loc="best")
        fig.tight_layout()
        line = np.column_stack([win[front][order], pts[front][order]])
        steps = np.linspace(0, 1, 20)[:, None]
        on_line = [a + steps * (b - a) for a, b in itertools.pairwise(line)]
        others = np.column_stack([np.delete(win, pick_index), np.delete(pts, pick_index)])
        _place_label(ax, "final pick", (win[pick_index], pts[pick_index]), np.vstack([others, *on_line]))
        _save(fig, path)
    return fig


def plot_finalists(finalists: np.ndarray, ratings: np.ndarray, names: Sequence[str], path: Path) -> plt.Figure:
    """Heatmap of each finalist's rating against each group."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(finalists) + 1.6))
        image = ax.imshow(ratings, cmap=BLUES, aspect="auto")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(finalists)), ["-".join(map(str, f)) for f in finalists])
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        middle = (ratings.min() + ratings.max()) / 2
        for r in range(ratings.shape[0]):
            for c in range(ratings.shape[1]):
                color = SURFACE if ratings[r, c] > middle else INK
                ax.text(c, r, f"{ratings[r, c]:.3f}", ha="center", va="center", fontsize=8, color=color)
        bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(color=AXIS, labelcolor=INK_2, labelsize=7.5)
        ax.set_title("Finalists: rating against each group (lambda = 0.5)")
        fig.tight_layout()
        renderer = fig.canvas.get_renderer()
        labels_left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
        # Line the title up with the row labels instead of the heatmap's left edge.
        ax.set_title(ax.get_title(loc="left"), x=ax.transAxes.inverted().transform((labels_left, 0))[0])
        _save(fig, path)
    return fig


def plot_tournaments(above_wins: np.ndarray, above_points: np.ndarray, path: Path) -> plt.Figure:
    """Distribution of the pick's rank across mock tournaments, by wins and by points."""
    bins = np.linspace(0, 1, 101)
    # Zoom to the data, but always show at least 0-50% so the cutoff has context on both sides.
    upper = min(1.0, max(0.5, np.ceil(max(np.max(above_wins), np.max(above_points)) * 10 + 1) / 10))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, data, label in zip(axes, [above_wins, above_points], ["wins", "points"]):
            ax.hist(data, bins=bins, color=BLUE, rwidth=0.85)
            ax.axvline(0.25, color=INK_2, linestyle="--", linewidth=1)
            ax.text(0.26, 0.95, "top-quartile cutoff", transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK_2, va="top")
            ax.set_title(f"Rank by {label}: top quartile in {(data < 0.25).mean():.0%} of tournaments")
            ax.xaxis.set_major_formatter(PERCENT)
            ax.set_xlim(0, upper)
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("Share of the field ranked above the pick")
        axes[0].set_ylabel("Mock tournaments")
        _save(fig, path)
    return fig


def plot_allocation(pick: np.ndarray, beaten: np.ndarray, path: Path) -> plt.Figure:
    """The final allocation and the share of the field it beats on each battlefield."""
    x = np.arange(1, len(pick) + 1)
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
        top.bar(x, pick, width=0.8, color=BLUE)
        for xi, v in zip(x, pick):
            top.text(xi, v + 0.6, str(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        top.set_ylim(0, max(pick) * 1.18)
        top.set_ylabel("Troops")
        top.set_title(f"Final allocation: {', '.join(map(str, pick))}")
        top.grid(axis="x", visible=False)
        bottom.bar(x, beaten, width=0.8, color=BLUE)
        bottom.yaxis.set_major_formatter(PERCENT)
        bottom.set_ylim(0, 1)
        bottom.set_ylabel("Share of field beaten")
        bottom.set_xlabel("Battlefield")
        bottom.set_xticks(x)
        bottom.grid(axis="x", visible=False)
        _save(fig, path)
    return fig
