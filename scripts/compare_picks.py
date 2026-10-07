"""Compare the current pick with the earlier one under the current and the earlier weight ranges.

uv run python scripts/compare_picks.py  (after scripts/run_pipeline.py; writes results/comparison.json)
"""

import json
from pathlib import Path

import numpy as np

from blotto.pipeline import Config, select_pick
from blotto.pool import field_from, read_chatbot_answers
from blotto.robust import BOUNDS, bounds_arrays, group_metrics, worst_case_rating
from blotto.scoring import play
from blotto.search import metrics
from blotto.validate import TOP_QUARTILE, mock_tournaments

ROOT = Path(__file__).resolve().parents[1]
# The first full run used wider weight ranges and chose this allocation.
EARLIER_PICK = [1, 4, 6, 1, 7, 28, 3, 5, 33, 12]
EARLIER_BOUNDS = {
    "riddler": (0.00, 0.20),
    "heuristics": (0.00, 0.15),
    "chatbots": (0.70, 0.95),
    "random": (0.00, 0.05),
    "counters": (0.05, 0.15),
}


def main() -> None:
    config = Config()
    # Same seed and first draw as the pipeline's main selection, so this rebuilds its pool exactly.
    sel = select_pick(config, ROOT / "data", np.random.default_rng(config.seed))
    current = json.loads((ROOT / "results" / "summary.json").read_text())["pick"]
    if sel.pick.tolist() != current:
        raise SystemExit("results/summary.json is out of date: run scripts/run_pipeline.py first")
    picks = {"current": current, "earlier": EARLIER_PICK}
    X = np.array(list(picks.values()))
    win, pts = group_metrics(X, sel.groups, sel.names)
    beaten: dict[str, dict[str, int]] = {p: {} for p in picks}
    for model, rows in read_chatbot_answers(ROOT / "data" / "chatbots" / "answers.csv").items():
        mine, theirs = play(X, rows)
        for k, p in enumerate(picks):
            beaten[p][model] = int((mine[k] > theirs[k]).sum())
    out = {
        "picks": picks,
        "win_rate_by_group": {p: dict(zip(sel.names, win[k].round(4).tolist())) for k, p in enumerate(picks)},
        "chatbot_answers_beaten": beaten,
        "by_bounds": {},
    }
    for label, bounds in [("current", BOUNDS), ("earlier", EARLIER_BOUNDS)]:
        lo, hi = bounds_arrays(sel.names, bounds)
        mid = (lo + hi) / 2
        center_win, center_pts = metrics(X, field_from(sel.groups, dict(zip(sel.names, mid / mid.sum()))))
        worst = worst_case_rating(win, pts, lo, hi)
        t = mock_tournaments(
            X, sel.groups, sel.names, np.random.default_rng(config.seed), n=config.tournaments, bounds=bounds
        )
        above_w, above_p = t["share_above_wins"], t["share_above_points"]
        out["by_bounds"][label] = {
            "bounds": {g: list(bounds[g]) for g in sel.names},
            **{
                p: {
                    "worst_case_rating": float(worst[k]),
                    "center_win_rate": float(center_win[k]),
                    "center_avg_points": float(center_pts[k]),
                    "top_quartile_both": float(t["top_quartile_both"][k]),
                    "median_share_above_wins": float(np.median(above_w[:, k])),
                    "median_share_above_points": float(np.median(above_p[:, k])),
                    "max_share_above_wins": float(above_w[:, k].max()),
                    "share_below_cutoff_wins": float((above_w[:, k] < TOP_QUARTILE).mean()),
                }
                for k, p in enumerate(picks)
            },
        }
    (ROOT / "results" / "comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    for label, rows in out["by_bounds"].items():
        print(f"{label} ranges:")
        for p, allocation in picks.items():
            r = rows[p]
            print(
                f"  {p:8s} {allocation}  worst case {r['worst_case_rating']:.4f}  center {r['center_win_rate']:.1%} / "
                f"{r['center_avg_points']:.1f} pts  top quartile {r['top_quartile_both']:.0%}  "
                f"median above by wins {r['median_share_above_wins']:.1%}"
            )
    for p in picks:
        print(f"{p} win rate by group: {out['win_rate_by_group'][p]}  chatbot answers beaten: {out['chatbot_answers_beaten'][p]}")


if __name__ == "__main__":
    main()
