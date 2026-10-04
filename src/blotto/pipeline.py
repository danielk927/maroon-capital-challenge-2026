"""End-to-end selection: pool, counters, candidates, robust pick, validation, figures, and summary."""

import json
from collections.abc import Collection
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from blotto import report
from blotto.counters import counter_loop
from blotto.pool import (
    RIDDLER_FILES,
    Field,
    Group,
    build_pool,
    chatbot_answer_report,
    field_from,
    make_group,
    read_allocations,
    read_chatbot_answers,
    read_riddler,
)
from blotto.robust import (
    BOUNDS,
    LAM_RANGE,
    Ranked,
    center_mix,
    generate_candidates,
    group_names,
    rank_candidates,
    robust_search,
)
from blotto.scoring import MAX_SCORE
from blotto.search import metrics
from blotto.validate import counter_check, heldout_report, mock_tournaments


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    chains: int = 32
    steps: int = 3000
    sample_size: int = 2000
    counter_rounds: int = 6
    counters_per_round: int = 5
    counter_min_l1: int = 6
    seeds_per_setting: int = 1
    robust_seeds: int = 2
    per_group_sample: int = 400
    keep: int = 300
    finalist_gap: float = 0.005
    max_finalists: int = 10
    tournaments: int = 1000
    heldout_riddler_round: int = 4
    heldout_chatbot_model: str = "gemini"

    @classmethod
    def quick(cls) -> Config:
        """Tiny settings for tests and smoke runs."""
        return cls(
            chains=4, steps=100, sample_size=300, counter_rounds=1, counters_per_round=2,
            robust_seeds=1, per_group_sample=100, keep=20, max_finalists=3, tournaments=10,
        )


@dataclass(frozen=True)
class Selection:
    groups: dict[str, Group]
    names: list[str]
    counters: np.ndarray
    ranked: Ranked
    finalists: np.ndarray
    tournaments: dict[str, np.ndarray]
    pick: np.ndarray
    pick_index: int


def select_pick(
    config: Config,
    data_dir: Path,
    rng: np.random.Generator,
    exclude_riddler_rounds: Collection[int] = (),
    exclude_chatbot_models: Collection[str] = (),
) -> Selection:
    """Build the pool, run the counter loop, generate and rank candidates, and choose the pick."""
    groups = build_pool(data_dir, rng, exclude_riddler_rounds, exclude_chatbot_models)
    counters = counter_loop(
        groups, center_mix(group_names(groups) + ["counters"]), rng,
        rounds=config.counter_rounds, per_round=config.counters_per_round, min_l1=config.counter_min_l1,
        chains=config.chains, steps=config.steps, sample_size=config.sample_size,
    )
    groups = {**groups, "counters": make_group(counters, rng)}
    names = group_names(groups)
    candidates = generate_candidates(
        groups, names, rng, config.chains, config.steps, config.sample_size, config.seeds_per_setting
    )
    starts = rank_candidates(np.vstack([candidates, counters]), groups, names, keep=config.chains).candidates
    robust = [
        robust_search(groups, names, starts, rng, config.steps, config.per_group_sample)
        for _ in range(config.robust_seeds)
    ]
    ranked = rank_candidates(np.vstack([candidates, counters, *robust]), groups, names, keep=config.keep)
    close = ranked.worst >= ranked.worst[0] - config.finalist_gap
    finalists = ranked.candidates[close][: config.max_finalists]
    tournaments = mock_tournaments(finalists, groups, names, rng, n=config.tournaments)
    pick_index = int(np.lexsort((-ranked.worst[: len(finalists)], -tournaments["top_quartile_both"]))[0])
    return Selection(groups, names, counters, ranked, finalists, tournaments, finalists[pick_index], pick_index)


def _uniform(entries: np.ndarray) -> Field:
    return Field(entries, np.full(len(entries), 1.0 / len(entries)))


def run(config: Config, data_dir: Path, out_dir: Path) -> dict:
    """Main selection, held-out diagnostic run, counter check, figures, and summary.json."""
    rng = np.random.default_rng(config.seed)
    sel = select_pick(config, data_dir, rng)
    center = field_from(sel.groups, center_mix(sel.names))

    # Held-out diagnostic: rerun without one Riddler round and one chatbot model, then compare like with like.
    answers = read_chatbot_answers(data_dir / "chatbots" / "answers.csv")
    held_model = config.heldout_chatbot_model if config.heldout_chatbot_model in answers else None
    held_round = config.heldout_riddler_round
    diag = select_pick(config, data_dir, rng, {held_round}, {held_model} if held_model else set())
    heldout = {
        "riddler": heldout_report(
            diag.pick,
            _uniform(read_riddler(data_dir / "riddler", exclude_rounds={held_round})),
            {f"round_{held_round}": _uniform(read_allocations(data_dir / "riddler" / RIDDLER_FILES[held_round]))},
        )
    }
    other_answers = [rows for model, rows in sorted(answers.items()) if model != held_model]
    if held_model and other_answers:
        heldout["chatbots"] = heldout_report(
            diag.pick, _uniform(np.vstack(other_answers)), {held_model: _uniform(answers[held_model])}
        )
    elif held_model:
        heldout["chatbots_skipped"] = f"no answers from models other than '{held_model}' to compare with"
    else:
        heldout["chatbots_skipped"] = f"no answers from model '{config.heldout_chatbot_model}'"

    n_final = len(sel.finalists)
    win_center, pts_center = metrics(sel.ranked.candidates, center)
    summary = {
        "pick": sel.pick.tolist(),
        "pick_vs_center_field": {
            "win_rate": float(win_center[sel.pick_index]),
            "avg_points": float(pts_center[sel.pick_index]),
        },
        "finalists": [
            {
                "allocation": sel.finalists[f].tolist(),
                "worst_case_rating": float(sel.ranked.worst[f]),
                "top_quartile_wins": float(sel.tournaments["top_quartile_wins"][f]),
                "top_quartile_points": float(sel.tournaments["top_quartile_points"][f]),
                "top_quartile_both": float(sel.tournaments["top_quartile_both"][f]),
                "win_rate_by_group": dict(zip(sel.names, sel.ranked.win[f].round(4).tolist())),
                "avg_points_by_group": dict(zip(sel.names, sel.ranked.pts[f].round(2).tolist())),
            }
            for f in range(n_final)
        ],
        "heldout": {"pick": diag.pick.tolist(), "same_as_final_pick": bool((diag.pick == sel.pick).all()), **heldout},
        "counter_check": counter_check(sel.pick, center, rng, config.chains, config.steps),
        "chatbot_answers": chatbot_answer_report(data_dir / "chatbots" / "answers.csv"),
        "counters": sel.counters.tolist(),
        "group_sizes": {g: len(sel.groups[g].entries) for g in sel.names},
        "bounds": {g: list(BOUNDS[g]) for g in sel.names},
        "lam_range": list(LAM_RANGE),
        "config": asdict(config),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    beaten = np.array([report.price_list(center, i, np.array([sel.pick[i]]))[0] for i in range(len(sel.pick))])
    ratings = 0.5 * sel.ranked.win[:n_final] + 0.5 * sel.ranked.pts[:n_final] / MAX_SCORE
    report.plot_pool(sel.groups, sel.names, center_mix(sel.names), out_dir / "pool.png")
    report.plot_price_lists(center, sel.pick, out_dir / "price_lists.png")
    report.plot_candidates(win_center, pts_center, sel.pick_index, out_dir / "candidates.png")
    report.plot_finalists(sel.finalists, ratings, sel.names, out_dir / "finalists.png")
    report.plot_tournaments(
        sel.tournaments["share_above_wins"][:, sel.pick_index],
        sel.tournaments["share_above_points"][:, sel.pick_index],
        out_dir / "tournaments.png",
    )
    report.plot_allocation(sel.pick, beaten, out_dir / "allocation.png")
    return summary
