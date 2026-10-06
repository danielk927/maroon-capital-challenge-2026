import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from blotto.pipeline import Config, run

DATA = Path(__file__).resolve().parents[1] / "data"
HEADER = "model,prompt_variant,sample," + ",".join(f"b{i}" for i in range(1, 11))
FIGURES = ["pool", "price_lists", "candidates", "finalists", "tournaments", "allocation"]
FIGURE_DATA = ["allocation", "chatbot_bids", "candidates", "tournaments"]


def test_quick_pipeline_end_to_end_with_claude_answers_only(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA / "riddler", data / "riddler")
    (data / "chatbots").mkdir(parents=True)
    (data / "chatbots" / "answers.csv").write_text(
        HEADER + "\nclaude,verbatim,1,0,0,0,0,0,20,20,20,20,20\nclaude,ask,1,2,4,5,7,9,11,13,15,16,18\n"
    )
    out = tmp_path / "results"
    summary = run(Config.quick(), data, out)
    pick = np.array(summary["pick"])
    assert pick.shape == (10,) and pick.sum() == 100 and (pick >= 0).all()
    assert json.loads((out / "summary.json").read_text())["pick"] == summary["pick"]
    assert set(summary["heldout"]["riddler"]) == {"seen", "round_4"}
    assert "chatbots" not in summary["heldout"]  # no chatgpt answers to hold out
    assert "chatgpt" in summary["heldout"]["chatbots_skipped"]
    assert summary["chatbot_answers"] == {"kept": {"claude": 2}, "skipped_rows": 0, "unknown_models": []}
    assert 1 <= len(summary["finalists"]) <= 3
    assert summary["finalists"][0]["worst_case_rating"] >= summary["finalists"][-1]["worst_case_rating"]
    for name in FIGURES:
        assert (out / f"{name}.png").stat().st_size > 10_000
    for name in FIGURE_DATA:
        assert (out / "figure_data" / f"{name}.csv").stat().st_size > 0
    assert len(summary["pick_share_beaten"]) == 10 and all(0 <= s <= 1 for s in summary["pick_share_beaten"])
    assert sum(o["share"] for o in summary["pick_outcomes"]) <= 1 + 1e-9
    assert set(summary["bonus_rates"]) == {"pick", "chatbot_mean"}
    assert [m["from"] for m in summary["one_troop_moves"]] == [i + 1 for i in range(10) if pick[i] > 0]
    ranks = summary["tournament_ranks"]
    assert 0 <= ranks["median_share_above_wins"] <= ranks["max_share_above_wins"] <= 1
    assert 0 <= ranks["first_by_points"] <= 1


def test_quick_pipeline_without_any_chatbot_answers_stops_with_a_clear_error(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA / "riddler", data / "riddler")
    with pytest.raises(ValueError, match="chatbot answers"):
        run(Config.quick(), data, tmp_path / "results")


def test_quick_pipeline_with_only_chatgpt_answers_keeps_them_and_skips_the_holdout(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA / "riddler", data / "riddler")
    (data / "chatbots").mkdir(parents=True)
    (data / "chatbots" / "answers.csv").write_text(HEADER + "\nchatgpt,verbatim,1,1,1,1,2,11,18,20,21,24,1\n")
    summary = run(Config.quick(), data, tmp_path / "results")
    assert "chatbots" in summary["group_sizes"]
    assert "other than 'chatgpt'" in summary["heldout"]["chatbots_skipped"]


def test_quick_pipeline_holds_out_chatgpt_answers(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA / "riddler", data / "riddler")
    (data / "chatbots").mkdir(parents=True)
    (data / "chatbots" / "answers.csv").write_text(
        HEADER + "\nclaude,verbatim,1,0,0,0,0,13,26,2,1,30,28\nchatgpt,verbatim,1,1,1,1,2,11,18,20,21,24,1\n"
    )
    summary = run(Config.quick(), data, tmp_path / "results")
    assert set(summary["heldout"]["chatbots"]) == {"seen", "chatgpt"}
    assert "chatbots_skipped" not in summary["heldout"]
