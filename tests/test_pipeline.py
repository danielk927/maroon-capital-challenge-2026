import json
import shutil
from pathlib import Path

import numpy as np

from blotto.pipeline import Config, run

DATA = Path(__file__).resolve().parents[1] / "data"
HEADER = "model,prompt_variant,sample," + ",".join(f"b{i}" for i in range(1, 11))
FIGURES = ["pool", "price_lists", "candidates", "finalists", "tournaments", "allocation"]


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


def test_quick_pipeline_without_any_chatbot_answers(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA / "riddler", data / "riddler")
    summary = run(Config.quick(), data, tmp_path / "results")
    assert sum(summary["pick"]) == 100
    assert summary["chatbot_answers"] == {"kept": {}, "skipped_rows": 0, "unknown_models": []}
    assert "chatbots" not in summary["group_sizes"]
    assert "chatgpt" in summary["heldout"]["chatbots_skipped"]


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
