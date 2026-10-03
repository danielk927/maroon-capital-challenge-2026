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
    assert "chatbots" not in summary["heldout"]  # no gemini answers to hold out
    assert 1 <= len(summary["finalists"]) <= 3
    assert summary["finalists"][0]["worst_case_rating"] >= summary["finalists"][-1]["worst_case_rating"]
    for name in FIGURES:
        assert (out / f"{name}.png").stat().st_size > 10_000
