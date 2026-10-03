import shutil
from pathlib import Path

import numpy as np

from blotto.pool import (
    Group,
    build_pool,
    field_from,
    heuristic_entries,
    jitter,
    make_group,
    random_splits,
    round_to_budget,
    sample_field,
)

DATA = Path(__file__).resolve().parents[1] / "data"
HEADER = "model,prompt_variant,sample," + ",".join(f"b{i}" for i in range(1, 11))


def test_round_to_budget():
    assert round_to_budget(np.arange(1, 11)).tolist() == [2, 4, 5, 7, 9, 11, 13, 15, 16, 18]
    assert round_to_budget(np.ones(3)).sum() == 100


def test_heuristics_are_valid_and_include_key_rules():
    H = heuristic_entries()
    assert len(H) == 44
    assert (H.sum(axis=1) == 100).all() and (H >= 0).all()
    assert len(np.unique(H, axis=0)) == len(H)
    rows = {tuple(r) for r in H.tolist()}
    for expected in [
        (10,) * 10,
        (2, 4, 5, 7, 9, 11, 13, 15, 16, 18),
        (1, 3, 5, 7, 9, 11, 13, 15, 17, 19),
        (0, 0, 0, 0, 0, 20, 20, 20, 20, 20),
        (0, 0, 0, 0, 0, 33, 17, 17, 17, 16),
        (0, 0, 0, 0, 0, 16, 21, 21, 21, 21),
        (0, 0, 0, 0, 0, 0, 25, 25, 25, 25),
    ]:
        assert expected in rows


def test_random_splits_are_valid():
    R = random_splits(30, np.random.default_rng(0))
    assert R.shape == (30, 10) and (R.sum(axis=1) == 100).all() and (R >= 0).all()


def test_jitter_keeps_allocations_valid_even_when_concentrated():
    rng = np.random.default_rng(0)
    E = np.array([[100] + [0] * 9, [10] * 10, [0] * 9 + [100]])
    for _ in range(200):
        J = jitter(E, rng)
        assert (J.sum(axis=1) == 100).all() and (J >= 0).all()
    J = jitter(np.array([[10] * 10] * 50), rng)
    moved = np.abs(J - 10).sum(axis=1)
    assert moved.max() <= 12 and (moved > 0).mean() > 0.9


def test_make_group_weights():
    rng = np.random.default_rng(0)
    g = make_group(np.array([[10] * 10, [20] * 5 + [0] * 5]), rng)
    assert g.entries.shape == (4, 10)
    assert np.isclose(g.weights.sum(), 1)
    assert np.allclose(g.weights[:2], 0.35) and np.allclose(g.weights[2:], 0.15)
    plain = make_group(np.array([[10] * 10]), rng, jitter_share=0.0)
    assert plain.entries.shape == (1, 10) and plain.weights.tolist() == [1.0]


def test_field_from_renormalizes_over_present_groups():
    a = Group(np.array([[10] * 10]), np.array([1.0]))
    b = Group(np.array([[20] * 5 + [0] * 5, [0] * 5 + [20] * 5]), np.array([0.5, 0.5]))
    f = field_from({"a": a, "b": b}, {"a": 0.2, "b": 0.6, "missing": 0.2})
    assert f.entries.shape == (3, 10)
    assert np.allclose(f.weights, [0.25, 0.375, 0.375])


def test_sample_field_keeps_group_shares():
    rng = np.random.default_rng(0)
    big = Group(random_splits(500, rng), np.full(500, 1 / 500))
    small = Group(np.array([[10] * 10]), np.array([1.0]))
    f = sample_field({"big": big, "small": small}, {"big": 0.75, "small": 0.25}, 100, rng)
    assert len(f.entries) == 76  # 75 sampled from big, plus the single small entry
    assert np.isclose(f.weights.sum(), 1)
    assert np.isclose(f.weights[-1], 0.25)


def _data_dir(tmp_path, answers=None):
    shutil.copytree(DATA / "riddler", tmp_path / "riddler")
    if answers is not None:
        (tmp_path / "chatbots").mkdir()
        (tmp_path / "chatbots" / "answers.csv").write_text(HEADER + "\n" + answers)
    return tmp_path


def test_build_pool_groups_and_exclusions(tmp_path):
    data = _data_dir(
        tmp_path,
        "claude,verbatim,1,0,0,0,0,0,20,20,20,20,20\ngemini,verbatim,1,10,10,10,10,10,10,10,10,10,10\n",
    )
    rng = np.random.default_rng(0)
    pool = build_pool(data, rng)
    assert list(pool) == ["riddler", "heuristics", "chatbots", "random"]
    assert len(pool["riddler"].entries) == 2 * 4510
    assert len(pool["heuristics"].entries) == 2 * 44
    assert len(pool["chatbots"].entries) == 4
    assert len(pool["random"].entries) == 1000
    for g in pool.values():
        assert np.isclose(g.weights.sum(), 1) and (g.entries.sum(axis=1) == 100).all()
    held = build_pool(data, rng, exclude_riddler_rounds={4}, exclude_chatbot_models={"gemini"})
    assert len(held["riddler"].entries) == 2 * (4510 - 938)
    assert len(held["chatbots"].entries) == 2


def test_build_pool_without_chatbot_answers(tmp_path):
    pool = build_pool(_data_dir(tmp_path), np.random.default_rng(0))
    assert list(pool) == ["riddler", "heuristics", "random"]
