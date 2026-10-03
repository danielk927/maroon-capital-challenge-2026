# Strategy Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline that turns a simulated applicant pool into the 10 troop counts to submit, plus the numbers and figures for the writeup.

**Architecture:** An exact lookup-table scorer feeds a weighted pool of simulated applicants (Riddler data, rules of thumb, chatbot answers, random splits, and counters that best-respond to the rest).
Simulated annealing plus a steepest-ascent polish finds strong allocations, and a pessimist's worst case over plausible field mixes picks among them.
Mock tournaments, a held-out run, and a counter check validate the pick.

**Tech Stack:** Python 3.14, uv, numpy, matplotlib, pytest.

**Spec:** `docs/specs/2026-10-03-strategy-design.md`

## Global Constraints

- Python 3.14 via uv (`requires-python = ">=3.14"`); runtime dependencies are numpy and matplotlib only; pytest is the only dev dependency.
- Code lives in `src/blotto/`, tests in `tests/`, and every command runs from the repo root through `uv run`.
- An allocation is an int64 array of 10 non-negative integers summing to 100; battlefield v is array index v - 1 and bit v - 1 of a win mask.
- Scores are integers equal to the true score times 100 (`SCORE_TABLE`); never compare float scores.
- Rating(lam) = lam * win rate + (1 - lam) * average points / 127.05.
- Weight bounds: riddler 20-50%, heuristics 10-30%, chatbots 15-50%, random 0-10%, counters 5-25%.
  The pessimist's lam range is [0.25, 0.75], and candidate searches use lam in {0.25, 0.5, 0.75}.
- Jittered copies carry 30% of a group's weight; the random group has none.
- All randomness flows from one `np.random.Generator` created from `Config.seed` (2026); never use numpy's global random state.
- Never use the em dash character in any file or commit message.
- Commit messages carry no Co-Authored-By or other attribution lines; the repo-local git identity (Doyoon Kim <dkim250810@gmail.com>) is already configured.
- In long Markdown files, put each full sentence on its own line.
- Every task ends with a commit and `git push` to `origin main`.

## Review Focus

1. No chatbot answers yet, or answers from only some models: the pool builds without a `chatbots` group, the weight bounds stay feasible, and the held-out run skips the missing model.
   Pinned by `test_build_pool_without_chatbot_answers` (Task 3), `test_rank_candidates_sorts_by_worst_case` without chatbots (Task 7), and the Claude-only pipeline test (Task 10).
2. Hand-typed rows in `answers.csv` with typos, missing cells, or sums other than 100: they are skipped and never crash the run.
   Pinned by `test_chatbot_answers_grouped_by_model_and_invalid_rows_skipped` (Task 2).
3. Concentrated allocations such as all 100 troops on one battlefield: jitter, neighbors, and annealing never produce negative counts or totals other than 100.
   Pinned by `test_jitter_keeps_allocations_valid_even_when_concentrated` (Task 3) and the concentrated start in `test_anneal_never_returns_worse_than_start` (Task 5).
4. Identical submissions in a field: they tie 0 to 0 and never count as a win for either side.
   Pinned by `test_play_identical_allocations_tie_at_zero` (Task 1) and the identical crowds in the mock tournament tests (Task 8).
5. Weight bounds that cannot sum to 1 after a group is missing or the bounds are edited: a clear `ValueError` naming the groups.
   Pinned by `test_infeasible_bounds_raise` (Task 7).

---

### Task 1: Project scaffold and exact scorer

**Files:**
- Create: `pyproject.toml`
- Create: `src/blotto/__init__.py`
- Create: `src/blotto/scoring.py`
- Test: `tests/test_scoring.py`
- Generated: `uv.lock`

**Interfaces:**
- Consumes: nothing.
- Produces: `N_FIELDS: int = 10`, `BUDGET: int = 100`, `MAX_SCORE: float = 127.05`, `BITS: np.ndarray` (int64, shape (10,)), `score_x100(won: Iterable[int]) -> int`, `mask_of(won: Iterable[int]) -> int`, `SCORE_TABLE: np.ndarray` (int64, shape (1024,)), `play(X: np.ndarray, Y: np.ndarray, chunk_pairs: int = 500_000) -> tuple[np.ndarray, np.ndarray]` returning (mine, theirs), each int64 with shape (len(X), len(Y)).

- [ ] **Step 1: Create the project files**

`pyproject.toml`:

```toml
[project]
name = "blotto"
version = "0.1.0"
description = "Strategy pipeline for the Maroon Capital 2026 challenge, a Colonel Blotto variant"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "matplotlib>=3.10",
    "numpy>=2.3",
]

[dependency-groups]
dev = [
    "pytest>=8.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/blotto"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/blotto/__init__.py`:

```python
"""Strategy pipeline for the Maroon Capital 2026 Blotto challenge."""
```

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, and installs numpy, matplotlib, pytest, and the `blotto` package in editable mode.

- [ ] **Step 2: Write the failing tests**

`tests/test_scoring.py`:

```python
from fractions import Fraction

import numpy as np

from blotto.scoring import MAX_SCORE, SCORE_TABLE, mask_of, play, score_x100


def reference_score(won):
    """Independent exact-fraction implementation of the three rules."""
    won = set(won)
    if not won:
        return Fraction(0)
    score = sum(Fraction(3, 2) * v if (v - 1 in won or v + 1 in won) else Fraction(v) for v in won)
    if sum(won) % 5 == 0:
        score *= Fraction(7, 5)
    if 6 in won:
        score *= Fraction(11, 10)
    return score


def test_problem_example():
    assert score_x100({3, 5, 6, 7}) == 3300


def test_known_values():
    assert score_x100(range(1, 11)) == 12705
    assert MAX_SCORE == 127.05
    assert score_x100(set()) == 0
    assert score_x100({10}) == 1400
    assert score_x100(range(6, 11)) == 9240
    assert score_x100({1, 6, 7, 8, 9, 10}) == 6710


def test_table_matches_reference_on_all_sets():
    for mask in range(1024):
        won = {v + 1 for v in range(10) if mask >> v & 1}
        assert int(SCORE_TABLE[mask]) == reference_score(won) * 100


def test_mask_of():
    assert mask_of({1}) == 1
    assert mask_of({10}) == 512
    assert SCORE_TABLE[mask_of({3, 5, 6, 7})] == 3300


def test_play_hand_checked_matchups():
    x = np.array([[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]])
    Y = np.array(
        [
            [2, 4, 5, 7, 9, 11, 13, 15, 16, 18],
            [0, 6, 7, 12, 12, 21, 3, 31, 3, 5],
            [10, 0, 0, 0, 0, 0, 0, 30, 30, 30],
        ]
    )
    mine, theirs = play(x, Y)
    assert mine.tolist() == [[9240, 3550, 2145]]
    assert theirs.tolist() == [[3150, 4180, 4150]]


def test_play_identical_allocations_tie_at_zero():
    x = np.array([[10] * 10])
    mine, theirs = play(x, x)
    assert mine[0, 0] == 0 and theirs[0, 0] == 0


def test_play_swapping_sides_swaps_scores():
    rng = np.random.default_rng(0)
    X = rng.multinomial(100, np.full(10, 0.1), size=7)
    Y = rng.multinomial(100, np.full(10, 0.1), size=5)
    mine, theirs = play(X, Y)
    mine2, theirs2 = play(Y, X)
    assert (mine == theirs2.T).all() and (theirs == mine2.T).all()


def test_play_chunking_gives_same_result():
    rng = np.random.default_rng(1)
    X = rng.multinomial(100, np.full(10, 0.1), size=50)
    Y = rng.multinomial(100, np.full(10, 0.1), size=40)
    a = play(X, Y)
    b = play(X, Y, chunk_pairs=7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.scoring'`.

- [ ] **Step 4: Implement the scorer**

`src/blotto/scoring.py`:

```python
"""Exact scoring for the Maroon Capital 2026 Blotto variant.

Battlefield v (1 to 10) is array index v - 1 and bit v - 1 of a win mask.
Scores are integers equal to the true score times 100, so ties compare exactly.
"""

from collections.abc import Iterable

import numpy as np

N_FIELDS = 10
BUDGET = 100
MAX_SCORE = 127.05
BITS = (1 << np.arange(N_FIELDS)).astype(np.int64)


def score_x100(won: Iterable[int]) -> int:
    """Total score times 100 for winning the battlefields with the given values."""
    won = set(won)
    if not won:
        return 0
    base = sum(won)
    adjacent = sum(v for v in won if v - 1 in won or v + 1 in won)
    total = 2 * (base - adjacent) + 3 * adjacent  # Rule 1, scaled by 2
    total *= 7 if base % 5 == 0 else 5  # Rule 2, scaled by 5 more
    total *= 11 if 6 in won else 10  # Rule 3, scaled by 10 more
    return total


def mask_of(won: Iterable[int]) -> int:
    """Win mask for a set of battlefield values."""
    return sum(1 << (v - 1) for v in set(won))


SCORE_TABLE = np.array(
    [score_x100(v + 1 for v in range(N_FIELDS) if mask >> v & 1) for mask in range(1 << N_FIELDS)],
    dtype=np.int64,
)


def play(X: np.ndarray, Y: np.ndarray, chunk_pairs: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Scores times 100 for every allocation in X against every allocation in Y.

    X is (K, 10) and Y is (N, 10). Returns (mine, theirs), each (K, N) int64.
    Rows of X are processed in chunks of about chunk_pairs matchups to bound memory.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.int64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.int64))
    mine = np.empty((len(X), len(Y)), dtype=np.int64)
    theirs = np.empty_like(mine)
    step = max(1, chunk_pairs // max(1, len(Y)))
    for start in range(0, len(X), step):
        x = X[start : start + step, None, :]
        mine[start : start + step] = SCORE_TABLE[(x > Y[None]) @ BITS]
        theirs[start : start + step] = SCORE_TABLE[(Y[None] > x) @ BITS]
    return mine, theirs
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit and push**

```bash
git add pyproject.toml uv.lock src/blotto/__init__.py src/blotto/scoring.py tests/test_scoring.py
git commit -m "Add exact scorer with a 1,024-entry lookup table"
git push
```

---

### Task 2: Riddler data and allocation loaders

**Files:**
- Create: `data/riddler/castle-solutions.csv`, `data/riddler/castle-solutions-2.csv`, `data/riddler/castle-solutions-3.csv`, `data/riddler/castle-solutions-4.csv`
- Create: `data/riddler/README.md`
- Create: `src/blotto/pool.py`
- Test: `tests/test_pool_loading.py`

**Interfaces:**
- Consumes: `BUDGET`, `N_FIELDS` from `blotto.scoring`.
- Produces: `RIDDLER_FILES: dict[int, str]` (round number to file name), `read_allocations(path: Path) -> np.ndarray` ((n, 10) int64), `read_riddler(riddler_dir: Path, exclude_rounds: Collection[int] = ()) -> np.ndarray`, `read_chatbot_answers(path: Path) -> dict[str, np.ndarray]` (lower-case model name to (n, 10) int64; empty dict when the file is missing).

- [ ] **Step 1: Download the Riddler data**

```bash
mkdir -p data/riddler
for f in castle-solutions.csv castle-solutions-2.csv castle-solutions-3.csv castle-solutions-4.csv; do
  curl -sSfL -o "data/riddler/$f" "https://raw.githubusercontent.com/fivethirtyeight/data/master/riddler-castles/$f"
done
ls -l data/riddler
```

Expected: four non-empty CSV files.

- [ ] **Step 2: Write the attribution README**

`data/riddler/README.md`:

```markdown
# Riddler castle submissions

These files are the submissions to FiveThirtyEight's "Battle for Riddler Nation" puzzles, rounds 1 to 4.
They are unchanged copies from https://github.com/fivethirtyeight/data/tree/master/riddler-castles.
FiveThirtyEight publishes this data under the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/).

The first 10 columns of each row are the troops sent to castles 1 to 10.
That game used plain scoring: castles worth 1 to 10 points, ties split, and no multipliers.
The pipeline uses these entries as a proxy for how people allocate troops on instinct.
```

- [ ] **Step 3: Commit and push the data**

```bash
git add data/riddler
git commit -m "Add FiveThirtyEight Riddler castle data (CC BY 4.0)"
git push
```

- [ ] **Step 4: Write the failing tests**

`tests/test_pool_loading.py`:

```python
from pathlib import Path

from blotto.pool import RIDDLER_FILES, read_allocations, read_chatbot_answers, read_riddler

DATA = Path(__file__).resolve().parents[1] / "data"
HEADER = "model,prompt_variant,sample," + ",".join(f"b{i}" for i in range(1, 11))


def test_riddler_counts_per_round():
    counts = [len(read_allocations(DATA / "riddler" / RIDDLER_FILES[r])) for r in (1, 2, 3, 4)]
    assert counts == [1349, 902, 1321, 938]


def test_riddler_exclude_round():
    assert len(read_riddler(DATA / "riddler")) == 4510
    assert len(read_riddler(DATA / "riddler", exclude_rounds={4})) == 4510 - 938


def test_read_allocations_skips_invalid_rows(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        "c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,why\n"
        "10,10,10,10,10,10,10,10,10,10,ok\n"
        "10,10,10,10,10,10,10,10,10,11,sums to 101\n"
        "-1,11,10,10,10,10,10,10,10,10,negative\n"
        "9.5,10.5,10,10,10,10,10,10,10,10,fractions\n"
        "a,b,c,d,e,f,g,h,i,j,text\n"
        "10,10,10\n"
    )
    assert read_allocations(p).tolist() == [[10] * 10]


def test_chatbot_answers_grouped_by_model_and_invalid_rows_skipped(tmp_path):
    p = tmp_path / "answers.csv"
    p.write_text(
        HEADER + "\n"
        "Claude,verbatim,1,0,0,0,0,0,20,20,20,20,20\n"
        "chatgpt,ask,1,10,10,10,10,10,10,10,10,10,10\n"
        "gemini,verbatim,1,10,10,10,10,10,10,10,10,10,9\n"
        "gemini,verbatim,2,x,10,10,10,10,10,10,10,10,10\n"
        "gemini,verbatim,3,10,10\n"
        ",verbatim,4,10,10,10,10,10,10,10,10,10,10\n"
    )
    answers = read_chatbot_answers(p)
    assert set(answers) == {"claude", "chatgpt"}
    assert answers["claude"].tolist() == [[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]]
    assert answers["chatgpt"].tolist() == [[10] * 10]


def test_missing_chatbot_file_gives_no_answers(tmp_path):
    assert read_chatbot_answers(tmp_path / "missing.csv") == {}
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pool_loading.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.pool'`.

- [ ] **Step 6: Implement the loaders**

`src/blotto/pool.py`:

```python
"""Simulated applicant pool: loading allocations, building groups, and weighted fields."""

import csv
from collections.abc import Collection
from pathlib import Path

import numpy as np

from blotto.scoring import BUDGET, N_FIELDS

RIDDLER_FILES = {
    1: "castle-solutions.csv",
    2: "castle-solutions-2.csv",
    3: "castle-solutions-3.csv",
    4: "castle-solutions-4.csv",
}


def _valid(a: list[float]) -> bool:
    return len(a) == N_FIELDS and all(v >= 0 and v == int(v) for v in a) and sum(a) == BUDGET


def read_allocations(path: Path) -> np.ndarray:
    """Valid allocations from a CSV whose first 10 columns are troop counts (header row skipped)."""
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            try:
                a = [float(x) for x in row[:N_FIELDS]]
            except ValueError:
                continue
            if _valid(a):
                rows.append([int(v) for v in a])
    return np.array(rows, dtype=np.int64).reshape(-1, N_FIELDS)


def read_riddler(riddler_dir: Path, exclude_rounds: Collection[int] = ()) -> np.ndarray:
    """All valid Riddler entries from the rounds not excluded."""
    return np.vstack(
        [read_allocations(riddler_dir / name) for rnd, name in RIDDLER_FILES.items() if rnd not in exclude_rounds]
    )


def read_chatbot_answers(path: Path) -> dict[str, np.ndarray]:
    """Valid allocations from answers.csv keyed by lower-case model name; invalid rows are skipped."""
    if not path.exists():
        return {}
    by_model: dict[str, list[list[int]]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model = (row.get("model") or "").strip().lower()
            try:
                a = [float(row[f"b{i}"]) for i in range(1, N_FIELDS + 1)]
            except (KeyError, TypeError, ValueError):
                continue
            if model and _valid(a):
                by_model.setdefault(model, []).append([int(v) for v in a])
    return {model: np.array(rows, dtype=np.int64) for model, rows in by_model.items()}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pool_loading.py -v`
Expected: 5 passed.

- [ ] **Step 8: Commit and push**

```bash
git add src/blotto/pool.py tests/test_pool_loading.py
git commit -m "Add Riddler and chatbot answer loaders"
git push
```

---

### Task 3: Pool groups, jittered copies, and weighted fields

**Files:**
- Modify: `src/blotto/pool.py` (replace the import block, add constants and functions)
- Test: `tests/test_pool.py`

**Interfaces:**
- Consumes: `read_riddler`, `read_chatbot_answers` (Task 2); `BUDGET`, `N_FIELDS` (Task 1).
- Produces:
  - `JITTER_SHARE: float = 0.3`, `VALUES: np.ndarray` (1 to 10).
  - `Group(entries: np.ndarray, weights: np.ndarray)` and `Field(entries: np.ndarray, weights: np.ndarray)`: frozen dataclasses; weights sum to 1.
  - `round_to_budget(raw: np.ndarray, budget: int = 100) -> np.ndarray`.
  - `heuristic_entries() -> np.ndarray`: 44 distinct rules of thumb.
  - `random_splits(n: int, rng: np.random.Generator, alphas: tuple[float, ...] = (0.5, 1.0, 2.0)) -> np.ndarray`.
  - `jitter(entries: np.ndarray, rng: np.random.Generator) -> np.ndarray`.
  - `make_group(entries: np.ndarray, rng: np.random.Generator, jitter_share: float = JITTER_SHARE) -> Group`.
  - `build_pool(data_dir: Path, rng: np.random.Generator, exclude_riddler_rounds: Collection[int] = (), exclude_chatbot_models: Collection[str] = ()) -> dict[str, Group]`: keys in the order riddler, heuristics, chatbots (only if answers exist), random.
  - `field_from(groups: Mapping[str, Group], mix: Mapping[str, float]) -> Field`.
  - `sample_field(groups: Mapping[str, Group], mix: Mapping[str, float], size: int, rng: np.random.Generator) -> Field`.
  - Mix weights are renormalized over the groups that are present, so a mix may name a group that does not exist yet.

- [ ] **Step 1: Write the failing tests**

`tests/test_pool.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pool.py -v`
Expected: FAIL with `ImportError: cannot import name 'Group' from 'blotto.pool'`.

- [ ] **Step 3: Replace the import block of `src/blotto/pool.py`**

Replace these lines at the top of `src/blotto/pool.py`:

```python
import csv
from collections.abc import Collection
from pathlib import Path
```

with:

```python
import csv
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
```

- [ ] **Step 4: Add the constants and dataclasses after `RIDDLER_FILES`**

Insert directly after the `RIDDLER_FILES = {...}` block in `src/blotto/pool.py`:

```python
JITTER_SHARE = 0.3
VALUES = np.arange(1, N_FIELDS + 1)


@dataclass(frozen=True)
class Group:
    """Allocations (n, 10) with weights (n,) that sum to 1 within the group."""

    entries: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class Field:
    """Opponents (N, 10) with weights (N,) that sum to 1."""

    entries: np.ndarray
    weights: np.ndarray
```

- [ ] **Step 5: Append the generators, groups, and fields to the end of `src/blotto/pool.py`**

```python
def round_to_budget(raw: np.ndarray, budget: int = BUDGET) -> np.ndarray:
    """Integers proportional to the non-negative shares in raw, summing to budget (largest remainder)."""
    raw = np.asarray(raw, dtype=float)
    scaled = raw / raw.sum() * budget
    out = np.floor(scaled).astype(np.int64)
    order = np.argsort(-(scaled - out), kind="stable")
    out[order[: budget - out.sum()]] += 1
    return out


def _nudged(a: np.ndarray) -> np.ndarray | None:
    """Each positive multiple of 5 raised by one troop, paid for by the lowest battlefield with troops."""
    targets = np.flatnonzero((a > 0) & (a % 5 == 0))
    if len(targets) == 0:
        return None
    out = a.copy()
    out[targets] += 1
    payer = np.flatnonzero(out)[0]
    out[payer] -= len(targets)
    return out if out[payer] >= 0 else None


def heuristic_entries() -> np.ndarray:
    """Rules of thumb a thoughtful applicant might write down in ten minutes (44 distinct rows)."""
    rows = [np.full(N_FIELDS, 10), np.arange(1, 20, 2)]
    rows += [round_to_budget(VALUES**p) for p in (0.5, 1.0, 2.0)]
    for k in range(3, 7):
        rows.append(round_to_budget((VALUES > N_FIELDS - k).astype(float)))
    for a in range(1, N_FIELDS + 1):
        for b in range(a + 1, N_FIELDS + 1):
            if sum(range(a, b + 1)) % 5:
                continue
            block = ((VALUES >= a) & (VALUES <= b)).astype(float)
            even = round_to_budget(block)
            rows.append(even)
            if a <= 6 <= b:
                heavy = block.copy()
                heavy[5] = 2.0
                rows.append(round_to_budget(heavy))
            nudged = _nudged(even)
            if nudged is not None:
                rows.append(nudged)
    return np.unique(np.array(rows, dtype=np.int64), axis=0)


def random_splits(n: int, rng: np.random.Generator, alphas: tuple[float, ...] = (0.5, 1.0, 2.0)) -> np.ndarray:
    """n random allocations drawn as multinomial(100, Dirichlet(alpha)), cycling alpha through alphas."""
    out = np.empty((n, N_FIELDS), dtype=np.int64)
    for i in range(n):
        out[i] = rng.multinomial(BUDGET, rng.dirichlet(np.full(N_FIELDS, alphas[i % len(alphas)])))
    return out


def jitter(entries: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One copy of each entry with 1 to 3 troops moved between random battlefields, once or twice."""
    out = np.array(entries, dtype=np.int64, copy=True).reshape(-1, N_FIELDS)
    for row in out:
        for _ in range(int(rng.integers(1, 3))):
            i = int(rng.choice(np.flatnonzero(row)))
            j = int(rng.choice(np.delete(np.arange(N_FIELDS), i)))
            k = min(int(rng.integers(1, 4)), int(row[i]))
            row[i] -= k
            row[j] += k
    return out


def make_group(entries: np.ndarray, rng: np.random.Generator, jitter_share: float = JITTER_SHARE) -> Group:
    """Entries plus one jittered copy each; the copies carry jitter_share of the group's weight."""
    entries = np.asarray(entries, dtype=np.int64).reshape(-1, N_FIELDS)
    n = len(entries)
    if jitter_share == 0:
        return Group(entries, np.full(n, 1.0 / n))
    weights = np.concatenate([np.full(n, (1 - jitter_share) / n), np.full(n, jitter_share / n)])
    return Group(np.vstack([entries, jitter(entries, rng)]), weights)


def build_pool(
    data_dir: Path,
    rng: np.random.Generator,
    exclude_riddler_rounds: Collection[int] = (),
    exclude_chatbot_models: Collection[str] = (),
) -> dict[str, Group]:
    """The non-strategic groups: riddler, heuristics, chatbots (when answers exist), and random."""
    groups = {
        "riddler": make_group(read_riddler(data_dir / "riddler", exclude_riddler_rounds), rng),
        "heuristics": make_group(heuristic_entries(), rng),
    }
    answers = read_chatbot_answers(data_dir / "chatbots" / "answers.csv")
    kept = [rows for model, rows in sorted(answers.items()) if model not in exclude_chatbot_models]
    if kept:
        groups["chatbots"] = make_group(np.vstack(kept), rng)
    groups["random"] = make_group(random_splits(1000, rng), rng, jitter_share=0.0)
    return groups


def _shares(groups: Mapping[str, Group], mix: Mapping[str, float]) -> dict[str, float]:
    present = {g: w for g, w in mix.items() if w > 0 and g in groups}
    total = sum(present.values())
    return {g: w / total for g, w in present.items()}


def field_from(groups: Mapping[str, Group], mix: Mapping[str, float]) -> Field:
    """Every entry of the mixed groups; mix weights are renormalized over the groups present."""
    shares = _shares(groups, mix)
    return Field(
        np.vstack([groups[g].entries for g in shares]),
        np.concatenate([groups[g].weights * s for g, s in shares.items()]),
    )


def sample_field(groups: Mapping[str, Group], mix: Mapping[str, float], size: int, rng: np.random.Generator) -> Field:
    """About `size` opponents, stratified so each group keeps its share of the weight."""
    entries, weights = [], []
    for g, s in _shares(groups, mix).items():
        group = groups[g]
        m = max(1, round(size * s))
        if m >= len(group.entries):
            entries.append(group.entries)
            weights.append(group.weights * s)
        else:
            idx = rng.choice(len(group.entries), size=m, replace=False, p=group.weights)
            entries.append(group.entries[idx])
            weights.append(np.full(m, s / m))
    return Field(np.vstack(entries), np.concatenate(weights))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pool.py tests/test_pool_loading.py -v`
Expected: 14 passed.

- [ ] **Step 7: Commit and push**

```bash
git add src/blotto/pool.py tests/test_pool.py
git commit -m "Add pool groups, jittered copies, and weighted fields"
git push
```

---

### Task 4: Problem statement and Claude chatbot answers

This task dispatches subagents, so the main session runs it, not an implementer subagent.

**Files:**
- Create: `docs/problem.md`
- Create: `data/chatbots/README.md`
- Create: `data/chatbots/answers.csv`

**Interfaces:**
- Consumes: `read_chatbot_answers` (Task 2) to verify the CSV.
- Produces: `data/chatbots/answers.csv` with header `model,prompt_variant,sample,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10` and rows with `model` = `claude`.

- [ ] **Step 1: Write the problem statement**

`docs/problem.md`:

```markdown
# Challenge problem (Maroon Capital, 2026)

The problem statement as given to applicants, with PDF line breaks and page numbers removed.

## Problem Statement

You are competing against other applicants.
Each person has a limited number of points to distribute across several battlefields.
For each battlefield, whoever allocates the most points wins that battlefield.
Your goal is to outperform the other applicants on as many battlefields as possible, forcing you to decide where to focus your resources and where you are willing to give ground.

There are 10 battlefields, worth 1, 2, 3 . . . 10 points respectively if they are won.
You have 100 units of troops in total to divide across the 10 battlefields.
You must devote strictly more troops than your opponent to win a battlefield.
In the case of a tie, no points are granted for that battlefield.
Your base score is the sum of points from battlefields won.
Additionally the following rules are imposed in order:

- If you win two adjacent battlefields, each score will be multiplied by 1.5.
  This will not "stack up" so if you have 1, 2, and 3, with 1 and 3 adjacent to 2, 2 will only be given the modifier once.
- If your base score is a multiple of 5, your score is further multiplied by 1.4.
- If you win the battlefield with value 6, your score is further multiplied by 1.1.

Your total score is the score after applying all three rules above.
You win against another applicant by having a higher total score.
As an example, if you win battlefields 3, 5, 6, and 7, your score will be calculated as follows:

- base score = 3 + 5 + 6 + 7 = 21
- Rule 1: score1 = 3 + 5 * 1.5 + 6 * 1.5 + 7 * 1.5 = 30
- Rule 2 - since base score equals 21, score is unchanged (score2 = score1 = 30)
- total score = score3 = 1.1 * score2 = 1.1 * 30 = 33

We will run everyone's submission against each other.
We will evaluate based on number of wins and total points.

## Submission

Submit your response in the Google Form by Tuesday, October 6th at midnight.
Your submission must include 10 non-negative integers summing to 100 which denote the number of troops you allocate to battlefield 1 through 10 respectively.
Your submission should also include a writeup explaining the reasoning for the strategy you chose.
The writeup should be a PDF with no more than 2 A4 pages.
We will review the written responses of candidates whose scores fall within the top quartile.
```

- [ ] **Step 2: Write the collection protocol**

`data/chatbots/README.md`:

```markdown
# Chatbot answers

`answers.csv` holds the allocations chatbots recommended when given the challenge problem.
They model applicants who ask an AI for their answer.

Columns:

- `model`: lower case, one of `claude`, `chatgpt`, or `gemini`.
- `prompt_variant`: `verbatim` or `ask`.
- `sample`: 1, 2, 3, and so on within each model and variant.
- `b1` to `b10`: troops on battlefields 1 to 10.

Protocol for ChatGPT and Gemini, collected by hand through each chatbot's web interface:

- Start a new chat for every sample, so earlier answers cannot influence later ones.
- `verbatim`: paste everything in `docs/problem.md` from "Problem Statement" to the end.
- `ask`: paste the same text, then add a final line: "What allocation should I submit?".
- Record the allocation the chatbot recommends exactly as given.
  The pipeline skips rows that are not 10 non-negative integers summing to 100.
- Aim for at least 15 answers per model: about 10 `verbatim` and 5 `ask`.

Claude answers came from fresh subagents that saw only the prompt.
Their prompts ended with one extra instruction, to answer without tools and to put the allocation on its own final line, so that replies could be parsed.
```

- [ ] **Step 3: Collect 15 Claude answers**

Dispatch 15 Agent calls in a single message, with `subagent_type: "general-purpose"` and `description: "Answer the challenge problem"`.
Samples 1-7 of `verbatim` and 1-3 of `ask` use `model: "sonnet"`; samples 8-10 of `verbatim` and 4-5 of `ask` use `model: "opus"`.

The `verbatim` prompt is the text of `docs/problem.md` from the line `## Problem Statement` to the end, then a blank line, then:

```text
Reply the way you normally would in a chat conversation. Do not use any tools or run any code. End your reply with your recommended allocation on its own line as 10 comma-separated integers.
```

The `ask` prompt is identical, except that the line `What allocation should I submit?` comes between the problem text and that final instruction.

- [ ] **Step 4: Record the answers**

For each reply, take the last line that contains exactly 10 integers.
Keep it if the integers are non-negative and sum to 100; otherwise drop that reply.
Write `data/chatbots/answers.csv` with this header and one row per kept reply, numbering `sample` within each variant:

```text
model,prompt_variant,sample,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10
```

Each row looks like `claude,verbatim,1,<b1>,...,<b10>` with the integers from that reply.

- [ ] **Step 5: Verify the file loads**

Run: `uv run python -c "from pathlib import Path; from blotto.pool import read_chatbot_answers; print({m: len(v) for m, v in read_chatbot_answers(Path('data/chatbots/answers.csv')).items()})"`
Expected: `{'claude': N}` where N is the number of kept replies (15 if every reply parsed).

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass (22).

- [ ] **Step 7: Commit and push**

```bash
git add docs/problem.md data/chatbots
git commit -m "Add problem statement and Claude chatbot answers"
git push
```

---

### Task 5: Rating and search

**Files:**
- Create: `src/blotto/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `Field` and `random_splits` (Task 3); `MAX_SCORE`, `N_FIELDS`, `play` (Task 1).
- Produces:
  - `Objective = Callable[[np.ndarray], np.ndarray]`: maps (K, 10) allocations to (K,) values.
  - `metrics(X: np.ndarray, field: Field) -> tuple[np.ndarray, np.ndarray]`: weighted win rate and average points in true score units, each (K,).
  - `rating(X: np.ndarray, field: Field, lam: float) -> np.ndarray`.
  - `MOVES: np.ndarray` (270, 10); `neighbors(x: np.ndarray) -> np.ndarray`.
  - `polish(x: np.ndarray, objective: Objective) -> tuple[np.ndarray, float]`.
  - `anneal(starts: np.ndarray, objective: Objective, rng: np.random.Generator, steps: int = 3000, t_start: float = 0.05, t_end: float = 0.001, max_move: int = 5) -> np.ndarray`.
  - `search(objective: Objective, starts: np.ndarray, rng: np.random.Generator, steps: int = 3000) -> tuple[np.ndarray, np.ndarray]`: distinct local optima sorted best first, with their values.

- [ ] **Step 1: Write the failing tests**

`tests/test_search.py`:

```python
import numpy as np

from blotto.pool import Field, random_splits
from blotto.scoring import play
from blotto.search import MOVES, anneal, metrics, neighbors, polish, rating, search


def uniform(entries):
    entries = np.asarray(entries)
    return Field(entries, np.full(len(entries), 1 / len(entries)))


def test_metrics_and_rating_hand_checked():
    field = uniform([[2, 4, 5, 7, 9, 11, 13, 15, 16, 18], [10, 0, 0, 0, 0, 0, 0, 30, 30, 30]])
    x = np.array([[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]])
    win_rate, avg_points = metrics(x, field)
    assert np.isclose(win_rate[0], 0.5)
    assert np.isclose(avg_points[0], (92.40 + 21.45) / 2)
    assert np.isclose(rating(x, field, 1.0)[0], 0.5)
    assert np.isclose(rating(x, field, 0.0)[0], (92.40 + 21.45) / 2 / 127.05)


def test_neighbors_are_valid_and_complete():
    assert MOVES.shape == (270, 10)
    nb = neighbors(np.array([10] * 10))
    assert len(nb) == 270 and (nb.sum(axis=1) == 100).all()
    edge = neighbors(np.array([100] + [0] * 9))
    assert len(edge) == 27 and (edge >= 0).all()  # only battlefield 1 can give: 9 targets x 3 sizes


def test_polish_reaches_a_local_optimum():
    field = uniform(random_splits(200, np.random.default_rng(0)))
    objective = lambda X: rating(X, field, 0.5)
    x, value = polish(np.array([10] * 10), objective)
    assert np.isclose(value, objective(x[None])[0])
    assert (objective(neighbors(x)) <= value).all()
    assert value >= objective(np.array([[10] * 10]))[0]


def test_search_beats_a_single_opponent_and_is_reproducible():
    opponent = np.array([10] * 10)
    objective = lambda X: rating(X, uniform([opponent]), 0.5)
    starts = random_splits(8, np.random.default_rng(1))
    X1, v1 = search(objective, starts, np.random.default_rng(2), steps=300)
    X2, v2 = search(objective, starts, np.random.default_rng(2), steps=300)
    assert (X1 == X2).all() and np.allclose(v1, v2)
    mine, theirs = play(X1[:1], opponent[None])
    assert mine[0, 0] > theirs[0, 0]
    assert (np.diff(v1) <= 0).all()
    assert len({tuple(r) for r in X1.tolist()}) == len(X1)


def test_anneal_never_returns_worse_than_start():
    field = uniform(random_splits(100, np.random.default_rng(3)))
    objective = lambda X: rating(X, field, 0.5)
    starts = np.vstack([random_splits(3, np.random.default_rng(4)), [[100] + [0] * 9]])
    out = anneal(starts, objective, np.random.default_rng(5), steps=200)
    assert (out.sum(axis=1) == 100).all() and (out >= 0).all()
    assert (objective(out) >= objective(starts) - 1e-12).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.search'`.

- [ ] **Step 3: Implement rating and search**

`src/blotto/search.py`:

```python
"""Rating allocations against a weighted field, and searching for the best ones."""

from collections.abc import Callable

import numpy as np

from blotto.pool import Field
from blotto.scoring import MAX_SCORE, N_FIELDS, play

Objective = Callable[[np.ndarray], np.ndarray]


def metrics(X: np.ndarray, field: Field) -> tuple[np.ndarray, np.ndarray]:
    """Weighted win rate and average points (true score units) of each row of X against the field."""
    mine, theirs = play(X, field.entries)
    return (mine > theirs) @ field.weights, (mine @ field.weights) / 100.0


def rating(X: np.ndarray, field: Field, lam: float) -> np.ndarray:
    """lam * win rate + (1 - lam) * average points / 127.05, for each row of X."""
    win_rate, avg_points = metrics(X, field)
    return lam * win_rate + (1 - lam) * avg_points / MAX_SCORE


def _moves(max_shift: int) -> np.ndarray:
    moves = []
    for i in range(N_FIELDS):
        for j in range(N_FIELDS):
            if i != j:
                for k in range(1, max_shift + 1):
                    m = np.zeros(N_FIELDS, dtype=np.int64)
                    m[i], m[j] = -k, k
                    moves.append(m)
    return np.array(moves)


MOVES = _moves(3)


def neighbors(x: np.ndarray) -> np.ndarray:
    """Every allocation reachable by moving 1 to 3 troops from one battlefield to another."""
    nb = np.asarray(x, dtype=np.int64)[None, :] + MOVES
    return nb[(nb >= 0).all(axis=1)]


def polish(x: np.ndarray, objective: Objective) -> tuple[np.ndarray, float]:
    """Steepest ascent over neighbors until none improves; returns the local optimum and its value."""
    x = np.asarray(x, dtype=np.int64)
    current = float(objective(x[None])[0])
    while True:
        nb = neighbors(x)
        values = objective(nb)
        best = int(np.argmax(values))
        if values[best] <= current:
            return x, current
        x, current = nb[best], float(values[best])


def anneal(
    starts: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    steps: int = 3000,
    t_start: float = 0.05,
    t_end: float = 0.001,
    max_move: int = 5,
) -> np.ndarray:
    """Simulated annealing on every start in parallel; returns the best allocation each chain visited."""
    X = np.array(starts, dtype=np.int64, copy=True)
    current = objective(X)
    best_X, best = X.copy(), current.copy()
    chains = np.arange(len(X))
    for t in np.geomspace(t_start, t_end, steps):
        i = rng.integers(0, N_FIELDS, len(X))
        j = (i + rng.integers(1, N_FIELDS, len(X))) % N_FIELDS
        k = np.minimum(rng.integers(1, max_move + 1, len(X)), X[chains, i])
        proposal = X.copy()
        proposal[chains, i] -= k
        proposal[chains, j] += k
        values = objective(proposal)
        delta = values - current
        accept = (k > 0) & ((delta >= 0) | (rng.random(len(X)) < np.exp(np.minimum(delta, 0) / t)))
        X[accept] = proposal[accept]
        current[accept] = values[accept]
        improved = current > best
        best_X[improved] = X[improved]
        best[improved] = current[improved]
    return best_X


def search(
    objective: Objective, starts: np.ndarray, rng: np.random.Generator, steps: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Anneal then polish every start; returns distinct local optima sorted best first, with their values."""
    found: dict[tuple[int, ...], float] = {}
    for x in anneal(starts, objective, rng, steps):
        optimum, value = polish(x, objective)
        found[tuple(optimum.tolist())] = value
    ordered = sorted(found.items(), key=lambda kv: -kv[1])
    return np.array([k for k, _ in ordered], dtype=np.int64), np.array([v for _, v in ordered])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/blotto/search.py tests/test_search.py
git commit -m "Add rating, simulated annealing, and steepest-ascent polish"
git push
```

---

### Task 6: Counter loop

**Files:**
- Create: `src/blotto/counters.py`
- Test: `tests/test_counters.py`

**Interfaces:**
- Consumes: `Group`, `make_group`, `random_splits`, `sample_field` (Task 3); `rating`, `search` (Task 5).
- Produces:
  - `pick_distinct(X: np.ndarray, k: int, min_l1: int) -> np.ndarray`: up to k rows of X (assumed sorted best first) that are pairwise at least min_l1 apart in L1 distance.
  - `counter_loop(groups: Mapping[str, Group], mix: Mapping[str, float], rng: np.random.Generator, rounds: int = 6, per_round: int = 5, min_l1: int = 6, chains: int = 32, steps: int = 3000, sample_size: int = 2000, lam: float = 0.5) -> np.ndarray`: the counters, shape (n, 10).
  - During the loop, the counters found so far form a `counters` group (with jittered copies) weighted by `mix["counters"]`; in round 1 that group does not exist yet and the mix renormalizes.

- [ ] **Step 1: Write the failing tests**

`tests/test_counters.py`:

```python
import numpy as np

from blotto.counters import counter_loop, pick_distinct
from blotto.pool import Field, Group
from blotto.search import metrics


def test_pick_distinct_respects_distance_and_order():
    X = np.array(
        [
            [10] * 10,
            [11, 9] + [10] * 8,
            [13, 7] + [10] * 8,
            [20, 0] + [10] * 8,
        ]
    )
    assert pick_distinct(X, k=3, min_l1=6).tolist() == [[10] * 10, [13, 7] + [10] * 8, [20, 0] + [10] * 8]
    assert pick_distinct(X, k=2, min_l1=6).tolist() == [[10] * 10, [13, 7] + [10] * 8]
    assert pick_distinct(X, k=5, min_l1=100).tolist() == [[10] * 10]


def test_counter_loop_produces_valid_counters_that_beat_the_crowd():
    rng = np.random.default_rng(0)
    crowd = Group(np.array([[10] * 10, [0, 0, 0, 0, 0, 20, 20, 20, 20, 20]]), np.array([0.5, 0.5]))
    counters = counter_loop(
        {"crowd": crowd}, {"crowd": 0.85, "counters": 0.15}, rng,
        rounds=2, per_round=2, chains=6, steps=200, sample_size=50,
    )
    assert counters.shape[1] == 10 and 2 <= len(counters) <= 4
    assert (counters.sum(axis=1) == 100).all() and (counters >= 0).all()
    win_rate, _ = metrics(counters[:1], Field(crowd.entries, crowd.weights))
    assert win_rate[0] >= 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_counters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.counters'`.

- [ ] **Step 3: Implement the counter loop**

`src/blotto/counters.py`:

```python
"""Counter loop: simulated applicants who best-respond to the rest of the field (fictitious play)."""

from collections.abc import Mapping

import numpy as np

from blotto.pool import Group, make_group, random_splits, sample_field
from blotto.scoring import N_FIELDS
from blotto.search import rating, search


def pick_distinct(X: np.ndarray, k: int, min_l1: int) -> np.ndarray:
    """Up to k rows of X (sorted best first), each at least min_l1 troops (L1) from those already picked."""
    chosen: list[np.ndarray] = []
    for x in X:
        if all(np.abs(x - c).sum() >= min_l1 for c in chosen):
            chosen.append(x)
            if len(chosen) == k:
                break
    return np.array(chosen, dtype=np.int64).reshape(-1, N_FIELDS)


def counter_loop(
    groups: Mapping[str, Group],
    mix: Mapping[str, float],
    rng: np.random.Generator,
    rounds: int = 6,
    per_round: int = 5,
    min_l1: int = 6,
    chains: int = 32,
    steps: int = 3000,
    sample_size: int = 2000,
    lam: float = 0.5,
) -> np.ndarray:
    """Allocations found by repeatedly best-responding to the pool plus every counter found so far."""
    counters = np.empty((0, N_FIELDS), dtype=np.int64)
    for _ in range(rounds):
        current = dict(groups)
        if len(counters):
            current["counters"] = make_group(counters, rng)
        field = sample_field(current, mix, sample_size, rng)
        X, _ = search(lambda Z, f=field: rating(Z, f, lam), random_splits(chains, rng), rng, steps)
        counters = np.vstack([counters, pick_distinct(X, per_round, min_l1)])
    return counters
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_counters.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/blotto/counters.py tests/test_counters.py
git commit -m "Add counter loop that simulates best-responding applicants"
git push
```

---

### Task 7: Robust selection

**Files:**
- Create: `src/blotto/robust.py`
- Test: `tests/test_robust.py`

**Interfaces:**
- Consumes: `Field`, `Group`, `random_splits`, `sample_field` (Task 3); `metrics`, `rating`, `search` (Task 5); `MAX_SCORE` (Task 1).
- Produces:
  - `BOUNDS: dict[str, tuple[float, float]]`, `LAM_RANGE = (0.25, 0.75)`, `LAMS = (0.25, 0.5, 0.75)`.
  - `group_names(groups: Mapping[str, Group]) -> list[str]`: groups present, in `BOUNDS` order.
  - `bounds_arrays(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]`: raises `ValueError` when no mix within the bounds sums to 1.
  - `center_mix(names: Sequence[str]) -> dict[str, float]`; `vertex_mixes(names: Sequence[str]) -> list[dict[str, float]]`.
  - `worst_case_value(ratings: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray`: (K, G) to (K,).
  - `worst_case_rating(win: np.ndarray, pts: np.ndarray, lo: np.ndarray, hi: np.ndarray, lam_range: tuple[float, float] = LAM_RANGE) -> np.ndarray`.
  - `group_metrics(X: np.ndarray, groups: Mapping[str, Group], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]`: (K, G) win rates and average points.
  - `generate_candidates(groups, names, rng, chains=32, steps=3000, sample_size=2000, seeds_per_setting=1) -> np.ndarray`.
  - `robust_search(groups, names, starts, rng, steps=3000, per_group_sample=400) -> np.ndarray`.
  - `Ranked(candidates, worst, win, pts)`: frozen dataclass, sorted by `worst` descending.
  - `rank_candidates(candidates: np.ndarray, groups, names, keep: int = 300) -> Ranked`.

- [ ] **Step 1: Write the failing tests**

`tests/test_robust.py`:

```python
import numpy as np
import pytest

from blotto.pool import Group, random_splits
from blotto.robust import (
    BOUNDS,
    LAM_RANGE,
    bounds_arrays,
    center_mix,
    group_metrics,
    group_names,
    rank_candidates,
    vertex_mixes,
    worst_case_rating,
    worst_case_value,
)

ALL = ["riddler", "heuristics", "chatbots", "random", "counters"]


def test_bounds_and_center_mix():
    lo, hi = bounds_arrays(ALL)
    assert lo.tolist() == [0.20, 0.10, 0.15, 0.00, 0.05]
    assert hi.tolist() == [0.50, 0.30, 0.50, 0.10, 0.25]
    mix = center_mix(ALL)
    assert np.isclose(sum(mix.values()), 1)
    assert all(BOUNDS[g][0] <= w <= BOUNDS[g][1] for g, w in mix.items())


def test_infeasible_bounds_raise():
    with pytest.raises(ValueError, match="random"):
        bounds_arrays(["random", "counters"])  # at most 0.35 in total


def test_vertex_mixes_are_feasible_extreme_points():
    V = vertex_mixes(ALL)
    assert len(V) == 21
    lo, hi = bounds_arrays(ALL)
    for mix in V:
        w = np.array([mix[g] for g in ALL])
        assert np.isclose(w.sum(), 1) and (w >= lo - 1e-9).all() and (w <= hi + 1e-9).all()
        assert np.sum((w > lo + 1e-9) & (w < hi - 1e-9)) <= 1


def test_greedy_worst_case_matches_brute_force_over_vertices():
    lo, hi = bounds_arrays(ALL)
    V = np.array([[m[g] for g in ALL] for m in vertex_mixes(ALL)])
    r = np.random.default_rng(0).random((100, 5))
    assert np.allclose(worst_case_value(r, lo, hi), (r @ V.T).min(axis=1))


def test_worst_case_rating_takes_the_worse_lambda_endpoint():
    lo, hi = bounds_arrays(ALL)
    win = np.full((1, 5), 0.9)
    pts = np.full((1, 5), 0.1 * 127.05)  # a points rating of 0.1 against every group
    expected = min(lam * 0.9 + (1 - lam) * 0.1 for lam in LAM_RANGE)
    assert np.isclose(worst_case_rating(win, pts, lo, hi)[0], expected)


def test_group_names_follow_bounds_order():
    g = Group(np.array([[10] * 10]), np.array([1.0]))
    assert group_names({"random": g, "riddler": g, "counters": g}) == ["riddler", "random", "counters"]


def test_rank_candidates_sorts_by_worst_case():
    rng = np.random.default_rng(0)
    names = ["riddler", "heuristics", "random", "counters"]  # no chatbots group: bounds stay feasible
    groups = {g: Group(random_splits(40, rng), np.full(40, 1 / 40)) for g in names}
    cands = random_splits(20, rng)
    ranked = rank_candidates(np.vstack([cands, cands[:3]]), groups, names, keep=10)
    assert len(ranked.candidates) == 10
    assert len({tuple(r) for r in ranked.candidates.tolist()}) == 10
    assert (np.diff(ranked.worst) <= 1e-12).all()
    lo, hi = bounds_arrays(names)
    assert np.allclose(ranked.worst, worst_case_rating(ranked.win, ranked.pts, lo, hi))
    win, pts = group_metrics(ranked.candidates, groups, names)
    assert np.allclose(win, ranked.win) and np.allclose(pts, ranked.pts)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_robust.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.robust'`.

- [ ] **Step 3: Implement robust selection**

`src/blotto/robust.py`:

```python
"""Robust selection: rate candidates per group and keep the best worst case over plausible field mixes."""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from blotto.pool import Field, Group, random_splits, sample_field
from blotto.scoring import MAX_SCORE
from blotto.search import metrics, rating, search

BOUNDS: dict[str, tuple[float, float]] = {
    "riddler": (0.20, 0.50),
    "heuristics": (0.10, 0.30),
    "chatbots": (0.15, 0.50),
    "random": (0.00, 0.10),
    "counters": (0.05, 0.25),
}
LAM_RANGE = (0.25, 0.75)
LAMS = (0.25, 0.5, 0.75)


def group_names(groups: Mapping[str, Group]) -> list[str]:
    """Names of the groups present, in BOUNDS order."""
    return [g for g in BOUNDS if g in groups]


def bounds_arrays(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Min and max weight per group; raises ValueError when no mix within the bounds sums to 1."""
    lo = np.array([BOUNDS[g][0] for g in names])
    hi = np.array([BOUNDS[g][1] for g in names])
    if lo.sum() > 1 + 1e-9 or hi.sum() < 1 - 1e-9:
        raise ValueError(
            f"weight bounds for {list(names)} cannot sum to 1 (minimums total {lo.sum():.2f}, maximums total {hi.sum():.2f})"
        )
    return lo, hi


def center_mix(names: Sequence[str]) -> dict[str, float]:
    """The midpoint of each group's bounds, normalized to sum to 1."""
    lo, hi = bounds_arrays(names)
    mid = (lo + hi) / 2
    return dict(zip(names, (mid / mid.sum()).tolist()))


def vertex_mixes(names: Sequence[str]) -> list[dict[str, float]]:
    """Extreme points of {lo <= w <= hi, sum(w) = 1}: every group at a bound except at most one."""
    lo, hi = bounds_arrays(names)
    found = set()
    for free in range(len(names)):
        others = [g for g in range(len(names)) if g != free]
        for at_max in itertools.product([False, True], repeat=len(others)):
            w = lo.copy()
            for g, up in zip(others, at_max):
                w[g] = hi[g] if up else lo[g]
            w[free] = 1 - w[others].sum()
            if lo[free] - 1e-9 <= w[free] <= hi[free] + 1e-9:
                found.add(tuple(float(v) for v in np.round(w, 9)))
    return [dict(zip(names, w)) for w in sorted(found)]


def worst_case_value(ratings: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per row, the minimum of ratings @ w over weights lo <= w <= hi summing to 1 (greedy fill)."""
    order = np.argsort(ratings, axis=1)
    w = np.tile(lo, (len(ratings), 1))
    remaining = np.full(len(ratings), 1 - lo.sum())
    rows = np.arange(len(ratings))
    for position in range(ratings.shape[1]):
        g = order[:, position]
        add = np.minimum(hi[g] - lo[g], remaining)
        w[rows, g] += add
        remaining -= add
    return (ratings * w).sum(axis=1)


def worst_case_rating(
    win: np.ndarray, pts: np.ndarray, lo: np.ndarray, hi: np.ndarray, lam_range: tuple[float, float] = LAM_RANGE
) -> np.ndarray:
    """Worst rating over mixes within the bounds and lam within lam_range; win and pts are (K, G)."""
    return np.minimum.reduce(
        [worst_case_value(lam * win + (1 - lam) * pts / MAX_SCORE, lo, hi) for lam in lam_range]
    )


def group_metrics(X: np.ndarray, groups: Mapping[str, Group], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Win rate and average points of each row of X against each named group, as (K, G) arrays."""
    results = [metrics(X, Field(groups[g].entries, groups[g].weights)) for g in names]
    return np.column_stack([r[0] for r in results]), np.column_stack([r[1] for r in results])


def generate_candidates(
    groups: Mapping[str, Group],
    names: Sequence[str],
    rng: np.random.Generator,
    chains: int = 32,
    steps: int = 3000,
    sample_size: int = 2000,
    seeds_per_setting: int = 1,
) -> np.ndarray:
    """Distinct local optima from searches under the center mix and every extreme mix, for each lam in LAMS."""
    found = []
    for mix in [center_mix(names), *vertex_mixes(names)]:
        for lam in LAMS:
            for _ in range(seeds_per_setting):
                field = sample_field(groups, mix, sample_size, rng)
                X, _ = search(lambda Z, f=field, l=lam: rating(Z, f, l), random_splits(chains, rng), rng, steps)
                found.append(X)
    return np.unique(np.vstack(found), axis=0)


def robust_search(
    groups: Mapping[str, Group],
    names: Sequence[str],
    starts: np.ndarray,
    rng: np.random.Generator,
    steps: int = 3000,
    per_group_sample: int = 400,
) -> np.ndarray:
    """Distinct local optima of the worst-case rating, with each group represented by a sample."""
    lo, hi = bounds_arrays(names)
    samples = {g: sample_field({g: groups[g]}, {g: 1.0}, per_group_sample, rng) for g in names}

    def objective(Z: np.ndarray) -> np.ndarray:
        results = [metrics(Z, samples[g]) for g in names]
        win = np.column_stack([r[0] for r in results])
        pts = np.column_stack([r[1] for r in results])
        return worst_case_rating(win, pts, lo, hi)

    X, _ = search(objective, starts, rng, steps)
    return X


@dataclass(frozen=True)
class Ranked:
    """Candidates sorted by worst-case rating (best first), with per-group metrics on the full groups."""

    candidates: np.ndarray  # (K, 10)
    worst: np.ndarray  # (K,)
    win: np.ndarray  # (K, G)
    pts: np.ndarray  # (K, G)


def rank_candidates(
    candidates: np.ndarray, groups: Mapping[str, Group], names: Sequence[str], keep: int = 300
) -> Ranked:
    """The top `keep` distinct candidates by worst-case rating against the full groups."""
    X = np.unique(np.asarray(candidates, dtype=np.int64), axis=0)
    lo, hi = bounds_arrays(names)
    win, pts = group_metrics(X, groups, names)
    worst = worst_case_rating(win, pts, lo, hi)
    order = np.argsort(-worst, kind="stable")[:keep]
    return Ranked(X[order], worst[order], win[order], pts[order])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_robust.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/blotto/robust.py tests/test_robust.py
git commit -m "Add robust selection with the pessimist's worst case"
git push
```

---

### Task 8: Validation

**Files:**
- Create: `src/blotto/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `Field`, `Group`, `jitter`, `random_splits` (Task 3); `bounds_arrays` (Task 7); `play` (Task 1); `metrics`, `rating`, `search` (Task 5).
- Produces:
  - `sample_mix(names: Sequence[str], rng: np.random.Generator) -> np.ndarray`: uniform over {lo <= w <= hi, sum(w) = 1}.
  - `mock_tournaments(finalists: np.ndarray, groups: Mapping[str, Group], names: Sequence[str], rng: np.random.Generator, n: int = 1000, size_range: tuple[int, int] = (100, 300)) -> dict[str, np.ndarray]`.
    Keys: `top_quartile_wins`, `top_quartile_points`, `top_quartile_both` (each (F,)), and `share_above_wins`, `share_above_points` (each (n, F)).
    `share_above_*` is the fraction of the field ranked strictly above the finalist, and top quartile means that fraction is below 0.25.
  - `heldout_report(pick: np.ndarray, seen: Field, heldout: Mapping[str, Field], lam: float = 0.5) -> dict[str, dict[str, float]]`.
  - `counter_check(pick: np.ndarray, pool_field: Field, rng: np.random.Generator, chains: int = 32, steps: int = 3000, copies: int = 50) -> dict`.
    Keys: `counter`, `pool_share_within_10`, `pick_win_rate_vs_counter_family`, `pick_avg_points_vs_counter_family`.

- [ ] **Step 1: Write the failing tests**

`tests/test_validate.py`:

```python
import numpy as np

from blotto.pool import Field, Group, random_splits
from blotto.robust import bounds_arrays
from blotto.scoring import play
from blotto.validate import counter_check, heldout_report, mock_tournaments, sample_mix

NAMES = ["riddler", "heuristics", "random", "counters"]
WEAK = np.array([[100] + [0] * 9])
STRONG = np.array([[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]])


def crowd_of(entry):
    return {g: Group(entry, np.array([1.0])) for g in NAMES}


def test_sample_mix_stays_within_bounds():
    rng = np.random.default_rng(0)
    lo, hi = bounds_arrays(NAMES)
    for _ in range(200):
        w = sample_mix(NAMES, rng)
        assert np.isclose(w.sum(), 1) and (w >= lo - 1e-12).all() and (w <= hi + 1e-12).all()


def test_round_robin_invariants():
    field = random_splits(60, np.random.default_rng(1))
    mine, theirs = play(field, field)
    wins, losses = mine > theirs, mine < theirs
    ties = ~wins & ~losses
    assert ((wins.sum(axis=1) + losses.sum(axis=1) + ties.sum(axis=1)) == 60).all()
    assert (wins == losses.T).all()
    assert (np.diag(mine) == 0).all()


def test_mock_tournaments_dominant_finalist_is_always_top():
    out = mock_tournaments(STRONG, crowd_of(WEAK), NAMES, np.random.default_rng(0), n=5, size_range=(10, 12))
    assert out["top_quartile_both"].tolist() == [1.0]
    assert out["share_above_wins"].shape == (5, 1) and (out["share_above_wins"] == 0).all()


def test_mock_tournaments_dominated_finalist_is_never_top():
    out = mock_tournaments(WEAK, crowd_of(STRONG), NAMES, np.random.default_rng(0), n=5, size_range=(10, 12))
    assert out["top_quartile_wins"].tolist() == [0.0]
    assert (out["share_above_wins"] == 1).all() and (out["share_above_points"] == 1).all()


def test_heldout_report_structure_and_values():
    pick = STRONG[0]
    seen = Field(np.array([[10] * 10]), np.array([1.0]))
    held = {"riddler_round_4": Field(np.array([[10, 0, 0, 0, 0, 0, 0, 30, 30, 30]]), np.array([1.0]))}
    report = heldout_report(pick, seen, held)
    assert set(report) == {"seen", "riddler_round_4"}
    assert report["seen"]["win_rate"] == 1.0 and np.isclose(report["seen"]["avg_points"], 92.40)
    assert report["riddler_round_4"]["win_rate"] == 0.0
    assert np.isclose(report["riddler_round_4"]["avg_points"], 21.45)


def test_counter_check_finds_a_counter_that_beats_the_pick():
    pick = np.array([10] * 10)
    pool = Field(np.array([[10] * 10, [0] * 9 + [100]]), np.array([0.5, 0.5]))
    result = counter_check(pick, pool, np.random.default_rng(0), chains=6, steps=300, copies=10)
    counter = np.array(result["counter"])
    mine, theirs = play(counter[None], pick[None])
    assert mine[0, 0] > theirs[0, 0]
    assert 0 <= result["pool_share_within_10"] <= 1
    assert 0 <= result["pick_win_rate_vs_counter_family"] <= 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.validate'`.

- [ ] **Step 3: Implement validation**

`src/blotto/validate.py`:

```python
"""Validation: mock tournaments, the held-out comparison, and the counter check."""

from collections.abc import Mapping, Sequence

import numpy as np

from blotto.pool import Field, Group, jitter, random_splits
from blotto.robust import bounds_arrays
from blotto.scoring import play
from blotto.search import metrics, rating, search

TOP_QUARTILE = 0.25


def sample_mix(names: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """A field mix drawn uniformly from {lo <= w <= hi, sum(w) = 1}."""
    lo, hi = bounds_arrays(names)
    while True:
        w = lo + (1 - lo.sum()) * rng.dirichlet(np.ones(len(names)))
        if (w <= hi + 1e-12).all():
            return w


def _draw_field(
    groups: Mapping[str, Group], names: Sequence[str], w: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    counts = rng.multinomial(size, w)
    return np.vstack(
        [
            groups[g].entries[rng.choice(len(groups[g].entries), size=c, p=groups[g].weights)]
            for g, c in zip(names, counts)
            if c > 0
        ]
    )


def mock_tournaments(
    finalists: np.ndarray,
    groups: Mapping[str, Group],
    names: Sequence[str],
    rng: np.random.Generator,
    n: int = 1000,
    size_range: tuple[int, int] = (100, 300),
) -> dict[str, np.ndarray]:
    """Play each finalist into n random realistic fields and record where it ranks on wins and on points."""
    finalists = np.atleast_2d(np.asarray(finalists, dtype=np.int64))
    above_wins = np.empty((n, len(finalists)))
    above_points = np.empty((n, len(finalists)))
    for t in range(n):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        field = _draw_field(groups, names, sample_mix(names, rng), size, rng)
        mine, theirs = play(field, field)
        field_wins = (mine > theirs).sum(axis=1)
        field_points = mine.sum(axis=1)
        f_mine, f_theirs = play(finalists, field)
        for f in range(len(finalists)):
            wins = (f_mine[f] > f_theirs[f]).sum()
            points = f_mine[f].sum()
            above_wins[t, f] = ((field_wins + (f_theirs[f] > f_mine[f])) > wins).sum() / size
            above_points[t, f] = ((field_points + f_theirs[f]) > points).sum() / size
    return {
        "top_quartile_wins": (above_wins < TOP_QUARTILE).mean(axis=0),
        "top_quartile_points": (above_points < TOP_QUARTILE).mean(axis=0),
        "top_quartile_both": ((above_wins < TOP_QUARTILE) & (above_points < TOP_QUARTILE)).mean(axis=0),
        "share_above_wins": above_wins,
        "share_above_points": above_points,
    }


def heldout_report(
    pick: np.ndarray, seen: Field, heldout: Mapping[str, Field], lam: float = 0.5
) -> dict[str, dict[str, float]]:
    """Win rate, average points, and rating at lam of the pick on the seen field and each held-out field."""
    x = np.asarray(pick, dtype=np.int64)[None]
    out = {}
    for name, field in {"seen": seen, **heldout}.items():
        win_rate, avg_points = metrics(x, field)
        out[name] = {
            "win_rate": float(win_rate[0]),
            "avg_points": float(avg_points[0]),
            "rating": float(rating(x, field, lam)[0]),
        }
    return out


def counter_check(
    pick: np.ndarray,
    pool_field: Field,
    rng: np.random.Generator,
    chains: int = 32,
    steps: int = 3000,
    copies: int = 50,
) -> dict:
    """The strongest counter to the pick, how much of the pool sits near it, and how the pick fares against it."""
    x = np.asarray(pick, dtype=np.int64)[None]
    target = Field(x, np.ones(1))
    X, _ = search(lambda Z: rating(Z, target, 0.5), random_splits(chains, rng), rng, steps)
    counter = X[0]
    near = np.abs(pool_field.entries - counter).sum(axis=1) <= 10
    family = np.vstack([counter[None], jitter(np.repeat(counter[None], copies, axis=0), rng)])
    win_rate, avg_points = metrics(x, Field(family, np.full(len(family), 1 / len(family))))
    return {
        "counter": counter.tolist(),
        "pool_share_within_10": float(pool_field.weights[near].sum()),
        "pick_win_rate_vs_counter_family": float(win_rate[0]),
        "pick_avg_points_vs_counter_family": float(avg_points[0]),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_validate.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/blotto/validate.py tests/test_validate.py
git commit -m "Add mock tournaments, held-out report, and counter check"
git push
```

---

### Task 9: Report figures

**Files:**
- Create: `src/blotto/report.py`
- Modify: `.gitignore` (add `.smoke/`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Field`, `Group` (Task 3); `bounds_arrays` (Task 7).
- Produces:
  - `price_list(field: Field, index: int, troops: np.ndarray) -> np.ndarray`: the weighted share of the field beaten at each troop count on one battlefield.
  - `pareto_mask(win: np.ndarray, pts: np.ndarray) -> np.ndarray`.
  - `plot_pool(groups: Mapping[str, Group], names: Sequence[str], mix: Mapping[str, float], path: Path) -> None`.
  - `plot_price_lists(field: Field, pick: np.ndarray, path: Path) -> None`.
  - `plot_candidates(win: np.ndarray, pts: np.ndarray, pick_index: int, path: Path) -> None`.
  - `plot_finalists(finalists: np.ndarray, ratings: np.ndarray, names: Sequence[str], path: Path) -> None`.
  - `plot_tournaments(above_wins: np.ndarray, above_points: np.ndarray, path: Path) -> None`.
  - `plot_allocation(pick: np.ndarray, beaten: np.ndarray, path: Path) -> None`.

Chart rules: these are static PNGs for a printed PDF, so the charts use a light surface only and have no hover layer.
The palette is blue `#2a78d6` with orange `#eb6834` (validated all-pairs for color vision deficiency) plus gray for context.
Each chart has one axis, text uses ink colors rather than series colors, and the heatmap uses a single-hue blue ramp.

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.report'`.

- [ ] **Step 3: Implement the figures**

`src/blotto/report.py`:

```python
"""Static figures for the writeup: light surface, validated palette, one axis per chart."""

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


def plot_pool(groups: Mapping[str, Group], names: Sequence[str], mix: Mapping[str, float], path: Path) -> None:
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


def plot_price_lists(field: Field, pick: np.ndarray, path: Path) -> None:
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
            ax.set_ylim(0, 1.02)
        fig.supxlabel("Troops placed on the battlefield", fontsize=9, color=INK_2)
        fig.supylabel("Share of field beaten", fontsize=9, color=INK_2)
        _save(fig, path)


def plot_candidates(win: np.ndarray, pts: np.ndarray, pick_index: int, path: Path) -> None:
    """Every candidate's win rate and average points, with the Pareto frontier and the pick."""
    front = pareto_mask(win, pts)
    order = np.argsort(win[front])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.scatter(win, pts, s=14, color=AXIS, label="candidate", zorder=2)
        ax.plot(win[front][order], pts[front][order], color=BLUE, linewidth=2, marker="o", markersize=5, label="Pareto frontier", zorder=3)
        ax.scatter([win[pick_index]], [pts[pick_index]], s=90, color=ORANGE, edgecolors=SURFACE, linewidths=2, label="final pick", zorder=4)
        ax.annotate(
            "final pick", (win[pick_index], pts[pick_index]), xytext=(-10, -14), textcoords="offset points",
            ha="right", va="top", color=INK, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"),
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(PERCENT)
        ax.set_xlabel("Win rate against the center-mix field")
        ax.set_ylabel("Average points per match")
        ax.set_title("Candidates: win rate versus points")
        ax.legend(loc="best")
        _save(fig, path)


def plot_finalists(finalists: np.ndarray, ratings: np.ndarray, names: Sequence[str], path: Path) -> None:
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
        _save(fig, path)


def plot_tournaments(above_wins: np.ndarray, above_points: np.ndarray, path: Path) -> None:
    """Distribution of the pick's rank across mock tournaments, by wins and by points."""
    bins = np.linspace(0, 1, 41)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        for ax, data, label in zip(axes, [above_wins, above_points], ["wins", "points"]):
            ax.hist(data, bins=bins, color=BLUE, rwidth=0.85)
            ax.axvline(0.25, color=INK_2, linestyle="--", linewidth=1)
            ax.text(0.26, 0.95, "top-quartile cutoff", transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK_2, va="top")
            ax.set_title(f"Rank by {label}: top quartile in {(data < 0.25).mean():.0%} of tournaments")
            ax.xaxis.set_major_formatter(PERCENT)
            ax.set_xlim(0, 1)
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("Share of the field ranked above the pick")
        axes[0].set_ylabel("Mock tournaments")
        _save(fig, path)


def plot_allocation(pick: np.ndarray, beaten: np.ndarray, path: Path) -> None:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Render sample figures from the real pool and look at them**

Add a line `.smoke/` to `.gitignore`, then run:

```bash
mkdir -p .smoke/figures && uv run python - << 'EOF'
from pathlib import Path
import numpy as np
from blotto import report
from blotto.pool import build_pool, field_from
from blotto.robust import center_mix, group_names
rng = np.random.default_rng(0)
groups = build_pool(Path("data"), rng)
names = group_names(groups)
field = field_from(groups, center_mix(names))
pick = np.array([1, 4, 7, 9, 13, 23, 26, 4, 6, 7])
out = Path(".smoke/figures")
report.plot_pool(groups, names, center_mix(names), out / "pool.png")
report.plot_price_lists(field, pick, out / "price_lists.png")
beaten = np.array([report.price_list(field, i, np.array([pick[i]]))[0] for i in range(10)])
report.plot_allocation(pick, beaten, out / "allocation.png")
EOF
```

Open the three PNGs in `.smoke/figures/` and check that no labels overlap, no text is clipped, and the titles fit.
Fix `report.py` and re-run Steps 4 and 5 if anything looks off.

- [ ] **Step 6: Commit and push**

```bash
git add src/blotto/report.py tests/test_report.py .gitignore
git commit -m "Add writeup figures"
git push
```

---

### Task 10: End-to-end pipeline and runner script

**Files:**
- Create: `src/blotto/pipeline.py`
- Create: `scripts/run_pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Config`: frozen dataclass with the run settings and `Config.quick()` for tests and smoke runs.
  - `Selection(groups, names, counters, ranked, finalists, tournaments, pick, pick_index)`: frozen dataclass; `pick_index` indexes both `finalists` and `ranked` rows.
  - `select_pick(config: Config, data_dir: Path, rng: np.random.Generator, exclude_riddler_rounds: Collection[int] = (), exclude_chatbot_models: Collection[str] = ()) -> Selection`.
  - `run(config: Config, data_dir: Path, out_dir: Path) -> dict`: writes `summary.json` and the six figures to `out_dir`, and returns the summary.
  - `scripts/run_pipeline.py [--quick] [--out PATH]`.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blotto.pipeline'`.

- [ ] **Step 3: Implement the pipeline**

`src/blotto/pipeline.py`:

```python
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
    def quick(cls) -> "Config":
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
```

`scripts/run_pipeline.py`:

```python
"""Run the strategy pipeline: uv run python scripts/run_pipeline.py [--quick] [--out PATH]."""

import argparse
import time
from pathlib import Path

from blotto.pipeline import Config, run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Maroon Capital 2026 strategy pipeline.")
    parser.add_argument("--quick", action="store_true", help="tiny settings for a smoke run")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="output directory")
    args = parser.parse_args()
    start = time.perf_counter()
    summary = run(Config.quick() if args.quick else Config(), ROOT / "data", args.out)
    print(f"pick: {summary['pick']}  ({time.perf_counter() - start:.0f}s, results in {args.out})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 1 passed (it takes about a minute).

- [ ] **Step 5: Smoke-run the script**

Run: `uv run python scripts/run_pipeline.py --quick --out .smoke/results`
Expected: prints `pick: [...]` with 10 integers summing to 100, and `.smoke/results/` contains `summary.json` and six PNGs.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass (46).

- [ ] **Step 7: Commit and push**

```bash
git add src/blotto/pipeline.py scripts/run_pipeline.py tests/test_pipeline.py
git commit -m "Add end-to-end pipeline and runner script"
git push
```

---

### Task 11: Full run, results, and README

**Files:**
- Create: `results/summary.json`, `results/*.png` (generated)
- Modify: `README.md`

**Interfaces:**
- Consumes: `scripts/run_pipeline.py` (Task 10).
- Produces: the committed results and the README the club's reviewers will see.

- [ ] **Step 1: Check which chatbot answers exist**

Run: `uv run python -c "from pathlib import Path; from blotto.pool import read_chatbot_answers; print({m: len(v) for m, v in read_chatbot_answers(Path('data/chatbots/answers.csv')).items()})"`
If `chatgpt` or `gemini` is missing, tell the user that this run uses only the answers present, and that the submission should come from a re-run after they add theirs.

- [ ] **Step 2: Run the full pipeline**

Run: `time uv run python scripts/run_pipeline.py`
Expected: prints `pick: [...]` in under 30 minutes and writes `results/summary.json` plus six PNGs.
If it takes longer than 30 minutes, stop and report the timing; the spec calls for adding a compiled evaluator in that case.

- [ ] **Step 3: Look at every figure**

Open each PNG in `results/` and check for overlapping labels, clipped text, and titles that do not fit.
Fix `report.py`, re-run `uv run pytest tests/test_report.py -q`, and repeat Step 2 if anything looks off.

- [ ] **Step 4: Sanity-check the summary**

Run:

```bash
uv run python - << 'EOF'
import json
s = json.load(open("results/summary.json"))
print("pick", s["pick"], "sum", sum(s["pick"]))
for f in s["finalists"]:
    print(f["allocation"], round(f["worst_case_rating"], 4), "top quartile (both)", f["top_quartile_both"])
print("held-out", json.dumps(s["heldout"], indent=1))
print("counter check", s["counter_check"])
EOF
```

Expected: the pick sums to 100, and the finalists' worst-case ratings decrease down the list.
In the held-out section, each held-out rating should be within about 0.05 of the matching `seen` rating; Riddler rounds differ by about that much on their own.
A held-out rating more than 0.05 below its `seen` rating suggests the search overfits; report it to the user before going further.

- [ ] **Step 5: Rewrite the README**

Replace `README.md` with the text below, then replace `PICK` with the `pick` list printed in Step 4:

````markdown
# Maroon Capital Challenge 2026

A computational strategy for the Maroon Capital 2026 challenge problem, a Colonel Blotto variant.
Each player splits 100 troops across 10 battlefields worth 1 to 10 points.
Bonus multipliers reward adjacent wins, base scores that are multiples of 5, and winning battlefield 6.
The full problem statement is in [`docs/problem.md`](docs/problem.md).

## Approach

1. Score every possible outcome exactly with a 1,024-entry lookup table.
2. Build a simulated applicant pool: real Riddler submissions, rules of thumb, chatbot answers, random splits, and counters that best-respond to the rest of the pool.
3. Search for the strongest allocations with simulated annealing and a steepest-ascent polish.
4. Pick the candidate with the best worst case when a pessimist chooses the pool mix and the wins-versus-points weighting.
5. Validate the pick with 1,000 mock tournaments, a held-out run, and a counter check.

The design is in [`docs/specs/`](docs/specs/), and the build plan is in [`docs/plans/`](docs/plans/).

## Results

The current pick is `PICK`.
The numbers for every finalist are in [`results/summary.json`](results/summary.json).

![Final allocation](results/allocation.png)

![Price lists](results/price_lists.png)

![Candidates](results/candidates.png)

![Mock tournaments](results/tournaments.png)

## Usage

```bash
uv sync
uv run pytest
uv run python scripts/run_pipeline.py
uv run python scripts/run_pipeline.py --quick --out .smoke/results
```

The full run takes about 5 minutes; the `--quick` run is a smoke test with tiny settings.

## Data

`data/riddler/` holds FiveThirtyEight's Riddler castle submissions, used under the Creative Commons Attribution 4.0 International License.
`data/chatbots/answers.csv` holds chatbot answers to the problem, collected as described in [`data/chatbots/README.md`](data/chatbots/README.md).
````

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass (46).

- [ ] **Step 7: Commit and push the results and README separately**

```bash
git add results
git commit -m "Add pipeline results"
git add README.md
git commit -m "Document approach, results, and usage in README"
git push
```
