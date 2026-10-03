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
