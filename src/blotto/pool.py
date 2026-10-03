"""Simulated applicant pool: loading allocations, building groups, and weighted fields."""

import csv
import io
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blotto.scoring import BUDGET, N_FIELDS

RIDDLER_FILES = {
    1: "castle-solutions.csv",
    2: "castle-solutions-2.csv",
    3: "castle-solutions-3.csv",
    4: "castle-solutions-4.csv",
}

KNOWN_CHATBOTS = ("chatgpt", "claude", "gemini")
CHATBOT_COLUMNS = ["model", *(f"b{i}" for i in range(1, N_FIELDS + 1))]
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


def _valid(a: list[float]) -> bool:
    return len(a) == N_FIELDS and all(math.isfinite(v) and v >= 0 and v == int(v) for v in a) and sum(a) == BUDGET


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


def _parse_chatbot_answers(path: Path) -> tuple[dict[str, list[list[int]]], int]:
    """Valid rows keyed by lower-case model name, plus the number of rows skipped as invalid."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0] if text.strip() else ",", delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    header = [name.strip().lower() for name in next(reader, [])]
    missing = [name for name in CHATBOT_COLUMNS if name not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)} (found: {', '.join(header)})")
    index = {name: header.index(name) for name in CHATBOT_COLUMNS}
    by_model: dict[str, list[list[int]]] = {}
    skipped = 0
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue  # blank line
        try:
            model = row[index["model"]].strip().lower()
            a = [float(row[index[f"b{i}"]]) for i in range(1, N_FIELDS + 1)]
        except (IndexError, ValueError):
            skipped += 1
            continue
        if model and _valid(a):
            by_model.setdefault(model, []).append([int(v) for v in a])
        else:
            skipped += 1
    return by_model, skipped


def read_chatbot_answers(path: Path) -> dict[str, np.ndarray]:
    """Valid allocations from answers.csv keyed by lower-case model name; invalid rows are skipped.

    Tolerates a byte-order mark, stray spaces, capitalized headers, and comma, semicolon, or tab delimiters.
    Raises ValueError when a required column (model, b1 to b10) is missing, since then no row could be read.
    """
    if not path.exists():
        return {}
    by_model, _ = _parse_chatbot_answers(path)
    return {model: np.array(rows, dtype=np.int64) for model, rows in by_model.items()}


def chatbot_answer_report(path: Path) -> dict:
    """Rows kept per model, rows skipped as invalid, and model names outside KNOWN_CHATBOTS."""
    if not path.exists():
        return {"kept": {}, "skipped_rows": 0, "unknown_models": []}
    by_model, skipped = _parse_chatbot_answers(path)
    return {
        "kept": {model: len(rows) for model, rows in sorted(by_model.items())},
        "skipped_rows": skipped,
        "unknown_models": sorted(model for model in by_model if model not in KNOWN_CHATBOTS),
    }


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
