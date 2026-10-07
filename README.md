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

The current pick is `[1, 2, 4, 1, 16, 29, 3, 4, 33, 7]`.
It is optimized against a field where chatbot answers (10 from Claude, 10 from ChatGPT) make up 85-98% of applicants.
Riddler entries get at most 5%, because Riddler Nation used plain scoring without the multipliers.
An earlier run with wider ranges (chatbots 70-95%, Riddler up to 20%) chose `[1, 4, 6, 1, 7, 28, 3, 5, 33, 12]`; `scripts/compare_picks.py` compares the two under both sets of ranges and writes `results/comparison.json`.
The numbers for every finalist are in [`results/summary.json`](results/summary.json).

![Final allocation](results/allocation.png)

![Price lists](results/price_lists.png)

![Candidates](results/candidates.png)

![Mock tournaments](results/tournaments.png)

## Writeup

The writeup is [`writeup/writeup.pdf`](writeup/writeup.pdf), built from [`writeup/writeup.tex`](writeup/writeup.tex).
Its figures are drawn with pgfplots from the CSV files the pipeline writes to `results/figure_data/`.

## Usage

```bash
uv sync
uv run pytest
uv run ruff check
uv run python scripts/run_pipeline.py
uv run python scripts/run_pipeline.py --quick --out .smoke/results
uv run python scripts/compare_picks.py
tectonic writeup/writeup.tex
```

The full run takes about 90 seconds, and the comparison about a minute; the `--quick` run is a smoke test with tiny settings.
The writeup builds with [Tectonic](https://tectonic-typesetting.github.io/) (`brew install tectonic`), which fetches the LaTeX packages it needs on first use.

## Data

`data/riddler/` holds FiveThirtyEight's Riddler castle submissions, used under the Creative Commons Attribution 4.0 International License.
`data/chatbots/answers.csv` holds chatbot answers to the problem, collected as described in [`data/chatbots/README.md`](data/chatbots/README.md).
