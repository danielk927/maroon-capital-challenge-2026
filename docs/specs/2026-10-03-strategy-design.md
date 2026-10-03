# Strategy Design: Maroon Capital Challenge 2026

Date: 2026-10-03.
Submission deadline: Tuesday, October 6, 2026 at midnight, via Google Form.

## 1. Goal

Submit 10 non-negative integers summing to 100 that rank as high as possible in a round robin against every other applicant.
The ranking uses an unknown linear combination of number of wins and total points, so the strategy must be strong on both.
Finishing in the top quartile is the minimum bar, because only those writeups are reviewed.

The pipeline must also produce the evidence for a writeup of at most 2 A4 pages: figures, tables, and the insights behind the final allocation.
Writing the PDF itself is out of scope for this spec.

## 2. Game rules as implemented

- 10 battlefields with values 1 to 10, and 100 troops split into non-negative integers.
- A player wins a battlefield only with strictly more troops.
  Ties give nobody the battlefield.
- Base score: the sum of the values of the battlefields won.
- Rule 1: each won battlefield adjacent to another won battlefield counts 1.5 times its value, at most once.
- Rule 2: if the base score is a multiple of 5, the score is multiplied by 1.4.
- Rule 3: if battlefield 6 is won, the score is multiplied by 1.1.
- The player with the higher total score wins the match.

Interpretations, stated in the writeup:

- Battlefields sit in a line, so 1 and 10 are not adjacent.
- Equal total scores are a win for neither player.
- "Total points" means the sum of a player's own total scores across all matches.

## 3. Facts from exploration that shape the design

- Only 1,024 sets of battlefields can be won, so every score is precomputed.
  The maximum score is 127.05.
- Scores are stored as integers (score × 100), so equal scores compare as exactly equal.
- The score is not monotone: in 556 of 5,120 cases, winning one more battlefield lowers the score.
  For example, {6..10} scores 92.4 but {1, 6..10} scores 67.1.
  Small allocation changes can swing results sharply, so the search works on exact integer allocations.
- No set of 3 or fewer battlefields beats the complementary set, but 26 sets of 4 do.
  For example, {2,6,8,9} beats the remaining six battlefields 51.59 to 50.4.
- Battlefields 5 and 10 never change whether the base score is a multiple of 5.
- Battlefield 6 is worth its 6 points plus 10% of everything else.
- In 4,510 real submissions to FiveThirtyEight's version of this game, 27-40% of per-battlefield allocations are multiples of 5.
  One troop above a round number therefore often flips many ties into wins.

## 4. Approach overview

1. Score matches exactly with a lookup table.
2. Build a pool of simulated applicants in groups.
3. Search for the allocations with the best rating against the pool.
4. Pick the candidate with the best worst case over plausible pool mixes and rating weights.
5. Validate the pick with mock tournaments, a held-out test, and a counter check.

Rejected alternatives:

- Reinforcement learning as the solver.
  The game is a single simultaneous move and any allocation can be rated exactly in milliseconds, so RL would be a slower, noisier optimizer of the same objective.
- A genetic algorithm tournament.
  Blotto populations cycle, and the end state resembles a field of optimizers rather than real applicants.
- Pure Nash equilibrium play.
  One fixed allocation is submitted into a field that is mostly not playing the equilibrium.

## 5. Components

### 5.1 Scoring (`blotto/scoring.py`)

- `SCORE_TABLE`: an integer array of length 1,024.
  Bit i of the index means battlefield i+1 was won, and the value is the score × 100.
- `play(X, Y)`: for candidates X (K × 10) and opponents Y (N × 10), returns both sides' scores (K × N) as integers.
  Large inputs are processed in chunks to bound memory.
- Match results derive from the scores: a win if mine > theirs, a loss if mine < theirs, and a tie otherwise.

### 5.2 Applicant pool (`blotto/pool.py`)

A pool is a set of named groups.
Each group is an integer array of allocations (n × 10) with equal weight inside the group.
A field mix assigns a weight to each group, and each opponent's weight is its group's weight divided by the group size.

| Group | Who it represents | Source |
|---|---|---|
| `riddler` | People playing on instinct | FiveThirtyEight riddler-castles rounds 1-4, 4,510 valid entries, committed to `data/riddler/` with CC BY 4.0 attribution |
| `heuristics` | Someone who thinks for 10 minutes | Generated as described below |
| `chatbots` | Applicants who ask an AI | `data/chatbots/answers.csv` |
| `random` | Noise | 1,000 splits drawn as multinomial(100, Dirichlet(α)) with α in {0.5, 1, 2} |
| `counters` | Applicants who think a few steps ahead | Produced by the counter loop in 5.4 |

Heuristics are generated, not hand-typed:

- An even split (10 each).
- Splits proportional to value^p for p in {0.5, 1, 2}, rounded to sum to 100.
- All-in on the top k battlefields for k from 3 to 6, split evenly.
- For each contiguous block whose base score is a multiple of 5 (for example 6-10, 5-10, 4-8, 6-9, 3-7, and 2-6), three variants.
  The variants are an even split over the block, a "heavy on 6" split, and a split with each allocation nudged one troop above a multiple of 5 where the budget allows.

Chatbot answers:

- Columns: `model`, `prompt_variant`, `sample`, `b1` to `b10`.
- Claude answers come from fresh subagents that see only the problem text.
- ChatGPT and Gemini answers are collected by pasting the problem text into each chatbot, since no API keys are configured.
- The target is at least 15 answers per model.
  Prompts are the problem text verbatim plus one short variant that ends with "What allocation should I submit?".

Jittered copies:

- Every group except `random` also gets copies of its entries with 1 to 3 troops moved between random battlefields, once or twice per copy.
- Within a group, 30% of the weight goes to jittered copies and 70% to originals.
  This keeps the search from relying on beating one exact number by one troop.

### 5.3 Rating and search (`blotto/search.py`)

Rating of an allocation x against a weighted pool:

- Win rate: the weighted share of opponents that x beats.
- Average points: the weighted mean of x's total score.
- Rating(λ) = λ × win rate + (1 − λ) × average points / 127.05.

Search:

- Each run starts from a random split, a heuristic, or a previous candidate.
- Simulated annealing phase: propose moving 1 to 5 troops from one battlefield to another.
  Improvements are always accepted, a worse move is accepted with probability exp(Δ / T), and T cools geometrically.
- Polish phase: steepest ascent over every shift of 1 to 3 troops between any two battlefields, until nothing improves.
  The up to 270 neighbors are rated in one batch.
  This certifies that no small change helps.
- Runs rate against a sample of about 2,000 opponents, stratified by group, for speed.
  Finalists are re-rated on the full pool.
- All randomness uses explicit seeds, so results are reproducible.

### 5.4 Counter loop (`blotto/counters.py`)

This simulates applicants who best-respond to the rest of the field.
It is fictitious play, equivalently PSRO with an exact best-response step.

- Start from the non-strategic groups, weighted at the centers of their bounds and normalized to sum to 1.
- Each round, search for the best allocations against the current pool.
  Keep the 5 best distinct results, meaning a pairwise L1 distance of at least 6 troops, and add them to `counters`.
- Run 6 rounds, producing 30 counters.
- Each round rates with λ = 0.5.

### 5.5 Robust selection (`blotto/robust.py`)

Candidates:

- Search results under many field mixes, λ in {0.25, 0.5, 0.75}, and several seeds.
  The field mixes are the center and the extreme points of the weight bounds.
- A direct search that maximizes the worst-case rating defined below.
- The 300 best distinct candidates are kept.

Worst case:

- Each candidate is rated against each group separately.
- The pessimist chooses group weights within the bounds below, and λ within [0.25, 0.75], to minimize the candidate's rating.
- Because the rating is linear in the weights, the pessimist's best weights come from a greedy fill.
  Every group gets its minimum, then the remaining weight goes to the groups where the candidate is weakest, up to each maximum.
  The worst λ is always an endpoint of its range.
- The pick is the candidate with the highest worst-case rating.
  Candidates within 0.005 of the best are compared by their mock tournament top-quartile frequency.

Initial weight bounds, which are judgment calls stated in the writeup:

| Group | Min | Max |
|---|---|---|
| `riddler` | 20% | 50% |
| `heuristics` | 10% | 30% |
| `chatbots` | 15% | 50% |
| `random` | 0% | 10% |
| `counters` | 5% | 25% |

### 5.6 Validation (`blotto/validate.py`)

- Mock tournaments, repeated 1,000 times: draw group weights uniformly within the bounds and a field size between 100 and 300.
  Sample that many applicants, add the finalist, play every matchup, and record the finalist's rank by wins and by points.
  Report the top-quartile frequency for each metric and for both at once.
- Held-out test: a separate diagnostic run of the full pipeline without Riddler round 4 and one chatbot model's answers.
  Its pick is rated against the excluded entries and compared with how it rates on the data it saw.
  The final pick comes from the run on all data.
- Counter check: search for the best response to the pick alone.
  Report the share of the pool within 10 troops (L1 distance) of that counter, and the pick's rating against the counter and its jittered copies.

### 5.7 Report (`blotto/report.py`)

Outputs go to `results/`, as JSON for numbers and PNG for figures:

- Pool composition.
- Price lists: the share of the pool beaten at each troop count, for each battlefield.
- Win rate versus average points for all candidates, with the Pareto frontier marked.
- Each finalist's rating against each group, as a heatmap.
- Mock tournament rank distributions.
- The final allocation, with the share of the pool beaten on each battlefield.

## 6. Data flow

`data/` → pool groups → counter loop → candidate search → robust selection → validation → `results/`.
One command, `uv run python scripts/run_pipeline.py`, runs everything from the committed data.

## 7. Testing

- Scoring: the problem's example ({3,5,6,7} scores 33), the full set (127.05), the non-monotone example (92.4 versus 67.1), the empty set (0), and battlefield 10 alone (14.0).
- Scoring cross-check: the integer table matches an independent exact-fraction implementation on all 1,024 sets.
- `play`: hand-checked matchups, such as 20 each on 6-10 beating the proportional split 92.40 to 31.50.
  Swapping the two sides swaps the scores.
- Search: against a single opponent, the result beats it.
  The polish output has no improving shift of 1 to 3 troops.
  Fixed seeds reproduce identical results.
- Robust selection: the greedy worst case matches a brute-force grid over weights on small examples.
- Validation: wins plus losses plus ties equals the number of matches, and round robin results are antisymmetric.

## 8. Repository layout and tooling

```
src/blotto/      scoring, pool, search, counters, robust, validate, report
scripts/         run_pipeline.py
data/            riddler/, chatbots/
results/         generated outputs, committed for the writeup
tests/
docs/specs/      this document
docs/plans/      the implementation plan
```

- Python 3.14 with uv, numpy, matplotlib, and pytest.
- Vectorized numpy throughout.
  A compiled evaluator is added only if profiling shows the full pipeline takes more than 30 minutes.
- Small, focused commits are pushed to `github.com/danielk927/maroon-capital-challenge-2026` as each component lands with its tests.

## 9. Risks

- The pool model is wrong.
  The robust selection, the held-out test, and the weight bounds limit the damage.
- The search overfits to exact numbers in the data.
  Jittered copies and the held-out test guard against this.
- The repository is public before the deadline, so other applicants could find the final numbers.
  This was an explicit choice.
