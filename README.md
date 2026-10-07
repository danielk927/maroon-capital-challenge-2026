# Maroon Capital Challenge 2026

This is my write-up, hope you guys enjoy!

## Result

The pick is `[1, 2, 4, 1, 16, 29, 3, 4, 33, 7]`.
Against the simulated field it beats 93% of opponents, and it finishes in the top quartile by wins and by points in all 1,000 mock tournaments.
It beats all 20 chatbot answers collected, in which GPT-6 Astra (Medium) and Opus 5.5 (High) were used.
The numbers for every finalist are in [`results/summary.json`](results/summary.json).

## How the pick was found

### 1. Treat it as a prediction problem

This problem does not have a definitive answer that wins, since every answer will always have a counter.
So I decided that the best submission depends on what the other applicants submit, and the pipeline works in three steps: model the field, search for the allocation that does best against it, and check that the answer doesn't do too poorly if the model is wrong.


### 2. Work out what the scoring rewards

Code: [`src/blotto/scoring.py`](src/blotto/scoring.py).

- Each player wins one of 2^10 = 1,024 sets of battlefields. The multipliers are 3/2, 7/5 and 11/10, so scores are stored as integers times 100 and ties compare exactly.
  The maximum score is 127.05.
- The multipliers compound. A set in which every battlefield has a won neighbor, whose base is a multiple of 5, and which includes battlefield 6 scores 1.5 × 1.4 × 1.1 = 2.31 times its base.
- Rule 2 depends only on values modulo 5.
- Winning more can score less.
- A player can win a match while losing the majority (six) battlefields


The pairs {2,3}, {5,6} and {9,10} have values summing to 35, and winning all 6 battlefield creates 35 * 2.31 = 80.85, which is ~64% of the maximum 127.05 when you win all 10 battlefields.
Less emphasis can be put on battlefields 5 and 10, because the base stays a multiple of 5: {2, 3, 6, 9, 10} still scores 64.68 and {2, 3, 5, 6, 9} scores 50.82.

### 3. Model the other applicants

Code: [`src/blotto/pool.py`](src/blotto/pool.py).

The simulated field has five groups.
Each group gets a range for its share of the field instead of a single number, because the shares are judgment calls.

| Group | Who it represents | Source | Share of the field |
|---|---|---|---|
| Chatbot answers | Applicants who ask an AI | 20 answers collected by hand, 10 from Claude and 10 from ChatGPT, each from a fresh chat with the problem pasted verbatim | 85-98% |
| Riddler entries | People playing on instinct | 4,510 submissions to FiveThirtyEight's Battle for Riddler Nation, the same board without the multipliers | 0-5% |
| Rules of thumb | Someone who thinks for ten minutes | 44 generated allocations: even splits, value-proportional splits, all-in on the top k battlefields, blocks whose base is a multiple of 5 | 0-5% |
| Random splits | Noise | 1,000 draws from a Dirichlet-multinomial | 0-2% |
| Counters | Applicants who think a step ahead | 30 best responses from the counter loop below | 2-10% |

Chatbot answers get most of the weight because AI use is encouraged, and I'd assume most, if not all applicants would use LLMs to generate their answer.
Most put 18 to 27 troops on battlefield 6, 21 to 32 on 9, 14 to 31 on 7 and 8 when they contest them, and at most 3 on battlefields 1 and 2.
Riddler entries are capped at 5% because Riddler Nation used plain scoring which do not use these multipliers, so those players are a weak model for this game.

Every entry except the random ones also gets a jittered copy, with 1 to 3 troops moved once or twice, and the copies carry 30% of each group's weight.
Real applicants may tweak the answers they get, so the jitter keeps the search from relying on beating one exact number by one troop.
The center mix uses the midpoint of each range, rescaled to sum to 100%, which puts chatbot answers at 88%.

### 4. Rate an allocation

Code: [`src/blotto/search.py`](src/blotto/search.py).

Final score is ranked on an unknown linear combination of wins and total points, so each allocation x gets a rating against a weighted field:

R(x) = λ · W(x) + (1 − λ) · P(x) / 127.05

W is the weighted win rate and P the weighted average points per match.
Dividing by the maximum score puts points on the same 0 to 1 scale as the win rate.
λ is unknown, so the selection step treats it as anywhere between an estimate of 0.25 and 0.75.

### 5. Search for strong allocations

Code: [`src/blotto/search.py`](src/blotto/search.py).

There are about 4.3 trillion ways to split 100 troops across 10 battlefields, which is far too many to check.
Simulated annealing runs 32 chains in parallel from random starts, each for 3,000 steps that move 1 to 5 troops between two battlefields.
It always keeps improvements and sometimes keeps a worse move while the temperature is high, so chains can escape dead ends.
A steepest-ascent polish then tries all 270 shifts of 1 to 3 troops and takes the best, repeating until no small change helps.
For speed the searches rate against a sample of 2,000 opponents, stratified by group, and the candidates are re-rated on the full pool of 10,208 entries.

### 6. Simulate strategic applicants

Code: [`src/blotto/counters.py`](src/blotto/counters.py).

Some applicants will try to beat the obvious answers.
The counter loop runs fictitious play, where it searches for the best responses to the current pool, adds the five best that differ by at least 6 troops, and repeats for six rounds.
The resulting 30 counters make sure the pick also faces people who try to out-think the chatbots.

### 7. Choose the pick that survives the worst case

Code: [`src/blotto/robust.py`](src/blotto/robust.py).

Candidates come from searches under the center mix and under every extreme mix of the ranges, for λ = 0.25, 0.5 and 0.75, plus two searches that maximize the worst case below, plus the counters, and the 300 best distinct candidates move on.

A pessimist then chooses the group shares inside the ranges and λ inside [0.25, 0.75] to make each candidate look as bad as possible, and the pick is the candidate whose worst case is best.
The rating is linear in the shares, so the pessimist's best move is a greedy fill: give every group its minimum, then pour the rest into the groups where the candidate does worst.
The worst λ is always one of the two endpoints.

The candidates within 0.005 of the best worst case, up to ten, become finalists, and each plays 1,000 mock tournaments.
The pick has the best top-quartile rate, with the worst case breaking ties.
This time ten finalists all made the top quartile in every tournament, and they came in three shapes: the pick and three close variants, two that concede battlefield 6 for 8 and 9, and four that buy 7 and 8 instead of 6 and 9.
The pick's shape had the best worst case (0.598) and beat more Riddler entries than the other shapes.

### 8. Validate

Code: [`src/blotto/validate.py`](src/blotto/validate.py).

- **Mock tournaments.**
  1,000 fields of 100 to 300 applicants, with group shares drawn uniformly inside the ranges, each playing a full round robin.
  The pick finished in the top quartile by wins and by points in all 1,000.
  The median share of the field ranked above it by wins was 3.5%, and in 72% of tournaments no applicant scored more total points.
- **Held-out data.**
  A rerun that never saw ChatGPT's answers or Riddler round 4 chose `[2, 3, 3, 6, 14, 28, 3, 4, 31, 6]`, the same shape.
  It beat 80% of the unseen ChatGPT answers, against 100% of the Claude answers it trained on, so expect some overfitting to the collected answers.
- **Counter check.**
  The best response to the pick alone, `[0, 4, 3, 4, 22, 0, 12, 10, 36, 9]`, beats it by giving up battlefields 1 and 6 and outbidding it almost everywhere else.
  No entry in the simulated field lies within 10 troops of that counter, so beating the pick this way takes someone who targets its exact numbers.

### 9. How the pick changed

The first full run used wider ranges: chatbots 70-95%, Riddler up to 20%, rules of thumb up to 15%, random up to 5%, and counters 5-15%.
It chose `[1, 4, 6, 1, 7, 28, 3, 5, 33, 12]`, the same three pairs with more troops on battlefield 10 and fewer on 5.
Because Riddler Nation used different rules, the second run narrowed the ranges to the ones in the table above and chose the current pick.

[`scripts/compare_picks.py`](scripts/compare_picks.py) compares the two picks on the same simulated pool, under both sets of ranges, and writes [`results/comparison.json`](results/comparison.json).
Under the old ranges, the current pick still makes the top quartile in all 1,000 mock tournaments, with a median of 5.7% of the field above it by wins against 10.3% for the earlier pick.
The cost is robustness to strategic players: the current pick beats 28% of the counters, while the earlier pick beats 57%.

### Why the pick works

| Battlefield | Troops | Field beaten | Reason |
|---|---|---|---|
| 1 | 1 | 45% | Ties the many one-troop bids, which decides matches (see below) |
| 2 | 2 | 81% | 18 of 20 chatbot answers bid 0 or 1 here |
| 3 | 4 | 77% | 18 of 20 chatbot answers bid 3 or fewer here |
| 4 | 1 | 12% | Ties one-troop bids, and with battlefield 1 keeps the base a multiple of 5 |
| 5 | 16 | 91% | Matches the highest chatbot bid; worth 0 mod 5, so losing it never costs the 1.4 |
| 6 | 29 | 97% | Two troops above the highest chatbot bid; triggers the 1.1 |
| 7 | 3 | 32% | Conceded: chatbots that contest it spend 14 to 25 |
| 8 | 4 | 20% | Conceded: chatbots that contest it spend 19 to 31 |
| 9 | 33 | 98% | One troop above the highest chatbot bid |
| 10 | 7 | 58% | Beats the low bids; worth 0 mod 5, so losing it never costs the 1.4 |

The pick's most common outcome is the full target set {2,3,5,6,9,10}, in 21% of matches.
Its base is a multiple of 5 in 52% of matches and it wins battlefield 6 in 97%, against 31% and 46% for the average chatbot answer.

The single troop on battlefield 1 shows how relative scoring works.
Against the ChatGPT answer `[1, 1, 3, 4, 1, 1, 21, 19, 28, 21]`, it ties their troop and holds their base at 29, so the pick wins 50.82 to 36.5.
Without it they win battlefield 1, their base reaches 30, their 1.4 multiplier fires, and the pick loses 50.82 to 52.5.
Moving that troop anywhere else lowers the win rate from 93.0% to 85.7% or less.

### Assumptions and limits

- Battlefields sit on a line, so 1 and 10 are not adjacent; equal totals are a win for neither player; and total points means the sum of a player's own scores.
- The chatbot share is the biggest judgment call.
  The pick beats 98.5% of chatbot answers but only 28% of the counters, so a field with many applicants running their own optimization would hurt it.
- The chatbot sample is small: 20 answers from two models and one prompt.
- The mock tournaments draw from the same pool that produced the pick, so they are optimistic; the held-out run is the fairer estimate.

The original design is in [`docs/specs/`](docs/specs/), and the build plan is in [`docs/plans/`](docs/plans/).

## Figures

![Final allocation](results/allocation.png)

![Price lists](results/price_lists.png)

![Candidates](results/candidates.png)

![Mock tournaments](results/tournaments.png)

`results/figure_data/` holds the data behind the figures as CSV files.

## Usage

```bash
uv sync
uv run pytest
uv run ruff check
uv run python scripts/run_pipeline.py
uv run python scripts/run_pipeline.py --quick --out .smoke/results
uv run python scripts/compare_picks.py
```

The full run takes about 90 seconds, and the comparison about a minute; the `--quick` run is a smoke test with tiny settings.

## Data

`data/riddler/` holds FiveThirtyEight's Riddler castle submissions, used under the Creative Commons Attribution 4.0 International License.
`data/chatbots/answers.csv` holds chatbot answers to the problem, collected as described in [`data/chatbots/README.md`](data/chatbots/README.md).
