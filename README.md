# Maroon Capital Challenge 2026

A computational strategy for the Maroon Capital 2026 challenge problem, a Colonel Blotto variant.
Each player splits 100 troops across 10 battlefields worth 1 to 10 points.
Bonus multipliers reward adjacent wins, base scores that are multiples of 5, and winning battlefield 6.

The approach: build a realistic pool of simulated opponents, search for the allocations that perform best against it, and pick the one that stays strong when the assumptions about the pool are wrong.

The design is in [`docs/specs/`](docs/specs/).

Status: in progress.
