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
