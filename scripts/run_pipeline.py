"""Run the strategy pipeline: uv run python scripts/run_pipeline.py [--quick] [--out PATH]."""

import argparse
import time
from pathlib import Path

from blotto.pipeline import Config, run
from blotto.pool import chatbot_answer_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Maroon Capital 2026 strategy pipeline.")
    parser.add_argument("--quick", action="store_true", help="tiny settings for a smoke run")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="output directory")
    args = parser.parse_args()
    answers = chatbot_answer_report(ROOT / "data" / "chatbots" / "answers.csv")
    kept = ", ".join(f"{model} {n}" for model, n in answers["kept"].items()) or "none"
    print(f"chatbot answers: {kept} ({answers['skipped_rows']} invalid rows skipped)")
    if answers["unknown_models"]:
        print(f"warning: unexpected chatbot model names: {', '.join(answers['unknown_models'])}")
    start = time.perf_counter()
    summary = run(Config.quick() if args.quick else Config(), ROOT / "data", args.out)
    print(f"pick: {summary['pick']}  ({time.perf_counter() - start:.0f}s, results in {args.out})")


if __name__ == "__main__":
    main()
