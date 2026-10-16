"""Task 7 -- judge pairwise comparison files using a Kaggle model-proxy judge.

Matches the manual's example invocation exactly:

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash \
        --batch_size 5 \
        --sleep_seconds 5 \
        --max_retries 3 \
        --reuse_judgments

Must be run from a Kaggle notebook configured as a "utility benchmark" with
model-proxy access enabled (manual Section 13.1) -- it will not work from a
plain local or Kaggle-without-benchmark-access environment.
"""

from __future__ import annotations

import argparse

from src.judging import run_judging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true")
    parser.add_argument(
        "--safety_mode",
        action="store_true",
        help="Append the safety addendum to the judge prompt (manual Section 13.2) -- use for the "
        "hh_rlhf-eval-set track, not for AlpacaEval/UltraFeedback.",
    )
    args = parser.parse_args()

    summary = run_judging(
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        reuse_judgments=args.reuse_judgments,
        safety_mode=args.safety_mode,
    )
    print(summary)


if __name__ == "__main__":
    main()
