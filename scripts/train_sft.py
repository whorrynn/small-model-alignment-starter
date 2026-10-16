"""Task 3 CLI entry point.

    python scripts/train_sft.py --run_name sft_run
"""

from __future__ import annotations

import argparse

from src.config import SFT_CONFIG
from src.sft_train import train_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="sft_run")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    best_checkpoint = train_sft(config=SFT_CONFIG, run_name=args.run_name, output_dir=args.output_dir)
    print("Best SFT checkpoint:", best_checkpoint)


if __name__ == "__main__":
    main()
