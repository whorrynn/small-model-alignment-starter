"""Task 4 CLI entry point -- run ONE from-scratch DPO track at a time.

    python scripts/train_dpo.py --track hh_rlhf --sft_checkpoint outputs/sft/sft_run/checkpoints/best
    python scripts/train_dpo.py --track ultrafeedback --sft_checkpoint outputs/sft/sft_run/checkpoints/best
"""

from __future__ import annotations

import argparse

from src.config import DPO_CONFIG
from src.dpo_train import train_dpo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["hh_rlhf", "ultrafeedback"], required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--run_name", default=None, help="defaults to '<track>_run'")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    best_checkpoint = train_dpo(
        track=args.track,
        sft_checkpoint_path=args.sft_checkpoint,
        config=DPO_CONFIG,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    print(f"Best {args.track} DPO checkpoint:", best_checkpoint)


if __name__ == "__main__":
    main()
