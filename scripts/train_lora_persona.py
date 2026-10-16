"""Task 5 CLI entry point.

    python scripts/train_lora_persona.py
"""

from __future__ import annotations

import os

# TRL's DPOTrainer refuses to run under nn.DataParallel, which HF Trainer
# silently uses whenever a single process sees more than one GPU (Kaggle T4 x2).
# Must be set before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.lora_persona_train import train_lora_persona  # noqa: E402

if __name__ == "__main__":
    train_lora_persona()
