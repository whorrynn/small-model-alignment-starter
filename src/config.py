"""Suggested hyperparameters, transcribed from the manual's reference tables
(Sections 9.2 and 10.4).

These are starting points, not fixed requirements -- see the manual's
"Guide Configurations, Not Sacred Hyperparameters" box. Edit freely, but if
you do, record the guide value, your value, and why in REPORT.md Section
14.3 (Parameter Choices and Deviations).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    context_length: int = 512
    train_examples: int = 30_000
    val_examples: int = 1_000
    per_device_batch_size: int = 2
    gradient_accumulation: int = 16
    learning_rate: float = 2e-5
    warmup_fraction: float = 0.03
    schedule: str = "cosine"
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 1
    seed: int = 42


@dataclass
class DPOConfig:
    beta: float = 0.1
    train_pairs: int = 15_000
    val_pairs: int = 1_000
    context_length: int = 512
    max_prompt_length: int = 256
    per_device_batch_size: int = 1
    gradient_accumulation: int = 32
    learning_rate: float = 1e-6
    lr_decay: str = "none"
    optimizer: str = "adamw"
    grad_clip: float = 1.0
    epochs: int = 1
    seed: int = 42


SFT_CONFIG = SFTConfig()
DPO_CONFIG = DPOConfig()
