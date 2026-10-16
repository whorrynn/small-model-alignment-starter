"""Small, non-assessed helpers shared across scripts (seeding, device
placement, directory bookkeeping). Nothing here is part of the graded
assignment math -- it's just infrastructure.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(index: int = 0):
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > index:
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record to a .jsonl metrics/log file, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_tokenizer(checkpoint_path: str | Path, fallback_model_name: str):
    """Tokenizer saved next to a checkpoint if present, else the base model's.
    SFT/DPO never change the vocabulary, so the two are interchangeable."""
    from transformers import AutoTokenizer

    if (Path(checkpoint_path) / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
