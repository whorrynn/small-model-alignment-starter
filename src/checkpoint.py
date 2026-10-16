"""Checkpoint saving/loading shared by the SFT and DPO training loops.

Manual: save only `best` and `final` (Kaggle disk is limited -- Sections
9.1 item 9, 10.1 item 6). Checkpoint selection metric differs by stage:
  - SFT: LOWER validation loss is better.
  - DPO: HIGHER validation reward accuracy is better (Section 10.5).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(model, optimizer, step: int, metrics: dict, path: str | Path) -> None:
    """Serialize model + optimizer state + step + metrics to `path`.

    Uses model.save_pretrained() when available (HF transformers models) --
    cheaper on Kaggle disk than a raw state-dict blob, per the manual's
    "Saving Work While Kaggle Jobs Run" guidance. Tokenizer is saved once by
    the caller (it never changes across checkpoints), not here.
    """
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path / "model_state_dict.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), path / "optimizer_state_dict.pt")

    metadata = {"step": step, "metrics": metrics}
    with open(path / "metadata.json", "w", encoding = "utf-8") as handle:
        json.dump(metadata, handle, indent = 2)


def load_checkpoint(path: str | Path, model, optimizer=None):
    """Inverse of save_checkpoint. Loads weights into `model` in place (and
    optimizer state into `optimizer` if given). Returns (step, metrics).
    """
    path = Path(path)

    safetensorsPath = path / "model.safetensors"
    if safetensorsPath.exists():
        from safetensors.torch import load_file

        stateDict = load_file(safetensorsPath)
    elif (path / "pytorch_model.bin").exists():
        stateDict = torch.load(path / "pytorch_model.bin", map_location = "cpu")
    else:
        stateDict = torch.load(path / "model_state_dict.pt", map_location = "cpu")
    model.load_state_dict(stateDict)

    optimizerPath = path / "optimizer_state_dict.pt"
    if optimizer is not None and optimizerPath.exists():
        optimizer.load_state_dict(torch.load(optimizerPath, map_location = "cpu"))

    with open(path / "metadata.json", encoding = "utf-8") as handle:
        metadata = json.load(handle)

    return metadata["step"], metadata["metrics"]


class BestCheckpointTracker:
    """Tracks the best metric seen so far and tells the training loop
    whether the current step should overwrite the "best" checkpoint.
    """

    def __init__(self, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best_value: float | None = None

    def is_better(self, metric_value: float) -> bool:
        isImprovement = self.best_value is None
        if not isImprovement:
            isImprovement = metric_value < self.best_value if self.mode == "min" else metric_value > self.best_value

        if isImprovement:
            self.best_value = metric_value

        return isImprovement
