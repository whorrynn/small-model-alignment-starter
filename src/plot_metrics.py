"""Training-curve plotting for REPORT.md Section 14.4 ("Training Behavior").

Manual: "Show readable plots for the training runs. Include training and
validation loss for SFT where available, and DPO loss plus at least one
preference diagnostic such as reward accuracy or reward margin for DPO.
Label axes, runs, checkpoints, and smoothing."
"""

from __future__ import annotations

from pathlib import Path

from src.utils import read_jsonl


def plot_sft_curves(metrics_path: str | Path, save_path: str | Path):
    """Read metrics.jsonl (written by src.sft_train.train_sft), plot training
    and validation loss against optimizer step on one figure.
    """
    records = read_jsonl(metrics_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents = True, exist_ok = True)

    import matplotlib.pyplot as plt

    steps = [record["step"] for record in records]
    valLoss = [record["val_loss"] for record in records]

    # The step-0 record has no train loss (nothing has been trained yet), so skip it here.
    trainSteps = []
    trainLoss = []
    for record in records:
        if record.get("train_loss") is not None:
            trainSteps.append(record["step"])
            trainLoss.append(record["train_loss"])

    figure, axis = plt.subplots(figsize = (7, 4.5))
    axis.plot(trainSteps, trainLoss, label = "train loss (last micro-batch)", color = "tab:blue", alpha = 0.5)
    axis.plot(steps, valLoss, label = "val loss (1k held-out)", color = "tab:orange", marker = "o", markersize = 3)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("SFT loss")
    axis.set_title("SFT training and validation loss")
    axis.legend()
    axis.grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi = 150)
    plt.close(figure)


def plot_dpo_curves(metrics_path: str | Path, save_path: str | Path):
    """Read metrics.jsonl (written by src.dpo_train.train_dpo), plot DPO loss
    and the reward-accuracy / reward-margin preference diagnostics against
    optimizer step as two side-by-side subplots.
    """
    records = read_jsonl(metrics_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents = True, exist_ok = True)

    import matplotlib.pyplot as plt

    steps = [record["step"] for record in records]
    dpoLoss = [record["dpo_loss"] for record in records]
    rewardAccuracy = [record["reward_accuracy"] for record in records]
    rewardMargin = [record["reward_margin"] for record in records]

    figure, (lossAxis, rewardAxis) = plt.subplots(1, 2, figsize = (12, 4.5))

    lossAxis.plot(steps, dpoLoss, color = "tab:blue")
    lossAxis.set_xlabel("optimizer step")
    lossAxis.set_ylabel("DPO loss")
    lossAxis.set_title("DPO validation loss")
    lossAxis.grid(alpha = 0.3)

    rewardAxis.plot(steps, rewardAccuracy, label = "reward accuracy", color = "tab:green")
    rewardAxis.plot(steps, rewardMargin, label = "reward margin", color = "tab:red")
    rewardAxis.set_xlabel("optimizer step")
    rewardAxis.set_ylabel("value")
    rewardAxis.set_title("DPO preference diagnostics")
    rewardAxis.legend()
    rewardAxis.grid(alpha = 0.3)

    figure.tight_layout()
    figure.savefig(save_path, dpi = 150)
    plt.close(figure)
