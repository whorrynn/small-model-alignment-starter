"""Task 4 -- two from-scratch DPO runs (HH-RLHF track, UltraFeedback track).

Must include (manual Section 10.1):
  1. Loading the SFT model as the trainable policy.
  2. Loading a frozen copy of the SFT model as the reference model.
  3. Computing response-only log-probs for chosen and rejected responses
     (src.logprobs.compute_response_logprobs_impl).
  4. Computing the DPO loss (src.losses.dpo_loss_impl).
  5. Tracking validation DPO loss, reward accuracy, and reward margin
     (src.losses.implicit_rewards / reward_accuracy_and_margin).
  6. Saving the best checkpoint by validation reward accuracy, plus final.

Kaggle memory tip from the manual: put the policy on cuda:0 and the frozen
reference on cuda:1; keep the trainable policy in fp32 when using gradient
scaling, the frozen reference may be loaded in fp16.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.checkpoint import BestCheckpointTracker, save_checkpoint
from src.config import DPOConfig, DPO_CONFIG, SFT_CONFIG
from src.data_dpo import load_preference_dataset
from src.losses import dpo_loss_from_logps_impl, dpo_loss_impl, implicit_rewards, reward_accuracy_and_margin
from src.logprobs import compute_response_logprobs_impl
from src.plot_metrics import plot_dpo_curves
from src.prompt_format import format_prompt
from src.utils import append_jsonl, load_tokenizer, pick_device, set_seed


def _evaluate_dpo(policy_model, reference_model, tokenizer, prompts, chosen, rejected, config, batch_size = 4):
    wasTraining = policy_model.training
    policy_model.eval()

    totalLoss, totalBatches = 0.0, 0
    chosenRewardBatches, rejectedRewardBatches = [], []

    # Same fp16 autocast as the training step (log_softmax still runs in fp32), so
    # validation numerics match training and eval isn't slowed down by fp32 matmuls.
    useAutocast = next(policy_model.parameters()).device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type = "cuda", dtype = torch.float16, enabled = useAutocast):
        for start in range(0, len(prompts), batch_size):
            batchPrompts = prompts[start:start + batch_size]
            batchChosen = chosen[start:start + batch_size]
            batchRejected = rejected[start:start + batch_size]
            if not batchPrompts:
                continue

            policyChosenLogps = compute_response_logprobs_impl(
                policy_model, tokenizer, batchPrompts, batchChosen, config.context_length
            )
            policyRejectedLogps = compute_response_logprobs_impl(
                policy_model, tokenizer, batchPrompts, batchRejected, config.context_length
            )
            referenceChosenLogps = compute_response_logprobs_impl(
                reference_model, tokenizer, batchPrompts, batchChosen, config.context_length
            ).to(policyChosenLogps.device)
            referenceRejectedLogps = compute_response_logprobs_impl(
                reference_model, tokenizer, batchPrompts, batchRejected, config.context_length
            ).to(policyRejectedLogps.device)

            loss = dpo_loss_from_logps_impl(
                policyChosenLogps, policyRejectedLogps, referenceChosenLogps, referenceRejectedLogps, config.beta
            )
            totalLoss += loss.item()
            totalBatches += 1

            chosenRewardBatches.append(implicit_rewards(policyChosenLogps, referenceChosenLogps, config.beta))
            rejectedRewardBatches.append(implicit_rewards(policyRejectedLogps, referenceRejectedLogps, config.beta))

    policy_model.train(wasTraining)

    allChosenRewards = torch.cat(chosenRewardBatches)
    allRejectedRewards = torch.cat(rejectedRewardBatches)
    accuracy, margin = reward_accuracy_and_margin(allChosenRewards, allRejectedRewards)

    return totalLoss / max(1, totalBatches), accuracy, margin


def train_dpo(
    track: str,
    sft_checkpoint_path: str | Path,
    config: DPOConfig = DPO_CONFIG,
    run_name: str | None = None,
    output_dir: str | Path = "outputs",
):
    """track is "hh_rlhf" or "ultrafeedback". Orchestrates one from-scratch
    DPO run starting from the SFT checkpoint: policy on cuda:0 (fp32,
    trainable), a frozen copy of the same checkpoint as the reference on
    cuda:1 if a second GPU is available (manual's Kaggle memory tip).

    Returns the path to the best checkpoint.
    """
    from transformers import AutoModelForCausalLM

    if run_name is None:
        run_name = f"{track}_run"

    set_seed(config.seed)

    policyDevice = pick_device(0)
    referenceDevice = pick_device(1) if torch.cuda.device_count() > 1 else policyDevice

    tokenizer = load_tokenizer(sft_checkpoint_path, SFT_CONFIG.model_name)

    policyModel = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, torch_dtype = torch.float32)
    policyModel = policyModel.to(policyDevice)
    policyModel.train()

    # fp32 even though the manual allows fp16: the reference has its own T4, and
    # keeping it identical to the policy makes the step-0 check exact
    # (loss = log 2, accuracy = 0) instead of being blurred by fp16 rounding.
    referenceModel = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, torch_dtype = torch.float32)
    referenceModel = referenceModel.to(referenceDevice)
    referenceModel.eval()
    for parameter in referenceModel.parameters():
        parameter.requires_grad_(False)

    trainPrompts, trainChosen, trainRejected, valPrompts, valChosen, valRejected = load_preference_dataset(
        track, config.train_pairs, config.val_pairs, config.seed
    )
    trainPrompts = [format_prompt(instruction) for instruction in trainPrompts]
    valPrompts = [format_prompt(instruction) for instruction in valPrompts]

    if config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(policyModel.parameters(), lr = config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(policyModel.parameters(), lr = config.learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled = policyDevice.type == "cuda")

    runDir = Path(output_dir) / f"dpo_{track}" / run_name
    metricsPath = runDir / "metrics.jsonl"
    bestCheckpointPath = runDir / "checkpoints" / "best"
    finalCheckpointPath = runDir / "checkpoints" / "final"
    checkpointTracker = BestCheckpointTracker(mode = "max")

    microStep = 0
    optimizerStep = 0
    optimizer.zero_grad()

    # Step-0 sanity point: policy == reference, so loss should be log(2) and accuracy 0 (strict >).
    initialLoss, initialAccuracy, initialMargin = _evaluate_dpo(
        policyModel, referenceModel, tokenizer, valPrompts, valChosen, valRejected, config
    )
    append_jsonl(
        metricsPath,
        {"step": 0, "dpo_loss": initialLoss, "reward_accuracy": initialAccuracy, "reward_margin": initialMargin},
    )

    for _epoch in range(config.epochs):
        for batchStart in range(0, len(trainPrompts), config.per_device_batch_size):
            batchPrompts = trainPrompts[batchStart:batchStart + config.per_device_batch_size]
            batchChosen = trainChosen[batchStart:batchStart + config.per_device_batch_size]
            batchRejected = trainRejected[batchStart:batchStart + config.per_device_batch_size]
            if not batchPrompts:
                continue

            with torch.autocast(
                device_type = policyDevice.type, dtype = torch.float16, enabled = policyDevice.type == "cuda"
            ):
                loss = dpo_loss_impl(
                    policyModel,
                    referenceModel,
                    tokenizer,
                    batchPrompts,
                    batchChosen,
                    batchRejected,
                    config.beta,
                    config.context_length,
                )
                scaledLoss = loss / config.gradient_accumulation

            scaler.scale(scaledLoss).backward()
            microStep += 1

            if microStep % config.gradient_accumulation != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policyModel.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            optimizerStep += 1

            if optimizerStep % 50 == 0:
                dpoLoss, rewardAccuracy, rewardMargin = _evaluate_dpo(
                    policyModel, referenceModel, tokenizer, valPrompts, valChosen, valRejected, config
                )
                append_jsonl(
                    metricsPath,
                    {
                        "step": optimizerStep,
                        "dpo_loss": dpoLoss,
                        "reward_accuracy": rewardAccuracy,
                        "reward_margin": rewardMargin,
                    },
                )

                if checkpointTracker.is_better(rewardAccuracy):
                    save_checkpoint(
                        policyModel,
                        optimizer,
                        optimizerStep,
                        {"dpo_loss": dpoLoss, "reward_accuracy": rewardAccuracy, "reward_margin": rewardMargin},
                        bestCheckpointPath,
                    )

    # Flush a trailing partial accumulation (15000 pairs % 32 = 24) and free the
    # gradients before the final eval, which otherwise runs with ~2 GB of stale
    # grads on the policy GPU (this exact pattern OOM'd the SFT final eval).
    if microStep % config.gradient_accumulation != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policyModel.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizerStep += 1
    optimizer.zero_grad(set_to_none = True)
    torch.cuda.empty_cache()

    finalDpoLoss, finalRewardAccuracy, finalRewardMargin = _evaluate_dpo(
        policyModel, referenceModel, tokenizer, valPrompts, valChosen, valRejected, config
    )
    finalMetrics = {
        "dpo_loss": finalDpoLoss,
        "reward_accuracy": finalRewardAccuracy,
        "reward_margin": finalRewardMargin,
    }
    save_checkpoint(policyModel, optimizer, optimizerStep, finalMetrics, finalCheckpointPath)
    append_jsonl(
        metricsPath,
        {
            "step": optimizerStep,
            "dpo_loss": finalDpoLoss,
            "reward_accuracy": finalRewardAccuracy,
            "reward_margin": finalRewardMargin,
            "final": True,
        },
    )

    for checkpointPath in (bestCheckpointPath, finalCheckpointPath):
        if checkpointPath.exists():
            tokenizer.save_pretrained(checkpointPath)

    plot_dpo_curves(metricsPath, Path("report_assets") / f"{run_name}_training_curves.png")

    return bestCheckpointPath
