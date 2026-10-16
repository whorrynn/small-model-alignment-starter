"""Task 3 -- from-scratch SFT training loop.

Must include (manual Section 9.1):
  1. Prompt formatting for instruction-response examples (src.prompt_format).
  2. Response-only loss masking (src.batching / src.losses.sft_loss_impl).
  3. Padding and attention masks (src.batching).
  4. Gradient accumulation.
  5. Mixed precision using fp16 autocast on T4 GPUs.
  6. FP32 trainable weights when using gradient scaling.
  7. Gradient clipping.
  8. Validation loss.
  9. Saving best and final checkpoints (src.checkpoint).

Suggested hyperparameters live in src/config.py (SFT_CONFIG) -- not sacred,
see the manual's "Guide Configurations, Not Sacred Hyperparameters" box.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from src.checkpoint import BestCheckpointTracker, save_checkpoint
from src.config import SFTConfig, SFT_CONFIG
from src.data_sft import load_sft_dataset
from src.losses import sft_loss_impl
from src.plot_metrics import plot_sft_curves
from src.prompt_format import format_prompt
from src.utils import append_jsonl, pick_device, set_seed


def _cosine_warmup_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _evaluate_sft_loss(model, tokenizer, prompts, responses, config, batch_size = 4):
    # Batch 4 (not 8): two full-vocab (151,936) fp32 tensors per batch sit next to
    # ~6 GB of weights + AdamW state on one T4; batch 8 OOM'd mid-run. 1000 % 4 == 0,
    # so the per-batch-mean average equals the batch-8 value.
    wasTraining = model.training
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    totalLoss, totalBatches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batchPrompts = prompts[start:start + batch_size]
            batchResponses = responses[start:start + batch_size]
            if not batchPrompts:
                continue
            loss = sft_loss_impl(model, tokenizer, batchPrompts, batchResponses, config.context_length)
            totalLoss += loss.item()
            totalBatches += 1

    model.train(wasTraining)
    return totalLoss / max(1, totalBatches)


def train_sft(config: SFTConfig = SFT_CONFIG, run_name: str = "sft_run", output_dir: str | Path = "outputs"):
    """Orchestrate the full SFT run: load data + model, train with gradient
    accumulation and fp16 autocast, validate periodically, checkpoint the
    best and final models, and plot the training curve.

    Returns the path to the best checkpoint.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(config.seed)
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype = torch.float32).to(device)
    model.train()

    trainPrompts, trainResponses, valPrompts, valResponses = load_sft_dataset(
        config.train_examples, config.val_examples, config.seed
    )
    trainPrompts = [format_prompt(instruction) for instruction in trainPrompts]
    valPrompts = [format_prompt(instruction) for instruction in valPrompts]

    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

    microBatchesPerEpoch = math.ceil(len(trainPrompts) / config.per_device_batch_size)
    optimizerStepsPerEpoch = math.ceil(microBatchesPerEpoch / config.gradient_accumulation)
    totalOptimizerSteps = optimizerStepsPerEpoch * config.epochs
    warmupSteps = max(1, int(config.warmup_fraction * totalOptimizerSteps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda = lambda step: _cosine_warmup_lambda(step, totalOptimizerSteps, warmupSteps)
    )
    scaler = torch.cuda.amp.GradScaler(enabled = device.type == "cuda")

    runDir = Path(output_dir) / "sft" / run_name
    metricsPath = runDir / "metrics.jsonl"
    bestCheckpointPath = runDir / "checkpoints" / "best"
    finalCheckpointPath = runDir / "checkpoints" / "final"
    checkpointTracker = BestCheckpointTracker(mode = "min")

    microStep = 0
    optimizerStep = 0
    optimizer.zero_grad()

    # Step-0 point: validation loss of the untrained base model, for the curve.
    initialValLoss = _evaluate_sft_loss(model, tokenizer, valPrompts, valResponses, config)
    append_jsonl(metricsPath, {"step": 0, "train_loss": None, "val_loss": initialValLoss, "lr": 0.0})

    for _epoch in range(config.epochs):
        for batchStart in range(0, len(trainPrompts), config.per_device_batch_size):
            batchPrompts = trainPrompts[batchStart:batchStart + config.per_device_batch_size]
            batchResponses = trainResponses[batchStart:batchStart + config.per_device_batch_size]
            if not batchPrompts:
                continue

            with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = device.type == "cuda"):
                loss = sft_loss_impl(model, tokenizer, batchPrompts, batchResponses, config.context_length)
                scaledLoss = loss / config.gradient_accumulation

            scaler.scale(scaledLoss).backward()
            microStep += 1

            if microStep % config.gradient_accumulation != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            optimizerStep += 1

            if optimizerStep % 50 == 0 or optimizerStep == totalOptimizerSteps:
                valLoss = _evaluate_sft_loss(model, tokenizer, valPrompts, valResponses, config)
                append_jsonl(
                    metricsPath,
                    {
                        "step": optimizerStep,
                        "train_loss": loss.item(),
                        "val_loss": valLoss,
                        "lr": scheduler.get_last_lr()[0],
                    },
                )

                if checkpointTracker.is_better(valLoss):
                    save_checkpoint(model, optimizer, optimizerStep, {"val_loss": valLoss}, bestCheckpointPath)

    # Flush a trailing partial accumulation (e.g. 15000 micro-batches % 16 = 8):
    # step on it, then free the gradients -- leaving ~2 GB of stale grads on the
    # GPU made the final full-vocab eval OOM on a T4.
    if microStep % config.gradient_accumulation != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizerStep += 1
    optimizer.zero_grad(set_to_none = True)
    torch.cuda.empty_cache()

    finalValLoss = _evaluate_sft_loss(model, tokenizer, valPrompts, valResponses, config)
    save_checkpoint(model, optimizer, optimizerStep, {"val_loss": finalValLoss}, finalCheckpointPath)
    for checkpointPath in (bestCheckpointPath, finalCheckpointPath):
        if checkpointPath.exists():
            tokenizer.save_pretrained(checkpointPath)

    plot_sft_curves(metricsPath, Path("report_assets") / f"{run_name}_training_curves.png")

    return bestCheckpointPath
