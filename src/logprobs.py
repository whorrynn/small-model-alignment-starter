"""Response-only log-probability computation, shared by SFT loss scoring
and DPO (both the from-scratch loss and the implicit-reward diagnostics).

manual Section 4.1: "Compute log probabilities over response tokens only,
not prompt or padding tokens." and "Preserve the input row order in all
returned tensors."
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.batching import build_sft_batch_impl


def compute_response_logprobs_impl(model, tokenizer, prompts, responses, max_length):
    """Return one SUMMED response-token log-probability per row.

    Note: this function never calls model.train()/model.eval() itself, so it
    never needs to restore the mode either -- it inherits whatever mode the
    caller left the model in. sft_loss_impl calls it with grad enabled during
    training; dpo_loss_impl wraps the reference-model calls in
    torch.no_grad() itself (Section 2.4.3 step 5).
    """
    batch = build_sft_batch_impl(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    inputIds = batch["input_ids"].to(device)
    attentionMask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids = inputIds, attention_mask = attentionMask).logits

    # Causal-LM shift: logits[:, t] predicts the token at position t + 1.
    shiftLogits = logits[:, :-1, :]
    shiftLabels = labels[:, 1:]

    logProbs = F.log_softmax(shiftLogits, dim = -1)
    responseMask = shiftLabels.ne(-100)
    safeLabels = shiftLabels.clamp_min(0)  # -100 is not a valid gather index
    tokenLogProbs = logProbs.gather(dim = -1, index = safeLabels.unsqueeze(-1)).squeeze(-1)
    tokenLogProbs = tokenLogProbs * responseMask

    return tokenLogProbs.sum(dim = -1)
