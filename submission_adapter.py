"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function is a thin
wrapper around the real implementation in src/ (batching.py, logprobs.py,
losses.py) -- the training mathematics itself lives there, not here, same
boundary pattern as PA1's tests/adapters.py -> src/*.py.

Fill in src/batching.py, src/logprobs.py, and src/losses.py first; once
those pass tests/test_public_adapter_smoke.py, this file needs no further
edits.
"""

from __future__ import annotations

from src.batching import build_dpo_batch_impl, build_sft_batch_impl
from src.logprobs import compute_response_logprobs_impl
from src.losses import dpo_loss_from_logps_impl, dpo_loss_impl, sft_loss_impl


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return build_sft_batch_impl(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return compute_response_logprobs_impl(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return sft_loss_impl(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return build_dpo_batch_impl(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return dpo_loss_from_logps_impl(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return dpo_loss_impl(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
