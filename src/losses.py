"""SFT and DPO loss implementations.

Formulas: manual Eq. (2) for SFT, Eq. (5)/(6) for DPO, Eq. (7)-(10) for the
implicit-reward validation diagnostics used by the DPO training loop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.logprobs import compute_response_logprobs_impl


def sft_loss_impl(model, tokenizer, prompts, responses, max_length):
    """Scalar response-only SFT loss.

    Reduction choice (a): mean of per-example summed response negative
    log-probs. Documented in REPORT.md Section 14.2 -- used consistently
    everywhere this loss is computed (train step, validation step).
    """
    responseLogps = compute_response_logprobs_impl(model, tokenizer, prompts, responses, max_length)
    return -responseLogps.mean()


def dpo_loss_from_logps_impl(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Manual Eq. (5)/(6), the pure-tensor DPO objective. Returns the scalar
    batch-mean of the per-example logistic loss.
    """
    deltaTheta = policy_chosen_logps - policy_rejected_logps
    deltaRef = reference_chosen_logps - reference_rejected_logps
    perExampleLoss = -F.logsigmoid(beta * (deltaTheta - deltaRef))
    return perExampleLoss.mean()


def dpo_loss_impl(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Full from-scratch DPO loss, manual Section 2.4.3 steps 1-8.

    The reference-model calls live inside torch.no_grad(), so no graph edge
    ever reaches reference_model's parameters -- loss.backward() cannot put
    a gradient there (verified by
    test_dpo_loss_is_scalar_and_does_not_update_reference_model).
    """
    policyChosenLogps = compute_response_logprobs_impl(policy_model, tokenizer, prompts, chosen_responses, max_length)
    policyRejectedLogps = compute_response_logprobs_impl(
        policy_model, tokenizer, prompts, rejected_responses, max_length
    )

    with torch.no_grad():
        referenceChosenLogps = compute_response_logprobs_impl(
            reference_model, tokenizer, prompts, chosen_responses, max_length
        )
        referenceRejectedLogps = compute_response_logprobs_impl(
            reference_model, tokenizer, prompts, rejected_responses, max_length
        )

    # Policy and reference may live on different devices (manual's Kaggle
    # memory tip: policy on cuda:0, frozen reference on cuda:1).
    referenceChosenLogps = referenceChosenLogps.to(policyChosenLogps.device)
    referenceRejectedLogps = referenceRejectedLogps.to(policyRejectedLogps.device)

    return dpo_loss_from_logps_impl(
        policyChosenLogps, policyRejectedLogps, referenceChosenLogps, referenceRejectedLogps, beta
    )


def implicit_rewards(policy_logps, reference_logps, beta):
    """Manual Eq. (7)-(8): r = beta * (policy_logps - reference_logps), elementwise."""
    return beta * (policy_logps - reference_logps)


def reward_accuracy_and_margin(chosen_rewards, rejected_rewards):
    """Manual Eq. (9)-(10). accuracy uses STRICT >, so an untrained policy
    (policy == reference, every reward pair tied) scores accuracy 0, not 0.5
    (manual Section 10.5 calls this out explicitly).
    """
    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
    margin = (chosen_rewards - rejected_rewards).mean().item()
    return accuracy, margin
