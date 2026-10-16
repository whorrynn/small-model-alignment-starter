"""Core batch-building logic behind submission_adapter.build_sft_batch and
build_dpo_batch. Kept out of submission_adapter.py on purpose -- the manual
says the adapter should be "a thin wrapper around your own implementation";
the actual tensor-building logic lives here.

Contract this must satisfy (manual Section 4.1 "Expected behavior", and
tests/test_public_adapter_smoke.py):
  - Pad shorter examples and truncate longer examples to max_length.
  - Mask prompt tokens AND padding tokens with -100 in `labels`.
  - Append an EOS token to each response before truncating to max_length
    (so the EOS token may be removed if the response gets truncated).
  - Compute log probabilities over response tokens only -- this module only
    builds the masked batch; src/logprobs.py does the scoring.
  - Preserve input row order in every returned tensor.
"""

from __future__ import annotations

import torch


def build_sft_batch_impl(tokenizer, prompts, responses, max_length):
    """Build a single (prompt + response) batch with response-only labels.

    Returns {"input_ids", "attention_mask", "labels"}, each a LongTensor of
    shape (len(prompts), L) where L is the longest truncated row (<= max_length).
    """
    eosId = tokenizer.eos_token_id
    padId = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eosId

    rowIds, rowLabels = [], []
    for prompt, response in zip(prompts, responses):
        promptIds = tokenizer.encode(prompt, add_special_tokens = False)
        responseIds = tokenizer.encode(response, add_special_tokens = False)
        if eosId is not None:
            responseIds = responseIds + [eosId]

        fullIds = (promptIds + responseIds)[:max_length]
        promptLen = min(len(promptIds), len(fullIds))  # clip in case the prompt alone overflows

        labels = list(fullIds)
        labels[:promptLen] = [-100] * promptLen

        rowIds.append(fullIds)
        rowLabels.append(labels)

    seqLen = max(len(ids) for ids in rowIds)
    batchSize = len(rowIds)

    inputIds = torch.full((batchSize, seqLen), padId, dtype = torch.long)
    attentionMask = torch.zeros((batchSize, seqLen), dtype = torch.long)
    labelsTensor = torch.full((batchSize, seqLen), -100, dtype = torch.long)

    for row, (ids, labels) in enumerate(zip(rowIds, rowLabels)):
        length = len(ids)
        inputIds[row, :length] = torch.tensor(ids, dtype = torch.long)
        attentionMask[row, :length] = 1
        labelsTensor[row, :length] = torch.tensor(labels, dtype = torch.long)

    return {"input_ids": inputIds, "attention_mask": attentionMask, "labels": labelsTensor}


def build_dpo_batch_impl(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Build chosen/rejected LM batches sharing the same prompts (manual Section 2.4.3 step 1)."""
    return {
        "chosen": build_sft_batch_impl(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch_impl(tokenizer, prompts, rejected_responses, max_length),
    }
