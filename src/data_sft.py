"""SFT dataset loading (manual Section 6.2 / Task 3).

This only needs to run on Kaggle (needs network + disk for the HF dataset
download); kept isolated here so it's easy to debug without touching the
training loop in src/sft_train.py.
"""

from __future__ import annotations


def load_sft_dataset(train_size: int = 30_000, val_size: int = 1_000, seed: int = 42):
    """
      - Loads an UltraChat-style instruction dataset via
        `datasets.load_dataset("HuggingFaceH4/ultrachat_200k", ...)`.
      - Converts each example into a single-turn (instruction, response)
        pair by keeping only the first user/assistant exchange.
      - Shuffles with the given seed and takes `train_size + val_size` examples
        total, split train/val.

    Return four parallel lists of strings:
        (train_prompts, train_responses, val_prompts, val_responses)

    `train_prompts`/`val_prompts` should be the RAW instruction text (apply
    src.prompt_format.format_prompt later, at batch-build time, not here --
    keeps this function reusable if the template changes).
    """
    from datasets import load_dataset

    rawDataset = load_dataset("HuggingFaceH4/ultrachat_200k", split = "train_sft")
    shuffledDataset = rawDataset.shuffle(seed = seed)
    totalNeeded = train_size + val_size
    selectedRows = shuffledDataset.select(range(min(totalNeeded, len(shuffledDataset))))

    prompts, responses = [], []
    for row in selectedRows:
        messages = row["messages"]
        firstUserMessage = next((m["content"] for m in messages if m["role"] == "user"), None)
        firstAssistantMessage = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if not firstUserMessage or not firstAssistantMessage:
            continue
        prompts.append(firstUserMessage)
        responses.append(firstAssistantMessage)

    trainPrompts = prompts[:train_size]
    trainResponses = responses[:train_size]
    valPrompts = prompts[train_size:train_size + val_size]
    valResponses = responses[train_size:train_size + val_size]

    return trainPrompts, trainResponses, valPrompts, valResponses
