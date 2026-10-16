"""Shared generation utility for baseline / SFT / DPO outputs (Tasks 2 and 6).

Manual Task 2: "Use greedy decoding for the main comparison unless you
justify another choice... Document any decoding changes so SFT and DPO are
compared fairly." Always use IDENTICAL decoding settings across the models
being compared in one evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.prompt_format import format_prompt


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    ids: list[str],
    generator_name: str,
    dataset_name: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    batch_size: int = 8,
):
    """Generate one response per prompt. Default decoding is greedy
    (do_sample=False) per the manual's Task 2.
    """
    device = next(model.parameters()).device
    wasTraining = model.training
    model.eval()

    originalPaddingSide = tokenizer.padding_side
    tokenizer.padding_side = "left"  # new tokens must follow the prompt, not right-padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    formattedPrompts = [format_prompt(instruction, tokenizer) for instruction in prompts]

    generations = []
    with torch.no_grad():
        for start in range(0, len(formattedPrompts), batch_size):
            batchFormatted = formattedPrompts[start:start + batch_size]
            batchIds = ids[start:start + batch_size]
            batchInstructions = prompts[start:start + batch_size]

            encoded = tokenizer(
                batchFormatted, return_tensors = "pt", padding = True, truncation = True, max_length = 512
            ).to(device)
            promptLength = encoded["input_ids"].shape[1]

            generatedIds = model.generate(
                input_ids = encoded["input_ids"],
                attention_mask = encoded["attention_mask"],
                max_new_tokens = max_new_tokens,
                do_sample = do_sample,
                temperature = temperature,
                top_p = top_p,
                repetition_penalty = repetition_penalty,
                no_repeat_ngram_size = no_repeat_ngram_size,
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            newTokens = generatedIds[:, promptLength:]
            decodedOutputs = tokenizer.batch_decode(newTokens, skip_special_tokens = True)

            for exampleId, instruction, output in zip(batchIds, batchInstructions, decodedOutputs):
                generations.append(
                    {
                        "id": exampleId,
                        "instruction": instruction,
                        "output": output.strip(),
                        "generator": generator_name,
                        "dataset": dataset_name,
                    }
                )

    tokenizer.padding_side = originalPaddingSide
    model.train(wasTraining)

    return generations


def save_generations(generations: list[dict], path: str | Path) -> None:
    """Trivial I/O, not assessed -- write `generations` as a JSON list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(generations, handle, indent=2, ensure_ascii=False)


def load_generations(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
