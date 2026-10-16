"""Preference-dataset loading for both from-scratch DPO tracks (Task 4) and
the provided LoRA persona data (Task 5) -- manual Section 6.3.
"""

from __future__ import annotations

import json
from pathlib import Path


def _parse_hh_pair(chosen_text: str, rejected_text: str):
    """HH-RLHF rows store the FULL conversation twice, once ending in the
    chosen reply and once ending in the rejected reply, e.g.
    "\\n\\nHuman: ...\\n\\nAssistant: ...". Keep only single-turn rows (exactly
    one "Human:" turn) per the manual's Section 6.3 single-turn requirement.
    """
    humanMarker = "\n\nHuman: "
    assistantMarker = "\n\nAssistant: "

    if chosen_text.count(humanMarker) != 1 or rejected_text.count(humanMarker) != 1:
        return None

    humanStart = chosen_text.index(humanMarker) + len(humanMarker)
    chosenAssistantStart = chosen_text.index(assistantMarker)
    rejectedAssistantStart = rejected_text.index(assistantMarker)

    prompt = chosen_text[humanStart:chosenAssistantStart].strip()
    chosenResponse = chosen_text[chosenAssistantStart + len(assistantMarker):].strip()
    rejectedResponse = rejected_text[rejectedAssistantStart + len(assistantMarker):].strip()

    if not prompt or not chosenResponse or not rejectedResponse:
        return None

    return prompt, chosenResponse, rejectedResponse


def _load_hh_rlhf_triples(limit: int, seed: int):
    from datasets import load_dataset

    rawDataset = load_dataset("Anthropic/hh-rlhf", split = "train").shuffle(seed = seed)

    prompts, chosenList, rejectedList = [], [], []
    for row in rawDataset:
        parsed = _parse_hh_pair(row["chosen"], row["rejected"])
        if parsed is None:
            continue
        prompt, chosenResponse, rejectedResponse = parsed
        prompts.append(prompt)
        chosenList.append(chosenResponse)
        rejectedList.append(rejectedResponse)
        if len(prompts) >= limit:
            break

    return prompts, chosenList, rejectedList


def _load_ultrafeedback_triples(limit: int, seed: int):
    from datasets import load_dataset

    rawDataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split = "train_prefs").shuffle(seed = seed)

    prompts, chosenList, rejectedList = [], [], []
    for row in rawDataset:
        chosenMessages = row["chosen"]
        rejectedMessages = row["rejected"]
        if len(chosenMessages) != 2 or len(rejectedMessages) != 2:
            continue  # keep single-turn rows only (manual Section 6.3)

        prompt = row["prompt"]
        chosenResponse = chosenMessages[-1]["content"]
        rejectedResponse = rejectedMessages[-1]["content"]
        if not prompt or not chosenResponse or not rejectedResponse:
            continue

        prompts.append(prompt)
        chosenList.append(chosenResponse)
        rejectedList.append(rejectedResponse)
        if len(prompts) >= limit:
            break

    return prompts, chosenList, rejectedList


def load_preference_dataset(track: str, train_size: int = 15_000, val_size: int = 1_000, seed: int = 42):
    """track is "hh_rlhf" (safety-oriented, manual Section 10.2) or
    "ultrafeedback" (general-helpfulness). Returns six parallel lists:
        (train_prompts, train_chosen, train_rejected,
         val_prompts, val_chosen, val_rejected)
    """
    totalNeeded = train_size + val_size

    if track == "hh_rlhf":
        prompts, chosenList, rejectedList = _load_hh_rlhf_triples(totalNeeded, seed)
    elif track == "ultrafeedback":
        prompts, chosenList, rejectedList = _load_ultrafeedback_triples(totalNeeded, seed)
    else:
        raise ValueError(f"Unknown track: {track!r} (expected 'hh_rlhf' or 'ultrafeedback')")

    trainPrompts = prompts[:train_size]
    trainChosen = chosenList[:train_size]
    trainRejected = rejectedList[:train_size]
    valPrompts = prompts[train_size:train_size + val_size]
    valChosen = chosenList[train_size:train_size + val_size]
    valRejected = rejectedList[train_size:train_size + val_size]

    return trainPrompts, trainChosen, trainRejected, valPrompts, valChosen, valRejected


def load_persona_dataset(path: str | Path = "data/persona_lora/ramsay_style_preferences.json"):
    """Provided, fixed data -- just load and return it; there's nothing to
    design here. The manual's formatting work (converting rows into TRL's
    conversational DPO record shape) happens in src/lora_persona_train.py,
    not here.

    Returns a list of {"prompt": ..., "chosen": ..., "rejected": ...} dicts,
    exactly as stored in the JSON file.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
