"""Task 2 -- build the fixed 300-prompt AlpacaEval subset (seed 42).

This is NOT provided in the starter repo (only the HH-RLHF-style safety
eval set is fixed for you, at eval_sets/hh_rlhf_eval_300_seed42.json). The
manual asks YOU to create this subset so every later comparison (SFT vs
base, DPO vs SFT, both UltraFeedback-track evaluations) reads from the same
reproducible 300 prompts: "To reproduce the default AlpacaEval evaluation
subset, use 300 prompts with seed 42."

Build it ONCE, save it, and have every later script (baseline generation,
SFT generation, DPO generation) load the SAVED file rather than re-sampling.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_fixed_alpacaeval_subset(
    n: int = 300,
    seed: int = 42,
    save_path: str | Path = "eval_sets/alpaca_eval_300_seed42.json",
):
    """
      - Loads the AlpacaEval prompt set (`datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]`).
      - Shuffles deterministically with `seed`, takes the first `n` prompts.
      - Saves as a JSON list of dicts shaped like eval_sets/hh_rlhf_eval_300_seed42.json's
        entries, e.g. {"id": "alpaca-eval-42-0000", "instruction": ...,
        "dataset": "alpaca_eval"}, to `save_path`.

    Return the list of prompt dicts. If `save_path` already exists, load and
    return it instead of re-sampling (keeps the subset stable across runs/
    Kaggle sessions without needing to re-set the seed carefully every time).
    """
    save_path = Path(save_path)
    if save_path.exists():
        with open(save_path, encoding="utf-8") as handle:
            return json.load(handle)

    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    # tatsu-lab/alpaca_eval is a script-based dataset, which datasets>=4 refuses
    # to load. Its "alpaca_eval" config reads exactly this JSON file (805 rows), so
    # loading the file and calling Dataset.shuffle(seed) should give the same
    # order as load_dataset(...)["eval"].shuffle(seed) on an older datasets version.
    jsonPath = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type = "dataset")
    with open(jsonPath, encoding = "utf-8") as handle:
        rawRows = json.load(handle)
    rawDataset = Dataset.from_list(rawRows)
    shuffledDataset = rawDataset.shuffle(seed = seed)
    selectedRows = shuffledDataset.select(range(min(n, len(shuffledDataset))))

    subset = [
        {
            "id": f"alpaca-eval-{seed}-{index:04d}",
            "instruction": row["instruction"],
            "dataset": "alpaca_eval",
        }
        for index, row in enumerate(selectedRows)
    ]

    save_path.parent.mkdir(parents = True, exist_ok = True)
    with open(save_path, "w", encoding = "utf-8") as handle:
        json.dump(subset, handle, indent = 2, ensure_ascii = False)

    return subset
