"""Task 6 -- build anonymized pairwise comparison files for judging.

Manual Task 6: "Build pairwise comparison files for DPO versus SFT.
Randomize response order in each pairwise file and preserve the mapping
from response position to model identity as hidden metadata. The judge
prompt must label the outputs only as Response A and Response B; it must
not reveal model names, checkpoint names, or whether an output came from
SFT or DPO."
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def build_pairwise_file(
    generations_a: list[dict],
    generations_b: list[dict],
    model_a_name: str,
    model_b_name: str,
    output_path: str | Path,
    seed: int = 42,
) -> None:
    """Writes one JSON object per line (JSONL) to `output_path`.

    `generations_a`/`generations_b` are lists of {"id", "instruction",
    "output", ...} dicts (src.generation's schema) for the SAME prompts, in
    the SAME order, from two different models.

    For each prompt, a seeded coin flip decides whether model_a's output
    becomes "response_a" or "response_b" -- so the judge never sees the same
    model in the same slot every time (position bias). Each record also
    carries a "position_to_model" field for src/judging.py's own bookkeeping
    only; that field is never sent to the judge model itself.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    coinFlips = random.Random(seed)

    with open(output_path, "w", encoding = "utf-8") as handle:
        for rowA, rowB in zip(generations_a, generations_b):
            assert rowA["id"] == rowB["id"], "generations_a/generations_b must be aligned by id"

            if coinFlips.random() < 0.5:
                responseA, responseB = rowA["output"], rowB["output"]
                positionToModel = {"A": model_a_name, "B": model_b_name}
            else:
                responseA, responseB = rowB["output"], rowA["output"]
                positionToModel = {"A": model_b_name, "B": model_a_name}

            record = {
                "id": rowA["id"],
                "instruction": rowA["instruction"],
                "response_a": responseA,
                "response_b": responseB,
                "position_to_model": positionToModel,
            }
            handle.write(json.dumps(record, ensure_ascii = False) + "\n")
