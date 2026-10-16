"""Task 6 -- generate SFT & DPO outputs on a fixed eval set and build the
anonymized pairwise comparison file for Task 7's judging.

    python scripts/generate_eval_outputs.py \
        --eval_set eval_sets/hh_rlhf_eval_300_seed42.json \
        --sft_checkpoint outputs/sft/sft_run/checkpoints/best \
        --dpo_checkpoint outputs/dpo_hh_rlhf/hh_rlhf_run/checkpoints/best \
        --dpo_track hh_rlhf \
        --output_dir outputs/eval

Run this once per DPO track (hh_rlhf against eval_sets/hh_rlhf_eval_300_seed42.json,
ultrafeedback against eval_sets/alpaca_eval_300_seed42.json). The manual's
"Generation Coherence Tip": use the SAME decoding settings for SFT and DPO
in a given comparison, so the judge is evaluating the models, not different
decoding policies -- the default args below keep both calls identical.

CLI wiring plus load_model_and_tokenizer (which loads whatever
src.checkpoint.save_checkpoint wrote via model.save_pretrained) are both
implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import SFT_CONFIG
from src.generation import generate_responses, save_generations
from src.pairwise import build_pairwise_file
from src.utils import load_tokenizer


def load_eval_set(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_model_and_tokenizer(checkpoint_path: str | Path):
    """Load a model + tokenizer from a checkpoint saved by
    src.checkpoint.save_checkpoint (model.save_pretrained / tokenizer saved
    once at train start).
    """
    from transformers import AutoModelForCausalLM

    # Checkpoints from before the tokenizer was saved alongside them fall back to
    # the base model's tokenizer (SFT/DPO never change the vocabulary).
    tokenizer = load_tokenizer(checkpoint_path, SFT_CONFIG.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    model.eval()

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--dpo_checkpoint", required=True)
    parser.add_argument("--dpo_track", required=True, help="hh_rlhf or ultrafeedback, used for output naming only")
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    args = parser.parse_args()

    eval_items = load_eval_set(args.eval_set)
    prompts = [item["instruction"] for item in eval_items]
    ids = [item["id"] for item in eval_items]

    decoding_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    sft_model, sft_tokenizer = load_model_and_tokenizer(args.sft_checkpoint)
    sft_generations = generate_responses(
        sft_model, sft_tokenizer, prompts, ids, generator_name="sft", dataset_name=args.dpo_track, **decoding_kwargs
    )
    save_generations(sft_generations, Path(args.output_dir) / f"sft_{args.dpo_track}.json")

    dpo_model, dpo_tokenizer = load_model_and_tokenizer(args.dpo_checkpoint)
    dpo_generations = generate_responses(
        dpo_model,
        dpo_tokenizer,
        prompts,
        ids,
        generator_name=f"dpo_{args.dpo_track}",
        dataset_name=args.dpo_track,
        **decoding_kwargs,
    )
    save_generations(dpo_generations, Path(args.output_dir) / f"dpo_{args.dpo_track}.json")

    pairs_path = Path(args.output_dir) / f"dpo_{args.dpo_track}_vs_sft_pairs.jsonl"
    build_pairwise_file(
        generations_a=sft_generations,
        generations_b=dpo_generations,
        model_a_name="sft",
        model_b_name=f"dpo_{args.dpo_track}",
        output_path=pairs_path,
    )
    print(f"Wrote pairwise comparison file to {pairs_path}")


if __name__ == "__main__":
    main()
