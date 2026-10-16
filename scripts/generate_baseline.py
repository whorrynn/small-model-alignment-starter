"""Task 2 -- generate baseline responses from the un-finetuned base model.

    python scripts/generate_baseline.py --output outputs/eval/baseline_alpaca_eval.json

CLI wiring plus the generation/subset logic it calls (src.alpaca_eval_subset,
src.generation) are both implemented; this only needs network + a GPU/CPU
with transformers installed, which is why it's meant to run on Kaggle.
"""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.alpaca_eval_subset import build_fixed_alpacaeval_subset
from src.config import SFT_CONFIG
from src.generation import generate_responses, save_generations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/eval/baseline_alpaca_eval.json")
    parser.add_argument("--n_prompts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subset = build_fixed_alpacaeval_subset(n=args.n_prompts, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(SFT_CONFIG.model_name)
    model = AutoModelForCausalLM.from_pretrained(SFT_CONFIG.model_name)

    generations = generate_responses(
        model,
        tokenizer,
        prompts=[item["instruction"] for item in subset],
        ids=[item["id"] for item in subset],
        generator_name="qwen2.5-0.5b-base",
        dataset_name="alpaca_eval",
    )
    save_generations(generations, args.output)
    print(f"Wrote {len(generations)} baseline generations to {args.output}")


if __name__ == "__main__":
    main()
