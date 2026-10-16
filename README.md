# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

### Local scaffold

This checkout has been extended with a full `src/`, `scripts/`, and
`REPORT.md` scaffold on top of the bare starter package above, mirroring the
`src/` + `tests/adapters.py` boundary used in PA1. Every function that needs
real implementation is a `TODO`-documented `NotImplementedError` stub citing
the exact manual section it satisfies; nothing in it is pre-solved.

- **`IMPLEMENTATION_PLAN.md`** -- the order to implement things in, and why.
- **`src/`** -- all training/eval logic (`prompt_format.py`, `batching.py`,
  `logprobs.py`, `losses.py`, `data_sft.py`, `data_dpo.py`, `checkpoint.py`,
  `sft_train.py`, `dpo_train.py`, `lora_persona_train.py`, `generation.py`,
  `alpaca_eval_subset.py`, `pairwise.py`, `judging.py`, `plot_metrics.py`,
  `config.py`, `utils.py`). `submission_adapter.py` is already wired to
  delegate into `src/batching.py`, `src/logprobs.py`, and `src/losses.py` --
  once those pass the public tests, the adapter needs no further edits.
- **`scripts/`** -- one CLI entry point per manual task (`task1_setup_check.py`
  through `judge_pairwise_kaggle.py`, the latter matching the manual's exact
  example invocation).
- **`REPORT.md`** -- pre-filled with the manual's Section 14.1-14.7 headings
  and a results-at-a-glance table.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.
