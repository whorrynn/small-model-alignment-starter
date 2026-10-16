# Implementation Order

Every function below is a `TODO`/`NotImplementedError` stub. This is the
order to fill them in, chosen so each phase is testable (or at least
runnable) before you move to the next, and so you never debug the hardest
part (reward GPU jobs) through the hardest tooling (the judge pipeline) at
the same time.

## Phase 0 -- local, no GPU, no network (do this first)

Pure logic you can test on CPU against the public adapter tests before ever
opening Kaggle.

1. `src/prompt_format.py::format_prompt` -- pick the template. Everything
   downstream depends on this being decided once.
2. `src/batching.py::build_sft_batch_impl` -- then
   `build_dpo_batch_impl` (trivially reuses the first).
3. `src/logprobs.py::compute_response_logprobs_impl`
4. `src/losses.py::sft_loss_impl`, `dpo_loss_from_logps_impl`, `dpo_loss_impl`
5. `src/losses.py::implicit_rewards`, `reward_accuracy_and_margin`
6. `src/checkpoint.py::BestCheckpointTracker.is_better`

**Checkpoint:** run
`PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py`
locally (needs `pip install torch pytest`, CPU is fine). All tests should
pass before you touch Kaggle. This is also most of what the *private*
grader checks, so get it solid here.

## Phase 1 -- Kaggle setup (cheap, do once)

7. `scripts/task1_setup_check.py` -- already implemented, just run it on
   Kaggle to confirm the environment (Task 1).
8. `src/checkpoint.py::save_checkpoint` / `load_checkpoint` -- needed by
   every training script from here on.
9. `src/alpaca_eval_subset.py::build_fixed_alpacaeval_subset` -- needed
   before baseline generation.

## Phase 2 -- baseline + SFT (Task 2, Task 3)

10. `src/generation.py::generate_responses` -- get this working against the
    plain base model first (`scripts/generate_baseline.py`); it's reused
    unchanged for every later generation step.
11. `src/data_sft.py::load_sft_dataset`
12. `src/sft_train.py::train_sft` -- the full loop. This is the single
    longest piece of new code in the assignment; budget real time for it.
    Run `scripts/train_sft.py`.
13. `src/plot_metrics.py::plot_sft_curves`

**Checkpoint:** you now have an SFT checkpoint, its training curve, and a
baseline-vs-SFT qualitative comparison you can eyeball before spending GPU
hours on DPO.

## Phase 3 -- from-scratch DPO (Task 4)

Your `dpo_loss_impl`/`dpo_loss_from_logps_impl` are already correct from
Phase 0 -- this phase is purely the training-loop orchestration around them.

14. `src/data_dpo.py::load_preference_dataset`
15. `src/dpo_train.py::train_dpo` -- run twice via `scripts/train_dpo.py`,
    once per track (`hh_rlhf`, `ultrafeedback`), both starting from the same
    SFT checkpoint.
16. `src/plot_metrics.py::plot_dpo_curves`

## Phase 4 -- LoRA persona run (Task 5)

Independent of Phases 2-3 besides needing `transformers`/`peft`/`trl`
installed -- can be done in parallel with Phase 3 if you have two Kaggle
sessions going.

17. `src/lora_persona_train.py::build_trl_dpo_records`
18. `src/lora_persona_train.py::train_lora_persona`

## Phase 5 -- evaluation pipeline (Task 6, Task 7)

19. `scripts/generate_eval_outputs.py::load_model_and_tokenizer`
20. `src/pairwise.py::build_pairwise_file`
21. `src/judging.py::JUDGE_PROMPT_TEMPLATE` + `judge_pairwise_batch`
22. `src/judging.py::run_judging`

Run `scripts/generate_eval_outputs.py` for both tracks, then
`scripts/judge_pairwise_kaggle.py` on each resulting pairs file (from a
Kaggle notebook with benchmark/model-proxy access enabled -- manual Section
13.1).

## Phase 6 -- report (Task 8)

23. Fill in `REPORT.md` (already scaffolded with the manual's exact
    Section 14.1-14.7 headings and a results-at-a-glance table). Write this
    *as you go* through Phases 2-5 rather than all at the end -- you need
    the deviation/evidence trail from Section 14.3 while it's fresh, and
    Kaggle sessions are temporary so screenshots/numbers can otherwise get
    lost.

---

### Why this order

- Phase 0 is where the private grader's teeth are (exact masking, EOS
  handling, gradient isolation between policy/reference) -- get it right
  against the fast, free, offline public tests before it's buried inside a
  2-hour GPU job.
- SFT must exist before either DPO run (both start from the SFT
  checkpoint), so Phase 2 blocks Phase 3.
- LoRA (Phase 4) only depends on the provided persona data, not on your SFT
  or DPO checkpoints -- it's a good thing to run on a second Kaggle session
  while a from-scratch DPO run occupies the first.
- Judging (Phase 5) is deliberately last: it needs real generations from
  real checkpoints, and its own reliability logic (retries, resumability)
  is easiest to get right once there's nothing else in flux underneath it.
