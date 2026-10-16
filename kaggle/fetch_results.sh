#!/usr/bin/env bash
# Download each Kaggle notebook's outputs into results/<notebook>/.
# Model weights and optimizer state (several GB) are skipped -- only metrics,
# plots, generations, and judgments are downloaded.
#
#   bash kaggle/fetch_results.sh                       # all six notebooks
#   bash kaggle/fetch_results.sh cs5326-pa2-judge      # just one
set -euo pipefail
cd "$(dirname "$0")/.."

# Kaggle's CLI crashes on non-ASCII log text under the Windows console encoding.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

SKIP_LARGE_FILES='^(?!.*(safetensors|optimizer_state_dict|pytorch_model|adapter_model)).*$'

if [ "$#" -gt 0 ]; then
  notebooks="$*"
else
  notebooks="cs5326-pa2-sft-run cs5326-pa2-lora cs5326-pa2-dpo-hh cs5326-pa2-dpo-uf cs5326-pa2-eval-gen cs5326-pa2-judge"
fi

for notebook in $notebooks; do
  echo "fetching $notebook"
  mkdir -p "results/$notebook"
  kaggle kernels output "rayyanhasib/$notebook" -p "results/$notebook" --file-pattern "$SKIP_LARGE_FILES" -o -q \
    || echo "fetch failed: $notebook (re-run this script to retry)"
done
