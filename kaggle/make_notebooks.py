"""Build the six Kaggle notebooks (plus their kernel-metadata.json files).

    python kaggle/make_notebooks.py

Writes kaggle/kaggle_kernels/<name>/. Push one with:

    kaggle kernels push -p kaggle/kaggle_kernels/<name>

Run order (each notebook reads the previous ones through kernel_sources):
    sft_run -> dpo_hh + dpo_uf -> eval_gen -> judge      (lora is independent)
"""

import json
from pathlib import Path

OWNER = "rayyanhasib"
REPO_DATASET = "rayyanhasib/cs5326-pa2-repo"
OUTPUT_DIR = Path(__file__).parent / "kaggle_kernels"


# ---------------------------------------------------------------------------
# Notebook cells
# ---------------------------------------------------------------------------

def codeCell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends = True),
    }


def markdownCell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends = True),
    }


def listFilesLoop(folder, pattern = "*", filesOnly = True):
    """Notebook code that prints every path under `folder` (needs `import pathlib`)."""
    loop = f"for p in sorted(pathlib.Path('{folder}').rglob('{pattern}')):\n"
    if filesOnly:
        loop += "    if p.is_file():\n"
        loop += "        print(p)\n"
    else:
        loop += "    print(p)\n"
    return loop


# Prints the GPUs (if any).
GPU_CELL = codeCell("""
import subprocess
try:
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print('no GPU (nvidia-smi not found)')
""")

# Installs the same package versions the manual recommends (skipped for the judge).
PIP_CELL = codeCell("""
%pip uninstall -y -q torchao
%pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors matplotlib "peft>=0.11.0" "trl>=0.9.0" "bitsandbytes>=0.43.0"
""")

# Copies the repo out of the (read-only) dataset into /kaggle/working/project,
# so every output is saved, and defines run(), which stops the notebook if a
# script fails (a plain "!python ..." cell hides failures).
SETUP_CELL = codeCell("""
import pathlib, os, sys, shutil

for root, dirs, files in os.walk('/kaggle/input'):
    depth = root.count(os.sep) - '/kaggle/input'.count(os.sep)
    if depth <= 3:
        print(root)
    if depth >= 4:
        dirs[:] = []

matches = [p.parent.parent for p in pathlib.Path('/kaggle/input').rglob('src/config.py')
           if (p.parent.parent / 'scripts' / 'train_sft.py').exists() and '/project/' not in str(p)]
assert matches, 'repo dataset (src/config.py + scripts/) not found under /kaggle/input'
repo_src = matches[0]
print('repo dataset found at', repo_src)

proj = pathlib.Path('/kaggle/working/project')
if not proj.exists():
    shutil.copytree(repo_src, proj)
os.chdir(proj)
sys.path.insert(0, str(proj))
os.environ['PYTHONPATH'] = str(proj)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import subprocess
def run(cmd):
    print('$', cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f'command failed with exit code {proc.returncode}: {cmd}')

import importlib
for name in ('torch', 'transformers', 'datasets', 'peft', 'trl'):
    try:
        print(name, importlib.import_module(name).__version__)
    except Exception as error:
        print(name, 'unavailable:', error)
""")

# Finds the SFT checkpoint saved by the cs5326-pa2-sft-run notebook.
FIND_SFT_CELL = codeCell("""
import glob
candidates = glob.glob('/kaggle/input/**/project/outputs/sft/sft_run/checkpoints/best', recursive=True)
assert candidates, "SFT checkpoint not found under /kaggle/input"
sft_ckpt = candidates[0]
print('Using SFT checkpoint:', sft_ckpt)
""")


# ---------------------------------------------------------------------------
# Writing one notebook + its metadata
# ---------------------------------------------------------------------------

def writeKernel(folderName, title, cells, kernelSources = None, useGpu = True, installPackages = True):
    folder = OUTPUT_DIR / folderName
    folder.mkdir(parents = True, exist_ok = True)
    codeFile = f"{title}.ipynb"

    allCells = [GPU_CELL]
    if installPackages:
        allCells.append(PIP_CELL)
    allCells.append(SETUP_CELL)
    allCells.extend(cells)

    notebook = {
        "cells": allCells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(folder / codeFile, "w", encoding = "utf-8") as handle:
        json.dump(notebook, handle, indent = 1)

    metadata = {
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": useGpu,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [REPO_DATASET],
        "kernel_sources": kernelSources or [],
        "competition_sources": [],
        "model_sources": [],
        "id": f"{OWNER}/{title}",
        "title": title,
        "code_file": codeFile,
    }
    if useGpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    with open(folder / "kernel-metadata.json", "w", encoding = "utf-8") as handle:
        json.dump(metadata, handle, indent = 2)

    print("wrote", folder)


# ---------------------------------------------------------------------------
# The six notebooks
# ---------------------------------------------------------------------------

def main():
    sftCells = [
        markdownCell("## Task 1 -- setup check"),
        codeCell('run(f"python scripts/task1_setup_check.py")'),
        markdownCell("## Task 2 -- baseline generation (also builds eval_sets/alpaca_eval_300_seed42.json)"),
        codeCell('run(f"python scripts/generate_baseline.py --output outputs/eval/baseline_alpaca_eval.json")'),
        markdownCell("## Task 3 -- SFT training, full scale (30k train / 1k val, per src/config.py)"),
        codeCell('run(f"python scripts/train_sft.py --run_name sft_run --output_dir outputs")'),
        markdownCell("## Confirm outputs"),
        codeCell(
            "import pathlib\n"
            + listFilesLoop("outputs")
            + listFilesLoop("eval_sets", filesOnly = False)
            + listFilesLoop("report_assets", filesOnly = False)
        ),
    ]
    writeKernel("sft_run", "cs5326-pa2-sft-run", sftCells)

    loraCells = [
        markdownCell("## Task 5 -- TRL DPOTrainer + PEFT/LoRA persona run"),
        codeCell('run(f"python scripts/train_lora_persona.py")'),
        markdownCell("## Confirm outputs"),
        codeCell("import pathlib\n" + listFilesLoop("outputs/lora_persona")),
    ]
    writeKernel("lora", "cs5326-pa2-lora", loraCells)

    dpoTracks = [
        ("dpo_hh", "cs5326-pa2-dpo-hh", "hh_rlhf", "HH-RLHF"),
        ("dpo_uf", "cs5326-pa2-dpo-uf", "ultrafeedback", "UltraFeedback"),
    ]
    for folderName, title, track, trackLabel in dpoTracks:
        dpoCells = [
            markdownCell("## Locate SFT checkpoint from kernel_sources"),
            FIND_SFT_CELL,
            markdownCell(f"## Task 4 -- from-scratch DPO, {trackLabel} track, full scale (15k pairs / 1k val, per src/config.py)"),
            codeCell(f'run(f"python scripts/train_dpo.py --track {track} --sft_checkpoint {{sft_ckpt}} --output_dir outputs")'),
            markdownCell("## Confirm outputs"),
            codeCell("import pathlib\n" + listFilesLoop(f"outputs/dpo_{track}")),
        ]
        writeKernel(folderName, title, dpoCells, kernelSources = [f"{OWNER}/cs5326-pa2-sft-run"])

    evalCells = [
        markdownCell("## Locate checkpoints and the alpaca-eval subset file from kernel_sources"),
        codeCell("""
import glob, shutil, os
sft_cands = glob.glob('/kaggle/input/**/project/outputs/sft/sft_run/checkpoints/best', recursive=True)
dpo_hh_cands = glob.glob('/kaggle/input/**/project/outputs/dpo_hh_rlhf/hh_rlhf_run/checkpoints/best', recursive=True)
dpo_uf_cands = glob.glob('/kaggle/input/**/project/outputs/dpo_ultrafeedback/ultrafeedback_run/checkpoints/best', recursive=True)
alpaca_cands = glob.glob('/kaggle/input/**/project/eval_sets/alpaca_eval_300_seed42.json', recursive=True)
assert sft_cands and dpo_hh_cands and dpo_uf_cands and alpaca_cands, (sft_cands, dpo_hh_cands, dpo_uf_cands, alpaca_cands)
sft_ckpt = sft_cands[0]
dpo_hh_ckpt = dpo_hh_cands[0]
dpo_uf_ckpt = dpo_uf_cands[0]
os.makedirs('eval_sets', exist_ok=True)
shutil.copy(alpaca_cands[0], 'eval_sets/alpaca_eval_300_seed42.json')
print('sft_ckpt', sft_ckpt)
print('dpo_hh_ckpt', dpo_hh_ckpt)
print('dpo_uf_ckpt', dpo_uf_ckpt)
"""),
        markdownCell("## Task 6 -- generation + pairwise file, HH-RLHF track"),
        codeCell('run(f"python scripts/generate_eval_outputs.py --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --sft_checkpoint {sft_ckpt} --dpo_checkpoint {dpo_hh_ckpt} --dpo_track hh_rlhf --output_dir outputs/eval")'),
        markdownCell("## Task 6 -- generation + pairwise file, UltraFeedback track (AlpacaEval prompts)"),
        codeCell('run(f"python scripts/generate_eval_outputs.py --eval_set eval_sets/alpaca_eval_300_seed42.json --sft_checkpoint {sft_ckpt} --dpo_checkpoint {dpo_uf_ckpt} --dpo_track ultrafeedback --output_dir outputs/eval")'),
        markdownCell("## Confirm outputs"),
        codeCell("import pathlib\n" + listFilesLoop("outputs/eval")),
    ]
    writeKernel(
        "eval_gen",
        "cs5326-pa2-eval-gen",
        evalCells,
        kernelSources = [f"{OWNER}/cs5326-pa2-sft-run", f"{OWNER}/cs5326-pa2-dpo-hh", f"{OWNER}/cs5326-pa2-dpo-uf"],
    )

    # The judge notebook must be set up once on the Kaggle website with
    # File > "Set as Benchmark Task" (manual Section 13.1). That image has no
    # torch and no GPU, so it skips the package install.
    judgeCommand = (
        "python scripts/judge_pairwise_kaggle.py --pairs_path {pairs} --output_path outputs/eval/{out}"
        " --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments"
    )
    hhCommand = judgeCommand.format(pairs = "{hh_pairs_path}", out = "dpo_hh_rlhf_vs_sft_judged.jsonl") + " --safety_mode"
    ufCommand = judgeCommand.format(pairs = "{uf_pairs_path}", out = "dpo_ultrafeedback_vs_sft_judged.jsonl")

    judgeCells = [
        markdownCell("## Confirm kaggle_benchmarks is importable (requires 'Set as Benchmark Task')"),
        codeCell("""
import kaggle_benchmarks
print('kaggle_benchmarks module found:', kaggle_benchmarks)
print([n for n in dir(kaggle_benchmarks) if not n.startswith('_')])
"""),
        markdownCell("## Locate pairwise files from kernel_sources"),
        codeCell("""
import glob
hh_pairs = glob.glob('/kaggle/input/**/project/outputs/eval/dpo_hh_rlhf_vs_sft_pairs.jsonl', recursive=True)
uf_pairs = glob.glob('/kaggle/input/**/project/outputs/eval/dpo_ultrafeedback_vs_sft_pairs.jsonl', recursive=True)
assert hh_pairs and uf_pairs, (hh_pairs, uf_pairs)
hh_pairs_path = hh_pairs[0]
uf_pairs_path = uf_pairs[0]
print('hh_pairs_path', hh_pairs_path)
print('uf_pairs_path', uf_pairs_path)
"""),
        markdownCell("## Task 7 -- judge HH-RLHF track (safety_mode on: safety-sensitive eval set)"),
        codeCell(f'run(f"{hhCommand}")'),
        markdownCell("## Task 7 -- judge UltraFeedback/AlpacaEval track"),
        codeCell(f'run(f"{ufCommand}")'),
        markdownCell("## Confirm outputs"),
        codeCell(
            "import pathlib\n"
            + listFilesLoop("outputs/eval", "*judged*", filesOnly = False)
            + listFilesLoop("outputs/eval", "*failures*", filesOnly = False)
        ),
    ]
    writeKernel(
        "judge",
        "cs5326-pa2-judge",
        judgeCells,
        kernelSources = [f"{OWNER}/cs5326-pa2-eval-gen"],
        useGpu = False,
        installPackages = False,
    )


if __name__ == "__main__":
    main()
