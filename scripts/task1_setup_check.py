"""Task 1 -- repository and Kaggle setup sanity check.

Run on Kaggle right after cloning the repo and installing requirements
(manual Task 1, items 1-4):

    python scripts/task1_setup_check.py

Fully implemented -- this is pure environment verification, not assignment
math.
"""

from __future__ import annotations

import torch


def main() -> None:
    print("torch:", torch.__version__)

    for package_name in ("transformers", "datasets", "accelerate", "peft", "trl"):
        try:
            module = __import__(package_name)
            print(f"{package_name}:", getattr(module, "__version__", "unknown"))
        except ImportError:
            print(f"{package_name}: NOT INSTALLED")

    if torch.cuda.is_available():
        print("GPUs visible:", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            print(f"  cuda:{index} -> {torch.cuda.get_device_name(index)}")
    else:
        print("No CUDA device visible (expected locally; Kaggle should show 2x Tesla T4).")


if __name__ == "__main__":
    main()
