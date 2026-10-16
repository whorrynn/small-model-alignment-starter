"""Single source of truth for prompt formatting.

Manual Section 10.3: "Choose an appropriate prompt format and use it
consistently for SFT, DPO, and evaluation. Document the exact template or
chat-template settings in the report."

Every other module (src/batching.py, src/generation.py, the SFT and DPO
training loops) must format prompts by calling format_prompt from HERE, not
by re-implementing their own template. That is what "consistently" means in
the manual -- SFT and DPO must see the exact same prompt shape the eval
scripts generate from, or the comparison in Task 8 (Q19-Q23-equivalent
report sections) is not apples-to-apples.
"""

from __future__ import annotations


def format_prompt(instruction: str, tokenizer=None) -> str:
    """Return the exact string fed to the tokenizer as the prompt half of a
    (prompt, response) training or generation pair.

    Template choice: a plain custom template, applied identically for the
    base Qwen2.5-0.5B model across SFT, DPO, and generation/evaluation. We
    do not use Qwen's native chat template here because Qwen2.5-0.5B is the
    *base* (non-instruct) checkpoint -- the chat template format is not
    something it has been trained to expect, and a custom template is
    trivial to keep byte-identical everywhere. `tokenizer` is accepted but
    unused, to keep the call signature uniform with call sites that pass it.

    Documented in REPORT.md Section 14.2.
    """
    return f"Instruction:\n{instruction}\n\nResponse:\n"
