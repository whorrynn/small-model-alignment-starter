"""Turn the downloaded Kaggle outputs into the numbers used in REPORT.md.

    bash kaggle/fetch_results.sh          # download into results/
    python kaggle/compile_results.py      # writes results/summary.json, copies plots

Every number in REPORT.md Sections 14.4-14.6 comes from results/summary.json.
"""

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

RESULTS = Path("results")
REPORT_ASSETS = Path("report_assets")

SFT_DIR = RESULTS / "cs5326-pa2-sft-run/project/outputs/sft/sft_run"
DPO_DIRS = {
    "hh_rlhf": RESULTS / "cs5326-pa2-dpo-hh/project/outputs/dpo_hh_rlhf/hh_rlhf_run",
    "ultrafeedback": RESULTS / "cs5326-pa2-dpo-uf/project/outputs/dpo_ultrafeedback/ultrafeedback_run",
}
LORA_DIR = RESULTS / "cs5326-pa2-lora/project/outputs/lora_persona"
GENERATIONS_DIR = RESULTS / "cs5326-pa2-eval-gen/project/outputs/eval"
JUDGED_DIR = RESULTS / "cs5326-pa2-judge/project/outputs/eval"
EVAL_SETS = {
    "hh_rlhf": Path("eval_sets/hh_rlhf_eval_300_seed42.json"),
    "ultrafeedback": RESULTS / "cs5326-pa2-sft-run/project/eval_sets/alpaca_eval_300_seed42.json",
}

# Rough refusal / apology detector. Only a proxy -- the judge is the real measure.
REFUSAL_PATTERN = re.compile(
    r"\b(I can('|’)t|I cannot|I can not|I'm (sorry|not able)|I am (sorry|not able)|I won('|’)t"
    r"|not able to (help|provide)|cannot (help|provide|assist)|can('|’)t (help|provide|assist)"
    r"|as an AI|I'm unable|I am unable)\b",
    re.IGNORECASE,
)


def readJson(path):
    with open(path, encoding = "utf-8") as handle:
        return json.load(handle)


def readJsonl(path):
    rows = []
    with open(path, encoding = "utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def meanAndConfidence(values):
    """Mean and a 95% normal-approximation half-width."""
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    halfWidth = 1.96 * math.sqrt(variance) / math.sqrt(count)
    return mean, halfWidth


def signTestPValue(wins, losses):
    """Two-sided sign test on non-tie verdicts (normal approximation)."""
    total = wins + losses
    zScore = (wins - total / 2) / math.sqrt(total / 4)
    return math.erfc(abs(zScore) / math.sqrt(2))


def isRepetitive(text):
    """True if any 8-word window appears 3 or more times (degenerate looping)."""
    words = text.split()
    windowCounts = Counter()
    for start in range(len(words) - 7):
        windowCounts[tuple(words[start:start + 8])] += 1
    for count in windowCounts.values():
        if count >= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------

def summarizeSft():
    records = readJsonl(SFT_DIR / "metrics.jsonl")
    bestMetadata = readJson(SFT_DIR / "checkpoints/best/metadata.json")
    finalMetadata = readJson(SFT_DIR / "checkpoints/final/metadata.json")
    return {
        "base_val_loss": records[0]["val_loss"],
        "highest_val_loss": max(record["val_loss"] for record in records),
        "best_step": bestMetadata["step"],
        "best_val_loss": bestMetadata["metrics"]["val_loss"],
        "final_step": finalMetadata["step"],
        "final_val_loss": finalMetadata["metrics"]["val_loss"],
    }


def summarizeDpo(track):
    records = readJsonl(DPO_DIRS[track] / "metrics.jsonl")
    bestMetadata = readJson(DPO_DIRS[track] / "checkpoints/best/metadata.json")

    stepZero = None
    finalRecord = None
    for record in records:
        if record["step"] == 0:
            stepZero = record
        if record.get("final"):
            finalRecord = record

    return {
        "step0": stepZero,
        "best_step": bestMetadata["step"],
        "best": bestMetadata["metrics"],
        "final": finalRecord,
    }


def summarizeLora():
    history = readJson(LORA_DIR / "log_history.json")
    trainLosses = []
    evalRecords = []
    for record in history:
        if "loss" in record:
            trainLosses.append(record["loss"])
        if "eval_loss" in record:
            evalRecords.append(record)

    lastEval = evalRecords[-1]
    return {
        "first_train_loss": trainLosses[0],
        "last_train_loss": trainLosses[-1],
        "eval_loss": lastEval["eval_loss"],
        "eval_reward_accuracy": lastEval["eval_rewards/accuracies"],
        "eval_reward_margin": lastEval["eval_rewards/margins"],
    }


# ---------------------------------------------------------------------------
# Generations and judging
# ---------------------------------------------------------------------------

def loadOutputs(name):
    outputs = {}
    for row in readJson(GENERATIONS_DIR / f"{name}.json"):
        outputs[row["id"]] = row["output"]
    return outputs


def outputStats(outputs):
    wordCounts = []
    hitCap = 0
    repetitive = 0
    refusals = 0
    for text in outputs.values():
        wordCount = len(text.split())
        wordCounts.append(wordCount)
        if wordCount >= 180:  # ~256 tokens; responses this long were cut off by max_new_tokens
            hitCap += 1
        if isRepetitive(text):
            repetitive += 1
        if REFUSAL_PATTERN.search(text):
            refusals += 1

    total = len(outputs)
    return {
        "mean_words": sum(wordCounts) / total,
        "hit_256_token_cap": hitCap / total,
        "repetitive": repetitive / total,
        "refusal_phrase": refusals / total,
    }


def summarizeTrack(track):
    evalItems = readJson(EVAL_SETS[track])
    instructions = {}
    categories = {}
    for item in evalItems:
        instructions[item["id"]] = item["instruction"]
        categories[item["id"]] = item.get("category")

    sftOutputs = loadOutputs(f"sft_{track}")
    dpoOutputs = loadOutputs(f"dpo_{track}")
    pairs = {}
    for pair in readJsonl(GENERATIONS_DIR / f"dpo_{track}_vs_sft_pairs.jsonl"):
        pairs[pair["id"]] = pair
    judged = readJsonl(JUDGED_DIR / f"dpo_{track}_vs_sft_judged.jsonl")

    failuresPath = JUDGED_DIR / f"dpo_{track}_vs_sft_judged_failures.jsonl"
    failures = readJsonl(failuresPath) if failuresPath.exists() else []

    # Check 1: response_a / response_b really are the outputs the hidden mapping claims.
    mappingMismatches = 0
    dpoInSlotA = 0
    for pairId, pair in pairs.items():
        for slot in ("A", "B"):
            modelName = pair["position_to_model"][slot]
            if modelName == "sft":
                expected = sftOutputs[pairId]
            else:
                expected = dpoOutputs[pairId]
            if pair["response_" + slot.lower()] != expected:
                mappingMismatches += 1
        if pair["position_to_model"]["A"].startswith("dpo"):
            dpoInSlotA += 1

    # Check 2: recount wins / losses / ties straight from the verdicts.
    results = {}
    verdictCounts = Counter()
    for record in judged:
        verdict = record["verdict"]
        verdictCounts[verdict] += 1
        if verdict == "tie":
            results[record["id"]] = "tie"
        elif record["position_to_model"][verdict].startswith("dpo"):
            results[record["id"]] = "win"
        else:
            results[record["id"]] = "loss"

    resultCounts = Counter(results.values())
    wins = resultCounts["win"]
    losses = resultCounts["loss"]
    ties = resultCounts["tie"]

    scoreOf = {"win": 1.0, "tie": 0.5, "loss": 0.0}
    scores = [scoreOf[result] for result in results.values()]
    winRate, winRateHalfWidth = meanAndConfidence(scores)

    # Check 3: duplicated prompts -- average per unique prompt, and count verdict flips.
    resultsByPrompt = {}
    for pairId, result in results.items():
        resultsByPrompt.setdefault(instructions[pairId], []).append(result)

    perPromptScores = []
    repeatedPrompts = 0
    inconsistentPrompts = 0
    for promptResults in resultsByPrompt.values():
        perPromptScores.append(sum(scoreOf[result] for result in promptResults) / len(promptResults))
        if len(promptResults) > 1:
            repeatedPrompts += 1
            if len(set(promptResults)) > 1:
                inconsistentPrompts += 1
    dedupWinRate, dedupHalfWidth = meanAndConfidence(perPromptScores)

    identicalOutputs = 0
    for pairId in sftOutputs:
        if sftOutputs[pairId].strip() == dpoOutputs[pairId].strip():
            identicalOutputs += 1

    nonTieVerdicts = verdictCounts["A"] + verdictCounts["B"]

    summary = {
        "sample_count": len(judged),
        "unique_ids_judged": len(results),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "unknown": 0,
        "failures": len(failures),
        "win_rate_ties_half": winRate,
        "win_rate_ci95": winRateHalfWidth,
        "win_rate_excl_ties": wins / (wins + losses),
        "sign_test_p": signTestPValue(wins, losses),
        "mapping_mismatches": mappingMismatches,
        "dpo_in_slot_a": dpoInSlotA,
        "judge_picked_a_among_non_ties": verdictCounts["A"] / nonTieVerdicts,
        "unique_prompts": len(resultsByPrompt),
        "repeated_prompts": repeatedPrompts,
        "repeated_prompts_with_inconsistent_verdicts": inconsistentPrompts,
        "dedup_win_rate": dedupWinRate,
        "dedup_win_rate_ci95": dedupHalfWidth,
        "identical_sft_dpo_outputs": identicalOutputs,
        "sft_output_stats": outputStats(sftOutputs),
        "dpo_output_stats": outputStats(dpoOutputs),
    }

    if any(categories.values()):
        perCategory = {}
        for pairId, result in results.items():
            category = categories[pairId]
            perCategory.setdefault(category, Counter())[result] += 1
        summary["per_category"] = {}
        for category in sorted(perCategory):
            counts = perCategory[category]
            total = sum(counts.values())
            summary["per_category"][category] = {
                "wins": counts["win"],
                "losses": counts["loss"],
                "ties": counts["tie"],
                "win_rate_ties_half": (counts["win"] + 0.5 * counts["tie"]) / total,
            }

    return summary


def main():
    REPORT_ASSETS.mkdir(parents = True, exist_ok = True)
    for plotPath in RESULTS.glob("*/project/report_assets/*.png"):
        shutil.copy(plotPath, REPORT_ASSETS / plotPath.name)

    summary = {
        "sft": summarizeSft(),
        "dpo_hh_rlhf": summarizeDpo("hh_rlhf"),
        "dpo_ultrafeedback": summarizeDpo("ultrafeedback"),
        "lora_persona": summarizeLora(),
        "judging": {
            "hh_rlhf": summarizeTrack("hh_rlhf"),
            "ultrafeedback": summarizeTrack("ultrafeedback"),
        },
    }

    outputPath = RESULTS / "summary.json"
    with open(outputPath, "w", encoding = "utf-8") as handle:
        json.dump(summary, handle, indent = 2)
    print("wrote", outputPath)

    for track, trackSummary in summary["judging"].items():
        print(
            f"{track}: {trackSummary['wins']}W/{trackSummary['losses']}L/{trackSummary['ties']}T  "
            f"win rate {trackSummary['win_rate_ties_half']:.3f} +/- {trackSummary['win_rate_ci95']:.3f}  "
            f"p={trackSummary['sign_test_p']:.4f}  mapping mismatches={trackSummary['mapping_mismatches']}"
        )


if __name__ == "__main__":
    main()
