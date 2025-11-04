# growth/data_loader.py
"""
Handles loading and balancing of the SNLI dataset.
"""
import json
import random
from pathlib import Path
from typing import List, Tuple


def load_snli(path: Path, limit: int) -> List[Tuple[str, str, str]]:
    """Loads SNLI data from a .jsonl file up to a limit."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gold = obj.get("gold_label", "").strip()
            if gold not in {"entailment", "neutral", "contradiction"}:
                continue
            s1, s2 = obj.get("sentence1", ""), obj.get("sentence2", "")
            if not s1 or not s2:
                continue
            data.append((s1, s2, gold))
            if len(data) >= limit:
                break
    return data


def balance_dataset(data: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Balances a dataset to have an equal number of samples per class."""
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    for _, _, g in data:
        if g in counts:
            counts[g] += 1

    # Handle cases where a class might be missing in a small sample
    valid_counts = [c for c in counts.values() if c > 0]
    if not valid_counts:
        return []  # Return empty if no valid labels found

    minc = min(valid_counts)

    balanced, seen = [], counts.fromkeys(counts, 0)
    random.shuffle(data)
    for s1, s2, g in data:
        if g in seen and seen[g] < minc:
            balanced.append((s1, s2, g))
            seen[g] += 1
    print(f"✅ Balanced {len(balanced)} samples ({minc} per class)")
    return balanced