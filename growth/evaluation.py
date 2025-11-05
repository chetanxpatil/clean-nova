"""
GrowthMind evaluation loop — refactored & modular.

Key improvements:
- Clear label/rule mapping utilities
- Isolated reward shaping (easy to tune)
- Clean training step wrapper
- Safer progress printing + optional test-case narration
- TQDM handled uniformly
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np

from core.semantic import IntentVector
from growth.mind.growth_mind import GrowthMind
from growth.phi_computer import phi_raw_for_pair
from growth.lattice_encoding import text_to_lattice
from growth.config import CONFIG
from growth.language import express_decision

try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None


# -------------------------------------------------------------------
# Constants & simple mappings
# -------------------------------------------------------------------

LABELS: Tuple[str, str, str] = ("entailment", "neutral", "contradiction")
LABEL_IDX: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}

GOLD_TO_RULE: Dict[str, str] = {
    "entailment": "merge",
    "contradiction": "branch",
    "neutral": "stabilize",
}

RULE_TO_LABEL: Dict[str, str] = {
    "merge": "entailment",
    "branch": "contradiction",
    "stabilize": "neutral",
    "revert": "neutral",  # treat revert as 'hold/neutral' for classification
}


# -------------------------------------------------------------------
# Config dataclasses
# -------------------------------------------------------------------

@dataclass
class EvalOptions:
    title: str
    is_training: bool = True
    show_test_examples: int = 5  # when not training
    progress: bool = True        # override or inherit CONFIG["progress"]


@dataclass
class RewardParams:
    correct_reward: float = +1.0
    incorrect_penalty: float = -1.0
    memory_phi_bias_scale: float = 0.05   # scales tanh(mean(prior))
    memory_bonus_scale: float = 0.1       # scales (1 - coupling_similarity)
    r2r_multiplier: float = 0.1
    r2r_max_bonus: float = 0.5


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def _use_tqdm(seq: Iterable, desc: str, enable: bool) -> Iterable:
    if enable and _tqdm is not None:
        return _tqdm(seq, desc=desc, dynamic_ncols=True)
    return seq


def _safe_mean(x: np.ndarray) -> float:
    try:
        return float(np.mean(x))
    except Exception:  # noqa: BLE001
        return 0.0


def _predict_label_from_rule(rule: str) -> str:
    return RULE_TO_LABEL.get(rule, "neutral")


def _update_confusion(cm: np.ndarray, gold: str, pred: str) -> None:
    cm[LABEL_IDX[gold], LABEL_IDX[pred]] += 1


def _progress_print(step: int, total: int, correct: int, phi_vals: List[float], mind: GrowthMind) -> None:
    ratio = step / max(1, total)
    acc = correct / max(1, step)
    phi_avg = _safe_mean(np.array(phi_vals))
    print(f"\n📍 Step {step:5d} / {total} ({ratio:.1%} complete)")
    print(f"   Accuracy: {acc:.3f} | Φ̄ = {phi_avg:+.3f}")
    try:
        print(f"   Policy Snapshot → {mind.policy.describe()}")
        mind.reflect()
    except Exception as e:  # noqa: BLE001
        print(f"   (Diagnostics skipped: {e})")


# -------------------------------------------------------------------
# Core math helpers
# -------------------------------------------------------------------

def _compute_phi(mind: GrowthMind, s1: str, s2: str) -> Tuple[float, float]:
    """Return (phi, phi_raw), applying mind.phi_sign."""
    phi_raw = float(phi_raw_for_pair(s1, s2))
    phi = float(mind.phi_sign) * phi_raw
    return phi, phi_raw


def _memory_keys(gold: str, pred: str, chosen_rule: str) -> Tuple[str, str]:
    """(store_key, recall_key)"""
    store_key = f"{GOLD_TO_RULE.get(gold, 'stabilize')}_{pred}"
    recall_key = f"{chosen_rule}_{gold}"
    return store_key, recall_key


def _reward_shaping(
    *,
    is_correct: bool,
    phi: float,
    prior: Optional[np.ndarray],
    coupling_delta: np.ndarray,
    params: RewardParams,
) -> Tuple[float, float]:
    """
    Compute total reward and potentially adjusted phi (memory bias).
    Returns (total_reward, adjusted_phi).
    """
    total = 0.0
    total += params.correct_reward if is_correct else params.incorrect_penalty

    # Memory shaping (if prior is available)
    if prior is not None:
        # small additive bias to phi from memory
        phi_bias = params.memory_phi_bias_scale * np.tanh(_safe_mean(prior))
        phi_adj = float(phi + phi_bias)

        # reward for being consistent with prior coupling (looser = higher)
        # similarity in [0, 1) via tanh(|prior - delta|)
        coupling_similarity = float(np.tanh(_safe_mean(np.abs(prior - coupling_delta))))
        memory_bonus = params.memory_bonus_scale * (1.0 - coupling_similarity)
        total += memory_bonus
    else:
        phi_adj = phi

    # Curiosity / R2R bonus: highest near uncertainty (|phi| ~ 0)
    confidence = abs(phi_adj)
    uncertainty = 1.0 - confidence
    curiosity = params.r2r_max_bonus * uncertainty * params.r2r_multiplier
    total += curiosity

    return total, phi_adj


def _train_one(
    *,
    mind: GrowthMind,
    s1: str,
    s2: str,
    gold: str,
    chosen_rule: str,
    phi: float,
    phi_raw: float,
    reward_params: RewardParams,
) -> None:
    """
    Execute the training updates (memory coupling; intent; policy update).
    """
    # Build lattices and coupling delta
    A = text_to_lattice(s1)
    B = text_to_lattice(s2)
    C = A.weights - B.weights  # coupling delta

    # Memory bookkeeping
    gold_rule = GOLD_TO_RULE.get(gold, "stabilize")
    store_key, recall_key = _memory_keys(gold, _predict_label_from_rule(chosen_rule), chosen_rule)

    # Persist coupling signal keyed by "what should have happened" vs "what did"
    mind.note_coupling(f"{gold_rule}_{_predict_label_from_rule(chosen_rule)}", C, phi, A.weights.shape, B.weights.shape)

    # Probe memory for prior to shape reward
    prior = mind.suggest_coupling(recall_key, A.weights.shape, B.weights.shape)

    # Intent assembly (phi may be nudged by memory shaping)
    intent = IntentVector(
        polarity=float(phi),
        raw_polarity=float(phi_raw),
        delta_energy=0.0,
        rotation_seq="",
        observer="Om",
    )

    # Reward shaping (extrinsic + memory + curiosity)
    is_correct = (_predict_label_from_rule(chosen_rule) == gold)
    total_reward, phi_adj = _reward_shaping(
        is_correct=is_correct,
        phi=phi,
        prior=prior,
        coupling_delta=C,
        params=reward_params,
    )
    intent.polarity = float(phi_adj)

    # Mind step & policy update
    mind.step(intent, note=f"{gold}->{_predict_label_from_rule(chosen_rule)}", external_correct=is_correct)
    mind.policy.update(chosen_rule, total_reward)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def evaluate_with_mind(
    pairs: List[Tuple[str, str, str]],
    phi_sign: float,
    neutral_band: float,
    title: str,
    mind: GrowthMind,
    is_training: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate (and optionally train) GrowthMind on (premise, hypothesis, gold) triples.

    Returns:
        confusion_matrix (3x3), accuracy
    """
    # Calibrate mind
    mind.neutral_band = float(neutral_band)
    mind.phi_sign = float(phi_sign)

    opts = EvalOptions(
        title=title,
        is_training=is_training,
        show_test_examples=5,
        progress=bool(CONFIG.get("progress", True)),
    )
    reward_params = RewardParams()

    # Stats
    cm = np.zeros((3, 3), dtype=int)
    phi_vals: List[float] = []
    rule_counts: Dict[str, int] = {"merge": 0, "branch": 0, "stabilize": 0, "revert": 0}
    correct = 0

    print(f"\n🔍 Evaluating {opts.title} on {len(pairs)} samples (Mind Neutral Band: {mind.neutral_band:.3f})...")
    it = _use_tqdm(pairs, desc=opts.title, enable=opts.progress)

    for i, (s1, s2, gold) in enumerate(it, start=1):
        # 1) Mind’s internal score
        phi, phi_raw = _compute_phi(mind, s1, s2)
        phi_vals.append(phi)

        # 2) Policy chooses a rule, map to classification
        rule = mind.choose_rule(phi)
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
        pred = _predict_label_from_rule(rule)

        # 3) Book-keeping
        is_correct = (pred == gold)
        correct += int(is_correct)
        _update_confusion(cm, gold, pred)

        # 4) Train (memory + policy) if requested
        if opts.is_training:
            _train_one(
                mind=mind,
                s1=s1,
                s2=s2,
                gold=gold,
                chosen_rule=rule,
                phi=phi,
                phi_raw=phi_raw,
                reward_params=reward_params,
            )
        else:
            # Lightweight narration for the first few examples
            if i <= max(0, int(opts.show_test_examples)):
                expression = express_decision(rule, phi)
                print("\n" + "-" * 20 + f" TEST EXAMPLE {i} " + "-" * 20)
                print(f"  Premise:    {s1}")
                print(f"  Hypothesis: {s2}")
                print(f"  Gold Label: {gold} ({'Correct' if is_correct else 'INCORRECT'})")
                print(f"  Mind's Internal Choice: {rule}")
                print(f"  Mind's Expression: {expression}")

        # 5) Progress snapshot
        if opts.progress and (i % 1000 == 0):
            _progress_print(i, len(pairs), correct, phi_vals, mind)

    # --- Final stats ---
    dist_by_pred = {l: int(sum(cm[:, j])) for j, l in enumerate(LABELS)}
    acc = correct / max(1, len(pairs))
    phi_array = np.array(phi_vals, dtype=float)
    phi_mean = _safe_mean(phi_array)
    phi_min = float(np.min(phi_array)) if phi_array.size else 0.0
    phi_max = float(np.max(phi_array)) if phi_array.size else 0.0

    total_seen = max(1, len(pairs))
    rule_dist = {k: f"{(v / total_seen) * 100:.1f}%" for k, v in rule_counts.items()}

    print(
        f"\n📊 {opts.title}: acc={acc:.3f}, "
        f"Φ mean={phi_mean:+.3f}, range=[{phi_min:+.3f},{phi_max:+.3f}], dist={dist_by_pred}"
    )
    print(f"   Rule Choices: {rule_dist}")

    header = " " * 13 + " ".join([f"{l:<13}" for l in LABELS])
    print("Confusion (Gold \\ Pred):")
    print(header)
    print("-" * len(header))
    for i, l in enumerate(LABELS):
        row = f"  {l:<13}" + " ".join([f"{cm[i, j]:<13}" for j in range(3)])
        print(row)

    return cm, acc
