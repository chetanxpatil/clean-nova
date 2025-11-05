"""
Contains the main evaluation loop for training and testing the GrowthMind.
"""
import numpy as np
from typing import List, Tuple
from core.semantic import IntentVector
from growth.mind.growth_mind import GrowthMind
from growth.phi_computer import phi_raw_for_pair
from growth.lattice_encoding import text_to_lattice
from growth.config import CONFIG
from growth.language import express_decision

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def evaluate_with_mind(pairs: List[Tuple[str, str, str]],
                       phi_sign: float,
                       neutral_band: float,
                       title: str,
                       mind: GrowthMind,
                       is_training: bool = True):
    """
    Runs an epoch of evaluation, optionally training the GrowthMind.
    """
    labels = ["entailment", "neutral", "contradiction"]
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((3, 3), dtype=int)
    phi_vals = []
    correct = 0

    # --- 1. Initialize Rule Counter ---
    rule_counts = {"merge": 0, "branch": 0, "stabilize": 0, "revert": 0}

    # --- CALIBRATE THE MIND ---
    mind.neutral_band = neutral_band
    mind.phi_sign = phi_sign

    print(f"\n🔍 Evaluating {title} on {len(pairs)} samples (Mind Neutral Band: {mind.neutral_band:.3f})...")

    it = tqdm(pairs, desc=title, dynamic_ncols=True) if (tqdm and CONFIG["progress"]) else pairs

    for i, (s1, s2, g) in enumerate(it, start=1):

        # 1. Get the raw signal
        phi_raw = phi_raw_for_pair(s1, s2)
        phi = mind.phi_sign * phi_raw
        phi_vals.append(phi)

        # 2. Ask the GrowthMind to choose a rule
        rule = mind.choose_rule(phi)

        # --- 2. Increment Rule Counter ---
        rule_counts[rule] += 1

        # 3. Map the MIND'S choice to a prediction
        if rule == "merge":
            pred = "entailment"
        elif rule == "branch":
            pred = "contradiction"
        else:  # "stabilize" or "revert"
            pred = "neutral"

        # 4. Check if the mind's prediction was correct
        is_correct = (pred == g)
        correct += is_correct
        cm[idx[g], idx[pred]] += 1

        # 5. Learn (update Q-values) if in "TRAIN" mode
        if is_training:
            intent = IntentVector(
                polarity=float(phi),
                raw_polarity=float(phi_raw),
                delta_energy=0.0,
                rotation_seq="",
                observer="Om"
            )

            # --- Integrate MemoryCoupling ---
            A = text_to_lattice(s1)
            B = text_to_lattice(s2)
            C = A.weights - B.weights  # Compute coupling

            # --- ================================== ---
            # ---        STEP 5.2 FEEDBACK FIX       ---
            # --- ================================== ---
            if g == "entailment":
                gold_rule = "merge"
            elif g == "contradiction":
                gold_rule = "branch"
            else:
                gold_rule = "stabilize"

            key = f"{gold_rule}_{pred}"
            # --- ================================== ---
            # ---            END OF FIX            ---
            # --- ================================== ---

            mind.note_coupling(key, C, phi, A.weights.shape, B.weights.shape)

            recall_key = f"{rule}_{g}"
            prior = mind.suggest_coupling(recall_key, A.weights.shape, B.weights.shape)

            if prior is not None:
                phi_bias = 0.05 * np.tanh(np.mean(prior))
                phi += phi_bias
                intent.polarity = float(phi)

                coupling_similarity = float(np.tanh(np.mean(np.abs(prior - C))))
                memory_bonus = 0.1 * (1.0 - coupling_similarity)
            else:
                memory_bonus = 0.0

            mind.step(intent, note=f"{g}->{pred}", external_correct=is_correct)
            mind.policy.update(rule, memory_bonus)

        # --- ================================== ---
        # ---   STEP 6: LANGUAGE EXPRESSION    ---
        # --- ================================== ---
        if not is_training and i <= 5:
            expression = express_decision(rule, phi)

            print("\n" + "-" * 20 + f" TEST EXAMPLE {i} " + "-" * 20)
            print(f"  Premise:    {s1}")
            print(f"  Hypothesis: {s2}")
            print(f"  Gold Label: {g} ({'Correct' if is_correct else 'INCORRECT'})")
            print(f"  Mind's Internal Choice: {rule}")
            print(f"  Mind's Expression: {expression}")
        # --- ================================== ---
        # ---          END OF STEP 6           ---
        # --- ================================== ---

        # 6. Print progress
        if i % 1000 == 0 and CONFIG["progress"]:
            progress_ratio = i / len(pairs)
            current_accuracy = correct / i
            phi_avg = np.mean(phi_vals)

            print(f"\n📍 Step {i:5d} / {len(pairs)} ({progress_ratio:.1%} complete)")
            print(f"   Accuracy: {current_accuracy:.3f} | Φ̄ = {phi_avg:+.3f}")
            print(f"   Policy Snapshot → {mind.policy.describe()}")

            try:
                mind.reflect()
            except AttributeError as e:
                print(f"   (No reflection available: {e})")

    # --- Final stats for the epoch ---
    dist = {l: int(sum(cm[:, i])) for i, l in enumerate(labels)}
    acc = correct / max(1, len(pairs))  # Avoid division by zero

    # --- 3. Calculate and Print Rule Distribution ---
    total_rules = max(1, len(pairs))
    rule_dist = {k: f"{(v / total_rules) * 100:.1f}%" for k, v in rule_counts.items()}

    print(f"\n📊 {title}: acc={acc:.3f}, Φ mean={np.mean(phi_vals):+.3f}, "
          f"range=[{min(phi_vals):+.3f},{max(phi_vals):+.3f}], dist={dist}")

    print(f"   Rule Choices: {rule_dist}")

    print("Confusion (Gold \\ Pred):")
    header = " " * 13 + " ".join([f"{l:<13}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, l in enumerate(labels):
        row = f"  {l:<13}" + " ".join([f"{cm[i, j]:<13}" for j in range(3)])
        print(row)

    # --- THIS IS THE FIX ---
    return cm, acc