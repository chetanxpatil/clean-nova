"""
Main script for SNLI + GrowthMind learning.
REFACTORED to use the new decoupled Tree-of-Thought architecture
with an external SearchOrchestrator.
"""
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- New Architectural Imports ---
from growth.mind.growth_mind import GrowthMind
from growth.mind.growth_tree import GrowthTree
from growth.mind.search_orchestrator import SearchOrchestrator
# --- NEW: Import the reward shaping logic ---
from growth.mind.reward import RewardParams, compute_total_reward
# ---------------------------------

from growth.config import CONFIG
from growth.utils import set_seed
from growth.data_loader import load_snli, balance_dataset
from growth.lattice_encoding import _get_embedder
from growth.calibration import calibrate_phi

# --- Core Imports for new loop ---
from core.semantic import IntentVector
from growth.phi_computer import phi_raw_for_pair

# --- NEW: Q-Learning Hyperparameters ---
LEARNING_RATE = 0.05
# -------------------------------------


def main():
    # --- Reproducibility & Paths (Unchanged) ---
    set_seed(CONFIG["seed"])
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"

    # --- Load SNLI data (Unchanged) ---
    train_raw = load_snli(DATA_DIR / "snli_1.0_train.jsonl", CONFIG["snli_train_limit"])
    test = load_snli(DATA_DIR / "snli_1.0_test.jsonl", CONFIG["snli_test_limit"])
    print(f"🌱 Loaded {len(train_raw)} train / {len(test)} test")
    train = balance_dataset(train_raw)

    # --- Initialize embedder (Unchanged) ---
    if CONFIG["use_embeddings"]:
        print("Initializing embedding model...")
        _ = _get_embedder()
        print("Embedding model ready.")

    # --- Calibrate (Unchanged) ---
    calib = calibrate_phi(train)
    phi_sign, band = calib["phi_sign"], calib["neutral_band"]

    # --- Initialize NEW Architecture ---

    # 1. Create the Mind (Policy/Memory Controller)
    mind = GrowthMind.from_config({
        "temperature": 0.25,
        "phi_damping": 0.95,
        "neutral_band": band,
    })

    # 2. Create the Reward object
    reward_params = RewardParams()

    print(f"🧠 GrowthMind Controller initialized | neutral_band={band:.3f} | φ_damping={mind.phi_damping}")

    # --- Split calibration and training phase (Unchanged) ---
    hstart = max(100, int(len(train) * CONFIG["calib_frac"]))
    train_phase_data = train[hstart:]

    # --- NEW Training Phase (Replaces evaluate_with_mind) ---
    print(f"🚀 Starting new Search Orchestrator training on {len(train_phase_data)} samples...")
    correct = 0
    total_samples = len(train_phase_data)
    cm_train = np.zeros((3, 3), dtype=int)
    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    RULE_TO_LABEL = {
        "merge": "entailment",
        "branch": "contradiction",
        "stabilize": "neutral",
        "origin": "neutral",
        "revert": "neutral"
    }

    for i, (s1, s2, gold_label) in enumerate(tqdm(train_phase_data, desc="TRAIN")):

        # 1. Create fresh Tree and Orchestrator
        tree = GrowthTree.create()
        orchestrator = SearchOrchestrator(mind, tree)

        # 2. Compute initial Intent
        phi_raw = phi_raw_for_pair(s1, s2)
        initial_intent = IntentVector(
            polarity=float(phi_raw * phi_sign),
            raw_polarity=float(phi_raw),
            delta_energy=0.0, rotation_seq="", observer="Om"
        )

        # 3. Run the "blind" search
        solution_node = orchestrator.run_search(
            initial_intent,
            note_prefix=f"sample_{i}"
        )

        # 4. Get the final prediction
        pred_label = RULE_TO_LABEL.get(solution_node.rule.split(':')[-1], "neutral")

        # 5. Book-keeping
        is_correct = (pred_label == gold_label)
        if is_correct:
            correct += 1
        cm_train[labels[gold_label], labels[pred_label]] += 1

        # --- 6. Policy Update (CRITICAL FIX: Incremental Q-Learning) ---

        # Calculate the rich, shaped reward
        total_reward = compute_total_reward( # <-- USING THE SHAPED REWARD
            mind=mind,
            solution_node=solution_node,
            is_correct=is_correct,
            params=reward_params
        )

        # Apply this reward to the whole path using incremental Q-learning
        curr = solution_node
        while curr and curr.parent is not None:
            rule_key = curr.rule.split(':')[-1]

            current_q = mind.policy.Q.get(rule_key, 0.0)

            # Temporal Difference Error: (Reward - Current Q)
            td_error = total_reward - current_q

            # Incremental Update: Q = Q + alpha * TD_Error
            new_q = current_q + LEARNING_RATE * td_error

            mind.policy.Q[rule_key] = new_q # Directly update the dictionary

            curr = curr.parent

        # --- 7. Update Mind Metabolism ---
        mind.update_temperature()

        # --- 8. Print progress ---
        if (i + 1) % 1000 == 0 and i > 0:
            current_acc = correct / (i + 1)
            print(f"\n📍 Step {i + 1} / {total_samples} ({ (i + 1) / total_samples :.1%} complete)")
            print(f"   Current Accuracy: {current_acc:.3f}")
            try:
                print(f"   Policy Snapshot → {mind.policy.describe()}")
            except Exception as e:
                print(f"   (Could not describe policy: {e})")

    acc_train = correct / max(1, total_samples)
    print(f"📊 TRAIN Complete: Accuracy = {acc_train:.3f}")

    # --- Meta-Cognition phase (Unchanged) ---
    mind.metacog_reflect_and_tune(cm_train, acc_train)

    # --- Testing phase (Skipped) ---
    print("Skipping TEST phase (requires separate non-training search loop).")

    # --- Reflective log and save state (Unchanged) ---
    print("\n" + "=" * 20 + " FINAL REFLECTION " + "=" * 20)
    try:
        mind.reflect()
    except Exception:
        pass

    mind.save_state()
    print("=" * 58)


if __name__ == "__main__":
    main()