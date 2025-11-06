"""
Main script for SNLI + GrowthMind learning.
REFACTORED to use the new decoupled Tree-of-Thought architecture
with an external SearchOrchestrator.
"""
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import sys  # For clean system exit/flush

# --- New Architectural Imports ---
from growth.mind.growth_mind import GrowthMind
from growth.mind.growth_tree import GrowthTree
from growth.mind.search_orchestrator import SearchOrchestrator
# ---------------------------------

# CRITICAL FIX: Ensure TENSORBOARD_WRITER is imported here
from growth.mind.reward import compute_total_reward, RewardParams, TENSORBOARD_WRITER

from growth.config import CONFIG
from growth.utils import set_seed
from growth.data_loader import load_snli, balance_dataset
from growth.lattice_encoding import _get_embedder
from growth.calibration import calibrate_phi

# --- Core Imports ---
from core.semantic import IntentVector
from growth.phi_computer import phi_raw_for_pair


# -------------------------------------------------------------------
# Helper: timestamped print
# -------------------------------------------------------------------
def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def initialize_environment():
    set_seed(CONFIG["seed"])
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    log("🧩 Environment initialized (random seed fixed).")
    return data_dir


def load_and_prepare_data(data_dir):
    log("📦 Loading SNLI dataset...")
    train_raw = load_snli(data_dir / "snli_1.0_train.jsonl", CONFIG["snli_train_limit"])
    test = load_snli(data_dir / "snli_1.0_test.jsonl", CONFIG["snli_test_limit"])
    log(f"🌱 Dataset loaded → Train: {len(train_raw):,} | Test: {len(test):,}")

    train = balance_dataset(train_raw)
    log(f"⚖️  Balanced training set → {len(train):,} samples (equalized per class).")
    return train, test


def initialize_embedder():
    if CONFIG["use_embeddings"]:
        log("🧠 Initializing embedding model...")
        _ = _get_embedder()
        log("✅ Embedding model ready.")


def calibrate_mind(train):
    log("🔧 Running Φ calibration on sample subset...")
    calib = calibrate_phi(train)
    phi_sign, band = calib["phi_sign"], calib["neutral_band"]
    log(f"✅ Calibration complete | sign={phi_sign:+.0f}, neutral_band={band:.3f}, acc={calib['acc']:.3f}")
    return phi_sign, band


def initialize_growthmind(neutral_band):
    mind = GrowthMind.from_config({
        "temperature": 0.25,
        "phi_damping": 0.90,  # aggressive exploitation
        "neutral_band": neutral_band,
    })
    log(f"🧩 GrowthMind Controller initialized.")
    log(f"    φ_damping={mind.phi_damping}, temperature={mind.temperature}, neutral_band={neutral_band:.3f}")
    return mind


# -------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------
def run_training(mind, train_data, phi_sign):
    log(f"🚀 Starting SearchOrchestrator training | Samples: {len(train_data):,}")
    correct = 0
    total_samples = len(train_data)
    cm_train = np.zeros((3, 3), dtype=int)

    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    rule_to_label = {
        "merge": "entailment",
        "branch": "contradiction",
        "stabilize": "neutral",
        "origin": "neutral",
        "revert": "neutral"
    }

    reward_params = RewardParams()

    for i, (s1, s2, gold_label) in enumerate(tqdm(train_data, desc="TRAIN")):
        tree = GrowthTree.create()
        orchestrator = SearchOrchestrator(mind, tree)

        phi_raw = phi_raw_for_pair(s1, s2)
        initial_intent = IntentVector(
            polarity=float(phi_raw * phi_sign),
            raw_polarity=float(phi_raw),
            delta_energy=0.0,
            rotation_seq="",
            observer="Om",
        )

        solution_node = orchestrator.run_search(
            initial_intent,
            note_prefix=f"sample_{i}"
        )

        pred_label = rule_to_label.get(solution_node.rule.split(":")[-1], "neutral")
        is_correct = (pred_label == gold_label)
        correct += int(is_correct)
        cm_train[labels[gold_label], labels[pred_label]] += 1

        total_path_reward = compute_total_reward(
            mind=mind,
            solution_node=solution_node,
            is_correct=is_correct,
            params=reward_params
        )

        # Policy Update
        curr = solution_node
        while curr and curr.parent is not None:
            rule_key = curr.rule.split(':')[-1]
            mind.policy.update(rule_key, total_path_reward)
            curr = curr.parent

        mind.update_temperature()

        # Periodic checkpoint logging
        if (i + 1) % 1000 == 0 or (i + 1) == total_samples:
            current_acc = correct / (i + 1)
            log(f"📍 Step {i + 1:,}/{total_samples:,} | Accuracy={current_acc:.3f}")
            try:
                if hasattr(orchestrator, 'node_counter'):
                    log(f"   🌳 Total nodes: {orchestrator.node_counter:,}")
            except Exception:
                pass

            try:
                desc = mind.policy.describe()
                log(f"   🧩 Policy snapshot → {desc}")
            except Exception:
                pass

    acc_train = correct / max(1, total_samples)
    log(f"📊 TRAIN Complete → Final Accuracy = {acc_train:.3f}")
    return cm_train, acc_train


# -------------------------------------------------------------------
# Finalization
# -------------------------------------------------------------------
def finalize_training(mind, cm_train, acc_train):
    log("🧠 Meta-cognition & reflection phase...")
    mind.metacog_reflect_and_tune(cm_train, acc_train)

    log("🧾 Skipping TEST phase (to be implemented later).")
    log("🪞 Running final reflection...")
    try:
        mind.reflect()
    except Exception:
        log("⚠️  Reflection failed (non-critical).")

    mind.save_state()
    log("💾 GrowthMind state saved.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    start = time.time()
    log("🌍 Initializing Livnium Growth training pipeline...")

    data_dir = initialize_environment()
    train, test = load_and_prepare_data(data_dir)
    initialize_embedder()

    phi_sign, band = calibrate_mind(train)
    mind = initialize_growthmind(band)

    hstart = max(100, int(len(train) * CONFIG["calib_frac"]))
    train_phase_data = train[hstart:]
    log(f"📘 Training subset selected → {len(train_phase_data):,} samples (post-calibration).")

    cm_train, acc_train = run_training(mind, train_phase_data, phi_sign)
    finalize_training(mind, cm_train, acc_train)

    # Safe TensorBoard closure
    try:
        log("💾 Flushing and closing TensorBoard writer.")
        TENSORBOARD_WRITER.close()
    except Exception as e:
        log(f"⚠️  TensorBoard writer close failed: {e}")

    elapsed = time.time() - start
    log(f"🏁 Total Runtime: {elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()
