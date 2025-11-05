# growth/main.py
"""
Main script for SNLI baseline + GrowthMind learning.

Orchestrates data loading, calibration, training, and testing.
"""
from pathlib import Path
from growth.config import CONFIG
from growth.utils import set_seed
from growth.data_loader import load_snli, balance_dataset
from growth.lattice_encoding import _get_embedder
from growth.calibration import calibrate_phi
from growth.evaluation import evaluate_with_mind
from growth.mind.growth_mind import GrowthMind


def main():
    # --- Reproducibility ---
    set_seed(CONFIG["seed"])

    # --- Paths ---
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"

    # --- Load SNLI data ---
    train_raw = load_snli(DATA_DIR / "snli_1.0_train.jsonl", CONFIG["snli_train_limit"])
    test = load_snli(DATA_DIR / "snli_1.0_test.jsonl", CONFIG["snli_test_limit"])
    print(f"🌱 Loaded {len(train_raw)} train / {len(test)} test")

    # --- Balance the training dataset ---
    train = balance_dataset(train_raw)

    # --- Initialize embedder early ---
    if CONFIG["use_embeddings"]:
        print("Initializing embedding model...")
        _ = _get_embedder()
        print("Embedding model ready.")

    # --- Calibrate Φ field on a subset ---
    calib = calibrate_phi(train)
    phi_sign, band = calib["phi_sign"], calib["neutral_band"]

    # --- Initialize GrowthMind ---
    mind = GrowthMind.from_config({
        "temperature": 0.10,
        "phi_damping": 0.90,
        "branch_var_threshold": 0.02,
        "neutral_band": band,  # Pass calibrated band
    })
    print(f"🧠 GrowthMind initialized | neutral_band={band:.3f} | φ_damping={mind.phi_damping}")

    # --- Split calibration and training phase ---
    hstart = max(100, int(len(train) * CONFIG["calib_frac"]))
    train_phase_data = train[hstart:]

    # --- Training phase ---
    # We capture the confusion matrix (cm) AND the final accuracy (acc)
    cm_train, acc_train = evaluate_with_mind(
        train_phase_data, phi_sign, band, "TRAIN", mind, is_training=True
    )

    # --- Meta-Cognition phase (Step 7) ---
    # The mind now reflects on its own training performance...
    mind.metacog_reflect_and_tune(cm_train, acc_train)
    # ...and will use its new, tuned 'neutral_band' for the TEST phase.

    # --- Testing phase (frozen learning) ---
    # We now pass 'mind.neutral_band', which may have been changed by Step 7.
    evaluate_with_mind(
        test, phi_sign, mind.neutral_band, "TEST", mind, is_training=False
    )

    # --- Reflective log and save state ---
    print("\n" + "=" * 20 + " FINAL REFLECTION " + "=" * 20)
    try:
        mind.reflect()
    except AttributeError:
        pass

    mind.save_state()
    print("=" * 58)


if __name__ == "__main__":
    main()