# grow.py
import sys
import json
from pathlib import Path

# --- Import Core Components ---
# NOTE: We need to import the simple evaluator and language module
from growth.evaluation import evaluate_with_mind
from growth.lattice_encoding import _get_embedder
from growth.phi_computer import phi_raw_for_pair
from growth.mind.growth_mind import GrowthMind
from growth.config import CONFIG
from growth.language import express_decision  # The mind's voice
import time


# ------------------------------------------------------------
# 1. Setup (Same as main.py, but faster)
# ------------------------------------------------------------
def setup_mind():
    """Initializes the embedder and a fresh GrowthMind instance."""
    # Initialize embedder early if needed (this can take a few seconds)
    if CONFIG["use_embeddings"]:
        print("🧠 Initializing Semantic Engine...")
        _get_embedder()

    # Load calibrated band (we assume 0.45, or read from saved state if possible)
    # Since we are just testing, we can hardcode the phi_sign and neutral_band
    # from our successful run for simplicity, or grab them from a saved file.

    # Using the fixed values from your successful run:
    phi_sign = 1.0
    band = 0.45

    mind = GrowthMind.from_config({
        "temperature": 0.10,
        "phi_damping": 0.90,
        "branch_var_threshold": 0.02,
        "neutral_band": band,
    })
    return mind, phi_sign, band


# ------------------------------------------------------------
# 2. Interactive Loop
# ------------------------------------------------------------
def interactive_run(mind, phi_sign, band):
    """Runs the main premise/hypothesis interaction loop."""
    print("\n--- Livnium Console Ready ---")
    print("Goal: Determine logical relationship (Entailment/Contradiction/Neutral).")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            premise = input("PREMISE > ").strip()
            if premise.lower() == 'exit':
                break

            hypothesis = input("HYPOTHESIS > ").strip()
            if hypothesis.lower() == 'exit':
                break

            if not premise or not hypothesis:
                continue

            # 1. Compute the core Phi signal
            phi_raw = phi_raw_for_pair(premise, hypothesis)
            phi = mind.phi_sign * phi_raw
            time.sleep(1.5)  # Pause for 1.5 seconds (you can change this value)

            # 2. Get the mind's decision rule (The 53% core logic)
            rule = mind.choose_rule(phi)
            print("🧠 Processing thought...")
            # 3. Get the mind's final expression (Step 6)
            expression = express_decision(rule, phi)

            print("\n" + "=" * 50)
            print(" Mind's Decision: " + expression)
            print("=" * 50 + "\n")

        except Exception as e:
            print(f"\n[ERROR] An internal error occurred: {e}")
            break


if __name__ == "__main__":
    try:
        mind, phi_sign, band = setup_mind()
        interactive_run(mind, phi_sign, band)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Required data not found. {e}")
    except Exception as e:
        print(f"FATAL ERROR: {e}")