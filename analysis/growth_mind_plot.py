
# --- plot_growthmind_stats_enhanced_fixed.py ---
"""
GrowthMind dynamics visualizer with insight overlays.
Shows Φ oscillations, entropy trends, and independent policy trajectories.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Path to your GrowthMind journal
JOURNAL_PATH = "/Users/chetanpatil/Desktop/clean-nova/brain/growth_journal.jsonl"

# Visualization mode: "relative" (normalize each policy separately) or "absolute" (shared scale)
NORMALIZE_MODE = "relative"


# -------------------------------------------------------------------
# Load journal
# -------------------------------------------------------------------
def load_journal(path=JOURNAL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No journal found at {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"📘 Loaded {len(records)} entries.")
    return records

# -------------------------------------------------------------------
# Extract statistics (with decay and normalization)
# -------------------------------------------------------------------
def extract_stats(records):
    steps, phi, entropy = [], [], []
    stabilize, merge, branch, revert = [], [], [], []

    # Initialize pi
    current_pi = {"stabilize": 0.25, "merge": 0.25, "branch": 0.25, "revert": 0.25}

    # Robust initialization: Check if the first record provides a full 'π'
    found_initial_pi = False
    for r in records:
        if "π" in r:
            current_pi = r["π"]
            found_initial_pi = True
            break
        elif "Δπ" in r and "Φ" in r:
            # If we start with deltas, we need a baseline
            current_pi = {"stabilize": 0.0, "merge": 0.0, "branch": 0.0, "revert": 0.0}
            break

    if not found_initial_pi and not any("Δπ" in r for r in records):
        # Edge case: No policy info at all
        current_pi = {"stabilize": 0.0, "merge": 0.0, "branch": 0.0, "revert": 0.0}

    for i, r in enumerate(records):
        if "Φ" not in r:
            continue

        steps.append(i)
        phi.append(r["Φ"])
        entropy.append(r.get("entropy", np.nan))

        # --- 1. Apply update (accumulation or override) ---
        if "Δπ" in r:
            for k, v in r["Δπ"].items():
                # Ensure key exists if Δπ is sparse
                current_pi[k] = current_pi.get(k, 0.0) + v
        elif "π" in r:
            # Full state override
            current_pi = r["π"]

        # --- 2. Applied Fix: Temporal Decay ---
        # Optional: introduce mild temporal decay for smoother evolution
        # (Applied *after* the update, but *before* normalization)
        for k in current_pi:
            current_pi[k] *= 0.985  # decay old influence slightly
        # ----------------------------------------

        # --- 3. Applied Fix: Normalization ---
        # Normalize *after* applying the update and decay
        total = sum(current_pi.values())
        if total != 0:
            for k in current_pi:
                current_pi[k] /= total
        # ----------------------------------

        # --- 4. Store the post-processed values ---
        stabilize.append(current_pi.get("stabilize", 0.0))
        merge.append(current_pi.get("merge", 0.0))
        branch.append(current_pi.get("branch", 0.0))
        revert.append(current_pi.get("revert", 0.0))

    return {
        "steps": np.array(steps),
        "phi": np.array(phi),
        "entropy": np.array(entropy),
        "stabilize": np.array(stabilize),
        "merge": np.array(merge),
        "branch": np.array(branch),
        "revert": np.array(revert),
    }


# -------------------------------------------------------------------
# Normalize for better visualization
# -------------------------------------------------------------------
def normalize_policy(stats, mode="relative"):
    if mode == "absolute":
        # Shared normalization across all policies
        all_vals = np.vstack(
            [stats["stabilize"], stats["merge"], stats["branch"], stats["revert"]]
        )
        max_val = np.max(np.abs(all_vals)) or 1.0
        for k in ["stabilize", "merge", "branch", "revert"]:
            stats[k] = stats[k] / max_val
    else:
        # Normalize each policy individually (relative comparison)
        for k in ["stabilize", "merge", "branch", "revert"]:
            vals = stats[k]
            min_val, max_val = np.min(vals), np.max(vals)
            stats[k] = (vals - min_val) / (max_val - min_val + 1e-9)
    return stats


# -------------------------------------------------------------------
# Plot GrowthMind dynamics
# -------------------------------------------------------------------
def plot_dynamics(s):
    s = normalize_policy(s, NORMALIZE_MODE)
    steps = s["steps"]
    φ_smooth = gaussian_filter1d(s["phi"], sigma=20)
    entropy_smooth = gaussian_filter1d(np.nan_to_num(s["entropy"]), sigma=20)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # --- Φ Dynamics ---
    axes[0].plot(steps, φ_smooth, color="teal", label="Φ (smoothed)", linewidth=1.6)
    axes[0].fill_between(steps, φ_smooth - 0.1, φ_smooth + 0.1, color="teal", alpha=0.2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Φ")
    axes[0].set_title("GrowthMind Φ Dynamics (smoothed oscillation band)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Policy Entropy ---
    axes[1].plot(
        steps, entropy_smooth, color="purple", label="Policy Entropy", linewidth=1.4
    )
    axes[1].axhline(1.5, color="gray", ls=":", lw=0.8)
    axes[1].set_ylabel("Entropy (bits)")
    axes[1].set_title("Policy Entropy Trend")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # --- Policy Evolution (independent line trajectories) ---
    axes[2].plot(steps, s["stabilize"], color="green", label="stabilize", linewidth=1.5)
    axes[2].plot(steps, s["merge"], color="blue", label="merge", linewidth=1.5)
    axes[2].plot(steps, s["branch"], color="orange", label="branch", linewidth=1.5)
    axes[2].plot(steps, s["revert"], color="red", label="revert", linewidth=1.5)
    axes[2].set_facecolor("white")
    axes[2].grid(alpha=0.3, linestyle="--", linewidth=0.7)
    axes[2].legend(frameon=True, loc="upper left")
    axes[2].set_ylabel("Normalized Q-weight")
    axes[2].set_title(
        f"Policy Evolution ({NORMALIZE_MODE.capitalize()} trajectories)"
    )

    # --- Stability vs Adaptation ---
    window = 200
    φ_var = np.array(
        [np.var(s["phi"][max(0, i - window): i + 1]) for i in range(len(s["phi"]))]
    )
    axes[3].plot(steps, φ_var, color="brown", label="Φ variance", linewidth=1.4)
    axes[3].plot(
        steps, entropy_smooth, color="purple", alpha=0.6, label="entropy", linewidth=1.2
    )
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Variance / Entropy")
    axes[3].set_title("Stability vs Adaptation")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    recs = load_journal(JOURNAL_PATH)
    stats = extract_stats(recs)
    plot_dynamics(stats)

