"""
GrowthMind dynamics visualizer with insight overlays.
Displays Φ oscillations, entropy trends, adaptive thresholds,
and evolving policy trajectories with stability–adaptation insight.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
JOURNAL_PATH = BASE_DIR / "brain" / "growth_journal.jsonl"
NORMALIZE_MODE = "relative"  # "relative" or "absolute"
OUT_PATH = BASE_DIR / "analysis" / "growth_dynamics.png"

RULES = ["stabilize", "merge", "branch", "revert"]


# -------------------------------------------------------------------
# Load journal
# -------------------------------------------------------------------
def load_journal(path: Path):
    """Load GrowthMind JSONL journal into a list of records."""
    if not path.exists():
        raise FileNotFoundError(f"No journal found at {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"📘 Loaded {len(records)} entries from {path}")
    return records


# -------------------------------------------------------------------
# Extract statistics (Φ, entropy, π, thresholds, etc.)
# -------------------------------------------------------------------
def extract_stats(records):
    steps, phi, entropy, temp = [], [], [], []
    stabilize, merge, branch, revert = [], [], [], []
    merge_thresh, branch_thresh = [], []

    current_pi = {k: 0.25 for k in RULES}
    found_initial_pi = False

    for r in records:
        if "π" in r:
            current_pi = r["π"]
            found_initial_pi = True
            break
        elif "Δπ" in r and "Φ" in r:
            current_pi = {k: 0.0 for k in RULES}
            break

    if not found_initial_pi and not any("Δπ" in r for r in records):
        current_pi = {k: 0.0 for k in RULES}

    for i, r in enumerate(records):
        if "Φ" not in r:
            continue

        steps.append(i)
        phi.append(r["Φ"])
        entropy.append(r.get("entropy", np.nan))
        temp.append(r.get("temperature", np.nan))

        # --- Threshold tracking ---
        th = r.get("thresholds", {})
        merge_thresh.append(th.get("merge_min_phi", np.nan))
        branch_thresh.append(th.get("branch_max_phi", np.nan))

        # --- Policy update ---
        if "Δπ" in r:
            for k, v in r["Δπ"].items():
                current_pi[k] = current_pi.get(k, 0.0) + v
        elif "π" in r:
            current_pi = r["π"]

        # Decay and normalize
        for k in current_pi:
            current_pi[k] *= 0.985
        total = sum(current_pi.values())
        if total != 0:
            for k in current_pi:
                current_pi[k] /= total

        stabilize.append(current_pi.get("stabilize", 0.0))
        merge.append(current_pi.get("merge", 0.0))
        branch.append(current_pi.get("branch", 0.0))
        revert.append(current_pi.get("revert", 0.0))

    return {
        "steps": np.array(steps),
        "phi": np.array(phi),
        "entropy": np.array(entropy),
        "temperature": np.array(temp),
        "stabilize": np.array(stabilize),
        "merge": np.array(merge),
        "branch": np.array(branch),
        "revert": np.array(revert),
        "merge_thresh": np.array(merge_thresh),
        "branch_thresh": np.array(branch_thresh),
    }


# -------------------------------------------------------------------
# Normalize for better visualization
# -------------------------------------------------------------------
def normalize_policy(stats, mode="relative"):
    """Normalize Q-policy trajectories for plotting clarity."""
    if mode == "absolute":
        all_vals = np.vstack([stats[k] for k in RULES])
        max_val = np.max(np.abs(all_vals)) or 1.0
        for k in RULES:
            stats[k] = stats[k] / max_val
    else:
        for k in RULES:
            vals = stats[k]
            min_val, max_val = np.min(vals), np.max(vals)
            stats[k] = (vals - min_val) / (max_val - min_val + 1e-9)
    return stats


# -------------------------------------------------------------------
# Plot GrowthMind dynamics
# -------------------------------------------------------------------
def plot_dynamics(s, records=None):
    """Generate multi-panel visualization of GrowthMind internal dynamics."""
    s = normalize_policy(s, NORMALIZE_MODE)
    steps = s["steps"]

    # Adaptive smoothing based on log length
    sigma = max(5, len(steps) // 200)
    φ_smooth = gaussian_filter1d(s["phi"], sigma=sigma)
    entropy_smooth = gaussian_filter1d(np.nan_to_num(s["entropy"]), sigma=sigma)
    temp_smooth = gaussian_filter1d(np.nan_to_num(s["temperature"]), sigma=sigma)

    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

    # --- Φ Dynamics + Adaptive Thresholds ---
    axes[0].plot(steps, φ_smooth, color="teal", label="Φ (smoothed)", linewidth=1.6)
    axes[0].fill_between(steps, φ_smooth - 0.1, φ_smooth + 0.1, color="teal", alpha=0.15)
    axes[0].plot(steps, s["merge_thresh"], "--", color="blue", alpha=0.6, label="merge_thresh")
    axes[0].plot(steps, s["branch_thresh"], "--", color="red", alpha=0.6, label="branch_thresh")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Φ (polarity)")
    axes[0].set_title("Φ Dynamics and Adaptive Thresholds")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Entropy and Temperature ---
    axes[1].plot(steps, entropy_smooth, color="purple", label="Entropy", linewidth=1.4)
    ax2 = axes[1].twinx()
    ax2.plot(steps, temp_smooth, color="orange", alpha=0.6, label="Temperature")
    axes[1].axhline(1.5, color="gray", ls=":", lw=0.8)
    axes[1].set_ylabel("Entropy (bits)", color="purple")
    ax2.set_ylabel("Temperature", color="orange")
    axes[1].set_title("Entropy–Temperature Coupling")
    axes[1].grid(alpha=0.3)

    # --- Policy Evolution (stacked) ---
    axes[2].stackplot(
        steps,
        s["stabilize"], s["merge"], s["branch"], s["revert"],
        labels=RULES,
        colors=["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3"],
        alpha=0.85
    )
    axes[2].set_ylabel("Normalized Policy Weight")
    axes[2].legend(loc="upper left")
    axes[2].set_title(f"Policy Evolution ({NORMALIZE_MODE.capitalize()} trajectories)")
    axes[2].grid(alpha=0.3)

    # --- Stability vs Adaptation ---
    window = 200
    φ_var = np.array([np.var(s["phi"][max(0, i - window): i + 1]) for i in range(len(s["phi"]))])
    gap = s["merge_thresh"] - s["branch_thresh"]
    axes[3].plot(steps, φ_var, color="brown", label="Φ variance", linewidth=1.4)
    axes[3].plot(steps, entropy_smooth, color="purple", alpha=0.6, label="entropy", linewidth=1.2)
    axes[3].plot(steps, gap, color="gray", alpha=0.5, label="Decision Gap", linewidth=1.2)
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Variance / Entropy / Gap")
    axes[3].set_title("Stability vs Adaptation")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()

    # Ensure output directory exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"✅ Saved visualization to {OUT_PATH}")
    plt.show()


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
def main():
    recs = load_journal(JOURNAL_PATH)
    stats = extract_stats(recs)
    plot_dynamics(stats, records=recs)


if __name__ == "__main__":
    main()
