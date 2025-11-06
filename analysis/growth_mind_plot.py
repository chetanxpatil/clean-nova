"""
GrowthMind dynamics visualizer (NEW ARCHITECTURE)
Displays Φ oscillations and the evolution of the
Policy Heuristic Scores (Q-values) from the search journal.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
JOURNAL_PATH = Path("/Users/chetanpatil/Desktop/clean-nova/brain/growth_journal.jsonl")
OUT_PATH = BASE_DIR / "analysis" / "growth_dynamics_new.png"


# -------------------------------------------------------------------
# Load and Extract
# -------------------------------------------------------------------
def load_and_extract_stats(path: Path):
    """Load the new SearchOrchestrator journal and extract key stats."""
    if not path.exists():
        raise FileNotFoundError(f"No journal found at {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)

                # Map to heuristic_score → fallback to f_score (used in A*)
                if "node_id" in entry:
                    entry["heuristic_score"] = entry.get("heuristic_score", entry.get("f_score", 0.0))

                    # Handle Φ (unicode key)
                    if "Φ" not in entry and "\u03a6" in entry:
                        entry["Φ"] = entry["\u03a6"]

                    records.append(entry)

            except json.JSONDecodeError:
                continue

            # ✅ Print every 1000 lines for progress visibility
            if i % 1000 == 0:
                print(f"🔄 Processed {i:,} journal lines...")

    print(f"📘 Loaded {len(records)} valid node entries from {path}")

    if not records:
        print("No valid node data found. Exiting.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Clean the rule name (e.g., "G1:merge" -> "merge")
    if "rule" in df.columns:
        df["rule"] = df["rule"].apply(lambda x: x.split(":")[-1] if isinstance(x, str) else x)
    else:
        df["rule"] = "unknown"

    # Extract data for plotting
    stats = {
        "steps": df.index.values,
        "phi": df["Φ"].values if "Φ" in df.columns else np.zeros(len(df)),
        "heuristic_score": df["heuristic_score"].values,
        "rules": df["rule"].values,
        "merge_scores": df[df["rule"] == "merge"]["heuristic_score"],
        "branch_scores": df[df["rule"] == "branch"]["heuristic_score"],
        "stabilize_scores": df[df["rule"] == "stabilize"]["heuristic_score"],
    }

    return stats


# -------------------------------------------------------------------
# Plot GrowthMind dynamics
# -------------------------------------------------------------------
def plot_dynamics(s):
    """Generate multi-panel visualization of the new architecture's dynamics."""

    steps = s["steps"]
    if len(steps) == 0:
        print("No data to plot.")
        return

    # Adaptive smoothing
    sigma = max(5, len(steps) // 200)

    # --- Layout using GridSpec ---
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0])

    plt.setp(ax0.get_xticklabels(), visible=False)

    # --- 1. Φ (Polarity) Dynamics ---
    phi_smooth = gaussian_filter1d(s["phi"], sigma=sigma)
    ax0.plot(steps, phi_smooth, color="teal", label="Φ (smoothed)", linewidth=1.6)
    ax0.fill_between(steps, phi_smooth - 0.1, phi_smooth + 0.1, color="teal", alpha=0.15)
    ax0.axhline(0, color="gray", ls="--", lw=0.8)
    ax0.set_ylabel("Φ (polarity)")
    ax0.set_title("Φ (Polarity) Dynamics & Policy Learning")
    ax0.legend()
    ax0.grid(alpha=0.3)

    # --- 2. Heuristic Score (Policy Q-Value) Evolution ---
    score_smooth = gaussian_filter1d(s["heuristic_score"], sigma=sigma)
    ax1.plot(steps, score_smooth, color="purple", label="Heuristic Score (Q-value)", linewidth=1.4)
    ax1.axhline(0, color="gray", ls=":", lw=0.8, label="Zero Reward")
    ax1.set_ylabel("Heuristic Score (Smoothed)")
    ax1.set_xlabel("Search Step")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- 3. Rule Score Distribution (Violin Plot) ---
    data_to_plot = [s["merge_scores"], s["branch_scores"], s["stabilize_scores"]]
    labels = ["Merge", "Branch", "Stabilize"]

    plot_data_filtered = [d for d in data_to_plot if len(d) > 0]
    plot_labels_filtered = [l for d, l in zip(data_to_plot, labels) if len(d) > 0]

    if plot_data_filtered:
        parts = ax2.violinplot(plot_data_filtered, showmeans=True, showmedians=False)
        ax2.set_xticks(np.arange(1, len(plot_labels_filtered) + 1))
        ax2.set_xticklabels(plot_labels_filtered)
        ax2.set_title("Policy Preference: Distribution of Heuristic Scores by Rule")
        ax2.set_ylabel("Heuristic Score (Q-Value)")
        ax2.axhline(0, color="gray", ls=":", lw=0.8)
        ax2.grid(alpha=0.3)

        colors = ["#8da0cb", "#fc8d62", "#66c2a5"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_edgecolor("black")
            pc.set_alpha(0.8)
    else:
        ax2.text(
            0.5,
            0.5,
            "No rule data to plot.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"✅ Saved new visualization to {OUT_PATH}")
    plt.show()


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
def main():
    try:
        stats = load_and_extract_stats(JOURNAL_PATH)
        if stats:
            plot_dynamics(stats)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
