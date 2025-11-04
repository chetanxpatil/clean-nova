# growth/utils.py
"""
General utilities: seeding and lattice visualization.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from core.lattice import LatticeState

# --- Determinism ---
def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# --- Diagnostic Visualization (Lattice Probe) ---
def visualize_lattice(lattice: LatticeState, title="Lattice Probe"):
    """
    Visual diagnostic for LatticeState.
    Works for both single-channel and multi-channel embeddings.
    """
    W = lattice.weights
    flat = W.flatten()
    normalized = (flat - np.min(flat)) / (np.max(flat) - np.min(flat) + 1e-8)

    print(f"\n🧩 {title}")
    print(f"  ΣSW = {lattice.total_sw():.2f}")
    print(f"  min={np.min(W):.3f}, max={np.max(W):.3f}, mean={np.mean(W):.3f}, std={np.std(W):.3f}")
    print(f"  bijective={lattice.is_bijective()} ({len(np.unique(lattice.cells))} unique symbols)")
    print(f"  Sample middle slice (z=1):\n{np.round(W[:, :, 1], 2)}")

    # --- Multi-slice visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        im = axes[i].imshow(W[:, :, i], cmap="plasma")
        axes[i].set_title(f"Slice z={i}")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- Optional 3D-style projection ---
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(np.arange(3), np.arange(3), np.arange(3))
    ax.scatter(X, Y, Z, c=W.flatten(), cmap="plasma", s=100 + 200 * normalized)
    ax.set_title("3D Lattice Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()