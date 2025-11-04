"""
Livnium Core — lattice.py (Updated for Dual-Core Architecture)
---------------------------------------------------------------
Implements Axiom A1 (Spatial Alphabet), A2 (Observer Anchor),
and A3 (Symbolic Weight Law).

Defines the canonical 3×3×3 Livnium cube — the atomic, reversible
computational geometry on which all higher reasoning operates.

Each cell corresponds bijectively to one symbol in Σ = {0, a–z}.
Each cell’s symbolic weight is determined by its exposure class (A3):
    Centers (f=1) → SW = 9
    Edges   (f=2) → SW = 18
    Corners (f=3) → SW = 27

The cube obeys the global invariant ΣSW = 486 at all times.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from core.conservation import conserve_ledger  # <-- NEW IMPORT

# -------------------------------------------------------------------
# Canonical Constants
# -------------------------------------------------------------------

GRID_SIZE = 3
SYMBOLS = list("abcdefghijklmnopqrstuvwxyz0")  # 27 symbols → 3×3×3 bijection
TOTAL_LEDGER_TARGET = 486  # Conservation constant (ΣSW)
ANCHORS = {"Om": (0, 0, 0), "Lo": (1, 1, 1)}  # Observer anchor positions


# -------------------------------------------------------------------
# Utility: face exposure and symbolic weight
# -------------------------------------------------------------------

def face_exposure(x: int, y: int, z: int) -> int:
    """Return the number of exposed faces for a coordinate in the 3×3×3 cube."""
    exposure = 0
    for coord in (x, y, z):
        if coord == 0 or coord == GRID_SIZE - 1:
            exposure += 1
    return exposure


def symbolic_weight(faces: int) -> float:
    """
    Apply Axiom A3 (Symbolic Weight Law):
        SW = 9 × f, where f = exposed faces (0–3).
    """
    if faces == 0:
        return 0.0
    return 9.0 * faces


# -------------------------------------------------------------------
# Lattice State
# -------------------------------------------------------------------

@dataclass
class LatticeState:
    """
    Represents the complete 3×3×3 Livnium cube.
    Encodes both the symbolic (Σ) and geometric (SW) structure.

    Anchors:
        Om → (0,0,0) : Global Observer (absolute reference)
        Lo → (1,1,1) : Local Observer (contextual frame)
    """
    cells: np.ndarray = field(default_factory=lambda: np.empty((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=str))
    weights: np.ndarray = field(default_factory=lambda: np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE)))
    anchors: dict[str, tuple[int, int, int]] = field(default_factory=lambda: dict(ANCHORS))

    # ---------------------------------------------------------------
    # Core properties and operations
    # ---------------------------------------------------------------

    def clone(self) -> LatticeState:
        """Return a deep copy of the current state."""
        return LatticeState(
            cells=self.cells.copy(),
            weights=self.weights.copy(),
            anchors=self.anchors.copy(),
        )

    def total_sw(self) -> float:
        """Compute total Symbolic Weight of the cube."""
        return float(np.sum(self.weights))

    # @conserve_ledger
    def normalize(self) -> None:
        """Rescale weights to enforce ΣSW = 486 (Conservation Law)."""
        total = self.total_sw()
        if total == 0:
            return
        factor = TOTAL_LEDGER_TARGET / total
        self.weights *= factor

    # @conserve_ledger
    def rebalance(self) -> None:
        """Rebuild weights using the exposure rule (A3)."""
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    f = face_exposure(x, y, z)
                    self.weights[x, y, z] = symbolic_weight(f)
        self.normalize()

    # ---------------------------------------------------------------
    # Symbol utilities
    # ---------------------------------------------------------------

    def get_symbol(self, x: int, y: int, z: int) -> str:
        return self.cells[x, y, z]

    def set_symbol(self, x: int, y: int, z: int, value: str) -> None:
        self.cells[x, y, z] = value

    # ---------------------------------------------------------------
    # Validation and integrity
    # ---------------------------------------------------------------

    def is_bijective(self) -> bool:
        """Check if all 27 symbols are unique (1:1 mapping)."""
        uniques, counts = np.unique(self.cells, return_counts=True)
        return len(uniques) == 27 and all(c == 1 for c in counts)

    def verify(self) -> bool:
        """Verify conservation and bijectivity."""
        return abs(self.total_sw() - TOTAL_LEDGER_TARGET) < 1e-6 and self.is_bijective()

    # ---------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        layout = "\n".join([f"Layer {z}:\n{self.cells[:, :, z]}" for z in range(GRID_SIZE)])
        om = self.anchors.get("Om", None)
        lo = self.anchors.get("Lo", None)
        return (
            f"<LatticeState ΣSW={self.total_sw():.2f} conserved={self.verify()}>\n"
            f"Anchors: Om={om}, Lo={lo}\n"
            f"{layout}"
        )


# -------------------------------------------------------------------
# Canonical construction
# -------------------------------------------------------------------

def canonical_symbol_layout() -> LatticeState:
    """Generate the canonical 3×3×3 Livnium lattice with bijective Σ mapping."""
    s = LatticeState()
    symbols_iter = iter(SYMBOLS)
    for z in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                s.cells[x, y, z] = next(symbols_iter)
                f = face_exposure(x, y, z)
                s.weights[x, y, z] = symbolic_weight(f)
    s.normalize()
    return s


def identity_state() -> LatticeState:
    """
    Return the canonical identity state (I-lattice).
    This is the unrotated, conserved base configuration of the system.
    """
    return canonical_symbol_layout().clone()


@conserve_ledger
def rebalance(state: LatticeState) -> None:
    """Convenience helper to rebalance a lattice state according to A3."""
    state.rebalance()


# -------------------------------------------------------------------
# Self-check
# -------------------------------------------------------------------

if __name__ == "__main__":
    lattice = canonical_symbol_layout()
    assert lattice.is_bijective(), "Lattice is not bijective!"
    assert abs(lattice.total_sw() - TOTAL_LEDGER_TARGET) < 1e-6, "Conservation error!"
    print("Anchors:", lattice.anchors)
    print("ΣSW =", lattice.total_sw())
    print("lattice.py canonical self-check passed ✓")
