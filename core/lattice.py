

from __future__ import annotations

from types import NoneType

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

from core.conservation import conserve_ledger  # Assuming this is available

# -------------------------------------------------------------------
# Canonical Constants
# -------------------------------------------------------------------

# ⚠️ --- THE SINGLE NUMBER TO CHANGE (N) ---
GLOBAL_GRID_SIZE = 3  # Set N here (Must be an odd integer >= 3). We set N=5 for the test.
# -----------------------------------------

GRID_SIZE = GLOBAL_GRID_SIZE
TOTAL_CELLS = GRID_SIZE ** 3


# --- DERIVED CONSTANTS (Automated Scaling) ---

def calculate_symbols(n: int) -> List[str]:
    """Generates an alphabet of N³ unique symbols (A1)."""
    return [str(i) for i in range(n ** 3)]


def calculate_target_sw(n: int) -> float:
    """
    Calculates Total Symbolic Weight (ΣSW) based on the generalized formula (D3).
    """
    if n % 2 == 0 or n < 3:
        raise ValueError("GRID_SIZE must be an odd integer >= 3.")

    N_minus_2 = n - 2

    # ΣSW(N) = 9 * [1*Centers + 2*Edges + 3*Corners]
    total_exposed_weight = (
            1 * 6 * (N_minus_2 ** 2) +  # Centers count (f=1)
            2 * 12 * N_minus_2 +  # Edges count (f=2)
            3 * 8  # Corners count (f=3)
    )
    return 9.0 * total_exposed_weight


SYMBOLS = calculate_symbols(GRID_SIZE)
TOTAL_LEDGER_TARGET = calculate_target_sw(GRID_SIZE)
ANCHOR_COORD = GRID_SIZE // 2
ANCHORS = {"Om": (ANCHOR_COORD, ANCHOR_COORD, ANCHOR_COORD),
           "Lo": (ANCHOR_COORD, ANCHOR_COORD, ANCHOR_COORD)}


# -------------------------------------------------------------------
# Utility: face exposure and symbolic weight
# -------------------------------------------------------------------

def face_exposure(x: int, y: int, z: int) -> int:
    """Return the number of exposed faces (0-3) for a coordinate in the N×N×N cube."""

    exposure = 0
    if x == 0 or x == GRID_SIZE - 1:
        exposure += 1
    if y == 0 or y == GRID_SIZE - 1:
        exposure += 1
    if z == 0 or z == GRID_SIZE - 1:
        exposure += 1

    # Crucial check for N > 3: Interior cells (not on the boundary) must have f=0.
    if GRID_SIZE > 3 and exposure == 0:
        return 0

    return exposure


def symbolic_weight(faces: int) -> float:
    """
    Apply Axiom A3 (Symbolic Weight Law): SW = 9 × f, where f = exposed faces (0–3).
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
    Represents the complete N×N×N Livnium cube.
    """
    # FIX: Remove default_factory lambdas to fix initialization size bug
    cells: np.ndarray = field(default=None)
    weights: np.ndarray = field(default=None)
    anchors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: dict(ANCHORS))

    # ---------------------------------------------------------------
    # Core properties and operations
    # ---------------------------------------------------------------

    def clone(self) -> "LatticeState":
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
        """Rescale weights to enforce ΣSW = TOTAL_LEDGER_TARGET (Conservation Law)."""
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

    def get_symbol(self, x: int, y: int, z: int) -> type[NoneType[Any, Any, Any]]:
        return self.cells[x, y, z]

    def set_symbol(self, x: int, y: int, z: int, value: str) -> None:
        self.cells[x, y, z] = value

    # ---------------------------------------------------------------
    # Validation and integrity
    # ---------------------------------------------------------------

    def is_bijective(self) -> bool:
        """Check if all N³ symbols are unique (1:1 mapping)."""
        uniques, counts = np.unique(self.cells, return_counts=True)
        return len(uniques) == TOTAL_CELLS and all(c == 1 for c in counts)

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
            f"<LatticeState N={GRID_SIZE} ΣSW={self.total_sw():.2f} conserved={self.verify()}>\n"
            f"Anchors: Om={om}, Lo={lo}\n"
            f"{layout}"
        )


# -------------------------------------------------------------------
# Canonical construction
# -------------------------------------------------------------------

def canonical_symbol_layout() -> LatticeState:
    """Generate the canonical N×N×N Livnium lattice with bijective Σ mapping."""

    # FIX: Initialize state with explicit size and data types for the current GRID_SIZE
    s = LatticeState(
        cells=np.empty((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=str),
        weights=np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE)),
    )

    symbols_iter = iter(SYMBOLS)

    for z in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                try:
                    s.cells[x, y, z] = next(symbols_iter)
                except StopIteration:
                    s.cells[x, y, z] = '!'

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

    print(f"\n--- Livnium N={GRID_SIZE} Self-Check ---")

    lattice = canonical_symbol_layout()

    # Check 1: Symbol Count
    assert len(np.unique(
        lattice.cells)) == TOTAL_CELLS, f"Symbol count error! Expected {TOTAL_CELLS}, got {len(np.unique(lattice.cells))}."

    # Check 2: Conservation
    target = TOTAL_LEDGER_TARGET
    current = lattice.total_sw()
    assert abs(current - target) < 1e-6, f"Conservation error! Target: {target:.2f}, Actual: {current:.2f}"

    print(f"Livnium N={GRID_SIZE} self-check passed ✓")
    print(f"Total Cells: {TOTAL_CELLS}")
    print(f"ΣSW Target: {TOTAL_LEDGER_TARGET:.2f}")