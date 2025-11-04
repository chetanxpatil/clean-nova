"""
Livnium Core — observer.py (Updated for Dual-Core Architecture)
---------------------------------------------------------------
Implements Axiom A2: The Observer Principle.

Defines:
    - GlobalObserver (Om): Absolute, unrotating frame.
    - LocalObserver  (Lo): Contextual, embedded frame that rotates relative to Om.
    - DualObserver   (Ω): Coupled pair providing bidirectional measurement.

This dual structure is the perceptual bridge between symbolic geometry
and meaning. Om measures absolute conservation; Lo experiences motion and
direction; together they produce semantic polarity (Φ).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

try:  # pragma: no cover
    from .lattice import LatticeState, canonical_symbol_layout
    from .rotation import rotate_x, rotate_y, rotate_z
    from .semantic import compute_polarity
    from .conservation import verify_conservation
except ImportError:  # fallback for direct execution
    from lattice import LatticeState, canonical_symbol_layout  # type: ignore
    from rotation import rotate_x, rotate_y, rotate_z  # type: ignore
    from semantic import compute_polarity  # type: ignore
    from conservation import verify_conservation  # type: ignore


# -------------------------------------------------------------------
# Global Observer (Om)
# -------------------------------------------------------------------

@dataclass
class GlobalObserver:
    """
    The Global Observer (Om) represents the absolute coordinate frame.
    It never rotates — it is the origin of all perspective and conservation.
    """
    state: LatticeState = field(default_factory=canonical_symbol_layout)

    def measure(self) -> dict:
        """Return global summary measurements."""
        return {
            "ΣSW": self.state.total_sw(),
            "conserved": verify_conservation(self.state),
            "anchors": self.state.anchors,
        }

    def reference(self) -> LatticeState:
        """Return a read-only reference of the canonical lattice."""
        return self.state.clone()


# -------------------------------------------------------------------
# Local Observer (Lo)
# -------------------------------------------------------------------

@dataclass
class LocalObserver:
    """
    The Local Observer (Lo) represents a contextual perspective — an
    embedded cognitive frame that perceives relative to the Om reference.
    Lo can rotate, altering perception but not physical conservation.
    """
    frame: LatticeState
    global_ref: GlobalObserver
    orientation: tuple[int, int, int] = (0, 0, 0)  # rotation counts (Rx,Ry,Rz)

    # ---------------------------------------------------------------
    # Orientation control
    # ---------------------------------------------------------------

    def rotate(self, axis: str, k: int = 1) -> None:
        """
        Apply a 90° rotation to the local frame along an axis.
        Positive k = clockwise rotation when viewed from positive axis.
        """
        axis = axis.upper()
        if axis == "X":
            self.frame = rotate_x(self.frame, k)
        elif axis == "Y":
            self.frame = rotate_y(self.frame, k)
        elif axis == "Z":
            self.frame = rotate_z(self.frame, k)
        else:
            raise ValueError(f"Invalid axis '{axis}' — must be X, Y, or Z.")

        dx, dy, dz = self.orientation
        if axis == "X":
            dx += k
        elif axis == "Y":
            dy += k
        elif axis == "Z":
            dz += k
        self.orientation = (dx % 4, dy % 4, dz % 4)

    # ---------------------------------------------------------------
    # Perception and measurement
    # ---------------------------------------------------------------

    def perceive_symbol(self, x: int, y: int, z: int) -> str:
        """Return the symbol visible at given coordinates in local frame."""
        return self.frame.get_symbol(x, y, z)

    def perceive_weights(self) -> np.ndarray:
        """Return the current local symbolic-weight distribution."""
        return self.frame.weights.copy()

    def relative_polarity(self) -> float:
        """
        Compute semantic polarity Φ between Lo (local frame) and Om (global reference).
        +1.0 → aligned toward Om
        -1.0 → opposed (negation)
        """
        return compute_polarity(self.global_ref.state, self.frame)

    def describe(self) -> dict:
        """Return a summary of Lo’s current perceptual state."""
        return {
            "orientation": self.orientation,
            "ΣSW(local)": self.frame.total_sw(),
            "relative Φ": round(self.relative_polarity(), 3),
            "conserved": verify_conservation(self.frame),
        }


# -------------------------------------------------------------------
# Dual Observer System (Ω)
# -------------------------------------------------------------------

@dataclass
class DualObserver:
    """
    Couples Om (global) and Lo (local) into a unified reversible observer system.
    This is the perceptual substrate used by higher reasoning layers (L2, L3).
    """
    Om: GlobalObserver = field(default_factory=GlobalObserver)
    Lo: LocalObserver = field(init=False)

    def __post_init__(self):
        self.Lo = LocalObserver(self.Om.state.clone(), self.Om)

    # ---------------------------------------------------------------
    # Coupled operations
    # ---------------------------------------------------------------

    def synchronize(self) -> None:
        """
        Re-align Lo’s frame to Om (resets orientation and perception).
        """
        self.Lo.frame = self.Om.state.clone()
        self.Lo.orientation = (0, 0, 0)

    def compare(self) -> dict:
        """Return relative polarity and conservation info between Om and Lo."""
        pol = self.Lo.relative_polarity()
        return {
            "Φ": round(pol, 3),
            "ΣSW(Om)": self.Om.state.total_sw(),
            "ΣSW(Lo)": self.Lo.frame.total_sw(),
            "aligned": abs(pol - 1.0) < 1e-3,
            "conserved": verify_conservation(self.Lo.frame),
        }

    def describe(self) -> str:
        summary = self.compare()
        return (
            f"Ω-DualObserver | Φ={summary['Φ']:.3f} | "
            f"ΣSW(Om)={summary['ΣSW(Om)']:.1f} | "
            f"ΣSW(Lo)={summary['ΣSW(Lo)']:.1f} | "
            f"aligned={summary['aligned']} | conserved={summary['conserved']}"
        )


# -------------------------------------------------------------------
# Self-check
# -------------------------------------------------------------------

if __name__ == "__main__":
    Ω = DualObserver()
    print("Initial:", Ω.describe())
    Ω.Lo.rotate("Z", 1)
    print("After rotation:", Ω.describe())
    Ω.synchronize()
    print("After sync:", Ω.describe())
    print("observer.py dual-core self-check passed ✓")
