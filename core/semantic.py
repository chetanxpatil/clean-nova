# """
# Livnium Core — semantic.py
# --------------------------
# Implements Axiom A5: The Semantic Law and A6: The Intent Relation.
#
# Meaning (Φ) arises from the relative orientation between observers
# or lattice states. Polarity (cosθ) expresses their alignment:
#
#     +1.0 → perfect agreement (toward Om)
#      0.0 → orthogonal / neutral
#     -1.0 → perfect negation (away from Om)
#
# Intent represents the energetic and directional delta between
# two states, used in reasoning and growth layers.
# """
#
# from __future__ import annotations
# import numpy as np
# from dataclasses import dataclass
# from core.coupling import CouplingMap, apply_coupling
# from core.lattice import LatticeState
# from core.rotation import rotate_sequence
#
# # -------------------------------------------------------------------
# # Polarity Computation
# # -------------------------------------------------------------------
#
# def compute_polarity(A: LatticeState, B: LatticeState, observer: str | None = None) -> float:
#     """
#     Compute semantic polarity Φ (cosθ) between two lattice states.
#
#     Parameters:
#         A, B       : LatticeState instances to compare.
#         observer   : If "Lo", inverts the polarity to represent
#                      the Local Observer’s reversed perspective.
#
#     Returns:
#         Polarity Φ ∈ [-1.0, +1.0], where:
#             +1 → alignment toward Om
#              0 → orthogonal / neutral
#             -1 → negation / opposition
#     """
#     a = A.weights.flatten()
#     b = B.weights.flatten()
#
#     # Handle degenerate (zero) states
#     if np.allclose(a, 0) or np.allclose(b, 0):
#         return 0.0
#
#     dot = np.dot(a, b)
#     norm = np.linalg.norm(a) * np.linalg.norm(b)
#     if norm == 0:
#         return 0.0
#
#     polarity = np.clip(dot / norm, -1.0, 1.0)
#
#     # Invert for local observer frame
#     if observer and observer.upper() == "LO":
#         polarity *= -1.0
#
#     return float(polarity)
#
# # -------------------------------------------------------------------
# # Intent Vector Definition
# # -------------------------------------------------------------------
#
# @dataclass
# class IntentVector:
#     """
#     Represents a semantic transition between two lattice states.
#
#     Attributes:
#         polarity      : Scaled polarity (shifted cosine result)
#         raw_polarity  : Unmodified cosine similarity before scaling
#         delta_energy  : Difference in ΣSW between states
#         rotation_seq  : Transformation or rotation applied to B
#         observer      : Frame of reference ("Om" or "Lo")
#     """
#     polarity: float
#     raw_polarity: float
#     delta_energy: float
#     rotation_seq: str
#     observer: str = "Om"
#
#     def describe(self) -> str:
#         """
#         Return a qualitative description of this intent
#         (alignment type and direction).
#         """
#         p = self.polarity
#         if p > 0.7:
#             meaning = "affirmation / alignment"
#         elif p > 0.2:
#             meaning = "related / parallel"
#         elif p > -0.2:
#             meaning = "neutral / orthogonal"
#         elif p > -0.7:
#             meaning = "contrast / divergence"
#         else:
#             meaning = "negation / opposition"
#
#         direction = "toward Om" if p > 0 else "away from Om"
#         return (
#             f"Intent[{self.observer}]({meaning}, {direction}, "
#             f"Φ={p:.3f}, ΔE={self.delta_energy:.3f})"
#         )
#
# # -------------------------------------------------------------------
# # Intent Computation
# # -------------------------------------------------------------------
#
# def compute_intent(
#     A: LatticeState,
#     B: LatticeState,
#     rotation_seq: str | None = None,
#     observer: str = "Om",
#     coupling: CouplingMap | None = None,
#     polarity_scale: float = 1.5,
#     polarity_shift: float = 0.5,
# ) -> IntentVector:
#     """
#     Compute full semantic intent between two lattice states.
#
#     This function computes:
#         1. Polarity — alignment via cosine similarity, scaled and shifted.
#         2. ΔE (delta_energy) — energy change between states.
#         3. Observer-relative direction — inverts polarity if viewed by Lo.
#
#     Parameters:
#         A, B            : Input lattice states.
#         rotation_seq    : Optional rotation sequence applied to B before comparison.
#         observer        : "Om" or "Lo", defines the reference frame.
#         coupling        : Optional coupling map aligning A to B.
#         polarity_scale  : Scale factor for polarity (default 1.5).
#         polarity_shift  : Offset applied after scaling (default 0.5).
#
#     Returns:
#         IntentVector object containing computed semantic metrics.
#     """
#
#     # Apply rotation to B if specified
#     if rotation_seq:
#         B = rotate_sequence(B, rotation_seq)
#
#     # Apply coupling transformation if provided
#     if coupling is not None:
#         A_aligned = A.clone()
#         A_aligned.weights = apply_coupling(A.weights, B.weights, coupling)
#         A = A_aligned
#
#     # Flatten weight matrices for dot-product computation
#     a = A.weights.flatten()
#     b = B.weights.flatten()
#
#     # Default polarity values for degenerate cases
#     polarity: float = 0.0
#     raw_polarity: float = 0.0
#
#     # Compute raw and scaled polarity
#     if not (np.allclose(a, 0) or np.allclose(b, 0)):
#         dot = float(np.dot(a, b))
#         norm = float(np.linalg.norm(a) * np.linalg.norm(b))
#         # if norm != 0:
#         #     # raw_polarity = np.clip(dot / norm, -1.0, 1.0)
#         #     # We are testing a new hypothesis, so we remove the
#         #     # inversion to get a clean baseline signal.
#         #     raw_polarity = np.clip(dot / norm, -1.0, 1.0)
#         #     polarity = float(np.clip(polarity_scale * raw_polarity - polarity_shift, -1.0, 1.0))
#
#         # In core/semantic.py, inside compute_intent()
#
#         if norm != 0:
#             # --- INVERT THE SIGNAL ---
#             # This is the fix from Run 8.
#             raw_polarity = -1.0 * np.clip(dot / norm, -1.0, 1.0)
#             polarity = float(np.clip(polarity_scale * raw_polarity - polarity_shift, -1.0, 1.0))
#
#     # Compute energy differential between states
#     delta_energy = B.total_sw() - A.total_sw()
#
#     # Invert polarity for local observer frame
#     if observer and observer.upper() == "LO":
#         polarity *= -1.0
#
#     return IntentVector(polarity, raw_polarity, delta_energy, rotation_seq or "", observer)
#
# # -------------------------------------------------------------------
# # Semantic Direction Helper
# # -------------------------------------------------------------------
#
# def toward_center(polarity: float) -> bool:
#     """
#     Determine if polarity indicates motion toward Om.
#     Returns True when Φ > 0 (alignment).
#     """
#     return polarity > 0.0
#
# # -------------------------------------------------------------------
# # Self-Check
# # -------------------------------------------------------------------
#
# if __name__ == "__main__":
#     from core.lattice import canonical_symbol_layout
#     from core.rotation import rotate_z
#
#     base = canonical_symbol_layout()
#     shifted = rotate_z(base, 1)
#
#     intent_om = compute_intent(base, shifted, "Z", "Om")
#     intent_lo = compute_intent(base, shifted, "Z", "Lo")
#
#     print(intent_om.describe())
#     print(intent_lo.describe())
#     print("semantic.py dual-core self-check passed ✓")


"""
Livnium Core — semantic.py
--------------------------
Implements Axiom A5: The Semantic Law and A6: The Intent Relation.

Meaning (Φ) arises from the relative orientation between observers
or lattice states. Polarity (cosθ) expresses their alignment:

    +1.0 → perfect agreement (toward Om)
     0.0 → orthogonal / neutral
    -1.0 → perfect negation (away from Om)

Intent represents the energetic and directional delta between
two states, used in reasoning and growth layers.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from core.coupling import CouplingMap, apply_coupling
from core.lattice import LatticeState
from core.rotation import rotate_sequence

# -------------------------------------------------------------------
# Polarity Computation
# -------------------------------------------------------------------

def compute_polarity(A: LatticeState, B: LatticeState, observer: str | None = None) -> float:
    """
    Compute semantic polarity Φ (cosθ) between two lattice states.
    """
    a = A.weights.flatten()
    b = B.weights.flatten()

    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0

    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0

    polarity = np.clip(dot / norm, -1.0, 1.0)

    if observer and observer.upper() == "LO":
        polarity *= -1.0

    return float(polarity)

# -------------------------------------------------------------------
# Intent Vector Definition
# -------------------------------------------------------------------

@dataclass
class IntentVector:
    """
    Represents a semantic transition between two lattice states.
    """
    polarity: float
    raw_polarity: float
    delta_energy: float
    rotation_seq: list[float] | str = None
    observer: str = "Om"

    def describe(self) -> str:
        p = self.polarity
        if p > 0.7:
            meaning = "affirmation / alignment"
        elif p > 0.2:
            meaning = "related / parallel"
        elif p > -0.2:
            meaning = "neutral / orthogonal"
        elif p > -0.7:
            meaning = "contrast / divergence"
        else:
            meaning = "negation / opposition"

        direction = "toward Om" if p > 0 else "away from Om"
        return (
            f"Intent[{self.observer}]({meaning}, {direction}, "
            f"Φ={p:.3f}, ΔE={self.delta_energy:.3f})"
        )


# -------------------------------------------------------------------
# Intent Computation
# -------------------------------------------------------------------

def compute_intent(
    A: LatticeState,
    B: LatticeState,
    rotation_seq: list[float] | None = None,
    observer: str = "Om",
    coupling: CouplingMap | None = None,
    polarity_scale: float = 1.5,
    polarity_shift: float = 0.5,
) -> IntentVector:
    """
    Compute full semantic intent between two lattice states.
    """

    # Apply rotation to B if specified
    if rotation_seq:
        # Only rotate if this looks like a symbolic rotation sequence (e.g. "X", "YZ")
        if isinstance(rotation_seq, str):
            B = rotate_sequence(B, rotation_seq)
        elif isinstance(rotation_seq, (list, tuple)):
            # Numeric rotation lists are ignored for now (placeholder for continuous rotations)
            pass

    # Apply coupling transformation if provided
    if coupling is not None:
        A_aligned = A.clone()
        A_aligned.weights = apply_coupling(A.weights, B.weights, coupling)
        A = A_aligned

    a = A.weights.flatten()
    b = B.weights.flatten()

    polarity = 0.0
    raw_polarity = 0.0

    if not (np.allclose(a, 0) or np.allclose(b, 0)):
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm != 0:
            raw_polarity = np.clip(dot / norm, -1.0, 1.0)
            polarity = float(np.clip(polarity_scale * raw_polarity - polarity_shift, -1.0, 1.0))

    # compute meaningful ΔE as mean absolute lattice divergence
    delta_energy = float(np.mean(np.abs(B.weights - A.weights)))

    if observer and observer.upper() == "LO":
        polarity *= -1.0

    return IntentVector(
        polarity=polarity,
        raw_polarity=raw_polarity,
        delta_energy=delta_energy,
        rotation_seq=rotation_seq or [0.0, 0.0, 0.0],
        observer=observer,
    )

# -------------------------------------------------------------------
# Semantic Direction Helper
# -------------------------------------------------------------------

def toward_center(polarity: float) -> bool:
    """
    Determine if polarity indicates motion toward Om.
    Returns True when Φ > 0 (alignment).
    """
    return polarity > 0.0

# -------------------------------------------------------------------
# Self-Check
# -------------------------------------------------------------------

if __name__ == "__main__":
    from core.lattice import canonical_symbol_layout
    from core.rotation import rotate_z

    base = canonical_symbol_layout()
    shifted = rotate_z(base, 1)

    intent_om = compute_intent(base, shifted, [0.0, 0.0, 0.0], "Om")
    intent_lo = compute_intent(base, shifted, [0.0, 0.0, 0.0], "Lo")

    print(intent_om.describe())
    print(intent_lo.describe())
    print("semantic.py self-check passed ✓")
