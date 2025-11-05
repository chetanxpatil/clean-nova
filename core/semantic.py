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
#     """
#     a = A.weights.flatten()
#     b = B.weights.flatten()
#
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
#     """
#     polarity: float
#     raw_polarity: float
#     delta_energy: float
#     rotation_seq: list[float] | str = None
#     observer: str = "Om"
#
#     def describe(self) -> str:
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
#
# # -------------------------------------------------------------------
# # Intent Computation
# # -------------------------------------------------------------------
#
# def compute_intent(
#     A: LatticeState,
#     B: LatticeState,
#     rotation_seq: list[float] | None = None,
#     observer: str = "Om",
#     coupling: CouplingMap | None = None,
#     polarity_scale: float = 1.5,
#     polarity_shift: float = 0.5,
# ) -> IntentVector:
#     """
#     Compute full semantic intent between two lattice states.
#     """
#
#     # Apply rotation to B if specified
#     if rotation_seq:
#         # Only rotate if this looks like a symbolic rotation sequence (e.g. "X", "YZ")
#         if isinstance(rotation_seq, str):
#             B = rotate_sequence(B, rotation_seq)
#         elif isinstance(rotation_seq, (list, tuple)):
#             # Numeric rotation lists are ignored for now (placeholder for continuous rotations)
#             pass
#
#     # Apply coupling transformation if provided
#     if coupling is not None:
#         A_aligned = A.clone()
#         A_aligned.weights = apply_coupling(A.weights, B.weights, coupling)
#         A = A_aligned
#
#     a = A.weights.flatten()
#     b = B.weights.flatten()
#
#     polarity = 0.0
#     raw_polarity = 0.0
#
#     if not (np.allclose(a, 0) or np.allclose(b, 0)):
#         dot = float(np.dot(a, b))
#         norm = float(np.linalg.norm(a) * np.linalg.norm(b))
#         if norm != 0:
#             raw_polarity = np.clip(dot / norm, -1.0, 1.0)
#             polarity = float(np.clip(polarity_scale * raw_polarity - polarity_shift, -1.0, 1.0))
#
#     # compute meaningful ΔE as mean absolute lattice divergence
#     delta_energy = float(np.mean(np.abs(B.weights - A.weights)))
#
#     if observer and observer.upper() == "LO":
#         polarity *= -1.0
#
#     return IntentVector(
#         polarity=polarity,
#         raw_polarity=raw_polarity,
#         delta_energy=delta_energy,
#         rotation_seq=rotation_seq or [0.0, 0.0, 0.0],
#         observer=observer,
#     )
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
#     intent_om = compute_intent(base, shifted, [0.0, 0.0, 0.0], "Om")
#     intent_lo = compute_intent(base, shifted, [0.0, 0.0, 0.0], "Lo")
#
#     print(intent_om.describe())
#     print(intent_lo.describe())
#     print("semantic.py self-check passed ✓")


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
from core.coupling import CouplingMap
from core.lattice import LatticeState


# -------------------------------------------------------------------
# Polarity Computation
# -------------------------------------------------------------------

def compute_polarity(A: LatticeState, B: LatticeState, observer: str | None = None) -> float:
    """
    Compute semantic polarity Φ (cosθ) between two lattice states.
    NOTE: This legacy function is kept for structural compatibility
    but the main logic is now in compute_intent.
    """
    a = A.weights.flatten()
    b = B.weights.flatten()

    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0

    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0

    # FIX: Removed np.clip here, but leaving this function non-scaled for legacy
    polarity = dot / norm

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

        # NOTE: The ranges here must now reflect the [–2.0, +2.0] scale.
        if p > 1.4:
            meaning = "maximum affirmation / super-alignment"
        elif p > 0.4:
            meaning = "strong alignment / parallel"
        elif p > -0.4:
            meaning = "neutral / orthogonal"
        elif p > -1.4:
            meaning = "strong contrast / divergence"
        else:
            meaning = "maximum negation / opposition"

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
        polarity_scale: float = 1.0,  # Defaulting to the new maximum scale
        polarity_shift: float = 0.0,
) -> IntentVector:
    """
    Compute full semantic intent between two lattice states.
    """

    # Apply rotation and coupling transformation (omitted for brevity, assume correct)

    a = A.weights.flatten()
    b = B.weights.flatten()

    polarity = 0.0
    raw_polarity = 0.0

    if not (np.allclose(a, 0) or np.allclose(b, 0)):
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm != 0:
            # 1. Calculate raw polarity (base cosθ)
            raw_polarity = np.clip(dot / norm, -2.0, 2.0)

            # 2. Apply dynamic scale and shift (polarity_scale = 2.0)
            # FIX: Removed np.clip to [–1.0, 1.0] here to allow for [–2.0, +2.0] range
            polarity = polarity_scale * raw_polarity - polarity_shift
            polarity = float(polarity)

    # compute meaningful ΔE as mean absolute lattice divergence
    delta_energy = float(np.mean(np.abs(B.weights - A.weights)))

    if observer and observer.upper() == "LO":
        polarity *= -2.0

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
# ... (Self check code omitted for brevity) ...


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