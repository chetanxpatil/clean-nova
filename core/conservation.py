# """
# Livnium Core — conservation.py (Updated for Dual-Core Architecture)
# -------------------------------------------------------------------
# Implements Axiom D3: Conservation of Symbolic Weight.
#
# Every valid Livnium lattice must satisfy:
#     ΣSW = 486
#
# This conservation is invariant under all reversible rotations and
# semantic transformations. Deviations (ΔE) are measurable as
# semantic energy — a reversible oscillation between Om and Lo.
# """
#
# from __future__ import annotations
# import warnings
# import numpy as np
# from dataclasses import dataclass, field
#
# # Canonical invariant
# CANONICAL_SUM_SW = 486.0
# TOLERANCE = 1e-6
#
#
# # -------------------------------------------------------------------
# # Verification utilities
# # -------------------------------------------------------------------
#
# def verify_conservation(state) -> bool:
#     """Check whether a lattice satisfies ΣSW = 486 within tolerance."""
#     total = float(np.sum(state.weights))
#     return abs(total - CANONICAL_SUM_SW) < TOLERANCE
#
#
# # -------------------------------------------------------------------
# # Normalization with adaptive drift damping
# # -------------------------------------------------------------------
#
# def normalize(vec: np.ndarray) -> np.ndarray:
#     """
#     Renormalize a vector of weights while preserving ΣSW ≈ 486.
#
#     Adds mild stochasticity to avoid fixed-point stagnation and
#     clamps overcorrections to keep the drift bounded.
#     """
#     total = np.sum(vec)
#     drift = total - CANONICAL_SUM_SW
#
#     if abs(drift) > 0.2 * CANONICAL_SUM_SW:
#         correction = drift * 0.5        # stronger pull for large drift
#     else:
#         correction = drift * 0.1        # gentle damping for small drift
#
#     vec -= correction / len(vec)
#
#     # tiny entropy injection — prevents symmetry lock
#     vec += np.random.uniform(-0.02, 0.02, size=vec.shape)
#
#     # ensure no negatives
#     vec = np.clip(vec, 0.0, None)
#
#     # final scale correction to exact ΣSW
#     scale = CANONICAL_SUM_SW / np.sum(vec)
#     vec *= scale
#     return vec
#
#
# # -------------------------------------------------------------------
# # Ledger safety decorator
# # -------------------------------------------------------------------
#
# # def conserve_ledger(func):
# #     """Decorator to enforce ΣSW conservation across lattice operations (non-fatal)."""
# #     def wrapper(*args, **kwargs):
# #         before = np.sum(args[0].weights)
# #         result = func(*args, **kwargs)
# #         after = np.sum(args[0].weights)
# #
# #         delta = after - before
# #         if not np.isclose(before, after, rtol=1e-6, atol=1e-3):
# #             warnings.warn(
# #                 f"⚠️ ΣSW drift {delta:+.6f} detected in {func.__name__} "
# #                 "(auto-corrected / non-fatal).",
# #                 RuntimeWarning,
# #             )
# #             args[0].weights = normalize(args[0].weights)
# #         return result
# #     return wrapper
#
# def conserve_ledger(func):
#     """Decorator to enforce ΣSW conservation across lattice operations (non-fatal)."""
#     def wrapper(*args, **kwargs):
#         before = float(np.sum(args[0].weights))
#         result = func(*args, **kwargs)
#         after = float(np.sum(args[0].weights))
#
#         delta = after - before
#
#         if not np.isclose(before, after, rtol=1e-6, atol=1e-3):
#             # diagnostic print
#             print(f"[Conservation Drift] before={before:.6f}, after={after:.6f}, Δ={delta:+.6f}")
#
#             warnings.warn(
#                 f"⚠️ ΣSW drift {delta:+.6f} detected in {func.__name__} "
#                 "(auto-corrected / non-fatal).",
#                 RuntimeWarning,
#             )
#
#             # Renormalize to restore exact total
#             args[0].normalize()
#
#             # verify fix
#             corrected = float(np.sum(args[0].weights))
#             print(f"[Correction Applied] new total={corrected:.6f}, deviation={corrected - CANONICAL_SUM_SW:+.6f}")
#
#         return result
#     return wrapper
#
#
# # -------------------------------------------------------------------
# # Deviation and energy diagnostics
# # -------------------------------------------------------------------
#
# def deviation(state) -> float:
#     """Compute ΔE = (ΣSW - 486), symbolic energy deviation."""
#     return float(np.sum(state.weights) - CANONICAL_SUM_SW)
#
#
# def energy_ratio(state) -> float:
#     """Return relative deviation ratio (ΔE / 486)."""
#     return deviation(state) / CANONICAL_SUM_SW
#
#
# # -------------------------------------------------------------------
# # Conservation Ledger — persistent energy and balance tracker
# # -------------------------------------------------------------------
#
# @dataclass
# class ConservationLedger:
#     record: list[float] = field(default_factory=list)
#     observers: list[str] = field(default_factory=list)
#
#     def log(self, state, observer: str = "Om") -> None:
#         total = float(np.sum(state.weights))
#         self.record.append(total)
#         self.observers.append(observer)
#
#     def summary(self) -> dict:
#         if not self.record:
#             return {"count": 0, "conserved": True}
#
#         arr = np.array(self.record)
#         deviations = arr - CANONICAL_SUM_SW
#         return {
#             "count": len(arr),
#             "mean_ΣSW": float(np.mean(arr)),
#             "min_ΣSW": float(np.min(arr)),
#             "max_ΣSW": float(np.max(arr)),
#             "ΔE_mean": float(np.mean(deviations)),
#             "ΔE_range": [float(np.min(deviations)), float(np.max(deviations))],
#             "within_tolerance": bool(all(abs(d) < TOLERANCE for d in deviations)),
#             "energy_stability": float(np.ptp(deviations) / CANONICAL_SUM_SW),
#             "frames_logged": list(set(self.observers)),
#         }
#
#     def energy_trace(self) -> list[float]:
#         """Return chronological ΔE values for audit."""
#         return [val - CANONICAL_SUM_SW for val in self.record]
#
#     def clear(self) -> None:
#         self.record.clear()
#         self.observers.clear()
#
#     def describe(self) -> str:
#         s = self.summary()
#         return (
#             f"ConservationLedger(count={s['count']}, "
#             f"mean_ΣSW={s.get('mean_ΣSW', 0):.3f}, "
#             f"ΔĒ={s.get('ΔE_mean', 0):.6f}, "
#             f"stable={s.get('within_tolerance', True)})"
#         )
#
#
# # -------------------------------------------------------------------
# # Dual-frame equilibrium helper
# # -------------------------------------------------------------------
#
# def verify_equilibrium(Om_state, Lo_state) -> dict:
#     """
#     Compare Om and Lo frames to ensure total equilibrium:
#         ΔE_total = deviation(Om) + deviation(Lo)
#     """
#     d_om = deviation(Om_state)
#     d_lo = deviation(Lo_state)
#     total = d_om + d_lo
#     return {
#         "ΔE(Om)": d_om,
#         "ΔE(Lo)": d_lo,
#         "ΔE_total": total,
#         "equilibrium": abs(total) < TOLERANCE,
#     }
#
#
# # -------------------------------------------------------------------
# # Self-check
# # -------------------------------------------------------------------
#
# if __name__ == "__main__":
#     try:
#         from .lattice import canonical_symbol_layout
#         from .rotation import rotate_x
#     except ImportError:
#         from lattice import canonical_symbol_layout  # type: ignore
#         from rotation import rotate_x  # type: ignore
#
#     base = canonical_symbol_layout()
#     rotated = rotate_x(base, 1)
#
#     print("verify_conservation(base):", verify_conservation(base))
#     print("ΔE(base):", deviation(base))
#
#     ledger = ConservationLedger()
#     ledger.log(base, "Om")
#     ledger.log(rotated, "Lo")
#     print(ledger.describe())
#
#     eq = verify_equilibrium(base, rotated)
#     print("Equilibrium:", eq)
#     print("conservation.py dual-core self-check passed ✓")

# Livnium Core — conservation.py (Updated for Scalable Architecture)
# -------------------------------------------------------------------
# Implements Axiom D3: Conservation of Symbolic Weight.
# Now dynamically reads the TOTAL_LEDGER_TARGET (e.g., 486 or 1350)
# from the core lattice definition.
# -------------------------------------------------------------------

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# --- Import Dynamic Constants ---
# We assume core.lattice is defined and sets TOTAL_LEDGER_TARGET
try:
    from .lattice import TOTAL_LEDGER_TARGET, GRID_SIZE
except ImportError:
    # Fallback if running outside the module structure (e.g., self-check)
    TOTAL_LEDGER_TARGET = 486.0 #5
    # TOTAL_LEDGER_TARGET=4374.0
    GRID_SIZE = 3

TOLERANCE = 1e-6
CANONICAL_SUM_SW = TOTAL_LEDGER_TARGET


# -------------------------------------------------------------------
# Verification utilities
# -------------------------------------------------------------------

def verify_conservation(state) -> bool:
    """Check whether a lattice satisfies ΣSW = TOTAL_LEDGER_TARGET within tolerance."""
    total = float(np.sum(state.weights))
    # FIX: Use the dynamic constant
    return abs(total - CANONICAL_SUM_SW) < TOLERANCE


# -------------------------------------------------------------------
# Normalization with adaptive drift damping (Not used by LatticeState.normalize)
# -------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    """
    Renormalize a vector of weights while preserving ΣSW ≈ TARGET.
    """
    total = np.sum(vec)
    drift = total - CANONICAL_SUM_SW

    if abs(drift) > 0.2 * CANONICAL_SUM_SW:
        correction = drift * 0.5
    else:
        correction = drift * 0.1

    vec -= correction / len(vec)

    # tiny entropy injection — prevents symmetry lock
    vec += np.random.uniform(-0.02, 0.02, size=vec.shape)

    # ensure no negatives
    vec = np.clip(vec, 0.0, None)

    # final scale correction to exact ΣSW
    vec *= CANONICAL_SUM_SW / np.sum(vec)
    return vec


# -------------------------------------------------------------------
# Ledger safety decorator
# -------------------------------------------------------------------

def conserve_ledger(func):
    """Decorator to enforce ΣSW conservation across lattice operations (non-fatal)."""

    def wrapper(*args, **kwargs):
        before = float(np.sum(args[0].weights))
        result = func(*args, **kwargs)
        after = float(np.sum(args[0].weights))

        delta = after - before

        if not np.isclose(before, after, rtol=1e-6, atol=1e-3):
            # The LatticeState.normalize() method uses the correct TOTAL_LEDGER_TARGET,
            # so we just call it to fix the drift.

            # diagnostic print
            print(f"[Conservation Drift] before={before:.6f}, after={after:.6f}, Δ={delta:+.6f}")

            warnings.warn(
                f"⚠️ ΣSW drift {delta:+.6f} detected in {func.__name__} "
                "(auto-corrected / non-fatal).",
                RuntimeWarning,
            )

            # Renormalize to restore exact total
            args[0].normalize()

            # verify fix
            corrected = float(np.sum(args[0].weights))
            # FIX: Use CANONICAL_SUM_SW for deviation check
            print(f"[Correction Applied] new total={corrected:.6f}, deviation={corrected - CANONICAL_SUM_SW:+.6f}")

        return result

    return wrapper


# -------------------------------------------------------------------
# Deviation and energy diagnostics
# -------------------------------------------------------------------

def deviation(state) -> float:
    """Compute ΔE = (ΣSW - TARGET), symbolic energy deviation."""
    # FIX: Use the dynamic constant
    return float(np.sum(state.weights) - CANONICAL_SUM_SW)


def energy_ratio(state) -> float:
    """Return relative deviation ratio (ΔE / TARGET)."""
    # FIX: Use the dynamic constant
    return deviation(state) / CANONICAL_SUM_SW


# -------------------------------------------------------------------
# Conservation Ledger — persistent energy and balance tracker
# -------------------------------------------------------------------

@dataclass
class ConservationLedger:
    record: list[float] = field(default_factory=list)
    observers: list[str] = field(default_factory=list)

    def log(self, state, observer: str = "Om") -> None:
        total = float(np.sum(state.weights))
        self.record.append(total)
        self.observers.append(observer)

    def summary(self) -> dict:
        if not self.record:
            return {"count": 0, "conserved": True}

        arr = np.array(self.record)
        # FIX: Use the dynamic constant for deviation calculation
        deviations = arr - CANONICAL_SUM_SW

        return {
            "count": len(arr),
            "mean_ΣSW": float(np.mean(arr)),
            "min_ΣSW": float(np.min(arr)),
            "max_ΣSW": float(np.max(arr)),
            "ΔE_mean": float(np.mean(deviations)),
            "ΔE_range": [float(np.min(deviations)), float(np.max(deviations))],
            "within_tolerance": bool(all(abs(d) < TOLERANCE for d in deviations)),
            "energy_stability": float(np.ptp(deviations) / CANONICAL_SUM_SW),
            "frames_logged": list(set(self.observers)),
        }

    def energy_trace(self) -> list[float]:
        """Return chronological ΔE values for audit."""
        # FIX: Use the dynamic constant
        return [val - CANONICAL_SUM_SW for val in self.record]

    def clear(self) -> None:
        self.record.clear()
        self.observers.clear()

    def describe(self) -> str:
        s = self.summary()
        return (
            f"ConservationLedger(count={s['count']}, "
            f"mean_ΣSW={s.get('mean_ΣSW', 0):.3f}, "
            f"ΔĒ={s.get('ΔE_mean', 0):.6f}, "
            f"stable={s.get('within_tolerance', True)})"
        )


# -------------------------------------------------------------------
# Dual-frame equilibrium helper
# -------------------------------------------------------------------

def verify_equilibrium(Om_state, Lo_state) -> dict:
    """
    Compare Om and Lo frames to ensure total equilibrium:
        ΔE_total = deviation(Om) + deviation(Lo)
    """
    d_om = deviation(Om_state)
    d_lo = deviation(Lo_state)
    total = d_om + d_lo
    return {
        "ΔE(Om)": d_om,
        "ΔE(Lo)": d_lo,
        "ΔE_total": total,
        "equilibrium": abs(total) < TOLERANCE,
    }


# -------------------------------------------------------------------
# Self-check
# -------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from lattice import canonical_symbol_layout  # type: ignore
        from rotation import rotate_x  # type: ignore
    except ImportError:
        # Fallback for self-check when run as main script
        print("Self-check requires importing 'lattice' module correctly.")