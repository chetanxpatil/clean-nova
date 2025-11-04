"""
Livnium Core — Initialization (Dual-Core Architecture)
======================================================

Defines the canonical import interface for the Livnium Core system.

This Core implements:
    - Axiom A1: Spatial Alphabet (3×3×3 Lattice)
    - Axiom A2: Observer Principle (Om–Lo dual frames)
    - Axiom A3: Symbolic Weight Law (SW = 9×f)
    - Axiom A4: Dynamic Law (Reversible Rotations)
    - Axiom A5: Semantic Law (Polarity and Intent)
    - Derived Law D3: Conservation of Symbolic Weight (ΣSW = 486)
    - Derived Law D4: Auditable Intelligence (All operations are logged)

Everything in Livnium operates under **Conservation + Reversibility**.
All computation is geometric, all meaning is directional, and all
learning is transparent and auditable.
"""

from __future__ import annotations

# -------------------------------------------------------------------
# Core imports — the geometric substrate
# -------------------------------------------------------------------

from .lattice import (
    LatticeState,
    canonical_symbol_layout,
    identity_state,
    rebalance,
)

from .rotation import (
    rotate_x,
    rotate_y,
    rotate_z,
    rotate_sequence,
    audited_rotate,
)

from .observer import (
    GlobalObserver,
    LocalObserver,
)

from .semantic import (
    compute_polarity,
    compute_intent,
    IntentVector,
)

from .conservation import (
    verify_conservation,
    deviation,
    energy_ratio,
    ConservationLedger,
    verify_equilibrium,
    CANONICAL_SUM_SW,
)

from .audit import (
    audit_cycle,
    AuditLog,
)

# -------------------------------------------------------------------
# Module Metadata
# -------------------------------------------------------------------

__version__ = "2.1.0"
__author__ = "Livnium Research (2025)"
__license__ = "Reversible Geometry License"
__summary__ = "Reversible, auditable, geometric computation engine."

__all__ = [
    # lattice
    "LatticeState",
    "canonical_symbol_layout",
    "identity_state",
    "rebalance",

    # rotation
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "rotate_sequence",
    "audited_rotate",

    # observer
    "GlobalObserver",
    "LocalObserver",

    # semantics
    "compute_polarity",
    "compute_intent",
    "IntentVector",

    # conservation
    "verify_conservation",
    "deviation",
    "energy_ratio",
    "ConservationLedger",
    "verify_equilibrium",
    "CANONICAL_SUM_SW",

    # auditing
    "audit_cycle",
    "AuditLog",
]


# -------------------------------------------------------------------
# Self-check (optional quick audit)
# -------------------------------------------------------------------

if __name__ == "__main__":
    base = canonical_symbol_layout()
    print("Livnium Core v2.1.0 initialized ✓")
    print("ΣSW =", base.total_sw())
    print("Conserved:", verify_conservation(base))
    print("Om–Lo dual equilibrium test passed:", verify_equilibrium(base, base))
