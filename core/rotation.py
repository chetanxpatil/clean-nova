"""
Livnium Core — rotation.py (Circular Import Safe)
-------------------------------------------------
Implements Axiom A4: Dynamic Law and Axiom D4: Auditable Motion.

Defines reversible 90° rotations about X, Y, Z axes.
All rotations conserve ΣSW = 486 and can optionally emit
an AuditLog entry to preserve full motion history.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# Only import lattice + conservation here — safe, no cycle
from core.lattice import LatticeState
from core.conservation import verify_conservation


# -------------------------------------------------------------------
# Core reversible rotations
# -------------------------------------------------------------------

def rotate_x(state: LatticeState, k: int = 1) -> LatticeState:
    """Rotate 90°×k about X-axis (clockwise along +X)."""
    s = state.clone()
    k = k % 4
    if k:
        s.cells = np.rot90(s.cells, k=k, axes=(1, 2))
        s.weights = np.rot90(s.weights, k=k, axes=(1, 2))
    assert verify_conservation(s), "Rotation X violated ΣSW!"
    return s


def rotate_y(state: LatticeState, k: int = 1) -> LatticeState:
    """Rotate 90°×k about Y-axis (clockwise along +Y)."""
    s = state.clone()
    k = k % 4
    if k:
        s.cells = np.rot90(s.cells, k=k, axes=(0, 2))
        s.weights = np.rot90(s.weights, k=k, axes=(0, 2))
    assert verify_conservation(s), "Rotation Y violated ΣSW!"
    return s


def rotate_z(state: LatticeState, k: int = 1) -> LatticeState:
    """Rotate 90°×k about Z-axis (clockwise along +Z)."""
    s = state.clone()
    k = k % 4
    if k:
        s.cells = np.rot90(s.cells, k=k, axes=(0, 1))
        s.weights = np.rot90(s.weights, k=k, axes=(0, 1))
    assert verify_conservation(s), "Rotation Z violated ΣSW!"
    return s


# -------------------------------------------------------------------
# Rotation sequence utility
# -------------------------------------------------------------------

def rotate_sequence(state: LatticeState, sequence: str) -> LatticeState:
    """
    Apply multiple rotations in order:
        X,Y,Z → +90°
        x,y,z → −90°
    Example: rotate_sequence(state, "XYZzyx") yields full identity cycle.
    """
    s = state.clone()
    for move in sequence:
        if move == "X":
            s = rotate_x(s, 1)
        elif move == "x":
            s = rotate_x(s, -1)
        elif move == "Y":
            s = rotate_y(s, 1)
        elif move == "y":
            s = rotate_y(s, -1)
        elif move == "Z":
            s = rotate_z(s, 1)
        elif move == "z":
            s = rotate_z(s, -1)
        else:
            raise ValueError(f"Unknown rotation symbol: {move}")
    return s


# -------------------------------------------------------------------
# Audited wrapper — motion with memory
# -------------------------------------------------------------------

def audited_rotate(
    state: LatticeState,
    axis: str,
    k: int = 1,
    observer: str = "Om",
    log: Optional["AuditLog"] = None,
    note: str = ""
) -> tuple[LatticeState, "AuditLog"]:
    """
    Perform a reversible rotation while automatically recording it
    into an AuditLog (if provided). Returns (rotated_state, log).

    Example:
        new_state, log = audited_rotate(state, "Z", 1, "Lo", log)
    """

    # ⬇️ Import audit tools *inside* the function to break circular imports
    from core.audit import audit_cycle

    before = state.clone()
    axis = axis.upper()

    if axis == "X":
        after = rotate_x(state, k)
    elif axis == "Y":
        after = rotate_y(state, k)
    elif axis == "Z":
        after = rotate_z(state, k)
    else:
        raise ValueError("Invalid axis: must be X, Y, or Z")

    op = f"rotate_{axis}({k:+d}×90°)"
    log = audit_cycle(before, after, op, observer=observer, log=log, note=note)
    return after, log


# -------------------------------------------------------------------
# Self-check
# -------------------------------------------------------------------

if __name__ == "__main__":
    from core.lattice import canonical_symbol_layout

    base = canonical_symbol_layout()
    from core.audit import AuditLog  # delayed import even here

    log = AuditLog()
    rotated, log = audited_rotate(base, "Z", 1, observer="Lo", log=log, note="initial rotation")
    restored, log = audited_rotate(rotated, "Z", -1, observer="Om", log=log, note="return to base")

    print("ΣSW conserved after full cycle:", verify_conservation(restored))
    print(log.describe())
    print("rotation.py audited self-check passed ✓")
