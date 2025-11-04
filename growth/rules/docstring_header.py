"""
Livnium Growth — rules.py
=========================
Implements the deterministic growth laws that govern structural cognition.

A Growth Rule is a reversible, geometric transformation applied to a
GrowthNode or pair of GrowthNodes.  Each rule must preserve:

    ΣSW == 486
    -1.0 ≤ Φ ≤ +1.0
    verify_conservation() == True

Growth Rules (canonical):
    G1 — Merge (alignment)
    G2 — Branch (divergence)
    G3 — Stabilize (rebalance after perturbation)
    G4 — Revert (undo last growth step)

All rules operate purely on geometric and semantic quantities:
    LatticeState, IntentVector, AuditLog
"""
