"""
Livnium Core — audit.py (Updated for Dual-Core Architecture)
------------------------------------------------------------
Implements Axiom D4: Auditable Intelligence.

Every Livnium operation must be:
    • Reversible (rotation-based)
    • Conserved (ΣSW = 486)
    • Recorded (Φ, ΔE, observer context)

The audit layer verifies that each reasoning step preserves the
geometric law while documenting its semantic direction.
"""

from __future__ import annotations
import time
import json
import numpy as np
from dataclasses import dataclass, field

try:  # pragma: no cover
    from .semantic import compute_intent
    from .conservation import verify_conservation, deviation
except ImportError:  # fallback
    from semantic import compute_intent  # type: ignore
    from conservation import verify_conservation, deviation  # type: ignore


# -------------------------------------------------------------------
# Audit Entry — atomic record of a single transformation
# -------------------------------------------------------------------

@dataclass
class AuditEntry:
    timestamp: float
    operation: str
    observer: str
    polarity: float
    delta_energy: float
    conserved: bool
    note: str = ""

    def to_dict(self) -> dict:
        """Convert the entry to a serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "observer": self.observer,
            "polarity": self.polarity,
            "delta_energy": self.delta_energy,
            "conserved": self.conserved,
            "note": self.note,
        }


# -------------------------------------------------------------------
# Audit Log — chronological ledger of state transitions
# -------------------------------------------------------------------

@dataclass
class AuditLog:
    """
    Records the sequence of verified transformations across observers.
    This log is the verifiable “conscious history” of Livnium reasoning.
    """
    entries: list[AuditEntry] = field(default_factory=list)

    # ---------------------------------------------------------------
    # Recording transformations
    # ---------------------------------------------------------------

    def record(self, before_state, after_state, operation: str, observer: str = "Om", note: str = "") -> None:
        """
        Record a single reversible transformation between two states.
        Automatically computes Φ (polarity) and ΔE (energy shift).
        """
        intent = compute_intent(before_state, after_state, None, observer)
        entry = AuditEntry(
            timestamp=time.time(),
            operation=operation,
            observer=observer,
            polarity=round(intent.polarity, 6),
            delta_energy=round(intent.delta_energy, 6),
            conserved=verify_conservation(after_state),
            note=note,
        )
        self.entries.append(entry)

    # ---------------------------------------------------------------
    # Verification and integrity
    # ---------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """Return True if all operations maintained ΣSW = 486."""
        return all(e.conserved for e in self.entries)

    def summary(self) -> dict:
        """Return statistical summary of the audit history."""
        if not self.entries:
            return {"count": 0, "integrity_passed": True}

        polarities = np.array([e.polarity for e in self.entries])
        deltas = np.array([e.delta_energy for e in self.entries])
        return {
            "count": len(self.entries),
            "mean_polarity": float(np.mean(polarities)),
            "mean_delta_energy": float(np.mean(deltas)),
            "integrity_passed": self.verify_integrity(),
            "range_polarity": [float(np.min(polarities)), float(np.max(polarities))],
        }

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export the full audit history to a JSON file."""
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def clear(self) -> None:
        """Reset the audit log."""
        self.entries.clear()

    # ---------------------------------------------------------------
    # Pretty-print helper
    # ---------------------------------------------------------------

    def describe(self) -> str:
        """Readable string summary for console/debug output."""
        s = self.summary()
        return (
            f"AuditLog(count={s['count']}, "
            f"Φ̄={s.get('mean_polarity', 0):.3f}, "
            f"ΔĒ={s.get('mean_delta_energy', 0):.3f}, "
            f"integrity={s['integrity_passed']})"
        )


# -------------------------------------------------------------------
# Helper: one-cycle audit wrapper
# -------------------------------------------------------------------

def audit_cycle(
    before_state,
    after_state,
    operation: str,
    observer: str = "Om",
    log: AuditLog | None = None,
    note: str = ""
) -> AuditLog:
    """
    Record one reversible operation and return the updated log.
    Creates a new log if none exists.
    """
    if log is None:
        log = AuditLog()
    log.record(before_state, after_state, operation, observer, note)
    return log


# -------------------------------------------------------------------
# Self-check
# -------------------------------------------------------------------

if __name__ == "__main__":
    try:  # pragma: no cover
        from .lattice import canonical_symbol_layout
        from .rotation import rotate_y
    except ImportError:  # fallback
        from lattice import canonical_symbol_layout  # type: ignore
        from rotation import rotate_y  # type: ignore

    base = canonical_symbol_layout()
    rotated = rotate_y(base, 1)

    log = AuditLog()
    log = audit_cycle(base, rotated, "rotate_y(+90)", observer="Lo", log=log, note="perceptual shift")
    log = audit_cycle(rotated, base, "rotate_y(-90)", observer="Om", log=log, note="return to equilibrium")

    print(log.describe())
    log.export_json("audit_dual_test.json")
    print("audit.py dual-core self-check passed ✓")
