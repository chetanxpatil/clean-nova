# --- growth/policy.py (Thread-Safe for Inward Growth) ---
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import threading # <-- ADDED for concurrency control

@dataclass
class PolicyPi:
    Q: Dict[str, float] = field(default_factory=lambda: {
        "merge": 0.0,
        "branch": 0.0,
        "stabilize": 0.50,
        "revert": 0.0
    })
    alpha = 0.05
    gamma: float = 0.9

    # ADDED: Lock to ensure thread-safe updates to the shared Q dictionary
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, rule: str, reward: float):
        """Basic Q-value update, protected by a lock to prevent race conditions."""
        with self._lock: # <-- THREAD-SAFETY HOOK
            q_old = self.Q.get(rule, 0.0)
            self.Q[rule] = q_old + self.alpha * (reward - q_old)

    def softmax_probs(self, temperature: float = 0.1) -> Dict[str, float]:
        # NOTE: Reads don't usually need a lock unless atomicity is critical,
        # but the array copy ensures stability for computation.
        q_vals = np.array(list(self.Q.values()))
        q_exp = np.exp(q_vals / max(temperature, 1e-6))
        probs = q_exp / np.sum(q_exp)
        return dict(zip(self.Q.keys(), probs))

    def describe(self, rules: list[str] = None) -> str:
        """Returns a string representation of the Q-values."""
        # Reads are safe outside the lock for diagnostics
        if rules is None:
            return "Q=" + ", ".join(f"{k}:{v:+.2f}" for k, v in sorted(self.Q.items()))

        display_q = {k: self.Q.get(k, 0.0) for k in rules}
        return "Q=" + ", ".join(f"{k}:{v:+.2f}" for k, v in sorted(display_q.items()))

    def entropy(self, temperature: float = 0.1) -> float:
        """Shannon entropy of the softmax policy at the given temperature."""
        p = np.asarray(list(self.softmax_probs(temperature).values()), dtype=float)
        return float(-np.sum(p * np.log2(p + 1e-9)))