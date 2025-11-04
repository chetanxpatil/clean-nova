# growth/policy.py
"""
Defines the PolicyPi class for managing Q-value-based policy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

@dataclass
class PolicyPi:
    Q: Dict[str, float] = field(default_factory=lambda: {"merge": 0.0, "branch": 0.0, "stabilize": 0.0, "revert": 0.0})
    alpha = 0.05  # or even 0.1 temporarily   # learning rate
    gamma: float = 0.9  # discount factor

    def update(self, rule: str, reward: float):
        """Basic Q-value update"""
        q_old = self.Q.get(rule, 0.0)
        self.Q[rule] = q_old + self.alpha * (reward - q_old)

    def softmax_probs(self, temperature: float = 0.1) -> Dict[str, float]:
        q_vals = np.array(list(self.Q.values()))
        q_exp = np.exp(q_vals / max(temperature, 1e-6))
        probs = q_exp / np.sum(q_exp)
        return dict(zip(self.Q.keys(), probs))

    def describe(self) -> str:
        return "Q=" + ", ".join(f"{k}:{v:+.2f}" for k, v in sorted(self.Q.items()))

    def entropy(self, temperature: float = 0.1) -> float:
        """Shannon entropy of the softmax policy at the given temperature."""
        p = np.asarray(list(self.softmax_probs(temperature).values()), dtype=float)
        return float(-np.sum(p * np.log2(p + 1e-9)))