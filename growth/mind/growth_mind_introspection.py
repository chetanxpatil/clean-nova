# growth/growth_mind_introspection.py
"""
Mixin class for GrowthMind to handle introspection, stats, and traversal.
"""
from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import time
import numpy as np

if TYPE_CHECKING:
    from growth.node.core_node import GrowthNode
    from growth.mind.growth_mind import GrowthMind

class GrowthMindIntrospectionMixin:
    def traverse_bfs(self: 'GrowthMind') -> Iterable['GrowthNode']:
        q = [self.root]
        seen = set()
        while q:
            n = q.pop(0)
            if id(n) in seen:
                continue
            seen.add(id(n))
            yield n
            q.extend(n.children)

    def stats(self: 'GrowthMind') -> dict:
        nodes = list(self.traverse_bfs())
        if not nodes:
            return {"count": 0}
        phis = [n.polarity for n in nodes]
        var = float(np.var(phis))

        if var < self.branch_var_threshold:
            # Give a small Q-reward to 'branch' to encourage exploration
            self.policy.update("branch", 0.1)

        return {
            "count": len(nodes),
            "depth_max": max(n.depth for n in nodes),
            "Φ_mean": float(np.mean(phis)),
            "Φ_var": var,
            "entropy": self.policy.entropy(self.temperature),
            "policy": dict(self.policy.Q),
        }

    def to_dict(self: 'GrowthMind') -> dict:
        return {
            "policy": dict(self.policy.Q),
            "journal": list(self.journal),
            "nodes": [
                {"depth": n.depth, "rule": n.rule, "Φ": n.polarity,
                 "children": [id(c) for c in n.children]}
                for n in self.traverse_bfs()
            ],
        }

    def reflect(self: 'GrowthMind'):
        """Print internal mind metrics for introspection."""
        stats = self.stats()
        policy = self.policy.softmax_probs(self.temperature) # Now uses the dynamic temperature
        print(f"\n🧠 GrowthMind Reflection:")
        print(f"   Nodes: {stats['count']}, depth_max={stats.get('depth_max', 0)}")
        print(f"   Φ̄={stats['Φ_mean']:+.3f}, σ²={stats['Φ_var']:.4f}, entropy={stats['entropy']:.3f}")
        print(f"   Policy → " + ", ".join(f"{k}:{policy[k]:.2f}" for k in policy))
        print(f"   Dynamic Temp: {self.temperature:.3f}") # Added for debugging

    def realign(self: 'GrowthMind', phi_sign: float, neutral_band: float):
        """Update internal calibration parameters."""
        self.phi_sign = phi_sign
        self.neutral_band = neutral_band
        self.journal.append({
            "t": time.time(),
            "event": "realign",
            "phi_sign": phi_sign,
            "neutral_band": neutral_band
        })