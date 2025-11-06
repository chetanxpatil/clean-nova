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
        # 🟢 CRITICAL FIX: Check if the root is initialized before traversal.
        if not hasattr(self, 'root') or self.root is None:
            return  # Return an empty iterable immediately

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
            # Ensures all expected keys are returned with safe defaults
            return {
                "count": 0,
                "depth_max": 0,
                "Φ_mean": 0.0,
                "Φ_var": 0.0,
                "entropy": 0.0,
                "policy": {},
            }
        phis = [n.polarity for n in nodes]
        var = float(np.var(phis))

        # Assumes branch_var_threshold and policy exist on self
        if hasattr(self, 'branch_var_threshold') and var < self.branch_var_threshold:
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
        """Print internal mind metrics for introspection. Must be robust to empty data."""

        # This explicit check here is now redundant but kept for safety,
        # as the traversal method now handles the empty root case first.

        stats = self.stats()

        if stats["count"] == 0:
            print(f"\n🧠 GrowthMind Reflection: No nodes found in tree.")
            return

        # Ensures policy metrics are calculated correctly even if Q is small
        policy = self.policy.softmax_probs(self.temperature)
        print(f"\n🧠 GrowthMind Reflection:")
        print(f"   Nodes: {stats['count']}, depth_max={stats.get('depth_max', 0)}")
        print(f"   Φ̄={stats['Φ_mean']:+.3f}, σ²={stats['Φ_var']:.4f}, entropy={stats['entropy']:.3f}")
        print(f"   Policy → " + ", ".join(f"{k}:{policy[k]:.2f}" for k in policy))
        print(f"   Dynamic Temp: {self.temperature:.3f}")