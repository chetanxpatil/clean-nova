"""
Livnium Growth — core_node.py
=============================
Defines the GrowthNode: the atomic cognitive cell of Livnium Growth.
Handles lattice evolution via rule-based transformations (merge, branch, stabilize, revert).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from core.lattice import LatticeState, rebalance
    from core.semantic import IntentVector, compute_intent
    from core.audit import AuditLog
    from growth.rules import apply_rule, GrowthResult
except ImportError:
    from core.lattice import LatticeState  # type: ignore
    from core.semantic import IntentVector, compute_intent  # type: ignore
    from core.audit import AuditLog  # type: ignore
    from growth.rules import apply_rule, GrowthResult  # type: ignore


@dataclass
class GrowthNode:
    """A single node representing one cognitive lattice state in the growth hierarchy."""
    state: LatticeState
    polarity: float = 0.0
    depth: int = 0
    rule: str = "origin"
    note: str = ""
    parent: Optional["GrowthNode"] = None
    children: List["GrowthNode"] = field(default_factory=list)
    log: AuditLog = field(default_factory=AuditLog)

    # ----------------------------------------------------------
    # Core growth transformations
    # ----------------------------------------------------------
    def merge_with(self, other: "GrowthNode") -> "GrowthNode":
        intent = compute_intent(self.state, other.state)
        result, log = apply_rule("merge", self.state, other.state, intent=intent, log=self.log)
        node = GrowthNode(result.new_state, result.new_polarity, self.depth + 1,
                          rule=result.rule, note=result.note, parent=self, log=log)
        self.children.append(node)
        return node

    def branch(self) -> "GrowthNode":
        intent = compute_intent(self.state, self.state, observer="Lo")
        intent.polarity *= -1.0
        result, log = apply_rule("branch", self.state, intent, log=self.log)
        node = GrowthNode(result.new_state, result.new_polarity, self.depth + 1,
                          rule=result.rule, note=result.note, parent=self, log=log)
        self.children.append(node)
        return node

    def stabilize(self) -> "GrowthNode":
        result, log = apply_rule("stabilize", self.state, log=self.log)
        node = GrowthNode(result.new_state, result.new_polarity, self.depth + 1,
                          rule=result.rule, parent=self, log=log)
        self.children.append(node)
        return node

    def revert(self) -> "GrowthNode":
        if not self.parent:
            raise ValueError("Cannot revert — node has no parent.")
        result, log = apply_rule("revert", self.state, self.parent.state, log=self.log)
        node = GrowthNode(result.new_state, result.new_polarity, self.depth + 1,
                          rule=result.rule, parent=self, log=log)
        self.children.append(node)
        return node
