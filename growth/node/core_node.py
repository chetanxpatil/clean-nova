"""
Livnium Growth — core_node.py
=============================
Defines the GrowthNode: the atomic cognitive cell of Livnium Growth.
This is a passive data structure managed by an external orchestrator.
(Refactored based on Section 3.3 & 5 of the architectural doc)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from core.lattice import LatticeState
    from core.audit import AuditLog
except ImportError:
    from core.lattice import LatticeState  # type: ignore
    from core.audit import AuditLog  # type: ignore


@dataclass
class GrowthNode:
    """
    A passive data structure representing one cognitive lattice state.
    It is a simple container for data, managed by an external orchestrator.
    """

    # --- Core Data Payload ---
    state: LatticeState
    polarity: float = 0.0
    note: str = ""

    # --- Structural/Persistent Identifiers (from Section 3.2) ---
    node_id: int = field(default=-1)
    parent_id: int = field(default=-1)

    # --- In-Memory Traversal & Tree Links ---
    parent: Optional["GrowthNode"] = None
    children: List["GrowthNode"] = field(default_factory=list)

    # --- Search & Heuristic Data (from Section 3.3 & 5.1) ---
    depth: int = 0
    rule: str = "origin"  # The rule that *created* this node
    search_state: str = "GENERATED" # e.g., GENERATED, EXPANDED, PRUNED

    # --- NEW A* SEARCH SCORES (Section 5) ---
    g_score: float = float('inf')  # The g(n) - cost from start to this node
    h_score: float = float('inf')  # The h(n) - estimated cost to goal (our Q-value)
    # (The old 'score' field is now replaced by g_score and h_score)

    # --- Auditing ---
    log: AuditLog = field(default_factory=AuditLog)

    def __repr__(self):
        """Provides a clean representation for logging and debugging."""
        # Calculate f_score for display, handling inf
        f_score_str = "inf"
        if self.g_score != float('inf') and self.h_score != float('inf'):
            f_score = self.g_score + self.h_score
            f_score_str = f"{f_score:+.2f}"

        return (f"<GrowthNode id={self.node_id} parent={self.parent_id} "
                f"rule={self.rule} Φ={self.polarity:+.2f} "
                f"f(n)={f_score_str} state={self.search_state}>")

    # --- ALL RULE-APPLICATION METHODS HAVE BEEN REMOVED ---
    # (merge_with, branch, stabilize, revert)
    #
    # This logic is now externalized to the SearchOrchestrator,
    # which fixes the "single-active-node" flaw and enables
    # true, decoupled tree search (Section 2.1 & 4.2).