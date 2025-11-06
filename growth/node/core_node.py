# ================================================================
# Livnium Growth — core_node.py (Throttled Print Mode)
# ================================================================
# Defines the GrowthNode: the atomic cognitive unit of Livnium Growth.
# Prints only once every 1000 node creations to track large-scale runs.
# ================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import time

# --- Core Livnium Imports ---
from core.lattice import LatticeState
from core.audit import AuditLog


# -------------------------------------------------------------------
# Global diagnostic state
# -------------------------------------------------------------------
_NODE_COUNTER = 0
_PRINT_INTERVAL = 1000  # print every Nth node


def _log_every_1000(msg: str):
    """Prints a throttled message exactly every 1000th node."""
    global _NODE_COUNTER
    if _NODE_COUNTER % _PRINT_INTERVAL == 0 and _NODE_COUNTER != 0:
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] {msg}")


# -------------------------------------------------------------------
# GrowthNode Definition
# -------------------------------------------------------------------
@dataclass
class GrowthNode:
    # --- Core Data Payload ---
    state: LatticeState
    polarity: float = 0.0
    note: str = ""

    # --- Structural / Persistent Identifiers ---
    node_id: int = field(default=-1)
    parent_id: int = field(default=-1)

    # --- In-Memory Traversal & Tree Links ---
    parent: Optional["GrowthNode"] = None
    children: List["GrowthNode"] = field(default_factory=list)

    # --- Search / Heuristic Data ---
    depth: int = 0
    rule: str = "origin"
    score: Optional[float] = None
    search_state: str = "GENERATED"

    # --- Auditing ---
    log: AuditLog = field(default_factory=AuditLog)

    # ----------------------------------------------------------------
    # Post-initialization hook (throttled print)
    # ----------------------------------------------------------------
    def __post_init__(self):
        global _NODE_COUNTER
        _NODE_COUNTER += 1

        # Print only when count hits exact multiples of 1000
        _log_every_1000(
            f"🧩 {_NODE_COUNTER:,} GrowthNodes created | "
            f"latest → id={self.node_id}, rule='{self.rule}', "
            f"Φ={self.polarity:+.3f}, depth={self.depth}, state={self.search_state}"
        )

    # ----------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------
    def __repr__(self):
        return (
            f"<GrowthNode id={self.node_id} parent={self.parent_id} "
            f"rule={self.rule} Φ={self.polarity:+.2f} state={self.search_state}>"
        )
