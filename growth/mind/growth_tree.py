# --- File: growth/growth_tree.py ---
from __future__ import annotations
from typing import Dict, TYPE_CHECKING
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

# Assuming AuditLog and LatticeState are in core.*
from core.audit import AuditLog
from core.lattice import LatticeState
from growth.node.core_node import GrowthNode


# -------------------------------------------------------------------
# TensorBoard Writer Initialization
# -------------------------------------------------------------------
TREE_TENSORBOARD_WRITER = SummaryWriter(
    f"runs/tree_growth_{time.strftime('%Y%m%d-%H%M%S')}"
)

# -------------------------------------------------------------------
# GrowthTree Class
# -------------------------------------------------------------------
class GrowthTree:
    """
    Manages the collection of GrowthNodes for a single search iteration.
    Provides O(1) lookup by node ID and maintains hierarchical structure.
    TensorBoard logs track node growth and mean depth evolution.
    """

    def __init__(self, root_node: GrowthNode):
        self.root = root_node
        # Dictionary to store all nodes for fast lookup by ID
        self.nodes: Dict[int, GrowthNode] = {root_node.node_id: root_node}
        self._node_counter = 1
        self._depth_sum = 0

    # ---------------------------------------------------------------
    # Factory Method
    # ---------------------------------------------------------------
    @staticmethod
    def create() -> 'GrowthTree':
        """
        Creates the initial root node and a new GrowthTree instance for a search.
        Ensures the root node has a NumPy array for 'weights' to prevent logging errors.
        """

        class SimpleStatePlaceholder:
            def __init__(self):
                # Ensure 'weights' exists for downstream logging
                self.weights = np.array([0.0])

        root_state = SimpleStatePlaceholder()
        root_node = GrowthNode(
            state=root_state,
            polarity=0.0,
            depth=0,
            rule="INIT:start",
            note="Root of search tree",
            parent=None,
            log=AuditLog(),
            node_id=0,
            parent_id=-1,
            search_state="GENERATED",
        )
        return GrowthTree(root_node)

    # ---------------------------------------------------------------
    # Node Management
    # ---------------------------------------------------------------
    def add_node(self, child_node: GrowthNode, parent_id: int):
        """
        Registers a new node, links it to its parent,
        and logs structural metrics to TensorBoard.
        """
        self.nodes[child_node.node_id] = child_node
        parent_node = self.nodes.get(parent_id)

        if parent_node is None:
            # Log missing-parent event silently
            TREE_TENSORBOARD_WRITER.add_scalar(
                "Tree/Orphan_Node_Event", 1, self._node_counter
            )
            return

        parent_node.children.append(child_node)

        # --- Update Stats ---
        self._node_counter += 1
        self._depth_sum += child_node.depth
        avg_depth = self._depth_sum / self._node_counter

        # --- TensorBoard Logging ---
        TREE_TENSORBOARD_WRITER.add_scalar("Tree/Node_Count", self._node_counter, self._node_counter)
        TREE_TENSORBOARD_WRITER.add_scalar("Tree/Average_Depth", avg_depth, self._node_counter)

        # Optional: count leaf ratio to monitor pruning vs. expansion
        leaf_count = sum(1 for n in self.nodes.values() if not n.children)
        TREE_TENSORBOARD_WRITER.add_scalar("Tree/Leaf_Ratio", leaf_count / max(1, self._node_counter), self._node_counter)

        # Ensure TensorBoard stays live-updated
        TREE_TENSORBOARD_WRITER.flush()
