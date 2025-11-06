from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from core.lattice import LatticeState, canonical_symbol_layout
from growth.node.core_node import GrowthNode  # Assumes core_node is updated


@dataclass
class GrowthTree:
    """
    Orchestrator class that manages the collection of GrowthNodes.
    It holds the tree structure (the "source of truth").
    (Based on Section 4.1 of the architectural doc)
    """
    root: GrowthNode

    # The 'nodes' dictionary allows O(1) lookup of any node by its ID
    nodes: Dict[int, GrowthNode] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the node registry with the root node."""
        if self.root.node_id != 0:
            raise ValueError("Root node must be initialized with node_id=0.")
        self.nodes[self.root.node_id] = self.root

    @classmethod
    def create(cls) -> GrowthTree:
        """Creates a new tree with a fresh root node."""
        root_node = GrowthNode(
            state=canonical_symbol_layout(),
            node_id=0,
            parent_id=-1,
            polarity=0.0,
            depth=0,
            rule="origin"
        )
        return cls(root=root_node)

    def add_node(self, child_node: GrowthNode, parent_id: int) -> None:
        """
        Adds a new node to the tree and links it.
        """
        parent_node = self.nodes.get(parent_id)
        if not parent_node:
            raise ValueError(f"Parent node with id {parent_id} not found.")

        self.nodes[child_node.node_id] = child_node
        parent_node.children.append(child_node)
        child_node.parent = parent_node  # Link in-memory reference

    def get_node(self, node_id: int) -> Optional[GrowthNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)