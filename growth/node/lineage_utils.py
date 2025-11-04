"""
lineage_utils.py
----------------
Utility functions for describing and tracing GrowthNode ancestry.
"""

from typing import List
from core_node import GrowthNode

def describe_node(node: GrowthNode) -> str:
    return f"<GrowthNode depth={node.depth} rule={node.rule} Φ={node.polarity:+.3f} children={len(node.children)}>"

def trace_lineage(node: GrowthNode) -> List[GrowthNode]:
    path = []
    while node:
        path.append(node)
        node = node.parent
    return list(reversed(path))
