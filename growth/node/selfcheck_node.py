"""
selfcheck_node.py
-----------------
Quick self-test for GrowthNode lineage and visualization.
"""

from core.lattice import canonical_symbol_layout
from core_node import GrowthNode
from lineage_utils import trace_lineage, describe_node
from visualize_tree import visualize_growth_tree

if __name__ == "__main__":
    base = canonical_symbol_layout()
    root = GrowthNode(base, polarity=1.0)
    n1 = root.merge_with(root)
    n2 = n1.branch()
    n3 = n2.stabilize()
    n4 = n3.revert()

    print("Lineage Trace:")
    for n in trace_lineage(n4):
        print(" ", describe_node(n))

    print("\nRendering growth tree...")
    visualize_growth_tree(root, title="Livnium Cognitive Growth Tree")
    print("\nGrowthNode self-check passed ✓")
