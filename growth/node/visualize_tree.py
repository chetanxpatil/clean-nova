"""
visualize_tree.py
-----------------
NetworkX + Matplotlib visualization of the cognitive growth hierarchy.
"""

import matplotlib.pyplot as plt
import networkx as nx
from core_node import GrowthNode

def visualize_growth_tree(root: GrowthNode, title="Growth Tree"):
    G = nx.DiGraph()

    def add_edges(node):
        label = f"{node.rule}\nΦ={node.polarity:+.2f}\nD={node.depth}"
        G.add_node(id(node), label=label)
        for child in node.children:
            G.add_edge(id(node), id(child))
            add_edges(child)

    add_edges(root)
    pos = nx.spring_layout(G, k=0.8, iterations=200, seed=42)
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos, with_labels=True, labels=labels,
        node_color='lightblue', node_size=1500,
        font_size=8, font_weight='bold', arrows=True, edge_color='gray'
    )
    plt.title(title)
    plt.axis('off')
    plt.show()
