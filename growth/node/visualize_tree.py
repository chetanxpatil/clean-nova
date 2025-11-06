"""
visualize_tot.py
----------------
Fully 3D Interactive Visualization Tool (Plotly for both ToT structure and Lattice State),
with solution path highlighting (bright red) and 1000-node subset view.
"""
import json
import matplotlib
# Use TkAgg for compatibility with NetworkX plotting backend.
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from typing import List, Dict, Any

# --- PLOTLY IMPORT ---
import plotly.graph_objects as go
from plotly.offline import plot
# ---------------------

# --- CONFIGURATION ---
JOURNAL_FILE = '/Users/chetanpatil/Desktop/clean-nova/brain/growth_journal.jsonl'
NODE_SIZE_BASE = 1500
NODE_SIZE_SCALE = 2000
LATTICE_DIMS = (3, 3, 3)
# ---------------------


def load_journal_data(filepath: str) -> List[Dict[str, Any]]:
    """Loads journal data from a JSON Lines (.jsonl) file, printing every 1000 lines processed."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if i % 1000 == 0:
                    print(f"🔄 Processed {i:,} journal lines...")

        if not data:
            with open(filepath, 'r') as f:
                data = json.load(f)

    except FileNotFoundError:
        print(f"Error: Journal file not found at '{filepath}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filepath}': {e}")
        print("Please ensure the file is valid JSON (or JSON Lines format).")
        sys.exit(1)

    print(f"📘 Loaded {len(data):,} total journal entries.")
    return [entry for entry in data if 'node_id' in entry and 'parent_id' in entry]


def _get_solution_path_nodes(journal_data, solution_node_id):
    """Traces the path backward from the solution to the root."""
    path_ids = set()
    node_map = {entry['node_id']: entry['parent_id'] for entry in journal_data}

    current_id = solution_node_id
    while current_id is not None and current_id != -1:
        if current_id in path_ids:
            # Prevent infinite loop if graph contains cycles
            break
        path_ids.add(current_id)
        current_id = node_map.get(current_id)
    return path_ids


def visualize_lattice_state_3d(phi_field_flat: List[float], node_id: int):
    """Renders the 3x3x3 lattice state as an interactive 3D scatter plot using Plotly."""
    try:
        phi_field = np.array(phi_field_flat).reshape(LATTICE_DIMS)
    except ValueError:
        print("⚠️ Warning: Could not reshape phi_field. Data missing or incorrect size.")
        return

    x, y, z = np.meshgrid(np.arange(LATTICE_DIMS[0]),
                          np.arange(LATTICE_DIMS[1]),
                          np.arange(LATTICE_DIMS[2]))
    x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
    phi_flat = phi_field.flatten()
    vmax = np.max(np.abs(phi_flat))

    hover_text = [f"Coord: ({xi}, {yi}, {zi})<br>Polarity (Φ): {phi:.4f}"
                  for xi, yi, zi, phi in zip(x_flat, y_flat, z_flat, phi_flat)]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_flat, y=y_flat, z=z_flat,
            mode='markers',
            marker=dict(
                size=10 + (np.abs(phi_flat) / vmax * 20),
                color=phi_flat,
                colorscale='RdBu',
                cmin=-vmax,
                cmax=vmax,
                colorbar=dict(title='Polarity', len=0.6),
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (Dim 1)', tickvals=np.arange(LATTICE_DIMS[0])),
            yaxis=dict(title='Y (Dim 2)', tickvals=np.arange(LATTICE_DIMS[1])),
            zaxis=dict(title='Z (Dim 3)', tickvals=np.arange(LATTICE_DIMS[2])),
            aspectmode='cube'
        ),
        title=f"Interactive 3D Lattice State (Node ID: {node_id})",
        margin=dict(r=0, l=0, b=0, t=0)
    )
    fig.show()


def visualize_from_journal_data(journal_data: List[Dict[str, Any]], title: str = "GrowthMind 3D Tree of Thought"):
    """Builds and renders the interactive 3D ToT graph using Plotly with highlighted solution path."""
    if not journal_data:
        print("No valid step data found to visualize.")
        return

    node_data_map = {}
    temp_G = nx.DiGraph()
    edge_x, edge_y, edge_z = [], [], []

    RULE_COLORS = {
        'merge': 'green', 'branch': 'red', 'stabilize': 'yellow',
        'revert': 'gray', 'origin': 'blue'
    }

    print(f"Building 3D graph structure with {len(journal_data):,} nodes...")

    for entry in journal_data:
        node_id = entry['node_id']
        parent_id = entry['parent_id']
        temp_G.add_node(node_id,
                        rule=entry.get('rule', 'origin').split(':')[-1],
                        depth=entry.get('depth', 0),
                        phi=entry.get('Φ', 0.0))
        if parent_id != -1 and parent_id in node_data_map:
            temp_G.add_edge(parent_id, node_id)
        node_data_map[node_id] = entry

    # --- Find and trace the best solution path ---
    best_node_id = max(journal_data, key=lambda x: x.get('h_score', -np.inf))['node_id']
    solution_path_ids = _get_solution_path_nodes(journal_data, best_node_id)
    print(f"🔍 Highlighting solution path with {len(solution_path_ids)} nodes (best node ID: {best_node_id})")

    pos = nx.spectral_layout(temp_G, dim=3)
    node_trace_data = {'x': [], 'y': [], 'z': [], 'marker_color': [], 'text': [], 'node_id': []}

    for node_id, coords in pos.items():
        data = node_data_map[node_id]
        rule = data['rule'].split(':')[-1]
        phi = data['Φ']

        # Highlight solution path in red
        color = 'red' if node_id in solution_path_ids else RULE_COLORS.get(rule, 'black')

        node_trace_data['x'].append(coords[0])
        node_trace_data['y'].append(coords[1])
        node_trace_data['z'].append(coords[2])
        node_trace_data['marker_color'].append(color)
        node_trace_data['text'].append(f"ID:{node_id} | Rule:{rule}<br>Φ:{phi:+.3f} | Depth:{data['depth']}")
        node_trace_data['node_id'].append(node_id)

        parent_id = data.get('parent_id')
        if parent_id != -1 and parent_id in node_data_map and parent_id in pos:
            parent_coords = pos[parent_id]

            # Red edges if both nodes are on solution path
            if node_id in solution_path_ids and parent_id in solution_path_ids:
                edge_color = 'rgba(255,0,0,0.8)'
            else:
                edge_color = 'rgba(150,150,150,0.4)'

            edge_x.extend([parent_coords[0], coords[0], None])
            edge_y.extend([parent_coords[1], coords[1], None])
            edge_z.extend([parent_coords[2], coords[2], None])

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                              line=dict(width=2, color='rgba(150,150,150,0.5)'),
                              hoverinfo='none', mode='lines', name='Edges')

    max_abs_phi = np.max([abs(data['Φ']) for data in node_data_map.values()]) if node_data_map else 0.1

    node_trace = go.Scatter3d(x=node_trace_data['x'], y=node_trace_data['y'], z=node_trace_data['z'],
                              mode='markers',
                              hovertext=node_trace_data['text'],
                              hoverinfo='text',
                              marker=dict(symbol='circle',
                                          size=[10 + 20 * (abs(node_data_map[nid]['Φ']) / max_abs_phi)
                                                for nid in node_trace_data['node_id']],
                                          color=node_trace_data['marker_color'],
                                          line=dict(color='black', width=1)))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False,
                      scene=dict(xaxis=dict(showticklabels=False, title=''),
                                 yaxis=dict(showticklabels=False, title=''),
                                 zaxis=dict(showticklabels=False, title='Depth')))

    print("\n🌐 Opening 3D ToT Structure in browser with solution path highlighted in red.")
    plot(fig, auto_open=True)


def main():
    print("--- GrowthMind ToT Visualizer ---")

    journal_data = load_journal_data(JOURNAL_FILE)
    if not journal_data:
        print("Visualization aborted due to no valid data.")
        return

    total_steps = len(journal_data)
    print(f"Loaded {total_steps:,} valid steps.")
    subset_data = journal_data[:1000]
    print(f"📊 Visualizing only the first {len(subset_data):,} steps out of {len(journal_data):,} total.")
    visualize_from_journal_data(subset_data, title=f"3D ToT Structure (First {len(subset_data):,} Steps)")


if __name__ == "__main__":
    main()
