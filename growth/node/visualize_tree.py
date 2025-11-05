"""
visualize_tot.py
----------------
Fully 3D Interactive Visualization Tool (Plotly for both ToT structure and Lattice State).
"""
import json
import matplotlib
# Use TkAgg for compatibility with NetworkX plotting backend.
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from typing import List, Dict, Any, Optional

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
    """Loads journal data from a JSON Lines (.jsonl) file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
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
    return [entry for entry in data if 'node_id' in entry and 'parent_id' in entry]


def visualize_lattice_state_3d(phi_field_flat: List[float], node_id: int):
    """
    Renders the 3x3x3 lattice state as an interactive 3D scatter plot using Plotly.
    The plot will open in your web browser, allowing rotation and zoom.
    """
    try:
        phi_field = np.array(phi_field_flat).reshape(LATTICE_DIMS)
    except ValueError:
        print("⚠️ Warning: Could not reshape phi_field. Data missing or incorrect size.")
        return

    # 1. Prepare Data for Scatter Plot
    x, y, z = np.meshgrid(np.arange(LATTICE_DIMS[0]),
                          np.arange(LATTICE_DIMS[1]),
                          np.arange(LATTICE_DIMS[2]))

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    phi_flat = phi_field.flatten()

    vmax = np.max(np.abs(phi_flat))

    hover_text = [f"Coord: ({xi}, {yi}, {zi})<br>Polarity (Φ): {phi:.4f}"
                  for xi, yi, zi, phi in zip(x_flat, y_flat, z_flat, phi_flat)]

    # 2. Create Plotly Figure
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
                colorbar=dict(title='Polarity ($\Phi$)', len=0.6),
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text'
        )
    ])

    # 3. Layout and Axes Configuration
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
    """
    Builds and renders the interactive 3D ToT graph using Plotly.
    """
    if not journal_data:
        print("No valid step data found to visualize.")
        return

    # 1. Prepare Data and Graphs
    node_data_map = {}
    temp_G = nx.DiGraph()

    # Store coordinates for Plotly
    coord_x, coord_y, coord_z = {}, {}, {}
    edge_x, edge_y, edge_z = [], [], []

    RULE_COLORS = {
        'merge': 'green', 'branch': 'red', 'stabilize': 'yellow',
        'revert': 'gray', 'origin': 'blue'
    }

    print(f"Building 3D graph structure with {len(journal_data)} nodes...")

    # A. Build the NetworkX graph and map data
    for entry in journal_data:
        node_id = entry['node_id']
        parent_id = entry['parent_id']

        # Build the graph structure for layout calculation
        temp_G.add_node(node_id,
                        rule=entry.get('rule', 'origin').split(':')[-1],
                        depth=entry.get('depth', 0),
                        phi=entry.get('Φ', 0.0)) # Store phi here for later size calculation

        if parent_id != -1 and parent_id in node_data_map:
            temp_G.add_edge(parent_id, node_id)

        node_data_map[node_id] = entry

    # B. Generate 3D Coordinates using NetworkX Layout (Only for nodes in the subset)
    pos = nx.spectral_layout(temp_G, dim=3)

    # C. Prepare Plotly Data
    node_trace_data = {'x': [], 'y': [], 'z': [], 'marker_color': [], 'text': [], 'node_id': []}

    for node_id, coords in pos.items():
        data = node_data_map[node_id]
        rule = data['rule'].split(':')[-1]
        phi = data['Φ']

        # 1. Node Data
        coord_x[node_id] = coords[0]
        coord_y[node_id] = coords[1]
        coord_z[node_id] = coords[2]

        node_trace_data['x'].append(coords[0])
        node_trace_data['y'].append(coords[1])
        node_trace_data['z'].append(coords[2])
        node_trace_data['marker_color'].append(RULE_COLORS.get(rule, 'black'))
        node_trace_data['text'].append(f"ID:{node_id} | Rule: {rule}<br>Φ:{phi:+.3f} | Depth:{data['depth']}")
        node_trace_data['node_id'].append(node_id)


        # 2. Edge Data
        parent_id = data.get('parent_id')
        if parent_id != -1 and parent_id in node_data_map:

            # --- FIX: Check if the parent_id is in the POS dictionary ---
            if parent_id in pos:
                parent_coords = pos[parent_id]
                parent_phi = node_data_map[parent_id]['Φ']
                # delta_phi = abs(phi - parent_phi) # (Not used in the trace, but useful for debugging)

                # Add line trace coordinates
                edge_x.extend([parent_coords[0], coords[0], None])
                edge_y.extend([parent_coords[1], coords[1], None])
                edge_z.extend([parent_coords[2], coords[2], None])
            # --- End Fix ---


    # 3. Create Plotly Figure
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                              line=dict(width=2, color='rgba(150,150,150,0.5)'),
                              hoverinfo='none', mode='lines', name='Edges')

    # Get max phi magnitude for sizing
    max_abs_phi = np.max([abs(data['Φ']) for data in node_data_map.values()]) if node_data_map else 0.1

    node_trace = go.Scatter3d(x=node_trace_data['x'], y=node_trace_data['y'], z=node_trace_data['z'],
                              mode='markers', # Removed text mode for cleaner 3D
                              hovertext=node_trace_data['text'],
                              hoverinfo='text',
                              marker=dict(symbol='circle',
                                          # Use stored phi magnitude for sizing
                                          size=[10 + 20 * (abs(node_data_map[nid]['Φ']) / max_abs_phi)
                                                for nid in node_trace_data['node_id']],
                                          color=node_trace_data['marker_color'],
                                          line=dict(color='black', width=1)))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False,
                      scene=dict(xaxis=dict(showticklabels=False, title=''),
                                 yaxis=dict(showticklabels=False, title=''),
                                 zaxis=dict(showticklabels=False, title='Depth')))

    print("\n🌐 Opening 3D ToT Structure in browser. Use the Plotly menu to rotate and zoom.")
    plot(fig, auto_open=True)


def main():
    """Main execution block to load data and run visualization."""
    print("--- GrowthMind ToT Visualizer ---")

    journal_data = load_journal_data(JOURNAL_FILE)
    if not journal_data:
        print("Visualization aborted due to no valid data.")
        return

    total_steps = len(journal_data)
    unique_parent_ids = set(entry['parent_id'] for entry in journal_data if entry.get('parent_id') != -1)
    if total_steps > 1 and len(unique_parent_ids) == total_steps - 1:
        print(f"⚠️ Warning: Detected {total_steps} steps with only {len(unique_parent_ids)} unique parents.")

    print(f"Loaded {total_steps} valid steps.")

    # Filter data for a small subset (e.g., first sample's path)
    subset_data = journal_data[:100]

    visualize_from_journal_data(subset_data,
                                title=f"3D ToT Structure (First {len(subset_data)} Steps)")


if __name__ == "__main__":
    main()