import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st


df = pd.read_csv(r"EFX_dataset25.csv")
df = df[["EFX_LEGSUBNAMEALL", "EFX_LEGDOMULTNAMEALL", "EFX_AFFLULTNAMEALL"]]
df = df.drop_duplicates()

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row["EFX_AFFLULTNAMEALL"], row["EFX_LEGDOMULTNAMEALL"])
    G.add_edge(row["EFX_LEGDOMULTNAMEALL"], row["EFX_LEGSUBNAMEALL"])

# Generate hierarchical layout (Left to Right)
pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")


# Function to generate cubic Bezier curve
def cubic_bezier_curve(x0, y0, x1, y1, num_points=30):
    """Generates a smooth cubic Bézier curve with two control points"""
    cx1, cy1 = (x0 + x1) / 2, y0 + np.random.uniform(-30, 30)  # First control point
    cx2, cy2 = (x0 + x1) / 2, y1 + np.random.uniform(-30, 30)  # Second control point

    # Generate t values
    t = np.linspace(0, 1, num_points)

    # Cubic Bézier formula
    bx = (
        (1 - t) ** 3 * x0
        + 3 * (1 - t) ** 2 * t * cx1
        + 3 * (1 - t) * t**2 * cx2
        + t**3 * x1
    )
    by = (
        (1 - t) ** 3 * y0
        + 3 * (1 - t) ** 2 * t * cy1
        + 3 * (1 - t) * t**2 * cy2
        + t**3 * y1
    )

    return bx, by


# Extract smooth edge coordinates
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    smooth_x, smooth_y = cubic_bezier_curve(x0, y0, x1, y1)

    edge_x.extend(smooth_x.tolist() + [None])  # Add None for Plotly line breaks
    edge_y.extend(smooth_y.tolist() + [None])

# Extract node coordinates
node_x = []
node_y = []
node_texts = []
node_text_positions = []

# Determine node types for text positioning
for node in G.nodes():
    node_x.append(pos[node][0])
    node_y.append(pos[node][1])
    node_texts.append(node)

    if G.in_degree(node) == 0:  # Root node (No incoming edges)
        node_text_positions.append("top center")
    elif G.out_degree(node) > 0:  # Middle nodes
        node_text_positions.append("top center")
    else:  # Leaf nodes (No outgoing edges)
        node_text_positions.append("middle right")

# Create figure
fig = go.Figure()

# Add wavy edges with cubic Bézier curves
fig.add_trace(
    go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="white"),
        hoverinfo="none",
    )
)

# Add nodes with different text positions
fig.add_trace(
    go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=10, color="lightblue"),
        text=node_texts,
        textposition=node_text_positions,
    )
)

# Layout settings
fig.update_layout(
    title="Left-to-Right Hierarchy Graph with Cubic Bézier Curves",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="black",
    height=1000,
)

st.set_page_config(layout="wide")
st.title("Organizational Hierarchy Visualization")
st.plotly_chart(fig, use_container_width=True)
