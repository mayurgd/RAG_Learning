import plotly.graph_objects as go
import networkx as nx
import pandas as pd
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

# Generate hierarchical layout (Top to Bottom)
pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")

# Identify leaf nodes and sort them alphabetically
leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
leaf_nodes = sorted(leaf_nodes)

# Assign levels based on y-coordinates
levels = {y: [] for x, y in pos.values()}
for node, (x, y) in pos.items():
    levels[y].append(node)

# Assign different sizes and colors per level
level_sizes = {y: 10 + 5 * i for i, y in enumerate(sorted(levels.keys()))}
level_colors = {
    y: f"rgb({100 + i * 50}, {50 + i * 30}, {150 - i * 30})"
    for i, y in enumerate(sorted(levels.keys()))
}

# Extract edge coordinates
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])  # Straight lines
    edge_y.extend([y0, y1, None])

# Extract node coordinates and labels
node_x = []
node_y = []
node_texts = []
node_text_positions = []
node_sizes = []
node_colors = []
annotations = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_sizes.append(level_sizes[y])
    node_colors.append(level_colors[y])

    if node in leaf_nodes:  # Ordered leaf nodes alphabetically
        annotations.append(
            dict(
                x=x - 5,
                y=y - 5,  # Shift text further below node
                text=node,  # Full name in annotation
                showarrow=False,
                textangle=30,
                font=dict(size=10),
            )
        )
    else:
        node_texts.append(node)  # Full name for non-leaf nodes
        node_text_positions.append("top right")

# Create figure
fig = go.Figure()

# Add edges as straight lines
fig.add_trace(
    go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="white"),
        hoverinfo="none",
    )
)

# Add nodes
fig.add_trace(
    go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if node_texts else "markers",
        marker=dict(size=node_sizes, color=node_colors),
        text=node_texts,  # Show full name for non-leaf nodes
        textposition=node_text_positions,
        hoverinfo="text",  # Show node name on hover
        hovertext=[node for node in G.nodes()],  # Full name on hover
    )
)

# Add annotations for leaf nodes
fig.update_layout(
    title="Top-Down Hierarchy Graph",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    annotations=annotations,
    height=1000,
)


# Streamlit UI
st.set_page_config(layout="wide")
st.title("Organizational Hierarchy Visualization")
st.plotly_chart(fig, use_container_width=True)
