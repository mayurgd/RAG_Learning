import math
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np


def create_circular_graph():

    def determine_text_position(angle):
        if 0 <= angle < math.radians(75):
            return "top right"
        elif math.radians(75) <= angle < math.radians(105):
            return "top center"
        elif math.radians(105) <= angle < math.radians(135):
            return "top left"
        elif math.radians(135) <= angle < math.radians(215):
            return "middle left"
        elif math.radians(215) <= angle < math.radians(255):
            return "bottom left"
        elif math.radians(255) <= angle < math.radians(285):
            return "bottom center"
        elif math.radians(285) <= angle < math.radians(315):
            return "bottom right"
        else:  # 315 - 360 degrees
            return "middle right"

    # # Create a dataframe from the given data
    # # If using your full dataframe, replace this with your actual dataframe
    # data = {
    #     'EFX_LEGSUBNAMEALL': ['CIS', 'DOBLIN INC', 'HEAT INDUSTRIES', 'CIS', 'HEAT INDUSTRIES'],
    #     'EFX_LEGDOMULTNAMEALL': ['DELOITTE US DENVER', 'DELOITTE US DENVER', 'DELOITTE US DENVER', 'DELOITTE US DENVER', 'DELOITTE US DENVER'],
    #     'EFX_AFFLULTNAMEALL': ['DELOITTE TOUCHE TOHMATSU LIMITED', 'DELOITTE TOUCHE TOHMATSU LIMITED', 'DELOITTE TOUCHE TOHMATSU LIMITED', 'DELOITTE TOUCHE TOHMATSU LIMITED', 'DELOITTE TOUCHE TOHMATSU LIMITED']
    # }
    # df = pd.DataFrame(data)

    # For the full dataframe, use your variable name instead
    df = pd.read_csv(r"EFX_dataset25.csv")
    df = df[["EFX_LEGSUBNAMEALL", "EFX_LEGDOMULTNAMEALL", "EFX_AFFLULTNAMEALL"]]
    df = df.drop_duplicates()
    df = df[df["EFX_LEGDOMULTNAMEALL"] != df["EFX_LEGSUBNAMEALL"]]

    # Get unique values for each level
    affil_names = df["EFX_AFFLULTNAMEALL"].unique()
    legdom_names = df["EFX_LEGDOMULTNAMEALL"].unique()
    legsub_names = df["EFX_LEGSUBNAMEALL"].unique()

    # Create a graph
    G = nx.DiGraph()

    # Add the nodes with level attributes
    for affil in affil_names:
        G.add_node(affil, level=0)
    for legdom in legdom_names:
        G.add_node(legdom, level=1)
    for legsub in legsub_names:
        G.add_node(legsub, level=2)

    # Add edges from affiliation to legal domain
    for affil in affil_names:
        for legdom in legdom_names:
            if (
                df[
                    (df["EFX_AFFLULTNAMEALL"] == affil)
                    & (df["EFX_LEGDOMULTNAMEALL"] == legdom)
                ].shape[0]
                > 0
            ):
                G.add_edge(affil, legdom)

    # Add edges from legal domain to legal sub
    legdom_to_legsub = {}
    for legdom in legdom_names:
        legdom_to_legsub[legdom] = []
        for legsub in sorted(legsub_names):  # Sorting legal sub names alphabetically
            if (
                df[
                    (df["EFX_LEGDOMULTNAMEALL"] == legdom)
                    & (df["EFX_LEGSUBNAMEALL"] == legsub)
                ].shape[0]
                > 0
            ):
                G.add_edge(legdom, legsub)
                legdom_to_legsub[legdom].append(legsub)

    # Get all nodes from the graph
    all_nodes = list(G.nodes())

    # Sort leaf nodes (nodes with no outgoing edges) alphabetically
    leaf_nodes = sorted([node for node in all_nodes if G.out_degree(node) == 0])

    # Custom positioning
    pos = {}
    node_textposition = {}  # Store text positions for each node

    # Root node (level 0) on the left
    root_nodes = [n for n in all_nodes if G.nodes[n]["level"] == 0]
    for i, node in enumerate(root_nodes):
        pos[node] = (-4.5, 0.5)
        node_textposition[node] = "top right"  # Right of the root node

    # Middle nodes (level 1) slightly to the right
    middle_nodes = [n for n in all_nodes if G.nodes[n]["level"] == 1]
    middle_y_spacing = 2.0  # Increased spacing between middle nodes
    for i, node in enumerate(middle_nodes):
        y_pos = i * middle_y_spacing - (len(middle_nodes) * middle_y_spacing / 2)
        pos[node] = (2, y_pos)
        node_textposition[node] = "top right"  # Default for middle nodes

    # Position leaf nodes in circles around their respective middle nodes
    for middle_node in middle_nodes:
        # Get leaf nodes connected to this middle node
        leaves = legdom_to_legsub.get(middle_node, [])
        if not leaves:
            continue

        # Increase radius based on number of leaves
        num_leaves = len(leaves)
        base_radius = 1.5  # Base radius
        # Dynamically adjust radius based on number of leaves
        circle_radius = base_radius + (0.1 * num_leaves)

        # Calculate positions in a circle
        angle_step = 2 * math.pi / num_leaves
        middle_x, middle_y = pos[middle_node]

        for i, leaf in enumerate(leaves):
            # Distribute leaves evenly in a circle
            angle = i * angle_step

            # Add small random variation to prevent exact overlaps
            jitter = 0.05
            rand_offset_x = np.random.uniform(-jitter, jitter)
            rand_offset_y = np.random.uniform(-jitter, jitter)

            x = middle_x + circle_radius * math.cos(angle) + rand_offset_x
            y = middle_y + circle_radius * math.sin(angle) + rand_offset_y
            pos[leaf] = (x, y)

            node_textposition[leaf] = determine_text_position(angle)

    # Make sure all nodes have positions (fix for KeyError)
    for node in all_nodes:
        if node not in pos:
            # Assign a default position for any missed nodes
            pos[node] = (4, 0)  # Off to the right side
            node_textposition[node] = "top right"  # Default position

    # Extract node positions for visualization
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_positions = []  # List to store text positions

    for node in all_nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_positions.append(node_textposition[node])

        # Assign colors and sizes based on level
        level = G.nodes[node]["level"]
        if level == 0:
            node_color.append("red")
            node_size.append(30)  # Larger size for root
        elif level == 1:
            node_color.append("orange")
            node_size.append(20)  # Medium size for middle
        else:
            node_color.append("blue")
            node_size.append(10)  # Smaller size for leaves

    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace with appropriate text positioning
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition=node_positions,  # Use the clock-based positioning
        marker=dict(color=node_color, size=node_size, line=dict(width=2)),
    )

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="black",
        ),
    )

    fig.update_layout(
        title="Organizational Hierarchy",
        font=dict(size=12),
        height=1200,
        width=1200,
    )

    return fig


df = pd.read_csv(r"EFX_dataset25.csv")
df = df[["EFX_LEGSUBNAMEALL", "EFX_LEGDOMULTNAMEALL", "EFX_AFFLULTNAMEALL"]]
df = df.drop_duplicates()
df = df[df["EFX_LEGDOMULTNAMEALL"] != df["EFX_LEGSUBNAMEALL"]]

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row["EFX_AFFLULTNAMEALL"], row["EFX_LEGDOMULTNAMEALL"])
    G.add_edge(row["EFX_LEGDOMULTNAMEALL"], row["EFX_LEGSUBNAMEALL"])


# Function to generate cubic Bezier curves
def cubic_bezier_curve(x0, y0, x1, y1, num_points=30):
    cx1, cy1 = (x0 + x1) / 2, y0 + np.random.uniform(-30, 30)
    cx2, cy2 = (x0 + x1) / 2, y1 + np.random.uniform(-30, 30)

    t = np.linspace(0, 1, num_points)
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


# Function to create left-to-right tree graph
def create_tree_graph():
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        smooth_x, smooth_y = cubic_bezier_curve(x0, y0, x1, y1)
        edge_x.extend(smooth_x.tolist() + [None])
        edge_y.extend(smooth_y.tolist() + [None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_texts = list(G.nodes())
    text_positions = [
        (
            "top center"
            if G.in_degree(node) == 0 or G.out_degree(node) > 0
            else "bottom center"
        )
        for node in G.nodes()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="white"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=10, color="lightblue"),
            text=node_texts,
            textposition=text_positions,
        )
    )
    fig.update_layout(
        title="Left-to-Right Hierarchy Graph",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=1000,
        plot_bgcolor="black",
    )
    return fig


# Function to create circular graph with leaf nodes
def create_circular_graph_old():
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_texts = list(G.nodes())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="white"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=10, color="lightblue"),
            text=node_texts,
            textposition="middle center",
        )
    )
    fig.update_layout(
        title="Circular Graph with Leaf Nodes",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=1000,
        plot_bgcolor="black",
    )
    return fig


# Streamlit UI
st.title("Network Graph Visualization")
view_option = st.selectbox(
    "Select Graph Type",
    ["Left-to-Right Tree", "Circular Leaf Nodes"],
)

# Show appropriate graph
if view_option == "Left-to-Right Tree":
    st.plotly_chart(create_tree_graph(), use_container_width=True)
else:
    st.plotly_chart(create_circular_graph(), use_container_width=True)
