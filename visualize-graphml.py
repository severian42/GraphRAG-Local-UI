import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Load the GraphML file
graph = nx.read_graphml('output/20240708-161630/artifacts/summarized_graph.graphml')

# Create a 3D spring layout with more separation
pos = nx.spring_layout(graph, dim=3, seed=42, k=0.5)

# Extract node positions
x_nodes = [pos[node][0] for node in graph.nodes()]
y_nodes = [pos[node][1] for node in graph.nodes()]
z_nodes = [pos[node][2] for node in graph.nodes()]

# Extract edge positions
x_edges = []
y_edges = []
z_edges = []

for edge in graph.edges():
    x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
    y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
    z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])

# Generate node colors based on a colormap
node_colors = [graph.degree(node) for node in graph.nodes()]
node_colors = np.array(node_colors)
node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())  # Normalize to [0, 1]

# Create the trace for edges
edge_trace = go.Scatter3d(
    x=x_edges, y=y_edges, z=z_edges,
    mode='lines',
    line=dict(color='lightgray', width=0.5),
    hoverinfo='none'
)

# Create the trace for nodes
node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(
        size=7,
        color=node_colors,
        colorscale='Viridis',  # Use a color scale for the nodes
        colorbar=dict(
            title='Node Degree',
            thickness=10,
            x=1.1,
            tickvals=[0, 1],
            ticktext=['Low', 'High']
        ),
        line=dict(width=1)
    ),
    text=[node for node in graph.nodes()],
    textposition="top center",
    textfont=dict(size=10, color='black'),
    hoverinfo='text'
)

# Create the 3D plot
fig = go.Figure(data=[edge_trace, node_trace])

# Update layout for better visualization
fig.update_layout(
    title='3D Graph Visualization',
    showlegend=False,
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            showarrow=False,
            text="Interactive 3D visualization of GraphML data",
            xref="paper",
            yref="paper",
            x=0,
            y=0
        )
    ]
)

# Show the plot
fig.show()
