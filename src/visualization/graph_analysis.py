import torch
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load graph
data_path = "outputs/visualizations/fraud_gnn_graph.pt"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

checkpoint = torch.load(data_path, weights_only=False)
data = checkpoint["data"]

edge_index = data.edge_index.numpy()
y = data.y.numpy()

# 🔥 Take subset (first 300 nodes)
num_nodes = 300

print(f"Building subgraph for the first {num_nodes} nodes...")
G = nx.DiGraph()

for i in range(edge_index.shape[1]):
    u, v = edge_index[0][i], edge_index[1][i]
    if u < num_nodes and v < num_nodes:
        # Avoid self-loops for cleaner visualization
        if u != v:
            G.add_edge(u, v)

# Node colors (fraud = red, normal = blue)
colors = []
for n in G.nodes():
    if n < len(y):
        colors.append("red" if y[n] == 1 else "royalblue")
    else:
        colors.append("gray") # Identity nodes etc

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.2, iterations=50)

print("Drawing graph...")
nx.draw(G, pos,
        node_color=colors,
        node_size=40,
        edge_color="lightgray",
        alpha=0.7,
        width=0.5,
        arrows=True,
        arrowsize=8)

plt.title("Transaction Subgraph (Red = Fraud, Blue = Normal)")
output_img = "outputs/visualizations/subgraph_viz.png"
plt.savefig(output_img, dpi=300)
print(f"Visualization saved to {output_img}")
plt.show()
