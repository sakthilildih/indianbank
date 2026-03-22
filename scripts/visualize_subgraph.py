import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import sys

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.subgraph_extractor import extract_2hop_subgraph

def main():
    data_path = "outputs/visualizations/fraud_gnn_graph.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print("Loading full graph...")
    checkpoint = torch.load(data_path, weights_only=False)
    data = checkpoint["data"]
    
    # We will choose a high-degree or suspected fraud node to extract around
    # If the graph has labels (`y`), we can find a fraud node (label = 1)
    # The first index where y == 1
    if hasattr(data, 'y') and data.y is not None:
        fraud_nodes = (data.y == 1).nonzero(as_tuple=True)[0]
        if len(fraud_nodes) > 0:
            target_node = fraud_nodes[0].item()
            print(f"Selected target node {target_node} (Known Fraud) for 2-hop subgraph.")
        else:
            target_node = 0
            print(f"No fraud labels found. Selected node {target_node}.")
    else:
        # Just pick a random highly connected node or node 0
        target_node = 0
        print(f"No labels found. Selected node {target_node}.")

    # For a transaction, we often use sender and receiver. Here we just use the selected node
    nodes_of_interest = [target_node]
    
    print(f"Extracting 2-hop subgraph around node(s): {nodes_of_interest}")
    sub_data, mapping = extract_2hop_subgraph(nodes_of_interest, data)
    
    print(f"Subgraph: {sub_data.num_nodes} nodes, {sub_data.num_edges} edges.")
    
    # Create networkx graph for visualization
    # We need a directed graph
    G = nx.DiGraph()
    
    # Add edges
    edge_index = sub_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u != v: # exclude self-loops for cleaner viz
            G.add_edge(u, v)
            
    # Draw graph
    plt.figure(figsize=(10, 8))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Color nodes
    colors = []
    node_sizes = []
    
    for n in G.nodes():
        # Check if the node is in our target group
        original_idx = sub_data.original_node_indices[n].item()
        if original_idx in nodes_of_interest:
            colors.append("orange") # Highlight focal nodes
            node_sizes.append(300)
        else:
            # Check if fraud
            fraud_status = False
            if hasattr(sub_data, 'y') and sub_data.y is not None:
                fraud_status = (sub_data.y[n] == 1)
            
            if fraud_status:
                colors.append("red")
            else:
                colors.append("royalblue")
            node_sizes.append(100)
            
    nx.draw(G, pos, 
            node_color=colors, 
            node_size=node_sizes, 
            edge_color="gray", 
            with_labels=False, 
            alpha=0.8,
            arrows=True,
            arrowsize=10)
            
    # Add a legend
    import matplotlib.patches as mpatches
    target_patch = mpatches.Patch(color='orange', label='Target Nodes')
    fraud_patch = mpatches.Patch(color='red', label='Fraud Nodes')
    normal_patch = mpatches.Patch(color='royalblue', label='Normal Nodes')
    plt.legend(handles=[target_patch, fraud_patch, normal_patch])
    
    plt.title(f"2-Hop Subgraph around Node(s) {nodes_of_interest}")
    
    os.makedirs("outputs/visualizations", exist_ok=True)
    out_path = "outputs/visualizations/2hop_subgraph_viz.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved visualization to {out_path}")
    
if __name__ == "__main__":
    main()
