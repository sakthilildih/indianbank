import torch
from torch_geometric.utils import k_hop_subgraph

def extract_2hop_subgraph(node_indices, data):
    """
    Extracts a 2-hop subgraph around provided node indices.
    
    Args:
        node_indices (list or torch.Tensor): The target node indices (e.g., sender and receiver).
        data (torch_geometric.data.Data): The full graph data object containing `x` and `edge_index`.
        
    Returns:
        torch_geometric.data.Data: A new Data object representing the 2-hop subgraph.
        dict: A mapping of original node indices to new subgraph node indices.
    """
    if isinstance(node_indices, list):
        node_indices = torch.tensor(node_indices, dtype=torch.long)
        
    # Get the 2-hop neighborhood subset
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_indices,
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Extract features for the subset
    sub_x = data.x[subset] if data.x is not None else None
    sub_y = data.y[subset] if hasattr(data, 'y') and data.y is not None else None
    
    # Extract edge attributes if they exist
    sub_edge_attr = None
    if getattr(data, 'edge_attr', None) is not None:
        sub_edge_attr = data.edge_attr[edge_mask]
        
    # Create the new Data object for the subgraph
    sub_data = data.__class__(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        y=sub_y
    )
    
    # Store the original indices for reference
    sub_data.original_node_indices = subset
    
    # mapping is for the nodes provided in node_indices -> new ids in the subgraph
    # e.g., if node 50 becomes node 2 in subgraph, original_to_subgraph_mapping[50] = 2
    original_to_subgraph_mapping = {n.item(): m.item() for n, m in zip(node_indices, mapping)}
    
    return sub_data, original_to_subgraph_mapping
