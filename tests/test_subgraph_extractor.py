import unittest
import torch
from torch_geometric.data import Data
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.subgraph_extractor import extract_2hop_subgraph

class TestSubgraphExtractor(unittest.TestCase):
    
    def setUp(self):
        """
        Create a simple dummy graph for testing:
             0 -- 1 -- 2 -- 3 -- 4
             |
             5
        We want to extract the 2-hop neighborhood around node 0.
        Expected nodes: 0, 1, 2, 5 (Node 3 is 3-hops away, Node 4 is 4-hops away).
        """
        # Node features
        self.x = torch.tensor([[i] for i in range(6)], dtype=torch.float)
        
        # Undirected edges
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 0]
        ], dtype=torch.long)
        
        self.data = Data(x=self.x, edge_index=edge_index)
        
    def test_extract_2hop_subgraph_single_node(self):
        target_nodes = [0]
        
        sub_data, mapping = extract_2hop_subgraph(target_nodes, self.data)
        
        # Nodes 0, 1, 2, 5 should be present
        expected_nodes_count = 4
        self.assertEqual(sub_data.num_nodes, expected_nodes_count)
        
        # Original node 0 should be mapped
        self.assertIn(0, mapping)
        
        # The stored original indices should contain 0, 1, 2, 5
        original_indices = sub_data.original_node_indices.tolist()
        for expected_node in [0, 1, 2, 5]:
            self.assertIn(expected_node, original_indices)
            
        # Nodes 3 and 4 should NOT be in the subgraph
        self.assertNotIn(3, original_indices)
        self.assertNotIn(4, original_indices)

    def test_extract_2hop_subgraph_multiple_nodes(self):
        # E.g., a transaction occurring between nodes 0 and 3
        # 2-hop around 0 includes: 0, 1, 2, 5
        # 2-hop around 3 includes: 3, 2, 4, 1
        # Combined: 0, 1, 2, 3, 4, 5 (Entire graph!)
        target_nodes = [0, 3]
        
        sub_data, mapping = extract_2hop_subgraph(target_nodes, self.data)
        
        self.assertEqual(sub_data.num_nodes, 6)
        self.assertIn(0, mapping)
        self.assertIn(3, mapping)

if __name__ == '__main__':
    unittest.main()
