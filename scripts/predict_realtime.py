import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.subgraph_extractor import extract_2hop_subgraph

# ---------------------------------------------------------
# 1. Model Definition (Must match training structure)
# ---------------------------------------------------------
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = SAGEConv(32, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.conv3 = SAGEConv(16, 8)
        self.bn3 = torch.nn.BatchNorm1d(8)
        self.conv4 = SAGEConv(8, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x

# ---------------------------------------------------------
# 2. Bootstrapping Dummy In-Memory State
# ---------------------------------------------------------
# In a real app, this graph is built once and kept in memory or Redis.
def bootstrap_in_memory_state():
    print("Bootstrapping baseline graph into memory...")
    try:
        from scripts.predict import build_test_graph, parse_test_data
        
        # Load the raw test data just to build the baseline topology
        df = pd.read_csv('data/raw/test_transactions.csv')
        df = parse_test_data(df)
        
        global_graph, account_to_id, num_accounts = build_test_graph(df)
        print(f"Memory Graph Ready: {global_graph.num_nodes} nodes, {global_graph.num_edges} edges.")
        return global_graph, account_to_id
    except Exception as e:
        print(f"Failed to bootstrap. Make sure 'predict.py' has generated data.\n{e}")
        sys.exit(1)

# ---------------------------------------------------------
# 3. Real-Time Processing Function
# ---------------------------------------------------------
def process_realtime_transaction(txn_json, global_graph, account_to_id, model, device):
    """
    Handles exactly what happens when a new transaction hits Kafka.
    """
    sender = txn_json.get("sender_account")
    receiver = txn_json.get("receiver_account")
    
    # 1. Map to Node IDs
    # If the account doesn't exist,For demo, we skip unknown nodes.
    if sender not in account_to_id or receiver not in account_to_id:
        return {"error": "Account not found in graph topology yet (cold start)."}
    
    sender_id = account_to_id[sender]
    receiver_id = account_to_id[receiver]
    
    # 2. Update Graph Features (SIMULATED)
    global_graph.x[sender_id][0] += 1  # out_degree
    global_graph.x[sender_id][2] += 1  # txn_count
    global_graph.x[receiver_id][1] += 1 # in_degree
    
    # 3. Append the new edge to the global graph
    new_edge = torch.tensor([[sender_id], [receiver_id]], dtype=torch.long)
    global_graph.edge_index = torch.cat([global_graph.edge_index, new_edge], dim=1)
    
    # 3b. Also append edge_attr (log_amount, time_norm, channel_id)
    # channel_map = {"UPI": 0, "IMPS": 1, "WEB": 2, "APP": 3, "ATM": 4}
    ch_id = {"UPI":0, "IMPS":1, "WEB":2, "APP":3, "ATM":4}.get(txn_json.get("channel", "UPI"), 0)
    amt_val = np.log1p(txn_json.get("amount", 0))
    new_edge_attr = torch.tensor([[amt_val, 1.0, float(ch_id)]], dtype=torch.float)
    if getattr(global_graph, 'edge_attr', None) is not None:
        global_graph.edge_attr = torch.cat([global_graph.edge_attr, new_edge_attr], dim=0)
    
    # 4. Extract 2-Hop Subgraph
    print(f"\nExtracting 2-hop neighborhood for Sender '{sender}' and Receiver '{receiver}'...")
    subgraph, mapping = extract_2hop_subgraph([sender_id, receiver_id], global_graph)
    print(f"-> 2-Hop Subgraph isolated: {subgraph.num_nodes} nodes, {subgraph.num_edges} edges.")
    
    # 5. Run Model Inference
    subgraph = subgraph.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(subgraph.x, subgraph.edge_index)
        probs = torch.softmax(logits, dim=1)
        
    # 6. Extract the sender's specific fraud score from the subgraph results
    subgraph_sender_id = mapping[sender_id]
    fraud_score = probs[subgraph_sender_id, 1].item()
    
    # Determine risk category
    if fraud_score > 0.6:
        risk = "HIGH RISK 🔴 (BLOCK)"
    elif fraud_score > 0.33:
        risk = "MEDIUM RISK 🟡 (REVIEW)"
    else:
        risk = "NORMAL 🟢 (ALLOW)"
        
    return {
        "txn_id": txn_json.get("txn_id"),
        "sender_account": sender,
        "receiver_account": receiver,
        "amount": txn_json.get("amount"),
        "fraud_score": round(fraud_score, 4),
        "decision": risk
    }

# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = GNNModel(in_channels=12) # 12 Features from IntelliTrace pipeline
    model_path = 'models/best_fraud_gnn_model.pth'
    if not os.path.exists(model_path):
        print("Model not found. Run train_model.py first.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    # Load Memory State
    global_graph, account_to_id = bootstrap_in_memory_state()
    
    test_sender = list(account_to_id.keys())[50]
    test_receiver = list(account_to_id.keys())[150]
    
    # SIM INCOMING KAFKA JSON MESSAGE
    incoming_kafka_message = {
  "txn_id": "TXN_LIVE_99999",
  "sender_account": "361374041380",
  "receiver_account": "757872634354",
  "amount": 85000,
  "channel": "UPI",
  "timestamp": 1735690000
}


    
    print("\n" + "="*50)
    print(" PROCESSING NEW INCOMING TRANSACTION")
    print("="*50)
    print(f"Payload: {incoming_kafka_message}")
    
    result = process_realtime_transaction(
        txn_json=incoming_kafka_message,
        global_graph=global_graph,
        account_to_id=account_to_id,
        model=model,
        device=device
    )
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
