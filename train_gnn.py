import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             recall_score, precision_score, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time

# 0. SET DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🚀 Focal Loss for Imbalanced Fraud Detection
def focal_loss(logits, targets, alpha=0.75, gamma=2, weight=None):
    ce = F.cross_entropy(logits, targets, reduction='none', weight=weight)
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()

# 1. LOAD GRAPH
print("STEP 1: LOADING GRAPH DATA")
data_path = "fraud_gnn_graph.pt"
import torch_geometric
checkpoint = torch.load(data_path, weights_only=False)
data = checkpoint["data"]
num_accounts = checkpoint["num_accounts"]
print(f"Loaded graph with {data.num_nodes} nodes and {data.num_edges} edges.")
print(f"Detected {num_accounts} account nodes for training.")

# 2. DEFINE DEEPER MODEL (GraphSAGE with BatchNorm)
class GNN(torch.nn.Module):
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
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x

# 3. TRAIN/TEST SPLIT (ON ACCOUNT NODES)
print("STEP 2: SPLITTING DATA")
idx = torch.arange(num_accounts)
train_idx, test_idx = train_test_split(
    idx.numpy(), test_size=0.2, stratify=data.y[:num_accounts].numpy(), random_state=42
)
train_idx = torch.tensor(train_idx).to(device)
test_idx = torch.tensor(test_idx).to(device)

# 4. HANDLE CLASS IMBALANCE
print("STEP 3: COMPUTING CLASS WEIGHTS")
weights = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=data.y[train_idx.cpu()].numpy()
)
weights = torch.tensor(weights, dtype=torch.float).to(device)
# 🔥 Restore Fraud Power: Increase weight to prioritize Fraud Recall
weights[1] *= 1.2
print(f"Class weights: Normal={weights[0]:.2f}, Fraud={weights[1]:.2f}")

# 5. TRAINING LOOP
print("STEP 4: STARTING TRAINING")
data = data.to(device)
model = GNN(data.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

best_loss = float("inf")
patience_limit = 20
counter = 0
t0 = time.time()

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = focal_loss(out[train_idx], data.y[train_idx], alpha=0.75, weight=weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = focal_loss(val_out[test_idx], data.y[test_idx], alpha=0.75, weight=weights)
            # Monitoring with 0.33 threshold
            probs_val = torch.softmax(val_out[test_idx], dim=1)[:, 1]
            preds_val = (probs_val > 0.33).long()
            y_true_test = data.y[test_idx].cpu().numpy()
            y_pred_test = preds_val.cpu().numpy()
            recall = recall_score(y_true_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Recall: {recall:.2f} | F1: {f1:.2f}")
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), "best_fraud_gnn_model.pth")
                best_metrics = (recall, f1)
            else:
                counter += 1
            if counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch}")
                break

print(f"Training completed in {time.time() - t0:.2f}s")

# 6. DYNAMIC THRESHOLD SEARCH
print("\nSTEP 5: DYNAMIC THRESHOLD SEARCH")
model.load_state_dict(torch.load("best_fraud_gnn_model.pth"))
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs_val = torch.softmax(out[test_idx], dim=1)[:, 1].cpu().numpy()
y_true = data.y[test_idx].cpu().numpy()

# User requested Best threshold: 0.33
best_thresh = 0.33
print(f"Using Optimal threshold: {best_thresh:.2f}")

# 7. FINAL EVALUATION
print("\nSTEP 6: FINAL EVALUATION")
final_preds = (probs_val > best_thresh).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, final_preds))

print("\nClassification Report:")
print(classification_report(y_true, final_preds, target_names=["Normal", "Fraud"]))

auc = roc_auc_score(y_true, probs_val)
print(f"ROC-AUC Score: {auc:.4f}")

# 🚀 Production-Style Risk Categorization
print("\n🚀 RISK CATEGORIZATION REPORT")
for i in range(min(10, len(test_idx))):
    prob = probs_val[i]
    risk = "HIGH RISK 🔴" if prob > 0.6 else ("MEDIUM RISK 🟡" if prob > 0.33 else "NORMAL 🟢")
    print(f"Account Node {test_idx[i].item()}: Score={prob:.3f} -> {risk}")

print("\n⚠️ Fraud Recall is the MOST important metric in banking systems")

torch.save(model.state_dict(), "fraud_gnn_model.pth")
print("\nFinal Model saved to fraud_gnn_model.pth")
