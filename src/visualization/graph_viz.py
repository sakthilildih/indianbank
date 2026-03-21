import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import time

print("STEP 1: LOAD DATA")
df = pd.read_csv("data/processed/transactions_parsed.csv")
df = df.sort_values("timestamp")

print("STEP 2: CREATE NODE INDEX")
# Accounts
accounts = pd.concat([df["sender_account"], df["receiver_account"]]).dropna().unique()
account_to_id = {acc: i for i, acc in enumerate(accounts)}

# Mobiles
mobiles = df["mobile"].dropna().unique()
mobile_to_id = {m: i + len(account_to_id) for i, m in enumerate(mobiles)}

# Names
names = df["name"].dropna().unique()
name_to_id = {n: i + len(account_to_id) + len(mobile_to_id) for i, n in enumerate(names)}

# Pincode
pins = df["pincode"].dropna().unique()
pin_to_id = {p: i + len(account_to_id) + len(mobile_to_id) + len(name_to_id)
             for i, p in enumerate(pins)}

num_nodes = len(account_to_id) + len(mobile_to_id) + len(name_to_id) + len(pin_to_id)
print(f"Total nodes: {num_nodes} (Accounts: {len(accounts)}, Mobiles: {len(mobiles)}, Names: {len(names)}, Pincodes: {len(pins)})")

print("STEP 3: BUILD EDGES + EDGE FEATURES")
edge_src = []
edge_dst = []
edge_attr = []

# Transaction edges
channel_map = {"UPI":0, "IMPS":1, "WEB":2, "APP":3, "ATM":4}

t0 = time.time()
min_time = df["timestamp"].min()
df["time_norm"] = df["timestamp"] - min_time

for _, row in df.iterrows():
    s = account_to_id[row["sender_account"]]
    r = account_to_id[row["receiver_account"]]

    edge_src.append(s)
    edge_dst.append(r)

    edge_attr.append([
        np.log1p(row["amount"]),        # amount normalized
        row["time_norm"] / 1e6,         # scale down
        channel_map.get(row["channel"], 0)
    ])

# Identity edges
for _, row in df.iterrows():
    acc = account_to_id[row["sender_account"]]

    # Mobile
    if pd.notna(row["mobile"]):
        edge_src.append(acc)
        edge_dst.append(mobile_to_id[row["mobile"]])
        edge_attr.append([0,0,0])

    # Name
    if pd.notna(row["name"]):
        edge_src.append(acc)
        edge_dst.append(name_to_id[row["name"]])
        edge_attr.append([0,0,0])

    # Pincode
    if pd.notna(row["pincode"]):
        edge_src.append(acc)
        edge_dst.append(pin_to_id[row["pincode"]])
        edge_attr.append([0,0,0])

print(f"Edges built in {time.time() - t0:.2f}s")
print(f"Total edges: {len(edge_src)}")

print("STEP 4: NODE FEATURES")
t0 = time.time()
x = np.zeros((num_nodes, 12))

# 1. Degree features
for s, d in zip(edge_src, edge_dst):
    x[s][0] += 1   # out_degree
    x[d][1] += 1   # in_degree

# Optional: Receiver-side feature (fan-in signal stronger)
recv_count = df.groupby("receiver_account")["sender_account"].nunique()
for acc, val in recv_count.items():
    if acc in account_to_id:
        x[account_to_id[acc]][1] += val

# 2. Transaction count
txn_counts = df["sender_account"].value_counts()
for acc, count in txn_counts.items():
    if acc in account_to_id:
        x[account_to_id[acc]][2] = count

# Optional: Channel diversity (fragmentation signal)
channel_div = df.groupby("sender_account")["channel"].nunique()
for acc, val in channel_div.items():
    if acc in account_to_id:
        x[account_to_id[acc]][10] = val

# 3. Unique receivers (fan-out signal)
uniq_recv = df.groupby("sender_account")["receiver_account"].nunique()
for acc, val in uniq_recv.items():
    if acc in account_to_id:
        x[account_to_id[acc]][3] = val

# 4. Amount features (structuring)
grp = df.groupby("sender_account")["amount"]

amount_mean = grp.mean()
amount_std = grp.std().fillna(0)
for acc in amount_mean.index:
    if acc in account_to_id:
        x[account_to_id[acc]][4] = amount_mean[acc]
        x[account_to_id[acc]][5] = amount_std[acc]

# 5. Time features (VERY IMPORTANT)
df["time_diff"] = df.groupby("sender_account")["timestamp"].diff().fillna(0)

time_gap = df.groupby("sender_account")["time_diff"].mean()

for acc, val in time_gap.items():
    if acc in account_to_id:
        x[account_to_id[acc]][6] = val

# 6. Burst score (fan-out speed)
burst = df.groupby("sender_account").apply(lambda g: (g["time_diff"] < 60).sum())

for acc, val in burst.items():
    if acc in account_to_id:
        x[account_to_id[acc]][7] = val

# 7. Identity features (mule detection)
mobile_count = df.groupby("mobile")["account_number"].nunique()

for _, row in df.drop_duplicates("sender_account").iterrows():
    acc = account_to_id.get(row["sender_account"])
    mob = row["mobile"]
    if acc is not None and pd.notna(mob) and mob in mobile_count:
        x[acc][8] = mobile_count[mob]

# 8. Pincode risk
pin_risk = df.groupby("pincode")["label"].mean()

for _, row in df.drop_duplicates("sender_account").iterrows():
    acc = account_to_id.get(row["sender_account"])
    pin = row["pincode"]
    if acc is not None and pd.notna(pin) and pin in pin_risk:
        x[acc][9] = pin_risk[pin]

# 9. Receiver Diversity (High Impact)
recv_div = df.groupby("sender_account")["receiver_account"].count()
for acc, val in recv_div.items():
    if acc in account_to_id:
        x[account_to_id[acc]][11] = val
x[:,6] = np.log1p(x[:,6])
x[:,11] = np.log1p(x[:,11])  # 🚀 Added log scaling to new feature
print(f"Node features built in {time.time() - t0:.2f}s")

print("STEP 5: LABELS (ACCOUNT ONLY)")
y = np.zeros(num_nodes)

fraud_acc = df.groupby("sender_account")["label"].max()

for acc, val in fraud_acc.items():
    if acc in account_to_id:
        y[account_to_id[acc]] = val

print("STEP 6: SCALING & CONVERT TO PYTORCH")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

edge_attr[:,1] = edge_attr[:,1] / edge_attr[:,1].max()

from torch_geometric.utils import add_self_loops
edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)

x_tensor   = torch.tensor(x, dtype=torch.float)
y_tensor   = torch.tensor(y, dtype=torch.long)

data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
print("------------------------------------------")
print("🧠 FINAL FEATURE VECTOR")
print("Index\tFeature")
print("0\tout_degree")
print("1\tin_degree")
print("2\ttxn_count")
print("3\tunique_receivers")
print("4\tavg_amount")
print("5\tstd_amount")
print("6\tavg_time_gap (log scaled)")
print("7\tburst_score")
print("8\tmobile_shared_count")
print("9\tpincode_risk")
print("10\tchannel_diversity")
print("11\treceiver_diversity")
print("------------------------------------------")
print(data)

checkpoint = {
    "data": data,
    "num_accounts": len(account_to_id)
}
torch.save(checkpoint, "outputs/visualizations/fraud_gnn_graph.pt")
print(f"Saved to outputs/visualizations/fraud_gnn_graph.pt (Accounts: {len(account_to_id)})")
