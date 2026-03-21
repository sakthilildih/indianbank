"""
═══════════════════════════════════════════════════════════════════════════════
    FRAUD DETECTION DEMO PIPELINE — End-to-End GNN Inference
═══════════════════════════════════════════════════════════════════════════════

Pipeline Stages:
  1️⃣  Generate synthetic test dataset with fraud patterns (NOT training data)
  2️⃣  Parse narration fields
  3️⃣  Build graph with node features
  4️⃣  Load trained GNN model
  5️⃣  Run inference & classify risk levels
  6️⃣  Output predictions & explanations
  7️⃣  Visualize fraud patterns and graph
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import uuid
import random
import datetime
import time
import warnings
from collections import defaultdict, Counter

# Force UTF-8 on Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings('ignore')

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ CONFIGURATION                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Risk thresholds
RISK_THRESHOLDS = {
    'HIGH': 0.6,      # > 0.6 → 🔴 HIGH RISK
    'MEDIUM': 0.33,   # 0.33-0.6 → 🟡 MEDIUM RISK
    'NORMAL': 0.0     # < 0.33 → 🟢 NORMAL
}

# Constants
EPOCH_START = 1_735_689_600  # 2025-01-01
EPOCH_END = 1_767_225_600    # 2026-01-01

HIGH_RISK_PINCODES = {
    600003, 600007, 600011, 600019, 600025, 600031,
    600037, 600043, 600051, 600063, 600071, 600082,
    600090, 600099,
}
ALL_PINCODES = list(range(600001, 600101))
NORMAL_PINCODES = [p for p in ALL_PINCODES if p not in HIGH_RISK_PINCODES]

CHANNELS = ["UPI", "IMPS", "WEB", "APP", "ATM"]
BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "PNB", "BOB", "CANARA"]
MALE_NAMES = ["Arjun","Karthik","Vikram","Rajesh","Suresh","Ramesh","Dinesh",
              "Ganesh","Venkat","Prasad","Sathish","Murugan","Selvam","Mani",
              "Rajan","Kumar","Sakthivel","Anand","Bala","Chandru","Deepak",
              "Elan","Farhan","Gopi","Hari","Iyappan","Jobi","Kiran","Lokesh"]
FEMALE_NAMES = ["Priya","Kavya","Lakshmi","Meena","Nithya","Rani","Saranya",
                "Thenmozhi","Uma","Vani","Abinaya","Bhavani","Chitra","Divya",
                "Ezhilarasi","Gowri","Hema","Indira","Janani","Kamala"]
ALL_NAMES = MALE_NAMES + FEMALE_NAMES

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ STEP 1: GENERATE TEST DATASET WITH FRAUD PATTERNS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def gen_mobile():
    return random.choice(["9","8","7","6"]) + str(random.randint(100_000_000, 999_999_999))

def gen_acc_num(used):
    while True:
        n = str(random.randint(100_000_000_000, 999_999_999_999))
        if n not in used:
            used.add(n)
            return n

def generate_test_dataset(num_accounts=5000, num_transactions=15000):
    """
    Generate synthetic UNSEEN test dataset with fraud patterns.
    Ensures different accounts from training data.
    """
    print("\n" + "="*80)
    print("🎯 STEP 1: GENERATE TEST DATASET WITH FRAUD PATTERNS")
    print("="*80)
    
    random.seed(123)  # Different seed from training (which used 42)
    np.random.seed(123)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1A: Create accounts (DIFFERENT FROM TRAINING)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[1.1] Generating {num_accounts} accounts...")
    
    accounts = []
    used_acc = set()
    
    for i in range(num_accounts):
        accounts.append({
            'account_number': gen_acc_num(used_acc),
            'name': random.choice(ALL_NAMES),
            'mobile': gen_mobile(),
            'pincode': random.choice(ALL_PINCODES),
            'bank': random.choice(BANKS),
            'product': random.choice(['Savings', 'Current', 'Credit'])
        })
    
    # Create clustering for shared mobile/identity (mule detection)
    mule_groups = random.sample(range(num_accounts), k=int(num_accounts * 0.1))
    shared_mobile = gen_mobile()
    for idx in mule_groups[:20]:
        accounts[idx]['mobile'] = shared_mobile
    
    print(f"✓ Created {num_accounts} accounts")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1B: Generate transactions with FRAUD PATTERNS
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[1.2] Generating {num_transactions} transactions with fraud patterns...")
    
    transactions = []
    fraud_accounts = set()  # Track which accounts are fraudulent
    normal_accounts = set()
    
    pattern_counts = defaultdict(int)
    txn_id_counter = 0
    
    # ~20% fraud-like behavior
    num_fraud = int(num_transactions * 0.20)
    num_normal = num_transactions - num_fraud
    
    # ╔─ FRAUD PATTERNS ─╗
    
    # Pattern 1: FAN-OUT (one sender → multiple receivers within 1min)
    print("  • Generating FAN-OUT pattern (sender→multiple receivers)...")
    for _ in range(int(num_fraud * 0.20)):
        sender = random.choice(accounts)
        fraud_accounts.add(sender['account_number'])
        base_time = random.randint(EPOCH_START, EPOCH_END - 300)
        
        # 1 sender sends to 4-8 receivers in <1min
        for recv_idx in range(random.randint(4, 8)):
            receiver = random.choice(accounts)
            txn_time = base_time + random.randint(0, 60)
            amount = random.randint(5000, 50000)
            
            narr = f"{random.choice(['UPI', 'IMPS'])}/{receiver['mobile'] if random.random() > 0.5 else receiver['account_number']}/{receiver['name']}/{receiver['account_number']}"
            
            transactions.append({
                'txn_id': f'TXND{txn_id_counter:08d}',
                'sender_account': sender['account_number'],
                'receiver_account': receiver['account_number'],
                'sender_name': sender['name'],
                'sender_mobile': sender['mobile'],
                'sender_pincode': sender['pincode'],
                'receiver_name': receiver['name'],
                'receiver_mobile': receiver['mobile'],
                'amount': amount,
                'timestamp': txn_time,
                'channel': 'UPI',
                'narration': narr,
                'label': 1,  # FRAUD
                'fraud_pattern': 'FAN_OUT'
            })
            txn_id_counter += 1
        pattern_counts['FAN_OUT'] += 1
    
    # Pattern 2: FAN-IN (multiple senders → one receiver)
    print("  • Generating FAN-IN pattern (multiple senders→one receiver)...")
    for _ in range(int(num_fraud * 0.20)):
        receiver = random.choice(accounts)
        fraud_accounts.add(receiver['account_number'])
        base_time = random.randint(EPOCH_START, EPOCH_END - 3600)
        
        # 5-10 senders send to same receiver within 1 hour
        for send_idx in range(random.randint(5, 10)):
            sender = random.choice(accounts)
            txn_time = base_time + random.randint(0, 3600)
            amount = random.randint(5000, 50000)
            
            narr = f"IMPS/{receiver['account_number']}/{receiver['name']}"
            
            transactions.append({
                'txn_id': f'TXND{txn_id_counter:08d}',
                'sender_account': sender['account_number'],
                'receiver_account': receiver['account_number'],
                'sender_name': sender['name'],
                'sender_mobile': sender['mobile'],
                'sender_pincode': sender['pincode'],
                'receiver_name': receiver['name'],
                'receiver_mobile': receiver['mobile'],
                'amount': amount,
                'timestamp': txn_time,
                'channel': 'IMPS',
                'narration': narr,
                'label': 1,  # FRAUD
                'fraud_pattern': 'FAN_IN'
            })
            txn_id_counter += 1
        pattern_counts['FAN_IN'] += 1
    
    # Pattern 3: CHAIN (A→B→C→D within minutes)
    print("  • Generating CHAIN pattern (A→B→C→D sequence)...")
    for _ in range(int(num_fraud * 0.15)):
        chain_accounts = random.sample(accounts, k=random.randint(3, 5))
        base_time = random.randint(EPOCH_START, EPOCH_END - 600)
        
        for i in range(len(chain_accounts) - 1):
            sender = chain_accounts[i]
            receiver = chain_accounts[i + 1]
            if i == 0:
                fraud_accounts.add(sender['account_number'])
            
            txn_time = base_time + (i * random.randint(60, 180))
            amount = random.randint(5000, 50000)
            
            narr = f"UPI/{receiver['mobile']}/{receiver['name']}/{receiver['account_number']}"
            
            transactions.append({
                'txn_id': f'TXND{txn_id_counter:08d}',
                'sender_account': sender['account_number'],
                'receiver_account': receiver['account_number'],
                'sender_name': sender['name'],
                'sender_mobile': sender['mobile'],
                'sender_pincode': sender['pincode'],
                'receiver_name': receiver['name'],
                'receiver_mobile': receiver['mobile'],
                'amount': amount,
                'timestamp': txn_time,
                'channel': 'UPI',
                'narration': narr,
                'label': 1,  # FRAUD
                'fraud_pattern': 'CHAIN'
            })
            txn_id_counter += 1
        pattern_counts['CHAIN'] += 1
    
    # Pattern 4: STRUCTURING (multiple similar amounts ~45k-50k)
    print("  • Generating STRUCTURING pattern (similar amounts to evade limits)...")
    for _ in range(int(num_fraud * 0.20)):
        sender = random.choice(accounts)
        fraud_accounts.add(sender['account_number'])
        base_time = random.randint(EPOCH_START, EPOCH_END - 86400)
        
        # Multiple txns of similar amount over days
        for day in range(random.randint(3, 7)):
            receiver = random.choice(accounts)
            txn_time = base_time + (day * 86400) + random.randint(0, 3600)
            amount = random.randint(45000, 50000)  # Just Below 50k limit
            
            narr = f"{'UPI' if random.random() > 0.5 else 'IMPS'}/{receiver['account_number']}/{receiver['name']}"
            
            transactions.append({
                'txn_id': f'TXND{txn_id_counter:08d}',
                'sender_account': sender['account_number'],
                'receiver_account': receiver['account_number'],
                'sender_name': sender['name'],
                'sender_mobile': sender['mobile'],
                'sender_pincode': sender['pincode'],
                'receiver_name': receiver['name'],
                'receiver_mobile': receiver['mobile'],
                'amount': amount,
                'timestamp': txn_time,
                'channel': random.choice(['UPI', 'IMPS']),
                'narration': narr,
                'label': 1,  # FRAUD
                'fraud_pattern': 'STRUCTURING'
            })
            txn_id_counter += 1
        pattern_counts['STRUCTURING'] += 1
    
    # Pattern 5: FRAGMENTATION (same sender→receiver via multiple channels)
    print("  • Generating FRAGMENTATION pattern (multi-channel usage)...")
    for _ in range(int(num_fraud * 0.15)):
        sender = random.choice(accounts)
        receiver = random.choice(accounts)
        fraud_accounts.add(sender['account_number'])
        base_time = random.randint(EPOCH_START, EPOCH_END - 3600)
        
        # Same pair via UPI, IMPS, APP within 1 hour
        for ch_idx, channel in enumerate(['UPI', 'IMPS', 'APP']):
            txn_time = base_time + (ch_idx * 600)
            amount = random.randint(10000, 30000)
            
            if channel == 'UPI':
                narr = f"UPI/{receiver['mobile']}/{receiver['name']}/{receiver['account_number']}"
            elif channel == 'IMPS':
                narr = f"IMPS/{receiver['account_number']}/{receiver['name']}"
            else:  # APP
                narr = f"APP-P2P/{receiver['mobile']}/{receiver['name']}/{receiver['account_number']}"
            
            transactions.append({
                'txn_id': f'TXND{txn_id_counter:08d}',
                'sender_account': sender['account_number'],
                'receiver_account': receiver['account_number'],
                'sender_name': sender['name'],
                'sender_mobile': sender['mobile'],
                'sender_pincode': sender['pincode'],
                'receiver_name': receiver['name'],
                'receiver_mobile': receiver['mobile'],
                'amount': amount,
                'timestamp': txn_time,
                'channel': channel,
                'narration': narr,
                'label': 1,  # FRAUD
                'fraud_pattern': 'FRAGMENTATION'
            })
            txn_id_counter += 1
        pattern_counts['FRAGMENTATION'] += 1
    
    # Pattern 6 & 7: High-risk pincode & shared mobile
    print("  • Generating SHARED IDENTITY & HIGH-RISK PINCODE patterns...")
    remaining_fraud = num_fraud - sum(pattern_counts.values())
    for _ in range(remaining_fraud):
        pattern = random.choice(['SHARED_IDENTITY', 'HIGH_RISK_LOCATION'])
        
        if pattern == 'SHARED_IDENTITY':
            # Multiple accounts with same mobile
            sender = random.choice(accounts)
            fraud_accounts.add(sender['account_number'])
            receiver = random.choice([a for a in accounts if a['mobile'] == sender['mobile']])
        else:
            # High-risk pincode transactions
            sender = random.choice(accounts)
            sender['pincode'] = random.choice(list(HIGH_RISK_PINCODES))
            fraud_accounts.add(sender['account_number'])
            receiver = random.choice(accounts)
        
        txn_time = random.randint(EPOCH_START, EPOCH_END)
        amount = random.randint(5000, 50000)
        narr = f"UPI/{receiver['mobile']}/{receiver['name']}/{receiver['account_number']}"
        
        transactions.append({
            'txn_id': f'TXND{txn_id_counter:08d}',
            'sender_account': sender['account_number'],
            'receiver_account': receiver['account_number'],
            'sender_name': sender['name'],
            'sender_mobile': sender['mobile'],
            'sender_pincode': sender['pincode'],
            'receiver_name': receiver['name'],
            'receiver_mobile': receiver['mobile'],
            'amount': amount,
            'timestamp': txn_time,
            'channel': random.choice(CHANNELS),
            'narration': narr,
            'label': 1,
            'fraud_pattern': pattern
        })
        txn_id_counter += 1
        pattern_counts[pattern] += 1
    
    # ╔─ NORMAL TRANSACTIONS ─╗
    print("  • Generating NORMAL transactions...")
    for _ in range(num_normal):
        sender = random.choice(accounts)
        receiver = random.choice(accounts)
        
        # Skip if too many fraud patterns on this account
        if len([t for t in transactions if t['sender_account'] == sender['account_number']]) > 50:
            continue
        
        normal_accounts.add(sender['account_number'])
        
        # Normal: occasional transactions, normal amounts, normal spacing
        txn_time = random.randint(EPOCH_START, EPOCH_END)
        amount = random.randint(1000, 100000)
        channel = random.choice(CHANNELS)
        
        if channel == 'UPI':
            narr = f"UPI/{receiver['mobile']}/{receiver['name']}/{receiver['account_number']}"
        elif channel == 'IMPS':
            narr = f"IMPS/{receiver['account_number']}/{receiver['name']}"
        elif channel == 'WEB':
            narr = f"WEB-TRF/{receiver['account_number']}/{receiver['name']}"
        elif channel == 'APP':
            narr = f"APP-P2P/{receiver['mobile']}/{receiver['name']}/{receiver['account_number']}"
        else:  # ATM
            narr = f"ATM-WDL/ATM{random.randint(1,500):05d}/Chennai"
        
        transactions.append({
            'txn_id': f'TXND{txn_id_counter:08d}',
            'sender_account': sender['account_number'],
            'receiver_account': receiver['account_number'],
            'sender_name': sender['name'],
            'sender_mobile': sender['mobile'],
            'sender_pincode': sender['pincode'],
            'receiver_name': receiver['name'],
            'receiver_mobile': receiver['mobile'],
            'amount': amount,
            'timestamp': txn_time,
            'channel': channel,
            'narration': narr,
            'label': 0,  # NORMAL
            'fraud_pattern': 'NORMAL'
        })
        txn_id_counter += 1
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save (label will be hidden during actual inference)
    df.to_csv('data/raw/test_transactions.csv', index=False)
    
    print(f"\n✓ Generated {len(df)} transactions")
    print(f"✓ Fraud transactions: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    print(f"✓ Normal transactions: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    print(f"\n📊 Fraud Pattern Distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {pattern:20s}: {count:4d}")
    
    return df, fraud_accounts, normal_accounts


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ STEP 2: PARSE NARRATION                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def parse_narration(row):
    """Extract channel, receiver info from narration."""
    narration = str(row['narration']) if pd.notna(row['narration']) else ''
    parts = narration.split('/')
    narr_type = parts[0] if parts else 'UNKNOWN'
    
    channel = 'UNKNOWN'
    recv_acc = 'UNKNOWN'
    recv_mob = None
    recv_name = None
    
    try:
        if narr_type == 'UPI':
            channel = 'UPI'
            recv_mob = parts[1] if len(parts) > 1 else None
            recv_name = parts[2] if len(parts) > 2 else None
            recv_acc = parts[3] if len(parts) > 3 else 'UNKNOWN'
        elif narr_type == 'IMPS':
            channel = 'IMPS'
            recv_acc = parts[1] if len(parts) > 1 else 'UNKNOWN'
            recv_name = parts[2] if len(parts) > 2 else None
        elif narr_type == 'WEB-TRF':
            channel = 'WEB'
            recv_acc = parts[1] if len(parts) > 1 else 'UNKNOWN'
            recv_name = parts[2] if len(parts) > 2 else None
        elif narr_type == 'APP-P2P':
            channel = 'APP'
            recv_mob = parts[1] if len(parts) > 1 else None
            recv_name = parts[2] if len(parts) > 2 else None
            recv_acc = parts[3] if len(parts) > 3 else 'UNKNOWN'
        elif narr_type == 'ATM-WDL':
            channel = 'ATM'
            recv_acc = str(row['sender_account'])
        else:
            channel = 'UNKNOWN'
    except:
        pass
    
    return pd.Series([channel, recv_acc, recv_mob, recv_name])

def parse_test_data(df):
    """Parse narration in test data."""
    print("\n" + "="*80)
    print("🎯 STEP 2: PARSE NARRATION")
    print("="*80)
    
    print("\n[2.1] Parsing narration fields...")
    df[['channel', 'receiver_account', 'receiver_mobile', 'receiver_name']] = \
        df.apply(parse_narration, axis=1)
    
    df['sender_account'] = df['sender_account']
    df['receiver_account'] = df['receiver_account'].fillna('UNKNOWN')
    
    print(f"✓ Parsed {len(df)} transactions")
    print(f"✓ Channel distribution:")
    print(df['channel'].value_counts().to_string())
    
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ STEP 3: BUILD GRAPH WITH NODE FEATURES                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class GNNModel(torch.nn.Module):
    """GNN model matching training architecture."""
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

def build_test_graph(df):
    """Build graph from test data."""
    print("\n" + "="*80)
    print("🎯 STEP 3: BUILD GRAPH WITH NODE FEATURES")
    print("="*80)
    
    df = df.sort_values('timestamp')
    
    # ──────────────────────────────────────────────────────────────────────────
    # Create node indices
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3.1] Creating node indices...")
    
    accounts = pd.concat([df['sender_account'], df['receiver_account']]).dropna().unique()
    account_to_id = {acc: i for i, acc in enumerate(accounts)}
    
    mobiles = df['sender_mobile'].dropna().unique()
    mobile_to_id = {m: i + len(account_to_id) for i, m in enumerate(mobiles)}
    
    names = df['sender_name'].dropna().unique()
    name_to_id = {n: i + len(account_to_id) + len(mobile_to_id) for i, n in enumerate(names)}
    
    pins = df['sender_pincode'].dropna().unique()
    pin_to_id = {p: i + len(account_to_id) + len(mobile_to_id) + len(name_to_id)
                 for i, p in enumerate(pins)}
    
    num_nodes = len(account_to_id) + len(mobile_to_id) + len(name_to_id) + len(pin_to_id)
    num_accounts = len(account_to_id)
    
    print(f"✓ Total nodes: {num_nodes}")
    print(f"  ├─ Accounts: {num_accounts}")
    print(f"  ├─ Mobiles: {len(mobiles)}")
    print(f"  ├─ Names: {len(names)}")
    print(f"  └─ Pincodes: {len(pins)}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Build edges
    # ──────────────────────────────────────────────────────────────────────────
    print("[3.2] Building edges...")
    
    edge_src = []
    edge_dst = []
    edge_attr = []
    
    channel_map = {"UPI": 0, "IMPS": 1, "WEB": 2, "APP": 3, "ATM": 4}
    
    # Transaction edges (account → account)
    min_time = df['timestamp'].min()
    df['time_norm'] = df['timestamp'] - min_time
    
    for _, row in df.iterrows():
        s = account_to_id[row['sender_account']]
        r = account_to_id.get(row['receiver_account'])
        if r is None:
            continue
        
        edge_src.append(s)
        edge_dst.append(r)
        edge_attr.append([
            np.log1p(row['amount']),
            row['time_norm'] / 1e6,
            channel_map.get(row['channel'], 0)
        ])
    
    # Identity edges
    for _, row in df.iterrows():
        acc = account_to_id[row['sender_account']]
        
        if pd.notna(row['sender_mobile']) and row['sender_mobile'] in mobile_to_id:
            edge_src.append(acc)
            edge_dst.append(mobile_to_id[row['sender_mobile']])
            edge_attr.append([0, 0, 0])
        
        if pd.notna(row['sender_name']) and row['sender_name'] in name_to_id:
            edge_src.append(acc)
            edge_dst.append(name_to_id[row['sender_name']])
            edge_attr.append([0, 0, 0])
        
        if pd.notna(row['sender_pincode']) and row['sender_pincode'] in pin_to_id:
            edge_src.append(acc)
            edge_dst.append(pin_to_id[row['sender_pincode']])
            edge_attr.append([0, 0, 0])
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    print(f"✓ Total edges: {len(edge_src)}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Compute node features (12 features per node)
    # ──────────────────────────────────────────────────────────────────────────
    print("[3.3] Computing node features...")
    
    x = np.zeros((num_nodes, 12))
    
    # Feature 0-1: Out-degree, In-degree
    for s, d in zip(edge_src, edge_dst):
        x[s][0] += 1
        x[d][1] += 1
    
    # Feature 2: Transaction count
    txn_counts = df['sender_account'].value_counts()
    for acc, count in txn_counts.items():
        if acc in account_to_id:
            x[account_to_id[acc]][2] = count
    
    # Feature 3: Unique receivers (fan-out signal)
    uniq_recv = df.groupby('sender_account')['receiver_account'].nunique()
    for acc, val in uniq_recv.items():
        if acc in account_to_id:
            x[account_to_id[acc]][3] = val
    
    # Feature 4-5: Amount mean & std (structuring signal)
    amount_mean = df.groupby('sender_account')['amount'].mean()
    amount_std = df.groupby('sender_account')['amount'].std().fillna(0)
    for acc in amount_mean.index:
        if acc in account_to_id:
            x[account_to_id[acc]][4] = amount_mean[acc]
            x[account_to_id[acc]][5] = amount_std[acc]
    
    # Feature 6-7: Time gap, Burst score (rapid transactions)
    df['time_diff'] = df.groupby('sender_account')['timestamp'].diff().fillna(0)
    time_gap = df.groupby('sender_account')['time_diff'].mean()
    for acc, val in time_gap.items():
        if acc in account_to_id:
            x[account_to_id[acc]][6] = val
    
    burst = df.groupby('sender_account').apply(lambda g: (g['time_diff'] < 60).sum())
    for acc, val in burst.items():
        if acc in account_to_id:
            x[account_to_id[acc]][7] = val
    
    # Feature 8: Mobile shared count (mule detection)
    mobile_count = df.groupby('sender_mobile')['sender_account'].nunique()
    for _, row in df.drop_duplicates('sender_account').iterrows():
        acc = account_to_id.get(row['sender_account'])
        mob = row['sender_mobile']
        if acc is not None and pd.notna(mob) and mob in mobile_count:
            x[acc][8] = mobile_count[mob]
    
    # Feature 9: Channel diversity (fragmentation signal)
    channel_div = df.groupby('sender_account')['channel'].nunique()
    for acc, val in channel_div.items():
        if acc in account_to_id:
            x[account_to_id[acc]][9] = val
    
    # Feature 10: Receiver diversity
    recv_div = df.groupby('sender_account')['receiver_account'].nunique()
    for acc, val in recv_div.items():
        if acc in account_to_id:
            x[account_to_id[acc]][10] = val
    
    # Feature 11: Pincode risk (high-risk pincodes)
    for _, row in df.drop_duplicates('sender_account').iterrows():
        acc = account_to_id.get(row['sender_account'])
        if acc is not None and row['sender_pincode'] in HIGH_RISK_PINCODES:
            x[acc][11] = 1.0
    
    x = torch.tensor(x, dtype=torch.float)
    
    print(f"✓ Node features shape: {x.shape}")
    
    # Create graph object
    graph_data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    return graph_data, account_to_id, num_accounts


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ STEP 4: LOAD MODEL & RUN INFERENCE                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_inference(graph_data, account_to_id, num_accounts, df):
    """Load model and run inference."""
    print("\n" + "="*80)
    print("🎯 STEP 4: LOAD MODEL & RUN INFERENCE")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[4.1] Using device: {device}")
    
    # Load model
    print("[4.2] Loading trained model...")
    model = GNNModel(graph_data.num_node_features)
    model.load_state_dict(torch.load('models/best_fraud_gnn_model.pth', weights_only=True))
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Run inference
    print("[4.3] Running inference...")
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
        probs = torch.softmax(logits, dim=1)
    
    fraud_scores = probs[:num_accounts, 1].cpu().numpy()
    
    # Create predictions
    id_to_account = {v: k for k, v in account_to_id.items()}
    predictions = []
    
    for acc_id in range(num_accounts):
        account = id_to_account[acc_id]
        score = fraud_scores[acc_id]
        
        if score > RISK_THRESHOLDS['HIGH']:
            risk_level = 'HIGH RISK 🔴'
        elif score > RISK_THRESHOLDS['MEDIUM']:
            risk_level = 'MEDIUM RISK 🟡'
        else:
            risk_level = 'NORMAL 🟢'
        
        predictions.append({
            'account_number': account,
            'fraud_score': score,
            'risk_level': risk_level,
            'account_id': acc_id
        })
    
    pred_df = pd.DataFrame(predictions).sort_values('fraud_score', ascending=False)
    
    print(f"\n✓ Inference completed")
    print(f"✓ High Risk: {(pred_df['fraud_score'] > RISK_THRESHOLDS['HIGH']).sum()}")
    print(f"✓ Medium Risk: {((pred_df['fraud_score'] > RISK_THRESHOLDS['MEDIUM']) & (pred_df['fraud_score'] <= RISK_THRESHOLDS['HIGH'])).sum()}")
    print(f"✓ Normal: {(pred_df['fraud_score'] <= RISK_THRESHOLDS['MEDIUM']).sum()}")
    
    return pred_df, fraud_scores, id_to_account


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ STEP 5: OUTPUT PREDICTIONS & EXPLANATIONS                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def generate_explanations(pred_df, df, account_to_id):
    """Generate detailed explanations for flagged accounts."""
    print("\n" + "="*80)
    print("🎯 STEP 5: PREDICTIONS & EXPLANATIONS")
    print("="*80)
    
    # High-risk accounts
    high_risk = pred_df[pred_df['fraud_score'] > RISK_THRESHOLDS['HIGH']].head(10)
    
    print("\n" + "🔴 " * 40)
    print("TOP 10 HIGH-RISK ACCOUNTS")
    print("🔴 " * 40)
    print("\n" + pred_df[['account_number', 'fraud_score', 'risk_level']].head(10).to_string(index=False))
    
    # Generate explanations for top flagged accounts
    print("\n\n" + "="*80)
    print("📋 DETAILED EXPLANATIONS FOR FLAGGED ACCOUNTS")
    print("="*80)
    
    explanations = {}
    
    for idx, row in high_risk.iterrows():
        account = row['account_number']
        score = row['fraud_score']
        acc_txns = df[df['sender_account'] == account]
        
        if len(acc_txns) == 0:
            continue
        
        # Compute signals
        burst_score = (acc_txns['timestamp'].diff() < 60).sum()
        receiver_diversity = acc_txns['receiver_account'].nunique()
        channel_diversity = acc_txns['channel'].nunique()
        unique_receivers = acc_txns['receiver_account'].nunique()
        
        # Mobile sharing
        my_mobile = acc_txns.iloc[0]['sender_mobile']
        mobile_sharing_count = len(df[df['sender_mobile'] == my_mobile]['sender_account'].unique()) - 1
        
        # Pincode risk
        my_pincode = acc_txns.iloc[0]['sender_pincode']
        is_high_risk_pincode = my_pincode in HIGH_RISK_PINCODES
        
        # Pattern detection
        patterns = acc_txns['fraud_pattern'].unique() if 'fraud_pattern' in acc_txns.columns else []
        
        explanation = f"""
┌─ Account: {account}
├─ Fraud Score: {score:.4f} (Threshold: {RISK_THRESHOLDS['HIGH']})
├─ Transactions: {len(acc_txns)}
└─ Risk Signals:
   ├─ ⚡ Burst Activity: {burst_score} rapid txns (<60s) — {'⚠️ VERY HIGH' if burst_score > 5 else '⚠️ HIGH' if burst_score > 2 else '✓ Normal'}
   ├─ 🌐 Receiver Diversity: {receiver_diversity} unique — {'⚠️ HIGH FAN-OUT' if receiver_diversity > 5 else '✓ Normal'}
   ├─ 🔀 Channel Diversity: {channel_diversity} channels — {'⚠️ FRAGMENTATION' if channel_diversity > 2 else '✓ Normal'}
   ├─ 📱 Shared Mobile Accounts: {mobile_sharing_count} — {'⚠️ MULE DETECTION' if mobile_sharing_count > 0 else '✓ Unique'}
   ├─ 📍 High-Risk Pincode: {'🔴 YES' if is_high_risk_pincode else '✓ No'} ({my_pincode})
   └─ 💰 Amount Pattern: {acc_txns['amount'].min()}-{acc_txns['amount'].max()} (std: {acc_txns['amount'].std():.0f}) — {'⚠️ SUSPICIOUSLY CONSISTENT' if acc_txns['amount'].std() < 5000 else '✓ Varied'}
"""
        explanations[account] = explanation
    
    # Print top 5 detailed explanations
    for i, (account, exp) in enumerate(list(explanations.items())[:5]):
        print(exp)
    
    # Save full results
    output_df = pred_df[['account_number', 'fraud_score', 'risk_level']].copy()
    output_df.to_csv('outputs/visualizations/fraud_predictions.csv', index=False)
    print(f"\n✓ Predictions saved to outputs/visualizations/fraud_predictions.csv")
    
    return pred_df, explanations


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ STEP 6-7: VISUALIZATIONS                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def create_visualizations(pred_df, df, account_to_id):
    """Create visualization plots."""
    print("\n" + "="*80)
    print("🎯 STEP 6: VISUALIZATIONS")
    print("="*80)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        import seaborn as sns
    except ImportError:
        print("⚠ Matplotlib/Seaborn not available - skipping visualizations")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 12)
    
    fig = plt.figure(figsize=(18, 14))
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Risk Score Distribution
    # ──────────────────────────────────────────────────────────────────────────
    ax1 = plt.subplot(2, 3, 1)
    scores = pred_df['fraud_score'].values
    
    ax1.hist(scores, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(RISK_THRESHOLDS['MEDIUM'], color='orange', linestyle='--', linewidth=2, label=f"Medium ({RISK_THRESHOLDS['MEDIUM']})")
    ax1.axvline(RISK_THRESHOLDS['HIGH'], color='red', linestyle='--', linewidth=2, label=f"High ({RISK_THRESHOLDS['HIGH']})")
    ax1.set_xlabel('Fraud Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Account Count', fontsize=11, fontweight='bold')
    ax1.set_title('📊 Fraud Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 2. Risk Level Pie Chart
    # ──────────────────────────────────────────────────────────────────────────
    ax2 = plt.subplot(2, 3, 2)
    
    high = (pred_df['fraud_score'] > RISK_THRESHOLDS['HIGH']).sum()
    medium = ((pred_df['fraud_score'] > RISK_THRESHOLDS['MEDIUM']) & 
              (pred_df['fraud_score'] <= RISK_THRESHOLDS['HIGH'])).sum()
    normal = (pred_df['fraud_score'] <= RISK_THRESHOLDS['MEDIUM']).sum()
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    sizes = [high, medium, normal]
    labels = [f'HIGH\n({high})', f'MEDIUM\n({medium})', f'NORMAL\n({normal})']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('🎯 Risk Level Distribution', fontsize=12, fontweight='bold')
    
    # ──────────────────────────────────────────────────────────────────────────
    # 3. Top 10 High Risk Accounts
    # ──────────────────────────────────────────────────────────────────────────
    ax3 = plt.subplot(2, 3, 3)
    
    top_fraud = pred_df.head(10).copy()
    top_fraud['display_account'] = top_fraud['account_number'].str[-6:]
    
    bars = ax3.barh(range(len(top_fraud)), top_fraud['fraud_score'].values, color='#e74c3c')
    ax3.axvline(RISK_THRESHOLDS['HIGH'], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_yticks(range(len(top_fraud)))
    ax3.set_yticklabels(top_fraud['display_account'].values, fontsize=9)
    ax3.set_xlabel('Fraud Score', fontsize=11, fontweight='bold')
    ax3.set_title('🔴 Top 10 High-Risk Accounts', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.0)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add score labels
    for i, (idx, row) in enumerate(top_fraud.iterrows()):
        ax3.text(row['fraud_score'] + 0.02, i, f"{row['fraud_score']:.3f}", 
                va='center', fontsize=8)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 4. Burst Activity vs Fraud Score
    # ──────────────────────────────────────────────────────────────────────────
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate burst scores
    burst_data = []
    for account in pred_df['account_number'].values[:100]:
        acc_txns = df[df['sender_account'] == account]
        if len(acc_txns) > 0:
            burst = (acc_txns['timestamp'].diff() < 60).sum()
            score = pred_df[pred_df['account_number'] == account]['fraud_score'].values[0]
            burst_data.append({'burst': burst, 'score': score})
    
    burst_df = pd.DataFrame(burst_data)
    scatter = ax4.scatter(burst_df['burst'], burst_df['score'], 
                         c=burst_df['score'], cmap='RdYlGn_r', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.axhline(RISK_THRESHOLDS['HIGH'], color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Burst Activity (rapid txns <60s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Fraud Score', fontsize=11, fontweight='bold')
    ax4.set_title('⚡ Burst Activity Signal', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Fraud Score')
    
    # ──────────────────────────────────────────────────────────────────────────
    # 5. Receiver Diversity vs Fraud Score
    # ──────────────────────────────────────────────────────────────────────────
    ax5 = plt.subplot(2, 3, 5)
    
    diversity_data = []
    for account in pred_df['account_number'].values[:100]:
        acc_txns = df[df['sender_account'] == account]
        if len(acc_txns) > 0:
            diversity = acc_txns['receiver_account'].nunique()
            score = pred_df[pred_df['account_number'] == account]['fraud_score'].values[0]
            diversity_data.append({'diversity': diversity, 'score': score})
    
    div_df = pd.DataFrame(diversity_data)
    scatter = ax5.scatter(div_df['diversity'], div_df['score'], 
                         c=div_df['score'], cmap='RdYlGn_r', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax5.axhline(RISK_THRESHOLDS['HIGH'], color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax5.set_xlabel('Unique Receivers (Fan-Out)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Fraud Score', fontsize=11, fontweight='bold')
    ax5.set_title('🌐 Fan-Out (Receiver Diversity)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Fraud Score')
    
    # ──────────────────────────────────────────────────────────────────────────
    # 6. Transaction Timeline for Selected Fraudster
    # ──────────────────────────────────────────────────────────────────────────
    ax6 = plt.subplot(2, 3, 6)
    
    # Get the highest risk account
    top_account = pred_df.iloc[0]['account_number']
    top_txns = df[df['sender_account'] == top_account].sort_values('timestamp')
    
    if len(top_txns) > 0:
        # Convert timestamps to relative hours
        min_ts = top_txns['timestamp'].min()
        hours = (top_txns['timestamp'] - min_ts) / 3600
        
        colors_by_channel = {'UPI': '#3498db', 'IMPS': '#2ecc71', 'WEB': '#9b59b6', 'APP': '#e74c3c', 'ATM': '#f39c12'}
        
        for channel in top_txns['channel'].unique():
            mask = top_txns['channel'] == channel
            ax6.scatter(hours[mask], top_txns[mask]['amount'].values, 
                       label=channel, s=100, alpha=0.7, 
                       color=colors_by_channel.get(channel, '#95a5a6'),
                       edgecolors='black', linewidth=0.5)
        
        ax6.set_xlabel('Time (hours from first transaction)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Transaction Amount (₹)', fontsize=11, fontweight='bold')
        ax6.set_title(f'📈 Timeline: Top Fraud Account\n({top_account[-6:]})', fontsize=12, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: outputs/visualizations/fraud_detection_analysis.png")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Additional: Pattern Detection Heatmap
    # ──────────────────────────────────────────────────────────────────────────
    if 'fraud_pattern' in df.columns:
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        pattern_risk = pd.DataFrame()
        for pattern in df['fraud_pattern'].unique():
            pattern_accounts = df[df['fraud_pattern'] == pattern]['sender_account'].unique()
            pattern_scores = pred_df[pred_df['account_number'].isin(pattern_accounts)]['fraud_score']
            pattern_risk = pd.concat([pattern_risk, pd.DataFrame({
                'Pattern': pattern,
                'Avg Fraud Score': [pattern_scores.mean()],
                'Account Count': [len(pattern_accounts)],
                'Max Score': [pattern_scores.max()]
            })], ignore_index=True)
        
        pattern_risk = pattern_risk.sort_values('Avg Fraud Score', ascending=False)
        
        bars = ax.barh(pattern_risk['Pattern'], pattern_risk['Avg Fraud Score'], color='#3498db')
        ax.axvline(RISK_THRESHOLDS['HIGH'], color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
        ax.axvline(RISK_THRESHOLDS['MEDIUM'], color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold')
        ax.set_xlabel('Average Fraud Score', fontsize=12, fontweight='bold')
        ax.set_title('🎯 Fraud Patterns vs Risk Score', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(pattern_risk.iterrows()):
            ax.text(row['Avg Fraud Score'] + 0.02, i, 
                   f"{row['Avg Fraud Score']:.3f} (n={row['Account Count']:.0f})", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/fraud_patterns_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: outputs/visualizations/fraud_patterns_analysis.png")
    
    print("✓ All visualizations created successfully!")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ MAIN PIPELINE                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    """Execute the complete fraud detection demo pipeline."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         🎯 FRAUD DETECTION DEMO PIPELINE — End-to-End GNN Inference       ║
║                                                                            ║
║  This pipeline demonstrates complete fraud detection workflow:             ║
║   1️⃣  Generate synthetic test dataset with fraud patterns                 ║
║   2️⃣  Parse narration fields                                              ║
║   3️⃣  Build heterogeneous graph with node features                        ║
║   4️⃣  Load trained GNN model                                              ║
║   5️⃣  Run inference & classify risk levels                                ║
║   6️⃣  Output predictions & detailed explanations                          ║
║   7️⃣  Visualize patterns and graph                                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Step 1: Generate test dataset
    test_df, fraud_accs, normal_accs = generate_test_dataset(
        num_accounts=5000, 
        num_transactions=15000
    )
    
    # Step 2: Parse narration
    test_df = parse_test_data(test_df)
    
    # Step 3: Build graph
    graph_data, account_to_id, num_accounts = build_test_graph(test_df)
    
    # Step 4: Load model & run inference
    pred_df, fraud_scores, id_to_account = run_inference(
        graph_data, account_to_id, num_accounts, test_df
    )
    
    # Step 5: Generate explanations
    pred_df, explanations = generate_explanations(pred_df, test_df, account_to_id)
    
    # Step 6-7: Create visualizations
    create_visualizations(pred_df, test_df, account_to_id)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"\n⏱  Total Time: {elapsed:.2f} seconds")
    print(f"\n📊 Summary:")
    print(f"   Total Accounts Analyzed: {len(pred_df)}")
    print(f"   High-Risk Accounts: {(pred_df['fraud_score'] > RISK_THRESHOLDS['HIGH']).sum()}")
    print(f"   Medium-Risk Accounts: {((pred_df['fraud_score'] > RISK_THRESHOLDS['MEDIUM']) & (pred_df['fraud_score'] <= RISK_THRESHOLDS['HIGH'])).sum()}")
    print(f"   Normal Accounts: {(pred_df['fraud_score'] <= RISK_THRESHOLDS['MEDIUM']).sum()}")
    
    print(f"\n📁 Output Files:")
    print(f"   ✓ test_transactions.csv — Full test dataset with fraud labels")
    print(f"   ✓ fraud_predictions.csv — Model predictions for all accounts")
    print(f"   ✓ fraud_detection_analysis.png — Comprehensive visualizations")
    print(f"   ✓ fraud_patterns_analysis.png — Fraud pattern analysis")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • Model detected {(pred_df['fraud_score'] > RISK_THRESHOLDS['HIGH']).sum()} high-risk accounts")
    print(f"   • Detection threshold (HIGH): {RISK_THRESHOLDS['HIGH']} | (MEDIUM): {RISK_THRESHOLDS['MEDIUM']}")
    print(f"   • Demo ready for presentation/validation")

if __name__ == '__main__':
    main()
