"""
Synthetic Banking Fraud Dataset Generator v2 — Enhanced for GNN Generalization
==============================================================================
Improvements over v1:
  1. Receiver disambiguation — UPI/APP narrations embed receiver_account
  2. Semi-normal burst behaviour  — 5-10% of normals do 2-4 quick txns (label=0)
  3. Weak / borderline fraud      — 20-30% soft fan-out with 2-3 receivers
  4. Structuring variability      — 30% of cases use ₹30k-₹60k range
  5. Bidirectional / cyclic flows — A→B→A and A↔B repeated patterns
  6. Stronger identity graph      — shared name+pincode clusters transact
  7. txn_id column                — unique per row, enables temporal ordering
  8. All v1 patterns preserved
  9. Fraud ratio kept at 15-25%
 10. Near-structuring normals     — similar amounts, label=0 (prevents overfitting)
 11. Cross-cluster noise          — normal account → fraud-cluster node (label=0)
 12. Fraud timestamp jitter       — ±5s noise on every fraud ts (prevents spacing memorisation)

Output: transactions_v2.csv  (same directory as this script)
"""

import random, csv, os, uuid, datetime, collections, sys

# Force UTF-8 output so Unicode arrows don't crash on Windows console
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

# ── Epoch window: 2025-01-01 → 2026-01-01 UTC ────────────────────────────────
EPOCH_START = 1_735_689_600
EPOCH_END   = 1_767_225_600

# ── High-risk pincodes ────────────────────────────────────────────────────────
HIGH_RISK_PINCODES = {
    600003, 600007, 600011, 600019, 600025, 600031,
    600037, 600043, 600051, 600063, 600071, 600082,
    600090, 600099,
}
ALL_PINCODES    = list(range(600001, 600101))
NORMAL_PINCODES = [p for p in ALL_PINCODES if p not in HIGH_RISK_PINCODES]

CHANNELS = ["UPI", "IMPS", "WEB", "APP", "ATM"]
BANKS    = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "PNB", "BOB", "CANARA"]
CITIES   = ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy",
            "Vellore", "Erode", "Tirunelveli", "Thanjavur", "Nagercoil"]
ATM_IDS  = [f"ATM{i:05d}" for i in range(1, 501)]

MALE_NAMES   = ["Arjun","Karthik","Vikram","Rajesh","Suresh","Ramesh","Dinesh",
                "Ganesh","Venkat","Prasad","Sathish","Murugan","Selvam","Mani",
                "Rajan","Kumar","Sakthivel","Anand","Bala","Chandru","Deepak",
                "Elan","Farhan","Gopi","Hari","Iyappan","Jobi","Kiran","Lokesh",
                "Manoj","Naveen","Ojas","Prem","Qasim","Rohan","Siva","Tamil",
                "Uday","Vivek","Wasim","Xavier","Yashwant","Zubair","Ajith",
                "Balaji","Chethan","David","Eswaran"]
FEMALE_NAMES = ["Priya","Kavya","Lakshmi","Meena","Nithya","Rani","Saranya",
                "Thenmozhi","Uma","Vani","Abinaya","Bhavani","Chitra","Divya",
                "Ezhilarasi","Gowri","Hema","Indira","Janani","Kamala","Latha",
                "Malathi","Nalini","Oviya","Padmaja","Radha","Swetha","Thamarai",
                "Usha","Vanitha","Yamuna","Zeenath","Akila","Brindha","Deepa",
                "Elakiya","Fathima","Gomathi","Honey","Isai"]
ALL_NAMES = MALE_NAMES + FEMALE_NAMES

# ══════════════════════════════════════════════════════════════════════════════
# 1. ACCOUNT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def gen_mobile():
    return random.choice(["9","8","7","6"]) + str(random.randint(100_000_000, 999_999_999))

def gen_acc_num(used):
    while True:
        n = str(random.randint(100_000_000_000, 999_999_999_999))
        if n not in used:
            used.add(n)
            return n

NUM_ACCOUNTS = 10_000
print("[1/5] Generating accounts …")

# ── Improvement 6: Shared name + pincode clusters ─────────────────────────────
# Create ~60 "cluster seeds": same name AND same pincode, 3-6 accounts each
CLUSTER_SEEDS = [(random.choice(ALL_NAMES), random.choice(ALL_PINCODES))
                 for _ in range(60)]
cluster_schedule = []
for seed in CLUSTER_SEEDS:
    size = random.randint(3, 6)
    cluster_schedule.extend([seed] * size)
random.shuffle(cluster_schedule)

# Shared mobile pool (10-12% of accounts share a mobile with 1-4 others)
shared_mobile_pool = [gen_mobile() for _ in range(int(NUM_ACCOUNTS * 0.11))]
shared_name_pool   = random.sample(ALL_NAMES, k=40)

acc_by_idx = {}
used_acc   = set()

for i in range(NUM_ACCOUNTS):
    # Cluster override (first N accounts get cluster identity)
    if cluster_schedule:
        forced_name, forced_pin = cluster_schedule.pop()
    else:
        forced_name, forced_pin = None, None

    # Mobile: 11% chance reuse from shared pool
    if shared_mobile_pool and random.random() < 0.11:
        mobile = random.choice(shared_mobile_pool)
    else:
        mobile = gen_mobile()

    name = forced_name if forced_name else (
        random.choice(shared_name_pool) if random.random() < 0.07
        else random.choice(ALL_NAMES)
    )
    pin = forced_pin if forced_pin else random.choice(ALL_PINCODES)

    acc_by_idx[i] = {
        "account_number":       gen_acc_num(used_acc),
        "name":                  name,
        "mobile":                mobile,
        "pincode":               pin,
        "account_product_type":  random.choice(["Savings", "Current"]),
    }

# ── Improvement 1: deterministic mobile → primary account mapping ─────────────
mobile_primary = {}          # mobile -> account_number (first-seen = primary)
for idx, acc in acc_by_idx.items():
    mob = acc["mobile"]
    if mob not in mobile_primary:
        mobile_primary[mob] = acc["account_number"]

# Indexes
mobile_to_indices = collections.defaultdict(list)
name_to_indices   = collections.defaultdict(list)
pin_to_indices    = collections.defaultdict(list)
for idx, acc in acc_by_idx.items():
    mobile_to_indices[acc["mobile"]].append(idx)
    name_to_indices[acc["name"]].append(idx)
    pin_to_indices[acc["pincode"]].append(idx)

# Groups for nesting (shared mobile ≥2)
shared_mob_groups = [v for v in mobile_to_indices.values() if len(v) >= 2]
# Groups for identity-cluster transactions (shared name+pin ≥2)
shared_cluster_groups = []
for idx in range(NUM_ACCOUNTS):
    a = acc_by_idx[idx]
    grp = [j for j in pin_to_indices[a["pincode"]]
           if acc_by_idx[j]["name"] == a["name"] and j != idx]
    if len(grp) >= 1:
        shared_cluster_groups.append([idx] + grp)

print(f"    {NUM_ACCOUNTS} accounts | {len(shared_mob_groups)} shared-mobile groups | "
      f"{len(shared_cluster_groups)} name+pin clusters")

# ══════════════════════════════════════════════════════════════════════════════
# 2. NARRATION BUILDERS  (Improvement 1: receiver_account embedded)
# ══════════════════════════════════════════════════════════════════════════════

def make_narration(channel, receiver):
    r = receiver
    if channel == "UPI":
        # v2: embed receiver account number for disambiguation
        return f"UPI/{r['mobile']}/{r['name']}/{r['account_number']}/{random.choice(BANKS)}"
    elif channel == "IMPS":
        return f"IMPS/{r['account_number']}/{r['name']}"
    elif channel == "WEB":
        return f"WEB-TRF/{r['account_number']}/{r['name']}"
    elif channel == "APP":
        # v2: embed receiver account number
        return f"APP-P2P/{r['mobile']}/{r['name']}/{r['account_number']}"
    elif channel == "ATM":
        return f"ATM-WDL/{random.choice(ATM_IDS)}/{random.choice(CITIES)}"
    return "MISC"

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSACTION FACTORY
# ══════════════════════════════════════════════════════════════════════════════

_txn_counter = 0
def make_tx(sender_idx, receiver_idx, channel, amount, timestamp, label):
    global _txn_counter
    _txn_counter += 1
    s = acc_by_idx[sender_idx]
    r = acc_by_idx[receiver_idx]
    # Improvement 12: add ±5s jitter to every fraud timestamp
    # This prevents model from memorising exact inter-arrival spacing
    if label == 1:
        timestamp = max(EPOCH_START, int(timestamp) + random.randint(-5, 5))
    return {
        "txn_id":               f"TXN{_txn_counter:08d}",
        "account_number":       s["account_number"],
        "name":                 s["name"],
        "mobile":               s["mobile"],
        "pincode":              s["pincode"],
        "account_product_type": s["account_product_type"],
        "narration":            make_narration(channel, r),
        "amount":               int(amount),
        "timestamp":            int(timestamp),
        "label":                label,
    }

def rand_ts():
    return random.randint(EPOCH_START, EPOCH_END)

def diff_idx(a):
    b = random.randint(0, NUM_ACCOUNTS - 1)
    return b if b != a else (b + 1) % NUM_ACCOUNTS

# ══════════════════════════════════════════════════════════════════════════════
# 4. NORMAL TRANSACTIONS  (~96,000 base)
# ══════════════════════════════════════════════════════════════════════════════

print("[2/5] Generating normal transactions …")

NORMAL_TARGET = 96_000
normal_txs = []

# 4a. Fully random normal (90% of normal budget)
plain_target = int(NORMAL_TARGET * 0.90)
while len(normal_txs) < plain_target:
    s = random.randint(0, NUM_ACCOUNTS - 1)
    r = diff_idx(s)
    normal_txs.append(make_tx(s, r,
        random.choice(CHANNELS),
        random.randint(100, 50_000),
        rand_ts(), 0))

# 4b. Improvement 2: Semi-normal burst (5-10% of accounts, 2-4 txns in 30-120s)
burst_accounts = random.sample(range(NUM_ACCOUNTS),
                               k=int(NUM_ACCOUNTS * random.uniform(0.05, 0.10)))
for acc_idx in burst_accounts:
    n = random.randint(2, 4)
    t0 = rand_ts()
    for j in range(n):
        ts = t0 + random.randint(0, 120) * j
        r  = diff_idx(acc_idx)
        normal_txs.append(make_tx(acc_idx, r,
            random.choice(CHANNELS),
            random.randint(500, 20_000),
            ts, 0))

# 4c. Improvement 5: Bidirectional / cyclic NORMAL (A→B→A)
bidi_normal = int(NORMAL_TARGET * 0.03)
for _ in range(bidi_normal):
    a = random.randint(0, NUM_ACCOUNTS - 1)
    b = diff_idx(a)
    t0  = rand_ts()
    amt = random.randint(500, 10_000)
    normal_txs.append(make_tx(a, b, random.choice(CHANNELS), amt, t0, 0))
    normal_txs.append(make_tx(b, a, random.choice(CHANNELS),
                               int(amt * random.uniform(0.80, 1.20)),
                               t0 + random.randint(300, 3600), 0))

# 4d. Improvement 10: Near-structuring NORMALS (label=0, similar amounts, spread times)
# Prevents model from learning "similar amounts always = fraud"
NEAR_STRUCT_ACCOUNTS = random.sample(range(NUM_ACCOUNTS),
                                     k=int(NUM_ACCOUNTS * 0.05))  # 5% of accounts
for acc_idx in NEAR_STRUCT_ACCOUNTS:
    n = random.randint(2, 3)
    base_amt = random.randint(20_000, 48_000)
    t0 = rand_ts()
    elapsed = 0
    for _ in range(n):
        # Spread over minutes-to-hours (not tightly clustered like fraud)
        elapsed += random.randint(600, 7_200)
        jitter = random.randint(-2_000, 2_000)
        amt = max(100, base_amt + jitter)
        r = diff_idx(acc_idx)
        normal_txs.append(make_tx(acc_idx, r, random.choice(CHANNELS), amt,
                                   t0 + elapsed, 0))

print(f"    Normal transactions : {len(normal_txs):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FRAUD PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

print("[3/5] Injecting fraud patterns …")
fraud_txs = []

# ── 5.1  FAN-OUT  (strong: 5-10 receivers, 30-60 sec) ────────────────────────
for _ in range(600):
    s = random.randint(0, NUM_ACCOUNTS - 1)
    n = random.randint(5, 10)
    receivers = random.sample([i for i in range(NUM_ACCOUNTS) if i != s], n)
    t0    = rand_ts()
    delta = random.uniform(30, 60) / n
    for j, r in enumerate(receivers):
        amt = int(random.randint(500, 15_000) * random.uniform(0.95, 1.05))
        fraud_txs.append(make_tx(s, r, random.choice(CHANNELS),
                                  amt, int(t0 + j * delta), 1))

# ── 5.1b  SOFT FAN-OUT  (Improvement 3: 2-3 receivers, slightly spread) ───────
soft_fanout = int(600 * 0.25)                                  # 25% extra soft
for _ in range(soft_fanout):
    s = random.randint(0, NUM_ACCOUNTS - 1)
    n = random.randint(2, 3)
    receivers = random.sample([i for i in range(NUM_ACCOUNTS) if i != s], n)
    t0 = rand_ts()
    for j, r in enumerate(receivers):
        ts = int(t0 + random.randint(60, 300) * j)             # slightly spread
        amt = int(random.randint(1_000, 20_000) * random.uniform(0.90, 1.10))
        fraud_txs.append(make_tx(s, r, random.choice(CHANNELS), amt, ts, 1))

# ── 5.2  FAN-IN  (5-10 senders → 1 receiver) ─────────────────────────────────
for _ in range(600):
    r = random.randint(0, NUM_ACCOUNTS - 1)
    n = random.randint(5, 10)
    senders = random.sample([i for i in range(NUM_ACCOUNTS) if i != r], n)
    t0    = rand_ts()
    delta = random.uniform(20, 60) / n
    for j, s in enumerate(senders):
        fraud_txs.append(make_tx(s, r, random.choice(CHANNELS),
                                  random.randint(1_000, 20_000),
                                  int(t0 + j * delta), 1))

# ── 5.2b  SOFT FAN-IN ─────────────────────────────────────────────────────────
soft_fanin = int(600 * 0.25)
for _ in range(soft_fanin):
    r = random.randint(0, NUM_ACCOUNTS - 1)
    n = random.randint(2, 3)
    senders = random.sample([i for i in range(NUM_ACCOUNTS) if i != r], n)
    t0 = rand_ts()
    for j, s in enumerate(senders):
        ts = int(t0 + random.randint(60, 300) * j)
        fraud_txs.append(make_tx(s, r, random.choice(CHANNELS),
                                  random.randint(500, 15_000), ts, 1))

# ── 5.3  CHAIN  (A→B→C→D in minutes) ─────────────────────────────────────────
for _ in range(500):
    length = random.randint(3, 6)
    chain  = random.sample(range(NUM_ACCOUNTS), length + 1)
    t0, elapsed = rand_ts(), 0
    amt = random.randint(5_000, 50_000)
    for k in range(length):
        elapsed += random.randint(30, 120)
        hop_amt  = max(100, int(amt * random.uniform(0.85, 1.0)))
        fraud_txs.append(make_tx(chain[k], chain[k+1],
                                  random.choice(["UPI","IMPS","WEB","APP"]),
                                  hop_amt, t0 + elapsed, 1))

# ── 5.4  STRUCTURING  (Improvement 4: 30% use ₹30k-₹60k range) ──────────────
for _ in range(500):
    s = random.randint(0, NUM_ACCOUNTS - 1)
    n = random.randint(3, 7)
    t0, elapsed = rand_ts(), 0
    for _ in range(n):
        elapsed += random.randint(30, 300)
        if random.random() < 0.30:                             # wider range
            base = random.randint(30_000, 60_000)
            jitter = random.randint(500, 2_000)
        else:                                                   # classic range
            base = random.randint(45_000, 49_000)
            jitter = random.randint(0, 500)
        amt = base + random.choice([-1, 1]) * jitter
        r = diff_idx(s)
        fraud_txs.append(make_tx(s, r, random.choice(CHANNELS),
                                  amt, t0 + elapsed, 1))

# ── 5.5  FRAGMENTATION  (same sender→receiver, ≥3 channels in 120s) ──────────
for _ in range(500):
    s = random.randint(0, NUM_ACCOUNTS - 1)
    r = diff_idx(s)
    t0, elapsed = rand_ts(), 0
    channels_used = random.sample(CHANNELS, k=random.randint(3, 5))
    for ch in channels_used:
        elapsed += random.randint(10, 60)
        fraud_txs.append(make_tx(s, r, ch,
                                  random.randint(1_000, 30_000),
                                  t0 + elapsed, 1))

# ── 5.6  NESTING  (shared-mobile accounts transact with each other) ───────────
nest_pool = list(shared_mob_groups)
for _ in range(450):
    if nest_pool:
        grp = random.choice(nest_pool)
    else:
        grp = random.sample(range(NUM_ACCOUNTS), 2)
    if len(grp) < 2:
        continue
    t0, elapsed = rand_ts(), 0
    for _ in range(random.randint(2, 5)):
        a, b = random.sample(grp, 2)
        elapsed += random.randint(30, 300)
        fraud_txs.append(make_tx(a, b, random.choice(CHANNELS),
                                  random.randint(1_000, 40_000),
                                  t0 + elapsed, 1))

# ── 5.7  JURISDICTION RISK  (high-risk pincode senders) ──────────────────────
# Mark enough accounts as high-risk
high_risk_idxs = [i for i,a in acc_by_idx.items()
                  if a["pincode"] in HIGH_RISK_PINCODES]
if len(high_risk_idxs) < 500:
    extra = random.sample([i for i in range(NUM_ACCOUNTS)
                           if i not in high_risk_idxs],
                          500 - len(high_risk_idxs))
    for idx in extra:
        acc_by_idx[idx]["pincode"] = random.choice(list(HIGH_RISK_PINCODES))
    high_risk_idxs = [i for i,a in acc_by_idx.items()
                      if a["pincode"] in HIGH_RISK_PINCODES]

for _ in range(900):
    s = random.choice(high_risk_idxs)
    r = diff_idx(s)
    t0, elapsed = rand_ts(), 0
    for _ in range(random.randint(2, 5)):
        elapsed += random.randint(10, 60)
        fraud_txs.append(make_tx(s, r, random.choice(CHANNELS),
                                  random.randint(500, 50_000),
                                  t0 + elapsed, 1))

# ── 5.8  BIDIRECTIONAL FRAUD  (Improvement 5: A→B→A cycles) ─────────────────
for _ in range(400):
    a = random.randint(0, NUM_ACCOUNTS - 1)
    b = diff_idx(a)
    t0    = rand_ts()
    amt   = random.randint(5_000, 40_000)
    hops  = random.randint(2, 4)
    ts    = t0
    direction = True
    for _ in range(hops):
        src, dst = (a, b) if direction else (b, a)
        ts += random.randint(30, 180)
        fraud_txs.append(make_tx(src, dst, random.choice(CHANNELS),
                                  int(amt * random.uniform(0.90, 1.10)),
                                  ts, 1))
        direction = not direction

# ── 5.9  IDENTITY-CLUSTER FRAUD  (Improvement 6: name+pin clusters transact) ─
cluster_pool = [g for g in shared_cluster_groups if len(g) >= 2]
for _ in range(400):
    if not cluster_pool:
        break
    grp = random.choice(cluster_pool)
    a, b = random.sample(grp, 2)
    t0, elapsed = rand_ts(), 0
    for _ in range(random.randint(2, 4)):
        elapsed += random.randint(20, 120)
        fraud_txs.append(make_tx(a, b, random.choice(CHANNELS),
                                  random.randint(1_000, 35_000),
                                  t0 + elapsed, 1))

# ── 5.10  CROSS-CLUSTER NOISE  (Improvement 11: normal → fraud-cluster node) ──
# A normal account sends a single ordinary transaction to a fraud-cluster node.
# label=0 — makes the graph edge-ambiguous, prevents "connected to fraud ⇒ fraud" rule.
if fraud_txs:
    # Build a fast set of fraud sender account_numbers
    fraud_acc_nums = {t["account_number"] for t in fraud_txs}
    # Map account_number -> idx for fast lookup
    acc_num_to_idx = {acc["account_number"]: idx for idx, acc in acc_by_idx.items()}
    fraud_sender_idxs = [acc_num_to_idx[acc] for acc in fraud_acc_nums
                         if acc in acc_num_to_idx]
    if fraud_sender_idxs:
        for _ in range(800):                          # 800 noisy cross-edges
            normal_sender = random.randint(0, NUM_ACCOUNTS - 1)
            fraud_node    = random.choice(fraud_sender_idxs)
            if normal_sender == fraud_node:
                continue
            normal_txs.append(make_tx(
                normal_sender, fraud_node,
                random.choice(CHANNELS),
                random.randint(100, 15_000),          # modest, non-suspicious amount
                rand_ts(), 0))                        # label = 0


print(f"    Fraud transactions  : {len(fraud_txs):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. COMBINE, BALANCE, SHUFFLE
# ══════════════════════════════════════════════════════════════════════════════

print("[4/5] Balancing and shuffling …")
all_txs    = normal_txs + fraud_txs
random.shuffle(all_txs)

total       = len(all_txs)
n_fraud     = sum(1 for t in all_txs if t["label"] == 1)
fraud_pct   = n_fraud / total * 100

print(f"    Before balancing → total={total:,}  fraud={n_fraud:,} ({fraud_pct:.1f}%)")

TARGET_MIN, TARGET_MAX = 0.15, 0.25

if fraud_pct < TARGET_MIN * 100:
    need = int(TARGET_MIN * total) - n_fraud
    extras = []
    for tx in random.choices(fraud_txs, k=need):
        t2 = dict(tx)
        t2["txn_id"]    = f"TXN{_txn_counter + 1:08d}"; _txn_counter += 1
        t2["timestamp"] = int(t2["timestamp"] + random.randint(1, 30))
        t2["amount"]    = int(t2["amount"] * random.uniform(0.97, 1.03))
        extras.append(t2)
    all_txs += extras
    random.shuffle(all_txs)

elif fraud_pct > TARGET_MAX * 100:
    surplus = n_fraud - int(TARGET_MAX * total)
    extras  = []
    for _ in range(surplus):
        s = random.randint(0, NUM_ACCOUNTS - 1)
        r = diff_idx(s)
        extras.append(make_tx(s, r, random.choice(CHANNELS),
                               random.randint(100, 50_000), rand_ts(), 0))
    all_txs += extras
    random.shuffle(all_txs)

total     = len(all_txs)
n_fraud   = sum(1 for t in all_txs if t["label"] == 1)
fraud_pct = n_fraud / total * 100

print(f"    After  balancing → total={total:,}  fraud={n_fraud:,} ({fraud_pct:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. WRITE CSV
# ══════════════════════════════════════════════════════════════════════════════

FIELDS = [
    "txn_id", "account_number", "name", "mobile", "pincode",
    "account_product_type", "narration", "amount", "timestamp", "label",
]

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "transactions_v2.csv")
print(f"[5/5] Writing → {OUTPUT}")

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(all_txs)

print(f"\n✅  Done!")
print(f"    Rows             : {total:,}")
print(f"    Fraud ratio      : {fraud_pct:.2f}%")
print(f"    Unique accounts  : {NUM_ACCOUNTS:,}")
print(f"    Output           : {OUTPUT}")
