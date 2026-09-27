import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
ACCESS_LOGS_PATH = 'access_logs.csv'
OUTPUT_CSV_PATH = 'synthetic_transactions.csv'
OUTPUT_PLOT_PATH = 'transaction_analysis.png'

# Normal Transaction Configuration
NORMAL_TXN_MIN_COUNT = 50
NORMAL_TXN_MAX_COUNT = 200
LOGNORMAL_MEAN = 10000
LOGNORMAL_STD = 25000
AMOUNT_MIN = 500
AMOUNT_MAX = 200000
TXN_TYPES = ['CREDIT', 'DEBIT', 'TRANSFER', 'PAYMENT']
TXN_TYPE_PROBS = [0.25, 0.35, 0.30, 0.10]
BUSINESS_HOURS_START = 9
BUSINESS_HOURS_END = 18
WEEKEND_PROB = 0.10
EXTERNAL_ACCOUNTS_COUNT = 500

# Scenario A: Temporal Link
TEMPORAL_LINK_MIN_DELAY_MINS = 2
TEMPORAL_LINK_MAX_DELAY_MINS = 8
TEMPORAL_LINK_MIN_AMT = 80000
TEMPORAL_LINK_MAX_AMT = 200000

# Scenario B: Structuring
STRUCTURING_ACCOUNTS_COUNT = 15
STRUCTURING_TXN_MIN = 5
STRUCTURING_TXN_MAX = 8
STRUCTURING_WINDOW_HOURS = 2
STRUCTURING_MIN_AMT = 30000
STRUCTURING_MAX_AMT = 49000

# Scenario C: Circular Transfer
CIRCULAR_GROUPS_COUNT = 5
CIRCULAR_WINDOW_HOURS = 4
CIRCULAR_MIN_AMT = 100000
CIRCULAR_MAX_AMT = 500000

# Clean Scenario: Busy Branch
CLEAN_BUSY_ACCOUNTS_COUNT = 10
CLEAN_BUSY_TXN_MIN = 20
CLEAN_BUSY_TXN_MAX = 30

# Validation Thresholds
FRAUD_RATE_MIN = 0.03
FRAUD_RATE_MAX = 0.08

# Set reproducible seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def get_random_business_timestamp(start_dt, end_dt):
    """Generate a random timestamp mostly during business hours on weekdays."""
    while True:
        delta = end_dt - start_dt
        random_second = random.randint(0, int(delta.total_seconds()))
        ts = start_dt + timedelta(seconds=random_second)
        
        is_weekend = ts.weekday() >= 5
        # 10% chance to allow weekend
        if is_weekend and random.random() > WEEKEND_PROB:
            continue
            
        if BUSINESS_HOURS_START <= ts.hour < BUSINESS_HOURS_END:
            return ts

def generate_lognormal_amount(mean, std, min_val, max_val):
    # Lognormal parameters
    sigma = np.sqrt(np.log(1 + (std/mean)**2))
    mu = np.log(mean) - (sigma**2)/2
    
    val = np.random.lognormal(mu, sigma)
    val = max(min_val, min(val, max_val))
    return round(val, 2)

def generate_external_account():
    return f"EXT_{random.randint(10000, 10000 + EXTERNAL_ACCOUNTS_COUNT - 1)}"

# =============================================================================
# SECTION 1 — Load Access Logs
# =============================================================================
print("=" * 50)
print("SECTION 1 — LOAD ACCESS LOGS")
print("=" * 50)

try:
    access_df = pd.read_csv(ACCESS_LOGS_PATH, parse_dates=['timestamp'])
except FileNotFoundError:
    print(f"Error: {ACCESS_LOGS_PATH} not found in the current directory.")
    exit(1)

unique_accounts = access_df['account_id'].unique()
unique_branches = access_df['branch_code'].unique()

# Create a mapping of account to its most frequent branch
account_branch_map = access_df.groupby('account_id')['branch_code'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()

# Dataset date range
min_date = access_df['timestamp'].min()
max_date = access_df['timestamp'].max()

print(f"Loaded {len(access_df)} access log records.")
print(f"Unique accounts found: {len(unique_accounts)}")
print(f"Unique branches found: {len(unique_branches)}")
print(f"Date range: {min_date} to {max_date}")
print("\n")


# =============================================================================
# SECTION 2 — Generate Normal Transactions
# =============================================================================
print("=" * 50)
print("SECTION 2 — GENERATE NORMAL TRANSACTIONS")
print("=" * 50)

normal_txns = []

for acc in unique_accounts:
    num_txns = random.randint(NORMAL_TXN_MIN_COUNT, NORMAL_TXN_MAX_COUNT)
    branch = account_branch_map.get(acc, random.choice(unique_branches))
    
    for _ in range(num_txns):
        amount = generate_lognormal_amount(LOGNORMAL_MEAN, LOGNORMAL_STD, AMOUNT_MIN, AMOUNT_MAX)
        txn_type = random.choices(TXN_TYPES, weights=TXN_TYPE_PROBS)[0]
        ts = get_random_business_timestamp(min_date, max_date)
        cp_id = generate_external_account()
        
        normal_txns.append({
            'account_id': acc,
            'amount': amount,
            'transaction_type': txn_type,
            'counterparty_id': cp_id,
            'timestamp': ts,
            'branch_code': branch,
            'is_fraud': 0,
            'scenario': 'normal'
        })

print(f"Total normal transactions generated: {len(normal_txns)}")
print("\n")

# =============================================================================
# SECTION 3 — Plant Suspicious Scenario A — Temporal Link
# =============================================================================
print("=" * 50)
print("SECTION 3 — PLANTING TEMPORAL LINK SCENARIO")
print("=" * 50)

suspicious_access = access_df[access_df['is_suspicious'] == 1].copy()
temporal_txns = []
temporal_pairs = []

for idx, row in suspicious_access.iterrows():
    delay_mins = random.uniform(TEMPORAL_LINK_MIN_DELAY_MINS, TEMPORAL_LINK_MAX_DELAY_MINS)
    txn_ts = row['timestamp'] + timedelta(minutes=delay_mins)
    
    # Check if transaction falls after our max_date, we cap it or allow it slightly over
    # but we'll just allow it to simulate real-time
    amount = round(random.uniform(TEMPORAL_LINK_MIN_AMT, TEMPORAL_LINK_MAX_AMT), 2)
    
    temporal_txns.append({
        'account_id': row['account_id'],
        'amount': amount,
        'transaction_type': 'TRANSFER',
        'counterparty_id': generate_external_account(),
        'timestamp': txn_ts,
        'branch_code': row['branch_code'],
        'is_fraud': 1,
        'scenario': 'temporal_link'
    })
    
    temporal_pairs.append({
        'access_time': row['timestamp'],
        'txn_time': txn_ts,
        'account': row['account_id'],
        'gap_mins': round(delay_mins, 2)
    })

print(f"Temporal link fraud transactions planted: {len(temporal_txns)}")
if temporal_pairs:
    print("Sample temporal link pairs (Access vs Transaction):")
    sample_pairs = temporal_pairs[:5]
    for p in sample_pairs:
        print(f"Acc: {p['account']} | Access: {p['access_time']} | Txn: {p['txn_time']} | Gap: {p['gap_mins']} mins")
print("\n")


# =============================================================================
# SECTION 4 — Plant Suspicious Scenario B — Structuring
# =============================================================================
print("=" * 50)
print("SECTION 4 — PLANTING STRUCTURING SCENARIO")
print("=" * 50)

structuring_txns = []
structuring_targets = random.sample(list(unique_accounts), min(STRUCTURING_ACCOUNTS_COUNT, len(unique_accounts)))

for acc in structuring_targets:
    branch = account_branch_map.get(acc)
    base_ts = get_random_business_timestamp(min_date, max_date - timedelta(hours=STRUCTURING_WINDOW_HOURS))
    num_txns = random.randint(STRUCTURING_TXN_MIN, STRUCTURING_TXN_MAX)
    
    for _ in range(num_txns):
        offset = timedelta(minutes=random.randint(1, STRUCTURING_WINDOW_HOURS * 60))
        txn_ts = base_ts + offset
        amount = round(random.uniform(STRUCTURING_MIN_AMT, STRUCTURING_MAX_AMT), 2)
        
        structuring_txns.append({
            'account_id': acc,
            'amount': amount,
            'transaction_type': 'TRANSFER',
            'counterparty_id': generate_external_account(),
            'timestamp': txn_ts,
            'branch_code': branch,
            'is_fraud': 1,
            'scenario': 'structuring'
        })

print(f"Structuring fraud transactions planted: {len(structuring_txns)}")
print("\n")


# =============================================================================
# SECTION 5 — Plant Suspicious Scenario C — Circular Transfer
# =============================================================================
print("=" * 50)
print("SECTION 5 — PLANTING CIRCULAR TRANSFER SCENARIO")
print("=" * 50)

circular_txns = []
for _ in range(CIRCULAR_GROUPS_COUNT):
    if len(unique_accounts) < 3:
        break
    group = random.sample(list(unique_accounts), 3)
    acc_A, acc_B, acc_C = group
    
    base_ts = get_random_business_timestamp(min_date, max_date - timedelta(hours=CIRCULAR_WINDOW_HOURS))
    amount = round(random.uniform(CIRCULAR_MIN_AMT, CIRCULAR_MAX_AMT), 2)
    
    # A -> B
    ts_1 = base_ts + timedelta(minutes=random.randint(5, 30))
    circular_txns.append({
        'account_id': acc_A, 'amount': amount, 'transaction_type': 'TRANSFER',
        'counterparty_id': acc_B, 'timestamp': ts_1, 
        'branch_code': account_branch_map.get(acc_A), 'is_fraud': 1, 'scenario': 'circular_transfer'
    })
    
    # B -> C
    ts_2 = ts_1 + timedelta(minutes=random.randint(10, 60))
    circular_txns.append({
        'account_id': acc_B, 'amount': amount, 'transaction_type': 'TRANSFER',
        'counterparty_id': acc_C, 'timestamp': ts_2, 
        'branch_code': account_branch_map.get(acc_B), 'is_fraud': 1, 'scenario': 'circular_transfer'
    })
    
    # C -> A
    ts_3 = ts_2 + timedelta(minutes=random.randint(10, 60))
    circular_txns.append({
        'account_id': acc_C, 'amount': amount, 'transaction_type': 'TRANSFER',
        'counterparty_id': acc_A, 'timestamp': ts_3, 
        'branch_code': account_branch_map.get(acc_C), 'is_fraud': 1, 'scenario': 'circular_transfer'
    })

print(f"Circular transfer fraud transactions planted: {len(circular_txns)}")
print("\n")


# =============================================================================
# SECTION 6 — Plant Clean Scenario — Busy Branch Day
# =============================================================================
print("=" * 50)
print("SECTION 6 — PLANTING CLEAN BUSY SCENARIO")
print("=" * 50)

clean_busy_txns = []
clean_targets = random.sample(list(unique_accounts), min(CLEAN_BUSY_ACCOUNTS_COUNT, len(unique_accounts)))

for acc in clean_targets:
    branch = account_branch_map.get(acc)
    busy_date = get_random_business_timestamp(min_date, max_date).replace(hour=0, minute=0, second=0)
    
    num_txns = random.randint(CLEAN_BUSY_TXN_MIN, CLEAN_BUSY_TXN_MAX)
    for _ in range(num_txns):
        ts = busy_date + timedelta(hours=random.randint(BUSINESS_HOURS_START, BUSINESS_HOURS_END - 1),
                                   minutes=random.randint(0, 59),
                                   seconds=random.randint(0, 59))
        
        amount = generate_lognormal_amount(LOGNORMAL_MEAN, LOGNORMAL_STD, AMOUNT_MIN, AMOUNT_MAX)
        txn_type = random.choices(TXN_TYPES, weights=TXN_TYPE_PROBS)[0]
        
        clean_busy_txns.append({
            'account_id': acc,
            'amount': amount,
            'transaction_type': txn_type,
            'counterparty_id': generate_external_account(),
            'timestamp': ts,
            'branch_code': branch,
            'is_fraud': 0,
            'scenario': 'clean_busy'
        })

print(f"Clean busy transactions generated: {len(clean_busy_txns)}")
print("\n")


# =============================================================================
# SECTION 7 — Combine and Save
# =============================================================================
print("=" * 50)
print("SECTION 7 — COMBINE AND SAVE")
print("=" * 50)

all_txns = normal_txns + temporal_txns + structuring_txns + circular_txns + clean_busy_txns
df_txns = pd.DataFrame(all_txns)

# Sort by timestamp
df_txns = df_txns.sort_values(by='timestamp').reset_index(drop=True)

# Add transaction_id
df_txns.insert(0, 'transaction_id', [f"TXN_{str(i+1).zfill(6)}" for i in range(len(df_txns))])

# Save
df_txns.to_csv(OUTPUT_CSV_PATH, index=False)

total_count = len(df_txns)
fraud_count = df_txns['is_fraud'].sum()
fraud_pct = (fraud_count / total_count) * 100 if total_count > 0 else 0

print(f"Final dataset saved to {OUTPUT_CSV_PATH}")
print(f"Total transactions: {total_count}")
print(f"Fraud transactions: {fraud_count} ({fraud_pct:.2f}%)")
print("Scenario breakdown:")
print(df_txns['scenario'].value_counts())
print(f"Date range: {df_txns['timestamp'].min()} to {df_txns['timestamp'].max()}")
print(f"Unique accounts covered: {df_txns['account_id'].nunique()}")
print("\n")


# =============================================================================
# SECTION 8 — Validation Checks
# =============================================================================
print("=" * 50)
print("SECTION 8 — VALIDATION CHECKS")
print("=" * 50)

# Check 1: Temporal Link matching
temporal_subset = df_txns[df_txns['scenario'] == 'temporal_link']
match_count = 0
for idx, row in temporal_subset.iterrows():
    # Find access logs for this account within 10 minutes prior
    acc = row['account_id']
    ts = row['timestamp']
    prior_10 = ts - timedelta(minutes=10)
    
    matches = access_df[(access_df['account_id'] == acc) & 
                        (access_df['timestamp'] >= prior_10) & 
                        (access_df['timestamp'] <= ts)]
    if not matches.empty:
        match_count += 1

print(f"Check 1: Temporal link transactions with valid prior access event: {match_count}/{len(temporal_subset)}")
if match_count != len(temporal_subset):
    print("WARNING: Some temporal link transactions lack a matching access log within 10 minutes!")

# Check 2: All accounts exist in access_logs
txn_accounts = set(df_txns['account_id'].unique())
access_accounts = set(access_df['account_id'].unique())
unmatched = txn_accounts - access_accounts
print(f"Check 2: Accounts matched: {len(txn_accounts) - len(unmatched)}, Unmatched: {len(unmatched)}")
if len(unmatched) > 0:
    print(f"WARNING: Found {len(unmatched)} accounts in transactions that do not exist in access_logs!")

# Check 3: Timestamps do not precede access logs min date
early_txns = df_txns[df_txns['timestamp'] < min_date]
print(f"Check 3: Transactions preceding earliest access log: {len(early_txns)}")
if len(early_txns) > 0:
    print(f"WARNING: {len(early_txns)} transactions occurred before the earliest access log!")

# Check 4: Fraud Rate
print(f"Check 4: Target Fraud Rate [{FRAUD_RATE_MIN*100:.1f}% - {FRAUD_RATE_MAX*100:.1f}%], Actual: {fraud_pct:.2f}%")
if not (FRAUD_RATE_MIN*100 <= fraud_pct <= FRAUD_RATE_MAX*100):
    print("WARNING: Fraud rate is outside the realistic threshold!")
print("\n")


# =============================================================================
# SECTION 9 — Quick Visualisation
# =============================================================================
print("=" * 50)
print("SECTION 9 — QUICK VISUALISATION")
print("=" * 50)

plt.figure(figsize=(20, 10))

# 1. Transaction Amount Distribution
plt.subplot(2, 2, 1)
normal_amts = df_txns[df_txns['is_fraud'] == 0]['amount']
fraud_amts = df_txns[df_txns['is_fraud'] == 1]['amount']
plt.hist(normal_amts, bins=50, alpha=0.5, label='Normal', density=True)
plt.hist(fraud_amts, bins=50, alpha=0.5, label='Fraud', density=True, color='red')
plt.title('Transaction Amount Distribution')
plt.legend()

# 2. Transaction Count Per Day
plt.subplot(2, 2, 2)
daily_counts = df_txns.groupby(df_txns['timestamp'].dt.date).size()
daily_counts.plot(kind='line', marker='o')
plt.title('Transaction Count Per Day')
plt.grid(True, alpha=0.3)

# 3. Fraud Scenario Breakdown
plt.subplot(2, 2, 3)
scenario_counts = df_txns[df_txns['is_fraud'] == 1]['scenario'].value_counts()
scenario_counts.plot(kind='bar', color='coral')
plt.title('Fraud Transactions by Scenario')
plt.xticks(rotation=45)

# 4. Hour of Day Distribution
plt.subplot(2, 2, 4)
hourly_counts = df_txns['timestamp'].dt.hour.value_counts().sort_index()
hourly_counts.plot(kind='bar', color='skyblue')
plt.title('Transaction Distribution by Hour of Day')

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH)
print(f"Visualizations saved to {OUTPUT_PLOT_PATH}")
print("=" * 50)
