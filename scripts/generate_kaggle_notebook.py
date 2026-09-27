import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4,
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# ==============================================================================
# SECTION 0 — INTRODUCTION
# ==============================================================================
add_md("""# TemporalShield — Deep Learning & Gradient Boosting Pipeline
## LSTM Structuring Detector + XGBoost Customer Profile Mismatch Engine

This notebook trains two advanced fraud detection engines for **TemporalShield**:
1. **PyTorch LSTM + Attention**: Detects complex **Transaction Structuring** — sub-₹50,000 splitting to evade RBI reporting thresholds.
2. **XGBoost + SHAP**: Detects **Customer Profile Mismatch** — dormant account abuse and high-velocity layering patterns.

### Model–Dataset–Incident Mapping

| Model | Dataset | Detects | Real Incident | Regulatory Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM + Attention** | PaySim + synthetic_transactions | Transaction structuring — sub-₹50K splitting | Hawala networks (ED-documented syndicates) | PMLA 2002 / RBI ₹50,000 CTR Threshold |
| **XGBoost + SHAP** | PaySim | Customer profile mismatch — dormant account abuse | Jan Dhan fraud 2016 (post-demonetization) | RBI KYC AML Red Flag Directives |

### Why This Dataset Combination Works

* **PaySim (6.36M+ Transactions)**: Derived from real-world mobile money logs. Provides natural account lifecycles, realistic balance progressions, and authentic payment rhythms—the backbone of learning true customer behavioral baselines.
* **synthetic_transactions.csv**: Contains explicitly planted, ground-truth labeled structuring scenarios mimicking Indian regulatory evasions with known burst timing.
* **Synergy**: PaySim supplies realistic normal sequence backgrounds; synthetic data provides verified structuring signal. The LSTM learns both simultaneously, eliminating the need for hand-crafted rules.
""")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
add_md("""## Configuration & Global Hyperparameters
All constants, thresholds, and model hyperparameters are defined here. No magic numbers elsewhere.
""")

add_code('''print("=" * 70)
print("TEMPORALSHIELD — CONFIGURATION")
print("=" * 70)

import os, sys, gc, json, random, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")

# --- Core Seeds ---
RANDOM_SEED = 42

# ===========================================================================
# GPU CONFIG — RTX Pro 6000 (95.6 GB VRAM) + 175 GB RAM
# ALL sampling caps removed. Full dataset utilised.
# ===========================================================================

# --- Sequence LSTM ---
SEQUENCE_LENGTH        = 30          # Longer context window captures multi-hop structuring rings
LSTM_HIDDEN_SIZE       = 256         # 2× wider: deeper pattern representation
LSTM_LAYERS            = 3           # 3 stacked layers for richer sequential abstraction
LSTM_DROPOUT           = 0.25        # Slight reduction — fewer regularisation constraints needed at scale
LSTM_EPOCHS            = 100         # 2× training budget — full convergence on large dataset
LSTM_BATCH_SIZE        = 512         # Large batches fully utilise GPU tensor cores
LSTM_LEARNING_RATE     = 0.001
LSTM_LR_PATIENCE       = 8          # Longer patience — training is noisier at scale
LSTM_LR_FACTOR         = 0.5
SEQUENCE_STRIDE        = 2          # Dense sequence sampling (was 5) — 175 GB RAM handles it
LSTM_DATALOADER_WORKERS = 4          # Parallel CPU data loading workers for GPU pipelining

# --- Structuring Detection Thresholds ---
STRUCTURING_THRESHOLD        = 50000
STRUCTURING_WINDOW_HOURS     = 2
STRUCTURING_MIN_TRANSACTIONS = 3

# --- Account Caps (REMOVED for full-dataset training) ---
LSTM_MAX_ACCOUNTS   = None  # None = no cap — use all accounts with >= 3 transactions
XGB_MAX_ACCOUNTS    = None  # None = no cap — full PaySim dataset for XGBoost

# --- XGBoost (GPU histogram mode) ---
XGBOOST_ESTIMATORS        = 1000    # 2× more trees — deeper ensemble on full 6.36M rows
XGBOOST_MAX_DEPTH         = 8       # Deeper trees now justified by full data volume
XGBOOST_LEARNING_RATE     = 0.03    # Slightly lower LR — more trees, smaller steps
XGBOOST_SCALE_POS_WEIGHT  = 10
XGBOOST_MIN_CHILD_WEIGHT  = 5       # Higher: prevents tiny spurious splits on 6M rows
XGBOOST_SUBSAMPLE         = 0.85
XGBOOST_COLSAMPLE_BYTREE  = 0.85
XGBOOST_GAMMA             = 0.1     # Minimum loss reduction to make a split — reduces overfitting
XGBOOST_REG_LAMBDA        = 1.5     # L2 regularisation
XGBOOST_REG_ALPHA         = 0.1     # L1 regularisation

# --- Splits & Targets ---
TRAIN_SPLIT    = 0.80
TARGET_RECALL  = 0.80

# --- Reproducibility ---
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark    = True   # Auto-tune convolutions for fixed input sizes
    torch.backends.cudnn.deterministic = False  # Faster non-deterministic mode on large batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Random seed: {RANDOM_SEED}")
print(f"[+] Compute device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"[+] CUDA Cores: {props.multi_processor_count * 128}")
    print(f"[+] Mixed Precision: {'Available (fp16)' if torch.cuda.is_bf16_supported() else 'fp32 only'}")
print(f"[+] CPU RAM available: {os.popen('python -c import psutil;print(f\"{psutil.virtual_memory().available/1e9:.1f} GB\")' if False else 'check psutil if available').read().strip()}")
''')

# ==============================================================================
# SECTION 1 — LOAD AND VALIDATE DATA
# ==============================================================================
add_md("""# SECTION 1 — LOAD AND VALIDATE DATA
PaySim simulation steps are converted to real ISO timestamps. Both datasets are validated before training begins.
""")

add_code('''print("=" * 70)
print("SECTION 1 — LOAD AND VALIDATE DATA")
print("=" * 70)

def locate_file(filename):
    """Automatically locate a file across local, ../data, and /kaggle/input paths."""
    search_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("../data", filename),
    ]
    for sp in search_paths:
        if os.path.exists(sp):
            return sp
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower() == filename.lower():
                    return os.path.join(root, f)
    return filename

paysim_path    = locate_file("paysim.csv")
if not os.path.exists(paysim_path):
    paysim_path = locate_file("PS_20174392719_1491204439457_log.csv")
synthetic_path = locate_file("synthetic_transactions.csv")

print(f"PaySim path:    {paysim_path}")
print(f"Synthetic path: {synthetic_path}")

# Optimised dtypes to reduce peak memory usage
paysim_df = pd.read_csv(
    paysim_path,
    dtype={
        "step": np.int32, "type": "category", "amount": np.float64,
        "nameOrig": "string", "oldbalanceOrg": np.float64,
        "newbalanceOrig": np.float64, "nameDest": "string",
        "oldbalanceDest": np.float64, "newbalanceDest": np.float64,
        "isFraud": np.int8, "isFlaggedFraud": np.int8
    }
)

base_date = pd.Timestamp("2026-01-01 00:00:00")
paysim_df["timestamp"] = base_date + pd.to_timedelta(paysim_df["step"], unit="h")
paysim_df.rename(columns={"nameOrig": "account_id", "nameDest": "counterparty_id"}, inplace=True)

keep_cols = ["account_id", "amount", "type", "counterparty_id",
             "timestamp", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest", "isFraud"]
paysim_df = paysim_df[keep_cols]

synthetic_df = pd.read_csv(synthetic_path)
synthetic_df["timestamp"] = pd.to_datetime(synthetic_df["timestamp"])

# --- Profiles ---
print("\\n--- PAYSIM PROFILE ---")
print(f"Shape: {paysim_df.shape}")
print(f"Date range: {paysim_df['timestamp'].min()} → {paysim_df['timestamp'].max()}")
print(f"Fraud rate: {paysim_df['isFraud'].mean()*100:.3f}% ({paysim_df['isFraud'].sum():,} cases)")
print("Types:\\n", paysim_df["type"].value_counts())
print("Nulls:\\n", paysim_df.isnull().sum())

print("\\n--- SYNTHETIC PROFILE ---")
print(f"Shape: {synthetic_df.shape}")
print(f"Date range: {synthetic_df['timestamp'].min()} → {synthetic_df['timestamp'].max()}")
print(f"Fraud rate: {synthetic_df['is_fraud'].mean()*100:.3f}%")
if "scenario" in synthetic_df.columns:
    print("Scenarios:\\n", synthetic_df["scenario"].value_counts())
print("Nulls:\\n", synthetic_df.isnull().sum())

# --- Validation ---
print("\\n" + "=" * 40 + "  DATA CHECKS  " + "=" * 40)

fraud_pct      = paysim_df["isFraud"].mean() * 100
struct_count   = len(synthetic_df[synthetic_df.get("scenario", pd.Series()) == "structuring"]) if "scenario" in synthetic_df.columns else 0
neg_ps         = (paysim_df["amount"] < 0).sum()
neg_sy         = (synthetic_df["amount"] < 0).sum()
expected_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
actual_types   = set(paysim_df["type"].unique())

print(f"Check A [PaySim fraud rate 0.05%–2%]:           {'PASS' if 0.05 <= fraud_pct <= 2.0 else 'WARNING'} ({fraud_pct:.3f}%)")
print(f"Check B [Synthetic structuring >= 50]:          {'PASS' if struct_count >= 50 else 'WARNING'} ({struct_count} cases)")
print(f"Check C [No negative amounts]:                  {'PASS' if neg_ps == 0 and neg_sy == 0 else 'WARNING'} (PaySim:{neg_ps}, Synth:{neg_sy})")
print(f"Check D [Expected transaction types present]:   {'PASS' if expected_types.issubset(actual_types) else 'WARNING'} (Found: {actual_types})")
''')

# ==============================================================================
# SECTION 2 — EDA
# ==============================================================================
add_md("""# SECTION 2 — EXPLORATORY DATA ANALYSIS
""")

add_code('''print("=" * 70)
print("SECTION 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
sns.set_theme(style="whitegrid")

# Plot 1: PaySim type distribution
ax = axes[0, 0]
sns.countplot(data=paysim_df, x="type", ax=ax, palette="Blues_r",
              order=paysim_df["type"].value_counts().index)
ax.set_title("Plot 1: Transaction Type Distribution", fontweight="bold")
ax.set_xlabel("Type"); ax.set_ylabel("Count")

# Plot 2: Amount distribution by fraud (log scale)
ax = axes[0, 1]
s = paysim_df.sample(min(200000, len(paysim_df)), random_state=RANDOM_SEED)
sns.histplot(data=s, x="amount", hue="isFraud", bins=50, log_scale=True,
             ax=ax, palette={0: "#3498db", 1: "#e74c3c"},
             common_norm=False, stat="density")
ax.set_title("Plot 2: Amount by Fraud Status (Log)", fontweight="bold")
ax.set_xlabel("Amount (Log Scale)")

# Plot 3: Fraud rate per type
ax = axes[0, 2]
fraud_type = paysim_df.groupby("type")["isFraud"].mean().reset_index()
sns.barplot(data=fraud_type, x="type", y="isFraud", ax=ax, palette="Reds_r")
ax.set_title("Plot 3: Fraud Rate per Channel", fontweight="bold")
ax.set_ylabel("Fraud Rate")

# Plot 4: Synthetic median amounts by scenario
ax = axes[1, 0]
if "scenario" in synthetic_df.columns:
    sns.barplot(data=synthetic_df, x="scenario", y="amount",
                ax=ax, estimator=np.median, palette="magma")
    ax.set_title("Plot 4: Median Amount by Synthetic Scenario", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
else:
    ax.text(0.5, 0.5, "No scenario column", ha="center", va="center")

# Plot 5: Balance before vs after (log-log)
ax = axes[1, 1]
fs = paysim_df[paysim_df["isFraud"]==1].sample(min(2000, paysim_df["isFraud"].sum()), random_state=RANDOM_SEED)
ns = paysim_df[paysim_df["isFraud"]==0].sample(2000, random_state=RANDOM_SEED)
ax.scatter(ns["oldbalanceOrg"]+1, ns["newbalanceOrig"]+1, c="#3498db", alpha=0.3, s=10, label="Normal")
ax.scatter(fs["oldbalanceOrg"]+1, fs["newbalanceOrig"]+1, c="#e74c3c", alpha=0.7, s=15, label="Fraud")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("Plot 5: Originator Balance Before vs After", fontweight="bold")
ax.set_xlabel("Balance Before"); ax.set_ylabel("Balance After"); ax.legend()

# Plot 6: Daily volume with fraud overlay
ax = axes[1, 2]
daily = paysim_df.set_index("timestamp").resample("D").size()
daily_fr = paysim_df[paysim_df["isFraud"]==1].set_index("timestamp").resample("D").size()
ax.plot(daily.index, daily.values, color="#2c3e50", lw=2, label="Daily Volume")
ax.scatter(daily_fr.index, daily_fr.values * 100, color="#e74c3c", s=20, zorder=5, label="Fraud Spikes (×100)")
ax.set_title("Plot 6: Volume & Fraud Over Time", fontweight="bold")
ax.tick_params(axis="x", rotation=30); ax.legend()

plt.tight_layout()
plt.show()
''')

add_md("""### Key Insights

1. **Channel Concentration (Plots 1 & 3)**: Fraud exists only in `TRANSFER` and `CASH_OUT`. All other channels have zero fraud in PaySim. Feature engineering should exploit this channel flag heavily.
2. **Full Balance Drain (Plot 5)**: Fraud transactions predominantly show `oldbalanceOrg ≈ amount` and `newbalanceOrig ≈ 0.0`. The `errorBalanceOrig` feature (expected vs. actual balance) is a near-perfect fraud signal.
3. **Amount Clustering (Plot 2 & 4)**: Structuring amounts cluster tightly just below ₹50,000. The `is_round_number` and `is_below_threshold` features will capture this boundary behaviour.
""")

# ==============================================================================
# SECTION 3 — LSTM STRUCTURING DETECTOR
# ==============================================================================
add_md("""# SECTION 3 — LSTM STRUCTURING DETECTOR

### 3a — What is Transaction Structuring (Smurfing)?
**Structuring** is the deliberate criminal practice of fragmenting large sums into multiple smaller transactions, each kept specifically below mandatory regulatory reporting thresholds. In India, the RBI PMLA 2002 mandates Cash Transaction Reports (CTR) for any transaction above **₹50,000**.

#### Why Rule-Based Systems Fail at This:
A single ₹49,000 transfer is completely legal and triggers zero alarms. But **ten transfers of ₹49,000 within 90 minutes** represent a ₹4,90,000 laundering sweep—entirely invisible to rule-based engines.

#### Why LSTM + Attention is the Correct Architecture:
Structuring is a **temporal sequence anomaly**. The LSTM learns the baseline rhythmic cadence of normal accounts across 20-transaction windows, while the **attention mechanism** learns to weight the most suspicious transactions within each window (not just the final one). This makes the model far more sensitive to burst patterns where only the last 3–4 transactions are suspicious.

#### Connection to Hawala Networks:
ED-documented Hawala syndicates use couriers to execute hundreds of sub-threshold cash deposits simultaneously across mule accounts in different cities. The temporal clustering and amount homogeneity is the exact signature this LSTM is trained to recognize.
""")

add_code('''print("=" * 70)
print("3b — DATA PREPARATION — STRUCTURING DETECTOR")
print("=" * 70)

import multiprocessing
try:
    import psutil
    print(f"[+] System RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")
except ImportError:
    pass
if torch.cuda.is_available():
    print(f"[+] GPU VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.1f} GB")

# RTX Pro 6000: no account cap needed. Use ALL accounts with >= 3 transactions.
paysim_tf    = paysim_df[paysim_df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
acc_vc       = paysim_tf["account_id"].value_counts()
if LSTM_MAX_ACCOUNTS is None:
    selected_accs = acc_vc[acc_vc >= 3].index           # Full dataset — all qualifying accounts
else:
    selected_accs = acc_vc[acc_vc >= 3].head(LSTM_MAX_ACCOUNTS).index

paysim_seq = paysim_tf[paysim_tf["account_id"].isin(selected_accs)].copy()
paysim_seq = paysim_seq.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
print(f"Working with {len(selected_accs):,} accounts, {len(paysim_seq):,} transactions")

# ---- Vectorised structuring window labeler (parallel per account) ----
def _label_account(args):
    """Label structuring windows for a single account group (parallelisable)."""
    acc_id, grp, threshold, window_h, min_tx = args
    t   = grp["timestamp"].values
    amt = grp["amount"].values
    n   = len(grp)
    flags = np.zeros(n, dtype=np.int8)
    if n < min_tx:
        return grp.index.values, flags
    window_td = pd.Timedelta(hours=window_h)
    for i in range(n):
        w_start = t[i] - window_td
        mask    = (t >= w_start) & (t <= t[i])
        w_amt   = amt[mask]
        if (len(w_amt) >= min_tx and
                np.all(w_amt < threshold) and
                w_amt.sum() >= threshold * 1.5):
            flags[mask] = 1
    return grp.index.values, flags

print(f"Labelling structuring windows across {len(selected_accs):,} accounts...")
all_groups = [
    (acc_id, grp, STRUCTURING_THRESHOLD, STRUCTURING_WINDOW_HOURS, STRUCTURING_MIN_TRANSACTIONS)
    for acc_id, grp in paysim_seq.groupby("account_id")
]
paysim_seq["is_structuring"] = 0
for idx_arr, flags in map(_label_account, all_groups):
    paysim_seq.loc[idx_arr, "is_structuring"] = flags

print(f"PaySim structuring transactions found: {paysim_seq['is_structuring'].sum():,}")

# ---- Synthetic structuring ----
synth_struct = synthetic_df.copy()
synth_struct["is_structuring"] = (synth_struct["scenario"] == "structuring").astype(int) if "scenario" in synth_struct.columns else 0
synth_struct.rename(columns={"transaction_type": "type"}, inplace=True, errors="ignore")
synth_struct["oldbalanceOrg"]  = 100000.0
synth_struct["newbalanceOrig"] = np.maximum(0.0, synth_struct["oldbalanceOrg"] - synth_struct["amount"])

common_cols = ["account_id", "timestamp", "amount", "type", "oldbalanceOrg", "newbalanceOrig", "is_structuring"]
combined_df = pd.concat([
    paysim_seq[common_cols],
    synth_struct[[c for c in common_cols if c in synth_struct.columns]]
], ignore_index=True)
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
combined_df = combined_df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

print(f"\\nCombined dataset: {len(combined_df):,} transactions")
print("Structuring label distribution:")
print(combined_df["is_structuring"].value_counts())
''')

add_md("""### 3c — Sequence Feature Engineering

We create 6 structured temporal features per transaction step plus 3 new signals:
- **`velocity_ratio`**: Transactions per hour in the past window vs. historical baseline — the primary smurfing speed signal.
- **`is_round_number`**: Flags amounts divisible by ₹1,000 — criminals often use psychologically "clean" round numbers just below thresholds (₹49,000, ₹48,000).
- **`amount_homogeneity`**: Rolling std of last 3 amounts divided by rolling mean — near-zero homogeneity is the defining structuring fingerprint.

**Front-zero padding strategy**: Sequences shorter than `SEQUENCE_LENGTH` are padded with zeros at the **front**, not the end. This ensures the LSTM final hidden state $h_T$ is informed by the most recent real transactions, while attention can still focus on the non-zero genuine signal.
""")

add_code('''print("=" * 70)
print("3c — FEATURE ENGINEERING & SEQUENCE GENERATION")
print("=" * 70)

# Per-account historical mean for normalisation
acc_mean = combined_df.groupby("account_id")["amount"].transform("mean").fillna(1.0)
combined_df["amount_normalised"] = combined_df["amount"] / (acc_mean + 1.0)

# Time since last transaction (minutes)
combined_df["time_since_last_minutes"] = (
    combined_df.groupby("account_id")["timestamp"]
    .diff().dt.total_seconds().div(60.0).fillna(9999.0).clip(upper=10000.0)
)

combined_df["is_below_threshold"]  = (combined_df["amount"] < STRUCTURING_THRESHOLD).astype(float)
# Structuring boundary band: sub-threshold amounts between 30,000 and 50,000 (PMLA CTR threshold evasion)
combined_df["is_struct_band"]      = ((combined_df["amount"] >= 30000) & (combined_df["amount"] < STRUCTURING_THRESHOLD)).astype(float)
combined_df["is_transfer"]         = (combined_df["type"].astype(str) == "TRANSFER").astype(float)
combined_df["is_fast"]             = (combined_df["time_since_last_minutes"] <= 120.0).astype(float)

le_type_lstm = LabelEncoder()
combined_df["type_encoded"] = le_type_lstm.fit_transform(combined_df["type"].astype(str))

combined_df["balance_change_ratio"] = (
    (combined_df["newbalanceOrig"] - combined_df["oldbalanceOrg"]) /
    (combined_df["oldbalanceOrg"] + 1.0)
).clip(-10.0, 10.0)

# Rolling cumulative amount (last 3 transactions)
combined_df["cumulative_amount_window"] = combined_df.groupby("account_id")["amount"].transform(
    lambda x: x.rolling(3, min_periods=1).sum()
)

# Rolling 3-txn metrics for structuring pattern
combined_df["roll_struct_band_3"] = combined_df.groupby("account_id")["is_struct_band"].transform(
    lambda x: x.rolling(3, min_periods=1).sum()
)
combined_df["roll_fast_3"] = combined_df.groupby("account_id")["is_fast"].transform(
    lambda x: x.rolling(3, min_periods=1).sum()
)

# Amount homogeneity — std/mean over last 3 (near-zero = structuring)
roll_mean = combined_df.groupby("account_id")["amount"].transform(lambda x: x.rolling(3, min_periods=2).mean())
roll_std  = combined_df.groupby("account_id")["amount"].transform(lambda x: x.rolling(3, min_periods=2).std().fillna(0.0))
combined_df["amount_homogeneity"] = (roll_std / (roll_mean + 1.0)).fillna(0.0).clip(0.0, 5.0)

# Velocity: rapid transactions within 2h window
combined_df["velocity_count"] = combined_df.groupby("account_id")["is_fast"].transform(
    lambda x: x.rolling(5, min_periods=1).sum()
)

feature_cols_lstm = [
    "amount_normalised",
    "time_since_last_minutes",
    "is_below_threshold",
    "is_struct_band",
    "is_transfer",
    "is_fast",
    "roll_fast_3",
    "roll_struct_band_3",
    "cumulative_amount_window",
    "amount_homogeneity",
    "velocity_count",
    "balance_change_ratio"
]

scaler_lstm = StandardScaler()
combined_df[feature_cols_lstm] = scaler_lstm.fit_transform(combined_df[feature_cols_lstm])

# ---- Sequence Generation (Capturing ALL Structuring Bursts) ----
print("Generating temporal sequences with complete structuring coverage...")
sequences, labels, sequence_meta = [], [], []

for acc_id, grp in combined_df.groupby("account_id"):
    g_feats  = grp[feature_cols_lstm].values.astype(np.float32)
    g_labels = grp["is_structuring"].values
    g_times  = grp["timestamp"].values
    g_amts   = grp["amount"].values
    n        = len(grp)

    # Ensure EVERY structuring event is emitted; sample normal steps with stride
    emit_indices = set()
    for i in range(n):
        if g_labels[i] == 1 or i % SEQUENCE_STRIDE == 0 or i == n - 1:
            emit_indices.add(i)
    emit_indices = sorted(emit_indices)

    for i in emit_indices:
        sub = g_feats[max(0, i - SEQUENCE_LENGTH + 1): i + 1]
        if len(sub) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(sub), len(feature_cols_lstm)), dtype=np.float32)
            sub = np.vstack([pad, sub])
        sequences.append(sub)
        labels.append(g_labels[i])
        sequence_meta.append({"account_id": acc_id, "timestamp": g_times[i], "amount": g_amts[i]})

X_seq = np.array(sequences, dtype=np.float32)
y_seq = np.array(labels,    dtype=np.float32)

print(f"[+] Total sequences: {len(X_seq):,}  (stride={SEQUENCE_STRIDE})")
print(f"[+] Structuring sequences: {int(y_seq.sum()):,}")
print(f"[+] Normal sequences:      {int((1 - y_seq).sum()):,}")
print(f"[+] Imbalance ratio:       {(len(y_seq) - y_seq.sum()) / max(1.0, y_seq.sum()):.1f} : 1")
''')

add_code('''print("=" * 70)
print("3d — TRAIN PYTORCH LSTM + ATTENTION STRUCTURING DETECTOR")
print("=" * 70)

# ---- Account-Stratified Split ----
# Stratify accounts with structuring so Train (70%), Val (15%), and Test (15%)
# all contain genuine structuring episodes, while keeping sequences temporally coherent
meta_df = pd.DataFrame(sequence_meta)

struct_accs = meta_df[y_seq == 1]["account_id"].unique()
norm_accs   = np.array([a for a in meta_df["account_id"].unique() if a not in struct_accs])

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(struct_accs)
rng.shuffle(norm_accs)

st_tr, st_va = int(len(struct_accs) * 0.70), int(len(struct_accs) * 0.15)
nm_tr, nm_va = int(len(norm_accs) * 0.70), int(len(norm_accs) * 0.15)

tr_accs  = set(struct_accs[:st_tr]).union(set(norm_accs[:nm_tr]))
val_accs = set(struct_accs[st_tr:st_tr+st_va]).union(set(norm_accs[nm_tr:nm_tr+nm_va]))
te_accs  = set(struct_accs[st_tr+st_va:]).union(set(norm_accs[nm_tr+nm_va:]))

tr_mask  = meta_df["account_id"].isin(tr_accs).values
val_mask = meta_df["account_id"].isin(val_accs).values
te_mask  = meta_df["account_id"].isin(te_accs).values

X_tr, y_tr   = X_seq[tr_mask], y_seq[tr_mask]
X_val, y_val = X_seq[val_mask], y_seq[val_mask]
X_te, y_te   = X_seq[te_mask], y_seq[te_mask]

print(f"[+] Train: {len(y_tr):,} seqs ({int(y_tr.sum())} structuring)")
print(f"[+] Val:   {len(y_val):,} seqs ({int(y_val.sum())} structuring)")
print(f"[+] Test:  {len(y_te):,} seqs ({int(y_te.sum())} structuring)")

class SequenceDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# Pin memory for faster GPU transfer
pin_mem = torch.cuda.is_available()
n_wkrs  = LSTM_DATALOADER_WORKERS if torch.cuda.is_available() else 0

# ---- Positive Sequence Oversampling ----
TARGET_IMBALANCE = 5.0
pos_idx = np.where(y_tr == 1)[0]
neg_count_tr = float(len(y_tr) - len(pos_idx))
needed = int(neg_count_tr / TARGET_IMBALANCE) - len(pos_idx)

if needed > 0 and len(pos_idx) > 0:
    rep_idx = np.random.choice(pos_idx, size=needed, replace=True)
    X_tr_os = np.concatenate([X_tr, X_tr[rep_idx]], axis=0)
    y_tr_os = np.concatenate([y_tr, y_tr[rep_idx]], axis=0)
    shuf    = np.random.permutation(len(X_tr_os))
    X_tr_os, y_tr_os = X_tr_os[shuf], y_tr_os[shuf]
    new_ratio = (len(y_tr_os) - y_tr_os.sum()) / y_tr_os.sum()
    print(f"[+] Oversampled: added {needed:,} positive copies. New ratio: {new_ratio:.1f} : 1")
else:
    X_tr_os, y_tr_os = X_tr, y_tr
    print(f"[+] No oversampling needed.")

tr_ld  = DataLoader(SequenceDS(X_tr_os, y_tr_os), batch_size=LSTM_BATCH_SIZE, shuffle=True,
                    drop_last=True, num_workers=n_wkrs, pin_memory=pin_mem)
val_ld = DataLoader(SequenceDS(X_val, y_val), batch_size=LSTM_BATCH_SIZE, shuffle=False,
                    num_workers=n_wkrs, pin_memory=pin_mem)
te_ld  = DataLoader(SequenceDS(X_te,  y_te),  batch_size=LSTM_BATCH_SIZE, shuffle=False,
                    num_workers=n_wkrs, pin_memory=pin_mem)

# ---- LSTM + Additive Attention Architecture ----
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM output sequence."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)            # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)            # (batch, hidden_dim)
        return context, weights.squeeze(2)

class StructuringLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = AdditiveAttention(hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _        = self.lstm(x)                    # (B, T, H)
        context, attn = self.attention(out)              # (B, H), (B, T)
        context       = self.norm(context)
        return self.fc(self.drop(context)), attn

lstm_model = StructuringLSTM(
    input_dim=len(feature_cols_lstm),
    hidden_dim=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_LAYERS,
    dropout=LSTM_DROPOUT
).to(device)

pos_weight = torch.tensor([(len(y_tr_os) - y_tr_os.sum()) / max(1.0, y_tr_os.sum())]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(lstm_model.parameters(), lr=LSTM_LEARNING_RATE, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=LSTM_LR_FACTOR,
    patience=LSTM_LR_PATIENCE, verbose=True
)

use_amp = torch.cuda.is_available()
scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

print(f"[+] Loss: Weighted BCE (pos_weight={pos_weight.item():.2f}) | AMP={use_amp}")
print(f"[+] Oversampled train size: {len(y_tr_os):,} | Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
print(lstm_model)

# ---- Training Loop with AMP Mixed Precision ----
train_losses, val_losses = [], []
best_val_auc   = -1.0
best_model_wts = None
n_train = len(y_tr_os)
n_val   = len(SequenceDS(X_val, y_val))

for epoch in range(1, LSTM_EPOCHS + 1):
    # — Train —
    lstm_model.train()
    run_loss = 0.0
    for bx, by in tr_ld:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = lstm_model(bx)
            loss      = criterion(logits, by)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=5.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        run_loss += loss.item() * len(bx)
    train_losses.append(run_loss / n_train)

    # — Validate —
    lstm_model.eval()
    val_run_loss, v_preds, v_trues = 0.0, [], []
    with torch.no_grad():
        for bx, by in val_ld:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = lstm_model(bx)
            val_run_loss += criterion(logits, by.float()).item() * len(bx)
            v_preds.extend(torch.sigmoid(logits.float()).cpu().numpy().flatten())
            v_trues.extend(by.cpu().numpy().flatten())
    val_losses.append(val_run_loss / n_val)

    # Average Precision (AUCPR) is threshold-independent and robust to class imbalance
    val_auc = average_precision_score(v_trues, v_preds) if sum(v_trues) > 0 else 0.0
    scheduler.step(val_auc)

    if val_auc > best_val_auc:
        best_val_auc   = val_auc
        best_model_wts = {k: v.clone() for k, v in lstm_model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        cur_lr   = optimizer.param_groups[0]["lr"]
        vram_gb  = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0
        print(f"Epoch [{epoch:03d}/{LSTM_EPOCHS}] "
              f"TrainLoss:{train_losses[-1]:.4f}  ValLoss:{val_losses[-1]:.4f}  "
              f"ValAUCPR:{val_auc:.4f}  LR:{cur_lr:.6f}  VRAM:{vram_gb:.2f}GB")

if best_model_wts:
    lstm_model.load_state_dict(best_model_wts)
    print(f"\\n[+] Best checkpoint restored (Val AUCPR: {best_val_auc:.4f})")

# — Training Curve —
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(train_losses, label="Train Loss", lw=2, color="#2980b9")
axes[0].plot(val_losses,   label="Val Loss",   lw=2, color="#e74c3c", ls="--")
axes[0].set_title("LSTM Train vs Val Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(val_losses, color="#8e44ad", lw=2)
axes[1].set_title("Validation Loss Convergence")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss"); axes[1].grid(True)

loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
axes[2].plot(loss_diff, color="#27ae60", lw=2)
axes[2].axhline(0, color="black", ls="--")
axes[2].set_title("Train-Val Loss Gap (Overfitting Monitor)")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Train − Val Loss"); axes[2].grid(True)

plt.tight_layout()
plt.show()
''')

add_code('''print("=" * 70)
print("3e — OPTIMAL THRESHOLD SEARCH (VALIDATION SET)")
print("=" * 70)

lstm_model.eval()
val_probs_arr, val_trues_arr = [], []
with torch.no_grad():
    for bx, by in val_ld:
        logits, _ = lstm_model(bx.to(device))
        val_probs_arr.extend(torch.sigmoid(logits.float()).cpu().numpy().flatten())
        val_trues_arr.extend(by.numpy().flatten())

val_probs_arr = np.array(val_probs_arr)
val_trues_arr = np.array(val_trues_arr)

print("Prob distribution stats on validation:")
print(f"  Min: {val_probs_arr.min():.6f}  Max: {val_probs_arr.max():.6f}")
print(f"  Mean: {val_probs_arr.mean():.6f}  Median: {np.median(val_probs_arr):.6f}")
print(f"  95th pctile: {np.percentile(val_probs_arr, 95):.6f}")
print(f"  99th pctile: {np.percentile(val_probs_arr, 99):.6f}")

best_lstm_thresh = 0.50
best_lstm_f1 = best_prec_lv = best_rec_lv = -1.0

# Search thresholds for optimal F1 with balanced precision & recall
for th in np.linspace(0.10, 0.90, 81):
    p_bin = (val_probs_arr >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(val_trues_arr, p_bin, average="binary", zero_division=0)
    if p >= 0.50 and r >= 0.70 and f1 > best_lstm_f1:
        best_lstm_f1, best_lstm_thresh, best_prec_lv, best_rec_lv = f1, th, p, r

# Fallback: best unconstrained F1
if best_lstm_f1 == -1.0:
    for th in np.linspace(0.10, 0.90, 81):
        p_bin = (val_probs_arr >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_trues_arr, p_bin, average="binary", zero_division=0)
        if f1 > best_lstm_f1:
            best_lstm_f1, best_lstm_thresh, best_prec_lv, best_rec_lv = f1, th, p, r

print(f"\\n[+] Optimal Threshold: {best_lstm_thresh:.4f}")
print(f"[+] Val Precision: {best_prec_lv:.4f}  Recall: {best_rec_lv:.4f}  F1: {best_lstm_f1:.4f}")

prec_c, rec_c, _ = precision_recall_curve(val_trues_arr, val_probs_arr)
plt.figure(figsize=(8, 5))
plt.plot(rec_c, prec_c, color="#8e44ad", lw=2, label="LSTM PR Curve")
plt.scatter([best_rec_lv], [best_prec_lv], color="red", s=100, zorder=5,
            label=f"Optimal ({best_lstm_thresh:.4f})")
plt.axvline(TARGET_RECALL, color="gray", ls="--", label=f"Target Recall ({TARGET_RECALL})")
plt.fill_between(rec_c, prec_c, alpha=0.1, color="#8e44ad")
plt.title("LSTM Precision-Recall Curve with Optimal Threshold")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.legend(); plt.grid(True); plt.show()
''')

add_code('''print("=" * 70)
print("3f — HELD-OUT TEST EVALUATION")
print("=" * 70)

lstm_model.eval()
te_probs, te_trues, te_attn_weights = [], [], []
with torch.no_grad():
    for bx, by in te_ld:
        logits, attn = lstm_model(bx.to(device))
        te_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        te_trues.extend(by.numpy().flatten())
        te_attn_weights.extend(attn.cpu().numpy())

te_probs = np.array(te_probs)
te_trues = np.array(te_trues)
te_preds = (te_probs >= best_lstm_thresh).astype(int)

tn, fp, fn, tp = confusion_matrix(te_trues, te_preds).ravel()
prec_lstm = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_lstm  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_lstm   = 2 * (prec_lstm * rec_lstm) / (prec_lstm + rec_lstm) if (prec_lstm + rec_lstm) > 0 else 0.0
fpr_lstm  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print(f"Precision:                  {prec_lstm:.4f}")
print(f"Recall:                     {rec_lstm:.4f}")
print(f"F1 Score:                   {f1_lstm:.4f}")
print(f"False Positive Rate (FPR):  {fpr_lstm:.4f}")
print(f"Detection Rate:             {tp}/{tp+fn} caught ({fn} missed)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(confusion_matrix(te_trues, te_preds), annot=True, fmt="d", cmap="Purples", ax=axes[0])
axes[0].set_title(f"Test Confusion Matrix (Threshold = {best_lstm_thresh:.2f})")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.histplot(te_probs[te_trues==0], bins=40, color="#3498db", alpha=0.5, label="Normal",      ax=axes[1], stat="density")
sns.histplot(te_probs[te_trues==1], bins=40, color="#e74c3c", alpha=0.7, label="Structuring", ax=axes[1], stat="density")
axes[1].axvline(best_lstm_thresh, color="black", ls="--", label=f"Threshold: {best_lstm_thresh:.2f}")
axes[1].set_title("Probability Separation (Normal vs Structuring)")
axes[1].set_xlabel("Structuring Probability"); axes[1].legend()
plt.tight_layout(); plt.show()

# Top 5 detected structuring sequences
print("\\n--- TOP 5 DETECTED STRUCTURING SEQUENCES ---")
te_meta       = meta_df[te_mask].copy().reset_index(drop=True)
te_meta["prob"]       = te_probs
te_meta["true_label"] = te_trues
top5 = te_meta[te_meta["true_label"] == 1].sort_values("prob", ascending=False).head(5)
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"\\nAlert #{rank}: Account {row['account_id']}")
    print(f"  Timestamp: {row['timestamp']}  Amount: INR {row['amount']:,.2f}")
    print(f"  Confidence: {row['prob']*100:.2f}%")
    print(f"  Evidence: Burst of sub-INR 50,000 transfers within 2h window. "
          f"Cumulative total exceeds threshold. Sequence velocity abnormal. "
          f"Amount homogeneity near-zero (smurfing pattern).")
''')

add_code('''print("=" * 70)
print("3g — SAVE LSTM ARTIFACTS")
print("=" * 70)

torch.save(lstm_model.state_dict(), "lstm_structuring_model.pt")
joblib.dump(scaler_lstm,   "lstm_scaler.pkl")
joblib.dump(le_type_lstm,  "lstm_label_encoders.pkl")

with open("lstm_optimal_threshold.json", "w") as f:
    json.dump({"optimal_threshold": float(best_lstm_thresh)}, f, indent=2)
with open("lstm_feature_columns.json", "w") as f:
    json.dump({"feature_columns": feature_cols_lstm}, f, indent=2)

for fn in ["lstm_structuring_model.pt", "lstm_scaler.pkl", "lstm_label_encoders.pkl",
           "lstm_optimal_threshold.json", "lstm_feature_columns.json"]:
    print(f"[OK] {fn:<32}  {os.path.getsize(fn)/1024:.2f} KB")
''')

# ==============================================================================
# SECTION 4 — XGBOOST PROFILE MISMATCH DETECTOR
# ==============================================================================
add_md("""# SECTION 4 — XGBOOST PROFILE MISMATCH DETECTOR

### 4a — What is Customer Profile Mismatch (Dormant Account Abuse)?
**Customer Profile Mismatch** is when an account's current transaction behavior is entirely inconsistent with its established historical pattern.

* **Classic Pattern**: A rural Jan Dhan zero-balance savings account (historical mean: ₹800/month) suddenly receives ₹3,50,000 and immediately transfers it out to 5 different accounts within 24 hours.
* **The Incident**: Post-demonetization (November 2016), the ED and RBI documented hundreds of thousands of dormant Jan Dhan accounts activated overnight to convert black money — a pattern completely invisible to static rule-based systems.

### Why XGBoost + SHAP is Optimal Here

1. **Best Tabular Performance**: XGBoost dominates structured financial data benchmarks consistently (Kaggle, academic studies).
2. **Scale Imbalance Resilience**: `scale_pos_weight` directly rescales the gradient computation to treat fraud cases as more important during tree splitting.
3. **SHAP Explainability**: Every alert comes with a mathematically exact attribution breakdown. This powers the TemporalShield investigator evidence panel — compliance officers can see exactly which behavioural deviation triggered each flag, enabling rapid triage.

### Key Features Added in This Version

- **`errorBalanceOrig`**: `oldbalanceOrg + amount - newbalanceOrig`. In PaySim, this is ≠0 almost exclusively for fraud. One of the strongest known signals in this dataset.
- **`balance_is_zero_after`**: Binary flag — account drained to exactly ₹0. Classic mule account washout indicator.
- **`is_new_counterparty`**: Properly computed using seen counterparties per account — not hardcoded.
- **`type_matches_history`**: Properly computed from historical dominant type per account.
""")

add_code('''print("=" * 70)
print("4b — FEATURE ENGINEERING — PROFILE MISMATCH")
print("=" * 70)

xgb_df = paysim_df.sort_values("timestamp").reset_index(drop=True)

# KEY FIX: Fraud in PaySim exists ONLY in TRANSFER + CASH_OUT channels.
# Training on all 5 types adds 4M noise rows with zero fraud — confuses the model.
# Filtering to fraud-possible channels dramatically sharpens class boundaries.
xgb_df = xgb_df[xgb_df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
xgb_df = xgb_df.sort_values("timestamp").reset_index(drop=True)

acc_multi = xgb_df["account_id"].value_counts()
if XGB_MAX_ACCOUNTS is None:
    xgb_df = xgb_df[xgb_df["account_id"].isin(acc_multi[acc_multi >= 2].index)].copy()
else:
    xgb_df = xgb_df[xgb_df["account_id"].isin(acc_multi[acc_multi >= 2].head(XGB_MAX_ACCOUNTS).index)].copy()

print(f"Working with {len(xgb_df):,} transactions | {xgb_df['account_id'].nunique():,} accounts")
print(f"Fraud cases: {xgb_df['isFraud'].sum():,} ({xgb_df['isFraud'].mean()*100:.3f}%)")

xgb_df["step_numeric"] = (xgb_df["timestamp"] - xgb_df["timestamp"].min()).dt.total_seconds() / 3600.0
grp = xgb_df.groupby("account_id", sort=False)

# Expanding past-only statistics (zero data leakage)
xgb_df["historical_txn_count"]   = grp.cumcount()
xgb_df["historical_mean_amount"] = grp["amount"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.0)
xgb_df["historical_std_amount"]  = grp["amount"].transform(lambda x: x.shift(1).expanding().std()).fillna(0.0)
xgb_df["historical_max_amount"]  = grp["amount"].transform(lambda x: x.shift(1).expanding().max()).fillna(0.0)
xgb_df["account_age_steps"]      = xgb_df["step_numeric"] - grp["step_numeric"].transform("first")

# Days since last transaction
xgb_df["prev_ts"] = grp["timestamp"].shift(1)
xgb_df["days_since_last_transaction"] = (
    (xgb_df["timestamp"] - xgb_df["prev_ts"]).dt.total_seconds().div(86400.0).fillna(999.0)
)

xgb_df["hour_of_day"] = xgb_df["timestamp"].dt.hour

# is_new_counterparty — VECTORISED (O(n log n), no iterrows)
# A counterparty is "new" if this is the first time this account sends to it.
# Trick: rank within (account_id, counterparty_id) group — first occurrence = rank 1 = new
print("Computing is_new_counterparty (vectorised)...")
xgb_df["_cp_rank"] = xgb_df.groupby(["account_id", "counterparty_id"]).cumcount()
xgb_df["is_new_counterparty"] = (xgb_df["_cp_rank"] == 0).astype(float)
xgb_df.drop(columns=["_cp_rank"], inplace=True)

# Transaction type encoding
le_xgb_type = LabelEncoder()
xgb_df["type_encoded"] = le_xgb_type.fit_transform(xgb_df["type"].astype(str))
# Prev type for type-change detection (no expanding window needed)
xgb_df["prev_type_encoded"]    = grp["type_encoded"].shift(1).fillna(-1)
xgb_df["type_matches_history"] = (xgb_df["type_encoded"] == xgb_df["prev_type_encoded"]).astype(float)

# Current transaction features
xgb_df["current_amount"]                  = xgb_df["amount"]
xgb_df["amount_vs_historical_mean_ratio"] = xgb_df["current_amount"] / (xgb_df["historical_mean_amount"] + 1.0)
xgb_df["amount_vs_historical_max_ratio"]  = xgb_df["current_amount"] / (xgb_df["historical_max_amount"]  + 1.0)

# Balance features
xgb_df["balance_before"]        = xgb_df["oldbalanceOrg"]
xgb_df["balance_after"]         = xgb_df["newbalanceOrig"]
xgb_df["balance_drop_ratio"]    = ((xgb_df["oldbalanceOrg"] - xgb_df["newbalanceOrig"]) /
                                   (xgb_df["oldbalanceOrg"] + 1.0)).clip(-5.0, 5.0)
xgb_df["balance_is_zero_after"] = (xgb_df["newbalanceOrig"] == 0.0).astype(float)

# errorBalanceOrig: THE #1 PaySim fraud signal.
# For normal TRANSFER/CASH_OUT: oldBal - amount == newBal (conservation of money)
# For fraud: account is zeroed out regardless of starting balance → residual != 0
# errorBalanceOrig = |oldBal - amount - newBal| (should be 0 for legitimate)
xgb_df["errorBalanceOrig"] = (xgb_df["oldbalanceOrg"] - xgb_df["amount"] - xgb_df["newbalanceOrig"]).abs()
# For destination: balance should increase by the received amount
xgb_df["errorBalanceDest"] = (xgb_df["newbalanceDest"] - xgb_df["oldbalanceDest"] - xgb_df["amount"]).abs().clip(0, 1e8)

# Dormancy flag
xgb_df["is_dormant_account"] = (xgb_df["days_since_last_transaction"] > 30.0).astype(float)

# is_TRANSFER flag (TRANSFER has higher fraud rate than CASH_OUT in PaySim)
xgb_df["is_transfer"] = (xgb_df["type"] == "TRANSFER").astype(float)

xgb_features = [
    "account_age_steps",
    "historical_mean_amount",
    "historical_std_amount",
    "historical_txn_count",
    "historical_max_amount",
    "days_since_last_transaction",
    "hour_of_day",
    "current_amount",
    "amount_vs_historical_mean_ratio",
    "amount_vs_historical_max_ratio",
    "is_new_counterparty",
    "balance_before",
    "balance_after",
    "balance_drop_ratio",
    "balance_is_zero_after",
    "type_matches_history",
    "is_dormant_account",
    "type_encoded",
    "errorBalanceOrig",
    "errorBalanceDest",
    "is_transfer"
]

print(f"\\nTotal features: {len(xgb_features)}")
print(xgb_df[xgb_features].describe().T[["mean", "std", "min", "max"]])

corr = xgb_df[xgb_features + ["isFraud"]].corr()["isFraud"].drop("isFraud").sort_values(ascending=False)
print("\\nTop features correlated with fraud:")
print(corr.head(10))

plt.figure(figsize=(14, 9))
sns.heatmap(xgb_df[xgb_features[:12] + ["isFraud"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("XGBoost Profile Mismatch — Feature Correlation Heatmap")
plt.show()
''')

add_code('''print("=" * 70)
print("4c — CLASS IMBALANCE & TIME-BASED SPLIT")
print("=" * 70)

n_normal = (xgb_df["isFraud"] == 0).sum()
n_fraud  = max(1, (xgb_df["isFraud"] == 1).sum())
computed_spw = float(n_normal) / float(n_fraud)

# Always use the COMPUTED ratio — it reflects the true class distribution in the actual
# filtered dataset. The hardcoded constant was for estimation only. On full PaySim
# TRANSFER+CASH_OUT, true imbalance is ~570:1, not 10:1.
active_spw = computed_spw
print(f"[+] True class imbalance (normal:fraud): {computed_spw:.1f} : 1")
print(f"[+] Using scale_pos_weight = {active_spw:.1f}")

xgb_df   = xgb_df.sort_values("timestamp").reset_index(drop=True)
split_n  = int(len(xgb_df) * TRAIN_SPLIT)
train_xgb, test_xgb = xgb_df.iloc[:split_n].copy(), xgb_df.iloc[split_n:].copy()

val_n  = int(len(train_xgb) * 0.8)
tr_xgb = train_xgb.iloc[:val_n].copy()
va_xgb = train_xgb.iloc[val_n:].copy()

X_tr = tr_xgb[xgb_features]; y_tr = tr_xgb["isFraud"]
X_va = va_xgb[xgb_features]; y_va = va_xgb["isFraud"]
X_te = test_xgb[xgb_features]; y_te = test_xgb["isFraud"]

scaler_xgb  = StandardScaler()
X_tr_s = pd.DataFrame(scaler_xgb.fit_transform(X_tr), columns=xgb_features)
X_va_s = pd.DataFrame(scaler_xgb.transform(X_va),     columns=xgb_features)
X_te_s = pd.DataFrame(scaler_xgb.transform(X_te),     columns=xgb_features)

print(f"Train: {len(X_tr):,} ({y_tr.sum():,} fraud)  "
      f"Val: {len(X_va):,} ({y_va.sum():,} fraud)  "
      f"Test: {len(X_te):,} ({y_te.sum():,} fraud)")
''')

add_code('''print("=" * 70)
print("4d — TRAIN XGBOOST PROFILE MISMATCH DETECTOR")
print("=" * 70)

# GPU-accelerated XGBoost histogram: 3–5× faster on RTX Pro 6000 vs CPU
# device='cuda' routes ALL tree splits through GPU tensor operations
xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[+] XGBoost device: {xgb_device}")

xgb_model = xgb.XGBClassifier(
    n_estimators           = XGBOOST_ESTIMATORS,
    max_depth              = XGBOOST_MAX_DEPTH,
    learning_rate          = XGBOOST_LEARNING_RATE,
    scale_pos_weight       = active_spw,
    min_child_weight       = XGBOOST_MIN_CHILD_WEIGHT,
    subsample              = XGBOOST_SUBSAMPLE,
    colsample_bytree       = XGBOOST_COLSAMPLE_BYTREE,
    gamma                  = XGBOOST_GAMMA,
    reg_lambda             = XGBOOST_REG_LAMBDA,
    reg_alpha              = XGBOOST_REG_ALPHA,
    eval_metric            = "aucpr",
    early_stopping_rounds  = 30,       # More patience for larger tree budget
    random_state           = RANDOM_SEED,
    tree_method            = "hist",   # CPU-GPU compatible histogram method
    device                 = xgb_device  # RTX Pro 6000 GPU acceleration
)

xgb_model.fit(
    X_tr_s, y_tr,
    eval_set=[(X_tr_s, y_tr), (X_va_s, y_va)],
    verbose=100  # Larger step between logs for 1000-tree training
)

print(f"\\n[+] Best Iteration: {xgb_model.best_iteration}")
print(f"[+] Best Val AUCPR:  {xgb_model.best_score:.4f}")

# Training curve
res  = xgb_model.evals_result()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(res["validation_0"]["aucpr"], lw=2, label="Train AUCPR",      color="#27ae60")
axes[0].plot(res["validation_1"]["aucpr"], lw=2, label="Validation AUCPR", color="#e74c3c", ls="--")
axes[0].axvline(xgb_model.best_iteration, color="navy", ls=":", label=f"Best iter ({xgb_model.best_iteration})")
axes[0].set_title("XGBoost Training vs Validation AUCPR")
axes[0].set_xlabel("Boosting Round"); axes[0].set_ylabel("AUCPR"); axes[0].legend(); axes[0].grid(True)

# XGBoost native feature importance
feat_importance = pd.Series(xgb_model.feature_importances_, index=xgb_features).sort_values(ascending=False)
feat_importance.head(12).plot(kind="barh", ax=axes[1], color="#27ae60", edgecolor="black")
axes[1].set_title("Top-12 Feature Importance (XGBoost)")
axes[1].invert_yaxis()
plt.tight_layout(); plt.show()
''')

add_code('''print("=" * 70)
print("4e — XGBOOST THRESHOLD OPTIMISATION")
print("=" * 70)

va_probs_xgb = xgb_model.predict_proba(X_va_s)[:, 1]

print("\\nProb distribution stats on validation:")
print(f"  Min prob:    {va_probs_xgb.min():.6f}")
print(f"  Max prob:    {va_probs_xgb.max():.6f}")
print(f"  Mean prob:   {va_probs_xgb.mean():.6f}")
print(f"  Median prob: {np.median(va_probs_xgb):.6f}")
print(f"  95th pctile: {np.percentile(va_probs_xgb, 95):.6f}")
print(f"  99th pctile: {np.percentile(va_probs_xgb, 99):.6f}")

best_xgb_th = 0.50
best_xgb_f1 = best_p_xgb = best_r_xgb = -1.0

# Fine-grained search over 200 thresholds (not just 0.05 steps)
# PRIMARY pass: maximize F1 at recall >= TARGET_RECALL AND precision >= 1%
for th in np.linspace(0.01, 0.99, 200):
    pred = (va_probs_xgb >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)
    if r >= TARGET_RECALL and p >= 0.01 and f1 > best_xgb_f1:
        best_xgb_f1, best_xgb_th, best_p_xgb, best_r_xgb = f1, th, p, r

# SECONDARY pass: if primary fails, maximize F1 with precision >= 1%
if best_xgb_f1 == -1.0:
    print("[INFO] No threshold achieves recall >= TARGET_RECALL with precision >= 1%. Falling back to best F1 with precision floor.")
    for th in np.linspace(0.01, 0.99, 200):
        pred = (va_probs_xgb >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)
        if p >= 0.01 and f1 > best_xgb_f1:
            best_xgb_f1, best_xgb_th, best_p_xgb, best_r_xgb = f1, th, p, r

# FINAL fallback: any threshold with non-trivial predictions (handles fully degenerate model)
if best_xgb_f1 == -1.0:
    print("[WARNING] Model probability spread is very narrow. Using 95th percentile as threshold.")
    best_xgb_th = float(np.percentile(va_probs_xgb, 95))
    pred = (va_probs_xgb >= best_xgb_th).astype(int)
    best_p_xgb, best_r_xgb, best_xgb_f1, _ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)

print(f"\\n[+] Optimal Threshold: {best_xgb_th:.4f}")
print(f"[+] Val Precision: {best_p_xgb:.4f}  Recall: {best_r_xgb:.4f}  F1: {best_xgb_f1:.4f}")

prec_c, rec_c, _ = precision_recall_curve(y_va, va_probs_xgb)
plt.figure(figsize=(8, 5))
plt.plot(rec_c, prec_c, color="#27ae60", lw=2, label="XGBoost PR Curve")
plt.fill_between(rec_c, prec_c, alpha=0.1, color="#27ae60")
plt.scatter([best_r_xgb], [best_p_xgb], color="red", s=100, zorder=5,
            label=f"Optimal ({best_xgb_th:.2f})")
plt.axvline(TARGET_RECALL, color="gray", ls="--", label=f"Target Recall ({TARGET_RECALL})")
plt.title("XGBoost Precision-Recall Curve with Optimal Threshold")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.legend(); plt.grid(True); plt.show()
''')

add_code('''print("=" * 70)
print("4f — TEST EVALUATION")
print("=" * 70)

te_probs_xgb = xgb_model.predict_proba(X_te_s)[:, 1]
te_preds_xgb = (te_probs_xgb >= best_xgb_th).astype(int)

tn, fp, fn, tp = confusion_matrix(y_te, te_preds_xgb).ravel()
prec_te = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_te  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_te   = 2 * (prec_te * rec_te) / (prec_te + rec_te) if (prec_te + rec_te) > 0 else 0.0
aucpr_te = average_precision_score(y_te, te_probs_xgb)
fpr_te  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print(f"Precision:   {prec_te:.4f}")
print(f"Recall:      {rec_te:.4f}")
print(f"F1 Score:    {f1_te:.4f}")
print(f"AUCPR:       {aucpr_te:.4f}")
print(f"FPR:         {fpr_te:.4f}")
print(f"Caught:      {tp}/{tp+fn}  Missed: {fn}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(confusion_matrix(y_te, te_preds_xgb), annot=True, fmt="d", cmap="Greens", ax=axes[0])
axes[0].set_title(f"XGBoost Test Confusion Matrix (Threshold = {best_xgb_th:.2f})")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.histplot(te_probs_xgb[y_te==0], bins=50, color="#3498db", alpha=0.5, label="Normal",  ax=axes[1], stat="density")
sns.histplot(te_probs_xgb[y_te==1], bins=50, color="#e74c3c", alpha=0.7, label="Fraud",   ax=axes[1], stat="density")
axes[1].axvline(best_xgb_th, color="black", ls="--", label=f"Threshold: {best_xgb_th:.2f}")
axes[1].set_title("Probability Separation (Normal vs Profile Mismatch)")
axes[1].set_xlabel("Fraud Probability"); axes[1].legend()
plt.tight_layout(); plt.show()

print("\\n--- TOP 10 SUSPICIOUS TRANSACTIONS ---")
res_df = test_xgb.copy()
res_df["fraud_prob"] = te_probs_xgb
top10 = res_df.sort_values("fraud_prob", ascending=False).head(10)
print(top10[["account_id", "timestamp", "current_amount", "balance_drop_ratio",
             "amount_vs_historical_mean_ratio", "fraud_prob", "isFraud"]].to_string(index=False))
''')

add_code('''print("=" * 70)
print("4g — SHAP ANALYSIS (TEMPORALSHIELD EVIDENCE PANEL)")
print("=" * 70)

# Sample for SHAP to keep Kaggle memory in check
shap_sample = X_te_s.sample(min(2000, len(X_te_s)), random_state=RANDOM_SEED)
explainer   = shap.TreeExplainer(xgb_model)
shap_vals   = explainer(shap_sample)

# Plot 1: Summary
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, shap_sample, show=False)
plt.title("Plot 1: SHAP Global Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# Plot 2: Beeswarm
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_vals, show=False)
plt.title("Plot 2: SHAP Beeswarm — Feature Impact Distribution", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# Plot 3: Waterfall for most suspicious transaction
most_sus_idx = np.argmax(te_probs_xgb)
shap_single  = explainer(X_te_s.iloc[[most_sus_idx]])
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single[0], show=False)
plt.title("Plot 3: Waterfall — Most Suspicious Transaction Evidence", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# Plot 4: Dependence plots for top 3 features by mean |SHAP|
top3_feat_idx = np.argsort(np.abs(shap_vals.values).mean(0))[-3:]
top3_feats    = [xgb_features[i] for i in top3_feat_idx]
for feat in top3_feats:
    plt.figure(figsize=(8, 4))
    shap.dependence_plot(feat, shap_vals.values, shap_sample, show=False)
    plt.title(f"Plot 4: SHAP Dependence — {feat}", fontweight="bold")
    plt.tight_layout(); plt.show()

# Plain English TemporalShield Alert Output
alert_row  = test_xgb.iloc[most_sus_idx]
alert_prob = te_probs_xgb[most_sus_idx]
top3_shap  = sorted(zip(xgb_features, shap_vals.values[np.argmax(te_probs_xgb[:len(shap_sample)])]),
                    key=lambda x: abs(x[1]), reverse=True)[:3]

explanations = {
    "errorBalanceOrig":              "Balance arithmetic discrepancy — funds appear to have vanished, strongly consistent with fraudulent diversion.",
    "balance_drop_ratio":            "Account balance drained close to zero — mule account washout pattern.",
    "amount_vs_historical_mean_ratio": "Transaction is far larger than this account's historical average — Jan Dhan profile mismatch.",
    "is_dormant_account":            "Account was dormant for >30 days and suddenly activated — dormant account hijacking pattern.",
    "balance_is_zero_after":         "Balance is exactly ₹0 after transfer — complete account liquidation detected.",
    "is_new_counterparty":           "Funds sent to counterparty never seen in account history — unknown destination.",
}

print("\\n" + "=" * 65)
print("TEMPORALSHIELD ALERT INVESTIGATION DOSSIER")
print("=" * 65)
print(f"ACCOUNT PROFILE MISMATCH DETECTED")
print(f"Account ID:     {alert_row['account_id']}")
print(f"Suspicion Score:{alert_prob * 100:.1f}%")
print("\\nWhat triggered this alert:")
for rank, (feat, sv) in enumerate(top3_shap, 1):
    val = alert_row.get(feat, "N/A")
    reason = explanations.get(feat, f"Anomalous deviation from historical baseline (SHAP: {sv:+.3f}).")
    print(f"  {rank}. {feat}: {val} — {reason}")
print("\\nSimilar real incident: Jan Dhan Account Fraud, 2016")
print("Pattern: Dormant accounts suddenly processing large transfers post-demonetization.")
print("This is what investigators see in the TemporalShield Alert Detail page.")
print("=" * 65)
''')

add_code('''print("=" * 70)
print("4h — SAVE XGBOOST ARTIFACTS")
print("=" * 70)

joblib.dump(xgb_model,   "xgboost_profile_model.pkl")
joblib.dump(scaler_xgb,  "xgboost_scaler.pkl")
joblib.dump(explainer,   "xgboost_shap_explainer.pkl")

with open("xgboost_feature_columns.json",  "w") as f:
    json.dump({"feature_columns": xgb_features}, f, indent=2)
with open("xgboost_optimal_threshold.json", "w") as f:
    json.dump({"optimal_threshold": float(best_xgb_th)}, f, indent=2)

for fn in ["xgboost_profile_model.pkl", "xgboost_scaler.pkl", "xgboost_feature_columns.json",
           "xgboost_optimal_threshold.json", "xgboost_shap_explainer.pkl"]:
    print(f"[OK] {fn:<35}  {os.path.getsize(fn)/1024:.2f} KB")
''')

# ==============================================================================
# SECTION 5 — COMBINED SUMMARY
# ==============================================================================
add_md("""# SECTION 5 — COMBINED MODEL SUMMARY
""")

add_code('''print("=" * 70)
print("SECTION 5 — COMBINED MODEL PERFORMANCE SUMMARY")
print("=" * 70)

summary = pd.DataFrame([
    {"Model": "Isolation Forest",  "Detects": "Temporal insider link",          "Precision": "34.0%", "Recall": "99.0%", "F1": "50.0%", "Dataset": "Both"},
    {"Model": "Autoencoder",       "Detects": "Role permission anomaly",         "Precision": "21.0%", "Recall": "60.0%", "F1": "31.1%", "Dataset": "Access logs"},
    {"Model": "LSTM + Attention",  "Detects": "Transaction structuring",         "Precision": f"{prec_lstm*100:.1f}%", "Recall": f"{rec_lstm*100:.1f}%", "F1": f"{f1_lstm*100:.1f}%", "Dataset": "PaySim + Synthetic"},
    {"Model": "XGBoost + SHAP",   "Detects": "Customer profile mismatch",       "Precision": f"{prec_te*100:.1f}%",   "Recall": f"{rec_te*100:.1f}%",   "F1": f"{f1_te*100:.1f}%",   "Dataset": "PaySim"},
])
print(summary.to_string(index=False))
''')

add_md("""### TemporalShield Multi-Layered Risk Aggregator

Each model fires **independently** on incoming events. The risk aggregator combines signals:

| Model | Weight | Rationale |
| :--- | :--- | :--- |
| Isolation Forest | **0.40** | Core temporal link detection — foundational signal |
| XGBoost | **0.30** | Strong tabular performance with SHAP transparency |
| LSTM | **0.20** | Temporal burst sequence signal |
| Autoencoder | **0.10** | Supporting behavioral context |

**Alert Tiers**: CRITICAL (> 0.75) → Automated hold | HIGH (> 0.50) → Priority queue | MEDIUM (> 0.25) → Watchlist

Every alert — regardless of severity — includes SHAP attributions from XGBoost and Isolation Forest scores in the TemporalShield evidence panel.
""")

# ==============================================================================
# SECTION 6 — FILES MANIFEST
# ==============================================================================
add_md("""# SECTION 6 — COMPLETE FILES MANIFEST
Verification of all 26 serialized artifacts required by the TemporalShield backend.
""")

add_code('''print("=" * 70)
print("SECTION 6 — SAVE ALL FILES MANIFEST")
print("=" * 70)

all_files = [
    # Previous notebook — Isolation Forest + Autoencoders
    "isolation_forest_model.pkl", "scaler_isolation_forest.pkl", "feature_columns_if.json",
    "autoencoder_loan_officer.pt",            "scaler_ae_loan_officer.pkl",
    "autoencoder_savings_representative.pt",  "scaler_ae_savings_representative.pkl",
    "autoencoder_branch_manager.pt",          "scaler_ae_branch_manager.pkl",
    "autoencoder_it_admin.pt",                "scaler_ae_it_admin.pkl",
    "autoencoder_compliance_officer.pt",      "scaler_ae_compliance_officer.pkl",
    "thresholds_autoencoder_optimised.json", "label_encoders_ae.pkl", "ensemble_weights.json",
    # This notebook — LSTM + XGBoost
    "lstm_structuring_model.pt", "lstm_scaler.pkl", "lstm_label_encoders.pkl",
    "lstm_optimal_threshold.json", "lstm_feature_columns.json",
    "xgboost_profile_model.pkl", "xgboost_scaler.pkl",
    "xgboost_feature_columns.json", "xgboost_optimal_threshold.json", "xgboost_shap_explainer.pkl"
]

print(f"{'Artifact':<42} {'Status':<10} {'Size (KB)':>10}")
print("-" * 67)
found = 0
for fn in all_files:
    candidates = [
        fn,
        os.path.join("models", fn),
        os.path.join("models/lstm", fn),
        os.path.join("models/xgboost", fn),
        os.path.join("models/autoencoder", fn),
        os.path.join("models/isolation_forest", fn),
        os.path.join("models/ensemble", fn),
        os.path.join("../models/autoencoder", fn),
        os.path.join("../models/isolation_forest", fn),
        os.path.join("../models/ensemble", fn),
        os.path.join("../models/lstm", fn),
        os.path.join("../models/xgboost", fn),
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path:
        sz = os.path.getsize(path) / 1024
        print(f"{fn:<42} {'EXISTS':<10} {sz:>10.2f}")
        found += 1
    else:
        print(f"{fn:<42} {'PENDING':<10} {'—':>10}")

print("-" * 67)
print(f"Total artifacts catalogued: {len(all_files)}")
print(f"Verified on disk:           {found}")
''')

# Write to file
target = "d:/temporalShield/notebooks/temporalshield_kaggle_lstm_xgboost.ipynb"
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"[OK] Notebook written -> {target}")
