# TemporalShield — Models & Detection Engines

This package houses the 5 core machine learning detection engines, the central SHAP evidence assembler, and all 26 serialized model artifacts.

---

## Directory & Package Structure

```
models/
├── __init__.py                           # Clean package exports for all 5 detectors
├── isolation_forest.py                   # Layer 1: Temporal Insider Link (4-min window)
├── autoencoder_role.py                   # Layer 2: Role Permission Abuse (5 role autoencoders)
├── lstm_structuring.py                   # Layer 3: Transaction Structuring / Smurfing (LSTM+Attn)
├── xgboost_profile.py                    # Layer 4: Customer Profile Mismatch (XGBoost + SHAP)
├── shap_explainer.py                     # Layer 5: Central SHAP Evidence Dossier Assembler
├── README.md                             # Architectural documentation & benchmarks
│
├── isolation_forest/                     # [Layer 1 Serialized Weights & Scalers]
│   ├── isolation_forest_model.pkl        (1,573 KB)
│   ├── scaler_isolation_forest.pkl       (1.17 KB)
│   └── feature_columns_if.json           (0.17 KB)
│
├── autoencoder/                          # [Layer 2 Serialized Weights & Scalers]
│   ├── autoencoder_branch_manager.pt     (20.25 KB)
│   ├── autoencoder_compliance_officer.pt (20.40 KB)
│   ├── autoencoder_it_admin.pt           (19.91 KB)
│   ├── autoencoder_loan_officer.pt       (20.18 KB)
│   ├── autoencoder_savings_rep.pt        (20.54 KB)
│   ├── scaler_ae_*.pkl                   (5 scalers, 1.12 KB each)
│   ├── label_encoders_ae.pkl             (1.11 KB)
│   └── thresholds_autoencoder_opt.json   (0.19 KB)
│
├── lstm/                                 # [Layer 3 Serialized Weights & Scalers]
│   ├── lstm_structuring_model.pt         (5,200 KB) — 3-layer LSTM with Bahdanau Attention
│   ├── lstm_scaler.pkl                   (1.35 KB)
│   ├── lstm_label_encoders.pkl           (0.51 KB)
│   ├── lstm_optimal_threshold.json       (0.03 KB)
│   └── lstm_feature_columns.json         (0.31 KB)
│
├── xgboost/                              # [Layer 4 Serialized Weights & Scalers]
│   ├── xgboost_profile_model.pkl         (33.03 KB) — 1000-tree GPU Hist XGBoost
│   ├── xgboost_scaler.pkl                (1.76 KB)
│   ├── xgboost_shap_explainer.pkl        (74.30 KB) — TreeSHAP Explainer
│   ├── xgboost_optimal_threshold.json    (0.04 KB)
│   └── xgboost_feature_columns.json      (0.58 KB)
│
└── ensemble/                             # [Layer 5 Ensemble Configuration]
    └── ensemble_weights.json             (IF: 0.40, XGB: 0.30, LSTM: 0.20, AE: 0.10)
```

---

## Clean Python Package Import

You can import any detector directly from `models`:

```python
from models import (
    TemporalInsiderDetector,
    RoleAnomalyDetector,
    StructuringDetector,
    ProfileMismatchDetector,
    SHAPEvidenceAssembler
)

# Example: Run structuring detection on a sequence of transactions
lstm = StructuringDetector()
result = lstm.predict(recent_transactions)
print(result["confidence_score"], result["structuring_detected"])
```

---

## Model Benchmark Summary

| Model | Detection Target | Precision | Recall | F1 Score | Indian Banking Incident Parallel |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Isolation Forest** | Temporal insider collusion | 34.0% | 99.0% | 50.0% | Citibank India (2010, ₹400 Cr) |
| **Autoencoders (x5)** | Role permission privilege abuse | 21.0% | 60.0% | 31.1% | Punjab National Bank (2018, ₹11,400 Cr) |
| **LSTM + Attention** | Hawala / structuring bursts | 51.9% | 66.7% | 58.3% | Hawala Syndicates (ED Ongoing) |
| **XGBoost + SHAP** | Mule washout & profile mismatch | 80.0% | 100.0% | 88.9% | Jan Dhan Demonetization (2016) |

---

## Total File Manifest
* **Python Modules**: 5 detectors + 1 evidence assembler + `__init__.py`
* **Serialized Model Artifacts**: 26 files (100% verified on disk)
