# TemporalShield 🛡️
### Multi-Tiered Financial Fraud Detection & Temporal Insider Threat Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU--Accelerated-green.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TemporalShield is an advanced AI/ML fraud intelligence platform engineered to detect sophisticated financial crimes that evade traditional static rule engines. It links internal bank employee access behaviors with outbound transactional fund flows, localized temporal burst anomalies, and account profile deviations.

---

## ⚡ Headline Capability: The 4-Minute Detection Window

In documented banking insider frauds (e.g., Citibank India 2010), rogue employees access high-net-worth or dormant customer accounts to harvest details or lower fraud flags immediately prior to an external transfer. 

TemporalShield's **Temporal Link Engine** models employee access events and subsequent customer transactions on the same account using unsupervised anomaly detection. When funds leave within **4 minutes** of internal employee lookup, TemporalShield calculates real-time collusion risk and triggers an immediate hold before settlement.

---

## 🧠 Multi-Layered Detection Architecture

TemporalShield operates **4 independent machine learning detection engines** synthesized by a multi-layered risk aggregation layer:

```
                                  Incoming Live Stream
                                           │
         ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
         ▼                   ▼                           ▼                   ▼
┌──────────────────┐┌──────────────────┐   ┌──────────────────┐┌──────────────────┐
│ Isolation Forest ││ Role Autoencoder │   │  LSTM+Attention  ││  XGBoost + SHAP  │
│  Temporal Link   ││ Privilege Abuse  │   │  Smurfing Burst  ││ Profile Mismatch │
└────────┬─────────┘└────────┬─────────┘   └────────┬─────────┘└────────┬─────────┘
         │ (Weight: 0.40)    │ (Weight: 0.10)       │ (Weight: 0.20)    │ (Weight: 0.30)
         └───────────────────┼──────────────────────┴───────────────────┘
                             ▼
             ┌───────────────────────────────┐
             │ Multi-Layer Risk Aggregator   │
             │   & Central SHAP Assembler    │
             └───────────────┬───────────────┘
                             ▼
             ┌───────────────────────────────┐
             │  Investigation Alert Dossier  │
             │  • Severity Tier (CRITICAL)   │
             │  • TreeSHAP Attributions      │
             │  • Indian Incident Benchmark  │
             │  • Automated Hold Action      │
             └───────────────────────────────┘
```

---

## 📊 Model Benchmark Matrix

| Layer | Target Threat | Precision | Recall | F1 Score | Indian Banking Case Benchmark |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Layer 1: Isolation Forest** | Temporal insider collusion (4-min window) | **34.0%** | **99.0%** | **50.0%** | Citibank India (2010, ₹400 Cr) |
| **Layer 2: Deep Autoencoders (x5)** | Role privilege abuse (off-hours / scope) | **21.0%** | **60.0%** | **31.1%** | Punjab National Bank (2018, ₹11,400 Cr) |
| **Layer 3: LSTM + Attention** | Hawala transaction structuring (sub-₹50k) | **51.9%** | **66.7%** | **58.3%** | Hawala Smurfing Syndicates (ED Ongoing) |
| **Layer 4: XGBoost + TreeSHAP** | Customer profile mismatch & dormant hijack | **80.0%** | **100.0%** | **88.9%** | Jan Dhan Post-Demonetization (2016) |

---

## 🏛️ Real-World Indian Incident Grounding

Every alert produced by TemporalShield is mapped to real-world Indian regulatory frameworks (RBI PMLA 2002, FIU-IND RFIs) and historical banking cases:

1. **Temporal Insider Link** $\rightarrow$ **Citibank India Wealth Scam (2010, ₹400 Cr)**: Relationship managers accessed accounts right before unauthorized transfers.
2. **Circular Transfers** $\rightarrow$ **Saradha Scam (2013, ₹2,500 Cr)**: Rapid layered fund movement across shell accounts.
3. **Smurfing / Structuring** $\rightarrow$ **Hawala Syndicates (Ongoing)**: Splitting sums into sub-₹50,000 bursts to bypass mandatory Cash Transaction Reports (CTR).
4. **Role Permission Abuse** $\rightarrow$ **PNB Brady House (2018, ₹11,400 Cr)**: Off-hours SWIFT terminal usage with zero CBS journal entries.
5. **Profile Mismatch** $\rightarrow$ **Jan Dhan Dormant Hijack (2016)**: Low-volume accounts suddenly processing massive transactions.

---

## 📁 Repository Structure

```
temporalShield/
│
├── api/                                  # API & Service Integration Layer
│   ├── __init__.py                       # Package export: EvidenceBuilderService
│   └── evidence_builder.py               # FastAPI-facing wrapper around SHAP evidence assembler
│
├── models/                               # Machine Learning Inference Engines & Weights
│   ├── __init__.py                       # Clean package exports for all 5 ML modules
│   ├── isolation_forest.py               # Temporal Insider Link Detector (4-min window)
│   ├── autoencoder_role.py               # Role Permission Anomaly Detectors (5 roles)
│   ├── lstm_structuring.py               # Transaction Structuring / Smurfing Detector
│   ├── xgboost_profile.py                # Customer Profile Mismatch & Dormant Abuse Detector
│   ├── shap_explainer.py                 # Central SHAP Evidence Dossier Assembler
│   ├── README.md                         # Detailed model directory manifest & benchmarks
│   │
│   ├── isolation_forest/                 # Serialized weights (model.pkl, scaler.pkl, cols.json)
│   ├── autoencoder/                      # 5 role PyTorch autoencoders (.pt), scalers, thresholds
│   ├── lstm/                             # PyTorch StructuringLSTM (.pt), scaler, thresholds
│   ├── xgboost/                          # XGBoost model (.pkl), scaler, SHAP explainer
│   └── ensemble/                         # Ensemble risk aggregator weights (weights.json)
│
├── data/                                 # Datasets & Ingestion Generators
│   ├── access_logs.csv                   # Historical & augmented access logs
│   ├── synthetic_transactions.csv        # Ground-truth synthetic transaction dataset
│   └── faker_access_logs.py              # Synthetic employee log generator with 5 roles & shifts
│
├── notebooks/                            # Evaluation & Presentation Notebooks
│   ├── model_training.ipynb              # Comprehensive 5-model pipeline for review
│   ├── temporalshield_kaggle_lstm_xgboost.ipynb # Kaggle-optimized GPU training notebook
│   └── temporalshield_models.ipynb       # Isolation Forest & Autoencoder training notebook
│
└── scripts/                              # Utility & Data Pipeline Scripts
    ├── faker_access_logs.py              # Employee access log generator
    ├── generate_kaggle_notebook.py       # Kaggle notebook generator
    ├── generate_full_model_training_nb.py# Full training notebook generator
    ├── run_models.py                     # Standalone CLI training & evaluation
    └── synthetic_transactions.py         # Transaction synthesizer
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Dnyanesh-29/TemporalShield.git
cd TemporalShield
pip install torch numpy pandas scikit-learn xgboost shap joblib matplotlib seaborn
```

### 2. Run All Detectors via Python API
```python
from models import (
    TemporalInsiderDetector,
    RoleAnomalyDetector,
    StructuringDetector,
    ProfileMismatchDetector,
    SHAPEvidenceAssembler
)
from api import EvidenceBuilderService

# Initialize the central evidence builder service
service = EvidenceBuilderService()

# Evaluate an incoming live transaction with recent employee access
dossier = service.evaluate_transaction_event(
    account_id="ACC_10017",
    transaction_event={
        "transaction_id": "TXN_99182",
        "amount": 48500.0,
        "type": "TRANSFER",
        "counterparty_id": "EXT_9999",
        "timestamp": "2026-09-01 10:03:45"
    },
    recent_employee_access={
        "employee_id": "EMP_0019",
        "role": "loan_officer",
        "account_id": "ACC_10017",
        "account_type": "savings",
        "timestamp": "2026-09-01 10:00:00"
    }
)

print(f"Risk Score: {dossier['composite_risk_score']}% ({dossier['severity']})")
print(f"Action: {dossier['recommended_action']}")
print(f"Incident: {dossier['historical_incident_benchmark']['incident_name']}")
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
