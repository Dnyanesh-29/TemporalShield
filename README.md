# TemporalShield 🛡️ (NexusTrace)
### Multi-Tiered Financial Fraud Detection & Temporal Insider Threat Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-00599c.svg)](https://networkx.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU--Accelerated-green.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TemporalShield (NexusTrace) is an enterprise-grade AI/ML fraud intelligence platform engineered to detect sophisticated financial crimes that evade traditional static rule engines. It links internal bank employee access behaviors with outbound transactional fund flows, localized temporal burst anomalies, circular money laundering rings, and account profile deviations.

---

## ⚡ Headline Capability: The 4-Minute Detection Window

In documented banking insider frauds (e.g., Citibank India 2010), rogue employees access high-net-worth or dormant customer accounts to harvest credentials or suppress fraud flags immediately prior to an external transfer.

TemporalShield's **Temporal Link Engine** monitors employee lookup events and subsequent customer transactions on the same account using unsupervised anomaly detection. When funds leave within **4 minutes** of internal employee lookup, TemporalShield calculates real-time collusion risk and triggers an automated hold before settlement.

---

## 🧠 End-to-End System Architecture

```
                       Live Event Ingestion / Replay
           ┌───────────────────────────────────────────────────┐
           │ access_logs.csv (63k+)   synthetic_txns.csv (7k+) │
           └─────────┬─────────────────────────┬───────────────┘
                     ▼                         ▼
           ┌──────────────────┐       ┌──────────────────┐
           │ Topic: access    │       │ Topic: txns      │
           └─────────┬────────┘       └────────┬─────────┘
                     └────────────┬────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ Stream Consumer Loop      │
                    │ (Kafka / In-Memory Queue) │
                    └─────────────┬─────────────┘
                                  ▼
           ┌─────────────────────────────────────────────┐
           │ Incremental Temporal Graph (NetworkX)       │
           │ • Dynamic Time Decay: w = exp(-λ·Δt)        │
           │ • 4-Minute Bi-Directional Cross-Check       │
           └──────────────────────┬──────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    ▼                             ▼                             ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Isolation Forest │     │ Role Autoencoder │     │ LSTM + Attention │
│ 4-Min Link Window│     │ Privilege Abuse  │     │ Smurfing Burst   │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │ (Weight: 0.35)         │ (Weight: 0.05)         │ (Weight: 0.15)
         └────────────────────────┼────────────────────────┘
                                  ▼
    ┌─────────────────────────────┼─────────────────────────────┐
    ▼                             ▼                             ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ XGBoost + SHAP   │     │ Circular Detector│     │ Node2Vec Embed   │
│ Profile Mismatch │     │ Saradha Cycles   │     │ Mule Clustering  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │ (Weight: 0.20)         │ (Weight: 0.25)         │
         └────────────────────────┼────────────────────────┘
                                  ▼
              ┌───────────────────────────────────────┐
              │ Central Multi-Tier Risk Aggregator    │
              │ Composite Score (0-100) & Severity    │
              │ (CRITICAL / HIGH / MEDIUM / LOW)      │
              └───────────────────┬───────────────────┘
                                  ▼
              ┌───────────────────────────────────────┐
              │ RESTful FastAPI Backend (/docs)       │
              │ • Live Graph Snapshots                │
              │ • Investigation Dossier & SHAP        │
              │ • Demo Scenario Planner Replay        │
              └───────────────────────────────────────┘
```

---

## 🏛️ Real-World Indian Incident Grounding & Regulatory Framework

Every alert produced by TemporalShield is mapped to real-world Indian regulatory directives (RBI PMLA 2002, FIU-IND Red Flag Indicators) and documented banking fraud cases:

| Threat Pattern | Detection Engine | Historical Case Parallel | Regulatory Grounding |
| :--- | :--- | :--- | :--- |
| **Temporal Insider Collusion** | Isolation Forest (4-min window) | **Citibank India Wealth Scam (2010, ₹400 Cr)**: Relationship manager accessed dormant accounts right before fraudulent fund diversion. | PMLA 2002 Section 3; RBI Master Direction on Fraud Classification (RBI/2016-17/13). |
| **Circular Transfer Rings** | Circular Detector + Node2Vec | **Saradha Syndicate Fraud (2013, ₹2,500 Cr)**: Rapid layered fund movement across shell accounts to fake commercial velocity. | FIU-IND Red Flag Indicator RFI-08 (Layered circular flows); SEBI CIS Regulations. |
| **Hawala Structuring / Smurfing** | LSTM + Multi-Head Attention | **Hawala Smurfing Syndicates (Ongoing)**: Splitting sums into sub-₹50,000 bursts to bypass mandatory Cash Transaction Reports (CTR). | PMLA 2002 Rule 3(1)(A); Mandatory reporting on transfers $> \text{INR 50,000}$. |
| **Privilege / Scope Abuse** | Deep Role Autoencoders (x5) | **Punjab National Bank (2018, ₹11,400 Cr)**: Off-hours terminal usage and LoU issuance with zero CBS journal entries. | Prevention of Corruption Act 1988, IPC 409; RBI Directive on SWIFT-CBS Integration. |
| **Dormant Account Hijacking** | XGBoost + TreeSHAP | **Jan Dhan Post-Demonetization (2016)**: Low-volume accounts suddenly processing massive electronic transfers. | Benami Transactions Amendment Act 2016; Income Tax 'Operation Clean Money'. |

---

## 📁 Repository Structure & Modules

```
TemporalShield/
│
├── api/                                  # API & Service Integration Layer
│   ├── __init__.py                       # Package exports
│   ├── main.py                           # FastAPI application exposing alerts, graph, and scenarios
│   ├── schemas.py                        # Pydantic contract between backend and frontend
│   └── evidence_builder.py               # SHAP Evidence Assembler wrapper
│
├── data/                                 # Ingestion, Replay & Offline Data Loaders
│   ├── kafka_producer.py                 # Timestamp-ordered dual-topic stream publisher (Kafka/Memory)
│   ├── kafka_consumer.py                 # Real-time event loop dispatching events to graph and ML
│   ├── scenario_planner.py               # Demo scenario replay controller (Citibank, Hawala, Saradha)
│   ├── event_bus.py                      # Thread-safe in-memory queues when Kafka is unavailable
│   ├── paysim_loader.py                  # PaySim dataset loader for offline LSTM/XGBoost training
│   ├── access_logs.csv                   # Synthetic bank employee access logs (63,000+ rows)
│   ├── synthetic_transactions.csv        # Synthetic transactions (7,143 rows) sharing ACC_XXXXX IDs
│   └── faker_access_logs.py              # Synthetic employee log generator with 5 roles & shifts
│
├── graph/                                # Incremental Graph Engine & Ring Detectors
│   ├── __init__.py                       # Package exports
│   ├── temporal_graph.py                 # Live NetworkX multigraph with exponential time decay
│   ├── circular_detector.py              # Simple cycle detector combined with Node2Vec ring cohesion
│   ├── node2vec_embed.py                 # Graph structural representation learning
│   └── risk_aggregator.py                # 5-model weighted ensemble with CRITICAL/HIGH/MEDIUM tiers
│
├── models/                               # Machine Learning Inference Engines & Weights
│   ├── __init__.py                       # Package exports
│   ├── isolation_forest.py               # Temporal Insider Link Detector (4-min window)
│   ├── autoencoder_role.py               # Role Permission Anomaly Detectors (5 roles)
│   ├── lstm_structuring.py               # Transaction Structuring / Smurfing Detector
│   ├── xgboost_profile.py                # Customer Profile Mismatch & Dormant Abuse Detector
│   ├── shap_explainer.py                 # Central SHAP Evidence Dossier Assembler
│   │
│   ├── isolation_forest/                 # Serialized model, scaler, feature columns
│   ├── autoencoder/                      # 5 role PyTorch autoencoders (.pt), scalers, thresholds
│   ├── lstm/                             # PyTorch StructuringLSTM (.pt), scaler, thresholds
│   ├── xgboost/                          # XGBoost model (.pkl), scaler, SHAP explainer
│   └── ensemble/                         # Aggregator weights config
│
├── requirements.txt                      # Complete Python package dependencies
└── README.md                             # Comprehensive documentation
```

---

## 🚀 How to Run the Project

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Dnyanesh-29/TemporalShield.git
cd TemporalShield

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install all required dependencies
pip install -r requirements.txt
```

---

### 2. Start the FastAPI Intelligence Backend

Start the FastAPI application with Uvicorn:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Once started:
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Root Healthcheck**: [http://localhost:8000/](http://localhost:8000/)

> **Note on Kafka:** Kafka is completely optional. If Apache Kafka is not running, the system automatically falls back to high-performance, thread-safe in-memory queues (`data/event_bus.py`). Everything works immediately with zero broker setup!

---

### 3. Demo Scenarios & Interactive Execution

TemporalShield comes with pre-configured demonstration scenarios that can be triggered on demand:

#### Available Scenarios:
1. `temporal_link`: 4-minute insider lookup before transfer (Citibank Scam).
2. `structuring`: Hawala smurfing burst below ₹50,000 threshold.
3. `circular_transfer`: Multi-hop laundering loop ($A \to B \to C \to A$) across shell accounts (Saradha Scam).
4. `clean_busy`: High-velocity benign commercial operations (validates $< 1.5\%$ false positive rate).

#### Trigger via cURL:
```bash
# Trigger the 4-Minute Insider Link scenario
curl -X POST "http://localhost:8000/scenario" \
     -H "Content-Type: application/json" \
     -d '{"scenario": "temporal_link", "speed_multiplier": 500.0}'

# Trigger Circular Transfer Ring scenario
curl -X POST "http://localhost:8000/scenario" \
     -H "Content-Type: application/json" \
     -d '{"scenario": "circular_transfer", "speed_multiplier": 500.0}'
```

#### Inspect Live Alerts & Graph:
```bash
# Get active alerts sorted by severity (CRITICAL first)
curl -X GET "http://localhost:8000/alerts"

# Retrieve full forensic evidence dossier for an alert
curl -X GET "http://localhost:8000/alerts/ALT_INIT_10017/evidence"

# Retrieve live graph snapshot for frontend rendering
curl -X GET "http://localhost:8000/graph"

# Reset all state and queues
curl -X DELETE "http://localhost:8000/reset"
```

---

### 4. Run Detectors via Python API

You can evaluate transactions directly inside Python scripts:

```python
from api import EvidenceBuilderService

service = EvidenceBuilderService()

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

print(f"Risk Score : {dossier['composite_risk_score']}% ({dossier['severity']})")
print(f"Action     : {dossier['recommended_action']}")
print(f"Incident   : {dossier['historical_incident_benchmark']['incident_name']}")
```

---

### 5. Offline PaySim Dataset Training (Optional)

The PaySim dataset (6M+ transactions) is used strictly for offline model training (LSTM sequences and XGBoost baselines). PaySim is **not** streamed via Kafka because it uses distinct account IDs (`C12345...`) incompatible with branch access logs.

To load and engineer features from PaySim:

```python
from data.paysim_loader import PaySimLoader

loader = PaySimLoader()
df_raw = loader.load_raw(nrows=50000)

# Build LSTM sequences for smurfing detection
X_lstm, y_lstm, lstm_features = loader.prepare_lstm_sequences(df_raw, sequence_length=20)

# Build XGBoost baseline deviation features
X_xgb, y_xgb, xgb_features = loader.prepare_xgboost_features(df_raw)
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root endpoint displaying system health and capabilities. |
| `GET` | `/alerts` | List of active alerts sorted by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`). |
| `GET` | `/alerts/{id}` | Detailed alert breakdown with feature drivers and recommended actions. |
| `GET` | `/alerts/{id}/evidence` | Full evidence dossier including TreeSHAP contributions, PMLA benchmark, and local sub-graph. |
| `GET` | `/graph` | Current snapshot of the temporal graph (nodes, edges, weights, time decay). |
| `GET` | `/scenarios` | Catalogue of all available demonstration scenarios and expected telemetry. |
| `POST` | `/scenario` | Trigger on-demand streaming replay of a pre-filtered scenario subset. |
| `GET` | `/stats` | Real-time system telemetry: alerts generated, false positive rate, patterns detected. |
| `DELETE` | `/reset` | Flush event queues, reset graph nodes, and restore baseline demo state. |

---

## 📜 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
