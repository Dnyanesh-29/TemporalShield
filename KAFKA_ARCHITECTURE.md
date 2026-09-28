# TemporalShield / NexusTrace — Kafka Streaming Architecture

> **Direct Answer:** **YES, the Kafka streaming logic is fully implemented and operational.**
> It is located across [`data/kafka_producer.py`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/data/kafka_producer.py), [`data/kafka_consumer.py`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/data/kafka_consumer.py), [`data/event_bus.py`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/data/event_bus.py), and [`data/scenario_planner.py`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/data/scenario_planner.py).

---

## 1. What Kafka Does and Does NOT Do in TemporalShield

A critical design principle of NexusTrace / TemporalShield is predictability and realistic replay:

- **What Kafka DOES NOT Do:**
  - Kafka **does NOT** create or invent synthetic data at runtime.
  - All event ground-truth was generated systematically and persisted in CSV format (`access_logs.csv` and `synthetic_transactions.csv`).
- **What Kafka DOES Do:**
  - Kafka acts as a **high-throughput, ordered delivery mechanism** (like a postman delivering pre-written letters one by one at a controlled pace).
  - It reads rows from both CSV files, sorts/interleaves them strictly by chronological timestamp, and publishes them as live JSON events.
  - This simulates a live core banking network where employee terminal lookups and transaction settlement requests arrive asynchronously in real time.

---

## 2. End-to-End Streaming Pipeline

```
                    ┌───────────────────────────┐
                    │ access_logs.csv (63,000+) │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ synthetic_txns.csv (7,143)│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Chronological Merger    │
                    │   & 500x Speed Slider     │
                    │ (data/kafka_producer.py)  │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼ [Kafka Connected]                             ▼ [Kafka Unavailable]
┌───────────────────────────────────┐           ┌───────────────────────────────────┐
│     Apache Kafka Broker           │           │ Thread-Safe In-Memory Queues      │
│  Topic 1: access_events           │           │ (data/event_bus.py)               │
│  Topic 2: transactions            │           │ • access_events_queue             │
│  Topic 3: alerts                  │           │ • transactions_queue              │
└─────────────────┬─────────────────┘           └─────────────────┬─────────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Stream Consumer Event Loop    │
                         │   (data/kafka_consumer.py)      │
                         └────────────────┬────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
     [When access_event arrives]                     [When transaction arrives]
     1. Add employee→account edge to graph           1. Add account→counterparty edge
     2. Run Autoencoder (Privilege Abuse)            2. Run LSTM Structuring (Smurfing)
     3. 4-Min Cross-Check vs recent txns             3. Run XGBoost Profile Mismatch
                                                     4. 4-Min Cross-Check vs access
                                                     5. Check Circular Rings (Saradha)
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │ Central Risk Aggregator         │
                         │ Weighted Ensemble + SHAP        │
                         └────────────────┬────────────────┘
                                          │
                                          ▼ (If Score >= Threshold)
                         ┌─────────────────────────────────┐
                         │ Investigation Alert Dossier     │
                         │ (CRITICAL / HIGH / MEDIUM)      │
                         └─────────────────────────────────┘
```

---

## 3. The Two Streaming Topics & Event Shapes

PaySim is **not** streamed through Kafka — PaySim uses disparate `C123456789` IDs incompatible with internal bank employee logs. Only synthetic datasets share the `ACC_XXXXX` identifier schema required for graph linking.

### Topic 1: `access_events`
Published whenever an internal bank employee searches, updates, or views an account:
```json
{
  "employee_id": "EMP_0019",
  "role": "loan_officer",
  "account_id": "ACC_10017",
  "account_type": "savings",
  "branch_code": "BR_006",
  "action_type": "read",
  "timestamp": "2026-09-01 10:00:00",
  "records_accessed": 1,
  "is_suspicious": 1
}
```

### Topic 2: `transactions`
Published whenever a customer executes an outbound or internal fund movement:
```json
{
  "transaction_id": "TXN_000042",
  "account_id": "ACC_10017",
  "amount": 48500.0,
  "transaction_type": "TRANSFER",
  "counterparty_id": "EXT_10481",
  "timestamp": "2026-09-01 10:02:30",
  "branch_code": "BR_006",
  "is_fraud": 1,
  "scenario": "temporal_link"
}
```

---

## 4. Key Streaming Behaviors

### 1. Chronological Interleaving
Transactions and employee accesses occur concurrently in the real world. The producer reads both files, parses their ISO timestamps, and executes a merge-sort so that events are emitted in strict chronological order across both topics.

### 2. 500x Speed Multiplier (Demo Compression)
In historical data, 7,143 transactions and 63,000+ access logs span 30 calendar days ($2,592,000$ seconds). Waiting 30 days is impossible during a live hackathon evaluation.
- The producer calculates the inter-arrival delta: $\Delta t_{\text{delay}} = \frac{t_{i} - t_{i-1}}{\text{speed\_multiplier}}$.
- Default multiplier `500.0` compresses 30 days into **~60 seconds of live streaming**.
- Micro-delays are clamped ($0.001\text{s} \le \text{delay} \le 0.15\text{s}$) to keep the UI interactive and responsive.

### 3. Scenario Mode (Instant Replay for Judges)
When an evaluator clicks **Trigger Scenario** on the frontend, the system doesn't wait for target events to appear in the 30-day timeline. It dynamically pulls only the pre-filtered subset for that scenario:
- **`temporal_link`**: Pre-filtered 266 events (Citibank India collusion benchmark).
- **`structuring`**: Pre-filtered 99 events (Hawala sub-50k smurfing bursts).
- **`circular_transfer`**: Pre-filtered 15 events (Saradha closed multi-hop laundering ring).
- **`clean_busy`**: Pre-filtered 255 events (High-volume legitimate traffic demonstrating zero false alarms).

### 4. Zero-Config Fallback Mode (In-Memory Queue)
If an evaluator or developer clones the repository on a machine **without** Apache Kafka installed:
- `StreamProducer` attempts connection to Kafka broker (`localhost:9092`).
- If connection fails or `kafka-python` is omitted, it automatically falls back to [`data/event_bus.py`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/data/event_bus.py).
- `StreamConsumer` consumes from thread-safe `queue.Queue` buffers with identical semantics, guaranteeing that the platform runs out of the box with zero setup hurdles.

---

## 5. How Consumer Routes Events to Models & Graph

The consumer loop in [`data/kafka_consumer.py`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/data/kafka_consumer.py) is the central heartbeat of TemporalShield:

1. **Access Event Arrival:**
   - Adds `Employee ➔ Account` edge to [`TemporalGraph`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/graph/temporal_graph.py) with exponential decay weight.
   - Evaluates `RoleAnomalyDetector` (Autoencoder) on employee shift, role permissions, and records queried.
   - Graph cross-checks whether any transactions left this account within the last 4 minutes.
2. **Transaction Event Arrival:**
   - Adds `Account ➔ Counterparty` edge to [`TemporalGraph`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/graph/temporal_graph.py).
   - Feeds the account's recent transaction window into `StructuringDetector` (LSTM + Attention).
   - Feeds profile baseline into `ProfileMismatchDetector` (XGBoost + SHAP).
   - Graph cross-checks whether an employee accessed this account within the preceding 4 minutes.
   - Runs `CircularFlowDetector` (DFS cycle search + Node2Vec similarity).
3. **Alert Generation:**
   - Feeds all active scores into [`RiskAggregator`](file:///c:/Users/purva/VS_Codes/Hackathons/TemporalShield/graph/risk_aggregator.py).
   - If composite risk exceeds threshold, creates an `AlertDetail` dossier and pushes to the alert queue.

---

## 6. How to Run and Control the Stream

### Option A: Running with In-Memory Mode (Default, Zero Setup)
The FastAPI backend starts the consumer automatically on boot:
```bash
uvicorn api.main:app --reload --port 8000
```
Then trigger streaming via cURL or Swagger:
```bash
curl -X POST "http://localhost:8000/scenario" \
     -H "Content-Type: application/json" \
     -d '{"scenario": "temporal_link", "speed_multiplier": 500.0}'
```

### Option B: Running with a Real Apache Kafka Broker
1. Start your Kafka broker (e.g., via Docker):
   ```bash
   docker run -d --name broker -p 9092:9092 apache/kafka:latest
   ```
2. Set the environment variable:
   ```bash
   export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
   ```
3. Initialize the producer with `use_kafka=True`:
   ```python
   from data.kafka_producer import StreamProducer
   producer = StreamProducer(use_kafka=True, bootstrap_servers="localhost:9092")
   producer.start_stream(scenario="temporal_link")
   ```
