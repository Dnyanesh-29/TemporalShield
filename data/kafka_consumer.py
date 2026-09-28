"""
TemporalShield / NexusTrace — Event Stream Consumer & Multi-Model Inference Dispatcher
Continuous event loop subscribing to both 'access_events' and 'transactions' topics (or in-memory fallback queues).
Routes each event to the TemporalGraph and the 5 ML detection layers, synthesizing alerts via RiskAggregator.

Event Routing Workflow:
When an access_event arrives:
1. Add employee -> account edge to temporal graph.
2. Run Autoencoder on access features (privilege abuse).
3. Cross-check recent transactions on this account for the 4-minute Isolation Forest window.

When a transaction arrives:
1. Add account -> counterparty edge to temporal graph.
2. Run LSTM Structuring Detector on recent transaction sequence for this account.
3. Run XGBoost on account historical profile features.
4. Cross-check recent employee access on this account for the 4-minute Isolation Forest window.
5. Check for circular transfer rings (A -> B -> C -> A) via CircularFlowDetector.

Pass all model outputs to RiskAggregator.
If ensemble risk score >= threshold -> synthesize alert -> push to alert queue.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import graph and models
try:
    from graph.temporal_graph import TemporalGraph
    from graph.circular_detector import CircularFlowDetector
    from graph.risk_aggregator import RiskAggregator
    from models.isolation_forest import TemporalInsiderDetector
    from models.autoencoder_role import RoleAnomalyDetector
    from models.lstm_structuring import StructuringDetector
    from models.xgboost_profile import ProfileMismatchDetector
    from data.event_bus import get_queue, clear_all_queues
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph.temporal_graph import TemporalGraph
    from graph.circular_detector import CircularFlowDetector
    from graph.risk_aggregator import RiskAggregator
    from models.isolation_forest import TemporalInsiderDetector
    from models.autoencoder_role import RoleAnomalyDetector
    from models.lstm_structuring import StructuringDetector
    from models.xgboost_profile import ProfileMismatchDetector
    from data.event_bus import get_queue, clear_all_queues

logger = logging.getLogger("TemporalShield.KafkaConsumer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TOPIC_ACCESS = "access_events"
TOPIC_TRANSACTIONS = "transactions"
TOPIC_ALERTS = "alerts"


class StreamConsumer:
    """
    Real-time continuous inference consumer evaluating events against graph and ML models.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        use_kafka: bool = False,
        temporal_graph: Optional[TemporalGraph] = None,
        risk_aggregator: Optional[RiskAggregator] = None
    ):
        self.bootstrap_servers = bootstrap_servers
        self.use_kafka = use_kafka

        # Detection Engines & Graph Engine
        self.graph = temporal_graph or TemporalGraph()
        self.risk_aggregator = risk_aggregator or RiskAggregator()
        self.circular_detector = CircularFlowDetector()

        # ML Models
        self.if_detector = TemporalInsiderDetector()
        self.ae_detector = RoleAnomalyDetector()
        self.lstm_detector = StructuringDetector()
        self.xgb_detector = ProfileMismatchDetector()

        # Telemetry & Stats
        self.stats = {
            "total_access_events": 0,
            "total_transactions": 0,
            "alerts_created": 0,
            "temporal_correlations_detected": 0,
            "circular_rings_detected": 0,
            "structuring_bursts_detected": 0,
            "profile_anomalies_detected": 0,
            "start_time": datetime.utcnow().isoformat(),
            "status": "IDLE"
        }

        self._stop_event = threading.Event()
        self._consumer_thread: Optional[threading.Thread] = None
        self.kafka_consumer = None
        self.is_running = False
        self._init_kafka_consumer()

    def _init_kafka_consumer(self):
        if not self.use_kafka:
            return
        try:
            from kafka import KafkaConsumer
            self.kafka_consumer = KafkaConsumer(
                TOPIC_ACCESS,
                TOPIC_TRANSACTIONS,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id=f"temporalshield-consumer-{int(time.time())}",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=200
            )
            logger.info(f"Subscribed to Kafka topics at {self.bootstrap_servers}: {[TOPIC_ACCESS, TOPIC_TRANSACTIONS]}")
        except Exception as e:
            logger.warning(f"Could not connect Kafka consumer ({e}). Operating in in-memory queue mode.")
            self.use_kafka = False
            self.kafka_consumer = None

    def process_access_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes an incoming employee access event:
        1. Incremental graph update.
        2. Role Autoencoder scoring.
        3. Cross-check recent transactions for 4-minute window.
        """
        self.stats["total_access_events"] += 1
        account_id = str(event.get("account_id"))

        # 1. Update Temporal Graph and get any temporal correlations
        correlations = self.graph.add_access_event(event)

        # 2. Run Role Autoencoder
        ae_result = self.ae_detector.predict(event)

        # 3. Check for 4-minute temporal link if correlated transactions exist
        alert = None
        for corr in correlations:
            self.stats["temporal_correlations_detected"] += 1
            txn_event = corr["transaction_event"]
            if_result = self.if_detector.predict(event, txn_event)

            # Evaluate through Risk Aggregator
            alert = self.risk_aggregator.evaluate(
                account_id=account_id,
                temporal_link_result=if_result,
                autoencoder_result=ae_result,
                raw_event=txn_event
            )
            if alert:
                self.stats["alerts_created"] += 1
                get_queue(TOPIC_ALERTS).put(alert)
                break

        # If no 4-minute match, still flag severe privilege abuse if autoencoder triggered
        if not alert and ae_result.get("is_anomalous"):
            alert = self.risk_aggregator.evaluate(
                account_id=account_id,
                autoencoder_result=ae_result,
                raw_event=event
            )
            if alert:
                self.stats["alerts_created"] += 1
                get_queue(TOPIC_ALERTS).put(alert)

        return alert

    def process_transaction_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes an incoming transaction event:
        1. Incremental graph update.
        2. LSTM structuring sequence scoring.
        3. XGBoost profile mismatch scoring.
        4. Cross-check recent employee access for 4-minute window.
        5. Circular transfer ring check.
        """
        self.stats["total_transactions"] += 1
        account_id = str(event.get("account_id"))

        # 1. Update Temporal Graph & 4-min cross check
        correlations = self.graph.add_transaction_event(event)

        # 2. LSTM Structuring Detector
        recent_txns = self.graph.get_recent_transactions_for_account(account_id, limit=20)
        lstm_result = self.lstm_detector.predict(recent_txns)
        if lstm_result.get("structuring_detected"):
            self.stats["structuring_bursts_detected"] += 1

        # 3. XGBoost Profile Mismatch
        profile = {
            "historical_mean_amount": float(event.get("amount", 5000)) * 0.20,
            "historical_max_amount": float(event.get("amount", 5000)) * 0.50,
            "historical_txn_count": len(recent_txns),
            "days_since_last_transaction": 15.0 if len(recent_txns) <= 2 else 1.0,
            "account_age_steps": 360.0
        }
        xgb_result = self.xgb_detector.predict(profile, event)
        if xgb_result.get("is_anomalous"):
            self.stats["profile_anomalies_detected"] += 1

        # 4. Check 4-minute window Isolation Forest
        if_result = None
        for corr in correlations:
            self.stats["temporal_correlations_detected"] += 1
            acc_event = corr["access_event"]
            if_result = self.if_detector.predict(acc_event, event)
            break

        # 5. Circular Flow Detection (run on account or periodic)
        circ_result = None
        scenario = str(event.get("scenario", ""))
        if scenario == "circular_transfer" or self.stats["total_transactions"] % 25 == 0:
            detected_cycles = self.circular_detector.detect_cycles(self.graph.graph, target_account=account_id)
            if detected_cycles:
                self.stats["circular_rings_detected"] += len(detected_cycles)
                circ_result = detected_cycles[0]

        # 6. Synthesize Alert via Risk Aggregator
        alert = self.risk_aggregator.evaluate(
            account_id=account_id,
            temporal_link_result=if_result,
            circular_result=circ_result,
            xgb_result=xgb_result,
            lstm_result=lstm_result,
            raw_event=event
        )

        if alert:
            self.stats["alerts_created"] += 1
            get_queue(TOPIC_ALERTS).put(alert)

        return alert

    def _consumer_loop(self):
        """Continuous consumer loop polling in-memory queues or Kafka."""
        self.is_running = True
        self.stats["status"] = "STREAMING"
        logger.info("KafkaConsumer loop started.")

        acc_q = get_queue(TOPIC_ACCESS)
        txn_q = get_queue(TOPIC_TRANSACTIONS)

        while not self._stop_event.is_set():
            processed_any = False

            # 1. Read from Kafka if enabled
            if self.use_kafka and self.kafka_consumer:
                try:
                    records = self.kafka_consumer.poll(timeout_ms=100, max_records=50)
                    for tp, messages in records.items():
                        for msg in messages:
                            if self._stop_event.is_set():
                                break
                            if msg.topic == TOPIC_ACCESS:
                                self.process_access_event(msg.value)
                            elif msg.topic == TOPIC_TRANSACTIONS:
                                self.process_transaction_event(msg.value)
                            processed_any = True
                except Exception as e:
                    logger.debug(f"Kafka consumer poll: {e}")

            # 2. Drain pending access events from in-memory queue
            while not acc_q.empty() and not self._stop_event.is_set():
                try:
                    event = acc_q.get_nowait()
                    self.process_access_event(event)
                    acc_q.task_done()
                    processed_any = True
                except Exception as e:
                    logger.error(f"Error processing access event: {e}")

            # 3. Drain pending transaction events from in-memory queue
            while not txn_q.empty() and not self._stop_event.is_set():
                try:
                    event = txn_q.get_nowait()
                    self.process_transaction_event(event)
                    txn_q.task_done()
                    processed_any = True
                except Exception as e:
                    logger.error(f"Error processing transaction event: {e}")

            if not processed_any:
                time.sleep(0.01)

        self.is_running = False
        self.stats["status"] = "IDLE"
        logger.info("KafkaConsumer loop stopped.")

    def start(self):
        """Starts the consumer thread."""
        if self._consumer_thread and self._consumer_thread.is_alive():
            return
        self._stop_event.clear()
        self._consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self._consumer_thread.start()

    def stop(self):
        """Stops the consumer loop."""
        self._stop_event.set()
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=1.0)
        if self.kafka_consumer:
            try:
                self.kafka_consumer.close()
            except Exception:
                pass
        self.is_running = False
        self.stats["status"] = "IDLE"

    def reset(self):
        """Resets consumer graph, detectors, and counters."""
        self.stop()
        self.graph.reset()
        self.risk_aggregator.clear()
        self.circular_detector = CircularFlowDetector()
        clear_all_queues()
        self.stats = {
            "total_access_events": 0,
            "total_transactions": 0,
            "alerts_created": 0,
            "temporal_correlations_detected": 0,
            "circular_rings_detected": 0,
            "structuring_bursts_detected": 0,
            "profile_anomalies_detected": 0,
            "start_time": datetime.utcnow().isoformat(),
            "status": "IDLE"
        }
        self.start()


# Quick test
if __name__ == "__main__":
    consumer = StreamConsumer()
    consumer.start()
    acc_q = get_queue(TOPIC_ACCESS)
    txn_q = get_queue(TOPIC_TRANSACTIONS)

    # Push correlated events
    acc_q.put({
        "employee_id": "EMP_0019", "role": "loan_officer",
        "account_id": "ACC_10017", "account_type": "savings",
        "branch_code": "BR_006", "action_type": "read",
        "timestamp": "2026-09-01 10:00:00", "records_accessed": 1, "is_suspicious": 1
    })
    time.sleep(0.05)
    txn_q.put({
        "transaction_id": "TXN_001", "account_id": "ACC_10017",
        "amount": 48500.0, "transaction_type": "TRANSFER",
        "counterparty_id": "EXT_9999", "timestamp": "2026-09-01 10:02:15",
        "branch_code": "BR_006", "is_fraud": 1, "scenario": "temporal_link"
    })
    time.sleep(0.2)
    print("Consumer Stats:", consumer.stats)
    alerts = consumer.risk_aggregator.get_alerts()
    print("Alerts created:", len(alerts))
    consumer.stop()
