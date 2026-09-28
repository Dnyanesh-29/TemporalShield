"""
TemporalShield / NexusTrace — Shared Event Bus
Provides thread-safe in-memory queues when Apache Kafka is unavailable (fallback mode).
Allows KafkaProducer and KafkaConsumer to run seamlessly out of the box with zero external broker setup.
"""

import queue
from typing import Dict, Any, Optional

# Global thread-safe queues for in-memory streaming
access_events_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=100000)
transactions_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=100000)
alerts_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=50000)


def get_queue(topic: str) -> "queue.Queue[Dict[str, Any]]":
    """Returns the in-memory queue corresponding to a Kafka topic."""
    if topic in ("access_events", "access_logs"):
        return access_events_queue
    elif topic in ("transactions", "synthetic_transactions"):
        return transactions_queue
    elif topic in ("alerts", "alert_events"):
        return alerts_queue
    else:
        return transactions_queue


def clear_all_queues():
    """Flushes all queued events."""
    for q in (access_events_queue, transactions_queue, alerts_queue):
        with q.mutex:
            q.queue.clear()
