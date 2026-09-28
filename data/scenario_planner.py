"""
TemporalShield / NexusTrace — Scenario Planner & Demo Controller
API-facing controller allowing the frontend / judges to trigger specific demo scenarios on demand.
Reads pre-filtered scenario subsets from existing CSV files and signals the StreamProducer
to stream them immediately. Does NOT generate any new data at runtime.

Scenario Subsets (Pre-filtered from ground truth):
1. temporal_link: 4-minute insider collusion window (Citibank India Wealth Scam benchmark)
2. structuring: Sub-50,000 Hawala smurfing burst evasion (ED Smurfing Syndicates benchmark)
3. circular_transfer: Closed loop shell account layering (Saradha Financial Scam benchmark)
4. clean_busy: High-velocity benign commercial operations baseline
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    from data.kafka_producer import StreamProducer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.kafka_producer import StreamProducer

logger = logging.getLogger("TemporalShield.ScenarioPlanner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCENARIO_MANIFEST = {
    "temporal_link": {
        "scenario_id": "temporal_link",
        "title": "4-Minute Insider Collusion Link",
        "historical_incident": "Citibank India Wealth Management Scam (2010)",
        "quantum": "INR 400 Crore",
        "pmla_context": "PMLA 2002 Section 3, RBI Master Direction on Fraud Classification",
        "description": "Bank loan officer accesses customer savings account. Within 4 minutes, large outbound fund transfers depart to external beneficiary accounts.",
        "target_model": "Isolation Forest (Temporal Link) + Role Autoencoder",
        "expected_events": 266,
        "recommended_speed": 500.0,
        "detection_window": "4 minutes"
    },
    "structuring": {
        "scenario_id": "structuring",
        "title": "Hawala Structuring / Smurfing Burst",
        "historical_incident": "Hawala Syndicates & ED Smurfing Operations (Ongoing)",
        "quantum": "Sub-INR 50,000 Fragmented Bursts (CTR Evasion)",
        "pmla_context": "PMLA 2002 Rule 3(1)(A) (Cash transaction reporting threshold evasion)",
        "description": "Rapid succession of 3+ transfers between INR 30,000 and INR 49,500 within 2 hours, cumulatively exceeding reporting limits without triggering CTR.",
        "target_model": "LSTM + Attention (Structuring Detector)",
        "expected_events": 99,
        "recommended_speed": 500.0,
        "detection_window": "2 hours"
    },
    "circular_transfer": {
        "scenario_id": "circular_transfer",
        "title": "Layered Circular Transfer Ring",
        "historical_incident": "Saradha Financial Syndicate Fraud (2013)",
        "quantum": "INR 2,500 Crore",
        "pmla_context": "SEBI Collective Investment Scheme Regulations, FIU-IND RFI-08",
        "description": "Multi-hop closed loop transfer (A ➔ B ➔ C ➔ A) across shell accounts with disguised round-tripping to inflate commercial velocity.",
        "target_model": "Circular Flow Detector + Node2Vec Ring Cohesion",
        "expected_events": 15,
        "recommended_speed": 200.0,
        "detection_window": "Real-time Graph Cycle"
    },
    "clean_busy": {
        "scenario_id": "clean_busy",
        "title": "Benign High-Velocity Branch Operations (Clean Baseline)",
        "historical_incident": "Legitimate Peak Banking Operations",
        "quantum": "Normal Commercial Payroll & Merchant Debits",
        "pmla_context": "Standard KYC & AML Audit Validation",
        "description": "High-volume branch transactions executed by authorized branch staff during standard hours. Demonstrates zero false positives.",
        "target_model": "All ML Engines (Low Anomaly Baseline)",
        "expected_events": 255,
        "recommended_speed": 500.0,
        "detection_window": "Standard Audit"
    }
}


class ScenarioPlanner:
    """
    Controller that coordinates on-demand scenario streaming via StreamProducer.
    """

    def __init__(self, producer: Optional[StreamProducer] = None):
        self.producer = producer or StreamProducer()

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Returns catalogue of all supported demonstration scenarios."""
        return list(SCENARIO_MANIFEST.values())

    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific scenario."""
        return SCENARIO_MANIFEST.get(scenario_id.lower())

    def trigger_scenario(self, scenario_id: str, speed_multiplier: Optional[float] = None) -> Dict[str, Any]:
        """
        Commands the StreamProducer to immediately filter and stream the specified scenario.
        """
        sid = scenario_id.lower()
        if sid not in SCENARIO_MANIFEST and sid != "all":
            raise ValueError(f"Unknown scenario '{scenario_id}'. Valid scenarios: {list(SCENARIO_MANIFEST.keys())}")

        scenario_info = SCENARIO_MANIFEST.get(sid, {
            "scenario_id": "all",
            "title": "Full Timeline Stream",
            "recommended_speed": 500.0
        })

        speed = speed_multiplier or scenario_info.get("recommended_speed", 500.0)

        logger.info(f"Triggering demo scenario '{sid}' at {speed}x speed multiplier...")
        self.producer.start_stream(scenario=sid, speed_multiplier=speed, async_run=True)

        return {
            "status": "STREAMING",
            "scenario_id": sid,
            "title": scenario_info.get("title"),
            "historical_incident": scenario_info.get("historical_incident"),
            "speed_multiplier": speed,
            "message": f"Scenario '{sid}' triggered successfully. Events streaming to Kafka/queue."
        }

    def stop(self) -> Dict[str, Any]:
        """Stops the active scenario stream."""
        self.producer.stop_stream()
        return {"status": "STOPPED", "message": "Event stream halted."}

    def get_status(self) -> Dict[str, Any]:
        """Returns the producer's current streaming state."""
        return {
            "is_streaming": self.producer.is_running,
            "current_scenario": self.producer.current_scenario,
            "events_published": self.producer.events_published_count,
            "speed_multiplier": self.producer.speed_multiplier
        }


# Quick test
if __name__ == "__main__":
    planner = ScenarioPlanner()
    print("Available scenarios:", [s["scenario_id"] for s in planner.list_scenarios()])
    status = planner.trigger_scenario("circular_transfer", speed_multiplier=200.0)
    print("Trigger result:", status)
