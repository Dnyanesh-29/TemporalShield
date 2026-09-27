"""
TemporalShield — API Evidence Builder
FastAPI-facing wrapper around the SHAP evidence assembler.
Exposes endpoints and helper functions for frontend alert investigation views.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import detector modules
try:
    from models.isolation_forest import TemporalInsiderDetector
    from models.autoencoder_role import RoleAnomalyDetector
    from models.lstm_structuring import StructuringDetector
    from models.xgboost_profile import ProfileMismatchDetector
    from models.shap_explainer import SHAPEvidenceAssembler
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.isolation_forest import TemporalInsiderDetector
    from models.autoencoder_role import RoleAnomalyDetector
    from models.lstm_structuring import StructuringDetector
    from models.xgboost_profile import ProfileMismatchDetector
    from models.shap_explainer import SHAPEvidenceAssembler


class EvidenceBuilderService:
    """Singleton service powering TemporalShield API evidence endpoints."""

    def __init__(self):
        self.if_detector = TemporalInsiderDetector()
        self.ae_detector = RoleAnomalyDetector()
        self.lstm_detector = StructuringDetector()
        self.xgb_detector = ProfileMismatchDetector()
        self.assembler = SHAPEvidenceAssembler()

    def evaluate_transaction_event(
        self,
        account_id: str,
        transaction_event: Dict[str, Any],
        recent_transactions: Optional[List[Dict[str, Any]]] = None,
        account_profile: Optional[Dict[str, Any]] = None,
        recent_employee_access: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs full multi-layer inference across all 4 detection engines and produces
        the unified investigation evidence dossier.
        """
        if_result = None
        ae_result = None
        lstm_result = None
        xgb_result = None

        # 1. Temporal Link & Role Anomaly (if employee access event provided)
        if recent_employee_access:
            if_result = self.if_detector.predict(recent_employee_access, transaction_event)
            ae_result = self.ae_detector.predict(recent_employee_access)
            # Combine drivers into access result
            if_result["reconstruction_error"] = ae_result.get("reconstruction_error")
            if_result["is_anomalous"] = ae_result.get("is_anomalous")
            if_result["top_unusual_features"] = ae_result.get("top_unusual_features")

        # 2. LSTM Structuring Detector
        txns_to_score = recent_transactions or [transaction_event]
        lstm_result = self.lstm_detector.predict(txns_to_score)

        # 3. XGBoost Customer Profile Mismatch Detector
        profile = account_profile or {
            "historical_mean_amount": float(transaction_event.get("amount", 5000)) * 0.15,
            "historical_max_amount": float(transaction_event.get("amount", 5000)) * 0.40,
            "historical_txn_count": 24,
            "days_since_last_transaction": 45.0,
            "account_age_steps": 720.0
        }
        xgb_result = self.xgb_detector.predict(profile, transaction_event)

        # 4. Assemble Evidence Dossier
        raw_events = {
            "access_event": recent_employee_access,
            "transaction_event": transaction_event
        }

        dossier = self.assembler.assemble(
            access_result=if_result,
            lstm_result=lstm_result,
            xgb_result=xgb_result,
            raw_events=raw_events
        )

        dossier["account_id"] = account_id
        dossier["transaction_id"] = transaction_event.get("transaction_id", f"TXN_{datetime.now().strftime('%f')}")

        return dossier

    def get_alert_dossier(self, alert_id: str) -> Dict[str, Any]:
        """
        Retrieves formatted dossier for the TemporalShield mock frontend alert view.
        """
        # Demo synthesized alert dossier matching real Jan Dhan / Hawala / Citibank scenarios
        return {
            "alert_id": alert_id,
            "account_id": "ACC_10017",
            "timestamp": datetime.now().isoformat(),
            "composite_risk_score": 86.4,
            "severity": "CRITICAL",
            "status": "UNDER_INVESTIGATION",
            "alert_title": "4-Minute Insider Link & Sub-Threshold Structuring Burst",
            "summary": "Employee EMP_0042 (Loan Officer) accessed customer savings account ACC_10017 at 14:10:00. Within 4 minutes, 4 sequential transfers of INR ~45,000 each departed to external accounts, totaling INR 182,700.",
            "modules": {
                "isolation_forest": {
                    "score": 0.88,
                    "flagged": True,
                    "insight": "Time gap of 3.8 minutes falls within the critical 4-minute insider collusion window."
                },
                "lstm_structuring": {
                    "score": 0.94,
                    "flagged": True,
                    "insight": "4 transactions below INR 50,000 executed within 75 minutes. Combined total INR 182,700 bypasses CTR."
                },
                "xgboost_profile": {
                    "score": 0.78,
                    "flagged": True,
                    "insight": "Account was inactive for 54 days before today's sudden burst."
                },
                "role_autoencoder": {
                    "score": 0.65,
                    "flagged": True,
                    "insight": "Loan Officer accessed a personal savings account outside standard lending workflow."
                }
            },
            "top_shap_features": [
                {"feature": "time_delta_minutes", "value": 3.8, "importance": "+0.34", "reason": "Immediate fund drainage after employee access"},
                {"feature": "sub_50k_burst_count", "value": 4, "importance": "+0.29", "reason": "PMLA cash reporting threshold evasion"},
                {"feature": "days_since_last_transaction", "value": 54, "importance": "+0.18", "reason": "Dormant account reactivation"}
            ],
            "historical_incident_parallel": {
                "name": "Citibank India Wealth Scam & Hawala Structuring",
                "quantum": "INR 400 Crore",
                "parallel_description": "Colluding staff accessed accounts followed immediately by structured transfers to bypass AML monitoring."
            },
            "recommended_action": "IMMEDIATE HOLD on outbound transfers from ACC_10017 and initiate formal inquiry into EMP_0042 access credentials."
        }


# Quick test
if __name__ == "__main__":
    service = EvidenceBuilderService()
    sample_acc = {
        "employee_id": "EMP_0019", "role": "loan_officer",
        "account_id": "ACC_10017", "account_type": "savings",
        "timestamp": "2026-09-01 10:00:00"
    }
    sample_tx = {
        "transaction_id": "TXN_99182",
        "account_id": "ACC_10017", "amount": 48500.0,
        "type": "TRANSFER", "counterparty_id": "EXT_9999",
        "timestamp": "2026-09-01 10:03:45"
    }
    dossier = service.evaluate_transaction_event("ACC_10017", sample_tx, recent_employee_access=sample_acc)
    print(json.dumps(dossier, indent=2))
