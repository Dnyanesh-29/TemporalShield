"""
TemporalShield / NexusTrace — Central Risk Aggregator & Alert Synthesis
Combines signals from all 5 ML & Graph modules:
1. Isolation Forest (Temporal Insider Link — 4-minute window) [Weight: 0.35]
2. Circular Detector (Layered Ring & Node2Vec Cohesion)      [Weight: 0.25]
3. XGBoost + TreeSHAP (Profile Mismatch & Dormant Abuse)     [Weight: 0.20]
4. LSTM + Attention (Sub-Threshold Structuring Burst)         [Weight: 0.15]
5. Role Autoencoder (Employee Privilege Deviation)           [Weight: 0.05]

Synthesizes multi-module scores into an overarching case-level priority score (0-100),
assigns standard severity tiers (CRITICAL / HIGH / MEDIUM / LOW),
and constructs the investigation alert dossier mapped to Indian regulatory benchmarks.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from models.shap_explainer import INCIDENT_CATALOG
except ImportError:
    INCIDENT_CATALOG = {
        "temporal_link": {
            "incident_name": "Citibank India Wealth Management Scam (2010)",
            "quantum": "INR 400 Crore",
            "description": "Relationship manager accessed dormant and HNW accounts to forge authorizations, moving funds out via unauthorized transfers right after internal logons.",
            "pmla_sections": "PMLA 2002 Section 3, IPC 420",
            "regulatory_trigger": "RBI Master Direction on Fraud Classification (RBI/2016-17/13)"
        },
        "circular_transfer": {
            "incident_name": "Saradha Financial Syndicate Fraud (2013)",
            "quantum": "INR 2,500 Crore",
            "description": "Layered fund diversions across 300+ shell accounts designed to obscure original source and create fictitious commercial velocity.",
            "pmla_sections": "SEBI Collective Investment Scheme Regulations, PMLA Section 4",
            "regulatory_trigger": "FIU-IND Red Flag Indicator RFI-08 (Layered circular flows)"
        },
        "structuring": {
            "incident_name": "Hawala Syndicates & ED Smurfing Operations (Ongoing)",
            "quantum": "Estimated INR 10,000+ Crore annually",
            "description": "Criminal syndicates fragment large proceeds into multiple sub-INR 50,000 deposits/transfers across mule networks to bypass mandatory Cash Transaction Reports (CTR).",
            "pmla_sections": "PMLA 2002 Rule 3(1)(A) (Cash transaction threshold evasions)",
            "regulatory_trigger": "RBI CTR Reporting Directive (Mandatory reporting > INR 50,000)"
        },
        "role_anomaly": {
            "incident_name": "Punjab National Bank (PNB) Brady House Fraud (2018)",
            "quantum": "INR 11,400 Crore",
            "description": "Bank officials issued unauthorized Letters of Undertaking (LoUs) via SWIFT terminals during odd hours without matching Core Banking System (CBS) journal entries.",
            "pmla_sections": "Prevention of Corruption Act 1988, IPC 409",
            "regulatory_trigger": "RBI Directive on SWIFT-CBS Integration & Role Separation"
        },
        "profile_mismatch": {
            "incident_name": "Jan Dhan Dormant Account Hijacking (2016, Post-Demonetization)",
            "quantum": "INR 42,000+ Crore anomalous surges",
            "description": "Dormant, low-volume savings accounts suddenly processing massive electronic credits and rapid cash withdrawals within days of cash currency replacement.",
            "pmla_sections": "Benami Transactions (Prohibition) Amendment Act 2016, PMLA Section 12",
            "regulatory_trigger": "Income Tax Department 'Operation Clean Money' Guidelines"
        }
    }

logger = logging.getLogger("TemporalShield.RiskAggregator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_WEIGHTS = {
    "temporal_link": 0.35,      # Core headline capability
    "circular_transfer": 0.25,  # Graph cycle detection
    "xgboost_profile": 0.20,   # Baseline deviation & SHAP
    "lstm_structuring": 0.15,  # Hawala smurfing
    "role_autoencoder": 0.05    # Staff privilege misuse
}


class RiskAggregator:
    """
    Ensemble evaluator aggregating inference signals across all 5 detection engines.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, alert_threshold: float = 0.45):
        self.weights = weights or DEFAULT_WEIGHTS
        self.alert_threshold = alert_threshold
        self.alert_history: List[Dict[str, Any]] = []

    def evaluate(
        self,
        account_id: str,
        temporal_link_result: Optional[Dict[str, Any]] = None,
        circular_result: Optional[Dict[str, Any]] = None,
        xgb_result: Optional[Dict[str, Any]] = None,
        lstm_result: Optional[Dict[str, Any]] = None,
        autoencoder_result: Optional[Dict[str, Any]] = None,
        raw_event: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates ensemble risk score and synthesizes alert object if threshold is exceeded.
        """
        scores = {}
        active_weights = {}
        fired_modules = []
        evidence_points = []
        shap_factors = []

        # 1. Temporal Link (Isolation Forest)
        if temporal_link_result:
            score = float(temporal_link_result.get("anomaly_score", 0.0))
            scores["temporal_link"] = score
            active_weights["temporal_link"] = self.weights["temporal_link"]
            if temporal_link_result.get("is_suspicious") or temporal_link_result.get("time_gap_minutes", 999) <= 4.0:
                fired_modules.append("Isolation Forest (4-Minute Temporal Link)")
                gap = temporal_link_result.get("time_gap_minutes", 0.0)
                evidence_points.append(f"Funds transferred within {gap:.1f} minutes of internal employee lookup (4-min window trigger).")
                for d in temporal_link_result.get("drivers", []):
                    evidence_points.append(d)

        # 2. Circular Detector (Graph Cycles)
        if circular_result:
            c_score = float(circular_result.get("anomaly_score", 0.85))
            scores["circular_transfer"] = c_score
            active_weights["circular_transfer"] = self.weights["circular_transfer"]
            fired_modules.append("Circular Transfer Detector (Saradha Ring)")
            path = circular_result.get("cycle_path", "")
            q = circular_result.get("total_quantum_inr", 0.0)
            evidence_points.append(f"Closed fund cycle detected: {path} with combined quantum of INR {q:,.2f}.")

        # 3. XGBoost (Profile Mismatch)
        if xgb_result:
            x_score = float(xgb_result.get("mismatch_score", 0.0))
            scores["xgboost_profile"] = x_score
            active_weights["xgboost_profile"] = self.weights["xgboost_profile"]
            if xgb_result.get("is_anomalous"):
                fired_modules.append("XGBoost + SHAP (Profile Deviation)")
                for tr in xgb_result.get("top_reasons", []):
                    shap_factors.append({
                        "feature": tr.get("feature"),
                        "importance": tr.get("shap_contribution", "+0.25"),
                        "reason": tr.get("explanation", "")
                    })
                    evidence_points.append(f"{tr.get('feature')}: {tr.get('explanation')}")

        # 4. LSTM (Structuring)
        if lstm_result:
            l_score = float(lstm_result.get("confidence_score", 0.0))
            scores["lstm_structuring"] = l_score
            active_weights["lstm_structuring"] = self.weights["lstm_structuring"]
            if lstm_result.get("structuring_detected"):
                fired_modules.append("LSTM + Attention (Structuring/Smurfing)")
                agg = lstm_result.get("aggregate_amount", 0.0)
                evidence_points.append(f"Sub-threshold structuring burst detected: aggregate INR {agg:,.2f} split below INR 50,000 CTR limit.")

        # 5. Role Autoencoder
        if autoencoder_result:
            err = float(autoencoder_result.get("reconstruction_error", 0.0))
            thresh = float(autoencoder_result.get("threshold", 0.05))
            ae_score = min(1.0, err / max(0.01, thresh))
            scores["role_autoencoder"] = ae_score
            active_weights["role_autoencoder"] = self.weights["role_autoencoder"]
            if autoencoder_result.get("is_anomalous"):
                fired_modules.append("Role Autoencoder (Privilege Abuse)")
                for uf in autoencoder_result.get("top_unusual_features", []):
                    evidence_points.append(f"Privilege baseline deviation: {uf}")

        # Compute weighted average
        if not active_weights:
            return None

        total_w = sum(active_weights.values())
        raw_composite = sum(scores[k] * active_weights[k] for k in scores) / total_w
        composite_score = round(raw_composite * 100.0, 1)

        # Always trigger alert if explicit fraud flag or high composite score
        is_fraud_flag = bool(raw_event and raw_event.get("is_fraud") == 1)
        if raw_composite < self.alert_threshold and not is_fraud_flag and not fired_modules:
            return None

        # Severity Determination
        if raw_composite >= 0.70 or is_fraud_flag or ("Isolation Forest (4-Minute Temporal Link)" in fired_modules and raw_composite >= 0.50):
            severity = "CRITICAL"
            recommended_action = "IMMEDIATE HOLD on account out-transfers; notify AML Compliance and initiate forensic employee session audit."
        elif raw_composite >= 0.50:
            severity = "HIGH"
            recommended_action = "PRIORITY REVIEW: Dispatch case to Senior Investigator; trigger telephonic step-up customer verification."
        elif raw_composite >= 0.35:
            severity = "MEDIUM"
            recommended_action = "WATCHLIST MONITOR: Flag account for 72-hour step-up OTP authentication and audit trail capture."
        else:
            severity = "LOW"
            recommended_action = "LOGGED: Low anomaly baseline recorded to compliance stream."

        # Incident Categorization
        if "Isolation Forest (4-Minute Temporal Link)" in fired_modules:
            inc_key = "temporal_link"
        elif "Circular Transfer Detector (Saradha Ring)" in fired_modules:
            inc_key = "circular_transfer"
        elif "LSTM + Attention (Structuring/Smurfing)" in fired_modules:
            inc_key = "structuring"
        elif "XGBoost + SHAP (Profile Deviation)" in fired_modules:
            inc_key = "profile_mismatch"
        else:
            inc_key = "role_anomaly"

        incident_info = INCIDENT_CATALOG.get(inc_key, INCIDENT_CATALOG["temporal_link"])
        alert_id = f"ALT_{int(time.time()*1000)}_{len(self.alert_history)+1}"

        alert = {
            "alert_id": alert_id,
            "account_id": account_id,
            "timestamp": (raw_event or {}).get("timestamp", datetime.utcnow().isoformat()),
            "composite_risk_score": composite_score,
            "severity": severity,
            "status": "ACTIVE",
            "alert_title": f"{severity} Anomaly: {' + '.join(fired_modules[:2]) if fired_modules else 'Behavioral Outlier'}",
            "summary": " | ".join(evidence_points[:3]) if evidence_points else "Multi-module anomaly threshold exceeded.",
            "fired_modules": fired_modules,
            "module_scores": {k: round(v, 3) for k, v in scores.items()},
            "top_shap_features": shap_factors[:4],
            "evidence_points": evidence_points[:6],
            "historical_incident_parallel": incident_info,
            "recommended_action": recommended_action,
            "raw_event": raw_event
        }

        self.alert_history.insert(0, alert)
        return alert

    def get_alerts(self, severity: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns sorted alerts by severity and timestamp."""
        sev_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        alerts = self.alert_history
        if severity:
            alerts = [a for a in alerts if a["severity"].upper() == severity.upper()]
        alerts.sort(key=lambda a: (sev_rank.get(a["severity"], 4), -a.get("composite_risk_score", 0)))
        return alerts[:limit]

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Finds an alert by its ID."""
        for a in self.alert_history:
            if a["alert_id"] == alert_id:
                return a
        return None

    def clear(self):
        """Clears all stored alerts."""
        self.alert_history.clear()


# Quick test
if __name__ == "__main__":
    agg = RiskAggregator()
    alt = agg.evaluate(
        account_id="ACC_10017",
        temporal_link_result={"anomaly_score": 0.88, "is_suspicious": True, "time_gap_minutes": 2.5, "drivers": ["2.5m gap"]},
        xgb_result={"mismatch_score": 0.72, "is_anomalous": True, "top_reasons": [{"feature": "amount", "explanation": "Huge surge"}]},
        raw_event={"timestamp": "2026-09-01 10:02:30", "is_fraud": 1}
    )
    print("Generated alert:", alt["alert_id"], alt["severity"], alt["composite_risk_score"])
