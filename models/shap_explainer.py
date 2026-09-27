"""
TemporalShield — Central SHAP Evidence Assembler
Combines signals from all detection layers (Isolation Forest, Autoencoder, LSTM, XGBoost)
into a unified, investigator-ready Alert Evidence Dossier mapped to historical Indian fraud incidents.
"""

import json
from datetime import datetime

# Documented Indian banking fraud incident library
INCIDENT_CATALOG = {
    "temporal_link": {
        "incident_name": "Citibank India Wealth Management Scam (2010)",
        "quantum": "INR 400 Crore",
        "description": "Relationship manager accessed dormant and HNW accounts to forge authorizations, moving funds out via unauthorized transfers right after internal logons.",
        "pmla_sections": "PMLA 2002 Section 3 (Offence of money laundering), IPC 420 (Cheating)",
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
        "pmla_sections": "Prevention of Corruption Act 1988, IPC 409 (Criminal breach of trust by banker)",
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


class SHAPEvidenceAssembler:
    """Assembles all model inferences into an investigator-ready TemporalShield Dossier."""

    def __init__(self, ensemble_weights=None):
        self.weights = ensemble_weights or {
            "isolation_forest": 0.40,
            "xgboost": 0.30,
            "lstm": 0.20,
            "autoencoder": 0.10
        }

    def assemble(self, access_result=None, txn_result=None, lstm_result=None, xgb_result=None, raw_events=None):
        """
        Combines detection outputs into a unified structured evidence package.
        """
        fired_modules = []
        scores = {}
        all_drivers = []
        shap_factors = []

        # 1. Isolation Forest (Temporal Link)
        if access_result:
            score = access_result.get("anomaly_score", 0.0)
            scores["isolation_forest"] = score
            if access_result.get("is_suspicious"):
                fired_modules.append("Isolation Forest (Temporal Link)")
                for d in access_result.get("drivers", []):
                    all_drivers.append({"module": "Temporal Link", "evidence": d})

        # 2. Role Autoencoder
        if access_result and "reconstruction_error" in access_result:
            ae_score = min(1.0, access_result.get("reconstruction_error", 0.0) / max(0.01, access_result.get("threshold", 0.05)))
            scores["autoencoder"] = ae_score
            if access_result.get("is_anomalous"):
                fired_modules.append("Role Autoencoder (Privilege Abuse)")
                for uf in access_result.get("top_unusual_features", []):
                    all_drivers.append({"module": "Role Autoencoder", "evidence": f"Unusual baseline: {uf}"})

        # 3. LSTM (Structuring)
        if lstm_result:
            l_score = lstm_result.get("confidence_score", 0.0)
            scores["lstm"] = l_score
            if lstm_result.get("structuring_detected"):
                fired_modules.append("LSTM + Attention (Structuring)")
                all_drivers.append({"module": "LSTM Structuring", "evidence": lstm_result.get("evidence", "Structuring detected")})

        # 4. XGBoost (Profile Mismatch)
        if xgb_result:
            x_score = xgb_result.get("mismatch_score", 0.0)
            scores["xgboost"] = x_score
            if xgb_result.get("is_anomalous"):
                fired_modules.append("XGBoost + SHAP (Profile Mismatch)")
                for tr in xgb_result.get("top_reasons", []):
                    shap_factors.append({
                        "feature": tr.get("feature"),
                        "contribution": tr.get("shap_contribution"),
                        "explanation": tr.get("explanation")
                    })
                    all_drivers.append({"module": "XGBoost SHAP", "evidence": f"{tr.get('feature')}: {tr.get('explanation')}"})

        # Multi-layer Risk Score Aggregation
        total_weight = sum(self.weights.values())
        composite_score = sum(scores.get(m, 0.0) * w for m, w in self.weights.items()) / total_weight

        # Severity classification
        if composite_score >= 0.70:
            severity = "CRITICAL"
            recommended_action = "IMMEDIATE HOLD: Automated temporary hold on beneficiary account and notify Anti-Money Laundering (AML) Compliance desk."
        elif composite_score >= 0.45:
            severity = "HIGH"
            recommended_action = "PRIORITY REVIEW: Route to Level-2 Fraud Investigator within 30 minutes; request telephonic step-up verification."
        elif composite_score >= 0.25:
            severity = "MEDIUM"
            recommended_action = "WATCHLIST MONITOR: Flag account for enhanced monitoring over next 72 hours; enforce lower OTP limits."
        else:
            severity = "LOW"
            recommended_action = "LOGGED: Normal event logged to audit stream with no active intervention required."

        # Incident mapping
        incident_key = "temporal_link"
        if "LSTM + Attention (Structuring)" in fired_modules:
            incident_key = "structuring"
        elif "XGBoost + SHAP (Profile Mismatch)" in fired_modules:
            incident_key = "profile_mismatch"
        elif "Role Autoencoder (Privilege Abuse)" in fired_modules:
            incident_key = "role_anomaly"

        historical_case = INCIDENT_CATALOG.get(incident_key, INCIDENT_CATALOG["temporal_link"])

        # Timeline reconstruction
        timeline = []
        if raw_events:
            acc_ev = raw_events.get("access_event")
            txn_ev = raw_events.get("transaction_event")
            if acc_ev:
                timeline.append({
                    "timestamp": acc_ev.get("timestamp"),
                    "actor": f"Employee {acc_ev.get('employee_id')} ({acc_ev.get('role')})",
                    "action": f"Accessed {acc_ev.get('account_id')} ({acc_ev.get('account_type')}) at branch {acc_ev.get('branch_code')}",
                    "significance": "Initial insider access event"
                })
            if txn_ev:
                timeline.append({
                    "timestamp": txn_ev.get("timestamp"),
                    "actor": f"Account {txn_ev.get('account_id')}",
                    "action": f"{txn_ev.get('type', 'TRANSFER')} of INR {float(txn_ev.get('amount', 0)):,.2f} to {txn_ev.get('counterparty_id', 'External')}",
                    "significance": "Fund diversion event"
                })

        dossier = {
            "dossier_id": f"TS-ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "composite_risk_score": round(composite_score * 100.0, 1),
            "severity": severity,
            "recommended_action": recommended_action,
            "fired_modules": fired_modules,
            "module_scores": {k: round(v, 4) for k, v in scores.items()},
            "key_drivers": all_drivers,
            "shap_attributions": shap_factors,
            "timeline": timeline,
            "historical_incident_benchmark": historical_case
        }
        return dossier


# Quick test
if __name__ == "__main__":
    assembler = SHAPEvidenceAssembler()
    mock_if = {"is_suspicious": True, "anomaly_score": 0.72, "drivers": ["Tight temporal link: 3.5 mins gap"]}
    mock_xgb = {"is_anomalous": True, "mismatch_score": 0.81, "top_reasons": [{"feature": "errorBalanceOrig", "shap_contribution": 0.12, "explanation": "Balance vanished"}]}
    mock_lstm = {"structuring_detected": True, "confidence_score": 0.89, "evidence": "4 sub-50k transfers totaling INR 180,000"}

    res = assembler.assemble(access_result=mock_if, xgb_result=mock_xgb, lstm_result=mock_lstm)
    print(json.dumps(res, indent=2))
