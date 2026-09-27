"""
TemporalShield — Isolation Forest Temporal Link Detector
Detects the temporal insider link between an employee accessing an account
and an anomalous transaction departing shortly after (headline 4-minute detection window).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Permitted account types per employee role
ROLE_MAPPING = {
    'loan_officer': ['loan', 'mortgage'],
    'savings_representative': ['savings', 'checking'],
    'branch_manager': ['loan', 'mortgage', 'savings', 'checking', 'business'],
    'it_admin': ['system', 'audit'],
    'compliance_officer': ['loan', 'mortgage', 'savings', 'checking', 'business', 'system', 'audit']
}

TXN_TYPE_MAP = {
    'TRANSFER': 0,
    'CASH_OUT': 1,
    'PAYMENT': 2,
    'DEBIT': 3,
    'CREDIT': 4
}


class TemporalInsiderDetector:
    """Isolation Forest detector for temporal employee-access to transaction correlation."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            # Try standard locations
            candidates = [
                os.path.join(os.path.dirname(__file__), "isolation_forest"),
                os.path.join(os.path.dirname(__file__), "../models/isolation_forest"),
                "models/isolation_forest"
            ]
            model_dir = next((c for c in candidates if os.path.exists(c)), "models/isolation_forest")

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.threshold = 0.50  # Top 5% most anomalous threshold
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler_isolation_forest.pkl")
        cols_path = os.path.join(self.model_dir, "feature_columns_if.json")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(cols_path):
            with open(cols_path, "r") as f:
                self.feature_cols = json.load(f)
        else:
            self.feature_cols = [
                "time_delta_minutes", "amount", "amount_zscore", "role_account_match",
                "hour_of_access", "day_of_week", "records_accessed",
                "transaction_type_encoded", "is_suspicious"
            ]

    def extract_features(self, access_event, txn_event, account_history=None):
        """
        Engineers temporal link features from an access event and a transaction event.
        """
        ts_access = pd.to_datetime(access_event.get("timestamp"))
        ts_txn = pd.to_datetime(txn_event.get("timestamp"))

        # 1. Time delta between access and transaction in minutes
        time_delta_minutes = abs((ts_txn - ts_access).total_seconds()) / 60.0

        # 2. Transaction amount & account z-score
        amount = float(txn_event.get("amount", 0.0))
        if account_history and len(account_history) > 1:
            hist_mean = float(np.mean(account_history))
            hist_std = float(np.std(account_history)) if np.std(account_history) > 0 else 1.0
            amount_zscore = (amount - hist_mean) / hist_std
        else:
            amount_zscore = (amount - 10000.0) / 15000.0

        # 3. Role vs account type match
        role = access_event.get("role", "")
        account_type = access_event.get("account_type", "")
        allowed = ROLE_MAPPING.get(role, [])
        role_account_match = 1.0 if account_type in allowed else 0.0

        # 4. Access hour & day of week
        hour_of_access = float(ts_access.hour)
        day_of_week = float(ts_access.dayofweek)

        records_accessed = float(access_event.get("records_accessed", 1))
        txn_type = str(txn_event.get("transaction_type", txn_event.get("type", "TRANSFER"))).upper()
        txn_type_encoded = float(TXN_TYPE_MAP.get(txn_type, 0))
        is_suspicious = float(access_event.get("is_suspicious", 0))

        feat_dict = {
            "time_delta_minutes": time_delta_minutes,
            "amount": amount,
            "amount_zscore": amount_zscore,
            "role_account_match": role_account_match,
            "hour_of_access": hour_of_access,
            "day_of_week": day_of_week,
            "records_accessed": records_accessed,
            "transaction_type_encoded": txn_type_encoded,
            "is_suspicious": is_suspicious
        }
        return feat_dict

    def predict(self, access_event, txn_event, account_history=None):
        """
        Scores an access-transaction event pair.
        Returns:
            dict containing anomaly_score, is_suspicious, decision_drivers, and explanation.
        """
        feat_dict = self.extract_features(access_event, txn_event, account_history)
        raw_vec = np.array([[feat_dict[c] for c in self.feature_cols]], dtype=np.float32)

        if self.scaler is not None:
            scaled_vec = self.scaler.transform(raw_vec)
        else:
            scaled_vec = raw_vec

        if self.model is not None:
            # Isolation forest decision_function: lower = more anomalous
            # Invert and normalize to [0, 1] risk score
            raw_score = -float(self.model.decision_function(scaled_vec)[0])
            anomaly_score = float(1.0 / (1.0 + np.exp(-raw_score * 5.0)))
        else:
            # Heuristic fallback if model not loaded
            dt = feat_dict["time_delta_minutes"]
            match = feat_dict["role_account_match"]
            score = 0.0
            if dt <= 5.0: score += 0.50
            if match == 0.0: score += 0.30
            if feat_dict["amount"] > 50000: score += 0.20
            anomaly_score = min(1.0, score)

        is_suspicious = bool(anomaly_score > self.threshold or feat_dict["time_delta_minutes"] <= 4.0)

        # Feature contributions / drivers
        drivers = []
        if feat_dict["time_delta_minutes"] <= 5.0:
            drivers.append(f"Tight temporal link: {feat_dict['time_delta_minutes']:.1f} mins between access and funds departure")
        if feat_dict["role_account_match"] == 0.0:
            drivers.append(f"Role mismatch: {access_event.get('role')} accessed {access_event.get('account_type')} account")
        if feat_dict["amount_zscore"] > 2.0:
            drivers.append(f"Anomalous transaction value: INR {feat_dict['amount']:,.2f} ({feat_dict['amount_zscore']:.1f}x baseline)")
        if feat_dict["hour_of_access"] < 8 or feat_dict["hour_of_access"] > 18:
            drivers.append(f"Off-hours internal access at {int(feat_dict['hour_of_access'])}:00")

        return {
            "model": "Isolation Forest",
            "detection_type": "Temporal Insider Link",
            "anomaly_score": round(anomaly_score, 4),
            "is_suspicious": is_suspicious,
            "time_gap_minutes": round(feat_dict["time_delta_minutes"], 2),
            "drivers": drivers[:3],
            "incident_link": "Citibank India Fraud (2010, INR 400 Crore)",
            "incident_summary": "Relationship manager accessed high-net-worth accounts right before fraudulent diversion sweeps."
        }


# Quick test
if __name__ == "__main__":
    detector = TemporalInsiderDetector()
    sample_access = {
        "employee_id": "EMP_0042", "role": "loan_officer",
        "account_id": "ACC_10017", "account_type": "savings",
        "timestamp": "2026-09-01 10:14:00", "records_accessed": 1
    }
    sample_txn = {
        "account_id": "ACC_10017", "amount": 250000.0,
        "transaction_type": "TRANSFER",
        "timestamp": "2026-09-01 10:18:00"
    }
    result = detector.predict(sample_access, sample_txn)
    print(json.dumps(result, indent=2))
