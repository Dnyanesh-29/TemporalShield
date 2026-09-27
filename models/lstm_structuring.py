"""
TemporalShield — LSTM Transaction Structuring Detector
Detects transaction splitting (smurfing) — breaking large transfers into sub-INR 50,000 chunks
to evade mandatory CTR reporting under India's PMLA (Prevention of Money Laundering Act).
"""

import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime

STRUCTURING_THRESHOLD = 50000.0  # RBI PMLA CTR reporting threshold
STRUCTURING_WINDOW_HOURS = 2.0


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM output sequence."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)            # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)            # (batch, hidden_dim)
        return context, weights.squeeze(2)


class StructuringLSTM(nn.Module):
    """3-layer LSTM with Bahdanau attention and LayerNorm."""
    def __init__(self, input_dim=12, hidden_dim=256, num_layers=3, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = AdditiveAttention(hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _        = self.lstm(x)
        context, attn = self.attention(out)
        context       = self.norm(context)
        return self.fc(self.drop(context)), attn


class StructuringDetector:
    """End-to-end inference wrapper for transaction structuring detection."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), "lstm"),
                os.path.join(os.path.dirname(__file__), "../models/lstm"),
                "models/lstm"
            ]
            model_dir = next((c for c in candidates if os.path.exists(c)), "models/lstm")

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.threshold = 0.50
        self.feature_cols = None
        self.sequence_length = 20
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.model_dir, "lstm_structuring_model.pt")
        scaler_path = os.path.join(self.model_dir, "lstm_scaler.pkl")
        thresh_path = os.path.join(self.model_dir, "lstm_optimal_threshold.json")
        cols_path = os.path.join(self.model_dir, "lstm_feature_columns.json")

        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                cfg = json.load(f)
                self.threshold = float(cfg.get("optimal_threshold", 0.50))

        if os.path.exists(cols_path):
            with open(cols_path, "r") as f:
                cfg = json.load(f)
                self.feature_cols = cfg.get("feature_columns", [])

        if not self.feature_cols:
            self.feature_cols = [
                "amount_normalised", "time_since_last_minutes", "is_below_threshold",
                "is_struct_band", "is_transfer", "is_fast", "roll_fast_3",
                "roll_struct_band_3", "cumulative_amount_window", "amount_homogeneity",
                "velocity_count", "balance_change_ratio"
            ]

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if os.path.exists(model_path):
            self.model = StructuringLSTM(
                input_dim=len(self.feature_cols),
                hidden_dim=256,
                num_layers=3,
                dropout=0.25
            )
            sd = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(sd)
            self.model.eval()

    def build_sequence(self, transactions):
        """
        Takes a list of transaction dictionaries (up to 20) and builds scaled tensor.
        """
        df = pd.DataFrame(transactions)
        if len(df) == 0:
            return np.zeros((1, self.sequence_length, len(self.feature_cols)), dtype=np.float32)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Feature calculations
        mean_amt = df["amount"].mean() if len(df) > 0 else 1.0
        df["amount_normalised"] = df["amount"] / (mean_amt + 1.0)
        df["time_since_last_minutes"] = (
            df["timestamp"].diff().dt.total_seconds().div(60.0).fillna(9999.0).clip(upper=10000.0)
        )
        df["is_below_threshold"] = (df["amount"] < STRUCTURING_THRESHOLD).astype(float)
        df["is_struct_band"] = ((df["amount"] >= 30000) & (df["amount"] < STRUCTURING_THRESHOLD)).astype(float)
        txn_types = df.get("transaction_type", df.get("type", pd.Series(["TRANSFER"]*len(df))))
        df["is_transfer"] = (txn_types.astype(str) == "TRANSFER").astype(float)
        df["is_fast"] = (df["time_since_last_minutes"] <= 120.0).astype(float)
        df["roll_fast_3"] = df["is_fast"].rolling(3, min_periods=1).sum()
        df["roll_struct_band_3"] = df["is_struct_band"].rolling(3, min_periods=1).sum()
        df["cumulative_amount_window"] = df["amount"].rolling(3, min_periods=1).sum()

        r_mean = df["amount"].rolling(3, min_periods=2).mean()
        r_std = df["amount"].rolling(3, min_periods=2).std().fillna(0.0)
        df["amount_homogeneity"] = (r_std / (r_mean + 1.0)).fillna(0.0).clip(0.0, 5.0)
        df["velocity_count"] = df["is_fast"].rolling(5, min_periods=1).sum()

        old_bal = df.get("oldbalanceOrg", pd.Series([100000.0]*len(df)))
        new_bal = df.get("newbalanceOrig", pd.Series([50000.0]*len(df)))
        df["balance_change_ratio"] = ((new_bal - old_bal) / (old_bal + 1.0)).clip(-10.0, 10.0)

        # Ensure all columns present
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        raw_feats = df[self.feature_cols].values.astype(np.float32)

        if self.scaler is not None:
            feats = self.scaler.transform(raw_feats)
        else:
            feats = raw_feats

        # Zero pad to sequence_length at the front
        n = len(feats)
        if n < self.sequence_length:
            pad = np.zeros((self.sequence_length - n, len(self.feature_cols)), dtype=np.float32)
            seq = np.vstack([pad, feats])
        else:
            seq = feats[-self.sequence_length:]

        return np.expand_dims(seq, axis=0)

    def predict(self, transactions):
        """
        Evaluates a transaction sequence for structuring.
        Inputs:
            transactions: list of dicts with keys: amount, timestamp, type, etc.
        Returns:
            dict containing structuring_detected, confidence_score, cluster_transactions, aggregate_amount.
        """
        if not transactions:
            return {
                "model": "LSTM + Attention",
                "detection_type": "Transaction Structuring (Smurfing)",
                "structuring_detected": False,
                "confidence_score": 0.0,
                "cluster_transactions": [],
                "aggregate_amount": 0.0
            }

        df = pd.DataFrame(transactions)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Rule-based cluster verification (PMLA: 3+ sub-50k txns within 2 hours summing > 75k)
        now_t = df["timestamp"].iloc[-1]
        w_start = now_t - pd.Timedelta(hours=STRUCTURING_WINDOW_HOURS)
        recent = df[(df["timestamp"] >= w_start) & (df["timestamp"] <= now_t)]
        sub_50k = recent[recent["amount"] < STRUCTURING_THRESHOLD]

        rule_matched = (len(sub_50k) >= 3 and sub_50k["amount"].sum() >= 75000.0)
        agg_amount = float(sub_50k["amount"].sum())

        # Neural sequence evaluation
        seq_tensor = torch.FloatTensor(self.build_sequence(transactions))
        confidence_score = 0.0
        attn_weights = []

        if self.model is not None:
            with torch.no_grad():
                logits, attn = self.model(seq_tensor)
                confidence_score = float(torch.sigmoid(logits).item())
                attn_weights = attn.squeeze(0).numpy().tolist()
        else:
            confidence_score = 0.90 if rule_matched else 0.10

        is_structuring = bool(confidence_score >= self.threshold or rule_matched)

        cluster_txns = []
        for _, row in sub_50k.iterrows():
            cluster_txns.append({
                "transaction_id": row.get("transaction_id", "N/A"),
                "timestamp": str(row["timestamp"]),
                "amount": float(row["amount"]),
                "type": row.get("transaction_type", row.get("type", "TRANSFER"))
            })

        evidence = (
            f"Detected {len(cluster_txns)} rapid transfers in a {STRUCTURING_WINDOW_HOURS}-hour window, "
            f"each below mandatory reporting threshold of INR 50,000. "
            f"Combined laundered value: INR {agg_amount:,.2f}."
        )

        return {
            "model": "LSTM + Attention",
            "detection_type": "Transaction Structuring (Smurfing)",
            "structuring_detected": is_structuring,
            "confidence_score": round(confidence_score, 4),
            "threshold": self.threshold,
            "cluster_count": len(cluster_txns),
            "cluster_transactions": cluster_txns,
            "aggregate_amount": round(agg_amount, 2),
            "evidence": evidence,
            "incident_link": "Hawala Syndicates & ED-Documented Smurfing Networks",
            "incident_summary": "Couriers execute clusters of sub-INR 50,000 transfers across mule networks to bypass RBI CTR filing."
        }


# Quick test
if __name__ == "__main__":
    detector = StructuringDetector()
    mock_txns = [
        {"transaction_id": "TXN_1", "amount": 42500.0, "type": "TRANSFER", "timestamp": "2026-09-01 14:10:00"},
        {"transaction_id": "TXN_2", "amount": 45100.0, "type": "TRANSFER", "timestamp": "2026-09-01 14:28:00"},
        {"transaction_id": "TXN_3", "amount": 48900.0, "type": "TRANSFER", "timestamp": "2026-09-01 14:52:00"},
        {"transaction_id": "TXN_4", "amount": 46200.0, "type": "TRANSFER", "timestamp": "2026-09-01 15:15:00"},
    ]
    res = detector.predict(mock_txns)
    print(json.dumps(res, indent=2))
