"""
TemporalShield — Role-Based Autoencoder Anomaly Detector
Learns normal access behavior per employee role.
When an employee accesses accounts or resources outside their operational profile,
reconstruction error spikes, signaling privilege abuse or unauthorized snooping.
"""

import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime

ROLES = [
    'loan_officer',
    'savings_representative',
    'branch_manager',
    'it_admin',
    'compliance_officer'
]

AE_FEATURES = [
    'hour_of_day', 'day_of_week', 'records_accessed',
    'action_type_encoded', 'account_type_encoded', 'branch_code_encoded',
    'is_weekend', 'is_off_hours', 'records_accessed_zscore',
    'access_frequency_today', 'same_branch_access', 'action_type_rarity'
]


class RoleAutoencoder(nn.Module):
    """Deep autoencoder with bottleneck dimension 4 for role access compression."""
    def __init__(self, input_dim=12):
        super(RoleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class RoleAnomalyDetector:
    """Manages role-specific autoencoders and reconstruction error evaluation."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), "autoencoder"),
                os.path.join(os.path.dirname(__file__), "../models/autoencoder"),
                "models/autoencoder"
            ]
            model_dir = next((c for c in candidates if os.path.exists(c)), "models/autoencoder")

        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_artifacts()

    def _load_artifacts(self):
        # Load thresholds
        thresh_path = os.path.join(self.model_dir, "thresholds_autoencoder_optimised.json")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                self.thresholds = json.load(f)

        # Load label encoders
        le_path = os.path.join(self.model_dir, "label_encoders_ae.pkl")
        if os.path.exists(le_path):
            self.label_encoders = joblib.load(le_path)

        # Load role models and scalers
        for role in ROLES:
            model_file = os.path.join(self.model_dir, f"autoencoder_{role}.pt")
            scaler_file = os.path.join(self.model_dir, f"scaler_ae_{role}.pkl")

            if os.path.exists(scaler_file):
                self.scalers[role] = joblib.load(scaler_file)

            if os.path.exists(model_file):
                model = RoleAutoencoder(len(AE_FEATURES)).to(self.device)
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.eval()
                self.models[role] = model

    def extract_features(self, access_event):
        """Builds the 12 input features for the role autoencoder."""
        ts = pd.to_datetime(access_event.get("timestamp", datetime.now()))
        hour = ts.hour
        day = ts.dayofweek
        records = float(access_event.get("records_accessed", 1))

        # Encoders
        def safe_encode(col, val):
            if col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    return float(le.transform([str(val)])[0])
                except Exception:
                    return 0.0
            return 0.0

        action_enc = safe_encode("action_type", access_event.get("action_type", "read"))
        acct_enc = safe_encode("account_type", access_event.get("account_type", "savings"))
        branch_enc = safe_encode("branch_code", access_event.get("branch_code", "BR_001"))

        is_weekend = 1.0 if day >= 5 else 0.0
        is_off_hours = 1.0 if (hour < 8 or hour > 18) else 0.0

        records_zscore = (records - 2.0) / 1.5
        access_freq = float(access_event.get("access_frequency_today", 3.0))
        same_branch = float(access_event.get("same_branch_access", 1.0))
        rarity = 0.85 if access_event.get("action_type") == "override" else 0.10

        return [
            float(hour), float(day), float(records),
            float(action_enc), float(acct_enc), float(branch_enc),
            float(is_weekend), float(is_off_hours), float(records_zscore),
            float(access_freq), float(same_branch), float(rarity)
        ]

    def predict(self, access_event):
        """
        Evaluates an employee access event against their role profile.
        Returns:
            dict containing reconstruction_error, threshold, is_anomalous, and feature deviations.
        """
        role = access_event.get("role", "savings_representative")
        features_raw = self.extract_features(access_event)
        raw_arr = np.array([features_raw], dtype=np.float32)

        scaler = self.scalers.get(role)
        if scaler is not None:
            scaled_arr = scaler.transform(raw_arr)
        else:
            scaled_arr = raw_arr

        model = self.models.get(role)
        threshold = float(self.thresholds.get(role, 0.05))

        if model is not None:
            with torch.no_grad():
                x_tensor = torch.FloatTensor(scaled_arr).to(self.device)
                reconstructed = model(x_tensor)
                per_feature_error = (x_tensor - reconstructed) ** 2
                mse = float(torch.mean(per_feature_error).item())
                feature_errors = per_feature_error.cpu().squeeze(0).numpy().tolist()
        else:
            # Fallback estimation
            mse = 0.12 if features_raw[7] == 1.0 else 0.02
            feature_errors = [0.0] * len(AE_FEATURES)

        is_anomalous = bool(mse > threshold)

        # Identify which features were poorly reconstructed
        top_error_indices = np.argsort(feature_errors)[::-1][:3]
        drivers = []
        for idx in top_error_indices:
            feat_name = AE_FEATURES[idx]
            val = features_raw[idx]
            drivers.append(f"{feat_name} (val: {val:.1f}, error: {feature_errors[idx]:.3f})")

        return {
            "model": "Role Autoencoder",
            "detection_type": "Role Permission Anomaly",
            "role": role,
            "reconstruction_error": round(mse, 5),
            "threshold": round(threshold, 5),
            "is_anomalous": is_anomalous,
            "top_unusual_features": drivers,
            "incident_link": "Punjab National Bank (PNB) Fraud (2018, INR 11,400 Crore)",
            "incident_summary": "Unauthorized SWIFT messages sent during off-hours with zero underlying core banking records."
        }


# Quick test
if __name__ == "__main__":
    detector = RoleAnomalyDetector()
    sample_off_hours = {
        "employee_id": "EMP_0012",
        "role": "savings_representative",
        "account_id": "ACC_10022",
        "account_type": "system",
        "branch_code": "BR_009",
        "action_type": "override",
        "timestamp": "2026-09-01 02:45:00",
        "records_accessed": 18
    }
    res = detector.predict(sample_off_hours)
    print(json.dumps(res, indent=2))
