"""
TemporalShield — PaySim Dataset Loader & Feature Preprocessor
Used EXCLUSIVELY for training the LSTM Structuring Detector and XGBoost Profile Mismatch model.

NOTE: PaySim (6M+ transactions) is NOT used for Kafka streaming.
It uses distinct account IDs (C123456789, M987654321) that are incompatible with bank employee access logs.
Only synthetic_transactions.csv (which shares ACC_XXXXX account IDs with access_logs.csv) streams through Kafka.

Models using PaySim:
- LSTM Structuring Detector: PaySim TRANSFER and CASH_OUT sequences.
- XGBoost Profile Mismatch: PaySim account history baselines.

Models NOT using PaySim:
- Isolation Forest: Uses access_logs + synthetic_transactions (requires ACC_XXXXX link).
- Role Autoencoder: Uses access_logs only.
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("TemporalShield.PaySimLoader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Default search paths for PaySim dataset
PAYSIM_POSSIBLE_NAMES = [
    "PS_20174392719_1491204439457_log.csv",
    "paysim.csv",
    "paysim_data.csv",
    "data/paysim.csv",
    "data/PS_20174392719_1491204439457_log.csv"
]


class PaySimLoader:
    """
    Loads, samples, and prepares the PaySim synthetic mobile money dataset
    for offline model training (LSTM sequences and XGBoost profile features).
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = self._resolve_path(data_path)

    def _resolve_path(self, user_path: Optional[str]) -> Optional[str]:
        if user_path and os.path.exists(user_path):
            return user_path

        # Look in workspace data dir and parent dirs
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(current_dir)

        search_dirs = [
            current_dir,
            workspace_dir,
            os.path.join(workspace_dir, "data"),
            os.path.join(workspace_dir, "dataset"),
            os.path.join(workspace_dir, "datasets")
        ]

        for s_dir in search_dirs:
            for name in PAYSIM_POSSIBLE_NAMES:
                candidate = os.path.join(s_dir, name)
                if os.path.exists(candidate):
                    return candidate
            # Check for any .csv with 'paysim' or 'PS_' in name
            matches = glob.glob(os.path.join(s_dir, "*paysim*.csv")) + glob.glob(os.path.join(s_dir, "PS_*.csv"))
            if matches:
                return matches[0]

        return None

    def load_raw(
        self,
        filter_types: Optional[Tuple[str, ...]] = ("TRANSFER", "CASH_OUT"),
        nrows: Optional[int] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Loads the PaySim dataset. PaySim contains only fraud on TRANSFER and CASH_OUT transactions.
        If file not found, generates a representative synthetic sample for development and testing.
        """
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading PaySim data from: {self.data_path}")
            df = pd.read_csv(self.data_path, nrows=nrows)

            if filter_types:
                df = df[df["type"].isin(filter_types)].copy()

            if sample_frac and 0.0 < sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

            logger.info(f"Loaded PaySim DataFrame with {len(df):,} rows and {len(df.columns)} columns.")
            return df
        else:
            logger.warning(
                "PaySim dataset not found on disk. Generating a synthetic fallback PaySim-format "
                "DataFrame for testing/dev (Kaggle download: ealaxi/paysim1)."
            )
            return self._generate_mock_paysim(n_samples=nrows or 5000, random_state=random_state)

    def _generate_mock_paysim(self, n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
        """Generates mock PaySim dataframe conforming to standard schema for unit testing/dev."""
        rng = np.random.default_rng(random_state)
        steps = rng.integers(1, 744, size=n_samples)
        types = rng.choice(["TRANSFER", "CASH_OUT"], size=n_samples, p=[0.45, 0.55])
        amounts = rng.lognormal(mean=9.5, sigma=1.2, size=n_samples).round(2)

        # 500 distinct origin accounts to form realistic sequences
        orig_accounts = [f"C{rng.integers(100000000, 999999999)}" for _ in range(500)]
        dest_accounts = [f"C{rng.integers(100000000, 999999999)}" for _ in range(1000)]

        name_orig = rng.choice(orig_accounts, size=n_samples)
        name_dest = rng.choice(dest_accounts, size=n_samples)

        old_bal_org = amounts * rng.uniform(0.8, 2.5, size=n_samples)
        new_bal_org = np.maximum(0.0, old_bal_org - amounts)
        old_bal_dest = rng.uniform(0, 500000, size=n_samples)
        new_bal_dest = old_bal_dest + amounts

        # ~1% fraud rate
        is_fraud = rng.choice([0, 1], size=n_samples, p=[0.988, 0.012])
        is_flagged = np.where((amounts > 200000) & (is_fraud == 1), 1, 0)

        df = pd.DataFrame({
            "step": steps,
            "type": types,
            "amount": amounts,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_bal_org,
            "newbalanceOrig": new_bal_org,
            "nameDest": name_dest,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged
        }).sort_values("step").reset_index(drop=True)

        return df

    def prepare_lstm_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 20,
        min_seq_len: int = 2,
        structuring_threshold: float = 50000.0
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepares sequential transaction tensors for training the LSTM Structuring Detector.
        Groups by nameOrig and creates multi-step sliding windows.
        
        Features engineered:
        - normalized amount
        - step difference (time proxy)
        - is_below_threshold (sub-50k structuring marker)
        - is_transfer
        - rolling velocity
        - balance change ratio
        """
        feature_cols = [
            "amount_norm", "time_since_last_steps", "is_below_threshold",
            "is_struct_band", "is_transfer", "rolling_velocity_3",
            "balance_change_ratio"
        ]

        sequences = []
        labels = []

        grouped = df.sort_values(["nameOrig", "step"]).groupby("nameOrig")

        for _, group in grouped:
            if len(group) < min_seq_len:
                continue

            grp = group.copy()
            amt_mean = grp["amount"].mean() if grp["amount"].mean() > 0 else 1.0
            grp["amount_norm"] = grp["amount"] / (amt_mean + 1.0)
            grp["time_since_last_steps"] = grp["step"].diff().fillna(0.0).clip(upper=100.0)
            grp["is_below_threshold"] = (grp["amount"] < structuring_threshold).astype(float)
            grp["is_struct_band"] = ((grp["amount"] >= 30000) & (grp["amount"] < structuring_threshold)).astype(float)
            grp["is_transfer"] = (grp["type"] == "TRANSFER").astype(float)
            grp["rolling_velocity_3"] = grp["amount"].rolling(3, min_periods=1).count()
            grp["balance_change_ratio"] = (
                (grp["newbalanceOrig"] - grp["oldbalanceOrg"]) / (grp["oldbalanceOrg"] + 1.0)
            ).clip(-10.0, 10.0)

            feats = grp[feature_cols].values.astype(np.float32)
            lbl = int(grp["isFraud"].max())

            # Pad or truncate to sequence_length
            n = len(feats)
            if n < sequence_length:
                pad = np.zeros((sequence_length - n, len(feature_cols)), dtype=np.float32)
                seq = np.vstack([pad, feats])
            else:
                seq = feats[-sequence_length:]

            sequences.append(seq)
            labels.append(lbl)

        if not sequences:
            return np.zeros((0, sequence_length, len(feature_cols)), dtype=np.float32), np.zeros((0,), dtype=np.int64), feature_cols

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        logger.info(f"Prepared {len(X)} LSTM sequences of shape ({sequence_length}, {len(feature_cols)})")
        return X, y, feature_cols

    def prepare_xgboost_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Engineers baseline profile and transaction mismatch features for XGBoost training.
        
        Features:
        - errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
        - errorBalanceDest = oldbalanceDest + amount - newbalanceDest
        - amount_vs_historical_mean_ratio
        - balance_drop_ratio
        - is_transfer
        """
        df_feat = df.copy()

        # Balance accounting discrepancies (standard PaySim signal)
        df_feat["errorBalanceOrig"] = df_feat["newbalanceOrig"] + df_feat["amount"] - df_feat["oldbalanceOrg"]
        df_feat["errorBalanceDest"] = df_feat["oldbalanceDest"] + df_feat["amount"] - df_feat["newbalanceDest"]

        # Historical profile estimation per account
        hist_stats = df_feat.groupby("nameOrig")["amount"].agg(["mean", "max", "count"]).rename(
            columns={"mean": "historical_mean_amount", "max": "historical_max_amount", "count": "historical_txn_count"}
        )
        df_feat = df_feat.merge(hist_stats, on="nameOrig", how="left")

        df_feat["amount_vs_historical_mean_ratio"] = (
            df_feat["amount"] / (df_feat["historical_mean_amount"] + 1.0)
        ).clip(0.0, 100.0)
        df_feat["amount_vs_historical_max_ratio"] = (
            df_feat["amount"] / (df_feat["historical_max_amount"] + 1.0)
        ).clip(0.0, 10.0)

        df_feat["balance_before"] = df_feat["oldbalanceOrg"]
        df_feat["balance_after"] = df_feat["newbalanceOrig"]
        df_feat["balance_drop_ratio"] = (
            (df_feat["oldbalanceOrg"] - df_feat["newbalanceOrig"]) / (df_feat["oldbalanceOrg"] + 1.0)
        ).clip(-10.0, 10.0)
        df_feat["balance_is_zero_after"] = (df_feat["newbalanceOrig"] <= 0.01).astype(float)
        df_feat["is_transfer"] = (df_feat["type"] == "TRANSFER").astype(float)
        df_feat["type_encoded"] = df_feat["type"].map({"TRANSFER": 4.0, "CASH_OUT": 1.0}).fillna(0.0)

        # Account age / step proxy
        df_feat["account_age_steps"] = df_feat["step"].astype(float)
        df_feat["days_since_last_transaction"] = 1.0
        df_feat["hour_of_day"] = (df_feat["step"] % 24).astype(float)
        df_feat["current_amount"] = df_feat["amount"].astype(float)
        df_feat["is_new_counterparty"] = 1.0
        df_feat["is_dormant_account"] = 0.0
        df_feat["type_matches_history"] = 1.0
        df_feat["historical_std_amount"] = 0.0

        xgb_feature_cols = [
            "account_age_steps", "historical_mean_amount", "historical_std_amount",
            "historical_txn_count", "historical_max_amount", "days_since_last_transaction",
            "hour_of_day", "current_amount", "amount_vs_historical_mean_ratio",
            "amount_vs_historical_max_ratio", "is_new_counterparty", "balance_before",
            "balance_after", "balance_drop_ratio", "balance_is_zero_after",
            "type_matches_history", "is_dormant_account", "type_encoded",
            "errorBalanceOrig", "errorBalanceDest", "is_transfer"
        ]

        X = df_feat[xgb_feature_cols].copy()
        y = df_feat["isFraud"].astype(int)

        logger.info(f"Prepared {len(X)} XGBoost feature vectors with {len(xgb_feature_cols)} features.")
        return X, y, xgb_feature_cols


# Standalone runner for testing
if __name__ == "__main__":
    loader = PaySimLoader()
    df_raw = loader.load_raw(nrows=1000)
    print("Loaded sample shape:", df_raw.shape)
    X_lstm, y_lstm, lstm_cols = loader.prepare_lstm_sequences(df_raw, sequence_length=10)
    print("LSTM X shape:", X_lstm.shape, "y shape:", y_lstm.shape)
    X_xgb, y_xgb, xgb_cols = loader.prepare_xgboost_features(df_raw)
    print("XGBoost X shape:", X_xgb.shape, "y shape:", y_xgb.shape)
