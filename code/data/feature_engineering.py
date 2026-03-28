"""
Feature engineering pipeline for fraud detection.
Extracts temporal, statistical, and behavioral features from transactions.
"""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Extract features from transaction data for ML models."""

    def __init__(self):
        self.feature_names = []
        self.categorical_mappings = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn categorical encodings from training data."""
        # Learn label encodings
        self.categorical_mappings["merchant_category"] = {
            cat: idx for idx, cat in enumerate(df["merchant_category"].unique())
        }
        self.categorical_mappings["location"] = {
            loc: idx for idx, loc in enumerate(df["location"].unique())
        }
        self.categorical_mappings["device"] = {
            dev: idx for idx, dev in enumerate(df["device"].unique())
        }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform transactions into feature matrix."""
        df = df.copy()

        # Parse timestamp
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

        # Temporal features
        df["hour"] = df["timestamp_dt"].dt.hour
        df["day_of_week"] = df["timestamp_dt"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

        # Amount features
        df["amount_log"] = np.log1p(df["amount"])

        # Categorical encodings
        for col, mapping in self.categorical_mappings.items():
            df[f"{col}_encoded"] = df[col].map(mapping).fillna(-1).astype(int)

        df["is_international"] = (
            ~df["location"].str.startswith("US", na=False)
        ).astype(int)

        # Velocity feature (already computed in generator)
        if "tx_count_1h" not in df.columns:
            df["tx_count_1h"] = 0

        # Device consistency (binary flag for web/mobile)
        df["is_card_present"] = (df["device"] == "pos").astype(int)

        # Select feature columns
        feature_cols = [
            "amount",
            "amount_log",
            "merchant_category_encoded",
            "location_encoded",
            "device_encoded",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",
            "is_business_hours",
            "is_international",
            "tx_count_1h",
            "is_card_present",
        ]

        self.feature_names = feature_cols

        # Return feature matrix
        X = df[feature_cols].copy()
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names
