"""
Deterministic synthetic transaction and fraud data generator.
Produces realistic financial transaction patterns with configurable fraud scenarios.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class UserProfile:
    """User spending profile."""

    user_id: str
    typical_amount_mean: float
    typical_amount_std: float
    typical_merchant_categories: List[str]
    typical_locations: List[str]
    fraud_propensity: float  # 0-1, probability of being targeted


class TransactionGenerator:
    """Deterministic synthetic transaction generator with fraud patterns."""

    def __init__(self, n_users: int = 1000, random_state: int = 42):
        self.n_users = n_users
        self.rng = np.random.RandomState(random_state)
        self.users = self._generate_user_profiles()

        # Merchant categories
        self.merchant_categories = [
            "grocery",
            "gas",
            "restaurant",
            "retail",
            "online",
            "travel",
            "entertainment",
            "utilities",
            "healthcare",
            "other",
        ]

        # Locations
        self.locations = [
            "US-NY",
            "US-CA",
            "US-TX",
            "US-FL",
            "US-IL",
            "UK-LON",
            "FR-PAR",
            "JP-TYO",
            "CN-BEJ",
            "ONLINE",
        ]

    def _generate_user_profiles(self) -> List[UserProfile]:
        """Generate diverse user spending profiles."""
        profiles = []
        for i in range(self.n_users):
            # Log-normal distribution for spending (realistic wealth distribution)
            amount_mean = self.rng.lognormal(mean=4.0, sigma=1.0)
            amount_std = amount_mean * self.rng.uniform(0.2, 0.5)

            # Random preferences for merchant categories (2-4 categories)
            n_categories = self.rng.randint(2, 5)
            preferred_categories = self.rng.choice(
                self.merchant_categories, size=n_categories, replace=False
            ).tolist()

            # Typical locations (1-3 locations)
            n_locations = self.rng.randint(1, 4)
            typical_locs = self.rng.choice(
                self.locations[:5], size=n_locations, replace=False  # Domestic mostly
            ).tolist()

            # Fraud propensity (most users low, some high-value targets)
            fraud_prop = self.rng.beta(a=1, b=10)  # Right-skewed

            profiles.append(
                UserProfile(
                    user_id=f"U{i:06d}",
                    typical_amount_mean=amount_mean,
                    typical_amount_std=amount_std,
                    typical_merchant_categories=preferred_categories,
                    typical_locations=typical_locs,
                    fraud_propensity=fraud_prop,
                )
            )

        return profiles

    def generate_normal_transaction(
        self, user: UserProfile, timestamp: datetime
    ) -> Dict:
        """Generate a normal (non-fraud) transaction."""
        amount = max(
            1.0, self.rng.normal(user.typical_amount_mean, user.typical_amount_std)
        )
        merchant_cat = self.rng.choice(user.typical_merchant_categories)
        location = self.rng.choice(user.typical_locations)

        # Time-of-day pattern (most transactions during business hours)

        return {
            "transaction_id": f"TX{self.rng.randint(0, 1e9):09d}",
            "user_id": user.user_id,
            "timestamp": timestamp.isoformat(),
            "amount": round(amount, 2),
            "merchant_category": merchant_cat,
            "location": location,
            "device": self.rng.choice(["mobile", "web", "pos"]),
            "is_fraud": 0,
            "fraud_type": None,
        }

    def generate_fraud_transaction(
        self, user: UserProfile, timestamp: datetime, fraud_type: str
    ) -> Dict:
        """Generate fraudulent transaction with specific fraud pattern."""
        base_tx = self.generate_normal_transaction(user, timestamp)

        if fraud_type == "amount_anomaly":
            # Unusually large transaction
            base_tx["amount"] = user.typical_amount_mean * self.rng.uniform(5, 15)
            base_tx["merchant_category"] = self.rng.choice(
                ["online", "travel", "retail"]
            )

        elif fraud_type == "location_anomaly":
            # Transaction from unusual location
            base_tx["location"] = self.rng.choice(
                [loc for loc in self.locations if loc not in user.typical_locations]
            )

        elif fraud_type == "velocity":
            # Multiple transactions in short time (handled at batch level)
            base_tx["amount"] = user.typical_amount_mean * self.rng.uniform(1.5, 3)

        elif fraud_type == "time_anomaly":
            # Transaction at unusual hour
            unusual_hour = self.rng.choice([2, 3, 4, 23])
            base_tx["timestamp"] = timestamp.replace(hour=unusual_hour).isoformat()

        elif fraud_type == "merchant_anomaly":
            # Unusual merchant category
            unusual_cats = [
                cat
                for cat in self.merchant_categories
                if cat not in user.typical_merchant_categories
            ]
            if unusual_cats:
                base_tx["merchant_category"] = self.rng.choice(unusual_cats)

        elif fraud_type == "account_takeover":
            # Multiple indicators: different location, large amount, unusual merchant
            base_tx["amount"] = user.typical_amount_mean * self.rng.uniform(3, 8)
            base_tx["location"] = self.rng.choice(self.locations[-3:])  # Foreign
            base_tx["device"] = "web"  # Changed device

        base_tx["is_fraud"] = 1
        base_tx["fraud_type"] = fraud_type

        return base_tx

    def generate_dataset(
        self, n_transactions: int = 10000, fraud_ratio: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate complete transaction dataset with fraud cases.

        Args:
            n_transactions: Total number of transactions
            fraud_ratio: Proportion of fraudulent transactions

        Returns:
            DataFrame with transaction data
        """
        transactions = []
        n_fraud = int(n_transactions * fraud_ratio)
        n_normal = n_transactions - n_fraud

        # Generate time range (90 days)
        start_date = datetime(2025, 1, 1)

        # Generate normal transactions
        for i in range(n_normal):
            user = self.rng.choice(self.users)
            # Random timestamp within 90 days
            days_offset = self.rng.uniform(0, 90)
            timestamp = start_date + timedelta(days=days_offset)
            tx = self.generate_normal_transaction(user, timestamp)
            transactions.append(tx)

        # Generate fraud transactions
        fraud_types = [
            "amount_anomaly",
            "location_anomaly",
            "velocity",
            "time_anomaly",
            "merchant_anomaly",
            "account_takeover",
        ]

        for i in range(n_fraud):
            # Target high-propensity users more often
            user_weights = [u.fraud_propensity for u in self.users]
            user_weights = np.array(user_weights) / sum(user_weights)
            user = self.rng.choice(self.users, p=user_weights)

            days_offset = self.rng.uniform(0, 90)
            timestamp = start_date + timedelta(days=days_offset)

            fraud_type = self.rng.choice(fraud_types)
            tx = self.generate_fraud_transaction(user, timestamp, fraud_type)
            transactions.append(tx)

        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(transactions)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add velocity features (transactions in past hour for each user)
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt").reset_index(drop=True)

        df["tx_count_1h"] = 0
        for user_id in df["user_id"].unique():
            user_mask = df["user_id"] == user_id
            user_df = df.loc[user_mask, ["timestamp_dt"]].copy()

            counts = []
            for idx, row in user_df.iterrows():
                time_window = row["timestamp_dt"] - timedelta(hours=1)
                # Single combined mask — no chained indexing, no index mismatch
                count = user_df.loc[
                    (user_df["timestamp_dt"] < row["timestamp_dt"])
                    & (user_df["timestamp_dt"] >= time_window)
                ].shape[0]
                counts.append(count)

            df.loc[user_mask, "tx_count_1h"] = counts

        df = df.drop("timestamp_dt", axis=1)

        return df

    def validate_distribution(self, df: pd.DataFrame) -> Dict:
        """Validate statistical properties of generated data."""
        stats = {
            "n_transactions": len(df),
            "n_users": df["user_id"].nunique(),
            "fraud_ratio": df["is_fraud"].mean(),
            "amount_mean": df["amount"].mean(),
            "amount_std": df["amount"].std(),
            "fraud_amount_mean": df[df["is_fraud"] == 1]["amount"].mean(),
            "normal_amount_mean": df[df["is_fraud"] == 0]["amount"].mean(),
            "fraud_types": df[df["is_fraud"] == 1]["fraud_type"]
            .value_counts()
            .to_dict(),
            "merchant_distribution": df["merchant_category"]
            .value_counts(normalize=True)
            .to_dict(),
            "location_distribution": df["location"]
            .value_counts(normalize=True)
            .to_dict(),
        }
        return stats


def main():
    """Generate and save synthetic dataset."""
    generator = TransactionGenerator(n_users=1000, random_state=42)

    # Generate train/val/test splits
    train_df = generator.generate_dataset(n_transactions=50000, fraud_ratio=0.02)
    val_df = generator.generate_dataset(n_transactions=10000, fraud_ratio=0.02)
    test_df = generator.generate_dataset(n_transactions=10000, fraud_ratio=0.02)

    # Save datasets
    train_df.to_csv("../data/transactions_train.csv", index=False)
    val_df.to_csv("../data/transactions_val.csv", index=False)
    test_df.to_csv("../data/transactions_test.csv", index=False)

    # Validate and save stats
    stats = generator.validate_distribution(train_df)
    with open("../data/data_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Generated datasets:")
    print(f"  Train: {len(train_df)} transactions ({train_df['is_fraud'].sum()} fraud)")
    print(f"  Val:   {len(val_df)} transactions ({val_df['is_fraud'].sum()} fraud)")
    print(f"  Test:  {len(test_df)} transactions ({test_df['is_fraud'].sum()} fraud)")
    print("\nStatistics saved to data_statistics.json")


if __name__ == "__main__":
    main()
