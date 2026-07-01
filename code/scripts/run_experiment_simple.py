"""
Simplified experiment runner using only sklearn (no xgboost dependency).
"""

import logging as _lg

# --- keep run output readable: suppress benign third-party noise (auto-added) ---
import os as _os
import warnings as _w

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
for _m in (
    r".*does not have valid feature names.*",
    r".*tight_layout.*",
    r".*Gym has been unmaintained.*",
    r".*not wrapped with a ``Monitor``.*",
):
    _w.filterwarnings("ignore", message=_m)
_w.filterwarnings("ignore", category=DeprecationWarning)
_w.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning as _CW

    _w.filterwarnings("ignore", category=_CW)
except Exception:
    pass
for _n in (
    "matplotlib",
    "PIL",
    "urllib3",
    "yfinance",
    "tensorflow",
    "absl",
    "gym",
    "gymnasium",
    "shap",
    "numba",
    "h5py",
):
    _lg.getLogger(_n).setLevel(_lg.ERROR)


def _silence_tqdm():
    try:
        import tqdm.std as _tstd

        _orig = _tstd.tqdm.__init__

        def _init(self, *a, **k):
            k["disable"] = True
            _orig(self, *a, **k)

        _tstd.tqdm.__init__ = _init
        try:
            from tqdm import auto as _ta

            if _ta.tqdm is not _tstd.tqdm:
                _o2 = _ta.tqdm.__init__

                def _init2(self, *a, **k):
                    k["disable"] = True
                    _o2(self, *a, **k)

                _ta.tqdm.__init__ = _init2
        except Exception:
            pass
    except Exception:
        pass


_silence_tqdm()
# --- end output cleanup ---

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# Simple config
class Config:
    def __init__(self):
        self.random_state = 42
        # Anchor to the code directory so paths resolve regardless of CWD.
        _code_dir = Path(__file__).resolve().parent.parent
        self.data_dir = _code_dir / "data"
        self.results_dir = _code_dir / "results"
        self.figures_dir = _code_dir / "figures"


def generate_synthetic_data(n_transactions=5000, fraud_ratio=0.02, random_state=42):
    """Generate simple synthetic transaction data."""
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud

    # Normal transactions
    normal_amounts = rng.lognormal(mean=4, sigma=1, size=n_normal)
    normal_hours = rng.choice(range(9, 22), size=n_normal)  # Business hours
    normal_velocity = rng.poisson(lam=0.5, size=n_normal)
    normal_international = rng.binomial(1, 0.1, size=n_normal)

    normal_data = {
        "transaction_id": [f"TX{i:09d}" for i in range(n_normal)],
        "user_id": [f"U{rng.randint(0, 1000):06d}" for _ in range(n_normal)],
        "amount": normal_amounts,
        "hour": normal_hours,
        "tx_count_1h": normal_velocity,
        "is_international": normal_international,
        "is_night": 0,
        "is_fraud": 0,
    }

    # Fraud transactions
    fraud_amounts = rng.lognormal(mean=4, sigma=1, size=n_fraud) * rng.uniform(
        3, 10, size=n_fraud
    )
    fraud_hours = rng.choice([2, 3, 4, 23], size=n_fraud)  # Unusual hours
    fraud_velocity = rng.poisson(lam=2, size=n_fraud) + 2  # Higher velocity
    fraud_international = rng.binomial(1, 0.7, size=n_fraud)  # More international

    fraud_data = {
        "transaction_id": [f"TX{i+n_normal:09d}" for i in range(n_fraud)],
        "user_id": [f"U{rng.randint(0, 1000):06d}" for _ in range(n_fraud)],
        "amount": fraud_amounts,
        "hour": fraud_hours,
        "tx_count_1h": fraud_velocity,
        "is_international": fraud_international,
        "is_night": 1,
        "is_fraud": 1,
    }

    # Combine
    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def extract_features(df):
    """Extract feature matrix."""
    feature_cols = ["amount", "hour", "tx_count_1h", "is_international", "is_night"]
    return df[feature_cols].values


def train_and_evaluate(config):
    """Main experiment."""
    print("=" * 60)
    print("FRAUD DETECTION EXPERIMENT - SIMPLIFIED")
    print("=" * 60)

    # Create directories
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "metrics").mkdir(exist_ok=True)
    (config.results_dir / "logs").mkdir(exist_ok=True)

    # Generate data
    print("\n1. Generating synthetic data...")
    train_df = generate_synthetic_data(
        n_transactions=5000, fraud_ratio=0.02, random_state=config.random_state
    )
    test_df = generate_synthetic_data(
        n_transactions=1000, fraud_ratio=0.02, random_state=config.random_state + 1
    )

    print(
        f"   Train: {len(train_df)} transactions ({train_df['is_fraud'].sum()} fraud)"
    )
    print(f"   Test: {len(test_df)} transactions ({test_df['is_fraud'].sum()} fraud)")

    # Save data
    train_df.to_csv(config.data_dir / "transactions_train.csv", index=False)
    test_df.to_csv(config.data_dir / "transactions_test.csv", index=False)

    # Extract features
    X_train = extract_features(train_df)
    y_train = train_df["is_fraud"].values
    X_test = extract_features(test_df)
    y_test = test_df["is_fraud"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Model 1: Isolation Forest
    print("\n2. Training Isolation Forest...")
    iso_model = IsolationForest(
        contamination=0.02, random_state=config.random_state, n_jobs=-1
    )
    iso_model.fit(X_train_scaled)

    iso_pred = (iso_model.predict(X_test_scaled) == -1).astype(int)
    iso_scores = -iso_model.score_samples(X_test_scaled)
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

    iso_metrics = compute_metrics(y_test, iso_pred, iso_scores)
    results["isolation_forest"] = iso_metrics
    print(
        f"   Precision: {iso_metrics['precision']:.3f}, Recall: {iso_metrics['recall']:.3f}, F1: {iso_metrics['f1_score']:.3f}, AUC: {iso_metrics['auc_roc']:.3f}"
    )

    # Model 2: Random Forest (sklearn alternative to XGBoost)
    print("\n3. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=config.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled)
    rf_scores = rf_model.predict_proba(X_test_scaled)[:, 1]

    rf_metrics = compute_metrics(y_test, rf_pred, rf_scores)
    results["random_forest"] = rf_metrics
    print(
        f"   Precision: {rf_metrics['precision']:.3f}, Recall: {rf_metrics['recall']:.3f}, F1: {rf_metrics['f1_score']:.3f}, AUC: {rf_metrics['auc_roc']:.3f}"
    )

    # Model 3: Ensemble (weighted average)
    print("\n4. Creating Ensemble...")
    ensemble_scores = 0.3 * iso_scores + 0.7 * rf_scores
    ensemble_pred = (ensemble_scores > 0.5).astype(int)

    ensemble_metrics = compute_metrics(y_test, ensemble_pred, ensemble_scores)
    results["ensemble"] = ensemble_metrics
    print(
        f"   Precision: {ensemble_metrics['precision']:.3f}, Recall: {ensemble_metrics['recall']:.3f}, F1: {ensemble_metrics['f1_score']:.3f}, AUC: {ensemble_metrics['auc_roc']:.3f}"
    )

    # Simulate detection latency
    detection_times = []
    print("\n5. Measuring detection latency...")
    for i in range(len(X_test)):
        start = time.time()
        _ = iso_model.predict([X_test_scaled[i]])
        _ = rf_model.predict([X_test_scaled[i]])
        detection_times.append((time.time() - start) * 1000)  # ms

    latency_stats = {
        "mean_ms": float(np.mean(detection_times)),
        "median_ms": float(np.median(detection_times)),
        "p95_ms": float(np.percentile(detection_times, 95)),
        "p99_ms": float(np.percentile(detection_times, 99)),
    }

    print(f"   Mean latency: {latency_stats['mean_ms']:.2f}ms")
    print(f"   P95 latency: {latency_stats['p95_ms']:.2f}ms")

    # Save results
    print("\n6. Saving results...")
    with open(config.results_dir / "metrics" / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(config.results_dir / "metrics" / "latency_stats.json", "w") as f:
        json.dump(latency_stats, f, indent=2)

    # Create comparison table
    comparison = pd.DataFrame(
        {
            "model": list(results.keys()),
            "precision": [results[m]["precision"] for m in results.keys()],
            "recall": [results[m]["recall"] for m in results.keys()],
            "f1_score": [results[m]["f1_score"] for m in results.keys()],
            "auc_roc": [results[m]["auc_roc"] for m in results.keys()],
        }
    )
    comparison.to_csv(
        config.results_dir / "metrics" / "model_comparison.csv", index=False
    )

    print(f"\nResults saved to {config.results_dir}/metrics/")
    W = 60
    print("\n" + "=" * W)
    print("FRAUD DETECTION - RESULTS".center(W))
    print("=" * W)
    print(f"\n{'Model':<20}{'Precision':>10}{'Recall':>9}{'F1':>8}{'AUC':>9}")
    print("-" * W)
    _best = comparison.loc[comparison["f1_score"].idxmax(), "model"]
    for _, r in comparison.iterrows():
        mark = "  <- best" if r["model"] == _best else ""
        print(
            f"{str(r['model']):<20}{r['precision']:>10.3f}{r['recall']:>9.3f}"
            f"{r['f1_score']:>8.3f}{r['auc_roc']:>9.3f}{mark}"
        )
    print("=" * W)

    return results, latency_stats, comparison


def compute_metrics(y_true, y_pred, y_scores):
    """Compute evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except:
        auc_roc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "accuracy": float((tp + tn) / len(y_true)),
    }


if __name__ == "__main__":
    config = Config()
    results, latency, comparison = train_and_evaluate(config)
