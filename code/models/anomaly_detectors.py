"""
ML-based anomaly detection models for fraud detection.
Implements unsupervised and supervised detectors.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb
from typing import Dict
import joblib


class AnomalyDetector:
    """Base class for anomaly detection models."""

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Train the detector."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1=fraud, 0=normal)."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probability scores."""
        raise NotImplementedError


class IsolationForestDetector(AnomalyDetector):
    """Unsupervised isolation forest anomaly detector."""

    def __init__(self, config):
        super().__init__(config)
        self.model = IsolationForest(
            contamination=config.model.isolation_forest_contamination,
            random_state=config.model.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Train isolation forest (unsupervised)."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Convert -1/1 to 0/1 (1=anomaly)
        return (predictions == -1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)  # Negative score = anomaly
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores


class XGBoostDetector(AnomalyDetector):
    """Supervised XGBoost classifier for fraud detection."""

    def __init__(self, config):
        super().__init__(config)
        # Handle class imbalance
        self.model = xgb.XGBClassifier(
            n_estimators=config.model.xgboost_n_estimators,
            max_depth=config.model.xgboost_max_depth,
            learning_rate=0.1,
            random_state=config.model.random_state,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost classifier with class weights."""
        X_scaled = self.scaler.fit_transform(X)

        # Compute class weights for imbalance
        n_samples = len(y)
        n_fraud = y.sum()
        n_normal = n_samples - n_fraud

        if n_fraud > 0:
            scale_pos_weight = n_normal / n_fraud
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(X_scaled, y)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get fraud probability scores."""
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]  # Probability of fraud class

    def get_feature_importance(self, feature_names) -> Dict[str, float]:
        """Get feature importance scores."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))


class EnsembleDetector(AnomalyDetector):
    """Ensemble of multiple detectors."""

    def __init__(self, config, detectors=None):
        super().__init__(config)
        if detectors is None:
            self.detectors = [IsolationForestDetector(config), XGBoostDetector(config)]
        else:
            self.detectors = detectors

        self.weights = [0.3, 0.7]  # XGBoost gets more weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all detectors in ensemble."""
        for detector in self.detectors:
            if isinstance(detector, XGBoostDetector):
                detector.fit(X, y)
            else:
                detector.fit(X)  # Unsupervised
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction via weighted voting."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability as weighted average."""
        scores = np.zeros(len(X))

        for detector, weight in zip(self.detectors, self.weights):
            scores += weight * detector.predict_proba(X)

        return scores / sum(self.weights)


def evaluate_detector(
    detector: AnomalyDetector, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate detector performance.

    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    try:
        auc_roc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc_roc = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "support_fraud": int(y_test.sum()),
        "support_normal": int(len(y_test) - y_test.sum()),
    }

    return metrics


def save_model(detector: AnomalyDetector, path: str):
    """Save trained detector to disk."""
    joblib.dump(detector, path)


def load_model(path: str) -> AnomalyDetector:
    """Load trained detector from disk."""
    return joblib.load(path)
