"""
Orchestrator coordinates all agents in the fraud detection pipeline.
"""

import sys

sys.path.append("..")

import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from agents.llm_agents import EvidenceAggregator, NarrativeGenerator
from agents.privacy_guard import PrivacyGuard
from config import get_config
from data.feature_engineering import FeatureEngineer
from models.anomaly_detectors import (
    EnsembleDetector,
    IsolationForestDetector,
    XGBoostDetector,
)


class FraudDetectionOrchestrator:
    """Coordinates multi-agent fraud detection pipeline."""

    def __init__(self, config=None):
        self.config = config or get_config()

        # Initialize agents
        self.feature_engineer = FeatureEngineer()
        self.privacy_guard = PrivacyGuard(self.config)
        self.evidence_aggregator = EvidenceAggregator(self.config)
        self.narrative_generator = NarrativeGenerator(self.config)

        # Initialize detectors
        self.detectors = {
            "isolation_forest": IsolationForestDetector(self.config),
            "xgboost": XGBoostDetector(self.config),
            "ensemble": EnsembleDetector(self.config),
        }

        # Tracking
        self.detection_times = []
        self.case_reports = []

    def train(self, train_df: pd.DataFrame):
        """Train all detectors on training data."""
        print("Training fraud detection models...")

        # Feature engineering
        self.feature_engineer.fit(train_df)
        X_train = self.feature_engineer.transform(train_df)
        y_train = train_df["is_fraud"].values

        # Train each detector
        for name, detector in self.detectors.items():
            print(f"  Training {name}...")
            start = time.time()

            if name in ["xgboost", "ensemble"]:
                detector.fit(X_train.values, y_train)
            else:
                detector.fit(X_train.values)

            elapsed = time.time() - start
            print(f"    Trained in {elapsed:.2f}s")

        print("Training complete.\n")

    def detect_single(self, transaction: Dict) -> Dict[str, Any]:
        """
        Process a single transaction through the detection pipeline.

        Returns:
            Detection result with evidence and narrative
        """
        start_time = time.time()

        # Step 1: Privacy Guard - redact PII
        redacted_tx = self.privacy_guard.redact_transaction(transaction)

        # Step 2: Feature engineering
        tx_df = pd.DataFrame([redacted_tx])
        features = self.feature_engineer.transform(tx_df)

        # Step 3: Run all detectors
        detector_scores = {}
        for name, detector in self.detectors.items():
            score = detector.predict_proba(features.values)[0]
            detector_scores[name] = float(score)

        # Step 4: Evidence aggregation
        feature_dict = features.iloc[0].to_dict()
        evidence = self.evidence_aggregator.aggregate(
            redacted_tx, detector_scores, feature_dict
        )

        # Step 5: Narrative generation (if flagged)
        narrative = None
        if evidence["fraud_score"] > 0.5:
            narrative = self.narrative_generator.generate_narrative(
                evidence, redacted_tx
            )

        # Step 6: Privacy Guard gates
        review_gate = self.privacy_guard.apply_investigator_review_gate(
            evidence["fraud_score"], evidence
        )

        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)

        result = {
            "transaction_id": transaction["transaction_id"],
            "fraud_score": evidence["fraud_score"],
            "risk_level": evidence["risk_level"],
            "is_flagged": evidence["fraud_score"] > 0.5,
            "score_isolation_forest": detector_scores.get("isolation_forest", 0.0),
            "score_xgboost": detector_scores.get("xgboost", 0.0),
            "score_ensemble": detector_scores.get("ensemble", 0.0),
            "narrative": narrative,
            "review_required": review_gate["review_required"],
            "detection_time_ms": detection_time * 1000,
            "processed_at": datetime.now().isoformat(),
            # Keep the full nested dicts available for callers that need them,
            # but they are not relied upon by evaluate().
            "_evidence": evidence,
            "_detector_scores": detector_scores,
        }

        return result

    def detect_batch(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process batch of transactions.

        Returns:
            DataFrame with detection results
        """
        print(f"Processing {len(transactions_df)} transactions...")

        results = []
        for row_num, (idx, row) in enumerate(transactions_df.iterrows(), start=1):
            tx_dict = row.to_dict()
            result = self.detect_single(tx_dict)
            results.append(result)

            if row_num % 1000 == 0:
                print(f"  Processed {row_num}/{len(transactions_df)}")

        results_df = pd.DataFrame(results)
        print("Batch processing complete.\n")

        return results_df

    def evaluate(
        self, test_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate detection performance.

        Returns:
            Metrics dictionary
        """
        y_true = test_df["is_fraud"].values
        y_pred = results_df["is_flagged"].values.astype(int)
        y_scores = results_df["fraud_score"].values

        # Import metrics
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_curve,
            precision_recall_fscore_support,
            roc_auc_score,
            roc_curve,
        )

        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        auc_roc = roc_auc_score(y_true, y_scores)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Detection latency stats
        detection_times = results_df["detection_time_ms"].values

        # Curves
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc_roc),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "accuracy": float((tp + tn) / len(y_true)),
            "detection_time_mean_ms": float(detection_times.mean()),
            "detection_time_median_ms": float(np.median(detection_times)),
            "detection_time_p95_ms": float(np.percentile(detection_times, 95)),
            "detection_time_p99_ms": float(np.percentile(detection_times, 99)),
            "curves": {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                "pr": {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()},
            },
        }

        return metrics

    def get_latency_stats(self) -> Dict[str, float]:
        """Get detection latency statistics."""
        times_ms = np.array(self.detection_times) * 1000

        return {
            "mean_ms": float(times_ms.mean()),
            "median_ms": float(np.median(times_ms)),
            "std_ms": float(times_ms.std()),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
        }
