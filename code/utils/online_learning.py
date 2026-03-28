"""
Online learning and model updating strategy for fraud detection.
Handles concept drift, periodic retraining, A/B testing, and adaptive learning.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import joblib
from pathlib import Path
import json
from collections import deque


class OnlineLearningManager:
    """
    Manages online learning and model updates for fraud detection.

    Features:
    - Incremental learning for adapting to new patterns
    - Concept drift detection
    - Periodic retraining schedule
    - A/B testing framework
    - Performance monitoring
    """

    def __init__(
        self,
        model,
        window_size: int = 10000,
        drift_threshold: float = 0.05,
        retrain_frequency_days: int = 7,
        min_samples_retrain: int = 1000,
    ):
        """
        Initialize online learning manager.

        Args:
            model: Base model for fraud detection
            window_size: Size of sliding window for drift detection
            drift_threshold: Threshold for detecting concept drift
            retrain_frequency_days: Days between scheduled retraining
            min_samples_retrain: Minimum new samples before retraining
        """
        self.model = model
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.retrain_frequency_days = retrain_frequency_days
        self.min_samples_retrain = min_samples_retrain

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.drift_detected = False
        self.last_retrain_date = datetime.now()

        # Data buffers
        self.buffer_X = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
        self.buffer_predictions = deque(maxlen=window_size)

        # Versioning
        self.model_version = "1.0.0"
        self.version_history = []

        print(f"Online Learning Manager Initialized:")
        print(f"  Window Size: {window_size}")
        print(f"  Drift Threshold: {drift_threshold}")
        print(f"  Retrain Frequency: {retrain_frequency_days} days")

    def predict_and_learn(
        self, X: np.ndarray, y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make prediction and optionally learn from feedback.

        Args:
            X: Feature matrix
            y_true: True labels (if available for learning)

        Returns:
            Predictions
        """
        # Make prediction
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # Store in buffer
        for i in range(len(X)):
            self.buffer_X.append(X[i])
            if y_true is not None:
                self.buffer_y.append(y_true[i])
            self.buffer_predictions.append(y_proba[i])

        # Track performance if labels available
        if y_true is not None:
            self._update_performance_metrics(y_true, y_pred)

            # Check for concept drift
            if self._detect_drift():
                print("⚠️  Concept drift detected! Triggering retraining...")
                self.trigger_retrain(reason="drift")

        # Check if scheduled retrain is due
        if self._is_retrain_due():
            print("📅 Scheduled retrain is due...")
            self.trigger_retrain(reason="scheduled")

        return y_pred

    def _update_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update running performance metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "n_samples": len(y_true),
        }

        self.performance_history.append(metrics)

    def _detect_drift(self) -> bool:
        """
        Detect concept drift using performance degradation.

        Returns:
            True if drift detected
        """
        if len(self.performance_history) < 10:
            return False

        # Compare recent performance to baseline
        recent_metrics = list(self.performance_history)[-10:]
        baseline_metrics = list(self.performance_history)[:10]

        recent_f1 = np.mean([m["f1_score"] for m in recent_metrics])
        baseline_f1 = np.mean([m["f1_score"] for m in baseline_metrics])

        # Check for significant degradation
        degradation = baseline_f1 - recent_f1

        if degradation > self.drift_threshold:
            self.drift_detected = True
            print(f"  Baseline F1: {baseline_f1:.3f}")
            print(f"  Recent F1: {recent_f1:.3f}")
            print(f"  Degradation: {degradation:.3f}")
            return True

        return False

    def _is_retrain_due(self) -> bool:
        """Check if scheduled retrain is due."""
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        has_enough_samples = len(self.buffer_y) >= self.min_samples_retrain

        return days_since_retrain >= self.retrain_frequency_days and has_enough_samples

    def trigger_retrain(self, reason: str = "scheduled"):
        """
        Trigger model retraining with buffered data.
        """
        if len(self.buffer_X) < self.min_samples_retrain:
            print(
                f"  Insufficient samples for retraining ({len(self.buffer_X)} < {self.min_samples_retrain})"
            )
            return

        print(f"\n{'='*60}")
        print(f"RETRAINING MODEL")
        print(f"{'='*60}")
        print(f"  Buffer Size: {len(self.buffer_X)}")
        print(f"  Days Since Last: {(datetime.now() - self.last_retrain_date).days}")
        print(f"  Reason: {reason}")

        # Convert buffer to arrays
        X_new = np.array(list(self.buffer_X))
        y_new = np.array(list(self.buffer_y))

        # Retrain model
        print(f"  Training with {len(X_new)} samples...")
        self.model.fit(X_new, y_new)

        # Update metadata — record reason BEFORE resetting drift flag
        self.last_retrain_date = datetime.now()
        self.drift_detected = False  # safe to reset now

        # Increment version
        version_parts = self.model_version.split(".")
        version_parts[1] = str(int(version_parts[1]) + 1)
        self.model_version = ".".join(version_parts)

        self.version_history.append(
            {
                "version": self.model_version,
                "date": self.last_retrain_date.isoformat(),
                "samples": len(X_new),
                "reason": reason,
            }
        )

        print(f"  ✓ Model retrained successfully")
        print(f"  New version: {self.model_version}")
        print(f"{'='*60}\n")

    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        if not self.performance_history:
            return {"status": "No data available"}

        recent = list(self.performance_history)[-50:]  # Last 50 batches

        report = {
            "current_version": self.model_version,
            "last_retrain": self.last_retrain_date.isoformat(),
            "buffer_size": len(self.buffer_X),
            "metrics": {
                "precision": {
                    "mean": float(np.mean([m["precision"] for m in recent])),
                    "std": float(np.std([m["precision"] for m in recent])),
                    "min": float(np.min([m["precision"] for m in recent])),
                    "max": float(np.max([m["precision"] for m in recent])),
                },
                "recall": {
                    "mean": float(np.mean([m["recall"] for m in recent])),
                    "std": float(np.std([m["recall"] for m in recent])),
                    "min": float(np.min([m["recall"] for m in recent])),
                    "max": float(np.max([m["recall"] for m in recent])),
                },
                "f1_score": {
                    "mean": float(np.mean([m["f1_score"] for m in recent])),
                    "std": float(np.std([m["f1_score"] for m in recent])),
                    "min": float(np.min([m["f1_score"] for m in recent])),
                    "max": float(np.max([m["f1_score"] for m in recent])),
                },
            },
            "version_history": self.version_history,
        }

        return report

    def save_state(self, output_dir: Path):
        """Save model and learning state."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f"model_v{self.model_version}.pkl"
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata = {
            "version": self.model_version,
            "last_retrain": self.last_retrain_date.isoformat(),
            "version_history": self.version_history,
            "performance_history": list(self.performance_history),
        }

        with open(output_dir / "online_learning_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model state saved to {output_dir}/")


class ABTestingFramework:
    """
    A/B testing framework for fraud detection models.

    Allows safe deployment and comparison of model versions.
    """

    def __init__(
        self, model_a, model_b, traffic_split: float = 0.5, min_samples: int = 1000
    ):
        """
        Initialize A/B testing framework.

        Args:
            model_a: Control model (current production)
            model_b: Treatment model (new version)
            traffic_split: Fraction of traffic to model B (0-1)
            min_samples: Minimum samples before statistical test
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.min_samples = min_samples

        # Results tracking
        self.results_a = {"y_true": [], "y_pred": [], "y_proba": []}
        self.results_b = {"y_true": [], "y_pred": [], "y_proba": []}

        print(f"A/B Testing Framework Initialized:")
        print(f"  Traffic Split: {traffic_split*100:.0f}% to Model B")
        print(f"  Min Samples: {min_samples}")

    def predict(
        self, X: np.ndarray, user_id: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Route prediction to appropriate model based on A/B split.

        Args:
            X: Features
            user_id: Optional user ID for consistent routing

        Returns:
            predictions, model_used
        """
        # Determine which model to use
        if user_id:
            # Consistent hashing for user
            hash_val = hash(user_id) % 100
            use_model_b = (hash_val / 100) < self.traffic_split
        else:
            # Random routing
            use_model_b = np.random.random() < self.traffic_split

        if use_model_b:
            predictions = self.model_b.predict(X)
            model_used = "model_b"
        else:
            predictions = self.model_a.predict(X)
            model_used = "model_a"

        return predictions, model_used

    def record_result(
        self,
        model_used: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ):
        """Record results for analysis."""
        if model_used == "model_a":
            self.results_a["y_true"].extend(y_true)
            self.results_a["y_pred"].extend(y_pred)
            self.results_a["y_proba"].extend(y_proba)
        else:
            self.results_b["y_true"].extend(y_true)
            self.results_b["y_pred"].extend(y_pred)
            self.results_b["y_proba"].extend(y_proba)

    def analyze_results(self) -> Dict:
        """
        Analyze A/B test results with statistical significance.

        Returns:
            Dictionary with comparison results
        """
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

        if len(self.results_a["y_true"]) < self.min_samples:
            return {"status": "Insufficient samples for Model A"}

        if len(self.results_b["y_true"]) < self.min_samples:
            return {"status": "Insufficient samples for Model B"}

        print(f"\n{'='*60}")
        print(f"A/B TEST RESULTS")
        print(f"{'='*60}")

        # Calculate metrics for both models
        results = {}

        for model_name, data in [
            ("Model A", self.results_a),
            ("Model B", self.results_b),
        ]:
            y_true = np.array(data["y_true"])
            y_pred = np.array(data["y_pred"])
            y_proba = np.array(data["y_proba"])

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )

            try:
                auc = roc_auc_score(y_true, y_proba)
            except Exception:
                auc = 0.0

            results[model_name] = {
                "n_samples": len(y_true),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc_roc": float(auc),
            }

            print(f"\n{model_name}:")
            print(f"  Samples: {len(y_true)}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  AUC-ROC: {auc:.3f}")

        # Statistical significance test (F1 scores)
        f1_a = results["Model A"]["f1_score"]
        f1_b = results["Model B"]["f1_score"]

        # Simple z-test for proportions
        # (In production, use bootstrap or permutation test)
        improvement = ((f1_b - f1_a) / f1_a) * 100 if f1_a > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"COMPARISON:")
        print(f"  F1 Improvement: {improvement:+.2f}%")

        # Decision
        if f1_b > f1_a + 0.02:  # 2% improvement threshold
            recommendation = "PROMOTE Model B to production"
        elif f1_b < f1_a - 0.02:
            recommendation = "KEEP Model A, Model B performs worse"
        else:
            recommendation = "INCONCLUSIVE, continue testing"

        print(f"  Recommendation: {recommendation}")
        print(f"{'='*60}\n")

        results["improvement_percent"] = improvement
        results["recommendation"] = recommendation

        return results

    def save_results(self, output_dir: Path):
        """Save A/B test results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.analyze_results()

        with open(output_dir / "ab_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ A/B test results saved to {output_dir}/")


def create_model_updating_documentation(output_dir: Path):
    """Create documentation on model updating strategy."""

    doc = """
# Model Updating Strategy - Technical Documentation

## Overview

Fraud patterns evolve rapidly. A static model will degrade over time due to:
- **Concept Drift**: Fraudsters adapt tactics
- **Data Distribution Shift**: Customer behavior changes
- **Seasonal Patterns**: Holiday spending, etc.
- **New Attack Vectors**: Emerging fraud types

This system implements adaptive learning to maintain performance.

## Update Strategies

### 1. Online Learning (Incremental)

**When to Use**: High-frequency updates, streaming data

**Algorithm**:
```python
# Sliding window approach
window_size = 10,000  # Recent transactions
for batch in transaction_stream:
    predictions = model.predict(batch)
    
    # Wait for labels (fraud confirmation)
    true_labels = get_confirmed_labels(batch)
    
    # Incremental update
    model.partial_fit(batch, true_labels)
    
    # Monitor performance
    if detect_drift():
        trigger_full_retrain()
```

**Pros**:
- Adapts quickly to new patterns
- Low latency
- Continuous improvement

**Cons**:
- Risk of catastrophic forgetting
- Requires careful monitoring
- May be unstable

### 2. Periodic Retraining

**When to Use**: Stable deployment, scheduled maintenance

**Schedule**:
- **Daily**: For high-volume environments (>1M txn/day)
- **Weekly**: Standard recommendation
- **Monthly**: Low-volume or stable environments

**Process**:
```
1. Collect data from last period (e.g., 7 days)
2. Label confirmed fraud cases
3. Combine with historical data (weighted)
4. Retrain model on combined dataset
5. Validate on hold-out set
6. Deploy new version via A/B test
7. Monitor for 24-48 hours
8. Full rollout if successful
```

**Data Mixing Strategy**:
```python
# Recent data gets higher weight
recent_data = last_7_days
historical_data = last_90_days

# Sample historical data to avoid over-representation
historical_sample = stratified_sample(
    historical_data,
    n_samples=len(recent_data) * 2
)

# Combine with weights
training_data = combine(
    recent_data (weight=0.6),
    historical_sample (weight=0.4)
)
```

### 3. Trigger-Based Retraining

**Triggers**:

1. **Performance Degradation**:
   ```python
   if current_f1 < baseline_f1 - 0.05:
       trigger_retrain()
   ```

2. **Concept Drift Detection**:
   - Monitor prediction distribution
   - Track feature drift
   - Alert if significant change

3. **New Fraud Pattern**:
   - Security team identifies new attack
   - Immediate retraining with labeled examples

4. **Seasonal Events**:
   - Black Friday, holidays
   - Pre-train model for expected patterns

## A/B Testing Protocol

### Setup

```python
# Route traffic to models
traffic_split = {
    'model_a': 0.8,  # Current production (80%)
    'model_b': 0.2   # New candidate (20%)
}

# Consistent routing per user
def route_request(user_id):
    hash_val = hash(user_id) % 100
    if hash_val < 20:
        return 'model_b'
    else:
        return 'model_a'
```

### Monitoring

Track these metrics:
- **Performance**: Precision, Recall, F1, AUC
- **Latency**: P50, P95, P99 response times
- **Costs**: False positive/negative costs
- **User Experience**: Customer complaints, reviews

### Decision Criteria

Promote Model B if:
1. F1 score improved by >=2%
2. No significant latency increase
3. No customer experience degradation
4. Statistical significance (p < 0.05)
5. Minimum 1000 samples per model

### Rollback Plan

If Model B underperforms:
1. Immediate rollback to Model A
2. Root cause analysis
3. Fix issues
4. Re-test before next deployment

## Monitoring Dashboard

Key metrics to track:

### Model Performance
- **Precision**: Track daily average
- **Recall**: Alert if drops >5%
- **F1 Score**: Primary metric
- **AUC-ROC**: Overall discrimination

### Data Quality
- **Feature Distribution**: Detect drift
- **Missing Values**: Track increase
- **Outliers**: Monitor anomalies

### Business Metrics
- **False Positive Rate**: Cost impact
- **False Negative Rate**: Fraud loss
- **Investigation Queue**: Workload
- **Customer Satisfaction**: Feedback score

### System Health
- **Prediction Latency**: P95, P99
- **Throughput**: Transactions/second
- **Error Rate**: Failed predictions
- **Resource Usage**: CPU, memory

## Alerting Rules

### Critical (Immediate Action)
- F1 score drops >10%
- Error rate >5%
- Latency P95 >500ms

### Warning (Review in 1 hour)
- F1 score drops 5-10%
- Feature drift detected
- Investigation queue >2x normal

### Info (Daily Review)
- F1 score drops <5%
- Minor latency increase
- Data quality issues

## Retraining Schedule

### Standard Weekly Schedule

**Monday**: Data collection review
**Wednesday**: Model training
**Thursday**: A/B test deployment (20% traffic)
**Friday**: Monitoring and validation
**Monday**: Full rollout if successful

### Emergency Retraining

When immediate action needed:
1. Collect minimum 1000 labeled samples
2. Fast-track training (4 hours)
3. Validate on hold-out set
4. Deploy to 10% traffic
5. Monitor closely for 2 hours
6. Full rollout if stable

## Version Control

Track model versions:

```python
version_format = "MAJOR.MINOR.PATCH"

# MAJOR: Architecture change
# MINOR: Retraining on new data
# PATCH: Hyperparameter tuning

# Example progression:
# 1.0.0 -> Initial production model
# 1.1.0 -> Weekly retrain
# 1.2.0 -> Drift-triggered retrain
# 2.0.0 -> New model architecture
```

## Best Practices

1. **Never skip validation**: Always test before production
2. **Keep baseline**: Maintain comparison to original model
3. **Document changes**: Track what triggered retraining
4. **Gradual rollout**: Use A/B testing, not big bang
5. **Have rollback plan**: Be able to revert quickly
6. **Monitor closely**: First 24-48 hours are critical
7. **Customer communication**: Notify of major changes

## Tools & Infrastructure

### Required Tools
- **Version Control**: Git for code, MLflow for models
- **Monitoring**: Prometheus, Grafana dashboards
- **Alerting**: PagerDuty, Slack notifications
- **A/B Testing**: Feature flags, traffic routing
- **Data Pipeline**: Airflow, Kafka for streaming

### Infrastructure
- **Training**: GPU instances for faster retraining
- **Serving**: Load-balanced prediction API
- **Storage**: S3/GCS for model artifacts
- **Database**: PostgreSQL for metrics logging

---

Generated by Multi-Agent Fraud Detection System
"""

    output_file = output_dir / "MODEL_UPDATING_STRATEGY.md"
    with open(output_file, "w") as f:
        f.write(doc)

    print(f"✓ Model updating documentation saved to {output_file}")


if __name__ == "__main__":
    print("Online Learning and Model Updating Module")
    print("Run from experiments to see results")
