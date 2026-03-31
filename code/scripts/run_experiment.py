"""
Main experiment runner - trains models, evaluates, and generates results.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import pandas as pd
from config import get_config
from data.feature_engineering import FeatureEngineer
from data.synthetic_generator import TransactionGenerator
from models.anomaly_detectors import (
    EnsembleDetector,
    IsolationForestDetector,
    XGBoostDetector,
    evaluate_detector,
)
from orchestrator.orchestrator import FraudDetectionOrchestrator


def generate_data(config, mode="quick"):
    """Generate synthetic transaction data."""
    print("=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)

    if mode == "quick":
        n_train, n_test = 5000, 1000
    else:
        n_train, n_test = 50000, 10000

    generator = TransactionGenerator(
        n_users=1000, random_state=config.model.random_state
    )

    train_df = generator.generate_dataset(n_transactions=n_train, fraud_ratio=0.02)
    test_df = generator.generate_dataset(n_transactions=n_test, fraud_ratio=0.02)

    print(
        f"Generated {len(train_df)} training transactions ({train_df['is_fraud'].sum()} fraud)"
    )
    print(
        f"Generated {len(test_df)} test transactions ({test_df['is_fraud'].sum()} fraud)"
    )

    # Save
    train_df.to_csv(config.data_dir / "transactions_train.csv", index=False)
    test_df.to_csv(config.data_dir / "transactions_test.csv", index=False)

    # Validate
    stats = generator.validate_distribution(train_df)
    with open(config.data_dir / "data_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Data saved to {config.data_dir}\n")

    return train_df, test_df


def train_baseline_models(config, train_df, test_df):
    """Train and evaluate baseline detectors."""
    print("=" * 60)
    print("TRAINING BASELINE MODELS")
    print("=" * 60)

    # Feature engineering
    fe = FeatureEngineer()
    X_train = fe.fit_transform(train_df)
    y_train = train_df["is_fraud"].values
    X_test = fe.transform(test_df)
    y_test = test_df["is_fraud"].values

    baselines = {}
    baseline_results = {}

    # Isolation Forest
    print("\n1. Isolation Forest (Unsupervised)")
    iso_detector = IsolationForestDetector(config)
    iso_detector.fit(X_train.values)
    iso_metrics = evaluate_detector(iso_detector, X_test.values, y_test)
    baselines["isolation_forest"] = iso_detector
    baseline_results["isolation_forest"] = iso_metrics
    print(
        f"   Precision: {iso_metrics['precision']:.3f}, Recall: {iso_metrics['recall']:.3f}, F1: {iso_metrics['f1_score']:.3f}, AUC: {iso_metrics['auc_roc']:.3f}"
    )

    # XGBoost
    print("\n2. XGBoost (Supervised)")
    xgb_detector = XGBoostDetector(config)
    xgb_detector.fit(X_train.values, y_train)
    xgb_metrics = evaluate_detector(xgb_detector, X_test.values, y_test)
    baselines["xgboost"] = xgb_detector
    baseline_results["xgboost"] = xgb_metrics
    print(
        f"   Precision: {xgb_metrics['precision']:.3f}, Recall: {xgb_metrics['recall']:.3f}, F1: {xgb_metrics['f1_score']:.3f}, AUC: {xgb_metrics['auc_roc']:.3f}"
    )

    # Ensemble
    print("\n3. Ensemble (Isolation Forest + XGBoost)")
    ensemble_detector = EnsembleDetector(config)
    ensemble_detector.fit(X_train.values, y_train)
    ensemble_metrics = evaluate_detector(ensemble_detector, X_test.values, y_test)
    baselines["ensemble"] = ensemble_detector
    baseline_results["ensemble"] = ensemble_metrics
    print(
        f"   Precision: {ensemble_metrics['precision']:.3f}, Recall: {ensemble_metrics['recall']:.3f}, F1: {ensemble_metrics['f1_score']:.3f}, AUC: {ensemble_metrics['auc_roc']:.3f}"
    )

    print("\nBaseline training complete.\n")

    return baselines, baseline_results


def run_full_pipeline(config, train_df, test_df):
    """Run full multi-agent pipeline."""
    print("=" * 60)
    print("RUNNING MULTI-AGENT PIPELINE")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = FraudDetectionOrchestrator(config)

    # Train
    orchestrator.train(train_df)

    # Detect on test set
    start_time = time.time()
    results_df = orchestrator.detect_batch(test_df)
    total_time = time.time() - start_time

    # Evaluate
    metrics = orchestrator.evaluate(test_df, results_df)

    print("\nMulti-Agent System Results:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  Detection Latency (mean): {metrics['detection_time_mean_ms']:.2f}ms")
    print(f"  Detection Latency (p95): {metrics['detection_time_p95_ms']:.2f}ms")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Throughput: {len(test_df)/total_time:.1f} tx/sec")

    return orchestrator, results_df, metrics


def save_results(config, baseline_results, multi_agent_metrics, test_df, results_df):
    """Save all experimental results."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Create results directories
    metrics_dir = config.results_dir / "metrics"
    logs_dir = config.results_dir / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save baseline results
    with open(metrics_dir / "baseline_metrics.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Save multi-agent results
    with open(metrics_dir / "multiagent_metrics.json", "w") as f:
        json.dump(multi_agent_metrics, f, indent=2)

    # Save detection results
    results_df.to_csv(logs_dir / "detection_results.csv", index=False)

    # Save ground truth for comparison
    test_df[["transaction_id", "user_id", "amount", "is_fraud", "fraud_type"]].to_csv(
        logs_dir / "ground_truth.csv", index=False
    )

    # Combined comparison table
    comparison = {
        "model": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "auc_roc": [],
    }

    for model_name, metrics in baseline_results.items():
        comparison["model"].append(model_name)
        comparison["precision"].append(metrics["precision"])
        comparison["recall"].append(metrics["recall"])
        comparison["f1_score"].append(metrics["f1_score"])
        comparison["auc_roc"].append(metrics["auc_roc"])

    comparison["model"].append("multi_agent_system")
    comparison["precision"].append(multi_agent_metrics["precision"])
    comparison["recall"].append(multi_agent_metrics["recall"])
    comparison["f1_score"].append(multi_agent_metrics["f1_score"])
    comparison["auc_roc"].append(multi_agent_metrics["auc_roc"])

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

    print(f"\nResults saved to {config.results_dir}")
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Run fraud detection experiment")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Experiment mode (quick=5k train, full=50k train)",
    )
    args = parser.parse_args()

    # Load config
    config = get_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FRAUD DETECTION MULTI-AGENT EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Random seed: {config.model.random_state}")
    print(f"Results directory: {config.results_dir}")
    print("=" * 60 + "\n")

    # Step 1: Generate data
    train_df, test_df = generate_data(config, mode=args.mode)

    # Step 2: Train baselines
    baselines, baseline_results = train_baseline_models(config, train_df, test_df)

    # Step 3: Run full multi-agent pipeline
    orchestrator, results_df, multi_agent_metrics = run_full_pipeline(
        config, train_df, test_df
    )

    # Step 4: Save results
    save_results(config, baseline_results, multi_agent_metrics, test_df, results_df)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Generate figures: python eval/generate_figures.py")
    print("  2. View results: ls results/metrics/")
    print("  3. Check logs: ls results/logs/")


if __name__ == "__main__":
    main()
