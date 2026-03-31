"""
Enhanced experiment runner with advanced features:
- Class imbalance handling comparison
- Cost-benefit analysis
- Advanced visualizations
- Production-ready evaluation
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from config import get_config
from data.feature_engineering import FeatureEngineer
from data.synthetic_generator import generate_synthetic_fraud_data
from eval.generate_figures import main as generate_basic_figures
from models.anomaly_detectors import evaluate_detector
from orchestrator.orchestrator import FraudDetectionOrchestrator

# Import new modules
try:
    from eval.advanced_visualizations import generate_all_advanced_figures
    from utils.cost_benefit_analysis import CostBenefitAnalyzer
    from utils.imbalance_handling import (
        ImbalanceComparison,
        create_imbalance_analysis_documentation,
    )
    from utils.online_learning import create_model_updating_documentation

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("⚠️  Advanced features not available. Install required dependencies.")


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Fraud Detection Experiments")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "advanced"],
        default="full",
        help="Experiment mode: quick (fast), full (standard), advanced (all features)",
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip data generation if data already exists",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Output directory for results",
    )
    return parser.parse_args()


def setup_directories(output_dir: str):
    """Create output directories."""
    base_dir = Path(output_dir)

    dirs = {
        "base": base_dir,
        "metrics": base_dir / "metrics",
        "models": base_dir / "models",
        "figures": base_dir.parent / "figures",
        "reports": base_dir / "reports",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def run_quick_experiment(config, dirs):
    """Run quick experiment for fast validation."""
    print("\n" + "=" * 70)
    print("QUICK EXPERIMENT MODE")
    print("=" * 70 + "\n")

    # Generate small dataset
    print("Generating small dataset...")
    df = generate_synthetic_fraud_data(
        n_samples=10000, fraud_ratio=0.02, random_state=config.model.random_state
    )

    # Split data
    train_size = int(len(df) * 0.7)
    test_size = int(len(df) * 0.15)

    train_df = df[:train_size]
    test_df = df[train_size : train_size + test_size]

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    # Train and evaluate
    orchestrator = FraudDetectionOrchestrator(config)
    orchestrator.train(train_df)

    results_df = orchestrator.detect_batch(test_df)
    metrics = orchestrator.evaluate(test_df, results_df)

    # Save results
    with open(dirs["metrics"] / "quick_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✓ Quick experiment completed!")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")

    return metrics


def run_full_experiment(config, dirs, skip_data_gen=False):
    """Run full experiment with all baseline models."""
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT MODE")
    print("=" * 70 + "\n")

    # Data generation
    data_path = Path("../data/synthetic_fraud_data.csv")

    if skip_data_gen and data_path.exists():
        print("Loading existing dataset...")
        df = pd.read_csv(data_path)
    else:
        print("Generating full dataset...")
        df = generate_synthetic_fraud_data(
            n_samples=100000, fraud_ratio=0.02, random_state=config.model.random_state
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)

    print(f"Dataset size: {len(df)}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")

    # Split data
    train_size = int(len(df) * config.train_ratio)
    val_size = int(len(df) * config.val_ratio)

    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size :]

    print("\nData splits:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df):.1%})")

    # Train orchestrator
    print("\n" + "-" * 70)
    orchestrator = FraudDetectionOrchestrator(config)
    orchestrator.train(train_df)

    # Evaluate individual models
    print("\n" + "-" * 70)
    print("EVALUATING INDIVIDUAL MODELS")
    print("-" * 70 + "\n")

    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_df)
    X_test = feature_engineer.transform(test_df)
    y_test = test_df["is_fraud"].values

    baseline_metrics = {}
    for name, detector in orchestrator.detectors.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_detector(detector, X_test.values, y_test)
        baseline_metrics[name] = metrics

        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}\n")

    # Evaluate full multi-agent system
    print("-" * 70)
    print("EVALUATING MULTI-AGENT SYSTEM")
    print("-" * 70 + "\n")

    results_df = orchestrator.detect_batch(test_df)
    system_metrics = orchestrator.evaluate(test_df, results_df)

    print("Multi-Agent System Performance:")
    print(f"  Precision: {system_metrics['precision']:.3f}")
    print(f"  Recall: {system_metrics['recall']:.3f}")
    print(f"  F1 Score: {system_metrics['f1_score']:.3f}")
    print(f"  AUC-ROC: {system_metrics['auc_roc']:.3f}")
    print(f"  Mean Latency: {system_metrics['detection_time_mean_ms']:.2f}ms")
    print(f"  P95 Latency: {system_metrics['detection_time_p95_ms']:.2f}ms")

    # Save metrics
    with open(dirs["metrics"] / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    with open(dirs["metrics"] / "system_metrics.json", "w") as f:
        json.dump(system_metrics, f, indent=2)

    # Create model comparison CSV
    comparison_data = []
    for name, metrics in baseline_metrics.items():
        comparison_data.append(
            {
                "model": name.replace("_", " ").title(),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics["auc_roc"],
            }
        )

    comparison_data.append(
        {
            "model": "Multi-Agent System",
            "precision": system_metrics["precision"],
            "recall": system_metrics["recall"],
            "f1_score": system_metrics["f1_score"],
            "auc_roc": system_metrics["auc_roc"],
        }
    )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(dirs["metrics"] / "model_comparison.csv", index=False)

    # Latency stats
    latency_stats = orchestrator.get_latency_stats()
    with open(dirs["metrics"] / "latency_stats.json", "w") as f:
        json.dump(latency_stats, f, indent=2)

    print("\n✓ Full experiment completed!")

    return baseline_metrics, system_metrics, test_df, results_df


def run_advanced_experiment(config, dirs, skip_data_gen=False):
    """Run advanced experiment with all new features."""
    if not ADVANCED_FEATURES_AVAILABLE:
        print("❌ Advanced features not available. Running full experiment instead...")
        return run_full_experiment(config, dirs, skip_data_gen)

    print("\n" + "=" * 70)
    print("ADVANCED EXPERIMENT MODE - ALL FEATURES")
    print("=" * 70 + "\n")

    # Run full experiment first
    baseline_metrics, system_metrics, test_df, results_df = run_full_experiment(
        config, dirs, skip_data_gen
    )

    # Load data
    data_path = Path("../data/synthetic_fraud_data.csv")
    df = pd.read_csv(data_path)

    train_size = int(len(df) * config.train_ratio)
    val_size = int(len(df) * config.val_ratio)

    train_df = df[:train_size]
    test_df_full = df[train_size + val_size :]

    # Feature engineering
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_df)
    X_train = feature_engineer.transform(train_df).values
    y_train = train_df["is_fraud"].values
    X_test = feature_engineer.transform(test_df_full).values
    y_test = test_df_full["is_fraud"].values

    # 1. CLASS IMBALANCE COMPARISON
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE HANDLING COMPARISON")
    print("=" * 70 + "\n")

    from xgboost import XGBClassifier

    imbalance_comp = ImbalanceComparison(random_state=config.model.random_state)
    imbalance_results = imbalance_comp.compare_techniques(
        X_train,
        y_train,
        X_test,
        y_test,
        model_class=XGBClassifier,
        model_params={
            "n_estimators": config.model.xgboost_n_estimators,
            "max_depth": config.model.xgboost_max_depth,
            "learning_rate": 0.1,
            "eval_metric": "aucpr",
        },
    )

    imbalance_comp.save_results(dirs["metrics"])
    report = imbalance_comp.generate_report()
    print(report)

    with open(dirs["reports"] / "imbalance_comparison.txt", "w") as f:
        f.write(report)

    # Create documentation
    create_imbalance_analysis_documentation(Path("../docs"))

    # 2. COST-BENEFIT ANALYSIS
    print("\n" + "=" * 70)
    print("COST-BENEFIT ANALYSIS")
    print("=" * 70 + "\n")

    cost_analyzer = CostBenefitAnalyzer(
        fp_cost=50.0,
        fn_cost=500.0,
        avg_transaction_value=150.0,
        monthly_transactions=1_000_000,
    )

    # Baseline costs (assume poor performance without system)
    y_pred_baseline = np.zeros_like(y_test)  # Predict all normal
    np.random.random(len(y_test)) * 0.3  # Low confidence

    baseline_costs = cost_analyzer.calculate_costs(y_test, y_pred_baseline)

    # System costs
    y_pred_system = results_df["is_flagged"].values[: len(y_test)]
    y_proba_system = results_df["fraud_score"].values[: len(y_test)]

    cost_analyzer.calculate_costs(y_test, y_pred_system)

    # Find optimal threshold
    optimal_threshold, optimal_costs = cost_analyzer.find_optimal_threshold(
        y_test, y_proba_system
    )

    # Threshold sensitivity
    sensitivity_df = cost_analyzer.threshold_sensitivity_analysis(
        y_test, y_proba_system
    )
    sensitivity_df.to_csv(dirs["metrics"] / "threshold_sensitivity.csv", index=False)

    # ROI calculation
    roi_metrics = cost_analyzer.calculate_roi(
        baseline_costs,
        optimal_costs,
        implementation_cost=300_000,
        annual_maintenance=100_000,
    )

    with open(dirs["metrics"] / "roi_analysis.json", "w") as f:
        json.dump(roi_metrics, f, indent=2)

    # Generate business report
    cost_analyzer.generate_business_report(
        baseline_costs,
        optimal_costs,
        roi_metrics,
        output_path=dirs["reports"] / "business_cost_benefit_report.txt",
    )

    cost_analyzer.save_analysis(dirs["metrics"])

    # 3. MODEL UPDATING DOCUMENTATION
    print("\n" + "=" * 70)
    print("GENERATING DOCUMENTATION")
    print("=" * 70 + "\n")

    create_model_updating_documentation(Path("../docs"))

    # 4. ADVANCED VISUALIZATIONS
    print("\n" + "=" * 70)
    print("GENERATING ADVANCED VISUALIZATIONS")
    print("=" * 70 + "\n")

    generate_all_advanced_figures(baseline_metrics, system_metrics, dirs["figures"])

    print("\n" + "=" * 70)
    print("ADVANCED EXPERIMENT COMPLETED!")
    print("=" * 70)
    print(f"\nResults saved to: {dirs['base']}")
    print(f"Figures saved to: {dirs['figures']}")
    print(f"Reports saved to: {dirs['reports']}")
    print("\nKey Findings:")
    print(f"  • Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  • Annual Savings: ${roi_metrics['annual_fraud_savings']:,.0f}")
    print(f"  • First Year ROI: {roi_metrics['first_year_roi_percent']:.1f}%")
    print(f"  • Payback Period: {roi_metrics['payback_period_months']:.1f} months")

    return {
        "baseline_metrics": baseline_metrics,
        "system_metrics": system_metrics,
        "imbalance_results": imbalance_results,
        "cost_analysis": roi_metrics,
        "optimal_threshold": optimal_threshold,
    }


def main():
    args = parse_args()

    print("=" * 70)
    print("MULTI-AGENT FRAUD DETECTION SYSTEM")
    print("Enhanced Experiment Runner")
    print("=" * 70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Setup
    config = get_config()
    dirs = setup_directories(args.output_dir)

    # Run experiment based on mode
    start_time = time.time()

    if args.mode == "quick":
        run_quick_experiment(config, dirs)
    elif args.mode == "full":
        run_full_experiment(config, dirs, args.skip_data_gen)
        # Generate basic figures
        print("\nGenerating basic figures...")
        generate_basic_figures()
    elif args.mode == "advanced":
        run_advanced_experiment(config, dirs, args.skip_data_gen)
        # Generate basic figures too
        print("\nGenerating basic figures...")
        generate_basic_figures()

    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETED IN {elapsed_time:.2f} SECONDS")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
