"""
Cost-benefit analysis module for fraud detection.
Provides business-oriented analysis of false positive/negative costs,
threshold selection, and ROI calculations.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class CostBenefitAnalyzer:
    """
    Analyzes cost-benefit tradeoffs in fraud detection.

    Key Metrics:
    - False Positive Cost: Cost of investigating a legitimate transaction
    - False Negative Cost: Cost of missing a fraudulent transaction
    - Threshold Optimization: Find optimal decision threshold
    - ROI Calculation: Business value of the system
    """

    def __init__(
        self,
        fp_cost: float = 50.0,
        fn_cost: float = 500.0,
        avg_transaction_value: float = 150.0,
        monthly_transactions: int = 1_000_000,
    ):
        """
        Initialize cost-benefit analyzer.

        Args:
            fp_cost: Cost per false positive (investigation cost)
            fn_cost: Cost per false negative (average fraud loss)
            avg_transaction_value: Average transaction amount
            monthly_transactions: Monthly transaction volume
        """
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.avg_transaction_value = avg_transaction_value
        self.monthly_transactions = monthly_transactions

        # Cost ratio
        self.cost_ratio = fn_cost / fp_cost

        print("Cost-Benefit Analyzer Initialized:")
        print(f"  FP Cost: ${fp_cost:.2f}")
        print(f"  FN Cost: ${fn_cost:.2f}")
        print(f"  Cost Ratio (FN/FP): {self.cost_ratio:.1f}x")

    def calculate_costs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate total costs for given predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary with cost breakdown
        """
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Direct costs
        fp_total_cost = fp * self.fp_cost
        fn_total_cost = fn * self.fn_cost
        total_cost = fp_total_cost + fn_total_cost

        # Cost per transaction
        cost_per_transaction = total_cost / len(y_true)

        # Annualized costs
        annual_transactions = self.monthly_transactions * 12
        annual_cost = cost_per_transaction * annual_transactions

        results = {
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "fp_cost_total": float(fp_total_cost),
            "fn_cost_total": float(fn_total_cost),
            "total_cost": float(total_cost),
            "cost_per_transaction": float(cost_per_transaction),
            "annual_cost_estimate": float(annual_cost),
            "investigation_workload": int(tp + fp),  # Total flagged cases
        }

        return results

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """
        Find optimal classification threshold that minimizes total cost.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            thresholds: Array of thresholds to test (optional)

        Returns:
            optimal_threshold, costs_at_optimal
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)

        costs = []
        details = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost_info = self.calculate_costs(y_true, y_pred)
            costs.append(cost_info["total_cost"])
            details.append(cost_info)

        costs = np.array(costs)
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_costs = details[optimal_idx]

        # Store for later analysis
        self.threshold_analysis = {
            "thresholds": thresholds.tolist(),
            "costs": costs.tolist(),
            "details": details,
            "optimal_threshold": float(optimal_threshold),
            "optimal_cost": float(optimal_costs["total_cost"]),
        }

        print("\nOptimal Threshold Analysis:")
        print(f"  Optimal Threshold: {optimal_threshold:.3f}")
        print(f"  Total Cost: ${optimal_costs['total_cost']:,.2f}")
        print(f"  FP Count: {optimal_costs['false_positives']}")
        print(f"  FN Count: {optimal_costs['false_negatives']}")
        print(f"  Annual Cost: ${optimal_costs['annual_cost_estimate']:,.2f}")

        return optimal_threshold, optimal_costs

    def threshold_sensitivity_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        cost_ratios: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Analyze how optimal threshold changes with different cost ratios.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            cost_ratios: Array of FN/FP cost ratios to test

        Returns:
            DataFrame with sensitivity analysis results
        """
        if cost_ratios is None:
            cost_ratios = np.array([1, 5, 10, 20, 50, 100, 200])

        results = []

        original_fp_cost = self.fp_cost
        original_fn_cost = self.fn_cost

        for ratio in cost_ratios:
            # Update costs
            self.fn_cost = original_fp_cost * ratio
            self.cost_ratio = ratio

            # Find optimal threshold
            optimal_threshold, costs = self.find_optimal_threshold(y_true, y_proba)

            results.append(
                {
                    "cost_ratio": ratio,
                    "optimal_threshold": optimal_threshold,
                    "total_cost": costs["total_cost"],
                    "fp_count": costs["false_positives"],
                    "fn_count": costs["false_negatives"],
                    "annual_cost": costs["annual_cost_estimate"],
                }
            )

        # Restore original costs
        self.fp_cost = original_fp_cost
        self.fn_cost = original_fn_cost
        self.cost_ratio = original_fn_cost / original_fp_cost

        df = pd.DataFrame(results)
        return df

    def calculate_roi(
        self,
        baseline_costs: Dict,
        system_costs: Dict,
        implementation_cost: float = 300_000,
        annual_maintenance: float = 100_000,
    ) -> Dict[str, float]:
        """
        Calculate Return on Investment (ROI) for the fraud detection system.

        Args:
            baseline_costs: Costs without the system
            system_costs: Costs with the system
            implementation_cost: One-time implementation cost
            annual_maintenance: Annual maintenance cost

        Returns:
            Dictionary with ROI metrics
        """
        # Annual savings
        baseline_annual = baseline_costs["annual_cost_estimate"]
        system_annual = system_costs["annual_cost_estimate"]
        annual_savings = baseline_annual - system_annual

        # Investigation time savings (assuming $50/hour, 30 min per case)
        baseline_investigations = baseline_costs["investigation_workload"]
        system_investigations = system_costs["investigation_workload"]
        investigation_hours_saved = (
            baseline_investigations - system_investigations
        ) * 0.5
        investigation_cost_saved = investigation_hours_saved * 50

        # Total annual benefit
        total_annual_benefit = annual_savings + investigation_cost_saved

        # First year ROI
        first_year_net = total_annual_benefit - implementation_cost - annual_maintenance
        first_year_roi = (
            first_year_net / (implementation_cost + annual_maintenance)
        ) * 100

        # Ongoing ROI (years 2+)
        ongoing_net = total_annual_benefit - annual_maintenance
        ongoing_roi = (ongoing_net / annual_maintenance) * 100

        # Payback period
        payback_months = (
            implementation_cost / (total_annual_benefit - annual_maintenance)
        ) * 12

        results = {
            "baseline_annual_cost": baseline_annual,
            "system_annual_cost": system_annual,
            "annual_fraud_savings": annual_savings,
            "investigation_cost_savings": investigation_cost_saved,
            "total_annual_benefit": total_annual_benefit,
            "implementation_cost": implementation_cost,
            "annual_maintenance_cost": annual_maintenance,
            "first_year_net_benefit": first_year_net,
            "first_year_roi_percent": first_year_roi,
            "ongoing_annual_roi_percent": ongoing_roi,
            "payback_period_months": payback_months,
            "3_year_total_benefit": total_annual_benefit * 3
            - implementation_cost
            - annual_maintenance * 3,
        }

        print(f"\n{'='*60}")
        print("ROI ANALYSIS")
        print(f"{'='*60}")
        print(f"Annual Fraud Savings: ${annual_savings:,.2f}")
        print(f"Investigation Savings: ${investigation_cost_saved:,.2f}")
        print(f"Total Annual Benefit: ${total_annual_benefit:,.2f}")
        print("\nFirst Year:")
        print(f"  Net Benefit: ${first_year_net:,.2f}")
        print(f"  ROI: {first_year_roi:.1f}%")
        print("\nOngoing (Year 2+):")
        print(f"  Annual Net: ${ongoing_net:,.2f}")
        print(f"  ROI: {ongoing_roi:.1f}%")
        print(f"\nPayback Period: {payback_months:.1f} months")
        print(f"3-Year Total Benefit: ${results['3_year_total_benefit']:,.2f}")
        print(f"{'='*60}\n")

        return results

    def generate_business_report(
        self,
        baseline_costs: Dict,
        system_costs: Dict,
        roi_metrics: Dict,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate comprehensive business-oriented report.

        Args:
            baseline_costs: Costs without system
            system_costs: Costs with system
            roi_metrics: ROI calculation results
            output_path: Path to save report (optional)

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("FRAUD DETECTION SYSTEM - BUSINESS COST-BENEFIT ANALYSIS")
        report.append("=" * 80)
        report.append("")

        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(
            f"Implementation Cost: ${roi_metrics['implementation_cost']:,.0f}"
        )
        report.append(
            f"Annual Maintenance: ${roi_metrics['annual_maintenance_cost']:,.0f}"
        )
        report.append(f"First Year ROI: {roi_metrics['first_year_roi_percent']:.1f}%")
        report.append(
            f"Payback Period: {roi_metrics['payback_period_months']:.1f} months"
        )
        report.append(
            f"3-Year Total Benefit: ${roi_metrics['3_year_total_benefit']:,.0f}"
        )
        report.append("")

        report.append("COST BREAKDOWN")
        report.append("-" * 80)

        report.append("\nBASELINE (Without ML System):")
        report.append(f"  False Positives: {baseline_costs['false_positives']:,}")
        report.append(f"  False Negatives: {baseline_costs['false_negatives']:,}")
        report.append(f"  FP Cost: ${baseline_costs['fp_cost_total']:,.2f}")
        report.append(f"  FN Cost: ${baseline_costs['fn_cost_total']:,.2f}")
        report.append(f"  Total Cost: ${baseline_costs['total_cost']:,.2f}")
        report.append(
            f"  Annual Estimate: ${baseline_costs['annual_cost_estimate']:,.0f}"
        )
        report.append(
            f"  Investigation Workload: {baseline_costs['investigation_workload']:,} cases"
        )

        report.append("\nWITH ML SYSTEM:")
        report.append(f"  False Positives: {system_costs['false_positives']:,}")
        report.append(f"  False Negatives: {system_costs['false_negatives']:,}")
        report.append(f"  FP Cost: ${system_costs['fp_cost_total']:,.2f}")
        report.append(f"  FN Cost: ${system_costs['fn_cost_total']:,.2f}")
        report.append(f"  Total Cost: ${system_costs['total_cost']:,.2f}")
        report.append(
            f"  Annual Estimate: ${system_costs['annual_cost_estimate']:,.0f}"
        )
        report.append(
            f"  Investigation Workload: {system_costs['investigation_workload']:,} cases"
        )

        report.append("\nIMPROVEMENTS:")
        fp_reduction = (
            (baseline_costs["false_positives"] - system_costs["false_positives"])
            / baseline_costs["false_positives"]
            * 100
        )
        fn_reduction = (
            (baseline_costs["false_negatives"] - system_costs["false_negatives"])
            / baseline_costs["false_negatives"]
            * 100
        )
        workload_reduction = (
            (
                baseline_costs["investigation_workload"]
                - system_costs["investigation_workload"]
            )
            / baseline_costs["investigation_workload"]
            * 100
        )

        report.append(f"  False Positive Reduction: {fp_reduction:.1f}%")
        report.append(f"  False Negative Reduction: {fn_reduction:.1f}%")
        report.append(f"  Investigation Workload Reduction: {workload_reduction:.1f}%")
        report.append(
            f"  Annual Cost Savings: ${roi_metrics['annual_fraud_savings']:,.0f}"
        )
        report.append("")

        report.append("FINANCIAL PROJECTIONS (3-Year Outlook)")
        report.append("-" * 80)

        year_1_benefit = roi_metrics["first_year_net_benefit"]
        year_2_3_benefit = (
            roi_metrics["total_annual_benefit"] - roi_metrics["annual_maintenance_cost"]
        )

        report.append("\nYear 1:")
        report.append(f"  Revenue/Savings: ${roi_metrics['total_annual_benefit']:,.0f}")
        report.append(f"  Implementation: -${roi_metrics['implementation_cost']:,.0f}")
        report.append(f"  Maintenance: -${roi_metrics['annual_maintenance_cost']:,.0f}")
        report.append(f"  Net Benefit: ${year_1_benefit:,.0f}")

        for year in [2, 3]:
            report.append(f"\nYear {year}:")
            report.append(
                f"  Revenue/Savings: ${roi_metrics['total_annual_benefit']:,.0f}"
            )
            report.append(
                f"  Maintenance: -${roi_metrics['annual_maintenance_cost']:,.0f}"
            )
            report.append(f"  Net Benefit: ${year_2_3_benefit:,.0f}")

        cumulative_3yr = year_1_benefit + 2 * year_2_3_benefit
        report.append(f"\n3-Year Cumulative Net Benefit: ${cumulative_3yr:,.0f}")
        report.append("")

        report.append("RISK FACTORS")
        report.append("-" * 80)
        report.append(
            "• Model performance may degrade over time (requires periodic retraining)"
        )
        report.append("• Fraud patterns evolve (requires continuous monitoring)")
        report.append("• Scaling costs may increase with transaction volume")
        report.append("• Initial false positive rate may impact customer satisfaction")
        report.append("• Regulatory compliance requirements may change")
        report.append("")

        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("1. Implement phased rollout to manage risk")
        report.append("2. Establish monthly performance monitoring")
        report.append("3. Plan for quarterly model retraining")
        report.append("4. Invest in customer communication for flagged transactions")
        report.append("5. Maintain baseline comparison metrics")
        report.append("6. Budget for scaling infrastructure in Year 2")
        report.append("")

        report.append("=" * 80)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"✓ Business report saved to {output_path}")

        return report_text

    def save_analysis(self, output_dir: Path):
        """Save all analysis results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save threshold analysis
        if hasattr(self, "threshold_analysis"):
            with open(output_dir / "threshold_analysis.json", "w") as f:
                json.dump(self.threshold_analysis, f, indent=2)

        print(f"✓ Cost-benefit analysis saved to {output_dir}/")


if __name__ == "__main__":
    print("Cost-Benefit Analysis Module")
    print("Run from experiments to see analysis results")
