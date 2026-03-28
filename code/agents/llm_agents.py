"""
LLM-powered agents for narrative generation and evidence aggregation.
Uses mock LLM for reproducibility (can be swapped with real API).
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class MockLLM:
    """Mock LLM for deterministic testing and reproducibility."""

    def __init__(self, config):
        self.config = config
        self.temperature = config.llm.temperature

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response based on prompt content."""
        # Simple deterministic responses for testing
        if "evidence" in prompt.lower():
            return self._generate_evidence_response()
        elif "narrative" in prompt.lower():
            return self._generate_narrative_response()
        else:
            return "Mock LLM response generated."

    def _generate_evidence_response(self) -> str:
        return json.dumps(
            {
                "suspicious_signals": ["amount_anomaly", "location_anomaly"],
                "confidence": 0.85,
                "reasoning": "Transaction amount significantly exceeds user baseline",
            }
        )

    def _generate_narrative_response(self) -> str:
        return (
            "FRAUD ALERT: Suspicious activity detected. "
            "Transaction amount ($2,450.00) is 8x user's typical spending. "
            "Location flagged as unusual (international). "
            "Recommend immediate investigation."
        )


class EvidenceAggregator:
    """Agent that consolidates signals from multiple detectors."""

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm or MockLLM(config)

    def aggregate(
        self, transaction: Dict, detector_scores: Dict[str, float], features: Dict
    ) -> Dict[str, Any]:
        """
        Aggregate evidence from multiple sources.

        Args:
            transaction: Raw transaction data
            detector_scores: Scores from each detector {model_name: score}
            features: Extracted features

        Returns:
            Evidence dictionary with consolidated assessment
        """
        # Weighted ensemble of detector scores
        weights = {"isolation_forest": 0.2, "xgboost": 0.5, "ensemble": 0.3}

        total_weight = sum(
            weight for name, weight in weights.items() if name in detector_scores
        )

        if total_weight > 0:
            weighted_score = (
                sum(
                    detector_scores.get(name, 0) * weight
                    for name, weight in weights.items()
                    if name in detector_scores
                )
                / total_weight
            )
        else:
            weighted_score = 0.0

        # Identify suspicious features
        suspicious_features = self._identify_suspicious_features(features, transaction)

        # Risk level
        if weighted_score > 0.8:
            risk_level = "HIGH"
        elif weighted_score > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        evidence = {
            "transaction_id": transaction["transaction_id"],
            "user_id": transaction["user_id"],
            "timestamp": transaction["timestamp"],
            "fraud_score": weighted_score,
            "risk_level": risk_level,
            "detector_scores": detector_scores,
            "suspicious_features": suspicious_features,
            "feature_values": self._extract_key_features(features, transaction),
            "aggregation_method": "weighted_ensemble",
            "aggregated_at": datetime.now().isoformat(),
        }

        return evidence

    def _identify_suspicious_features(
        self, features: Dict, transaction: Dict
    ) -> List[str]:
        """Identify which features are suspicious."""
        suspicious = []

        # Amount checks
        if features.get("amount", 0) > 500:
            suspicious.append("high_amount")

        # Time checks
        if features.get("is_night", 0) == 1:
            suspicious.append("unusual_time")

        # Location checks
        if features.get("is_international", 0) == 1:
            suspicious.append("international_transaction")

        # Velocity checks
        if features.get("tx_count_1h", 0) > 3:
            suspicious.append("high_velocity")

        return suspicious

    def _extract_key_features(self, features: Dict, transaction: Dict) -> Dict:
        """Extract key feature values for explanation."""
        return {
            "amount": transaction.get("amount"),
            "merchant_category": transaction.get("merchant_category"),
            "location": transaction.get("location"),
            "hour": features.get("hour"),
            "tx_count_1h": features.get("tx_count_1h"),
            "is_international": features.get("is_international"),
        }


class NarrativeGenerator:
    """Agent that generates human-readable fraud explanations."""

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm or MockLLM(config)

    def generate_narrative(self, evidence: Dict, transaction: Dict) -> str:
        """
        Generate human-readable narrative explaining fraud detection.

        Args:
            evidence: Evidence dictionary from aggregator
            transaction: Original transaction

        Returns:
            Human-readable narrative string
        """
        # Use template-based generation for determinism
        narrative = self._template_based_narrative(evidence, transaction)

        return narrative

    def _format_amount(self, amount) -> str:

        if amount is None:
            return "N/A"
        try:
            return f"${float(amount):.2f}"
        except (TypeError, ValueError):
            return str(amount)

    def _template_based_narrative(self, evidence: Dict, transaction: Dict) -> str:
        """Generate narrative using templates."""
        risk_level = evidence["risk_level"]
        fraud_score = evidence["fraud_score"]
        suspicious_features = evidence["suspicious_features"]
        features = evidence["feature_values"]

        amount_str = self._format_amount(features.get("amount"))

        # Header
        header = f"FRAUD ALERT [{risk_level} RISK]\n"
        header += f"Transaction ID: {transaction['transaction_id']}\n"
        header += f"User: {transaction['user_id']}\n"
        header += f"Fraud Score: {fraud_score:.3f}\n\n"

        # Summary
        summary = "SUSPICIOUS ACTIVITY DETECTED\n"
        summary += (
            f"Transaction of {amount_str} at {features.get('merchant_category', 'N/A')} "
            f"from location {features.get('location', 'N/A')}.\n\n"
        )

        # Evidence
        evidence_text = "KEY INDICATORS:\n"
        for feat in suspicious_features:
            if feat == "high_amount":
                evidence_text += (
                    f"  • Transaction amount ({amount_str}) unusually high\n"
                )
            elif feat == "unusual_time":
                hour = features.get("hour")
                hour_str = f"{hour}:00" if hour is not None else "N/A"
                evidence_text += f"  • Transaction at unusual hour ({hour_str})\n"
            elif feat == "international_transaction":
                evidence_text += f"  • International transaction from {features.get('location', 'N/A')}\n"
            elif feat == "high_velocity":
                evidence_text += f"  • High transaction velocity ({features.get('tx_count_1h', 'N/A')} in past hour)\n"

        # Recommendation
        if risk_level == "HIGH":
            recommendation = "\nRECOMMENDATION: Immediate investigator review and potential account freeze.\n"
        elif risk_level == "MEDIUM":
            recommendation = (
                "\nRECOMMENDATION: Flag for investigator review within 24 hours.\n"
            )
        else:
            recommendation = (
                "\nRECOMMENDATION: Monitor transaction; no immediate action required.\n"
            )

        narrative = header + summary + evidence_text + recommendation

        return narrative

    def generate_case_report(
        self, evidence: Dict, transaction: Dict, narrative: str
    ) -> Dict[str, Any]:
        """Generate structured case report for investigators."""
        report = {
            "report_id": f"RPT-{transaction['transaction_id']}",
            "generated_at": datetime.now().isoformat(),
            "transaction": transaction,
            "evidence": evidence,
            "narrative": narrative,
            "status": "pending_review",
            "priority": evidence["risk_level"],
            "assigned_to": None,
            "actions_taken": [],
        }

        return report
