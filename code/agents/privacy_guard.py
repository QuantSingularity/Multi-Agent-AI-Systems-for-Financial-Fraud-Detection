"""
Privacy Guard agent for PII redaction and data protection.
Implements deterministic rules and policy enforcement.
"""

import hashlib
import re
from typing import Any, Dict, List


class PrivacyGuard:
    """Agent for PII redaction and privacy policy enforcement."""

    def __init__(self, config):
        self.config = config
        self.redact_enabled = config.privacy.enabled
        self.redact_fields = config.privacy.redact_fields
        self.log_redacted = config.privacy.log_redacted

        # PII patterns
        self.patterns = {
            "card_number": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.\\s]?\d{3}[-.\\s]?\d{4}\b"),
            "account_number": re.compile(r"\b\d{10,12}\b"),
        }

        self.redaction_log = []

    def redact_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact PII from transaction data.

        Args:
            transaction: Raw transaction dictionary

        Returns:
            Redacted transaction with PII removed
        """
        if not self.redact_enabled:
            return transaction

        redacted = transaction.copy()
        redacted_fields = []

        # Redact configured fields
        for field in self.redact_fields:
            if field in redacted:
                original_value = str(redacted[field])
                redacted[field] = self._redact_value(field, original_value)
                redacted_fields.append(field)

        # Add metadata
        redacted["_privacy_redacted"] = True
        redacted["_redacted_fields"] = redacted_fields

        if self.log_redacted:
            self.redaction_log.append(
                {
                    "transaction_id": transaction.get("transaction_id"),
                    "redacted_fields": redacted_fields,
                    "timestamp": transaction.get("timestamp"),
                }
            )

        return redacted

    def _redact_value(self, field_name: str, value: str) -> str:
        """Redact a specific field value."""
        # Hash for consistency (same value always same hash)
        hashed = hashlib.sha256(value.encode()).hexdigest()[:8]

        if field_name == "card_number":
            # Keep last 4 digits if available
            if len(value) >= 4:
                return f"****-****-****-{value[-4:]}"
            return "****-****-****-****"

        elif field_name == "account_number":
            return f"ACCT-{hashed}"

        elif field_name == "ssn":
            return "***-**-****"

        elif field_name == "email":
            return f"user{hashed}@redacted.com"

        elif field_name == "phone":
            return "***-***-****"

        else:
            return f"[REDACTED-{hashed}]"

    def redact_text(self, text: str) -> str:
        """Redact PII from free-form text (narratives, logs)."""
        if not self.redact_enabled:
            return text

        redacted_text = text

        for field_name, pattern in self.patterns.items():
            matches = pattern.findall(redacted_text)
            for match in matches:
                redacted_value = self._redact_value(field_name, match)
                redacted_text = redacted_text.replace(match, redacted_value)

        return redacted_text

    def enforce_rate_limit(
        self, user_id: str, recent_flags: List[Dict]
    ) -> Dict[str, Any]:
        """
        Enforce rate limiting on fraud flags per user.

        Args:
            user_id: User identifier
            recent_flags: List of recent fraud flags for this user

        Returns:
            Policy decision dict with allow/deny and reason
        """
        max_flags = self.config.safeguards.max_flags_per_user_per_day

        if len(recent_flags) >= max_flags:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded: {len(recent_flags)}/{max_flags} flags in 24h",
                "action": "defer_to_investigator",
            }

        return {
            "allowed": True,
            "reason": f"Within rate limit: {len(recent_flags)}/{max_flags} flags",
            "action": "proceed",
        }

    def check_false_positive_throttle(self, recent_accuracy: float) -> Dict[str, Any]:
        """
        Check if system false positive rate is too high.

        Args:
            recent_accuracy: Recent precision/accuracy metric

        Returns:
            Policy decision dict
        """
        threshold = 1.0 - self.config.safeguards.false_positive_throttle_ratio

        if recent_accuracy < threshold:
            return {
                "allowed": False,
                "reason": f"False positive rate too high: accuracy={recent_accuracy:.2f}",
                "action": "throttle_system",
                "recommendation": "Review model performance and recalibrate",
            }

        return {
            "allowed": True,
            "reason": f"False positive rate acceptable: accuracy={recent_accuracy:.2f}",
            "action": "proceed",
        }

    def apply_investigator_review_gate(
        self, fraud_score: float, evidence: Dict
    ) -> Dict[str, Any]:
        """
        Determine if investigator review is required before action.

        Args:
            fraud_score: Model confidence score
            evidence: Evidence dictionary

        Returns:
            Gate decision dict
        """
        if not self.config.safeguards.investigator_review_required:
            return {"review_required": False, "auto_action": True}

        # High-confidence cases can auto-flag
        if fraud_score > 0.95:
            return {
                "review_required": False,
                "auto_action": True,
                "reason": "High confidence score",
            }

        # Medium confidence requires review
        if fraud_score > 0.7:
            return {
                "review_required": True,
                "auto_action": False,
                "reason": "Medium confidence - investigator review recommended",
                "priority": "high",
            }

        # Low confidence - require review before flagging
        return {
            "review_required": True,
            "auto_action": False,
            "reason": "Low confidence - investigator review required",
            "priority": "medium",
        }

    def get_redaction_summary(self) -> Dict[str, Any]:
        """Get summary of redaction operations."""
        return {
            "total_redactions": len(self.redaction_log),
            "redacted_transactions": len(
                set(
                    r["transaction_id"]
                    for r in self.redaction_log
                    if r["transaction_id"] is not None
                )
            ),
            "log": self.redaction_log[-100:],  # Last 100 entries
        }
