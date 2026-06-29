"""
Test suite for the Multi-Agent Financial Fraud Detection system.

Covers synthetic data generation (including a regression test for the
initialization-order bug), feature engineering, the anomaly detectors, the
orchestrator, the privacy guard, cost-benefit analysis, and config path
anchoring.
"""

import numpy as np
import pandas as pd
import pytest
from config import get_config
from data.feature_engineering import FeatureEngineer
from data.synthetic_generator import TransactionGenerator, generate_synthetic_fraud_data
from models.anomaly_detectors import (
    EnsembleDetector,
    IsolationForestDetector,
    XGBoostDetector,
)


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture(scope="module")
def dataset():
    """A small reproducible labelled dataset."""
    gen = TransactionGenerator(n_users=100, random_state=42)
    train = gen.generate_dataset(n_transactions=800, fraud_ratio=0.05)
    test = gen.generate_dataset(n_transactions=300, fraud_ratio=0.05)
    return train, test


@pytest.fixture(scope="module")
def features(dataset):
    """Numeric feature matrices derived from the dataset."""
    train, test = dataset
    fe = FeatureEngineer()
    fe.fit(train)
    X_train = fe.transform(train).values
    X_test = fe.transform(test).values
    y_train = train["is_fraud"].values
    y_test = test["is_fraud"].values
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
class TestSyntheticGenerator:
    def test_generator_constructs(self):
        """Regression test: the generator initialised attributes in the wrong
        order, so construction raised AttributeError on ``merchant_categories``.
        """
        gen = TransactionGenerator(n_users=10, random_state=0)
        assert len(gen.users) == 10
        assert len(gen.merchant_categories) > 0
        assert len(gen.locations) > 0

    def test_generate_dataset_schema(self, dataset):
        train, _ = dataset
        for col in ["transaction_id", "amount", "merchant_category", "is_fraud"]:
            assert col in train.columns
        assert set(train["is_fraud"].unique()).issubset({0, 1})

    def test_fraud_ratio_approximately_correct(self):
        gen = TransactionGenerator(n_users=100, random_state=7)
        df = gen.generate_dataset(n_transactions=1000, fraud_ratio=0.05)
        assert len(df) == 1000
        assert 0.02 < df["is_fraud"].mean() < 0.10

    def test_reproducible(self):
        a = TransactionGenerator(n_users=50, random_state=123).generate_dataset(
            300, 0.05
        )
        b = TransactionGenerator(n_users=50, random_state=123).generate_dataset(
            300, 0.05
        )
        pd.testing.assert_frame_equal(a, b)

    def test_generate_synthetic_fraud_data_wrapper(self):
        df = generate_synthetic_fraud_data(
            n_samples=500, fraud_ratio=0.03, random_state=42
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500
        assert "is_fraud" in df.columns
        assert df["is_fraud"].sum() > 0


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class TestFeatureEngineering:
    def test_transform_is_numeric_and_finite(self, features):
        X_train, _, X_test, _ = features
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 300
        assert X_train.shape[1] == X_test.shape[1]
        assert np.isfinite(X_train).all()
        assert np.isfinite(X_test).all()


# --------------------------------------------------------------------------- #
# Anomaly detectors
# --------------------------------------------------------------------------- #
class TestDetectors:
    def test_isolation_forest(self, config, features):
        X_train, y_train, X_test, _ = features
        det = IsolationForestDetector(config)
        det.fit(X_train, y_train)
        preds = det.predict(X_test)
        proba = det.predict_proba(X_test)
        assert preds.shape[0] == X_test.shape[0]
        assert set(np.unique(preds)).issubset({0, 1})
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_xgboost(self, config, features):
        X_train, y_train, X_test, _ = features
        det = XGBoostDetector(config)
        det.fit(X_train, y_train)
        preds = det.predict(X_test)
        proba = det.predict_proba(X_test)
        assert preds.shape[0] == X_test.shape[0]
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_ensemble(self, config, features):
        X_train, y_train, X_test, _ = features
        det = EnsembleDetector(config)
        det.fit(X_train, y_train)
        preds = det.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]
        assert set(np.unique(preds)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class TestOrchestrator:
    def test_train_and_detect_single(self, config, dataset):
        from orchestrator.orchestrator import FraudDetectionOrchestrator

        train, test = dataset
        orch = FraudDetectionOrchestrator(config)
        orch.train(train)

        transaction = test.iloc[0].to_dict()
        result = orch.detect_single(transaction)
        assert isinstance(result, dict)
        # The decision should expose some notion of fraud probability / flag.
        keys = {k.lower() for k in result.keys()}
        assert any(
            "fraud" in k or "score" in k or "decision" in k or "prob" in k for k in keys
        )


# --------------------------------------------------------------------------- #
# Privacy guard
# --------------------------------------------------------------------------- #
class TestPrivacyGuard:
    def test_redacts_pii_in_text(self, config):
        from agents.privacy_guard import PrivacyGuard

        guard = PrivacyGuard(config)
        text = "Card 4111-1111-1111-1111 SSN 123-45-6789 email john@example.com"
        redacted = guard.redact_text(text)
        assert "4111-1111-1111-1111" not in redacted
        assert "123-45-6789" not in redacted
        assert "john@example.com" not in redacted


# --------------------------------------------------------------------------- #
# Cost-benefit analysis
# --------------------------------------------------------------------------- #
class TestCostBenefit:
    def test_find_optimal_threshold(self):
        from utils.cost_benefit_analysis import CostBenefitAnalyzer

        rng = np.random.RandomState(0)
        y_true = rng.binomial(1, 0.1, 500)
        y_score = np.clip(y_true * 0.6 + rng.rand(500) * 0.4, 0, 1)
        analyzer = CostBenefitAnalyzer()
        threshold, costs = analyzer.find_optimal_threshold(y_true, y_score)
        assert 0.0 <= float(threshold) <= 1.0
        assert isinstance(costs, dict)
        assert "total_cost" in costs


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_paths_anchored_and_absolute(self, config):
        for path in (config.data_dir, config.results_dir, config.figures_dir):
            assert path.is_absolute(), f"{path} should be absolute (CWD-independent)"
        assert config.data_dir.name == "data"
        assert config.results_dir.name == "results"
        assert config.figures_dir.name == "figures"
