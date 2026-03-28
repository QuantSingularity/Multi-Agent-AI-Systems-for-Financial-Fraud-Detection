"""
Configuration management for fraud detection multi-agent system.
Handles environment variables, model configs, and orchestration policies.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    isolation_forest_contamination: float = 0.1
    autoencoder_latent_dim: int = 8
    autoencoder_epochs: int = 50
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    random_state: int = 42


@dataclass
class LLMConfig:
    """Configuration for LLM agents."""

    provider: str = "mock"  # mock, openai, anthropic
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 1000
    api_key: Optional[str] = None


@dataclass
class PrivacyConfig:
    """Privacy and PII redaction configuration."""

    enabled: bool = True
    redact_fields: List[str] = field(
        default_factory=lambda: [
            "card_number",
            "ssn",
            "account_number",
            "email",
            "phone",
        ]
    )
    log_redacted: bool = True


@dataclass
class OrchestrationConfig:
    """Orchestrator behavior configuration."""

    mode: str = "sequential"  # sequential, parallel, adaptive
    max_parallel_agents: int = 3
    timeout_seconds: int = 30
    retry_attempts: int = 2


@dataclass
class SafeguardsConfig:
    """Safety constraints and rate limits."""

    max_flags_per_user_per_day: int = 10
    false_positive_throttle_ratio: float = 0.3
    investigator_review_required: bool = True
    kill_switch_enabled: bool = True


@dataclass
class SystemConfig:
    """Master configuration for the entire system."""

    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    safeguards: SafeguardsConfig = field(default_factory=SafeguardsConfig)

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("../data"))
    results_dir: Path = field(default_factory=lambda: Path("../results"))
    figures_dir: Path = field(default_factory=lambda: Path("../figures"))

    # Experiment settings
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cv_folds: int = 5

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = cls()

        if "model" in config_dict and isinstance(config_dict["model"], dict):
            config.model = ModelConfig(**config_dict["model"])
        if "llm" in config_dict and isinstance(config_dict["llm"], dict):
            config.llm = LLMConfig(**config_dict["llm"])
        if "privacy" in config_dict and isinstance(config_dict["privacy"], dict):
            config.privacy = PrivacyConfig(**config_dict["privacy"])
        if "orchestration" in config_dict and isinstance(
            config_dict["orchestration"], dict
        ):
            config.orchestration = OrchestrationConfig(**config_dict["orchestration"])
        if "safeguards" in config_dict and isinstance(config_dict["safeguards"], dict):
            config.safeguards = SafeguardsConfig(**config_dict["safeguards"])

        # Scalar overrides
        for key in ("train_ratio", "val_ratio", "test_ratio", "cv_folds"):
            if key in config_dict:
                setattr(config, key, config_dict[key])

        for key in ("data_dir", "results_dir", "figures_dir"):
            if key in config_dict:
                setattr(config, key, Path(config_dict[key]))

        return config

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Load API keys from environment
        config.llm.api_key = os.getenv("OPENAI_API_KEY") or os.getenv(
            "ANTHROPIC_API_KEY"
        )
        if config.llm.api_key:
            config.llm.provider = (
                "openai" if "OPENAI_API_KEY" in os.environ else "anthropic"
            )

        # Override with env vars if present
        if os.getenv("RANDOM_SEED"):
            config.model.random_state = int(os.getenv("RANDOM_SEED"))

        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "llm": {k: v for k, v in self.llm.__dict__.items() if k != "api_key"},
            "privacy": self.privacy.__dict__,
            "orchestration": self.orchestration.__dict__,
            "safeguards": self.safeguards.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_config() -> SystemConfig:
    """Get system configuration with environment variable overrides."""
    config = SystemConfig.from_env()
    return config
