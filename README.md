# Multi-Agent AI Systems for Financial Fraud Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](code/requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, reproducible multi-agent system for real-time financial fraud detection. The `FraudDetectionOrchestrator` combines ensemble ML models (XGBoost, Isolation Forest) for high-speed scoring with LLM agents for evidence aggregation, narrative generation, and privacy-compliant case reporting.

---

## Table of Contents

- [Overview](#overview)
- [Agentic Workflow](#agentic-workflow)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Privacy and Compliance](#privacy-and-compliance)
- [License](#license)

---

## Overview

| Feature                   | Description                                                                                                 |
| :------------------------ | :---------------------------------------------------------------------------------------------------------- |
| **Hybrid Detection**      | Ensemble ML (XGBoost, Isolation Forest) for fast scoring combined with LLM agents for complex case analysis |
| **Privacy-First Design**  | Privacy Guard Agent enforces PII redaction and rate limiting before any LLM or investigator exposure        |
| **Explainability**        | Narrative Generator produces human-readable case reports addressing the GDPR Right to Explanation           |
| **Real-Time Performance** | Mean detection latency of 127ms per transaction, suitable for high-volume environments                      |
| **Cost-Benefit Analysis** | Threshold optimization by quantifying the financial impact of false positives and false negatives           |
| **Kubernetes Deployment** | K8s manifests and Helm charts for scalable cloud deployment                                                 |

---

## Agentic Workflow

The `FraudDetectionOrchestrator` (`code/orchestrator/orchestrator.py`) coordinates six components.

| Component               | Type      | Function                                                                                             |
| :---------------------- | :-------- | :--------------------------------------------------------------------------------------------------- |
| **Feature Engineer**    | Utility   | Extracts temporal, statistical, and behavioral features from raw transactions                        |
| **Privacy Guard**       | Agent     | Redacts PII and enforces policy gates before downstream processing                                   |
| **Anomaly Detector**    | Model     | Runs the ML ensemble to assign a raw fraud probability score                                         |
| **Evidence Aggregator** | LLM Agent | Consolidates ML scores, feature importance, and policy violations into a structured evidence package |
| **Narrative Generator** | LLM Agent | Produces a concise, human-readable case report from the evidence package                             |
| **Online Learning**     | Utility   | Supports incremental model updates to adapt to emerging fraud patterns                               |

---

## Repository Structure

| Path                 | Description                                                |
| :------------------- | :--------------------------------------------------------- |
| `code/orchestrator/` | Main workflow coordinator                                  |
| `code/agents/`       | LLM agents and Privacy Guard                               |
| `code/models/`       | Isolation Forest, XGBoost, and Ensemble implementations    |
| `code/data/`         | Synthetic data generation and feature engineering          |
| `code/utils/`        | Cost-benefit analysis, imbalance handling, online learning |
| `code/eval/`         | Evaluation scripts and figure generation                   |
| `code/scripts/`      | Experiment entry points                                    |
| `k8s/`               | Kubernetes deployment manifests                            |
| `helm/`              | Helm chart for parameterized deployment                    |
| `figures/`           | Generated plots and visualizations                         |

---

## Quick Start

### Prerequisites

- Docker 20.10+ (recommended)
- Python 3.9+ (for local run)

### Docker

```bash
git clone https://github.com/quantsingularity/Multi-Agent-AI-Systems-for-Financial-Fraud-Detection.git
cd Multi-Agent-AI-Systems-for-Financial-Fraud-Detection

docker build -t fraud-detection-agents .
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/figures:/app/figures \
  fraud-detection-agents python code/scripts/run_experiment.py --mode full
```

Results and figures are saved to `results/` and `figures/` on the host.

---

## Results

### Model Comparison

| Model                       | Precision | Recall | F1 Score | AUC-ROC  |
| :-------------------------- | :-------- | :----- | :------- | :------- |
| Isolation Forest            | 0.42      | 0.68   | 0.52     | 0.84     |
| XGBoost                     | 0.71      | 0.83   | 0.76     | 0.92     |
| Ensemble Detector           | 0.74      | 0.81   | 0.77     | 0.93     |
| **Full Multi-Agent System** | **0.78**  | 0.79   | **0.78** | **0.94** |

### Operational Performance

| Metric                  | Value          |
| :---------------------- | :------------- |
| Mean Detection Latency  | 127ms          |
| P95 Detection Latency   | 340ms          |
| Investigator Time Saved | ~67% reduction |

---

## Privacy and Compliance

| Safeguard            | Description                                                                           |
| :------------------- | :------------------------------------------------------------------------------------ |
| PII Redaction        | Policy-driven redaction before any LLM or investigator exposure — GDPR/CCPA aligned   |
| Audit Logs           | Timestamped trace of all agent decisions and scores for regulatory review             |
| Narrative Generation | Evidence-backed rationale per flagged transaction addressing the Right to Explanation |

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
