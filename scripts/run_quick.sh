#!/bin/bash
# Quick run: sklearn-only experiment (no xgboost dependency).
set -e
cd "$(dirname "$0")/.."
echo "=========================================="
echo "Multi-Agent Fraud Detection - Quick Run"
echo "=========================================="
export PYTHONPATH="$(pwd)/code:${PYTHONPATH}"
python code/scripts/run_experiment_simple.py
echo ""
echo "Quick run complete. Results in code/results/"
