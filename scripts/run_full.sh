#!/bin/bash
# Full run: complete multi-agent experiment.
set -e
cd "$(dirname "$0")/.."
echo "=========================================="
echo "Multi-Agent Fraud Detection - Full Run"
echo "=========================================="
export PYTHONPATH="$(pwd)/code:${PYTHONPATH}"
python code/scripts/run_experiment.py
echo ""
echo "Full run complete. Results in code/results/"
