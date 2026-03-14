#!/usr/bin/env bash
# Run all solvers on all datasets, then generate CSV report
set -euo pipefail
cd "$(dirname "$0")"

# Activate local venv
source .venv/bin/activate

SOLVERS_DIR="solvers"
DATASETS_DIR="datasets"
RESULTS_DIR="results"
FILTER="${1:-}"

export PYTHONPATH="."

for solver in "$SOLVERS_DIR"/*.py; do
    solver_name=$(basename "$solver" .py)
    if [[ -n "$FILTER" && "$solver_name" != "$FILTER" ]]; then
        continue
    fi
    solver_results_dir="$RESULTS_DIR/$solver_name"
    mkdir -p "$solver_results_dir"

    for dataset in "$DATASETS_DIR"/*.json; do
        dataset_name=$(basename "$dataset" .json)
        out_json="$solver_results_dir/${dataset_name}.json"

        echo "=== $solver_name × $dataset_name ==="
        python3 "$solver" "$dataset" "$out_json" 2>&1 || true
        echo ""
    done
done

echo "Running benchmark report..."
python3 benchmark_report.py "$RESULTS_DIR" benchmark.csv

echo ""
echo "==============================="
echo "CSV saved to benchmark.csv"
echo ""
column -t -s',' benchmark.csv
