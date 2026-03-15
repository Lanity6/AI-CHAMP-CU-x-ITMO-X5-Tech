#!/bin/bash
# Run LNS solver multiple times on hard dataset for averaging.
# Usage: bash brute_force_vs_algo/run_multi_lns.sh

set -e

N_RUNS=5
DATASET="brute_force_vs_algo/brute_data/hard_test.json"
OUT_DIR="brute_force_vs_algo/brute_result"

echo "=== Running LNS $N_RUNS times on hard dataset ==="

for i in $(seq 1 $N_RUNS); do
    OUT="$OUT_DIR/res_hard_lns_run${i}.json"
    echo "--- Run $i/$N_RUNS -> $OUT ---"
    PYTHONPATH=. python3 solvers/lns_solver.py "$DATASET" "$OUT"
    echo ""
done

echo "=== All $N_RUNS LNS runs complete ==="
