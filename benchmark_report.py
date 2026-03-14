#!/usr/bin/env python3
"""Extract metrics from solver result JSONs and write a CSV report."""

import csv
import json
import os
import sys
from pathlib import Path


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "benchmark.csv"

    rows = []
    for jf in sorted(results_dir.rglob("*.json")):
        solver_name = jf.parent.name if jf.parent != results_dir else jf.stem
        with open(jf, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        for task in tasks:
            m = task.get("metrics", {})
            rows.append({
                "solver": solver_name,
                "file": jf.name,
                "task_id": task.get("task_id", ""),
                "solver_version": task.get("solver_version", ""),
                "solve_time_ms": task.get("solve_time_ms", ""),
                "placements_count": len(task.get("placements", [])),
                "unplaced_count": len(task.get("unplaced", [])),
                "valid": m.get("valid", ""),
                "final_score": m.get("final_score", ""),
                "volume_utilization": m.get("volume_utilization", ""),
                "item_coverage": m.get("item_coverage", ""),
                "fragility_score": m.get("fragility_score", ""),
                "time_score": m.get("time_score", ""),
                "error": m.get("error", ""),
            })

    if not rows:
        print("No result JSONs found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")

    # Print summary per solver / dataset
    from collections import defaultdict
    by_solver = defaultdict(list)
    for r in rows:
        by_solver[(r["solver"], r["file"])].append(r)

    print(f"\n{'Solver':<20} {'Dataset':<25} {'Tasks':>5} {'Valid':>5} {'Avg Score':>10}")
    print("-" * 70)
    for (solver, fname), file_rows in sorted(by_solver.items()):
        total = len(file_rows)
        valid = sum(1 for r in file_rows if r["valid"] is True)
        scores = [r["final_score"] for r in file_rows if isinstance(r["final_score"], (int, float))]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{solver:<20} {fname:<25} {total:>5} {valid:>5} {avg:>10.4f}")


if __name__ == "__main__":
    main()
