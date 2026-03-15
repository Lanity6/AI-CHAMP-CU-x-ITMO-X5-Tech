#!/usr/bin/env python3
"""Compare brute_force vs LNS scores across all test datasets.

Supports multiple LNS runs (res_*_lns_run{i}.json) for error bars.
Falls back to single-run files (res_*_lns.json) if multi-run not found.

Usage:
    python brute_force_vs_algo/plot_comparison.py
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats


def load_metrics(path):
    """Load task_id -> metrics dict from result file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for r in data:
        tid = r["task_id"]
        m = r.get("metrics", {})
        result[tid] = {
            "final_score": m.get("final_score", 0.0),
            "fragility_score": m.get("fragility_score", 0.0),
            "volume_utilization": m.get("volume_utilization", 0.0),
            "item_coverage": m.get("item_coverage", 0.0),
        }
    return result


def load_multi_run_scores(result_dir, prefix, n_runs=5):
    """Load scores from multiple LNS runs. Returns {task_id: [score1, ..., scoreN]}.

    Looks for files like res_small_lns_run1.json ... res_small_lns_run5.json.
    If not found, falls back to res_small_lns.json (single run).
    """
    multi = {}
    found_runs = 0
    for i in range(1, n_runs + 1):
        path = result_dir / f"res_{prefix}_lns_run{i}.json"
        if path.exists():
            metrics = load_metrics(path)
            for tid, m in metrics.items():
                multi.setdefault(tid, []).append(m["final_score"])
            found_runs += 1

    if found_runs == 0:
        # Fallback to single run
        single = result_dir / f"res_{prefix}_lns.json"
        if single.exists():
            metrics = load_metrics(single)
            for tid, m in metrics.items():
                multi[tid] = [m["final_score"]]

    return multi


def load_multi_run_frag(result_dir, prefix, n_runs=5):
    """Load fragility scores from multiple runs."""
    multi = {}
    found_runs = 0
    for i in range(1, n_runs + 1):
        path = result_dir / f"res_{prefix}_lns_run{i}.json"
        if path.exists():
            metrics = load_metrics(path)
            for tid, m in metrics.items():
                multi.setdefault(tid, []).append(m["fragility_score"])
            found_runs += 1

    if found_runs == 0:
        single = result_dir / f"res_{prefix}_lns.json"
        if single.exists():
            metrics = load_metrics(single)
            for tid, m in metrics.items():
                multi[tid] = [m["fragility_score"]]

    return multi


result_dir = Path("brute_force_vs_algo/brute_result")

# Load BF results (deterministic for small/medium, may have multiple runs for hard)
bf_all = {}
bf_frag = {}
bf_multi = {}  # for hard tasks with LDS (stochastic)

datasets = ["small", "medium", "hard"]

for ds in datasets:
    bf_path = result_dir / f"res_{ds}_brute.json"
    if bf_path.exists():
        metrics = load_metrics(bf_path)
        bf_all.update({tid: m["final_score"] for tid, m in metrics.items()})
        bf_frag.update({tid: m["fragility_score"] for tid, m in metrics.items()})

    # Check for multi-run BF (hard tasks use LDS which is stochastic)
    for i in range(1, 6):
        bf_run = result_dir / f"res_{ds}_brute_run{i}.json"
        if bf_run.exists():
            metrics = load_metrics(bf_run)
            for tid, m in metrics.items():
                bf_multi.setdefault(tid, []).append(m["final_score"])

# Load LNS results (potentially multiple runs)
lns_multi = {}
lns_frag_multi = {}
for ds in datasets:
    lns_multi.update(load_multi_run_scores(result_dir, ds))
    lns_frag_multi.update(load_multi_run_frag(result_dir, ds))

# Compute mean/std for LNS
lns_mean = {tid: np.mean(scores) for tid, scores in lns_multi.items()}
lns_std = {tid: np.std(scores) for tid, scores in lns_multi.items()}
lns_frag_mean = {tid: np.mean(scores) for tid, scores in lns_frag_multi.items()}
lns_frag_std = {tid: np.std(scores) for tid, scores in lns_frag_multi.items()}

# BF: use multi-run mean/std if available, else single value
bf_mean = {}
bf_std = {}
for tid in bf_all:
    if tid in bf_multi:
        bf_mean[tid] = np.mean(bf_multi[tid])
        bf_std[tid] = np.std(bf_multi[tid])
    else:
        bf_mean[tid] = bf_all[tid]
        bf_std[tid] = 0.0

# Group tasks by dataset
groups = {
    "Small (6-10 items)\nExact BF": [f"small_test_{i:03d}" for i in range(1, 31)],
    "Medium (8-10 items)\nExact BF": [f"medium_test_{i:03d}" for i in range(1, 31)],
    "Hard (18-25 items)\nLDS mode": [f"hard_test_{i:03d}" for i in range(1, 31)],
}

# Filter to tasks present in both
task_ids = []
for g in groups.values():
    task_ids.extend([t for t in g if t in bf_mean and t in lns_mean])

if not task_ids:
    print("No matching tasks found. Run solvers first.")
    exit(1)

bf_scores = [bf_mean[t] for t in task_ids]
lns_scores = [lns_mean[t] for t in task_ids]
bf_errs = [bf_std[t] for t in task_ids]
lns_errs = [lns_std[t] for t in task_ids]

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={"height_ratios": [3, 1.5, 1.2]})

x = np.arange(len(task_ids))
width = 0.35

# 1) Score comparison with error bars
ax1 = axes[0]
ax1.bar(x - width/2, bf_scores, width, yerr=bf_errs, capsize=2,
        label="Brute Force", color="#2196F3", edgecolor="white", alpha=0.85)
ax1.bar(x + width/2, lns_scores, width, yerr=lns_errs, capsize=2,
        label="LNS", color="#FF9800", edgecolor="white", alpha=0.85)

ymin = min(min(bf_scores), min(lns_scores)) - 0.05
ymax = max(max(bf_scores), max(lns_scores)) + 0.05
ax1.set_ylim(max(0, ymin), min(1.05, ymax))
ax1.set_ylabel("Final Score", fontsize=12)
ax1.set_title("Brute Force vs LNS — Score Comparison", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=6)
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(axis="y", alpha=0.3)

# Group separators and labels
pos = 0
for name, ids in groups.items():
    count = len([t for t in ids if t in bf_mean and t in lns_mean])
    if count > 0:
        mid = pos + count / 2 - 0.5
        ax1.text(mid, min(1.05, ymax) - 0.005, name, ha="center", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        if pos > 0:
            ax1.axvline(pos - 0.5, color="gray", linestyle="--", alpha=0.4)
        pos += count

# 2) Delta chart with Wilcoxon test
ax2 = axes[1]
deltas = [bf - lns for bf, lns in zip(bf_scores, lns_scores)]
colors = ["#4CAF50" if d > 0.001 else "#F44336" if d < -0.001 else "#9E9E9E" for d in deltas]
ax2.bar(x, deltas, 0.6, color=colors, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Delta (BF - LNS)", fontsize=11)

wins_bf = sum(1 for d in deltas if d > 0.001)
wins_lns = sum(1 for d in deltas if d < -0.001)
ties = sum(1 for d in deltas if abs(d) <= 0.001)

# Wilcoxon signed-rank test
try:
    stat, p_value = stats.wilcoxon(bf_scores, lns_scores)
    p_text = f"p={p_value:.4f}" if p_value >= 0.0001 else f"p<0.0001"
    wilcoxon_text = f" | Wilcoxon: W={stat:.0f}, {p_text}"
except ValueError:
    wilcoxon_text = ""

ax2.set_title(
    f"BF wins {wins_bf}, LNS wins {wins_lns}, ties {ties}, "
    f"avg delta = {np.mean(deltas):+.4f}{wilcoxon_text}",
    fontsize=11
)
ax2.set_xticks(x)
ax2.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=6)
ax2.grid(axis="y", alpha=0.3)

# Add group separators
pos = 0
for name, ids in groups.items():
    count = len([t for t in ids if t in bf_mean and t in lns_mean])
    if count > 0:
        if pos > 0:
            ax2.axvline(pos - 0.5, color="gray", linestyle="--", alpha=0.4)
        pos += count

# 3) Fragility comparison with error bars
ax3 = axes[2]
bf_f = [bf_frag.get(t, 0) for t in task_ids]
lns_f_mean = [lns_frag_mean.get(t, 0) for t in task_ids]
lns_f_err = [lns_frag_std.get(t, 0) for t in task_ids]

ax3.bar(x - width/2, bf_f, width, label="BF fragility", color="#2196F3",
        alpha=0.7, edgecolor="white")
ax3.bar(x + width/2, lns_f_mean, width, yerr=lns_f_err, capsize=2,
        label="LNS fragility", color="#FF9800", alpha=0.7, edgecolor="white")
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Fragility Score", fontsize=11)
ax3.set_title("Fragility Score (1.0 = no violations)", fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=6)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("brute_force_vs_algo/brute_force_comparison.png", dpi=150)
print("Saved brute_force_vs_algo/brute_force_comparison.png")
plt.show()
