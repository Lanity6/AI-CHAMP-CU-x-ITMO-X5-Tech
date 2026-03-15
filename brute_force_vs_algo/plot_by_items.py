#!/usr/bin/env python3
"""Plot score delta (BF - LNS) vs number of items, sorted by item count.

Supports multiple LNS runs for error bars and confidence bands.

Usage:
    python brute_force_vs_algo/plot_by_items.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["task_id"]: r.get("metrics", {}).get("final_score", 0.0) for r in data}


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["task_id"]: sum(b["quantity"] for b in t["boxes"]) for t in data}


def load_multi_run(result_dir, prefix, n_runs=5):
    """Load scores from multiple runs. Returns {task_id: [scores]}."""
    multi = {}
    for i in range(1, n_runs + 1):
        path = result_dir / f"res_{prefix}_lns_run{i}.json"
        if path.exists():
            scores = load_results(path)
            for tid, s in scores.items():
                multi.setdefault(tid, []).append(s)

    if not multi:
        single = result_dir / f"res_{prefix}_lns.json"
        if single.exists():
            scores = load_results(single)
            for tid, s in scores.items():
                multi[tid] = [s]

    return multi


result_dir = Path("brute_force_vs_algo/brute_result")
data_dir = Path("brute_force_vs_algo/brute_data")

# Collect all data
bf_scores = {}
bf_multi = {}
lns_multi = {}
item_counts = {}

datasets = ["small", "medium", "hard"]

for ds in datasets:
    data_path = data_dir / f"{ds}_test.json"
    if data_path.exists():
        item_counts.update(load_tasks(data_path))

    bf_path = result_dir / f"res_{ds}_brute.json"
    if bf_path.exists():
        bf_scores.update(load_results(bf_path))

    # Multi-run BF (for hard/LDS tasks)
    for i in range(1, 6):
        bf_run = result_dir / f"res_{ds}_brute_run{i}.json"
        if bf_run.exists():
            scores = load_results(bf_run)
            for tid, s in scores.items():
                bf_multi.setdefault(tid, []).append(s)

    lns_multi.update(load_multi_run(result_dir, ds))

# Compute mean/std
bf_mean = {}
bf_std = {}
for tid in bf_scores:
    if tid in bf_multi:
        bf_mean[tid] = np.mean(bf_multi[tid])
        bf_std[tid] = np.std(bf_multi[tid])
    else:
        bf_mean[tid] = bf_scores[tid]
        bf_std[tid] = 0.0

lns_mean = {tid: np.mean(s) for tid, s in lns_multi.items()}
lns_std = {tid: np.std(s) for tid, s in lns_multi.items()}

# Build sorted list by item count
task_ids = sorted(
    set(bf_mean.keys()) & set(lns_mean.keys()) & set(item_counts.keys()),
    key=lambda t: (item_counts[t], t)
)

if not task_ids:
    print("No matching tasks found. Run solvers first.")
    exit(1)

counts = [item_counts[t] for t in task_ids]
bf_vals = [bf_mean[t] for t in task_ids]
lns_vals = [lns_mean[t] for t in task_ids]
bf_errs = [bf_std[t] for t in task_ids]
lns_errs = [lns_std[t] for t in task_ids]
deltas = [bf_mean[t] - lns_mean[t] for t in task_ids]
# Error propagation for delta: sqrt(bf_std^2 + lns_std^2)
delta_errs = [np.sqrt(bf_std[t]**2 + lns_std[t]**2) for t in task_ids]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12),
                                gridspec_kw={"height_ratios": [2, 1.5]})

x = np.arange(len(task_ids))
width = 0.35

# Top: scores side by side with error bars
ax1.bar(x - width/2, bf_vals, width, yerr=bf_errs, capsize=2,
        label="Brute Force", color="#2196F3", edgecolor="white", alpha=0.85)
ax1.bar(x + width/2, lns_vals, width, yerr=lns_errs, capsize=2,
        label="LNS", color="#FF9800", edgecolor="white", alpha=0.85)

labels = [f"{task_ids[i]}\n({counts[i]} items)" for i in range(len(task_ids))]
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
ymin = min(min(bf_vals), min(lns_vals)) - 0.05
ymax = max(max(bf_vals), max(lns_vals)) + 0.05
ax1.set_ylim(max(0, ymin), min(1.05, ymax))
ax1.set_ylabel("Final Score", fontsize=12)
ax1.set_title("BF vs LNS — Scores sorted by item count (left = fewer items)",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(axis="y", alpha=0.3)

# Bottom: delta vs item count with error bars + trend line + confidence band
bar_colors = ["#4CAF50" if d > 0.001 else "#F44336" if d < -0.001 else "#9E9E9E"
              for d in deltas]
ax2.bar(x, deltas, 0.6, yerr=delta_errs, capsize=2,
        color=bar_colors, edgecolor="white", alpha=0.85)
ax2.axhline(0, color="black", linewidth=0.5)

# Trend line with confidence band
counts_arr = np.array(counts, dtype=float)
deltas_arr = np.array(deltas, dtype=float)

slope, intercept, r_value, p_value, std_err = stats.linregress(counts_arr, deltas_arr)

x_trend = np.linspace(counts_arr.min(), counts_arr.max(), 200)
y_trend = slope * x_trend + intercept

# Confidence band (95%)
n = len(counts_arr)
x_mean = counts_arr.mean()
se = np.sqrt(np.sum((deltas_arr - (slope * counts_arr + intercept))**2) / (n - 2))
ci = stats.t.ppf(0.975, n - 2) * se * np.sqrt(1/n + (x_trend - x_mean)**2 / np.sum((counts_arr - x_mean)**2))

# Map x_trend to bar positions for plotting
x_trend_pos = np.interp(x_trend, counts_arr, x)

ax2.plot(x_trend_pos, y_trend, color="red", linewidth=2, linestyle="--",
         label=f"Trend: {slope:+.4f} x items {intercept:+.4f} (R²={r_value**2:.3f}, p={p_value:.4f})")
ax2.fill_between(x_trend_pos, y_trend - ci, y_trend + ci,
                 color="red", alpha=0.1, label="95% CI")

ax2.set_xticks(x)
ax2.set_xticklabels([str(c) for c in counts], fontsize=7)
ax2.set_xlabel("Number of items", fontsize=12)
ax2.set_ylabel("Delta (BF - LNS)", fontsize=11)

# Wilcoxon test
try:
    w_stat, w_p = stats.wilcoxon(bf_vals, lns_vals)
    wilcoxon_text = f" | Wilcoxon p={w_p:.4f}" if w_p >= 0.0001 else " | Wilcoxon p<0.0001"
except ValueError:
    wilcoxon_text = ""

ax2.set_title(
    f"Score delta vs item count — "
    f"BF wins {sum(1 for d in deltas if d > 0.001)}/{len(deltas)}"
    f"{wilcoxon_text}",
    fontsize=12
)
ax2.legend(fontsize=9, loc="best")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("brute_force_vs_algo/brute_force_by_items.png", dpi=150)
print("Saved brute_force_vs_algo/brute_force_by_items.png")
plt.show()
