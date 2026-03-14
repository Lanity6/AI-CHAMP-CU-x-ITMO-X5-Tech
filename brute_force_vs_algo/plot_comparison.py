#!/usr/bin/env python3
"""Compare brute_force vs solve_x5 scores on small datasets."""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["task_id"]: r.get("metrics", {}).get("final_score", 0.0) for r in data}

# Load all result files
bf_small = load_scores("brute_force_vs_algo/brute_result/res_brute_force.json")
x5_small = load_scores("brute_force_vs_algo/brute_result/res_solve_x5.json")
bf_med = load_scores("brute_force_vs_algo/brute_result/res_medium_brute.json")
x5_med = load_scores("brute_force_vs_algo/brute_result/res_medium_solve_x5.json")

# Merge
bf_all = {**bf_small, **bf_med}
x5_all = {**x5_small, **x5_med}

task_ids = sorted(set(bf_all.keys()) & set(x5_all.keys()))
bf_scores = [bf_all[t] for t in task_ids]
x5_scores = [x5_all[t] for t in task_ids]

# Plot
x = np.arange(len(task_ids))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

# Bar chart
bars_bf = ax1.bar(x - width/2, bf_scores, width, label="Brute Force", color="#2196F3", edgecolor="white")
bars_x5 = ax1.bar(x + width/2, x5_scores, width, label="solve_x5", color="#FF9800", edgecolor="white")

ymin = min(min(bf_scores), min(x5_scores)) - 0.03
ymax = max(max(bf_scores), max(x5_scores)) + 0.03
ax1.set_ylim(ymin, ymax)
ax1.set_ylabel("Final Score")
ax1.set_title("Brute Force vs solve_x5 — Score Comparison (small datasets, exact mode)")
ax1.set_xticks(x)
ax1.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=9)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Add score labels on bars
for bar in bars_bf:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
for bar in bars_x5:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

# Delta chart
deltas = [bf - x5 for bf, x5 in zip(bf_scores, x5_scores)]
colors = ["#4CAF50" if d > 0 else "#F44336" if d < 0 else "#9E9E9E" for d in deltas]
ax2.bar(x, deltas, 0.6, color=colors, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Delta (BF - x5)")
ax2.set_title(f"Advantage: Brute Force wins {sum(1 for d in deltas if d > 0)}/{len(deltas)} tasks, avg delta = {np.mean(deltas):+.4f}")
ax2.set_xticks(x)
ax2.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("brute_force_vs_algo/brute_force_comparison.png", dpi=150)
print("Saved brute_force_vs_algo/brute_force_comparison.png")
plt.show()
