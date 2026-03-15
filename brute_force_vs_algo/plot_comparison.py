#!/usr/bin/env python3
"""Compare brute_force vs solve_x5 scores across all test datasets."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["task_id"]: r.get("metrics", {}).get("final_score", 0.0) for r in data}

def load_frag(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["task_id"]: r.get("metrics", {}).get("fragility_score", 0.0) for r in data}

# Load all result files
result_dir = Path("brute_force_vs_algo/brute_result")

bf_all = {}
x5_all = {}
bf_frag = {}
x5_frag = {}

pairs = [
    ("res_brute_force.json", "res_solve_x5.json"),
    ("res_medium_brute.json", "res_medium_solve_x5.json"),
    ("res_hard_brute.json", "res_hard_solve_x5.json"),
]

for bf_file, x5_file in pairs:
    bf_path = result_dir / bf_file
    x5_path = result_dir / x5_file
    if bf_path.exists() and x5_path.exists():
        bf_all.update(load_scores(bf_path))
        x5_all.update(load_scores(x5_path))
        bf_frag.update(load_frag(bf_path))
        x5_frag.update(load_frag(x5_path))

# Group by dataset
groups = {
    "Small (6-8 items)\nExact mode": ["small_test_001", "small_test_002"],
    "Medium (9-12 items)\nExact mode": [f"med_{i:03d}" for i in range(1, 11)],
    "Hard (20-25 items)\nLDS mode": [f"hard_{i:03d}" for i in range(1, 11)],
}

task_ids = []
for g in groups.values():
    task_ids.extend([t for t in g if t in bf_all and t in x5_all])

bf_scores = [bf_all[t] for t in task_ids]
x5_scores = [x5_all[t] for t in task_ids]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

x = np.arange(len(task_ids))
width = 0.35

# 1) Score comparison
ax1 = axes[0]
bars_bf = ax1.bar(x - width/2, bf_scores, width, label="Brute Force", color="#2196F3", edgecolor="white")
bars_x5 = ax1.bar(x + width/2, x5_scores, width, label="solve_x5", color="#FF9800", edgecolor="white")

ymin = min(min(bf_scores), min(x5_scores)) - 0.03
ymax = max(max(bf_scores), max(x5_scores)) + 0.03
ax1.set_ylim(ymin, ymax)
ax1.set_ylabel("Final Score")
ax1.set_title("Brute Force vs solve_x5 — Score Comparison", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=7)
ax1.legend(fontsize=11)
ax1.grid(axis="y", alpha=0.3)

# Group separators
pos = 0
for name, ids in groups.items():
    count = len([t for t in ids if t in bf_all and t in x5_all])
    if count > 0:
        mid = pos + count / 2 - 0.5
        ax1.text(mid, ymax - 0.005, name, ha="center", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        if pos > 0:
            ax1.axvline(pos - 0.5, color="gray", linestyle="--", alpha=0.4)
        pos += count

# 2) Delta chart
ax2 = axes[1]
deltas = [bf - x5 for bf, x5 in zip(bf_scores, x5_scores)]
colors = ["#4CAF50" if d > 0.001 else "#F44336" if d < -0.001 else "#9E9E9E" for d in deltas]
ax2.bar(x, deltas, 0.6, color=colors, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Delta (BF - x5)")
wins = sum(1 for d in deltas if d > 0.001)
ties = sum(1 for d in deltas if abs(d) <= 0.001)
ax2.set_title(f"BF wins {wins}/{len(deltas)}, ties {ties}, avg delta = {np.mean(deltas):+.4f}", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=7)
ax2.grid(axis="y", alpha=0.3)

# 3) Fragility comparison
ax3 = axes[2]
bf_f = [bf_frag.get(t, 0) for t in task_ids]
x5_f = [x5_frag.get(t, 0) for t in task_ids]
ax3.bar(x - width/2, bf_f, width, label="BF fragility", color="#2196F3", alpha=0.7, edgecolor="white")
ax3.bar(x + width/2, x5_f, width, label="x5 fragility", color="#FF9800", alpha=0.7, edgecolor="white")
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Fragility Score")
ax3.set_title("Fragility Score (1.0 = no violations)", fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=7)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("brute_force_vs_algo/brute_force_comparison.png", dpi=150)
print("Saved brute_force_vs_algo/brute_force_comparison.png")
plt.show()
