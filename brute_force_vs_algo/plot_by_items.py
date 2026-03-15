#!/usr/bin/env python3
"""Plot score delta (BF - x5) vs number of items, sorted by item count."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["task_id"]: r.get("metrics", {}).get("final_score", 0.0) for r in data}

def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["task_id"]: sum(b["quantity"] for b in t["boxes"]) for t in data}

result_dir = Path("brute_force_vs_algo/brute_result")
data_dir = Path("brute_force_vs_algo/brute_data")

# Collect all data
bf_scores = {}
x5_scores = {}
item_counts = {}

pairs = [
    ("small_test.json", "res_brute_force.json", "res_solve_x5.json"),
    ("medium_test.json", "res_medium_brute.json", "res_medium_solve_x5.json"),
    ("hard_test.json", "res_hard_brute.json", "res_hard_solve_x5.json"),
]

for data_file, bf_file, x5_file in pairs:
    item_counts.update(load_tasks(data_dir / data_file))
    bf_scores.update(load_results(result_dir / bf_file))
    x5_scores.update(load_results(result_dir / x5_file))

# Build sorted list by item count
task_ids = sorted(
    set(bf_scores.keys()) & set(x5_scores.keys()) & set(item_counts.keys()),
    key=lambda t: (item_counts[t], t)
)

counts = [item_counts[t] for t in task_ids]
deltas = [bf_scores[t] - x5_scores[t] for t in task_ids]
bf_vals = [bf_scores[t] for t in task_ids]
x5_vals = [x5_scores[t] for t in task_ids]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1.5]})

x = np.arange(len(task_ids))
width = 0.35

# Top: scores side by side, sorted by item count
ax1.bar(x - width/2, bf_vals, width, label="Brute Force", color="#2196F3", edgecolor="white")
ax1.bar(x + width/2, x5_vals, width, label="solve_x5", color="#FF9800", edgecolor="white")

labels = [f"{task_ids[i]}\n({counts[i]} items)" for i in range(len(task_ids))]
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ymin = min(min(bf_vals), min(x5_vals)) - 0.03
ymax = max(max(bf_vals), max(x5_vals)) + 0.03
ax1.set_ylim(ymin, ymax)
ax1.set_ylabel("Final Score")
ax1.set_title("Scores sorted by number of items (left = fewer items)", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Bottom: delta vs item count as scatter + trend
ax2.bar(x, deltas, 0.6,
        color=["#4CAF50" if d > 0.001 else "#F44336" if d < -0.001 else "#9E9E9E" for d in deltas],
        edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.5)

# Trend line
z = np.polyfit(counts, deltas, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(counts), max(counts), 100)
ax2.plot(
    # map x_trend back to bar positions
    np.interp(x_trend, counts, x),
    p(x_trend),
    color="red", linewidth=2, linestyle="--", label=f"Trend: delta = {z[0]:.4f} × items + {z[1]:.4f}"
)

ax2.set_xticks(x)
ax2.set_xticklabels([str(c) for c in counts], fontsize=8)
ax2.set_xlabel("Number of items")
ax2.set_ylabel("Delta (BF - x5)")
ax2.set_title("Score advantage of Brute Force grows with item count", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("brute_force_vs_algo/brute_force_by_items.png", dpi=150)
print("Saved brute_force_vs_algo/brute_force_by_items.png")
plt.show()
