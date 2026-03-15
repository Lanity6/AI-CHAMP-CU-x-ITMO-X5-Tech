#!/usr/bin/env python3
"""Plot average final_score per solver per dataset from benchmark CSV."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark.csv"

df = pd.read_csv(csv_path)
df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

avg = df.groupby(["solver", "file"])["final_score"].mean().reset_index()
pivot = avg.pivot(index="file", columns="solver", values="final_score")

# Sort solvers (columns) by overall mean score ascending
solver_order = pivot.mean().sort_values(ascending=True).index
pivot = pivot[solver_order]

datasets = pivot.index.tolist()
solvers = pivot.columns.tolist()
n_datasets = len(datasets)
n_solvers = len(solvers)

fig, ax = plt.subplots(figsize=(max(10, n_datasets * 2), 6))

cmap = plt.colormaps.get_cmap("tab10").resampled(n_solvers)
solver_colors = {s: cmap(i) for i, s in enumerate(solvers)}

group_width = 0.8
bar_width = group_width / n_solvers if n_solvers else 0.8

for g_idx, dataset in enumerate(datasets):
    scores = pivot.loc[dataset]
    # Sort solvers by score ascending within this group
    sorted_solvers = scores.sort_values(ascending=True).index.tolist()

    for bar_idx, solver in enumerate(sorted_solvers):
        val = scores[solver]
        if pd.notna(val):
            x = g_idx - group_width / 2 + bar_width * (bar_idx + 0.5)
            ax.bar(x, val, width=bar_width * 0.9,
                   color=solver_colors[solver], edgecolor="white",
                   label=solver if g_idx == 0 else None)

ax.set_xlabel("Dataset")
ax.set_ylabel("Average Score")
ax.set_title("Average Final Score per Solver / Dataset (sorted by score)")
ax.set_xticks(range(n_datasets))
ax.set_xticklabels(datasets, rotation=30, ha="right")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Solver")

ymin = max(0, pivot.min().min() - 0.05)
ymax = min(1, pivot.max().max() + 0.05)
ax.set_ylim(ymin, ymax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
print("Saved benchmark_plot.png")
plt.show()
