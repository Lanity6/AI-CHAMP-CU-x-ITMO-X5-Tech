#!/usr/bin/env python3
"""Plot average final_score per solver per dataset from benchmark CSV."""

import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark.csv"

df = pd.read_csv(csv_path)
df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

avg = df.groupby(["solver", "file"])["final_score"].mean().reset_index()
pivot = avg.pivot(index="file", columns="solver", values="final_score")

ax = pivot.plot(kind="bar", figsize=(10, 6))
ax.set_xlabel("Dataset")
ax.set_ylabel("Average Score")
ax.set_title("Average Final Score per Solver / Dataset")
ax.set_xticklabels(pivot.index, rotation=30, ha="right")
ax.legend(title="Solver")
ymin = max(0, pivot.min().min() - 0.05)
ymax = min(1, pivot.max().max() + 0.05)
ax.set_ylim(ymin, ymax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
print(f"Saved benchmark_plot.png")
plt.show()
