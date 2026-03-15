#!/usr/bin/env python3
"""Generate extended brute-force benchmark datasets (30 tasks per difficulty).

Uses the project's dataset_generation.generator for SKU archetypes and pallets.

Usage:
    PYTHONPATH=. python3 brute_force_vs_algo/generate_brute_datasets.py
"""

import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_generation.generator import (
    PALLETS, FOOD_RETAIL_ARCHETYPES, create_box, set_seed
)

NUM_TASKS = 30
OUT_DIR = Path("brute_force_vs_algo/brute_data")


def _total_items(boxes):
    return sum(b["quantity"] for b in boxes)


def generate_task(task_id: str, target_items_min: int, target_items_max: int,
                  seed: int, max_qty_per_sku: int = 50) -> dict:
    """Generate a single task with total items in [target_items_min, target_items_max].

    Strategy: pick random SKU types and adjust quantities to hit the target range.
    max_qty_per_sku caps quantity per SKU to avoid pathological BB cases.
    """
    set_seed(seed)
    pallet = random.choice(PALLETS)

    keys = list(FOOD_RETAIL_ARCHETYPES.keys())

    # Choose number of SKU types based on difficulty
    if target_items_max <= 10:
        k = random.randint(2, 4)
    elif target_items_max <= 14:
        k = random.randint(3, 5)
    else:
        k = random.randint(3, 6)

    chosen_keys = random.sample(keys, k=min(k, len(keys)))

    # First pass: create boxes with quantity=1 each
    boxes = []
    for key in chosen_keys:
        boxes.append(create_box(key, 1, 1))

    # Distribute remaining items randomly among SKUs (respecting max_qty_per_sku)
    current = _total_items(boxes)
    target = random.randint(target_items_min, target_items_max)
    remaining = target - current

    while remaining > 0:
        # Only add to SKUs that haven't reached the cap
        eligible = [i for i in range(len(boxes)) if boxes[i]["quantity"] < max_qty_per_sku]
        if not eligible:
            break
        idx = random.choice(eligible)
        max_add = min(remaining, max_qty_per_sku - boxes[idx]["quantity"])
        add = min(max_add, random.randint(1, max(1, remaining)))
        boxes[idx]["quantity"] += add
        remaining -= add

    return {
        "task_id": task_id,
        "pallet": {
            "type_id": pallet["id"],
            "length_mm": pallet["length_mm"],
            "width_mm": pallet["width_mm"],
            "max_height_mm": pallet["max_height_mm"],
            "max_weight_kg": pallet["max_weight_kg"],
        },
        "boxes": boxes
    }


def generate_dataset(prefix: str, items_min: int, items_max: int,
                     base_seed: int, max_qty_per_sku: int = 50) -> list:
    tasks = []
    for i in range(1, NUM_TASKS + 1):
        tid = f"{prefix}_{i:03d}"
        task = generate_task(tid, items_min, items_max, seed=base_seed + i,
                             max_qty_per_sku=max_qty_per_sku)
        actual = _total_items(task["boxes"])
        tasks.append(task)
        print(f"  {tid}: {len(task['boxes'])} SKU types, {actual} total items")
    return tasks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        # (name, items_min, items_max, base_seed, max_qty_per_sku)
        ("small_test", 6, 10, 1000, 4),
        ("medium_test", 8, 10, 2000, 4),
        ("hard_test", 18, 25, 3000, 8),
    ]

    for name, lo, hi, base_seed, max_qty in configs:
        print(f"\n=== Generating {name}.json ({NUM_TASKS} tasks, {lo}-{hi} items) ===")
        tasks = generate_dataset(name, lo, hi, base_seed, max_qty_per_sku=max_qty)

        counts = [_total_items(t["boxes"]) for t in tasks]
        print(f"  Items range: {min(counts)}-{max(counts)}, "
              f"mean={sum(counts)/len(counts):.1f}")

        path = OUT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {path}")

    print("\nDone! Generated 3 datasets with 30 tasks each.")


if __name__ == "__main__":
    main()
