import json
from pathlib import Path

from generator import generate_scenario

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
DATA_DIR.mkdir(exist_ok=True)

SCENARIO_TYPES = ["heavy_water", "fragile_tower", "liquid_tetris", "random_mixed"]
N = 200

NUM_FILES = 3

for file_idx in range(1, NUM_FILES + 1):
    dataset = []
    seed_offset = (file_idx - 1) * N
    for i in range(N):
        global_i = seed_offset + i
        scenario_type = SCENARIO_TYPES[global_i % len(SCENARIO_TYPES)]
        task_id = f"{scenario_type}_{global_i:04d}"
        case = generate_scenario(task_id=task_id, scenario_type=scenario_type, seed=global_i)
        case["pallet"].pop("type_id", None)
        dataset.append(case)

    output_path = DATA_DIR / f"dataset_200_{file_idx}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved: {len(dataset)} cases -> {output_path}")
