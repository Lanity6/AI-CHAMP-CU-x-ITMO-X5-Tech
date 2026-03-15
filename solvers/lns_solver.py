"""
LNS (Large Neighborhood Search) 3D Bin Packing Solver — X5 Tech Smart Packing

Начальное решение: жадный алгоритм с трёхуровневой сортировкой
  Группа 0: обычные коробки (normal)      → кладутся первыми, FFD по объёму
  Группа 1: fragile=True                  → после обычных
  Группа 2: stackable=False               → последними (на самый верх)

Repair:  Beam Search с delta-state + PlacedGrid (пространственная сетка O(k))
Destroy: 5 эвристик с адаптивными весами
"""

import json
import sys
import time
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np

# Импортируем из greedy_algorithm.py
from solvers.greedy_algorithm import (
    get_rotations,
    find_z_by_gravity,
    check_support,
    check_fragility,
    has_3d_collision,
    xy_overlap,
    PlacedGrid,
    _np_find_z,
    _np_collision,
    _np_support,
    _np_fragility,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — все параметры алгоритма
# ──────────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    "time_budget_s":        9.0,
    "early_stop_patience":  15,
    "ils_kicks":            3,      # ILS: перезапусков после early_stop (0=выкл.)

    "destroy_k_fraction":   0.12,
    "destroy_k_min":        3,
    "destroy_k_max":        12,

    "beam_width":           4,
    "beam_eps_limit":       25,
    "repair_queue_limit":   25,

    "repair_restarts":      2,

    # Phase 0: multi-start greedy — перебор случайных порядков внутри групп.
    "greedy_restarts":      300,    # макс. число перезапусков (обрывается по времени)
    "greedy_phase_s":       0.5,    # бюджет Phase 0; короткий → LNS стартует раньше,
                                    # больше шансов найти улучшение до t=1s (time_score=1.0)

    "accept_criterion":     "sa",
    "sa_initial_temp":      0.015,
    "sa_cooling_rate":      0.92,
    "sa_min_temp":          1e-4,

    "adaptive_weights":     True,
    "adaptive_decay":       0.9,
    "adaptive_reward":      1.0,

    "solver_version":       "lns-1.0",
    "verbose":              True,
}

CONFIG_HQ: Dict[str, Any] = {
    "time_budget_s":        28.0,
    "early_stop_patience":  30,
    "ils_kicks":            5,

    "destroy_k_fraction":   0.30,
    "destroy_k_min":        3,
    "destroy_k_max":        30,

    "beam_width":           8,
    "beam_eps_limit":       50,
    "repair_queue_limit":   40,

    "repair_restarts":      6,

    "greedy_restarts":      200,
    "greedy_phase_s":       3.0,    # в HQ больше времени на Phase 0

    "accept_criterion":     "sa",
    "sa_initial_temp":      0.02,
    "sa_cooling_rate":      0.97,
    "sa_min_temp":          1e-4,

    "adaptive_weights":     True,
    "adaptive_decay":       0.9,
    "adaptive_reward":      1.0,

    "solver_version":       "lns-hq-1.0",
    "verbose":              True,
}

# ──────────────────────────────────────────────────────────────────────────────
# Приоритетная сортировка товаров
# ──────────────────────────────────────────────────────────────────────────────

def item_sort_key(item: Dict, all_heavy: bool = False) -> Tuple:
    """3-уровневый приоритет: normal → fragile → non-stackable. FFD внутри группы.

    all_heavy=True: все товары задачи тяжелее 2 кг → на fragile-товар никто не встанет.
    В этом случае fragile переносится в group=2 (как non-stackable), иначе он займёт
    место на полу и заблокирует столбец над собой.
    """
    if not item["stackable"]:
        group = 2
    elif item["fragile"] and all_heavy:
        group = 2  # effectively non-stackable: все соседи >2 кг → ничего сверху не встанет
    elif item["fragile"]:
        group = 1
    else:
        group = 0
    return (group, -(item["l"] * item["w"] * item["h"]))


# ──────────────────────────────────────────────────────────────────────────────
# Жадный солвер с заданным порядком
# Использует PlacedGrid для O(k) геометрии вместо O(n) цикла
# ──────────────────────────────────────────────────────────────────────────────

def greedy_with_order(
    items: List[Dict],
    pallet: Dict,
    prefer_elevated_fragile: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Размещает items в заданном порядке (Extreme Points + Gravity Drop).
    PlacedGrid даёт O(k) проверку геометрии вместо O(n).
    Возвращает (placed_records, unplaced_items).

    prefer_elevated_fragile=True: для fragile-товаров предпочитать z>0 перед z=0.
    Применяется когда все товары тяжелее 2 кг: тогда ничто не встанет на вино,
    и размещение вина на полу блокирует весь столбец над ним.
    """
    PL = pallet["length_mm"]
    PW = pallet["width_mm"]
    PH = pallet["max_height_mm"]
    max_weight = pallet["max_weight_kg"]

    grid = PlacedGrid(PL, PW)
    placed: List[Dict] = []
    unplaced: List[Dict] = []
    total_weight = 0.0

    eps: List[Tuple[int, int, int]] = [(0, 0, 0)]
    eps_set: set = {(0, 0, 0)}

    for item in items:
        if total_weight + item["weight"] > max_weight + 1e-6:
            unplaced.append({**item, "reason": "weight_limit_exceeded"})
            continue

        rotations = get_rotations(item["l"], item["w"], item["h"], item["strict_upright"])
        iw = item["weight"]
        best_score: Optional[Tuple] = None
        best_placement: Optional[Tuple] = None

        for (ex, ey, _ez) in eps:
            for (dl, dw, dh, code) in rotations:
                x2, y2 = ex + dl, ey + dw
                if x2 > PL or y2 > PW:
                    continue

                # Один grid lookup на (ex, ey, dl, dw) — переиспользуем для 4 проверок
                cands = grid.candidates_xy(ex, ey, x2, y2)

                z = grid.find_z(ex, ey, x2, y2, cands)
                if z is None:
                    continue
                if z + dh > PH:
                    continue
                if grid.collision_3d(ex, ey, z, x2, y2, z + dh, cands):
                    continue
                if not grid.check_support(ex, ey, z, x2, y2, dl * dw, cands):
                    continue
                if not grid.check_fragility(ex, ey, z, x2, y2, iw, cands):
                    continue

                # prefer_elevated_fragile: fragile на полу — худший тир (1),
                # fragile повыше и обычные — лучший тир (0). Внутри тира: lowest z wins.
                floor_penalty = (
                    1 if prefer_elevated_fragile and item["fragile"] and z == 0
                    else 0
                )
                score = (floor_penalty, z, ex, ey)
                if best_score is None or score < best_score:
                    best_score = score
                    best_placement = (ex, ey, z, dl, dw, dh, code)

        if best_placement is not None:
            px, py, pz, dl, dw, dh, code = best_placement
            rec: Dict[str, Any] = {
                "sku_id":         item["sku_id"],
                "instance_index": item["instance_index"],
                "x_min": px,       "x_max": px + dl,
                "y_min": py,       "y_max": py + dw,
                "z_min": pz,       "z_max": pz + dh,
                "weight":         item["weight"],
                "fragile":        item["fragile"],
                "stackable":      item["stackable"],
                "rotation_code":  code,
                "l": item["l"], "w": item["w"], "h": item["h"],
                "strict_upright": item["strict_upright"],
            }
            grid.add(rec)
            placed.append(rec)
            total_weight += item["weight"]

            for ep in ((px + dl, py, pz), (px, py + dw, pz), (px, py, pz + dh)):
                if ep not in eps_set:
                    eps_set.add(ep)
                    eps.append(ep)
        else:
            unplaced.append({**item, "reason": "no_space"})

    return placed, unplaced


# ──────────────────────────────────────────────────────────────────────────────
# Формула оценки (совпадает с validator.py)
# ──────────────────────────────────────────────────────────────────────────────

def count_frag_violations(placed: List[Dict]) -> int:
    """
    Считает пары (тяжёлый >2 кг поверх хрупкого).
    Numpy-ускоренная: O(n_tops × n_frags) с numpy вместо O(n²) Python.
    """
    if not placed:
        return 0
    c = np.array(
        [(b["x_min"], b["x_max"], b["y_min"], b["y_max"], b["z_min"], b["z_max"]) for b in placed],
        dtype=np.int64,
    )
    weights = np.array([b["weight"] for b in placed], dtype=np.float64)
    fragile = np.array([b["fragile"] for b in placed], dtype=bool)

    top_mask = weights > 2.0
    if not top_mask.any() or not fragile.any():
        return 0

    tops  = c[top_mask]
    bots  = c[fragile]
    bot_z = bots[:, 5]

    count = 0
    for top in tops:
        level = bot_z == top[4]
        if not level.any():
            continue
        bc = bots[level]
        dx = np.minimum(top[1], bc[:, 1]) - np.maximum(top[0], bc[:, 0])
        dy = np.minimum(top[3], bc[:, 3]) - np.maximum(top[2], bc[:, 2])
        count += int(((dx > 0) & (dy > 0)).sum())
    return count


def compute_score(
    placed: List[Dict],
    pallet: Dict,
    total_items: int,
    elapsed_ms: int,
) -> float:
    """Итоговый балл: 0.5·vol + 0.3·coverage + 0.1·fragility + 0.1·time."""
    pallet_vol = pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    placed_vol = sum(
        (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) * (b["z_max"] - b["z_min"])
        for b in placed
    )
    vol_util = placed_vol / pallet_vol if pallet_vol > 0 else 0.0
    item_cov  = len(placed) / total_items if total_items > 0 else 0.0
    frag_sc   = max(0.0, 1.0 - 0.05 * count_frag_violations(placed))
    if elapsed_ms <= 1000:
        time_sc = 1.0
    elif elapsed_ms <= 5000:
        time_sc = 0.7
    elif elapsed_ms <= 30000:
        time_sc = 0.3
    else:
        time_sc = 0.0
    return 0.50 * vol_util + 0.30 * item_cov + 0.10 * frag_sc + 0.10 * time_sc


def partial_score(
    placed_vol: float,
    placed_count: int,
    frag_violations: int,
    pallet_vol: float,
    total_items: int,
) -> float:
    """Эвристическая оценка частичного состояния для beam search."""
    vol_util = placed_vol / pallet_vol if pallet_vol > 0 else 0.0
    item_cov = placed_count / total_items if total_items > 0 else 0.0
    return 0.50 * vol_util + 0.30 * item_cov - 0.005 * frag_violations


# ──────────────────────────────────────────────────────────────────────────────
# Destroy-эвристики
# ──────────────────────────────────────────────────────────────────────────────

def _compute_k(n_placed: int, config: Dict) -> int:
    k = max(config["destroy_k_min"], int(config["destroy_k_fraction"] * n_placed))
    return min(k, config["destroy_k_max"], n_placed)


def destroy_random(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """Случайное удаление k коробок."""
    indices = set(rng.sample(range(len(placed)), k))
    remaining = [b for i, b in enumerate(placed) if i not in indices]
    removed   = [b for i, b in enumerate(placed) if i in indices]
    return remaining, removed


def destroy_worst_height(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """Удалить k коробок с наибольшим z_max."""
    ranked = sorted(range(len(placed)), key=lambda i: placed[i]["z_max"], reverse=True)
    indices = set(ranked[:k])
    remaining = [b for i, b in enumerate(placed) if i not in indices]
    removed   = [placed[i] for i in ranked[:k]]
    return remaining, removed


def destroy_cluster(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """Удалить k ближайших коробок к случайной коробке-центру. Numpy-ускоренная."""
    seed = placed[rng.randrange(len(placed))]
    cx = (seed["x_min"] + seed["x_max"]) * 0.5
    cy = (seed["y_min"] + seed["y_max"]) * 0.5
    cz = (seed["z_min"] + seed["z_max"]) * 0.5

    centers = np.array(
        [((b["x_min"] + b["x_max"]) * 0.5,
          (b["y_min"] + b["y_max"]) * 0.5,
          (b["z_min"] + b["z_max"]) * 0.5)
         for b in placed],
        dtype=np.float64,
    )
    dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2 + (centers[:, 2] - cz) ** 2
    ranked  = np.argsort(dist2)[:k].tolist()
    indices = set(ranked)
    remaining = [b for i, b in enumerate(placed) if i not in indices]
    removed   = [placed[i] for i in ranked]
    return remaining, removed


def destroy_fragility_violation(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """
    Удалить тяжёлые (>2 кг) коробки, лежащие на хрупких. Numpy-ускоренная.
    Если нарушений меньше k — добить случайными.
    """
    if not placed:
        return placed, []

    c = np.array(
        [(b["x_min"], b["x_max"], b["y_min"], b["y_max"], b["z_min"], b["z_max"]) for b in placed],
        dtype=np.int64,
    )
    weights = np.array([b["weight"] for b in placed], dtype=np.float64)
    fragile = np.array([b["fragile"] for b in placed], dtype=bool)

    viol_indices: set = set()
    top_idx = np.where(weights > 2.0)[0]
    if top_idx.size and fragile.any():
        bots   = c[fragile]
        bot_z  = bots[:, 5]
        for i in top_idx:
            t = c[i]
            level = bot_z == t[4]
            if not level.any():
                continue
            bc = bots[level]
            dx = np.minimum(t[1], bc[:, 1]) - np.maximum(t[0], bc[:, 0])
            dy = np.minimum(t[3], bc[:, 3]) - np.maximum(t[2], bc[:, 2])
            if ((dx > 0) & (dy > 0)).any():
                viol_indices.add(int(i))

    indices = list(viol_indices)
    if len(indices) < k:
        cands = [i for i in range(len(placed)) if i not in viol_indices]
        extra = rng.sample(cands, min(k - len(indices), len(cands)))
        indices.extend(extra)
    else:
        indices = indices[:k]

    index_set = set(indices)
    remaining = [b for i, b in enumerate(placed) if i not in index_set]
    removed   = [placed[i] for i in indices]
    return remaining, removed


def destroy_unplaced_neighbor(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """Удалить коробки с объёмом, близким к среднему объёму непомещённых."""
    if not unplaced:
        return destroy_random(placed, unplaced, k, rng)

    avg_vol = sum(it["l"] * it["w"] * it["h"] for it in unplaced) / len(unplaced)
    inv_avg = 1.0 / max(avg_vol, 1)

    def score(b: Dict) -> float:
        vol = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) * (b["z_max"] - b["z_min"])
        return abs(vol - avg_vol) * inv_avg + b["z_min"] / 10000.0

    ranked  = sorted(range(len(placed)), key=lambda i: score(placed[i]))
    indices = set(ranked[:k])
    remaining = [b for i, b in enumerate(placed) if i not in indices]
    removed   = [placed[i] for i in ranked[:k]]
    return remaining, removed


def destroy_full_restart(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """Полный перезапуск."""
    return [], list(placed)


def destroy_small_placed(
    placed: List[Dict], unplaced: List[Dict], k: int, rng: random.Random
) -> Tuple[List[Dict], List[Dict]]:
    """
    Удалить k наименьших (по объёму) размещённых коробок.
    Даёт beam search шанс вставить крупные непомещённые коробки вместо них.
    Эффективно для сценариев с ограничением по весу (heavy_water).
    """
    ranked = sorted(range(len(placed)),
                    key=lambda i: (placed[i]["x_max"] - placed[i]["x_min"]) *
                                  (placed[i]["y_max"] - placed[i]["y_min"]) *
                                  (placed[i]["z_max"] - placed[i]["z_min"]))
    indices = set(ranked[:k])
    remaining = [b for i, b in enumerate(placed) if i not in indices]
    removed   = [placed[i] for i in ranked[:k]]
    return remaining, removed


DESTROY_HEURISTICS: Dict[str, Any] = {
    "random":              destroy_random,
    "worst_height":        destroy_worst_height,
    "cluster":             destroy_cluster,
    "fragility_violation": destroy_fragility_violation,
    "unplaced_neighbor":   destroy_unplaced_neighbor,
    "full_restart":        destroy_full_restart,
    "small_placed":        destroy_small_placed,
}


def stabilize_placed(placed: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Каскадно удаляет "зависшие" коробки после destroy.
    Numpy-ускоренная: O(n) vectorized per pass вместо O(n²) Python.
    """
    extra_removed: List[Dict] = []
    changed = True
    while changed and placed:
        changed = False
        c = np.array(
            [(b["x_min"], b["x_max"], b["y_min"], b["y_max"], b["z_min"], b["z_max"])
             for b in placed],
            dtype=np.int64,
        )
        stable: List[Dict] = []
        for i, b in enumerate(placed):
            if b["z_min"] == 0:
                stable.append(b)
                continue
            z = b["z_min"]
            mask = c[:, 5] == z
            mask[i] = False
            area = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            if not mask.any() or area == 0:
                extra_removed.append(b)
                changed = True
                continue
            supp = c[mask]
            dx = np.maximum(0, np.minimum(b["x_max"], supp[:, 1]) - np.maximum(b["x_min"], supp[:, 0]))
            dy = np.maximum(0, np.minimum(b["y_max"], supp[:, 3]) - np.maximum(b["y_min"], supp[:, 2]))
            if float((dx * dy).sum()) / area >= 0.60:
                stable.append(b)
            else:
                extra_removed.append(b)
                changed = True
        placed = stable
    return placed, extra_removed


def _select_heuristic(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(weights.values())
    r     = rng.random() * total
    cumul = 0.0
    for h, w in weights.items():
        cumul += w
        if r <= cumul:
            return h
    return list(weights.keys())[-1]


def _update_weights(
    weights: Dict[str, float], heuristic: str, improved: bool, config: Dict
) -> None:
    decay  = config["adaptive_decay"]
    reward = config["adaptive_reward"] if improved else 0.0
    for h in weights:
        weights[h] = decay * weights[h] + (reward if h == heuristic else 0.0)
        weights[h] = max(0.05, weights[h])


# ──────────────────────────────────────────────────────────────────────────────
# Beam Search Repair
#
# Два ключевых ускорения:
# 1. delta-state: каждое состояние луча хранит только маленький delta-список
#    (новые коробки repair), а не копию всего placed_fixed → O(delta) вместо O(N)
# 2. fix_grid: PlacedGrid для placed_fixed (строится один раз) даёт O(k) геометрию
#    для фиксированной части. delta-список мал, проверяется Python-циклом.
# ──────────────────────────────────────────────────────────────────────────────

def _prune_eps(
    eps_set: set, limit: int, PL: int, PW: int, PH: int
) -> List[Tuple[int, int, int]]:
    """Оставить не более limit точек внутри паллеты, сортировка по (z, x, y).
    List comprehension + sort без lambda (tuple natural sort быстрее)."""
    # Перекладываем в (z,x,y) для натуральной сортировки без lambda
    valid = [(z, x, y) for (x, y, z) in eps_set if x < PL and y < PW and z < PH]
    valid.sort()  # натуральная сортировка кортежей = (z, x, y) = нужный порядок
    return [(x, y, z) for (z, x, y) in valid[:limit]]


def _recompute_eps(placed: List[Dict]) -> set:
    eps: set = {(0, 0, 0)}
    for b in placed:
        eps.add((b["x_max"], b["y_min"], b["z_min"]))
        eps.add((b["x_min"], b["y_max"], b["z_min"]))
        eps.add((b["x_min"], b["y_min"], b["z_max"]))
    return eps


def _delta_find_z(
    cx: int, cy: int, x2: int, y2: int,
    fix_grid: PlacedGrid, fix_cands: Set[int],
    delta: List[Dict],
) -> Optional[int]:
    """Gravity drop: O(k) grid для fix + Python-цикл для delta.
    Overlap check: x2>bx1 and bx2>cx (≡ min(x2,bx2)>max(cx,bx1)) — без min/max overhead."""
    z = fix_grid.find_z(cx, cy, x2, y2, fix_cands)
    if z is None:
        return None
    for b in delta:
        if x2 > b["x_min"] and b["x_max"] > cx and y2 > b["y_min"] and b["y_max"] > cy:
            if not b["stackable"]:
                return None
            bz = b["z_max"]
            if bz > z:
                z = bz
    return z


def _delta_collision(
    cx: int, cy: int, cz: int, x2: int, y2: int, z2: int,
    fix_grid: PlacedGrid, fix_cands: Set[int],
    delta: List[Dict],
) -> bool:
    """3D-коллизия: O(k) grid для fix + Python-цикл для delta."""
    if fix_grid.collision_3d(cx, cy, cz, x2, y2, z2, fix_cands):
        return True
    for b in delta:
        if (x2 > b["x_min"] and b["x_max"] > cx and
                y2 > b["y_min"] and b["y_max"] > cy and
                z2 > b["z_min"] and b["z_max"] > cz):
            return True
    return False


def _delta_support(
    cx: int, cy: int, z: int, x2: int, y2: int, area: int,
    fix_grid: PlacedGrid, fix_cands: Set[int],
    delta: List[Dict],
) -> bool:
    """Проверка опоры ≥60%: O(k) grid для fix + Python-цикл для delta."""
    if z == 0:
        return True
    if area == 0:
        return False
    support = 0.0
    boxes = fix_grid.boxes
    for idx in fix_cands:
        b = boxes[idx]
        if b["z_max"] == z:
            dx = min(x2, b["x_max"]) - max(cx, b["x_min"])
            dy = min(y2, b["y_max"]) - max(cy, b["y_min"])
            if dx > 0 and dy > 0:
                support += dx * dy
    for b in delta:
        if b["z_max"] == z:
            dx = min(x2, b["x_max"]) - max(cx, b["x_min"])
            dy = min(y2, b["y_max"]) - max(cy, b["y_min"])
            if dx > 0 and dy > 0:
                support += dx * dy
    return support / area >= 0.60


def _delta_fragility(
    cx: int, cy: int, z: int, x2: int, y2: int, weight: float,
    fix_grid: PlacedGrid, fix_cands: Set[int],
    delta: List[Dict],
) -> bool:
    """Fragility: O(k) grid для fix + Python-цикл для delta."""
    if weight <= 2.0 or z == 0:
        return True
    boxes = fix_grid.boxes
    for idx in fix_cands:
        b = boxes[idx]
        if b["fragile"] and b["z_max"] == z:
            if x2 > b["x_min"] and b["x_max"] > cx and y2 > b["y_min"] and b["y_max"] > cy:
                return False
    for b in delta:
        if b["fragile"] and b["z_max"] == z:
            if x2 > b["x_min"] and b["x_max"] > cx and y2 > b["y_min"] and b["y_max"] > cy:
                return False
    return True


def beam_search_repair(
    placed_fixed: List[Dict],
    items_queue: List[Dict],
    pallet: Dict,
    total_items: int,
    config: Dict,
) -> List[Dict]:
    """
    Beam Search поверх placed_fixed.

    Оптимизации:
    - fix_grid: PlacedGrid для placed_fixed → O(k) геометрия для фиксированной части
    - delta-state: каждый луч хранит только delta (новые коробки), не копируя placed_fixed
    - Один grid lookup на (ex,ey,dl,dw) → переиспользуется для 4 геометрических проверок

    Хрупкость — мягкий штраф (не жёсткий reject).
    """
    PL, PW, PH = pallet["length_mm"], pallet["width_mm"], pallet["max_height_mm"]
    max_weight  = pallet["max_weight_kg"]
    pallet_vol  = PL * PW * PH
    BEAM_W      = config["beam_width"]
    EPS_LIM     = config["beam_eps_limit"]

    # Строим PlacedGrid для placed_fixed ОДИН РАЗ
    fix_grid = PlacedGrid.from_placed(placed_fixed, PL, PW)

    n_fix       = len(placed_fixed)
    init_weight = sum(b["weight"] for b in placed_fixed)
    init_vol    = sum(
        (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) * (b["z_max"] - b["z_min"])
        for b in placed_fixed
    )
    init_eps = _prune_eps(_recompute_eps(placed_fixed), EPS_LIM, PL, PW, PH)

    # Состояние луча: (delta, total_w, eps_list, placed_vol, placed_count, frag_v)
    # delta — маленький список Dict (коробки, добавленные в этом repair)
    # placed_fixed не копируется — доступен через fix_grid
    beams = [([], init_weight, init_eps, init_vol, n_fix, 0)]

    for item in items_queue:
        rotations = get_rotations(item["l"], item["w"], item["h"], item["strict_upright"])
        candidates: List[Tuple] = []
        iw = item["weight"]

        for (delta, total_w, eps_list, placed_vol, placed_count, frag_v) in beams:
            item_placed = False
            weight_ok   = total_w + iw <= max_weight + 1e-6

            if not weight_ok:
                # Вес исчерпан — луч продолжается без этой коробки
                ps = partial_score(placed_vol, placed_count, frag_v, pallet_vol, total_items)
                candidates.append((ps, delta, total_w, eps_list, placed_vol, placed_count, frag_v))
                continue

            for (ex, ey, _) in eps_list:
                for (dl, dw, dh, code) in rotations:
                    x2, y2 = ex + dl, ey + dw
                    if x2 > PL or y2 > PW:
                        continue

                    # Один grid lookup на (ex,ey,dl,dw)
                    fix_cands = fix_grid.candidates_xy(ex, ey, x2, y2)

                    z = _delta_find_z(ex, ey, x2, y2, fix_grid, fix_cands, delta)
                    if z is None:
                        continue
                    if z + dh > PH:
                        continue
                    if _delta_collision(ex, ey, z, x2, y2, z + dh, fix_grid, fix_cands, delta):
                        continue
                    if not _delta_support(ex, ey, z, x2, y2, dl * dw, fix_grid, fix_cands, delta):
                        continue

                    frag_ok    = _delta_fragility(ex, ey, z, x2, y2, iw, fix_grid, fix_cands, delta)
                    new_frag_v = frag_v + (0 if frag_ok else 1)

                    new_box: Dict[str, Any] = {
                        "sku_id":         item["sku_id"],
                        "instance_index": item["instance_index"],
                        "x_min": ex,      "x_max": ex + dl,
                        "y_min": ey,      "y_max": ey + dw,
                        "z_min": z,       "z_max": z  + dh,
                        "weight":         iw,
                        "fragile":        item["fragile"],
                        "stackable":      item["stackable"],
                        "rotation_code":  code,
                        "l": item["l"],   "w": item["w"],   "h": item["h"],
                        "strict_upright": item["strict_upright"],
                    }

                    # delta маленький → copy дёшево
                    new_delta = delta + [new_box]
                    new_vol   = placed_vol + dl * dw * dh
                    new_count = placed_count + 1
                    new_w     = total_w + iw

                    new_eps_set = set(eps_list)
                    new_eps_set.add((ex + dl, ey,      z))
                    new_eps_set.add((ex,      ey + dw, z))
                    new_eps_set.add((ex,      ey,      z + dh))
                    new_eps_list = _prune_eps(new_eps_set, EPS_LIM, PL, PW, PH)

                    ps = partial_score(new_vol, new_count, new_frag_v, pallet_vol, total_items)
                    candidates.append((ps, new_delta, new_w, new_eps_list,
                                       new_vol, new_count, new_frag_v))
                    item_placed = True

            if not item_placed:
                ps = partial_score(placed_vol, placed_count, frag_v, pallet_vol, total_items)
                candidates.append((ps, delta, total_w, eps_list,
                                   placed_vol, placed_count, frag_v))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [c[1:] for c in candidates[:BEAM_W]]

    # Лучший луч: reconstruct placed_fixed + delta (один раз, O(N+delta))
    best = max(
        beams,
        key=lambda b: partial_score(b[3], b[4], b[5], pallet_vol, total_items),
    )
    return placed_fixed + best[0]


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _box_to_item(box: Dict) -> Dict:
    return {
        "sku_id":         box["sku_id"],
        "instance_index": box["instance_index"],
        "l":              box["l"],
        "w":              box["w"],
        "h":              box["h"],
        "weight":         box["weight"],
        "fragile":        box["fragile"],
        "stackable":      box["stackable"],
        "strict_upright": box["strict_upright"],
    }


def _build_output(
    placed: List[Dict],
    unplaced: List[Dict],
    request: Dict,
    solve_time_ms: int,
    solver_version: str,
) -> Dict:
    placements_out = []
    for b in placed:
        dl = b["x_max"] - b["x_min"]
        dw = b["y_max"] - b["y_min"]
        dh = b["z_max"] - b["z_min"]
        placements_out.append({
            "sku_id":         b["sku_id"],
            "instance_index": b["instance_index"],
            "position": {"x_mm": b["x_min"], "y_mm": b["y_min"], "z_mm": b["z_min"]},
            "dimensions_placed": {"length_mm": dl, "width_mm": dw, "height_mm": dh},
            "rotation_code": b["rotation_code"],
        })

    unplaced_map: Dict[str, Dict] = {}
    for it in unplaced:
        rec = unplaced_map.setdefault(
            it["sku_id"], {"count": 0, "reason": it.get("reason", "no_space")}
        )
        rec["count"] += 1
    unplaced_out = [
        {"sku_id": sid, "quantity_unplaced": r["count"], "reason": r["reason"]}
        for sid, r in unplaced_map.items()
    ]

    return {
        "task_id":        request["task_id"],
        "solver_version": solver_version,
        "solve_time_ms":  solve_time_ms,
        "placements":     placements_out,
        "unplaced":       unplaced_out,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Repair: beam search + multi-start wrapper
# ──────────────────────────────────────────────────────────────────────────────

def _perturb_order(
    items: List[Dict],
    rng: random.Random,
    sort_key=None,
) -> List[Dict]:
    """Случайная перестановка ВНУТРИ каждой приоритетной группы."""
    if sort_key is None:
        sort_key = item_sort_key
    groups: Dict[int, List[Dict]] = {}
    for it in items:
        g = sort_key(it)[0]
        groups.setdefault(g, []).append(it)
    result: List[Dict] = []
    for g in sorted(groups.keys()):
        bucket = list(groups[g])
        rng.shuffle(bucket)
        result.extend(bucket)
    return result


def _score_placed(placed: List[Dict], pallet_vol: float, total_items: int) -> float:
    """Быстрая оценка placed для выбора лучшего repair-результата."""
    if not placed:
        return 0.0
    vol = sum(
        (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) * (b["z_max"] - b["z_min"])
        for b in placed
    )
    fv = count_frag_violations(placed)
    return partial_score(vol, len(placed), fv, pallet_vol, total_items)


def repair(
    placed_fixed: List[Dict],
    items_to_repair: List[Dict],
    pallet: Dict,
    total_items: int,
    config: Dict,
    rng: random.Random,
) -> List[Dict]:
    """Repair: beam search (основной) + N_restarts перестановок очереди."""
    pallet_vol = pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]

    best_placed = beam_search_repair(placed_fixed, items_to_repair, pallet, total_items, config)
    best_sc = _score_placed(best_placed, pallet_vol, total_items)

    for _ in range(config.get("repair_restarts", 3)):
        shuffled  = _perturb_order(items_to_repair, rng)
        candidate = beam_search_repair(placed_fixed, shuffled, pallet, total_items, config)
        sc = _score_placed(candidate, pallet_vol, total_items)
        if sc > best_sc:
            best_sc     = sc
            best_placed = candidate

    return best_placed


# ──────────────────────────────────────────────────────────────────────────────
# Главная функция LNS
# ──────────────────────────────────────────────────────────────────────────────

def solve_lns(request: Dict, config: Dict = CONFIG) -> Dict:
    t_start = time.time()
    rng = random.Random(42)

    pallet = request["pallet"]
    total_items_count = sum(b["quantity"] for b in request["boxes"])

    items: List[Dict] = []
    for box in request["boxes"]:
        for idx in range(box["quantity"]):
            items.append({
                "sku_id":         box["sku_id"],
                "instance_index": idx,
                "l":              box["length_mm"],
                "w":              box["width_mm"],
                "h":              box["height_mm"],
                "weight":         box["weight_kg"],
                "strict_upright": box["strict_upright"],
                "fragile":        box["fragile"],
                "stackable":      box.get("stackable", True),
            })

    # Если все товары тяжелее 2 кг, fragile-товары фактически нестекируемые:
    # ни один соседний товар не встанет на fragile, поэтому они должны укладываться
    # ПОСЛЕДНИМИ (как non-stackable), иначе займут пол и заблокируют пространство над собой.
    all_heavy = all(it["weight"] > 2.0 for it in items)
    _sort_key = lambda it: item_sort_key(it, all_heavy)

    # ── Фаза 1: Начальное жадное решение ──────────────────────────────────────
    items.sort(key=_sort_key)
    best_placed, best_unplaced = greedy_with_order(items, pallet, prefer_elevated_fragile=all_heavy)
    best_time_ms = int((time.time() - t_start) * 1000)
    best_score   = compute_score(best_placed, pallet, total_items_count, best_time_ms)
    greedy_score = best_score

    if config["verbose"]:
        print(
            f"[LNS] {request['task_id']} | "
            f"Greedy: score={best_score:.4f}  "
            f"placed={len(best_placed)}/{total_items_count}  "
            f"time={best_time_ms}ms",
            file=sys.stderr,
        )

    # ── Фаза 0: Multi-start greedy — перебор случайных порядков внутри групп ──
    #
    # Deterministic restarts: explore weight-ordered and alternative-group greedy variants.
    #
    # solver_sort: solver.py-style grouping ((not stackable) or fragile → last).
    #   Лучше для stackable=False+non-fragile (water) которые в solver.py попадают
    #   в ту же группу, что и fragile-товары, и кладутся раньше.
    # light-first: лёгкие товары в начале группы → лучше item_coverage при
    #              weight-constrained сценариях (heavy_water).
    # heavy-first: тяжёлые товары в начале группы → тяжёлые (и нестекируемые)
    #              укладываются сначала, лёгкие fragile складываются поверх
    #              (fragile_tower: eggs first → chips stack on eggs).
    def _solver_sort_key(it: Dict) -> Tuple:
        is_special = (not it["stackable"]) or it["fragile"]
        return (1 if is_special else 0, -(it["l"] * it["w"] * it["h"]))

    det_restarts = [
        # solver.py-стиль: (fragile OR non-stackable) → группа 1 вместе
        ("solver",    lambda it: _solver_sort_key(it)),
        # light-first: лёгкие сначала → больше штук при weight-limit
        ("lf",        lambda it: (_sort_key(it)[0],  it["weight"], -(it["l"]*it["w"]*it["h"]))),
        # heavy-first: тяжёлые сначала → лёгкие fragile кладутся поверх
        ("hf",        lambda it: (_sort_key(it)[0], -it["weight"], -(it["l"]*it["w"]*it["h"]))),
        # footprint-desc: большой след сначала → лучше покрытие пола
        ("footprint", lambda it: (_sort_key(it)[0], -(it["l"]*it["w"]), -it["h"])),
        # height-asc: низкие сначала → больше слоёв
        ("height",    lambda it: (_sort_key(it)[0],  it["h"], -(it["l"]*it["w"]*it["h"]))),
    ]
    for _tag, _key in det_restarts:
        items_ord = sorted(items, key=_key)
        placed_ord, unplaced_ord = greedy_with_order(items_ord, pallet, prefer_elevated_fragile=all_heavy)
        t_ms_ord  = int((time.time() - t_start) * 1000)
        score_ord = compute_score(placed_ord, pallet, total_items_count, t_ms_ord)
        if score_ord > best_score + 1e-6:
            best_placed, best_unplaced = placed_ord, unplaced_ord
            best_score, best_time_ms   = score_ord, t_ms_ord
            if config["verbose"]:
                d = score_ord - greedy_score
                print(f"[LNS]   {_tag}_restart  greedy+  score={score_ord:.4f} ({d:+.4f} vs orig)",
                      file=sys.stderr)

    greedy_restarts = config.get("greedy_restarts", 100)
    t_phase0_end    = t_start + config.get("greedy_phase_s", 0.9)
    n_restarts_done = 0
    for restart_i in range(1, greedy_restarts + 1):
        if time.time() > t_phase0_end:
            break
        rng_r   = random.Random(restart_i * 31 + 7)
        items_r = _perturb_order(items, rng_r, _sort_key)
        placed_r, unplaced_r = greedy_with_order(items_r, pallet, prefer_elevated_fragile=all_heavy)
        t_ms    = int((time.time() - t_start) * 1000)
        score_r = compute_score(placed_r, pallet, total_items_count, t_ms)
        n_restarts_done += 1
        if score_r > best_score + 1e-6:
            best_placed   = placed_r
            best_unplaced = unplaced_r
            best_score    = score_r
            best_time_ms  = t_ms
            if config["verbose"]:
                d = score_r - greedy_score
                sign = "+" if d >= 0 else ""
                print(
                    f"[LNS]   restart={restart_i}  greedy+  "
                    f"score={score_r:.4f} ({sign}{d:.4f} vs orig)",
                    file=sys.stderr,
                )
    if config["verbose"]:
        print(
            f"[LNS]   phase0: {n_restarts_done} restarts, best={best_score:.4f}",
            file=sys.stderr,
        )
    greedy_score = best_score  # дельта LNS считается от лучшего greedy (включая рестарты)

    t_deadline    = t_start + config["time_budget_s"]
    SAFETY_MARGIN = 0.35

    heuristic_weights = {h: 1.0 for h in DESTROY_HEURISTICS}
    iteration  = 0
    no_improve = 0
    n_kicks    = 0
    sa_temp    = config["sa_initial_temp"]

    # SA требует разделения «текущего» и «лучшего» решений:
    # cur_* — рабочее состояние (SA может принять ухудшение)
    # best_* — лучшее найденное (только улучшения)
    cur_placed, cur_unplaced, cur_score = best_placed, best_unplaced, best_score

    # ── Фаза 2: LNS-цикл ──────────────────────────────────────────────────────
    while time.time() < t_deadline - SAFETY_MARGIN:
        if not cur_placed:
            break
        iteration += 1

        if config["adaptive_weights"]:
            hname = _select_heuristic(heuristic_weights, rng)
        else:
            hname = rng.choice(list(DESTROY_HEURISTICS.keys()))

        k = _compute_k(len(cur_placed), config)

        # ── Специальная ветка: full_restart ───────────────────────────────────
        # Если задача маленькая (n_placed ≤ q_limit): beam search — лучший поиск.
        # Если задача большая (n_placed > q_limit): beam берёт только q_limit items
        # и выбрасывает остальные → катастрофа. Используем жадный рестарт.
        if hname == "full_restart":
            q_limit_fr  = config.get("repair_queue_limit", 25)
            all_items = sorted(
                [_box_to_item(b) for b in cur_placed] + cur_unplaced,
                key=_sort_key,
            )
            if len(cur_placed) <= q_limit_fr:
                # Маленькая задача: beam search на всех items (старое поведение)
                items_r = _perturb_order(all_items, rng, _sort_key)
                items_r.sort(key=_sort_key)
                new_placed_fr = repair([], items_r, pallet, total_items_count, config, rng)
                new_unplaced_fr_keys = {(b["sku_id"], b["instance_index"]) for b in new_placed_fr}
                new_unplaced_fr = [it for it in all_items
                                   if (it["sku_id"], it["instance_index"]) not in new_unplaced_fr_keys]
            else:
                # Большая задача: полный жадный рестарт с перемешанным порядком
                items_r = _perturb_order(all_items, rng, _sort_key)
                new_placed_fr, new_unplaced_fr = greedy_with_order(items_r, pallet, prefer_elevated_fragile=all_heavy)
            elapsed_ms_fr = int((time.time() - t_start) * 1000)
            new_score_fr  = compute_score(new_placed_fr, pallet, total_items_count, elapsed_ms_fr)

            accepted_fr      = False
            best_improved_fr = new_score_fr > best_score + 1e-6
            if config["accept_criterion"] == "sa":
                if new_score_fr >= cur_score - 1e-9:
                    accepted_fr = True
                elif sa_temp > config["sa_min_temp"]:
                    prob = math.exp((new_score_fr - cur_score) / sa_temp)
                    accepted_fr = rng.random() < prob
            else:
                accepted_fr = best_improved_fr
            sa_temp = max(sa_temp * config["sa_cooling_rate"], config["sa_min_temp"])

            if accepted_fr:
                cur_placed, cur_unplaced, cur_score = new_placed_fr, new_unplaced_fr, new_score_fr
            if best_improved_fr:
                best_placed, best_unplaced = new_placed_fr, new_unplaced_fr
                best_score, best_time_ms   = new_score_fr, elapsed_ms_fr
                no_improve = 0
                if config["verbose"]:
                    d = new_score_fr - greedy_score
                    print(
                        f"[LNS]   iter={iteration:3d}  IMPROVED  "
                        f"score={new_score_fr:.4f} ({d:+.4f} vs greedy)  "
                        f"placed={len(new_placed_fr)}/{total_items_count}  "
                        f"heuristic=full_restart",
                        file=sys.stderr,
                    )
            else:
                no_improve += 1
            if config["adaptive_weights"]:
                _update_weights(heuristic_weights, hname, best_improved_fr, config)
            if no_improve >= config["early_stop_patience"]:
                max_kicks = config.get("ils_kicks", 0)
                if n_kicks >= max_kicks or time.time() >= t_deadline - SAFETY_MARGIN:
                    if config["verbose"]:
                        print(
                            f"[LNS]   iter={iteration:3d}  early_stop "
                            f"(patience={config['early_stop_patience']}, kicks={n_kicks})",
                            file=sys.stderr,
                        )
                    break
                n_kicks  += 1
                no_improve = 0
                sa_temp    = config["sa_initial_temp"]
                all_items_kick = sorted(
                    [_box_to_item(b) for b in best_placed] + best_unplaced,
                    key=_sort_key,
                )
                items_kick_r = _perturb_order(all_items_kick, rng, _sort_key)
                cur_placed, cur_unplaced = greedy_with_order(items_kick_r, pallet, prefer_elevated_fragile=all_heavy)
                _t = int((time.time() - t_start) * 1000)
                cur_score = compute_score(cur_placed, pallet, total_items_count, _t)
                if config["verbose"]:
                    print(
                        f"[LNS]   iter={iteration:3d}  ILS_kick={n_kicks}  "
                        f"restart_score={cur_score:.4f}",
                        file=sys.stderr,
                    )
            continue  # ← skip standard destroy+repair pipeline for this iteration
        # ── Конец ветки full_restart ──────────────────────────────────────────

        remaining, removed = DESTROY_HEURISTICS[hname](cur_placed, cur_unplaced, k, rng)

        remaining, cascade_removed = stabilize_placed(remaining)
        removed = removed + cascade_removed

        removed_all    = [_box_to_item(b) for b in removed]
        unplaced_items = sorted(cur_unplaced, key=_sort_key)
        q_limit        = config.get("repair_queue_limit", 25)

        # Cap removed items to q_limit (largest-volume first, so big items are repaired)
        removed_sorted   = sorted(removed_all, key=lambda x: -(x["l"] * x["w"] * x["h"]))
        removed_in_q     = removed_sorted[:q_limit]
        removed_overflow = removed_sorted[q_limit:]   # will go to new_unplaced

        extra_slots     = max(0, q_limit - len(removed_in_q))
        items_to_repair = removed_in_q + unplaced_items[:extra_slots]
        items_to_repair.sort(key=_sort_key)

        new_placed = repair(
            remaining, items_to_repair, pallet, total_items_count, config, rng
        )

        placed_keys = {(b["sku_id"], b["instance_index"]) for b in new_placed}
        new_unplaced = (
            [
                {**it, "reason": it.get("reason", "no_space")}
                for it in items_to_repair
                if (it["sku_id"], it["instance_index"]) not in placed_keys
            ]
            + [{"sku_id": it["sku_id"], "instance_index": it["instance_index"],
                "l": it["l"], "w": it["w"], "h": it["h"],
                "weight": it["weight"], "fragile": it["fragile"],
                "stackable": it["stackable"], "strict_upright": it["strict_upright"],
                "reason": "no_space"}
               for it in removed_overflow]
            + [
                {**it, "reason": it.get("reason", "no_space")}
                for it in unplaced_items[extra_slots:]
            ]
        )

        elapsed_ms = int((time.time() - t_start) * 1000)
        new_score  = compute_score(new_placed, pallet, total_items_count, elapsed_ms)

        # ── Критерий принятия ─────────────────────────────────────────────────
        # accepted: обновляем текущее рабочее решение (для следующей итерации)
        # best_improved: обновляем глобальный лучший результат
        accepted      = False
        best_improved = new_score > best_score + 1e-6

        if config["accept_criterion"] == "hill_climb":
            accepted = best_improved
        elif config["accept_criterion"] == "sa":
            if new_score >= best_score - 1e-9:
                accepted = True
            elif sa_temp > config["sa_min_temp"]:
                prob = math.exp((new_score - best_score) / sa_temp)
                accepted = rng.random() < prob
            sa_temp = max(sa_temp * config["sa_cooling_rate"], config["sa_min_temp"])

        # Обновляем текущее рабочее решение (SA может принять ухудшение)
        if accepted:
            cur_placed   = new_placed
            cur_unplaced = new_unplaced
            cur_score    = new_score

        # Обновляем глобальный лучший результат
        if best_improved:
            best_placed   = new_placed
            best_unplaced = new_unplaced
            best_score    = new_score
            best_time_ms  = elapsed_ms

        # early_stop: считаем только итерации БЕЗ реального улучшения лучшего решения
        if best_improved:
            no_improve = 0
            if config["verbose"]:
                delta = new_score - greedy_score
                sign  = "+" if delta >= 0 else ""
                print(
                    f"[LNS]   iter={iteration:3d}  IMPROVED  "
                    f"score={new_score:.4f} ({sign}{delta:.4f} vs greedy)  "
                    f"placed={len(new_placed)}/{total_items_count}  "
                    f"heuristic={hname}",
                    file=sys.stderr,
                )
        else:
            no_improve += 1

        if config["adaptive_weights"]:
            _update_weights(heuristic_weights, hname, best_improved, config)

        if no_improve >= config["early_stop_patience"]:
            max_kicks = config.get("ils_kicks", 0)
            if n_kicks >= max_kicks or time.time() >= t_deadline - SAFETY_MARGIN:
                if config["verbose"]:
                    print(
                        f"[LNS]   iter={iteration:3d}  early_stop "
                        f"(patience={config['early_stop_patience']}, kicks={n_kicks})",
                        file=sys.stderr,
                    )
                break
            n_kicks  += 1
            no_improve = 0
            sa_temp    = config["sa_initial_temp"]
            all_items_kick = sorted(
                [_box_to_item(b) for b in best_placed] + best_unplaced,
                key=_sort_key,
            )
            items_kick_r = _perturb_order(all_items_kick, rng, _sort_key)
            cur_placed, cur_unplaced = greedy_with_order(items_kick_r, pallet, prefer_elevated_fragile=all_heavy)
            _t = int((time.time() - t_start) * 1000)
            cur_score = compute_score(cur_placed, pallet, total_items_count, _t)
            if config["verbose"]:
                print(
                    f"[LNS]   iter={iteration:3d}  ILS_kick={n_kicks}  "
                    f"restart_score={cur_score:.4f}",
                    file=sys.stderr,
                )

    # ── Фаза 3: Post-fill — вставка непомещённых товаров в оставшееся пространство ──
    # Запускаем один beam-search-repair поверх best_placed с очередью из best_unplaced.
    # Стоит недорого (~0.1–0.3s) и может добрать несколько непомещённых штук.
    if best_unplaced:
        unplaced_sorted_pf = sorted(best_unplaced, key=_sort_key)
        q_pf = config.get("repair_queue_limit", 25)
        fill_placed = beam_search_repair(
            best_placed, unplaced_sorted_pf[:q_pf], pallet, total_items_count, config
        )
        fill_ms    = int((time.time() - t_start) * 1000)
        fill_score = compute_score(fill_placed, pallet, total_items_count, fill_ms)
        if fill_score > best_score + 1e-6:
            placed_keys_pf = {(b["sku_id"], b["instance_index"]) for b in fill_placed}
            best_unplaced  = [
                it for it in best_unplaced
                if (it["sku_id"], it["instance_index"]) not in placed_keys_pf
            ]
            best_placed   = fill_placed
            best_score    = fill_score
            best_time_ms  = fill_ms
            if config["verbose"]:
                d = fill_score - greedy_score
                print(
                    f"[LNS]   post_fill  score={fill_score:.4f} ({d:+.4f} vs greedy)  "
                    f"placed={len(best_placed)}/{total_items_count}",
                    file=sys.stderr,
                )

    # ── Финал ─────────────────────────────────────────────────────────────────
    if config["verbose"]:
        final_sc = compute_score(best_placed, pallet, total_items_count, best_time_ms)
        total_ms = int((time.time() - t_start) * 1000)
        delta    = final_sc - greedy_score
        sign     = "+" if delta >= 0 else ""
        print(
            f"[LNS] FINAL: score={final_sc:.4f} ({sign}{delta:.4f} vs greedy)  "
            f"placed={len(best_placed)}/{total_items_count}  "
            f"iters={iteration}  best_time={best_time_ms}ms  total={total_ms}ms",
            file=sys.stderr,
        )

    return _build_output(best_placed, best_unplaced, request, best_time_ms,
                         config["solver_version"])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    from validator import evaluate_solution

    cfg = dict(CONFIG)
    cfg["verbose"] = False

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        tasks = json.load(f)

    resps = []
    ts = 0.0
    vc = 0
    for i, task in enumerate(tasks):
        resp = solve_lns(task, cfg)
        resps.append(resp)
        r = evaluate_solution(task, resp)
        tid = task["task_id"]
        if r["valid"]:
            s = r["final_score"]
            ts += s
            vc += 1
            m = r["metrics"]
            resp["metrics"] = {
                "valid": True, "final_score": s,
                "volume_utilization": m["volume_utilization"],
                "item_coverage": m["item_coverage"],
                "fragility_score": m["fragility_score"],
                "time_score": m["time_score"],
            }
            print(f"[{i+1:4d}/{len(tasks)}] {tid}: score={s:.4f} "
                  f"vol={m['volume_utilization']:.4f} cov={m['item_coverage']:.4f} "
                  f"frag={m['fragility_score']:.4f} time={m['time_score']:.4f} "
                  f"({len(resp['placements'])} placed, {resp['solve_time_ms']}ms)")
        else:
            resp["metrics"] = {"valid": False, "final_score": 0.0, "error": r["error"]}
            print(f"[{i+1:4d}/{len(tasks)}] {tid}: INVALID - {r['error']}")

    avg = ts / len(tasks) if tasks else 0.0
    print(f"\n{'='*60}\nTasks: {len(tasks)} | Valid: {vc} | Avg score: {avg:.4f}")
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(resps, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {sys.argv[2]}")


if __name__ == "__main__":
    main()
