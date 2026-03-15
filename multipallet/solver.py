"""
Greedy 3D Bin Packing Solver — X5 Tech Smart Packing
Algorithm: Extreme Points + Gravity Drop (Bottom-Left-Fill heuristic)
"""

import json
import math
import time
import argparse
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# Rotation codes: maps (L, W, H) original dims
# to (placed_length, placed_width, placed_height)
# ──────────────────────────────────────────────
ROTATION_CODES: Dict[str, Tuple[int, int, int]] = {
    "LWH": (0, 1, 2),  # z = H
    "LHW": (0, 2, 1),  # z = W
    "WLH": (1, 0, 2),  # z = H
    "WHL": (1, 2, 0),  # z = L
    "HLW": (2, 0, 1),  # z = W
    "HWL": (2, 1, 0),  # z = L
}


@lru_cache(maxsize=2048)
def get_rotations(
    l: int, w: int, h: int, strict_upright: bool
) -> Tuple[Tuple[int, int, int, str], ...]:
    """
    Cached. Returns valid (dl, dw, dh, rotation_code) tuples.
    strict_upright=True: placed height must equal original height_mm.
    Deduplicates identical dimension triples to reduce redundant checks.
    """
    dims = [l, w, h]
    seen: set = set()
    result = []
    for code, (i, j, k) in ROTATION_CODES.items():
        dl, dw, dh = dims[i], dims[j], dims[k]
        if strict_upright and dh != h:
            continue
        key = (dl, dw, dh)
        if key not in seen:
            seen.add(key)
            result.append((dl, dw, dh, code))
    return tuple(result)


# ──────────────────────────────────────────────
# Original Python geometry helpers
# (kept for backward compatibility — imported by lns_solver.py)
# ──────────────────────────────────────────────

def xy_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    """XY overlap area between two rectangles."""
    dx = max(0, min(ax2, bx2) - max(ax1, bx1))
    dy = max(0, min(ay2, by2) - max(ay1, by1))
    return dx * dy


def has_3d_collision(
    cx: int, cy: int, cz: int,
    dl: int, dw: int, dh: int,
    placed: List[Dict[str, Any]]
) -> bool:
    """True if proposed box (cx,cy,cz)+(dl,dw,dh) collides with any placed box."""
    x2, y2, z2 = cx + dl, cy + dw, cz + dh
    for b in placed:
        if (min(x2, b["x_max"]) > max(cx, b["x_min"]) and
                min(y2, b["y_max"]) > max(cy, b["y_min"]) and
                min(z2, b["z_max"]) > max(cz, b["z_min"])):
            return True
    return False


def find_z_by_gravity(
    cx: int, cy: int,
    dl: int, dw: int,
    placed: List[Dict[str, Any]]
) -> Optional[int]:
    """Drop box footprint to the lowest valid z. Returns z (0 = pallet floor)."""
    x2, y2 = cx + dl, cy + dw
    z = 0
    for b in placed:
        if min(x2, b["x_max"]) > max(cx, b["x_min"]) and min(y2, b["y_max"]) > max(cy, b["y_min"]):
            if not b.get("stackable", True):
                return None
            if b["z_max"] > z:
                z = b["z_max"]
    return z


def check_support(
    cx: int, cy: int, z: int,
    dl: int, dw: int,
    placed: List[Dict[str, Any]]
) -> bool:
    """Returns True if ≥60% of box base is supported. z=0 → always True."""
    if z == 0:
        return True
    x2, y2 = cx + dl, cy + dw
    area = dl * dw
    if area == 0:
        return False
    support = 0.0
    for b in placed:
        if b["z_max"] == z:
            support += xy_overlap(cx, cy, x2, y2, b["x_min"], b["y_min"], b["x_max"], b["y_max"])
    return support / area >= 0.60


def check_fragility(
    cx: int, cy: int, z: int,
    dl: int, dw: int,
    item_weight: float,
    placed: List[Dict[str, Any]]
) -> bool:
    """Returns True if placing this box here does NOT create a fragility violation."""
    if item_weight <= 2.0 or z == 0:
        return True
    x2, y2 = cx + dl, cy + dw
    for b in placed:
        if b["fragile"] and b["z_max"] == z:
            if xy_overlap(cx, cy, x2, y2, b["x_min"], b["y_min"], b["x_max"], b["y_max"]) > 0:
                return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Numpy batch geometry (used for count_frag_violations, stabilize_placed, etc.)
# coords: np.ndarray shape (n, 6) int64
#         columns: x_min, x_max, y_min, y_max, z_min, z_max
# ──────────────────────────────────────────────────────────────────────────────

def _np_find_z(
    cx: int, cy: int, dl: int, dw: int,
    coords: np.ndarray, stackable: np.ndarray,
) -> Optional[int]:
    """Vectorized gravity drop over numpy placed array."""
    if len(coords) == 0:
        return 0
    x2, y2 = cx + dl, cy + dw
    ox = np.minimum(x2, coords[:, 1]) - np.maximum(cx, coords[:, 0])
    oy = np.minimum(y2, coords[:, 3]) - np.maximum(cy, coords[:, 2])
    mask = (ox > 0) & (oy > 0)
    if not mask.any():
        return 0
    if not stackable[mask].all():
        return None
    return int(coords[mask, 5].max())


def _np_collision(
    cx: int, cy: int, cz: int, dl: int, dw: int, dh: int,
    coords: np.ndarray,
) -> bool:
    """Vectorized 3-D collision check."""
    if len(coords) == 0:
        return False
    x2, y2, z2 = cx + dl, cy + dw, cz + dh
    ox = np.minimum(x2, coords[:, 1]) - np.maximum(cx, coords[:, 0])
    oy = np.minimum(y2, coords[:, 3]) - np.maximum(cy, coords[:, 2])
    oz = np.minimum(z2, coords[:, 5]) - np.maximum(cz, coords[:, 4])
    return bool(((ox > 0) & (oy > 0) & (oz > 0)).any())


def _np_support(
    cx: int, cy: int, z: int, dl: int, dw: int,
    coords: np.ndarray,
) -> bool:
    """Vectorized support check (≥60% base area supported)."""
    if z == 0:
        return True
    area = dl * dw
    if area == 0:
        return False
    mask = coords[:, 5] == z
    if not mask.any():
        return False
    c = coords[mask]
    dx = np.maximum(0, np.minimum(cx + dl, c[:, 1]) - np.maximum(cx, c[:, 0]))
    dy = np.maximum(0, np.minimum(cy + dw, c[:, 3]) - np.maximum(cy, c[:, 2]))
    return float((dx * dy).sum()) / area >= 0.60


def _np_fragility(
    cx: int, cy: int, z: int, dl: int, dw: int,
    weight: float, coords: np.ndarray, fragile: np.ndarray,
) -> bool:
    """Vectorized fragility check. Returns True if no violation."""
    if weight <= 2.0 or z == 0:
        return True
    mask = fragile & (coords[:, 5] == z)
    if not mask.any():
        return True
    c = coords[mask]
    x2, y2 = cx + dl, cy + dw
    dx = np.minimum(x2, c[:, 1]) - np.maximum(cx, c[:, 0])
    dy = np.minimum(y2, c[:, 3]) - np.maximum(cy, c[:, 2])
    return not bool(((dx > 0) & (dy > 0)).any())


# ──────────────────────────────────────────────────────────────────────────────
# PlacedGrid — пространственная сетка для O(k) геометрических проверок
# Вместо O(n) цикла по всем коробкам используем O(k) проверку кандидатов из сетки,
# где k = среднее число коробок в ячейках, перекрываемых заданным прямоугольником.
# Для паллеты 1200×800 с сеткой 100мм и 100 коробками: k ≈ 5-15 вместо n=100.
# ──────────────────────────────────────────────────────────────────────────────

_GRID_RES = 100  # мм на ячейку


class PlacedGrid:
    """
    Spatial hash grid for fast O(k) box lookups in XY.
    k = average candidates in footprint cells ≪ n.
    """
    __slots__ = ("gx", "gy", "cells", "boxes", "_RES")

    def __init__(self, PL: int, PW: int, resolution: int = _GRID_RES) -> None:
        self._RES = resolution
        self.gx = math.ceil(PL / resolution) + 1
        self.gy = math.ceil(PW / resolution) + 1
        self.cells: List[List[List[int]]] = [
            [[] for _ in range(self.gy)] for _ in range(self.gx)
        ]
        self.boxes: List[Dict] = []

    def add(self, box: Dict) -> None:
        """Register a box in all overlapping grid cells. O(cells in footprint)."""
        idx = len(self.boxes)
        self.boxes.append(box)
        R, gx, gy = self._RES, self.gx, self.gy
        for ix in range(box["x_min"] // R, min((box["x_max"] - 1) // R + 1, gx)):
            row = self.cells[ix]
            for iy in range(box["y_min"] // R, min((box["y_max"] - 1) // R + 1, gy)):
                row[iy].append(idx)

    def candidates_xy(self, cx: int, cy: int, x2: int, y2: int) -> Set[int]:
        """
        Deduplicated set of box indices whose footprint MAY overlap (cx,cy)-(x2,y2).
        O(footprint_cells × boxes_per_cell).
        Uses set.update (batch C-level) for faster deduplication.
        """
        R, gx, gy = self._RES, self.gx, self.gy
        seen: Set[int] = set()
        for ix in range(cx // R, min((x2 - 1) // R + 1, gx)):
            row = self.cells[ix]
            for iy in range(cy // R, min((y2 - 1) // R + 1, gy)):
                seen.update(row[iy])
        return seen

    # ── Geometry methods accept precomputed candidates for reuse ──────────────
    # Overlap check shortcut: min(x2,bx2)>max(cx,bx1) ↔ x2>bx1 and bx2>cx
    # (valid because x2>cx always, bx2>bx1 always for positive-size boxes)

    def find_z(self, cx: int, cy: int, x2: int, y2: int, cands: Set[int]) -> Optional[int]:
        """Gravity drop using precomputed candidate set."""
        z = 0
        boxes = self.boxes
        for idx in cands:
            b = boxes[idx]
            if x2 > b["x_min"] and b["x_max"] > cx and y2 > b["y_min"] and b["y_max"] > cy:
                if not b["stackable"]:
                    return None
                bz = b["z_max"]
                if bz > z:
                    z = bz
        return z

    def collision_3d(
        self, cx: int, cy: int, cz: int, x2: int, y2: int, z2: int, cands: Set[int]
    ) -> bool:
        """3D collision check using precomputed candidates."""
        boxes = self.boxes
        for idx in cands:
            b = boxes[idx]
            if (x2 > b["x_min"] and b["x_max"] > cx and
                    y2 > b["y_min"] and b["y_max"] > cy and
                    z2 > b["z_min"] and b["z_max"] > cz):
                return True
        return False

    def check_support(
        self, cx: int, cy: int, z: int, x2: int, y2: int, area: int, cands: Set[int]
    ) -> bool:
        """Support ≥60% using precomputed candidates."""
        if z == 0:
            return True
        if area == 0:
            return False
        support = 0.0
        boxes = self.boxes
        for idx in cands:
            b = boxes[idx]
            if b["z_max"] == z:
                dx = min(x2, b["x_max"]) - max(cx, b["x_min"])
                dy = min(y2, b["y_max"]) - max(cy, b["y_min"])
                if dx > 0 and dy > 0:
                    support += dx * dy
        return support / area >= 0.60

    def check_fragility(
        self, cx: int, cy: int, z: int, x2: int, y2: int, weight: float, cands: Set[int]
    ) -> bool:
        """Fragility check using precomputed candidates."""
        if weight <= 2.0 or z == 0:
            return True
        boxes = self.boxes
        for idx in cands:
            b = boxes[idx]
            if b["fragile"] and b["z_max"] == z:
                if x2 > b["x_min"] and b["x_max"] > cx and y2 > b["y_min"] and b["y_max"] > cy:
                    return False
        return True

    @classmethod
    def from_placed(cls, placed: List[Dict], PL: int, PW: int) -> "PlacedGrid":
        """Build grid from an existing placed list."""
        g = cls(PL, PW)
        for b in placed:
            g.add(b)
        return g


# ──────────────────────────────────────────────
# Внутренние хелперы солвера
# ──────────────────────────────────────────────

def _expand_and_sort(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand SKU quantities and FFD-sort items."""
    items: List[Dict[str, Any]] = []
    for box in request["boxes"]:
        for idx in range(box["quantity"]):
            items.append({
                "sku_id": box["sku_id"],
                "instance_index": idx,
                "l": box["length_mm"],
                "w": box["width_mm"],
                "h": box["height_mm"],
                "weight": box["weight_kg"],
                "strict_upright": box["strict_upright"],
                "fragile": box["fragile"],
                "stackable": box.get("stackable", True),
            })

    def sort_key(x):
        is_special = (not x["stackable"]) or x["fragile"]
        return (1 if is_special else 0, -(x["l"] * x["w"] * x["h"]))

    items.sort(key=sort_key)
    return items


def _item_impossible(
    item: Dict[str, Any], PL: int, PW: int, PH: int, max_weight: float
) -> Tuple[bool, str]:
    """True if item cannot fit on any fresh empty pallet."""
    if item["weight"] > max_weight + 1e-6:
        return True, "weight_limit_exceeded"
    for (dl, dw, dh, _) in get_rotations(item["l"], item["w"], item["h"], item["strict_upright"]):
        if dl <= PL and dw <= PW and dh <= PH:
            return False, ""
    return True, "no_space"


def _pack_items_onto_pallet(
    items: List[Dict[str, Any]],
    PL: int, PW: int, PH: int,
    max_weight: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
    Pack items onto a single fresh pallet using Extreme Points + Gravity Drop.

    Returns:
        placements_raw  — list of placement dicts (same shape as output placements)
        remaining_items — items not placed; each has "_skip_reason" key
        total_weight    — total weight placed (kg)
    """
    grid = PlacedGrid(PL, PW)
    placements_raw: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    total_weight = 0.0

    eps: List[Tuple[int, int, int]] = [(0, 0, 0)]
    eps_set: set = {(0, 0, 0)}

    for item in items:
        if total_weight + item["weight"] > max_weight + 1e-6:
            remaining.append({**item, "_skip_reason": "weight_limit_exceeded"})
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

                score = (z, ex, ey)
                if best_score is None or score < best_score:
                    best_score = score
                    best_placement = (ex, ey, z, dl, dw, dh, code)

        if best_placement is not None:
            px, py, pz, dl, dw, dh, code = best_placement

            box_record = {
                "sku_id": item["sku_id"],
                "x_min": px,       "x_max": px + dl,
                "y_min": py,       "y_max": py + dw,
                "z_min": pz,       "z_max": pz + dh,
                "weight": item["weight"],
                "fragile": item["fragile"],
                "stackable": item["stackable"],
            }
            grid.add(box_record)
            total_weight += item["weight"]

            for ep in ((px + dl, py, pz), (px, py + dw, pz), (px, py, pz + dh)):
                if ep not in eps_set:
                    eps_set.add(ep)
                    eps.append(ep)

            placements_raw.append({
                "sku_id": item["sku_id"],
                "instance_index": item["instance_index"],
                "position": {"x_mm": px, "y_mm": py, "z_mm": pz},
                "dimensions_placed": {"length_mm": dl, "width_mm": dw, "height_mm": dh},
                "rotation_code": code,
            })
        else:
            remaining.append({**item, "_skip_reason": "no_space"})

    return placements_raw, remaining, total_weight


# ──────────────────────────────────────────────
# Core solver (PlacedGrid geometry + lru_cache rotations + O(1) eps set)
# ──────────────────────────────────────────────

def solve_task(request: Dict[str, Any]) -> Dict[str, Any]:
    t_start = time.time()

    pallet = request["pallet"]
    PL = pallet["length_mm"]
    PW = pallet["width_mm"]
    PH = pallet["max_height_mm"]
    max_weight = pallet["max_weight_kg"]

    items = _expand_and_sort(request)
    placements_raw, unplaced_items, _ = _pack_items_onto_pallet(items, PL, PW, PH, max_weight)

    unplaced_map: Dict[str, Dict[str, Any]] = {}
    for item in unplaced_items:
        rec = unplaced_map.setdefault(
            item["sku_id"], {"count": 0, "reason": item["_skip_reason"]}
        )
        rec["count"] += 1

    unplaced_out = [
        {"sku_id": sku_id, "quantity_unplaced": rec["count"], "reason": rec["reason"]}
        for sku_id, rec in unplaced_map.items()
    ]

    return {
        "task_id": request["task_id"],
        "solver_version": "greedy-1.0",
        "solve_time_ms": int((time.time() - t_start) * 1000),
        "placements": placements_raw,
        "unplaced": unplaced_out,
    }


# ──────────────────────────────────────────────
# Multi-pallet solver
# ──────────────────────────────────────────────

def solve_task_multi(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Multi-pallet greedy solver.
    Opens new pallets as needed until all items are placed or physically impossible.

    Returns a list of solution dicts (same structure as solve_task), one per pallet.
    The last dict's 'unplaced' contains items that can't fit on any pallet.
    """
    t_start = time.time()

    pallet = request["pallet"]
    PL = pallet["length_mm"]
    PW = pallet["width_mm"]
    PH = pallet["max_height_mm"]
    max_weight = pallet["max_weight_kg"]

    items = _expand_and_sort(request)

    # Pre-filter physically impossible items (too big or too heavy for any pallet)
    truly_unplaced: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for item in items:
        impossible, reason = _item_impossible(item, PL, PW, PH, max_weight)
        if impossible:
            truly_unplaced.append({**item, "_skip_reason": reason})
        else:
            remaining.append(item)

    results: List[Dict[str, Any]] = []

    while remaining:
        placements_raw, leftover, _ = _pack_items_onto_pallet(remaining, PL, PW, PH, max_weight)

        # Safety: if nothing placed on a fresh pallet, move remaining to unplaced and stop
        if len(placements_raw) == 0:
            truly_unplaced.extend({**item, "_skip_reason": "no_space"} for item in remaining)
            break

        results.append({
            "task_id": request["task_id"],
            "solver_version": "greedy-1.0-multi",
            "solve_time_ms": int((time.time() - t_start) * 1000),
            "placements": placements_raw,
            "unplaced": [],
        })

        remaining = leftover

    # If nothing was packed at all, create a single empty result
    if not results:
        results.append({
            "task_id": request["task_id"],
            "solver_version": "greedy-1.0-multi",
            "solve_time_ms": int((time.time() - t_start) * 1000),
            "placements": [],
            "unplaced": [],
        })

    # Aggregate truly unplaced into the last pallet's unplaced field
    if truly_unplaced:
        unplaced_map: Dict[str, Dict[str, Any]] = {}
        for item in truly_unplaced:
            rec = unplaced_map.setdefault(
                item["sku_id"], {"count": 0, "reason": item["_skip_reason"]}
            )
            rec["count"] += 1
        results[-1]["unplaced"] = [
            {"sku_id": sku_id, "quantity_unplaced": rec["count"], "reason": rec["reason"]}
            for sku_id, rec in unplaced_map.items()
        ]

    return results


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="X5 Greedy 3D Packing Solver")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single request JSON file")
    group.add_argument("--dataset", help="Dataset JSON file (array of tasks)")
    parser.add_argument("--output", help="Output file (default: response_<task_id>.json)")
    parser.add_argument(
        "--multi-pallet", action="store_true", default=False,
        help="Enable multi-pallet mode: overflow items onto additional pallets"
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            request = json.load(f)

        if args.multi_pallet:
            solutions = solve_task_multi(request)
            out_path = args.output or PROJECT_ROOT / f"response_{request['task_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(solutions, f, indent=2, ensure_ascii=False)

            total_placed = sum(len(s["placements"]) for s in solutions)
            total_unplaced = sum(
                sum(u["quantity_unplaced"] for u in s["unplaced"]) for s in solutions
            )
            total_time = solutions[-1]["solve_time_ms"]
            print(f"Solved '{request['task_id']}' (multi-pallet) -> {out_path}")
            print(f"  Pallets used: {len(solutions)}")
            for i, s in enumerate(solutions):
                placed = len(s["placements"])
                unplaced = sum(u["quantity_unplaced"] for u in s["unplaced"])
                weight = sum(
                    p["dimensions_placed"]["length_mm"] for p in s["placements"]
                )  # placeholder — we don't store weight per placement
                suffix = f"  unplaced={unplaced}" if unplaced else ""
                print(f"  Pallet {i}: placed={placed}{suffix}")
            print(f"  Total placed: {total_placed}")
            print(f"  Total unplaced: {total_unplaced}")
            print(f"  Time:     {total_time} ms")

        else:
            solution = solve_task(request)
            out_path = args.output or PROJECT_ROOT / f"response_{request['task_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(solution, f, indent=2, ensure_ascii=False)
            print(f"Solved '{request['task_id']}' -> {out_path}")
            print(f"  Placed:   {len(solution['placements'])}")
            print(f"  Unplaced: {sum(u['quantity_unplaced'] for u in solution['unplaced'])}")
            print(f"  Time:     {solution['solve_time_ms']} ms")

    elif args.dataset:
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if args.multi_pallet:
            all_solutions = []
            for i, request in enumerate(dataset):
                sols = solve_task_multi(request)
                all_solutions.append(sols)
                placed = sum(len(s["placements"]) for s in sols)
                unplaced = sum(
                    sum(u["quantity_unplaced"] for u in s["unplaced"]) for s in sols
                )
                pallets = len(sols)
                t_ms = sols[-1]["solve_time_ms"]
                print(f"[{i+1:4d}/{len(dataset)}] {request['task_id']:30s}  "
                      f"placed={placed:4d}  unplaced={unplaced:4d}  "
                      f"pallets={pallets:2d}  time={t_ms:4d}ms")
            out_path = args.output or PROJECT_ROOT / "solution_dataset_multi.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_solutions, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(all_solutions)} solutions -> {out_path}")

        else:
            solutions = []
            for i, request in enumerate(dataset):
                sol = solve_task(request)
                solutions.append(sol)
                placed = len(sol["placements"])
                unplaced = sum(u["quantity_unplaced"] for u in sol["unplaced"])
                print(f"[{i+1:4d}/{len(dataset)}] {request['task_id']:30s}  "
                      f"placed={placed:4d}  unplaced={unplaced:4d}  "
                      f"time={sol['solve_time_ms']:4d}ms")
            out_path = args.output or PROJECT_ROOT / "solution_dataset.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(solutions, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(solutions)} solutions → {out_path}")


if __name__ == "__main__":
    main()
