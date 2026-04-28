"""
Brute-Force 3D Bin Packing Solver — conditions Smart Packing
Algorithm: DFS Branch-and-Bound with extreme points.
  - Exact mode (N <= 12): full BB search, optimal guarantee
  - LDS mode (N > 12): limited-discrepancy search, anytime best result
"""

import json
import time
import sys
from typing import List, Dict, Any, Optional, Tuple, Set

# Reuse from greedy_algorithm
from solvers.greedy_algorithm import (
    get_rotations,
    PlacedGrid,
)

EXACT_THRESHOLD = 12  # items <= this → full BB; above → LDS
LDS_TIME_LIMIT_S = 25.0  # anytime budget for LDS mode
LDS_MAX_DISCREPANCIES = 3  # max deviations from greedy order


# ──────────────────────────────────────────────────────────────────────────────
# BacktrackablePacker — PlacedGrid wrapper with undo support
# ──────────────────────────────────────────────────────────────────────────────

class BacktrackablePacker:
    """
    Wraps PlacedGrid to support add/undo_last for DFS backtracking.
    Tracks which grid cells each box occupies so we can remove it.
    """

    def __init__(self, PL: int, PW: int, PH: int, max_weight: float):
        self.PL = PL
        self.PW = PW
        self.PH = PH
        self.max_weight = max_weight
        self.grid = PlacedGrid(PL, PW)
        self.total_weight = 0.0
        # Stack of (box_record, cell_entries) for undo
        self._history: List[Tuple[Dict, List[Tuple[int, int]]]] = []
        # Extreme points with undo stack
        self.eps: List[Tuple[int, int, int]] = [(0, 0, 0)]
        self.eps_set: Set[Tuple[int, int, int]] = {(0, 0, 0)}
        self._eps_len_stack: List[int] = []

    def can_place(
        self, cx: int, cy: int, dl: int, dw: int, dh: int, weight: float,
        fragile: bool, cands: Optional[Set[int]] = None
    ) -> Optional[int]:
        """
        Check if box can be placed at (cx, cy) with gravity drop.
        Returns z coordinate if valid, None otherwise.
        """
        x2, y2 = cx + dl, cy + dw
        if x2 > self.PL or y2 > self.PW:
            return None
        if self.total_weight + weight > self.max_weight + 1e-6:
            return None

        if cands is None:
            cands = self.grid.candidates_xy(cx, cy, x2, y2)

        z = self.grid.find_z(cx, cy, x2, y2, cands)
        if z is None:
            return None
        if z + dh > self.PH:
            return None
        if self.grid.collision_3d(cx, cy, z, x2, y2, z + dh, cands):
            return None
        if not self.grid.check_support(cx, cy, z, x2, y2, dl * dw, cands):
            return None
        if not self.grid.check_fragility(cx, cy, z, x2, y2, weight, cands):
            return None
        return z

    def add(self, box_record: Dict, weight: float) -> None:
        """Place a box, recording grid cells for undo."""
        # Save EP state
        self._eps_len_stack.append(len(self.eps))

        # Track which cells we add to
        idx = len(self.grid.boxes)
        self.grid.boxes.append(box_record)
        R = self.grid._RES
        gx, gy = self.grid.gx, self.grid.gy
        touched_cells: List[Tuple[int, int]] = []

        for ix in range(box_record["x_min"] // R,
                        min((box_record["x_max"] - 1) // R + 1, gx)):
            row = self.grid.cells[ix]
            for iy in range(box_record["y_min"] // R,
                            min((box_record["y_max"] - 1) // R + 1, gy)):
                row[iy].append(idx)
                touched_cells.append((ix, iy))

        self._history.append((box_record, touched_cells))
        self.total_weight += weight

        # Add new extreme points
        px, py, pz = box_record["x_min"], box_record["y_min"], box_record["z_min"]
        dl = box_record["x_max"] - px
        dw = box_record["y_max"] - py
        dh = box_record["z_max"] - pz
        for ep in ((px + dl, py, pz), (px, py + dw, pz), (px, py, pz + dh)):
            if ep not in self.eps_set:
                self.eps_set.add(ep)
                self.eps.append(ep)

    def undo_last(self) -> None:
        """Remove the last placed box, restoring grid state."""
        box_record, touched_cells = self._history.pop()
        idx = len(self.grid.boxes) - 1

        # Remove from grid cells
        for ix, iy in touched_cells:
            cell = self.grid.cells[ix][iy]
            # idx is always the last element we appended
            cell.pop()

        self.grid.boxes.pop()
        self.total_weight -= box_record["weight"]

        # Restore extreme points
        old_len = self._eps_len_stack.pop()
        while len(self.eps) > old_len:
            ep = self.eps.pop()
            self.eps_set.discard(ep)

    @property
    def placed_count(self) -> int:
        return len(self.grid.boxes)

    @property
    def placed_volume(self) -> int:
        return sum(
            (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) * (b["z_max"] - b["z_min"])
            for b in self.grid.boxes
        )


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_score(
    placed_vol: int, pallet_vol: int,
    placed_n: int, total_n: int,
    frag_violations: int, time_ms: int,
) -> float:
    vol_util = placed_vol / pallet_vol if pallet_vol > 0 else 0.0
    item_cov = placed_n / total_n if total_n > 0 else 0.0
    frag_score = max(0.0, 1.0 - 0.05 * frag_violations)
    if time_ms <= 1000:
        time_score = 1.0
    elif time_ms <= 5000:
        time_score = 0.7
    elif time_ms <= 30000:
        time_score = 0.3
    else:
        time_score = 0.0
    return 0.50 * vol_util + 0.30 * item_cov + 0.10 * frag_score + 0.10 * time_score


def upper_bound(
    placed_vol: int, remaining_vol: int, pallet_vol: int,
    placed_n: int, remaining_n: int, total_n: int,
) -> float:
    """Optimistic upper bound: assume all remaining items fit, no frag violations, best time."""
    vol_util = (placed_vol + remaining_vol) / pallet_vol if pallet_vol > 0 else 0.0
    vol_util = min(vol_util, 1.0)
    item_cov = (placed_n + remaining_n) / total_n if total_n > 0 else 0.0
    return 0.50 * vol_util + 0.30 * item_cov + 0.10 * 1.0 + 0.10 * 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Count fragility violations for current packer state
# ──────────────────────────────────────────────────────────────────────────────

def count_frag_violations(boxes: List[Dict]) -> int:
    violations = 0
    for top in boxes:
        if top["weight"] <= 2.0:
            continue
        for bot in boxes:
            if not bot["fragile"]:
                continue
            if bot["z_max"] == top["z_min"]:
                x_ov = min(top["x_max"], bot["x_max"]) - max(top["x_min"], bot["x_min"])
                y_ov = min(top["y_max"], bot["y_max"]) - max(top["y_min"], bot["y_min"])
                if x_ov > 0 and y_ov > 0:
                    violations += 1
    return violations


# ──────────────────────────────────────────────────────────────────────────────
# BB-DFS core
# ──────────────────────────────────────────────────────────────────────────────

def solve_task(request: Dict[str, Any]) -> Dict[str, Any]:
    t_start = time.time()

    pallet = request["pallet"]
    PL = pallet["length_mm"]
    PW = pallet["width_mm"]
    PH = pallet["max_height_mm"]
    max_weight = pallet["max_weight_kg"]
    pallet_vol = PL * PW * PH

    # Expand items
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
                "volume": box["length_mm"] * box["width_mm"] * box["height_mm"],
            })

    total_n = len(items)

    # Sort: large items first (harder to place), special items last
    def sort_key(x):
        is_special = (not x["stackable"]) or x["fragile"]
        return (1 if is_special else 0, -x["volume"])

    items.sort(key=sort_key)

    # Precompute remaining volumes for pruning
    remaining_vols = [0] * (total_n + 1)
    for i in range(total_n - 1, -1, -1):
        remaining_vols[i] = remaining_vols[i + 1] + items[i]["volume"]

    use_exact = total_n <= EXACT_THRESHOLD

    # ── State for best solution found ──
    best_score = [0.0]
    best_placements: List[List[Dict[str, Any]]] = [[]]
    best_placed_items: List[List[Dict[str, Any]]] = [[]]

    packer = BacktrackablePacker(PL, PW, PH, max_weight)

    # Track placed items for output reconstruction
    placed_items_stack: List[Dict[str, Any]] = []

    def record_solution():
        """Record current packer state as a candidate solution."""
        elapsed_ms = int((time.time() - t_start) * 1000)
        frag_v = count_frag_violations(packer.grid.boxes)
        sc = compute_score(
            packer.placed_volume, pallet_vol,
            packer.placed_count, total_n,
            frag_v, elapsed_ms,
        )
        if sc > best_score[0]:
            best_score[0] = sc
            # Rebuild placements output
            out = []
            for item, box in zip(placed_items_stack, packer.grid.boxes):
                dl = box["x_max"] - box["x_min"]
                dw = box["y_max"] - box["y_min"]
                dh = box["z_max"] - box["z_min"]
                out.append({
                    "sku_id": item["sku_id"],
                    "instance_index": item["instance_index"],
                    "position": {"x_mm": box["x_min"], "y_mm": box["y_min"], "z_mm": box["z_min"]},
                    "dimensions_placed": {"length_mm": dl, "width_mm": dw, "height_mm": dh},
                    "rotation_code": item["_rot_code"],
                    "layer": 0,
                })
            best_placements[0] = out
            best_placed_items[0] = [it["sku_id"] for it in placed_items_stack]

    def _try_place_item(item: Dict, packer: BacktrackablePacker):
        """Try placing item at all EPs × rotations. Returns list of (z, x, y, dl, dw, dh, code)."""
        rotations = get_rotations(item["l"], item["w"], item["h"], item["strict_upright"])
        candidates = []
        iw = item["weight"]

        for (ex, ey, _ez) in packer.eps:
            for (dl, dw, dh, code) in rotations:
                x2, y2 = ex + dl, ey + dw
                if x2 > PL or y2 > PW:
                    continue
                cands = packer.grid.candidates_xy(ex, ey, x2, y2)
                z = packer.can_place(ex, ey, dl, dw, dh, iw, item["fragile"], cands)
                if z is not None:
                    candidates.append((z, ex, ey, dl, dw, dh, code))

        # Sort: bottom-left-fill (lowest z, then x, then y)
        candidates.sort()
        return candidates

    # ── Symmetry breaking: for identical SKU instances, enforce instance order ──
    def is_canonical(item_idx: int) -> bool:
        """Skip if a previous instance of the same SKU is still unplaced."""
        item = items[item_idx]
        if item["instance_index"] == 0:
            return True
        # Check if any earlier instance of this SKU is unplaced
        for j in range(item_idx):
            if (items[j]["sku_id"] == item["sku_id"] and
                    items[j]["instance_index"] == item["instance_index"] - 1):
                # This earlier instance must be in placed_items_stack
                if items[j] not in placed_items_stack:
                    return False
                break
        return True

    # Pre-build placed set for faster symmetry check
    placed_set: Set[int] = set()  # indices into items[]

    if use_exact:
        # ── EXACT: Full branch-and-bound ──
        node_count = [0]

        def bb_dfs(level: int):
            """
            At each level, we try to place items[level] or skip it.
            We branch on: place at each valid (position, rotation) OR skip.
            """
            node_count[0] += 1

            # Record current state as solution
            record_solution()

            if level >= total_n:
                return

            # Pruning: upper bound
            ub = upper_bound(
                packer.placed_volume, remaining_vols[level], pallet_vol,
                packer.placed_count, total_n - level, total_n,
            )
            if ub <= best_score[0]:
                return

            item = items[level]

            # Symmetry breaking
            if not is_canonical(level):
                # Must skip this item (previous instance not placed)
                bb_dfs(level + 1)
                return

            # Weight pruning
            if packer.total_weight + item["weight"] > max_weight + 1e-6:
                bb_dfs(level + 1)
                return

            # Get all valid placements
            placements = _try_place_item(item, packer)

            # Branch 1: try each placement
            for (z, px, py, dl, dw, dh, code) in placements:
                box_record = {
                    "sku_id": item["sku_id"],
                    "x_min": px, "x_max": px + dl,
                    "y_min": py, "y_max": py + dw,
                    "z_min": z, "z_max": z + dh,
                    "weight": item["weight"],
                    "fragile": item["fragile"],
                    "stackable": item["stackable"],
                }
                item["_rot_code"] = code
                placed_items_stack.append(item)
                placed_set.add(level)
                packer.add(box_record, item["weight"])

                bb_dfs(level + 1)

                packer.undo_last()
                placed_set.discard(level)
                placed_items_stack.pop()

            # Branch 2: skip this item entirely
            bb_dfs(level + 1)

        bb_dfs(0)

    else:
        # ── LDS: Limited Discrepancy Search ──
        # Greedy order = first candidate at each level (best by z,x,y)
        # Discrepancy = choosing a non-first candidate or skipping

        time_limit = LDS_TIME_LIMIT_S

        def lds_dfs(level: int, disc_remaining: int):
            """LDS DFS: disc_remaining = number of discrepancies still allowed."""
            if time.time() - t_start > time_limit:
                return

            record_solution()

            if level >= total_n:
                return

            # Pruning
            ub = upper_bound(
                packer.placed_volume, remaining_vols[level], pallet_vol,
                packer.placed_count, total_n - level, total_n,
            )
            if ub <= best_score[0]:
                return

            item = items[level]

            if not is_canonical(level):
                lds_dfs(level + 1, disc_remaining)
                return

            if packer.total_weight + item["weight"] > max_weight + 1e-6:
                lds_dfs(level + 1, disc_remaining)
                return

            placements = _try_place_item(item, packer)

            if not placements:
                # No valid placement → skip (no discrepancy cost)
                lds_dfs(level + 1, disc_remaining)
                return

            # Greedy choice: first placement (0 discrepancies)
            z, px, py, dl, dw, dh, code = placements[0]
            box_record = {
                "sku_id": item["sku_id"],
                "x_min": px, "x_max": px + dl,
                "y_min": py, "y_max": py + dw,
                "z_min": z, "z_max": z + dh,
                "weight": item["weight"],
                "fragile": item["fragile"],
                "stackable": item["stackable"],
            }
            item["_rot_code"] = code
            placed_items_stack.append(item)
            placed_set.add(level)
            packer.add(box_record, item["weight"])

            lds_dfs(level + 1, disc_remaining)

            packer.undo_last()
            placed_set.discard(level)
            placed_items_stack.pop()

            # Discrepancy branches: try alternative placements or skip
            if disc_remaining > 0:
                # Try alternative placements (up to a few)
                for alt_idx in range(1, min(len(placements), 4)):
                    if time.time() - t_start > time_limit:
                        return
                    z2, px2, py2, dl2, dw2, dh2, code2 = placements[alt_idx]
                    box_record2 = {
                        "sku_id": item["sku_id"],
                        "x_min": px2, "x_max": px2 + dl2,
                        "y_min": py2, "y_max": py2 + dw2,
                        "z_min": z2, "z_max": z2 + dh2,
                        "weight": item["weight"],
                        "fragile": item["fragile"],
                        "stackable": item["stackable"],
                    }
                    item["_rot_code"] = code2
                    placed_items_stack.append(item)
                    placed_set.add(level)
                    packer.add(box_record2, item["weight"])

                    lds_dfs(level + 1, disc_remaining - 1)

                    packer.undo_last()
                    placed_set.discard(level)
                    placed_items_stack.pop()

                # Skip this item entirely (1 discrepancy)
                if time.time() - t_start > time_limit:
                    return
                lds_dfs(level + 1, disc_remaining - 1)

        # Run LDS with increasing discrepancy budget
        for max_disc in range(LDS_MAX_DISCREPANCIES + 1):
            if time.time() - t_start > time_limit:
                break
            lds_dfs(0, max_disc)

    elapsed_ms = int((time.time() - t_start) * 1000)

    # Build unplaced
    placed_skus: Dict[str, int] = {}
    for p in best_placements[0]:
        placed_skus[p["sku_id"]] = placed_skus.get(p["sku_id"], 0) + 1

    unplaced_out = []
    for box in request["boxes"]:
        placed_qty = placed_skus.get(box["sku_id"], 0)
        unplaced_qty = box["quantity"] - placed_qty
        if unplaced_qty > 0:
            unplaced_out.append({
                "sku_id": box["sku_id"],
                "quantity_unplaced": unplaced_qty,
                "reason": "no_space",
            })

    return {
        "task_id": request["task_id"],
        "solver_version": "brute-force-1.0",
        "solve_time_ms": elapsed_ms,
        "placements": best_placements[0],
        "unplaced": unplaced_out,
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    from validator import evaluate_solution

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        tasks = json.load(f)

    resps = []
    ts = 0.0
    vc = 0
    for i, task in enumerate(tasks):
        resp = solve_task(task)
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
