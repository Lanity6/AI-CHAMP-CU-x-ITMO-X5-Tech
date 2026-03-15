import json
from pathlib import Path

import numpy as np
import pyvista as pv


VALID_ROTATIONS = {"LWH", "LHW", "WLH", "WHL", "HLW", "HWL"}


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_box(x, y, z, sx, sy, sz):
    return pv.Box(bounds=(x, x + sx, y, y + sy, z, z + sz))


def build_boxes_index(input_data: dict) -> dict:
    return {box["sku_id"]: box for box in input_data.get("boxes", [])}


def get_box_type_key(box: dict) -> tuple:
    return (
        box.get("description", "").strip(),
        box.get("length_mm"),
        box.get("width_mm"),
        box.get("height_mm"),
        bool(box.get("strict_upright", False)),
        bool(box.get("fragile", False)),
        bool(box.get("stackable", True)),
    )


def detect_pallet_type(length_mm: int, width_mm: int) -> str:
    mapping = {
        (1200, 800): "EUR / EPAL",
        (1200, 1000): "FIN",
        (1219, 1016): "GMA / 48x40",
    }
    if (length_mm, width_mm) in mapping:
        return mapping[(length_mm, width_mm)]
    if (width_mm, length_mm) in mapping:
        return mapping[(width_mm, length_mm)]
    return f"Unknown ({length_mm}x{width_mm})"


def get_rotated_dimensions(box_info: dict, rotation_code: str) -> tuple[int, int, int]:
    dims = {
        "L": box_info["length_mm"],
        "W": box_info["width_mm"],
        "H": box_info["height_mm"],
    }
    if rotation_code not in VALID_ROTATIONS:
        raise ValueError(f"Unsupported rotation_code: {rotation_code}")
    return (
        dims[rotation_code[0]],
        dims[rotation_code[1]],
        dims[rotation_code[2]],
    )


def resolve_layers(placements: list[dict]) -> list[dict]:
    if not placements:
        return placements
    if any("layer" in p for p in placements):
        return placements
    unique_z = sorted({p["position"]["z_mm"] for p in placements})
    z_to_layer = {z: i + 1 for i, z in enumerate(unique_z)}
    return [{**p, "layer": z_to_layer[p["position"]["z_mm"]]} for p in placements]


def enrich_placements_with_input_data(input_data: dict, result_data: dict) -> list:
    boxes_index = build_boxes_index(input_data)
    placements = resolve_layers(result_data.get("placements", []))
    enriched = []
    for placement in placements:
        sku_id = placement["sku_id"]
        source_box = boxes_index.get(sku_id)
        if source_box is None:
            raise KeyError(
                f"SKU '{sku_id}' from result.json not found in input.json "
                f"for task_id '{input_data.get('task_id')}'"
            )
        sx, sy, sz = get_rotated_dimensions(
            source_box, placement.get("rotation_code", "LWH")
        )
        enriched.append(
            {
                **placement,
                "visual_props": {
                    "description": source_box.get("description"),
                    "strict_upright": source_box.get("strict_upright", False),
                    "fragile": source_box.get("fragile", False),
                    "stackable": source_box.get("stackable", True),
                    "sku_id": sku_id,
                },
                "resolved_dimensions": {"x_mm": sx, "y_mm": sy, "z_mm": sz},
            }
        )
    return enriched


def build_legend_description(
    description: str, strict_upright: bool, fragile: bool, stackable: bool
) -> str:
    parts = [description.strip()]
    if strict_upright:
        parts.append("upright")
    if stackable is False:
        parts.append("top only")
    elif fragile is True:
        parts.append("fragile")
    parts = [p for p in parts if p and str(p).strip()]
    return " - ".join(parts)


def set_visibility(actors, visible: bool):
    for actor in actors:
        actor.SetVisibility(visible)


# ---------------------------------------------------------------------------
# Batched line rendering
# ---------------------------------------------------------------------------

def build_lines_polydata(segments):
    """Build a single PolyData from a list of ((x1,y1,z1),(x2,y2,z2)) tuples."""
    n = len(segments)
    if n == 0:
        return pv.PolyData()
    points = np.empty((2 * n, 3), dtype=np.float64)
    for i, (p1, p2) in enumerate(segments):
        points[2 * i] = p1
        points[2 * i + 1] = p2
    cells = np.empty(3 * n, dtype=np.int64)
    for i in range(n):
        cells[3 * i] = 2
        cells[3 * i + 1] = 2 * i
        cells[3 * i + 2] = 2 * i + 1
    poly = pv.PolyData()
    poly.points = points
    poly.lines = cells
    return poly


def _collect_stripes_xy(x, y, z, sx, sy, spacing=35):
    lines = []
    s = 0
    while s <= sx + sy:
        x1 = max(0, s - sy)
        y1 = s - x1
        x2 = min(sx, s)
        y2 = s - x2
        if 0 <= y1 <= sy and 0 <= y2 <= sy:
            lines.append(((x + x1, y + y1, z), (x + x2, y + y2, z)))
        s += spacing
    return lines


def _collect_stripes_xz(x, y, z, sx, sz, spacing=35):
    lines = []
    s = 0
    while s <= sx + sz:
        x1 = max(0, s - sz)
        z1 = s - x1
        x2 = min(sx, s)
        z2 = s - x2
        if 0 <= z1 <= sz and 0 <= z2 <= sz:
            lines.append(((x + x1, y, z + z1), (x + x2, y, z + z2)))
        s += spacing
    return lines


def _collect_stripes_yz(x, y, z, sy, sz, spacing=35):
    lines = []
    s = 0
    while s <= sy + sz:
        y1 = max(0, s - sz)
        z1 = s - y1
        y2 = min(sy, s)
        z2 = s - y2
        if 0 <= z1 <= sz and 0 <= z2 <= sz:
            lines.append(((x, y + y1, z + z1), (x, y + y2, z + z2)))
        s += spacing
    return lines


def _collect_cross_xy(x, y, z, sx, sy):
    return [
        ((x, y, z), (x + sx, y + sy, z)),
        ((x + sx, y, z), (x, y + sy, z)),
    ]


def add_fragile_pattern(plotter, x, y, z, sx, sy, sz, spacing=35, color="black"):
    """Add fragile hatching as a SINGLE batched mesh (instead of ~140 individual actors)."""
    lines = []
    lines += _collect_stripes_xy(x, y, z, sx, sy, spacing)
    lines += _collect_stripes_xy(x, y, z + sz, sx, sy, spacing)
    lines += _collect_stripes_xz(x, y, z, sx, sz, spacing)
    lines += _collect_stripes_xz(x, y + sy, z, sx, sz, spacing)
    lines += _collect_stripes_yz(x, y, z, sy, sz, spacing)
    lines += _collect_stripes_yz(x + sx, y, z, sy, sz, spacing)
    if not lines:
        return []
    mesh = build_lines_polydata(lines)
    actor = plotter.add_mesh(mesh, color=color, line_width=1)
    return [actor]


def add_upright_marks(plotter, x, y, z, sx, sy, sz, color="black"):
    """Add upright cross marks as a SINGLE batched mesh (instead of 4 individual actors)."""
    lines = []
    lines += _collect_cross_xy(x, y, z, sx, sy)
    lines += _collect_cross_xy(x, y, z + sz, sx, sy)
    if not lines:
        return []
    mesh = build_lines_polydata(lines)
    actor = plotter.add_mesh(mesh, color=color, line_width=2)
    return [actor]
