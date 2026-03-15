import json
import math
from pathlib import Path

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


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(255, round(r * 255)))),
        int(max(0, min(255, round(g * 255)))),
        int(max(0, min(255, round(b * 255)))),
    )


def build_global_box_type_color_map(inputs_raw):
    input_list = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]

    seen = set()
    type_keys = []

    for input_case in input_list:
        for box in input_case.get("boxes", []):
            type_key = get_box_type_key(box)
            if type_key not in seen:
                seen.add(type_key)
                type_keys.append(type_key)

    if len(type_keys) > 1000:
        raise ValueError(f"Too many box types: {len(type_keys)} > 1000")

    colors = []
    for i in range(1000):
        h = i / 1000.0
        s = 0.68 + 0.20 * ((i % 4) / 3.0)
        v = 0.78 + 0.18 * ((i % 5) / 4.0)

        r, g, b = hsv_to_rgb(h, s, v)
        colors.append(rgb_to_hex(r, g, b))

    return {type_key: colors[i] for i, type_key in enumerate(type_keys)}


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

    out = []
    for placement in placements:
        out.append({**placement, "layer": z_to_layer[placement["position"]["z_mm"]]})
    return out


def enrich_placements_with_input_data(input_data: dict, result_data: dict) -> list:
    boxes_index = build_boxes_index(input_data)
    placements = resolve_layers(result_data.get("placements", []))
    enriched = []

    for placement in placements:
        sku_id = placement["sku_id"]
        source_box = boxes_index.get(sku_id)
        if source_box is None:
            raise KeyError(
                f"SKU '{sku_id}' from result.json not found in input.json for task_id '{input_data.get('task_id')}'"
            )

        sx, sy, sz = get_rotated_dimensions(
            source_box,
            placement.get("rotation_code", "LWH"),
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
                "resolved_dimensions": {
                    "x_mm": sx,
                    "y_mm": sy,
                    "z_mm": sz,
                },
            }
        )

    return enriched


def add_line(plotter, p1, p2, color="black", width=2):
    return plotter.add_mesh(pv.Line(p1, p2), color=color, line_width=width)


def add_cross_on_xy_face(plotter, x, y, z, sx, sy, color="black", width_px=2):
    actors = []
    actors.append(add_line(plotter, (x, y, z), (x + sx, y + sy, z), color=color, width=width_px))
    actors.append(add_line(plotter, (x + sx, y, z), (x, y + sy, z), color=color, width=width_px))
    return actors


def add_diagonal_stripes_xy(plotter, x, y, z, sx, sy, spacing=35, color="black", width_px=1):
    actors = []
    s = 0
    while s <= sx + sy:
        x1 = max(0, s - sy)
        y1 = s - x1
        x2 = min(sx, s)
        y2 = s - x2
        if 0 <= y1 <= sy and 0 <= y2 <= sy:
            actors.append(
                add_line(
                    plotter,
                    (x + x1, y + y1, z),
                    (x + x2, y + y2, z),
                    color=color,
                    width=width_px,
                )
            )
        s += spacing
    return actors


def add_diagonal_stripes_xz(plotter, x, y, z, sx, sz, spacing=35, color="black", width_px=1):
    actors = []
    s = 0
    while s <= sx + sz:
        x1 = max(0, s - sz)
        z1 = s - x1
        x2 = min(sx, s)
        z2 = s - x2
        if 0 <= z1 <= sz and 0 <= z2 <= sz:
            actors.append(
                add_line(
                    plotter,
                    (x + x1, y, z + z1),
                    (x + x2, y, z + z2),
                    color=color,
                    width=width_px,
                )
            )
        s += spacing
    return actors


def add_diagonal_stripes_yz(plotter, x, y, z, sy, sz, spacing=35, color="black", width_px=1):
    actors = []
    s = 0
    while s <= sy + sz:
        y1 = max(0, s - sz)
        z1 = s - y1
        y2 = min(sy, s)
        z2 = s - y2
        if 0 <= z1 <= sz and 0 <= z2 <= sz:
            actors.append(
                add_line(
                    plotter,
                    (x, y + y1, z + z1),
                    (x, y + y2, z + z2),
                    color=color,
                    width=width_px,
                )
            )
        s += spacing
    return actors


def add_fragile_pattern(plotter, x, y, z, sx, sy, sz, spacing=35, color="black"):
    actors = []
    actors += add_diagonal_stripes_xy(plotter, x, y, z, sx, sy, spacing=spacing, color=color)
    actors += add_diagonal_stripes_xy(plotter, x, y, z + sz, sx, sy, spacing=spacing, color=color)
    actors += add_diagonal_stripes_xz(plotter, x, y, z, sx, sz, spacing=spacing, color=color)
    actors += add_diagonal_stripes_xz(plotter, x, y + sy, z, sx, sz, spacing=spacing, color=color)
    actors += add_diagonal_stripes_yz(plotter, x, y, z, sy, sz, spacing=spacing, color=color)
    actors += add_diagonal_stripes_yz(plotter, x + sx, y, z, sy, sz, spacing=spacing, color=color)
    return actors


def add_upright_marks(plotter, x, y, z, sx, sy, sz, color="black"):
    actors = []
    actors += add_cross_on_xy_face(plotter, x, y, z, sx, sy, color=color, width_px=2)
    actors += add_cross_on_xy_face(plotter, x, y, z + sz, sx, sy, color=color, width_px=2)
    return actors


def set_visibility(actors, visible: bool):
    for actor in actors:
        actor.SetVisibility(visible)


def normalize_cases(inputs_raw, results_raw):
    input_list = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]
    result_list = results_raw if isinstance(results_raw, list) else [results_raw]

    input_by_task_id = {}
    for input_case in input_list:
        task_id = input_case.get("task_id")
        if not task_id:
            raise ValueError("input.json contains item without task_id")
        input_by_task_id[task_id] = input_case

    cases = []
    for idx, result_case in enumerate(result_list):
        task_id = result_case.get("task_id")
        if not task_id:
            raise ValueError(f"result.json item #{idx} has no task_id")

        input_case = input_by_task_id.get(task_id)
        if input_case is None:
            raise ValueError(f"No input case found for task_id '{task_id}'")

        cases.append((input_case, result_case, idx))

    return cases


def build_legend_description(description: str, strict_upright: bool, fragile: bool, stackable: bool) -> str:
    parts = [description.strip()]

    if strict_upright:
        parts.append("upright")
    if stackable is False:
        parts.append("top only")
    elif fragile is True:
        parts.append("fragile")

    parts = [p for p in parts if p and str(p).strip()]
    return " - ".join(parts)


def compute_offsets(cases):
    max_len = max(task["pallet"]["length_mm"] for task, _, _ in cases)
    max_wid = max(task["pallet"]["width_mm"] for task, _, _ in cases)

    gap_x = max(500, int(max_len * 0.45))
    gap_y = max(500, int(max_wid * 0.55))

    cols = math.ceil(math.sqrt(len(cases)))
    offsets = []

    for idx in range(len(cases)):
        row = idx // cols
        col = idx % cols
        ox = col * (max_len + gap_x)
        oy = row * (max_wid + gap_y)
        offsets.append((ox, oy, 0))

    return offsets


def prepare_case(plotter, input_data, result_data, box_type_color_map, offset):
    ox, oy, oz = offset
    placements = enrich_placements_with_input_data(input_data, result_data)
    boxes_index = build_boxes_index(input_data)

    pallet = input_data["pallet"]
    container_length = pallet["length_mm"]
    container_width = pallet["width_mm"]
    container_height = pallet["max_height_mm"]
    pallet_type = detect_pallet_type(container_length, container_width)

    all_actors = []
    base_actors = []
    item_actors = []
    layer_actors = {}
    type_legend = {}

    pallet_plane = pv.Plane(
        center=(ox + container_length / 2, oy + container_width / 2, oz - 1),
        direction=(0, 0, 1),
        i_size=container_length,
        j_size=container_width,
        i_resolution=1,
        j_resolution=1,
    )
    pallet_actor = plotter.add_mesh(
        pallet_plane,
        color="#d6b48a",
        show_edges=False,
        lighting=False,
    )
    all_actors.append(pallet_actor)
    base_actors.append(pallet_actor)

    container_mesh = pv.Box(
        bounds=(ox, ox + container_length, oy, oy + container_width, oz, oz + container_height)
    )
    container_actor = plotter.add_mesh(
        container_mesh,
        style="wireframe",
        color="black",
        line_width=1.5,
        opacity=0.35,
    )
    all_actors.append(container_actor)
    base_actors.append(container_actor)

    for placement in placements:
        pos = placement["position"]
        props = placement["visual_props"]
        dims = placement["resolved_dimensions"]
        layer = placement.get("layer", 1)
        sku_id = placement["sku_id"]

        source_box = boxes_index[sku_id]
        box_type_key = get_box_type_key(source_box)
        color = box_type_color_map[box_type_key]

        x = ox + pos["x_mm"]
        y = oy + pos["y_mm"]
        z = oz + pos["z_mm"]

        sx = dims["x_mm"]
        sy = dims["y_mm"]
        sz = dims["z_mm"]

        strict_upright = props["strict_upright"]
        fragile = props["fragile"]
        stackable = props["stackable"]

        if box_type_key not in type_legend:
            type_legend[box_type_key] = {
                "color": color,
                "description": props.get("description") or sku_id,
                "strict_upright": strict_upright,
                "fragile": fragile,
                "stackable": stackable,
            }

        current_item_actors = []
        mesh = make_box(x, y, z, sx, sy, sz)

        if stackable is False:
            actor = plotter.add_mesh(
                mesh,
                style="wireframe",
                color=color,
                line_width=2.0,
                opacity=1.0,
            )
            current_item_actors.append(actor)
        else:
            actor = plotter.add_mesh(
                mesh,
                color=color,
                opacity=1.0,
                show_edges=True,
                edge_color="black",
                line_width=0.5,
            )
            current_item_actors.append(actor)

        if strict_upright:
            current_item_actors += add_upright_marks(plotter, x, y, z, sx, sy, sz, color="black")

        if fragile:
            current_item_actors += add_fragile_pattern(
                plotter, x, y, z, sx, sy, sz, spacing=35, color="black"
            )

        all_actors.extend(current_item_actors)
        item_actors.extend(current_item_actors)
        layer_actors.setdefault(layer, []).extend(current_item_actors)

    return {
        "task_id": input_data.get("task_id", "N/A"),
        "pallet_type": pallet_type,
        "container_length": container_length,
        "container_width": container_width,
        "container_height": container_height,
        "all_actors": all_actors,
        "base_actors": base_actors,
        "item_actors": item_actors,
        "layer_actors": layer_actors,
        "type_legend": type_legend,
        "offset": offset,
    }


INPUT_JSON_PATH = "../datasets/dataset_100.json"
RESULT_JSON_PATH = "../results/lns_solver/dataset_100.json"

inputs_raw = load_json(INPUT_JSON_PATH)
results_raw = load_json(RESULT_JSON_PATH)

cases = normalize_cases(inputs_raw, results_raw)
if not cases:
    raise ValueError("No cases found")

box_type_color_map = build_global_box_type_color_map(inputs_raw)
offsets = compute_offsets(cases)

plotter = pv.Plotter(window_size=(1900, 1100))
plotter.set_background("white")
plotter.show_axes()
plotter.camera_position = "iso"

prepared_cases = []
for (input_case, result_case, original_index), offset in zip(cases, offsets):
    case_data = prepare_case(plotter, input_case, result_case, box_type_color_map, offset)
    case_data["original_index"] = original_index
    prepared_cases.append(case_data)

ui_state = {
    "current_case_index": 0,
    "show_all_button": None,
    "layer_widgets": [],
    "layer_states": {},
    "layer_label_names": [],
    "legend_text_names": [],
    "main_text_name": "main_text",
    "case_label_name": "case_label",
    "layers_header_name": "layers_header",
    "show_all_label_name": "show_all_label",
    "legend_header_name": "legend_header",
}


def set_case_full_visibility(case_index: int):
    case_data = prepared_cases[case_index]
    set_visibility(case_data["all_actors"], True)


def highlight_active_case(case_index: int):
    for idx, case_data in enumerate(prepared_cases):
        container_actor = case_data["base_actors"][1]
        if idx == case_index:
            container_actor.GetProperty().SetColor(0.85, 0.15, 0.15)
            container_actor.GetProperty().SetLineWidth(3.0)
            container_actor.GetProperty().SetOpacity(0.8)
        else:
            container_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
            container_actor.GetProperty().SetLineWidth(1.5)
            container_actor.GetProperty().SetOpacity(0.35)


def clear_layer_ui():
    for widget in ui_state["layer_widgets"]:
        try:
            widget.Off()
        except Exception:
            pass
        try:
            widget.SetCurrentRenderer(None)
        except Exception:
            pass
        try:
            widget.SetInteractor(None)
        except Exception:
            pass
    ui_state["layer_widgets"].clear()

    if ui_state["show_all_button"] is not None:
        try:
            ui_state["show_all_button"].Off()
        except Exception:
            pass
        try:
            ui_state["show_all_button"].SetCurrentRenderer(None)
        except Exception:
            pass
        try:
            ui_state["show_all_button"].SetInteractor(None)
        except Exception:
            pass
        ui_state["show_all_button"] = None

    for text_name in ui_state["layer_label_names"]:
        try:
            plotter.remove_actor(text_name)
        except Exception:
            pass
    ui_state["layer_label_names"].clear()

    ui_state["layer_states"] = {}
    plotter.render()


def clear_legend_ui():
    for text_name in ui_state["legend_text_names"]:
        try:
            plotter.remove_actor(text_name)
        except Exception:
            pass
    ui_state["legend_text_names"].clear()


def update_header(case_index: int):
    case_data = prepared_cases[case_index]

    main_text = (
        "3D Bin Packing Visualizer\n"
        f"Task ID: {case_data['task_id']}\n"
        f"Pallet type: {case_data['pallet_type']}\n"
        f"Size: {case_data['container_length']} x "
        f"{case_data['container_width']} x {case_data['container_height']} mm"
    )

    plotter.add_text(
        main_text,
        position="upper_left",
        font_size=11,
        color="black",
        name=ui_state["main_text_name"],
        font="arial",
    )

    plotter.add_text(
        f"Active pallet {case_index + 1} / {len(prepared_cases)}",
        position=(20, 120),
        font_size=10,
        color="black",
        name=ui_state["case_label_name"],
        font="arial",
    )


def rebuild_legend_ui(case_index: int):
    clear_legend_ui()

    case_data = prepared_cases[case_index]
    legend_items = sorted(case_data["type_legend"].items(), key=lambda x: x[1]["description"])

    legend_x = 1380

    plotter.add_text(
        "Legend",
        position=(legend_x, 960),
        font_size=12,
        color="black",
        name=ui_state["legend_header_name"],
        font="arial",
    )

    y = 920
    for idx, (_, item) in enumerate(legend_items):
        name = f"legend_{idx}"

        text = build_legend_description(
            description=item["description"],
            strict_upright=item["strict_upright"],
            fragile=item["fragile"],
            stackable=item["stackable"],
        )

        plotter.add_text(
            f"■ {text}",
            position=(legend_x, y),
            font_size=10,
            color=item["color"],
            name=name,
            font="arial",
        )
        ui_state["legend_text_names"].append(name)

        y -= 28
        if y < 60:
            break


def apply_layer_visibility():
    case_index = ui_state["current_case_index"]
    case_data = prepared_cases[case_index]
    any_enabled = any(ui_state["layer_states"].values())

    set_visibility(case_data["base_actors"], True)
    set_visibility(case_data["item_actors"], False)

    if any_enabled:
        for layer, enabled in ui_state["layer_states"].items():
            if enabled:
                for actor in case_data["layer_actors"].get(layer, []):
                    actor.SetVisibility(True)

    plotter.render()


def on_layer_toggle(layer_value, state):
    ui_state["layer_states"][layer_value] = bool(state)

    try:
        if ui_state["show_all_button"] is not None:
            ui_state["show_all_button"].GetRepresentation().SetState(0)
    except Exception:
        pass

    apply_layer_visibility()


def on_show_all(state):
    case_index = ui_state["current_case_index"]
    case_data = prepared_cases[case_index]

    if state:
        for layer in ui_state["layer_states"]:
            ui_state["layer_states"][layer] = False

        for widget in ui_state["layer_widgets"]:
            try:
                widget.GetRepresentation().SetState(0)
            except Exception:
                pass

        set_visibility(case_data["base_actors"], True)
        set_visibility(case_data["item_actors"], True)
    else:
        set_visibility(case_data["base_actors"], True)
        set_visibility(case_data["item_actors"], False)

    plotter.render()


def rebuild_layer_ui(case_index: int):
    clear_layer_ui()

    case_data = prepared_cases[case_index]
    available_layers = sorted(case_data["layer_actors"].keys())
    ui_state["layer_states"] = {layer: False for layer in available_layers}

    plotter.add_text(
        "Layers",
        position=(20, 840),
        font_size=12,
        color="black",
        name=ui_state["layers_header_name"],
        font="arial",
    )

    plotter.add_text(
        "Show all",
        position=(60, 795),
        font_size=10,
        color="black",
        name=ui_state["show_all_label_name"],
        font="arial",
    )

    ui_state["show_all_button"] = plotter.add_checkbox_button_widget(
        callback=on_show_all,
        value=True,
        position=(20, 795),
        size=20,
        color_on="green",
        color_off="lightgray",
        border_size=1,
    )

    start_y = 755
    step_y = 32

    for idx, layer_value in enumerate(available_layers):
        y_pos = start_y - idx * step_y
        if y_pos < 40:
            break

        label_name = f"layer_label_{case_index}_{layer_value}"
        ui_state["layer_label_names"].append(label_name)

        plotter.add_text(
            f"Layer {layer_value}",
            position=(60, y_pos),
            font_size=10,
            color="black",
            name=label_name,
            font="arial",
        )

        widget = plotter.add_checkbox_button_widget(
            callback=lambda state, lv=layer_value: on_layer_toggle(lv, state),
            value=False,
            position=(20, y_pos),
            size=20,
            color_on="dodgerblue",
            color_off="lightgray",
            border_size=1,
        )
        ui_state["layer_widgets"].append(widget)

    plotter.render()


def switch_case(new_index: int):
    old_index = ui_state["current_case_index"]
    if new_index == old_index:
        return

    set_case_full_visibility(old_index)

    ui_state["current_case_index"] = new_index
    highlight_active_case(new_index)
    update_header(new_index)
    rebuild_layer_ui(new_index)
    rebuild_legend_ui(new_index)

    set_case_full_visibility(new_index)
    plotter.render()


def go_prev():
    new_index = (ui_state["current_case_index"] - 1) % len(prepared_cases)
    switch_case(new_index)


def go_next():
    new_index = (ui_state["current_case_index"] + 1) % len(prepared_cases)
    switch_case(new_index)


center_x = 900

plotter.add_checkbox_button_widget(
    callback=lambda state: go_prev() if state else None,
    value=False,
    position=(center_x - 90, 45),
    size=22,
    color_on="lightgray",
    color_off="lightgray",
    border_size=1,
)

plotter.add_checkbox_button_widget(
    callback=lambda state: go_next() if state else None,
    value=False,
    position=(center_x + 40, 45),
    size=22,
    color_on="lightgray",
    color_off="lightgray",
    border_size=1,
)

plotter.add_text(
    "<-",
    position=(center_x - 96, 18),
    font_size=16,
    color="black",
    name="prev_label",
    font="arial",
)

plotter.add_text(
    "->",
    position=(center_x + 36, 18),
    font_size=16,
    color="black",
    name="next_label",
    font="arial",
)

highlight_active_case(0)
update_header(0)
rebuild_layer_ui(0)
rebuild_legend_ui(0)

plotter.show()
