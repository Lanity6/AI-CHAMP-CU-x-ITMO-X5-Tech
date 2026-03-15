import math
import random

import pyvista as pv

from viz_utils import (
    add_fragile_pattern,
    add_upright_marks,
    build_boxes_index,
    build_legend_description,
    detect_pallet_type,
    enrich_placements_with_input_data,
    get_box_type_key,
    load_json,
    make_box,
    set_visibility,
)


MAX_CASES = 9
RANDOM_SEED = 42

FONT_REGULAR = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
FONT_SEMIBOLD = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"
FONT_LIGHT = "/usr/share/fonts/truetype/lato/Lato-Light.ttf"

COLOR_TITLE = "#1e293b"
COLOR_SUBTITLE = "#475569"
COLOR_LABEL = "#334155"
COLOR_HEADER = "#0f172a"


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
    """Assign colors by description so that e.g. all 'Water Pack' boxes share one color."""
    input_list = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]

    # Collect unique descriptions and all type_keys
    desc_order = []
    desc_seen = set()
    all_type_keys = []

    for input_case in input_list:
        for box in input_case.get("boxes", []):
            desc = box.get("description", "").strip()
            type_key = get_box_type_key(box)
            all_type_keys.append((type_key, desc))
            if desc not in desc_seen:
                desc_seen.add(desc)
                desc_order.append(desc)

    # Generate one color per unique description
    colors = []
    for i in range(max(len(desc_order), 1)):
        h = i / max(len(desc_order), 1)
        s = 0.68 + 0.20 * ((i % 4) / 3.0)
        v = 0.78 + 0.18 * ((i % 5) / 4.0)
        r, g, b = hsv_to_rgb(h, s, v)
        colors.append(rgb_to_hex(r, g, b))

    desc_to_color = {desc: colors[i] for i, desc in enumerate(desc_order)}

    # Map every type_key to the color of its description
    result = {}
    for type_key, desc in all_type_keys:
        if type_key not in result:
            result[type_key] = desc_to_color[desc]

    return result


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

    # Limit number of cases to avoid performance issues
    total = len(cases)
    if total > MAX_CASES:
        rng = random.Random(RANDOM_SEED)
        selected_indices = sorted(rng.sample(range(total), MAX_CASES))
        cases = [cases[i] for i in selected_indices]

    return cases


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
        pallet_plane, color="#d6b48a", show_edges=False, lighting=False,
    )
    all_actors.append(pallet_actor)
    base_actors.append(pallet_actor)

    container_mesh = pv.Box(
        bounds=(ox, ox + container_length, oy, oy + container_width, oz, oz + container_height)
    )
    container_actor = plotter.add_mesh(
        container_mesh, style="wireframe", color="black", line_width=1.5, opacity=0.35,
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
            actor = plotter.add_mesh(mesh, style="wireframe", color=color, line_width=2.0, opacity=1.0)
        else:
            actor = plotter.add_mesh(mesh, color=color, opacity=1.0, show_edges=True, edge_color="black", line_width=0.5)
        current_item_actors.append(actor)

        if strict_upright:
            current_item_actors += add_upright_marks(plotter, x, y, z, sx, sy, sz, color="black")

        if fragile:
            current_item_actors += add_fragile_pattern(plotter, x, y, z, sx, sy, sz, spacing=35, color="black")

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
plotter.set_background("#f8fafc")
plotter.show_axes()

prepared_cases = []
for (input_case, result_case, original_index), offset in zip(cases, offsets):
    case_data = prepare_case(plotter, input_case, result_case, box_type_color_map, offset)
    case_data["original_index"] = original_index
    prepared_cases.append(case_data)

MAX_VISIBLE_LAYERS = 20

ui_state = {
    "current_case_index": 0,
    "show_all_button": None,
    "layer_widgets": [],
    "layer_states": {},
    "layer_label_names": [],
    "legend_text_names": [],
    "legend_color_widgets": [],
    "title_name": "title_text",
    "task_id_name": "task_id_text",
    "pallet_info_name": "pallet_info_text",
    "size_info_name": "size_info_text",
    "case_label_name": "case_label",
    "layers_header_name": "layers_header",
    "show_all_label_name": "show_all_label",
    "legend_header_name": "legend_header",
    "layer_scroll_offset": 0,
    "available_layers": [],
    "scroll_up_widget": None,
    "scroll_down_widget": None,
    "scroll_up_label": "scroll_up_label",
    "scroll_down_label": "scroll_down_label",
    "scroll_info_label": "scroll_info_label",
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


def _kill_widget(w):
    try:
        w.Off()
    except Exception:
        pass
    try:
        w.SetEnabled(0)
    except Exception:
        pass
    try:
        w.GetRepresentation().SetVisibility(0)
    except Exception:
        pass
    try:
        w.SetCurrentRenderer(None)
    except Exception:
        pass
    try:
        w.SetInteractor(None)
    except Exception:
        pass


def clear_layer_ui():
    for widget in ui_state["layer_widgets"]:
        _kill_widget(widget)
    ui_state["layer_widgets"].clear()

    if ui_state["show_all_button"] is not None:
        _kill_widget(ui_state["show_all_button"])
        ui_state["show_all_button"] = None

    if ui_state["scroll_up_widget"] is not None:
        _kill_widget(ui_state["scroll_up_widget"])
        ui_state["scroll_up_widget"] = None
    if ui_state["scroll_down_widget"] is not None:
        _kill_widget(ui_state["scroll_down_widget"])
        ui_state["scroll_down_widget"] = None

    for text_name in ui_state["layer_label_names"]:
        try:
            plotter.remove_actor(text_name)
        except Exception:
            pass
    ui_state["layer_label_names"].clear()

    for name in (ui_state["scroll_up_label"], ui_state["scroll_down_label"], ui_state["scroll_info_label"]):
        try:
            plotter.remove_actor(name)
        except Exception:
            pass

    plotter.render()


def clear_legend_ui():
    for text_name in ui_state["legend_text_names"]:
        try:
            plotter.remove_actor(text_name)
        except Exception:
            pass
    ui_state["legend_text_names"].clear()

    for w in ui_state["legend_color_widgets"]:
        _kill_widget(w)
    ui_state["legend_color_widgets"].clear()


def update_header(case_index: int):
    case_data = prepared_cases[case_index]

    plotter.add_text(
        "3D Bin Packing Visualizer",
        position=(20, 1050), font_size=16, shadow=True,
        color=COLOR_TITLE, name=ui_state["title_name"],
        font_file=FONT_BOLD,
    )

    plotter.add_text(
        f"Task ID: {case_data['task_id']}",
        position=(20, 1020), font_size=11,
        color=COLOR_SUBTITLE, name=ui_state["task_id_name"],
        font_file=FONT_REGULAR,
    )

    plotter.add_text(
        f"Pallet: {case_data['pallet_type']}",
        position=(20, 995), font_size=11,
        color=COLOR_SUBTITLE, name=ui_state["pallet_info_name"],
        font_file=FONT_REGULAR,
    )

    plotter.add_text(
        f"{case_data['container_length']} \u00d7 {case_data['container_width']} \u00d7 {case_data['container_height']} mm",
        position=(20, 970), font_size=10,
        color=COLOR_SUBTITLE, name=ui_state["size_info_name"],
        font_file=FONT_LIGHT,
    )

    plotter.add_text(
        f"Active pallet {case_index + 1} / {len(prepared_cases)}",
        position=(20, 940), font_size=10,
        color=COLOR_LABEL, name=ui_state["case_label_name"],
        font_file=FONT_SEMIBOLD,
    )


def rebuild_legend_ui(case_index: int):
    clear_legend_ui()

    # Merge legends from all cases, deduplicate by description
    seen_desc = set()
    legend_items = []
    for cd in prepared_cases:
        for key, val in cd["type_legend"].items():
            desc = val.get("description", "")
            if desc not in seen_desc:
                seen_desc.add(desc)
                legend_items.append((key, val))
    legend_items.sort(key=lambda x: x[1]["description"])

    legend_x = 1380

    plotter.add_text(
        "Legend", position=(legend_x, 1050), font_size=13,
        color=COLOR_HEADER, name=ui_state["legend_header_name"],
        font_file=FONT_BOLD,
    )

    swatch_size = 14
    y = 1018
    for idx, (_, item) in enumerate(legend_items):
        text_name = f"legend_text_{idx}"

        text = build_legend_description(
            description=item["description"],
            strict_upright=item["strict_upright"],
            fragile=item["fragile"],
            stackable=item["stackable"],
        )

        w = plotter.add_checkbox_button_widget(
            callback=lambda state: None,
            value=True, position=(legend_x, y), size=swatch_size,
            color_on=item["color"], color_off=item["color"], border_size=1,
        )
        ui_state["legend_color_widgets"].append(w)

        plotter.add_text(
            text, position=(legend_x + swatch_size + 8, y), font_size=10,
            color=COLOR_LABEL, name=text_name,
            font_file=FONT_REGULAR,
        )
        ui_state["legend_text_names"].append(text_name)

        y -= 26
        if y < 60:
            break


def apply_layer_visibility():
    any_enabled = any(ui_state["layer_states"].values())

    for case_data in prepared_cases:
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
    if state:
        for layer in ui_state["layer_states"]:
            ui_state["layer_states"][layer] = False

        for widget in ui_state["layer_widgets"]:
            try:
                widget.GetRepresentation().SetState(0)
            except Exception:
                pass

        for case_data in prepared_cases:
            set_visibility(case_data["base_actors"], True)
            set_visibility(case_data["item_actors"], True)
    else:
        for case_data in prepared_cases:
            set_visibility(case_data["base_actors"], True)
            set_visibility(case_data["item_actors"], False)

    plotter.render()


def _render_layer_page():
    """Render the current page of layers based on scroll offset."""
    case_index = ui_state["current_case_index"]
    available_layers = ui_state["available_layers"]
    offset = ui_state["layer_scroll_offset"]
    total = len(available_layers)

    for widget in ui_state["layer_widgets"]:
        _kill_widget(widget)
    ui_state["layer_widgets"].clear()

    for text_name in ui_state["layer_label_names"]:
        try:
            plotter.remove_actor(text_name)
        except Exception:
            pass
    ui_state["layer_label_names"].clear()

    visible = available_layers[offset:offset + MAX_VISIBLE_LAYERS]
    start_y = 822
    step_y = 30

    for idx, layer_value in enumerate(visible):
        y_pos = start_y - idx * step_y

        label_name = f"layer_label_{case_index}_{layer_value}"
        ui_state["layer_label_names"].append(label_name)

        plotter.add_text(
            f"Layer {layer_value}", position=(60, y_pos), font_size=10,
            color=COLOR_LABEL, name=label_name,
            font_file=FONT_REGULAR,
        )

        is_on = ui_state["layer_states"].get(layer_value, False)
        widget = plotter.add_checkbox_button_widget(
            callback=lambda state, lv=layer_value: on_layer_toggle(lv, state),
            value=is_on, position=(20, y_pos), size=20,
            color_on="dodgerblue", color_off="lightgray", border_size=1,
        )
        ui_state["layer_widgets"].append(widget)

    if total > MAX_VISIBLE_LAYERS:
        end = min(offset + MAX_VISIBLE_LAYERS, total)
        plotter.add_text(
            f"{offset + 1}-{end} / {total}",
            position=(60, 822 - len(visible) * step_y), font_size=9,
            color=COLOR_SUBTITLE, name=ui_state["scroll_info_label"],
            font_file=FONT_LIGHT,
        )
    else:
        try:
            plotter.remove_actor(ui_state["scroll_info_label"])
        except Exception:
            pass

    plotter.render()


def _scroll_layers_up(_state=None):
    if ui_state["layer_scroll_offset"] > 0:
        ui_state["layer_scroll_offset"] -= MAX_VISIBLE_LAYERS
        if ui_state["layer_scroll_offset"] < 0:
            ui_state["layer_scroll_offset"] = 0
        _render_layer_page()


def _scroll_layers_down(_state=None):
    total = len(ui_state["available_layers"])
    if ui_state["layer_scroll_offset"] + MAX_VISIBLE_LAYERS < total:
        ui_state["layer_scroll_offset"] += MAX_VISIBLE_LAYERS
        _render_layer_page()


def rebuild_layer_ui(case_index: int):
    clear_layer_ui()

    # Collect layers from ALL cases
    all_layers = set()
    for cd in prepared_cases:
        all_layers.update(cd["layer_actors"].keys())
    available_layers = sorted(all_layers)

    old_states = ui_state["layer_states"]
    new_states = {}
    for layer in available_layers:
        new_states[layer] = old_states.get(layer, False)
    ui_state["layer_states"] = new_states
    ui_state["available_layers"] = available_layers
    ui_state["layer_scroll_offset"] = 0

    plotter.add_text(
        "Layers", position=(20, 890), font_size=13,
        color=COLOR_HEADER, name=ui_state["layers_header_name"],
        font_file=FONT_BOLD,
    )

    plotter.add_text(
        "Show all", position=(60, 858), font_size=10,
        color=COLOR_LABEL, name=ui_state["show_all_label_name"],
        font_file=FONT_REGULAR,
    )

    ui_state["show_all_button"] = plotter.add_checkbox_button_widget(
        callback=on_show_all, value=True, position=(20, 858),
        size=20, color_on="green", color_off="lightgray", border_size=1,
    )

    if len(available_layers) > MAX_VISIBLE_LAYERS:
        scroll_x = 160
        ui_state["scroll_up_widget"] = plotter.add_checkbox_button_widget(
            callback=lambda state: _scroll_layers_up(),
            value=False, position=(scroll_x, 862), size=18,
            color_on="steelblue", color_off="steelblue", border_size=1,
        )
        plotter.add_text(
            "\u25b2", position=(scroll_x + 4, 862),
            font_size=9, color="white", name=ui_state["scroll_up_label"],
            font="arial",
        )
        ui_state["scroll_down_widget"] = plotter.add_checkbox_button_widget(
            callback=lambda state: _scroll_layers_down(),
            value=False, position=(scroll_x + 26, 862), size=18,
            color_on="steelblue", color_off="steelblue", border_size=1,
        )
        plotter.add_text(
            "\u25bc", position=(scroll_x + 30, 862),
            font_size=9, color="white", name=ui_state["scroll_down_label"],
            font="arial",
        )

    _render_layer_page()


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
btn_size = 36

plotter.add_checkbox_button_widget(
    callback=lambda state: go_prev(),
    value=False, position=(center_x - btn_size - 10, 20),
    size=btn_size, color_on="steelblue", color_off="steelblue", border_size=2,
)
plotter.add_text(
    "<", position=(center_x - btn_size + 2, 25),
    font_size=12, color="white", name="prev_label", font="arial",
)

plotter.add_checkbox_button_widget(
    callback=lambda state: go_next(),
    value=False, position=(center_x + 10, 20),
    size=btn_size, color_on="steelblue", color_off="steelblue", border_size=2,
)
plotter.add_text(
    ">", position=(center_x + 22, 25),
    font_size=12, color="white", name="next_label", font="arial",
)

highlight_active_case(0)
update_header(0)
rebuild_layer_ui(0)
rebuild_legend_ui(0)

# Camera outside the scene looking at the grid center
all_offsets = [c["offset"] for c in prepared_cases]
max_ox = max(ox for ox, _, _ in all_offsets)
max_oy = max(oy for _, oy, _ in all_offsets)
max_l = max(c["container_length"] for c in prepared_cases)
max_w = max(c["container_width"] for c in prepared_cases)
max_h = max(c["container_height"] for c in prepared_cases)
scene_cx = (max_ox + max_l) / 2
scene_cy = (max_oy + max_w) / 2
scene_cz = max_h / 2
dist = max(max_ox + max_l, max_oy + max_w, max_h) * 1.8
plotter.camera_position = [
    (scene_cx + dist, scene_cy + dist * 0.6, scene_cz + dist * 0.8),
    (scene_cx, scene_cy, scene_cz),
    (0, 0, 1),
]

plotter.show()
