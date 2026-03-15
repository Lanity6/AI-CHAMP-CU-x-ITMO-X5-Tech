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


MAX_CASES = 3
RANDOM_SEED = 42


def normalize_cases(inputs_raw, results_raw):
    input_list = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]
    result_list = results_raw if isinstance(results_raw, list) else [results_raw]

    if len(input_list) != len(result_list):
        raise ValueError("input_data.json и packing_result.json должны содержать одинаковое число кейсов")

    total = len(input_list)
    if total == 0:
        return []

    k = min(MAX_CASES, total)
    rng = random.Random(RANDOM_SEED)
    selected_indices = sorted(rng.sample(range(total), k))

    cases = []
    for idx in selected_indices:
        input_case = input_list[idx]
        result_case = result_list[idx]

        if input_case.get("task_id") != result_case.get("task_id"):
            raise ValueError(
                f"Несовпадение task_id на индексе {idx}: "
                f"{input_case.get('task_id')} != {result_case.get('task_id')}"
            )

        cases.append((input_case, result_case, idx))

    return cases


def build_global_box_type_color_map(cases):
    palette = [
        "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#06b6d4",
        "#8b5cf6", "#84cc16", "#f97316", "#3b82f6", "#ec4899",
        "#14b8a6", "#eab308", "#f43f5e", "#22c55e", "#0ea5e9",
        "#a855f7", "#65a30d", "#fb7185", "#0891b2", "#7c3aed",
        "#2563eb", "#059669", "#dc2626", "#9333ea", "#0f766e",
        "#c2410c", "#1d4ed8", "#15803d", "#be123c", "#7e22ce",
    ]

    seen = set()
    type_keys = []

    for input_case, _, _ in cases:
        for box in input_case.get("boxes", []):
            type_key = get_box_type_key(box)
            if type_key not in seen:
                seen.add(type_key)
                type_keys.append(type_key)

    if len(type_keys) > len(palette):
        raise ValueError("Not enough unique colors for all box types")

    return {type_key: palette[i] for i, type_key in enumerate(type_keys)}


def prepare_case(plotter, input_data, result_data, box_type_color_map):
    placements = enrich_placements_with_input_data(input_data, result_data)
    boxes_index = build_boxes_index(input_data)

    pallet = input_data["pallet"]
    container_length = pallet["length_mm"]
    container_width = pallet["width_mm"]
    container_height = pallet["max_height_mm"]
    pallet_type = detect_pallet_type(container_length, container_width)

    all_actors = []
    layer_actors = {}
    type_legend = {}

    pallet_plane = pv.Plane(
        center=(container_length / 2, container_width / 2, -1),
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

    container_mesh = pv.Box(bounds=(0, container_length, 0, container_width, 0, container_height))
    container_actor = plotter.add_mesh(
        container_mesh, style="wireframe", color="black", line_width=1.5, opacity=0.35,
    )
    all_actors.append(container_actor)

    for placement in placements:
        pos = placement["position"]
        props = placement["visual_props"]
        dims = placement["resolved_dimensions"]
        layer = placement.get("layer", 1)
        sku_id = placement["sku_id"]

        source_box = boxes_index[sku_id]
        box_type_key = get_box_type_key(source_box)
        color = box_type_color_map[box_type_key]

        x, y, z = pos["x_mm"], pos["y_mm"], pos["z_mm"]
        sx, sy, sz = dims["x_mm"], dims["y_mm"], dims["z_mm"]

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

        item_actors = []
        mesh = make_box(x, y, z, sx, sy, sz)

        if stackable is False:
            actor = plotter.add_mesh(mesh, style="wireframe", color=color, line_width=2.0, opacity=1.0)
        else:
            actor = plotter.add_mesh(mesh, color=color, opacity=1.0, show_edges=True, edge_color="black", line_width=0.5)
        item_actors.append(actor)

        if strict_upright:
            item_actors += add_upright_marks(plotter, x, y, z, sx, sy, sz, color="black")

        if fragile:
            item_actors += add_fragile_pattern(plotter, x, y, z, sx, sy, sz, spacing=35, color="black")

        all_actors.extend(item_actors)
        layer_actors.setdefault(layer, []).extend(item_actors)

    return {
        "task_id": input_data.get("task_id", "N/A"),
        "pallet_type": pallet_type,
        "container_length": container_length,
        "container_width": container_width,
        "container_height": container_height,
        "all_actors": all_actors,
        "layer_actors": layer_actors,
        "type_legend": type_legend,
    }


INPUT_JSON_PATH = "../datasets/dataset_100.json"
RESULT_JSON_PATH = "../results/lns_solver/dataset_100.json"

inputs_raw = load_json(INPUT_JSON_PATH)
results_raw = load_json(RESULT_JSON_PATH)
cases = normalize_cases(inputs_raw, results_raw)

if not cases:
    raise ValueError("No cases found")

box_type_color_map = build_global_box_type_color_map(cases)

plotter = pv.Plotter(window_size=(1550, 950))
plotter.set_background("white")
plotter.show_axes()

prepared_cases = []
for input_case, result_case, original_index in cases:
    case_data = prepare_case(plotter, input_case, result_case, box_type_color_map)
    case_data["original_index"] = original_index
    prepared_cases.append(case_data)

for i, case_data in enumerate(prepared_cases):
    set_visibility(case_data["all_actors"], i == 0)

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
        main_text, position="upper_left", font_size=11,
        color="black", name=ui_state["main_text_name"], font="arial",
    )

    plotter.add_text(
        f"Case {case_index + 1} / {len(prepared_cases)}",
        position=(20, 120), font_size=10,
        color="black", name=ui_state["case_label_name"], font="arial",
    )


def rebuild_legend_ui(case_index: int):
    clear_legend_ui()

    case_data = prepared_cases[case_index]
    legend_items = sorted(case_data["type_legend"].items(), key=lambda x: x[1]["description"])

    legend_x = 1080

    plotter.add_text(
        "Legend", position=(legend_x, 860), font_size=12,
        color="black", name=ui_state["legend_header_name"], font="arial",
    )

    y = 820
    for idx, (_, item) in enumerate(legend_items):
        name = f"legend_{idx}"

        text = build_legend_description(
            description=item["description"],
            strict_upright=item["strict_upright"],
            fragile=item["fragile"],
            stackable=item["stackable"],
        )

        plotter.add_text(
            f"\u25a0 {text}", position=(legend_x, y), font_size=10,
            color=item["color"], name=name, font="arial",
        )
        ui_state["legend_text_names"].append(name)

        y -= 28
        if y < 80:
            break


def apply_layer_visibility():
    case_data = prepared_cases[ui_state["current_case_index"]]
    any_enabled = any(ui_state["layer_states"].values())

    set_visibility(case_data["all_actors"], False)

    if any_enabled:
        if len(case_data["all_actors"]) >= 2:
            case_data["all_actors"][0].SetVisibility(True)
            case_data["all_actors"][1].SetVisibility(True)

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
    case_data = prepared_cases[ui_state["current_case_index"]]

    if state:
        for layer in ui_state["layer_states"]:
            ui_state["layer_states"][layer] = False

        for widget in ui_state["layer_widgets"]:
            try:
                widget.GetRepresentation().SetState(0)
            except Exception:
                pass

        set_visibility(case_data["all_actors"], True)
    else:
        set_visibility(case_data["all_actors"], False)

    plotter.render()


def rebuild_layer_ui(case_index: int):
    clear_layer_ui()

    case_data = prepared_cases[case_index]
    available_layers = sorted(case_data["layer_actors"].keys())
    ui_state["layer_states"] = {layer: False for layer in available_layers}

    plotter.add_text(
        "Layers", position=(20, 700), font_size=12,
        color="black", name=ui_state["layers_header_name"], font="arial",
    )

    plotter.add_text(
        "Show all", position=(60, 655), font_size=10,
        color="black", name=ui_state["show_all_label_name"], font="arial",
    )

    ui_state["show_all_button"] = plotter.add_checkbox_button_widget(
        callback=on_show_all, value=True, position=(20, 655),
        size=20, color_on="green", color_off="lightgray", border_size=1,
    )

    start_y = 615
    step_y = 32

    for idx, layer_value in enumerate(available_layers):
        y_pos = start_y - idx * step_y
        if y_pos < 40:
            break

        label_name = f"layer_label_{case_index}_{layer_value}"
        ui_state["layer_label_names"].append(label_name)

        plotter.add_text(
            f"Layer {layer_value}", position=(60, y_pos), font_size=10,
            color="black", name=label_name, font="arial",
        )

        widget = plotter.add_checkbox_button_widget(
            callback=lambda state, lv=layer_value: on_layer_toggle(lv, state),
            value=False, position=(20, y_pos), size=20,
            color_on="dodgerblue", color_off="lightgray", border_size=1,
        )
        ui_state["layer_widgets"].append(widget)

    plotter.render()


def switch_case(new_index: int):
    old_index = ui_state["current_case_index"]
    if new_index == old_index:
        return

    set_visibility(prepared_cases[old_index]["all_actors"], False)
    set_visibility(prepared_cases[new_index]["all_actors"], True)

    ui_state["current_case_index"] = new_index
    update_header(new_index)
    rebuild_layer_ui(new_index)
    rebuild_legend_ui(new_index)
    plotter.render()


def go_prev():
    new_index = (ui_state["current_case_index"] - 1) % len(prepared_cases)
    switch_case(new_index)


def go_next():
    new_index = (ui_state["current_case_index"] + 1) % len(prepared_cases)
    switch_case(new_index)


center_x = 720
btn_size = 36

plotter.add_checkbox_button_widget(
    callback=lambda state: go_prev(),
    value=False, position=(center_x - btn_size - 10, 20),
    size=btn_size, color_on="steelblue", color_off="steelblue", border_size=2,
)
plotter.add_text(
    "\u25c0", position=(center_x - btn_size - 2, 22),
    font_size=14, color="white", name="prev_label", font="arial",
)

plotter.add_checkbox_button_widget(
    callback=lambda state: go_next(),
    value=False, position=(center_x + 10, 20),
    size=btn_size, color_on="steelblue", color_off="steelblue", border_size=2,
)
plotter.add_text(
    "\u25b6", position=(center_x + 18, 22),
    font_size=14, color="white", name="next_label", font="arial",
)

update_header(0)
rebuild_layer_ui(0)
rebuild_legend_ui(0)

# Camera outside the scene looking at the pallet center
case0 = prepared_cases[0]
cx = case0["container_length"] / 2
cy = case0["container_width"] / 2
cz = case0["container_height"] / 2
dist = max(case0["container_length"], case0["container_width"], case0["container_height"]) * 2.5
plotter.camera_position = [
    (cx + dist, cy + dist * 0.6, cz + dist * 0.8),
    (cx, cy, cz),
    (0, 0, 1),
]

plotter.show()
