"""Lossless input tables, with critical geometry fields explicitly named.

Only the transport representation changes. Population counts, physical settings,
numbers, units, candidate order, and every forecast remain request-specific.
Unknown future section shapes use the original JSON for the whole request.
"""
import copy
import json

VERSION = "shepherd-tables-v2"
GEOMETRY_NAMED = (
    "target_id", "target_goalward_speed", "drive_point_boundary_margin",
    "estimated_travel_frames",
)
GEOMETRY_VALUES = (
    "target_status", "target_goal_distance", "target_spread", "target_outward_speed",
    "dog_to_target", "dog_to_drive_point", "dog_approach_alignment",
)
FORECAST_COLUMNS = (
    "centroid_progress", "final_centroid_distance",
    "max_spread_change", "final_max_spread", "mean_spread_change", "final_mean_spread",
    "inside_goal_change", "final_inside_goal", "predicted_goal_exits",
    "predicted_splitting_frames", "peak_out_of_bounds", "minimum_boundary_margin",
    "dog_travel", "predicted_success",
)
TYPE_COLUMNS = (
    "count", "mean_goal_distance", "max_goal_distance", "mean_spread",
    "mean_goalward_speed", "mean_outward_speed",
)
SECTIONS = {
    "cohesion_candidate", "hard_to_guide_candidate", "type_geometry",
    "mode_forecasts", "repair_candidates",
}

TABLE_INSTRUCTIONS = "\n\n" + "\n".join((
    "INPUT TRANSPORT: If input_format is shepherd-tables-v2, read its state as follows; "
    "otherwise the input is the original named JSON state. All original numbers, units, "
    "booleans, sheep identities and candidate ordering are unchanged. No fields are inferred or omitted.",
    "Global state fields and type_c_response retain their original names and values.",
    "Geometry objects (cohesion_candidate, hard_to_guide_candidate, and candidate geometry) "
    "keep target_id, target_goalward_speed, drive_point_boundary_margin and estimated_travel_frames "
    "explicitly named. Their v array columns are: " + ",".join(GEOMETRY_VALUES) + ".",
    "Every forecast OBJECT keeps horizon_frames explicitly named. Its v ARRAY uses columns: " + ",".join(FORECAST_COLUMNS) + ".",
    "type_geometry maps each original type name to an ARRAY with columns: " + ",".join(TYPE_COLUMNS) + ".",
    "mode_forecasts maps each original mode name to its forecast OBJECT. "
    "repair_candidates maps each original mode to an ordered list of objects "
    "{geometry: geometry object, forecast: forecast OBJECT}. Sheep IDs are identities, never row indices. "
    "Compare a mode's candidates only within that mode. Current drive_point_boundary_margin "
    "and predicted minimum_boundary_margin are different quantities. "
    "No change to the response schema or mode/target permissions.",
))


class UnsupportedShape(ValueError):
    pass


def _row(value, columns):
    if not isinstance(value, dict) or set(value) != set(columns):
        raise UnsupportedShape()
    return [copy.deepcopy(value[key]) for key in columns]


def _geometry(value):
    _row(value, GEOMETRY_NAMED + GEOMETRY_VALUES)
    return {**{key: copy.deepcopy(value[key]) for key in GEOMETRY_NAMED},
            "v": [copy.deepcopy(value[key]) for key in GEOMETRY_VALUES]}


def _forecast(value):
    _row(value, ("horizon_frames",) + FORECAST_COLUMNS)
    return {"horizon_frames": copy.deepcopy(value["horizon_frames"]),
            "v": [copy.deepcopy(value[key]) for key in FORECAST_COLUMNS]}


def pack(state):
    """Return a table envelope; raise UnsupportedShape rather than discard fields."""
    if not isinstance(state, dict) or not SECTIONS <= state.keys() or "input_format" in state:
        raise UnsupportedShape()
    for key in ("type_geometry", "mode_forecasts", "repair_candidates"):
        if not isinstance(state[key], dict):
            raise UnsupportedShape()
    result = copy.deepcopy(state)
    for key in ("cohesion_candidate", "hard_to_guide_candidate"):
        result[key] = _geometry(state[key])
    result["type_geometry"] = {name: _row(value, TYPE_COLUMNS) for name, value in state["type_geometry"].items()}
    result["mode_forecasts"] = {mode: _forecast(value) for mode, value in state["mode_forecasts"].items()}
    candidates = {}
    for mode, values in state["repair_candidates"].items():
        if not isinstance(values, list):
            raise UnsupportedShape()
        candidates[mode] = []
        for value in values:
            if not isinstance(value, dict) or set(value) != set(GEOMETRY_NAMED + GEOMETRY_VALUES) | {"forecast"}:
                raise UnsupportedShape()
            candidates[mode].append({
                "geometry": _geometry({key: value[key] for key in GEOMETRY_NAMED + GEOMETRY_VALUES}),
                "forecast": _forecast(value["forecast"]),
            })
    result["repair_candidates"] = candidates
    return {"input_format": VERSION, "state": result}


def unpack(payload):
    """Inverse used for losslessness checks; inference permissions use original state."""
    if set(payload) != {"input_format", "state"} or payload["input_format"] != VERSION:
        raise UnsupportedShape()

    def row(values, columns):
        if not isinstance(values, list) or len(values) != len(columns):
            raise UnsupportedShape()
        return dict(zip(columns, copy.deepcopy(values)))

    def geometry(value):
        if not isinstance(value, dict) or set(value) != set(GEOMETRY_NAMED) | {"v"}:
            raise UnsupportedShape()
        return {**{key: copy.deepcopy(value[key]) for key in GEOMETRY_NAMED},
                **row(value["v"], GEOMETRY_VALUES)}

    def forecast(value):
        if not isinstance(value, dict) or set(value) != {"horizon_frames", "v"}:
            raise UnsupportedShape()
        return {"horizon_frames": copy.deepcopy(value["horizon_frames"]),
                **row(value["v"], FORECAST_COLUMNS)}

    state = copy.deepcopy(payload["state"])
    for key in ("cohesion_candidate", "hard_to_guide_candidate"):
        state[key] = geometry(state[key])
    state["type_geometry"] = {name: row(value, TYPE_COLUMNS) for name, value in state["type_geometry"].items()}
    state["mode_forecasts"] = {mode: forecast(value) for mode, value in state["mode_forecasts"].items()}
    state["repair_candidates"] = {
        mode: [{**geometry(value["geometry"]), "forecast": forecast(value["forecast"])}
               for value in values] for mode, values in state["repair_candidates"].items()
    }
    return state


def encode_json(original):
    """Use original JSON unchanged if a future section has an unfamiliar shape."""
    state = json.loads(original)
    try:
        payload = pack(state)
    except UnsupportedShape:
        return original
    # Validate every transformation before exposing it to a control decision.
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(unpack(payload)) != canonical(state):
        raise ValueError("Input table round-trip validation failed")
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
