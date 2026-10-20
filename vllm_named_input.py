"""Named JSON deduplication; never remove differing geometry or forecast fields."""
import copy
import json

VERSION = "shepherd-named-dedup-v1"
DEFAULTS = {
    "cohesion_candidate": "COHESION_PRIORITY",
    "hard_to_guide_candidate": "HARD_TO_GUIDE",
}


def instructions(mode_selection=True, target_selection=True, compact=False):
    """Same decision policy, consolidated once; no positional numeric encoding."""
    text = """You select a high-level mode, optional supplied repair target, and hold for a 2D swarm shepherding controller. The low-level controller keeps moving while you reason and computes all movement. Prioritize successful arrival, then limiting dispersion and unsafe boundary approaches. You cannot change coordinates, forces, distances, speeds, coefficients, or invent modes or missing state.

PERMISSIONS AND FALLBACK
The response schema's mode enum is the complete set of admissible modes; other forecasts give context, not permission. COHESION_PRIORITY requires flock_splitting; HARD_TO_GUIDE requires stagnation. Calls normally have both alarms; their rule fallback is HARD_TO_GUIDE. Prefer rule_mode unless another admissible mode clearly improves current geometry and short-horizon estimates without meaningfully worsening goal progress, dispersion, predicted goal exits, or boundary safety. Avoid a repair mode if its alarm looks brief or nearly resolved: API latency may invalidate it. Select target_sheep_id only from the chosen mode's repair_candidates (at most three, including its eligible rule target), or null for low-level rule selection. TARGET_TRACKING requires null.

MODE AND TARGET JUDGMENT
COHESION_PRIORITY collects a selected sheep, defaulting to a predicted outward outlier. Favor it for high/worsening spread, an outward-moving cohesion candidate, or a Type B main dispersion target; with both alarms, state and forecast must justify overriding HARD_TO_GUIDE. HARD_TO_GUIDE drives a selected sheep toward the goal, defaulting to a predicted goal-distance laggard. Favor it for poor progress, weak candidate goalward motion, or a Type A main lagging target. TARGET_TRACKING suits ordinary drive/collect already recovering the flock or repair likely to prolong a poor trajectory. Compare candidate geometry and individual forecasts; never switch mode or target solely due to sheep type.
Use dog travel, approach alignment, and drive-point boundary margin to avoid slow/risky interventions. Alignment near 1 means the dog is on the desired side; negative drive-point margin means outside the workspace. Positive goalward speed is toward the goal; outward speed is velocity relative to the moving flock centroid projected away from it, positive when separating from the flock. Type A has reduced dog repulsion and normal flocking/boundary response; Type B has weak cohesion and stronger separation; Type C has normal flocking and stronger near-dog repulsion per type_c_response when present.

FORECASTS AND EXECUTION
mode_forecasts are independent 30-frame shadow simulations using the same kinematic model for the three modes, not guarantees of long-term success. Favor more centroid_progress and inside_goal_change; less final_max_spread, predicted_goal_exits, predicted_splitting_frames, and peak_out_of_bounds. Goal exits and out-of-bounds motion are strong costs; use static geometry to interpret trade-offs. At response arrival the interface rechecks mode AND target, rejecting choices Pareto-dominated by another admissible mode or that mode's rule target; this filter is not proof of safety. Use supplied values exactly. Choose hold_frames 15-60, normally near the 30-frame horizon, shorter for fast-changing/risky states.

NAMED INPUT
All numbers retain their names, units, precision, and candidate order. If input_format is shepherd-named-dedup-v1, data is under state. If cohesion_candidate or hard_to_guide_candidate has geometry_from_repair_candidate=true, its omitted geometry equals the same target_id row in repair_candidates.COHESION_PRIORITY or repair_candidates.HARD_TO_GUIDE respectively. Read that row, then override with the default object's explicit fields (which may differ). Defaults without that marker are complete. No forecasts are removed. Current drive_point_boundary_margin and forecast minimum_boundary_margin differ. Otherwise read the original named JSON state."""
    if not mode_selection:
        text += "\nTARGET-ONLY: Copy rule_mode exactly; choose its supplied target or null. hold_frames holds only the target, never the mode; a rule mode change releases it."
    if not target_selection:
        text += "\nMODE-ONLY: Always return target_sheep_id=null; candidates are mode-selection context and the low-level rule selects targets."
    if compact:
        text += "\nRETURN only JSON m,t,h: m=0 for TARGET_TRACKING, 1 for COHESION_PRIORITY, 2 for HARD_TO_GUIDE; t=target_sheep_id or null; h=hold_frames. Apply all permissions above. Omit reason and other fields."
    else:
        text += "\nRETURN only schema JSON mode,target_sheep_id,hold_frames,reason; reason under 20 words."
    return text


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def pack(state):
    if not isinstance(state, dict) or "default_target_ids" in state or "input_format" in state:
        raise ValueError("Unsupported named input shape")
    result = copy.deepcopy(state)
    groups = state.get("repair_candidates", {})
    if not isinstance(groups, dict):
        raise ValueError("Unsupported candidates")
    for key, mode in DEFAULTS.items():
        geometry = state.get(key)
        if not isinstance(geometry, dict) or type(geometry.get("target_id")) is not int:
            continue
        if "geometry_from_repair_candidate" in geometry:
            raise ValueError("Reserved geometry field")
        rows = groups.get(mode, [])
        if not isinstance(rows, list):
            raise ValueError("Unsupported candidates")
        matches = [row for row in rows if isinstance(row, dict) and row.get("target_id") == geometry["target_id"]]
        # Exact serialized equality also preserves signed zero and int/float types.
        if len(matches) == 1:
            shared = {k: v for k, v in matches[0].items() if k != "forecast"}
            if shared.keys() != geometry.keys():
                continue
            explicit = {k: copy.deepcopy(v) for k, v in geometry.items()
                        if k == "target_id" or canonical(v) != canonical(shared[k])}
            if len(explicit) < len(geometry):
                result[key] = {**explicit, "geometry_from_repair_candidate": True}
    return {"input_format": VERSION, "state": result}


def unpack(payload):
    if not isinstance(payload, dict) or set(payload) != {"input_format", "state"} or payload["input_format"] != VERSION:
        raise ValueError("Unsupported named input envelope")
    state = copy.deepcopy(payload["state"])
    for key, mode in DEFAULTS.items():
        geometry = state.get(key, {})
        if isinstance(geometry, dict) and geometry.get("geometry_from_repair_candidate") is True:
            rows = [r for r in state["repair_candidates"][mode] if r["target_id"] == geometry["target_id"]]
            if len(rows) != 1:
                raise ValueError("Missing or ambiguous default target")
            state[key] = {**{k: copy.deepcopy(v) for k, v in rows[0].items() if k != "forecast"},
                          **{k: copy.deepcopy(v) for k, v in geometry.items() if k != "geometry_from_repair_candidate"}}
    return state


def encode_json(original):
    state = json.loads(original)
    try:
        payload = pack(state)
    except ValueError:
        return original
    if canonical(unpack(payload)) != canonical(state):
        raise ValueError("Named input round-trip validation failed")
    return canonical(payload)
