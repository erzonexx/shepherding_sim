"""Diagnose controller behavior without rotating or overwriting experiment logs."""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment import SwarmEnv
from run_experiments import CASES, make_detector, temporary_config
from high_level_controller import build_controller


def segment_distances(points, start, end):
    """Euclidean distances to a finite segment, including a stationary segment."""
    delta = end - start
    length_squared = float(delta @ delta)
    if length_squared == 0:
        return np.linalg.norm(points - start, axis=1)
    fractions = np.clip((points - start) @ delta / length_squared, 0, 1)
    return np.linalg.norm(points - (start + fractions[:, None] * delta), axis=1)


def motion_diagnostics(env, positions_before, velocities_before, dog_before):
    """Geometry and exposure observations, not attribution of a failure cause."""
    goal_vectors = positions_before - env.goal_pos
    goal_distances = np.linalg.norm(goal_vectors, axis=1)
    inside = goal_distances < config.GOAL_RADIUS
    radial = goal_vectors / (goal_distances[:, None] + config.EPSILON)
    from_dog = positions_before - env.dog_pos
    dog_distances = np.linalg.norm(from_dog, axis=1)
    # Match the force used during this update, after dog motion and before sheep motion.
    weights = np.where(env.sheep_status == "A",
                       config.WEIGHT_DOG_REPULSION * config.WEIGHT_DOG_REPULSION_A_FACTOR,
                       config.WEIGHT_DOG_REPULSION)
    response = np.where(env.sheep_status == "A", 1 - config.INERTIA_FACTOR_A, 1.)
    dog_delta_v = from_dog * (weights * response / (dog_distances + 1e-6) ** 2)[:, None]
    dog_delta_v[dog_distances > config.DOG_SENSING_RANGE] = 0
    outward = np.sum(dog_delta_v * radial, axis=1)
    outward_velocity = np.sum(velocities_before * radial, axis=1)
    target = env.target_sheep_id
    gap = float(np.linalg.norm(env.drive_point - env.dog_pos))
    row = {
        "dog_x_before": float(dog_before[0]), "dog_y_before": float(dog_before[1]),
        "dog_x_after": float(env.dog_pos[0]), "dog_y_after": float(env.dog_pos[1]),
        "dog_goal_distance": float(np.linalg.norm(env.dog_pos - env.goal_pos)),
        "drive_point_gap_before": float(np.linalg.norm(env.drive_point - dog_before)),
        "drive_point_gap_after": gap,
        "near_drive_point": gap <= config.DOG_MAX_SPEED,
        "inside_sheep_outward_velocity_count": int(np.sum(inside & (outward_velocity > 0))),
        "inside_sheep_outward_dog_response_count": int(np.sum(inside & (outward > 0))),
        "inside_sheep_outward_dog_response_sum": float(np.maximum(outward[inside], 0).sum()),
        "target_goal_distance_before": None if target is None else float(goal_distances[target]),
        "target_dog_distance": None if target is None else float(dog_distances[target]),
        "target_outward_dog_response": None if target is None else float(outward[target]),
        "target_outward_velocity_before": None if target is None else float(outward_velocity[target]),
        "planned_segment_min_inside_distance": None,
        "executed_segment_min_inside_distance": None,
    }
    if np.any(inside):
        row["planned_segment_min_inside_distance"] = float(segment_distances(positions_before[inside], dog_before, env.drive_point).min())
        row["executed_segment_min_inside_distance"] = float(segment_distances(positions_before[inside], dog_before, env.dog_pos).min())
    return row


def observe(env, state_frame):
    positions = env.sheep_pos
    center = positions.mean(axis=0)
    goal_distances = np.linalg.norm(positions - env.goal_pos, axis=1)
    spread = np.linalg.norm(positions - center, axis=1)
    excursions = np.maximum(np.maximum(-positions, positions - config.SPACE_SIZE), 0)
    outside = np.any(excursions > 0, axis=1)
    return {
        "state_frame": state_frame,
        "centroid_distance": float(np.linalg.norm(center - env.goal_pos)),
        "max_goal_distance": float(goal_distances.max()),
        "inside_goal": int(np.sum(goal_distances < config.GOAL_RADIUS)),
        "max_spread": float(spread.max()),
        "mean_spread": float(spread.mean()),
        "outside_count": int(outside.sum()),
        "max_excursion": float(excursions.max()),
        "outside_ids": ";".join(str(i) for i in np.flatnonzero(outside)),
        "goal_x": float(env.goal_pos[0]), "goal_y": float(env.goal_pos[1]),
    }


def summarize(states, actions, success, step_calls, digest):
    first_outside = next((s["state_frame"] for s in states if s["outside_count"]), None)
    peak = max(states, key=lambda s: s["inside_goal"])
    closest = min(states, key=lambda s: s["centroid_distance"])
    mode_switches = 0
    target_switches = 0
    same_action_target_switches = 0
    short_mode_runs = 0
    run_length = 1
    for previous, current in zip(actions, actions[1:]):
        if current["mode"] != previous["mode"]:
            mode_switches += 1
            short_mode_runs += run_length <= 5
            run_length = 1
        else:
            run_length += 1
        if (previous["target_id"] is not None and current["target_id"] is not None
                and previous["target_id"] != current["target_id"]):
            target_switches += 1
            same_action_target_switches += (
                previous["mode"] == current["mode"]
                and previous["action"] == current["action"]
            )
    source_counts = Counter(a.get("decision_source", "unknown") for a in actions)
    status_counts = Counter(a.get("high_level_status") for a in actions)
    status_counts.pop(None, None)
    request_ids = {a.get("llm_request_id") for a in actions if a.get("llm_request_id")}
    response_ids = {
        a.get("llm_request_id")
        for a in actions
        if a.get("decision_source") not in (None, "rule")
        and a.get("llm_request_id")
    }
    latencies = sorted({
        int(a["llm_latency_frames"])
        for a in actions
        if a.get("llm_latency_frames") is not None
    })
    response_records = {}
    for action in actions:
        request_id = action.get("llm_request_id")
        if request_id and action.get("decision_source") not in (None, "rule"):
            response_records.setdefault(request_id, action)
    return {
        "success_reported_by_env": success,
        "all_inside_at_final_state": states[-1]["inside_goal"] == config.NUM_SHEEP,
        "step_calls": step_calls,
        "physics_updates": len(actions),
        "trajectory_sha256": digest,
        "first_outside_state_frame": first_outside,
        "outside_state_samples": sum(s["outside_count"] > 0 for s in states),
        "max_excursion": max(s["max_excursion"] for s in states),
        "peak_inside_goal": peak["inside_goal"],
        "peak_inside_state_frame": peak["state_frame"],
        "final_inside_goal": states[-1]["inside_goal"],
        "closest_centroid_state": closest,
        "mode_switches": mode_switches,
        "completed_mode_runs_at_most_5_updates": short_mode_runs,
        "target_switches": target_switches,
        "same_mode_action_target_switches": same_action_target_switches,
        "individual_goal_exits": sum(a["goal_exits"] for a in actions),
        "decision_source_counts": dict(source_counts),
        "high_level_status_counts": dict(status_counts),
        "high_level_request_count": len(request_ids),
        "high_level_response_count": len(response_ids),
        "high_level_rejected_response_count": sum(
            a.get("high_level_status") == "rejected" for a in actions
        ),
        "high_level_latency_frames": latencies,
        "high_level_latency_seconds": [
            record["llm_latency_seconds"]
            for record in response_records.values()
            if record.get("llm_latency_seconds") is not None
        ],
        "high_level_cache_hit_count": sum(
            bool(record.get("llm_cache_hit"))
            for record in response_records.values()
        ),
        "high_level_input_tokens": sum(
            int(record.get("llm_input_tokens") or 0)
            for record in response_records.values()
        ),
        "high_level_output_tokens": sum(
            int(record.get("llm_output_tokens") or 0)
            for record in response_records.values()
        ),
        "high_level_selections_evaluated": sum(
            record.get("llm_reference_mode") is not None
            for record in response_records.values()
        ),
        "high_level_selections_correct": sum(
            record.get("llm_selection_correct") is True
            for record in response_records.values()
        ),
    }


def diagnose(case, seed, max_steps, controller_kind="rule"):
    with temporary_config(
        USE_FIXED_SEED=True, RANDOM_SEED=seed,
        NUM_ABNORMAL_A=case["num_abnormal_a"],
        NUM_ABNORMAL_B=case["num_abnormal_b"],
        REPAIR_ENABLED=case["repair_enabled"], MAX_STEPS=max_steps,
    ):
        env = SwarmEnv(controller=build_controller(controller_kind))
        detector = make_detector()
        states = [observe(env, 0)]
        actions = []
        digest = hashlib.sha256()
        success = False
        for step in range(max_steps):
            alarms, _, _ = detector.detect(env.sheep_pos, env.goal_pos)
            positions_before = env.sheep_pos.copy()
            velocities_before = env.sheep_vel.copy()
            dog_before = env.dog_pos.copy()
            inside_before = np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1) < config.GOAL_RADIUS
            success = bool(env.step(alarms))
            if (
                controller_kind in ("openai", "deepseek")
                and getattr(env.controller, "pending_request", None) is not None
            ):
                time.sleep(config.LLM_PENDING_POLL_SECONDS)
            for array in (env.sheep_pos, env.sheep_vel, env.dog_pos):
                digest.update(array.tobytes())
            # step() checks success before moving: a terminal call is not an action.
            if success:
                break
            inside_after = np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1) < config.GOAL_RADIUS
            states.append(observe(env, len(actions) + 1))
            actions.append({
                "from_state_frame": len(actions),
                "to_state_frame": len(actions) + 1,
                "alarm": ";".join(alarms) or "none",
                "mode": env.current_mode,
                "decision_source": env.decision_source,
                "decision_issued_for_frame": env.decision_frame,
                "high_level_status": env.decision_metadata.get("high_level_status"),
                "llm_request_id": env.decision_metadata.get("request_id"),
                "llm_trigger_frame": env.decision_metadata.get("trigger_frame"),
                "llm_trigger_alarm": env.decision_metadata.get("trigger_alarm"),
                "llm_trigger_alarms": env.decision_metadata.get("trigger_alarms"),
                "llm_state_summary": env.decision_metadata.get("state_summary"),
                "llm_selected_mode": env.decision_metadata.get("selected_mode"),
                "llm_reference_mode": env.decision_metadata.get("reference_mode"),
                "llm_selection_correct": env.decision_metadata.get("selection_correct"),
                "llm_latency_frames": env.decision_metadata.get("latency_frames"),
                "llm_latency_seconds": env.decision_metadata.get("latency_seconds"),
                "llm_reason": env.decision_metadata.get("reason"),
                "llm_model": env.decision_metadata.get("model"),
                "llm_prompt_version": env.decision_metadata.get("prompt_version"),
                "llm_cache_hit": env.decision_metadata.get("cache_hit"),
                "llm_response_id": env.decision_metadata.get("response_id"),
                "llm_response_status": env.decision_metadata.get("response_status"),
                "llm_incomplete_reason": env.decision_metadata.get("incomplete_reason"),
                "llm_input_tokens": env.decision_metadata.get("input_tokens"),
                "llm_output_tokens": env.decision_metadata.get("output_tokens"),
                "llm_backend_error": env.decision_metadata.get("backend_error"),
                "llm_rejection_reason": env.decision_metadata.get("rejection_reason"),
                "action": env.control_action,
                "target_id": env.target_sheep_id,
                "navigation_point_x": float(env.navigation_point[0]),
                "navigation_point_y": float(env.navigation_point[1]),
                "drive_point_x": float(env.drive_point[0]),
                "drive_point_y": float(env.drive_point[1]),
                "drive_point_outside": bool(np.any((env.drive_point < 0) | (env.drive_point > config.SPACE_SIZE))),
                "goal_exits": int(np.sum(inside_before & ~inside_after)),
                "goal_entries": int(np.sum(~inside_before & inside_after)),
                **motion_diagnostics(env, positions_before, velocities_before, dog_before),
            })
        summary = summarize(states, actions, success, step + 1, digest.hexdigest())
        summary.update({"case": case["case"], "seed": seed, "A": config.NUM_ABNORMAL_A,
                        "B": config.NUM_ABNORMAL_B, "repair": config.REPAIR_ENABLED,
                        "goal_retention_enabled": config.GOAL_RETENTION_ENABLED,
                        "repair_target_selection_enabled": config.REPAIR_TARGET_SELECTION_ENABLED,
                        "controller": controller_kind})
        env.close()
        return summary, states, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[70])
    parser.add_argument("--cases", nargs="+", choices=[c["case"] for c in CASES], default=[c["case"] for c in CASES])
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "logs" / "diagnostics",
    )
    parser.add_argument("--compare-repair-targeting", action="store_true", help="Compare legacy and predictive repair-target selection")
    parser.add_argument("--summary-only", action="store_true", help="Skip per-frame CSVs for batch comparisons")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run console output")
    parser.add_argument("--controller", choices=("rule", "mock-llm", "openai", "deepseek"), default="rule")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.controller in ("openai", "deepseek"):
        try:
            probe = build_controller(args.controller)
        except RuntimeError as exc:
            raise SystemExit(f"Controller setup failed: {exc}") from None
        probe.close()
    output = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    settings = {k: v for k, v in vars(config).items() if k.isupper() and isinstance(v, (str, int, float, bool))}
    metadata = {
        "base_config": settings, "seeds": args.seeds, "cases": args.cases,
        "max_steps": args.max_steps, "numpy_version": np.__version__,
        "compare_repair_targeting": args.compare_repair_targeting,
        "controller": args.controller,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "environment.py", "control_types.py", "low_level_controller.py",
                "high_level_controller.py", "responses_api_backend.py",
                "detector.py", "config.py",
            )
        } | {
            "tools/diagnose_failures.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
        "frame_convention": "state_frame 0 is initial; each action connects pre/post states. Terminal success calls do not move agents.",
        "switch_convention": "Adjacent executed actions only; initial selection is not a switch. Target switches exclude transitions to/from no individual target.",
        "short_run_convention": "Completed mode runs <=5 updates; last censored run excluded. Diagnostic count, not a proven fault.",
        "motion_convention": "Route distances use pre-update sheep positions. Planned segment is not necessarily executed. near_drive_point means post-dog gap <= DOG_MAX_SPEED. Outward dog response is radial velocity increment before speed clipping, not net motion or a causal estimate. All geometry is pre-sheep-motion unless named otherwise.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summaries = []
    for seed in dict.fromkeys(args.seeds):
        for case in CASES:
            if case["case"] not in args.cases:
                continue
            variants = (
                [False, True]
                if args.compare_repair_targeting
                else [config.REPAIR_TARGET_SELECTION_ENABLED]
            )
            for repair_targeting in variants:
                with temporary_config(
                    REPAIR_TARGET_SELECTION_ENABLED=repair_targeting
                ):
                    summary, states, actions = diagnose(
                        case, seed, args.max_steps, controller_kind=args.controller
                    )
                prefix = (
                    f"{case['case']}_seed{seed}_"
                    f"repairtarget{int(repair_targeting)}"
                )
                if args.controller != "rule":
                    prefix += f"_{args.controller}"
                if not args.summary_only:
                    pd.DataFrame(states).to_csv(output / f"{prefix}_states.csv", index=False)
                    pd.DataFrame(actions).to_csv(output / f"{prefix}_actions.csv", index=False)
                summaries.append(summary)
                if args.quiet:
                    continue
                if args.summary_only:
                    print(f"{prefix}: success={summary['success_reported_by_env']}, "
                          f"calls={summary['step_calls']}", flush=True)
                else:
                    print(json.dumps(summary), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Diagnostics saved to {output.resolve()}")


if __name__ == "__main__":
    main()
