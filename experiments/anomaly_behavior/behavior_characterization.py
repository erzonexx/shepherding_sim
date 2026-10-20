"""Small deterministic trials that characterize the current anomaly model."""

import csv
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
from detector import Detector
from dynamics import advance_sheep
from environment import SwarmEnv
from low_level_controller import LowLevelController
from low_level_controller import rule_mode_for_alarms
from high_level_controller import HighLevelController, MockLLMBackend
from run_experiments import (
    make_detector, parse_args, run_case, select_cases, temporary_config,
    write_experiment_metadata, write_summary,
)


DOG_RESPONSE_DISTANCES = (3.0, 10.0, 20.0, 40.0, 60.0)
DISPERSION_SAMPLE_FRAMES = (0, 1, 10, 30, 60)
CONTROLLED_DOG_DISTANCES = (3.0, 10.0, 20.0, 25.0, 40.0, 60.0)


def dog_response_curve(distances=DOG_RESPONSE_DISTANCES, statuses=("Normal", "A")):
    """Measure one-frame motion caused only by a stationary dog."""
    rows = []
    for distance in distances:
        for status in statuses:
            positions = np.array([[100.0, 100.0]])
            velocities = np.zeros_like(positions)
            next_positions, next_velocities = advance_sheep(
                positions,
                velocities,
                np.array([status], dtype=object),
                np.array([100.0 - distance, 100.0]),
                np.array([180.0, 180.0]),
                config,
            )
            rows.append({
                "status": status,
                "dog_distance": float(distance),
                "displacement": float(np.linalg.norm(next_positions[0] - positions[0])),
                "speed": float(np.linalg.norm(next_velocities[0])),
            })
    return rows


def type_b_dispersion_trial(
    frames=60,
    sample_frames=DISPERSION_SAMPLE_FRAMES,
    statuses=("Normal", "B"),
    observation_decimals=None,
):
    """Compare focus behaviors in the same compact flock without dog exposure."""
    initial_positions = np.array([
        [96.0, 96.0], [100.0, 96.0], [104.0, 96.0],
        [96.0, 100.0], [100.0, 100.0], [104.0, 100.0],
        [96.0, 104.0], [100.0, 104.0], [104.0, 104.0],
    ])
    focus_id = 5
    requested_frames = sorted(set(sample_frames) | {0, frames})
    rows = []
    for focus_status in statuses:
        positions = initial_positions.copy()
        velocities = np.zeros_like(positions)
        sheep_statuses = np.array(["Normal"] * len(positions), dtype=object)
        sheep_statuses[focus_id] = focus_status
        previous_observed = None
        # The distant goal leaves retention inactive without changing its settings.
        for frame in range(frames + 1):
            observed = (np.round(positions, observation_decimals)
                        if observation_decimals is not None else None)
            if frame in requested_frames:
                center = np.mean(positions, axis=0)
                spread = np.linalg.norm(positions - center, axis=1)
                row = {
                    "variant": ("all_normal" if focus_status == "Normal"
                                else f"one_type_{focus_status.lower()}"),
                    "frame": frame,
                    "focus_distance_to_centroid": float(spread[focus_id]),
                    "mean_spread": float(np.mean(spread)),
                    "max_spread": float(np.max(spread)),
                    "focus_speed": float(np.linalg.norm(velocities[focus_id])),
                }
                if observed is not None:
                    observed_spread = np.linalg.norm(observed - observed.mean(axis=0), axis=1)
                    row["observed_focus_distance_to_centroid"] = float(observed_spread[focus_id])
                    row["observed_focus_speed"] = (
                        float(np.linalg.norm(observed[focus_id] - previous_observed[focus_id]))
                        if previous_observed is not None else 0.0)
                rows.append(row)
            previous_observed = observed
            if frame == frames:
                break
            positions, velocities = advance_sheep(
                positions,
                velocities,
                sheep_statuses,
                np.array([-1000.0, -1000.0]),
                np.array([180.0, 180.0]),
                config,
            )
    return rows


def normal_initial_split_screen(seed_count=1000):
    """Measure frame-zero splitting alarms for normal initialized flocks."""
    rows = []
    with temporary_config(
        NUM_ABNORMAL_A=0,
        NUM_ABNORMAL_B=0,
        REPAIR_ENABLED=False,
        USE_FIXED_SEED=True,
    ):
        for seed in range(seed_count):
            with temporary_config(RANDOM_SEED=seed):
                env = SwarmEnv()
                detector = Detector(
                    config.DANGER_DISPERSION_THRESHOLD,
                    config.DANGER_MEAN_SPREAD_THRESHOLD,
                    config.DANGER_STAGNATION_FRAMES,
                    config.DANGER_STAGNATION_THRESHOLD,
                )
                alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
                rows.append({
                    "seed": seed,
                    "max_spread": metrics["dispersion"],
                    "mean_spread": metrics["mean_spread"],
                    "splitting_alarm": "flock_splitting" in alarms,
                })
                env.close()
    return rows


def splitting_trajectory_screen(seed_count=100, frames=60):
    """Compare raw splitting alarms in matched normal and Type B runs."""
    sample_frames = tuple(frame for frame in (0, 10, 30, 60) if frame <= frames)
    summary = {}
    for population, type_b_count in (("normal", 0), ("type_b", 4)):
        samples = {frame: [] for frame in sample_frames}
        late_alarm = []
        with temporary_config(
            NUM_ABNORMAL_A=0,
            NUM_ABNORMAL_B=type_b_count,
            REPAIR_ENABLED=False,
            USE_FIXED_SEED=True,
        ):
            for seed in range(seed_count):
                with temporary_config(RANDOM_SEED=seed):
                    env = SwarmEnv()
                    detector = Detector(
                        config.DANGER_DISPERSION_THRESHOLD,
                        config.DANGER_MEAN_SPREAD_THRESHOLD,
                        config.DANGER_STAGNATION_FRAMES,
                        config.DANGER_STAGNATION_THRESHOLD,
                    )
                    had_late_alarm = False
                    for frame in range(frames + 1):
                        alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
                        split = "flock_splitting" in alarms
                        if frame in samples:
                            samples[frame].append((
                                split,
                                metrics["dispersion"],
                                metrics["mean_spread"],
                            ))
                        if frame >= 10 and split:
                            had_late_alarm = True
                        if frame < frames and env.step([]):
                            break
                    late_alarm.append(had_late_alarm)
                    env.close()
        summary[population] = {
            "sample_alarm_rates": {
                frame: float(np.mean([row[0] for row in values]))
                for frame, values in samples.items()
            },
            "sample_mean_max_spread": {
                frame: float(np.mean([row[1] for row in values]))
                for frame, values in samples.items()
            },
            "sample_mean_spread": {
                frame: float(np.mean([row[2] for row in values]))
                for frame, values in samples.items()
            },
            "any_alarm_from_frame_10": float(np.mean(late_alarm)),
        }
    return summary


def stress_alarm_screen(seed_count=20, max_steps=1000):
    """Count alarm opportunities in mixed stress runs without repair."""
    from run_experiments import CASES
    from tools.diagnose_failures import diagnose

    case = next(case for case in CASES if case["case"] == "C07")
    rows = []
    for seed in range(seed_count):
        result, _, actions = diagnose(case, seed, max_steps)
        alarm_sets = [set(action["alarm"].split(";")) for action in actions]
        split_frames = [
            action["from_state_frame"] for action, alarms in zip(actions, alarm_sets)
            if "flock_splitting" in alarms
        ]
        stagnation_frames = [
            action["from_state_frame"] for action, alarms in zip(actions, alarm_sets)
            if "stagnation" in alarms
        ]
        rows.append({
            "seed": seed,
            "success": bool(result["success_reported_by_env"]),
            "splitting_frames": len(split_frames),
            "first_splitting_frame": split_frames[0] if split_frames else None,
            "last_splitting_frame": split_frames[-1] if split_frames else None,
            "stagnation_frames": len(stagnation_frames),
            "first_stagnation_frame": (
                stagnation_frames[0] if stagnation_frames else None
            ),
            "concurrent_frames": sum(
                {"flock_splitting", "stagnation"} <= alarms
                for alarms in alarm_sets
            ),
        })
    return {
        "run_count": seed_count,
        "runs_with_splitting": sum(row["splitting_frames"] > 0 for row in rows),
        "runs_with_stagnation": sum(row["stagnation_frames"] > 0 for row in rows),
        "runs_with_concurrent_alarms": sum(
            row["concurrent_frames"] > 0 for row in rows
        ),
        "runs_with_only_early_splitting": sum(
            row["splitting_frames"] > 0
            and row["last_splitting_frame"] < config.DANGER_STAGNATION_FRAMES - 1
            for row in rows
        ),
        "total_splitting_frames": sum(row["splitting_frames"] for row in rows),
        "total_stagnation_frames": sum(row["stagnation_frames"] for row in rows),
        "total_concurrent_frames": sum(row["concurrent_frames"] for row in rows),
        "runs": rows,
    }


def build_report(initial_seed_count=1000, dispersion_frames=60):
    dog_rows = dog_response_curve()
    dispersion_rows = type_b_dispersion_trial(frames=dispersion_frames)
    initial_rows = normal_initial_split_screen(initial_seed_count)

    normal_response = next(
        row["speed"] for row in dog_rows
        if row["status"] == "Normal" and row["dog_distance"] == 20.0
    )
    type_a_response = next(
        row["speed"] for row in dog_rows
        if row["status"] == "A" and row["dog_distance"] == 20.0
    )
    normal_final = next(
        row for row in dispersion_rows
        if row["variant"] == "all_normal" and row["frame"] == dispersion_frames
    )
    type_b_final = next(
        row for row in dispersion_rows
        if row["variant"] == "one_type_b" and row["frame"] == dispersion_frames
    )
    initial_split_rate = float(np.mean([
        row["splitting_alarm"] for row in initial_rows
    ]))
    response_ratio = type_a_response / normal_response if normal_response else None
    dispersion_ratio = (
        type_b_final["focus_distance_to_centroid"]
        / normal_final["focus_distance_to_centroid"]
        if normal_final["focus_distance_to_centroid"] else None
    )

    summary = {
        "parameters": {
            "dog_response_a_factor": config.WEIGHT_DOG_REPULSION_A_FACTOR,
            "type_b_cohesion_factor": config.WEIGHT_COHESION_B_FACTOR,
            "type_b_separation_factor": config.WEIGHT_SEPARATION_B_FACTOR,
            "normal_separation_radius": config.SEPARATION_RADIUS_NORMAL,
            "type_b_separation_radius": config.SEPARATION_RADIUS_B,
            "initial_spread_radius": config.SHEEP_START_RADIUS,
            "splitting_max_threshold": config.DANGER_DISPERSION_THRESHOLD,
            "splitting_mean_threshold": config.DANGER_MEAN_SPREAD_THRESHOLD,
        },
        "type_a": {
            "dog_distance": 20.0,
            "normal_one_frame_speed": normal_response,
            "type_a_one_frame_speed": type_a_response,
            "type_a_to_normal_response_ratio": response_ratio,
        },
        "type_b": {
            "frames": dispersion_frames,
            "normal_focus_distance": normal_final["focus_distance_to_centroid"],
            "type_b_focus_distance": type_b_final["focus_distance_to_centroid"],
            "type_b_to_normal_focus_distance_ratio": dispersion_ratio,
            "normal_mean_spread": normal_final["mean_spread"],
            "type_b_mean_spread": type_b_final["mean_spread"],
        },
        "normal_initialization": {
            "seed_count": initial_seed_count,
            "frame_zero_splitting_rate": initial_split_rate,
            "mean_max_spread": float(np.mean([row["max_spread"] for row in initial_rows])),
            "mean_mean_spread": float(np.mean([row["mean_spread"] for row in initial_rows])),
        },
        "review_flags": {
            "type_a_response_below_5_percent_of_normal": (
                response_ratio is not None and response_ratio < 0.05
            ),
            "normal_initial_split_rate_above_5_percent": initial_split_rate > 0.05,
        },
    }
    return summary, dog_rows, dispersion_rows, initial_rows


def write_characterization(
    output_root=None,
    initial_seed_count=1000,
    dispersion_frames=60,
):
    """Write one timestamped behavior report and its raw tables."""
    root = Path(output_root or config.PROJECT_ROOT / "logs" / "anomaly_behavior")
    output = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    summary, dog_rows, dispersion_rows, initial_rows = build_report(
        initial_seed_count=initial_seed_count,
        dispersion_frames=dispersion_frames,
    )
    (output / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output / "type_a_dog_response.csv", dog_rows)
    _write_csv(output / "type_b_dispersion.csv", dispersion_rows)
    _write_csv(output / "normal_initial_split.csv", initial_rows)
    return output, summary


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_controlled_diagnosis_trial(output_root=None):
    """Measure ideal exposure signals; no inference or online control changes."""
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    statuses = ("Normal", "A", "B", "C")
    protocol = {
        "api_calls": 0,
        "statuses": statuses,
        "dog_distances": CONTROLLED_DOG_DISTANCES,
        "dog_trial": "Reset one sheep at [100,100] with zero velocity; dog on negative x axis; one physics update per distance/type.",
        "flock_trial": "Reset identical nine-sheep 4-unit grid, zero velocities, eight Normal peers, focus ID5; dog out of sensing range; 60 updates.",
        "goal": [180.0, 180.0],
        "observation_decimals": 2,
        "production_settings_changed": False,
        "replication": "Deterministic controlled configurations, not independent random-seed trials.",
        "limits": "Known Normal peers and isolated exposure are ideal conditions, not available automatically in a mixed flock. No classification accuracy, repair benefit or LLM benefit is measured.",
        "settings": {key: getattr(config, key) for key in (
            "SHEEP_MAX_SPEED", "WEIGHT_COHESION", "WEIGHT_DOG_REPULSION",
            "WEIGHT_SEPARATION", "WEIGHT_DOG_REPULSION_A_FACTOR",
            "WEIGHT_COHESION_B_FACTOR", "WEIGHT_SEPARATION_B_FACTOR",
            "SEPARATION_RADIUS_NORMAL", "SEPARATION_RADIUS_B", "DOG_SENSING_RANGE",
            "TYPE_C_SENSITIVITY_RADIUS", "WEIGHT_DOG_REPULSION_C_FACTOR",
            "GOAL_RETENTION_ENABLED", "GOAL_RADIUS", "SPACE_SIZE",
        )},
    }
    (output / "controlled_diagnosis_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    dog_rows = dog_response_curve(CONTROLLED_DOG_DISTANCES, statuses)
    for row in dog_rows:
        # Motion is along x from an integer initial coordinate, so this matches rounded positions.
        row["observed_displacement"] = float(np.round(100.0 + row["displacement"], 2) - 100.0)
    flock_rows = type_b_dispersion_trial(
        frames=60, sample_frames=range(61), statuses=statuses, observation_decimals=2)
    summary = {"dog_response": {}, "no_dog_flock": {}}
    for distance in CONTROLLED_DOG_DISTANCES:
        rows = [row for row in dog_rows if row["dog_distance"] == distance]
        summary["dog_response"][str(distance)] = {
            row["status"]: {"speed": row["speed"], "observed_displacement": row["observed_displacement"]}
            for row in rows}
    for status in statuses:
        variant = "all_normal" if status == "Normal" else f"one_type_{status.lower()}"
        rows = [row for row in flock_rows if row["variant"] == variant]
        summary["no_dog_flock"][status] = {
            "initial_distance": rows[0]["focus_distance_to_centroid"],
            "final_distance": rows[-1]["focus_distance_to_centroid"],
            "mean_distance_frames_1_60": statistics.mean(row["focus_distance_to_centroid"] for row in rows[1:]),
            "observed_mean_distance_frames_1_60": statistics.mean(row["observed_focus_distance_to_centroid"] for row in rows[1:]),
            "max_distance": max(row["focus_distance_to_centroid"] for row in rows),
        }
    report = {"api_calls": 0, "dog_configurations": len(dog_rows),
              "flock_trajectories": len(statuses), "summary": summary, "limits": protocol["limits"]}
    _write_csv(output / "controlled_dog_response.csv", dog_rows)
    _write_csv(output / "controlled_no_dog_flock.csv", flock_rows)
    (output / "controlled_diagnosis_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return output, report


BLIND_DIAGNOSIS_CHOICES = ("normal_response", "weak_dog_response", "dispersing",
                           "enhanced_near_dog_response", "insufficient_evidence")
BLIND_DIAGNOSIS_PROMPT = "shepherd-blind-behavior-v1"
BLIND_DIAGNOSIS_INSTRUCTIONS = """Diagnose observed behavior of one sheep from an anonymized
motion clip. You receive ONLY measured geometry and motion, not identities, actual force
coefficients, type counts, controller decisions or model forecasts. Choose one diagnosis:
normal_response (ordinary response without sustained dispersal), weak_dog_response
(weak observable response to a nearby dog), dispersing (persistent outward motion/weak
group following), enhanced_near_dog_response (unusually strong response on close approach),
or insufficient_evidence. These are hypotheses, not guaranteed causal identifications.
Peer reference sheep are unlabeled and can themselves be heterogeneous. Compare similar
exposure and flock geometry, and distinguish dog response from cohesion, collisions and
goal proximity. Coordinates are rounded to 0.01; velocities/accelerations are finite
differences. Speed saturation can hide force-response differences. Do NOT infer a hidden
type from an ID or invent a parameter. Abstain when the clip cannot distinguish causes.
Return a JSON object with diagnosis, up to three supporting relative evidence_frames
chosen ONLY from the displayed sampled frames,
and a brief reason (at most 240 characters). Do not propose control actions."""


def blind_motion_payload(data, focus_id, start_frame, goal_position):
    """Explicit geometry-only whitelist; labels cannot enter derived features."""
    columns = ["Frame", "Agent_Type", "Agent_ID", "X", "Y"]
    observed = data.loc[data.Frame.between(start_frame - 2, start_frame + 60), columns]
    frames = list(range(start_frame - 2, start_frame + 61))
    sheep = observed.loc[observed.Agent_Type.eq("Sheep")]
    dogs = observed.loc[observed.Agent_Type.eq("Dog")].set_index("Frame")
    ids = sorted(sheep.Agent_ID.unique().tolist())
    if focus_id not in ids or start_frame < 2:
        raise ValueError("Focus ID and two preceding observations are required")
    positions = []
    for frame in frames:
        rows = sheep.loc[sheep.Frame.eq(frame)].set_index("Agent_ID").reindex(ids)
        positions.append(rows[["X", "Y"]].to_numpy(dtype=float))
    positions = np.asarray(positions)
    dog_positions = dogs.reindex(frames)[["X", "Y"]].to_numpy(dtype=float)
    if not np.isfinite(positions).all() or not np.isfinite(dog_positions).all():
        raise ValueError("Blind clip must have complete finite observations")
    index = ids.index(focus_id)
    velocity = np.diff(positions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    current, previous = positions[2:], positions[1:-1]
    velocity = velocity[1:]
    center = current.mean(axis=1)
    spread = np.linalg.norm(current - center[:, None, :], axis=2)
    previous_spread = np.linalg.norm(previous - previous.mean(axis=1)[:, None, :], axis=2)
    exposure = previous - dog_positions[2:, None, :]
    distance = np.linalg.norm(exposure, axis=2)
    directions = exposure / np.maximum(distance[..., None], 1e-9)
    radial_acceleration = np.sum(acceleration * directions, axis=2)
    radial_speed = np.sum(velocity * directions, axis=2)
    outward = np.sum(velocity * (current - center[:, None, :])
                     / np.maximum(spread[..., None], 1e-9), axis=2)
    speed = np.linalg.norm(velocity, axis=2)
    empirical_ceiling = float(np.quantile(np.delete(speed, index, axis=1), .95))
    focus_response, peer_response = [], []
    samples = []
    for t in range(61):
        comparable = ((np.abs(distance[t] - distance[t, index]) <= 5.)
                      & (np.abs(previous_spread[t] - previous_spread[t, index]) <= 10.))
        comparable[index] = False
        peer_acceleration = float(np.median(radial_acceleration[t, comparable])) if comparable.sum() >= 3 else None
        if distance[t, index] <= 25. and peer_acceleration is not None and peer_acceleration >= .02:
            focus_response.append(float(radial_acceleration[t, index]))
            peer_response.append(peer_acceleration)
        if t % 5 == 0:
            neighbors = np.linalg.norm(current[t] - current[t, index], axis=1)
            neighbors[index] = np.inf
            samples.append({"relative_frame": t, "focus_position": current[t, index].tolist(),
                            "dog_position": dog_positions[t+2].tolist(), "flock_centroid": center[t].tolist(),
                            "focus_velocity_estimate": np.round(velocity[t, index], 3).tolist(),
                            "dog_exposure_distance": round(float(distance[t, index]), 3),
                            "focus_spread": round(float(spread[t, index]), 3),
                            "outward_speed": round(float(outward[t, index]), 3),
                            "dog_radial_acceleration_estimate": round(float(radial_acceleration[t, index]), 3),
                            "comparable_peer_count": int(comparable.sum()),
                            "comparable_peer_median_radial_acceleration": round(peer_acceleration, 3) if peer_acceleration is not None else None,
                            "nearest_neighbor_distance": round(float(neighbors.min()), 3)})
    response_ratio = float(np.median(focus_response) / np.median(peer_response)) if peer_response else None
    features = {"minimum_dog_exposure_distance": round(float(distance[:, index].min()), 3),
                "dog_exposure_distance_span": round(float(np.ptp(distance[:, index])), 3),
                "close_frames": int(np.sum(distance[:, index] <= 25.)),
                "comparable_response_samples": len(peer_response),
                "median_response_ratio_to_unlabeled_peers": round(response_ratio, 3) if response_ratio is not None else None,
                "spread_change": round(float(spread[-1, index] - spread[0, index]), 3),
                "median_outward_speed": round(float(np.median(outward[:, index])), 3),
                "fraction_near_observed_peer_speed_ceiling": round(float(np.mean(speed[:, index] >= .97 * empirical_ceiling)), 3)}
    return {"observation": "61 frames, rounded positions; sampled every5; finite-difference velocity and acceleration",
            "goal_position": [float(value) for value in goal_position], "samples": samples, "features": features}


def blind_numeric_diagnosis(payload):
    """Predeclared observational thresholds; no true coefficients or identities."""
    f = payload["features"]
    if (f["minimum_dog_exposure_distance"] > 25. and f["spread_change"] >= 5.
            and f["median_outward_speed"] >= .08):
        return "dispersing"
    ratio = f["median_response_ratio_to_unlabeled_peers"]
    if (f["comparable_response_samples"] < 6 or ratio is None
            or f["dog_exposure_distance_span"] < 5.
            or f["fraction_near_observed_peer_speed_ceiling"] >= .8):
        return "insufficient_evidence"
    if ratio <= .5:
        return "weak_dog_response"
    if ratio >= 1.5:
        return "enhanced_near_dog_response"
    if .75 <= ratio <= 1.25 and abs(f["spread_change"]) < 3.:
        return "normal_response"
    return "insufficient_evidence"


def mixed_observation_windows(data, first_frame=150):
    """Scan chronological, nonoverlapping windows using only logged geometry."""
    observed = data.loc[:, ["Frame", "Agent_Type", "Agent_ID", "X", "Y"]]
    sheep = observed.loc[observed.Agent_Type.eq("Sheep")]
    ids = sorted(sheep.Agent_ID.unique().tolist())
    if len(ids) < 2 or first_frame < 2:
        raise ValueError("At least two sheep and two preceding frames are required")
    frames = np.arange(int(observed.Frame.min()), int(observed.Frame.max()) + 1)
    grid = pd.MultiIndex.from_product([frames, ids], names=["Frame", "Agent_ID"])
    positions = sheep.set_index(["Frame", "Agent_ID"])[["X", "Y"]].reindex(grid).to_numpy().reshape(len(frames), len(ids), 2)
    dogs = observed.loc[observed.Agent_Type.eq("Dog")].set_index("Frame")[["X", "Y"]].reindex(frames).to_numpy()
    if not np.isfinite(positions).all() or not np.isfinite(dogs).all():
        raise ValueError("Complete finite sheep/dog observations are required")
    rows = []
    for start in range(max(first_frame, int(frames[0]) + 2), int(frames[-1]) - 59, 61):
        offset = start - int(frames[0])
        clip = positions[offset-2:offset+61]
        current, previous = clip[2:], clip[1:-1]
        velocity = np.diff(clip, axis=0)[1:]
        acceleration = np.diff(clip, n=2, axis=0)
        spread = np.linalg.norm(current - current.mean(axis=1)[:, None, :], axis=2)
        previous_spread = np.linalg.norm(previous - previous.mean(axis=1)[:, None, :], axis=2)
        exposure = previous - dogs[offset:offset+61, None, :]
        distance = np.linalg.norm(exposure, axis=2)
        directions = exposure / np.maximum(distance[..., None], 1e-9)
        radial_acceleration = np.sum(acceleration * directions, axis=2)
        outward = np.sum(velocity * (current - current.mean(axis=1)[:, None, :])
                         / np.maximum(spread[..., None], 1e-9), axis=2)
        speed = np.linalg.norm(velocity, axis=2)
        comparable = ((np.abs(distance[:, :, None] - distance[:, None, :]) <= 5.)
                      & (np.abs(previous_spread[:, :, None] - previous_spread[:, None, :]) <= 10.))
        comparable[:, np.arange(len(ids)), np.arange(len(ids))] = False
        enough_peers = comparable.sum(axis=2) >= 3
        peer_values = np.where(comparable, radial_acceleration[:, None, :], np.nan)
        peer_acceleration = np.full(distance.shape, np.nan)
        peer_acceleration[enough_peers] = np.nanmedian(peer_values[enough_peers], axis=1)
        for index, agent_id in enumerate(ids):
            ceiling = float(np.quantile(np.delete(speed, index, axis=1), .95))
            eligible = (distance[:, index] <= 25.) & (peer_acceleration[:, index] >= .02)
            ratio = (float(np.median(radial_acceleration[eligible, index])
                           / np.median(peer_acceleration[eligible, index])) if eligible.any() else None)
            features = {
                "minimum_dog_exposure_distance": round(float(distance[:, index].min()), 3),
                "dog_exposure_distance_span": round(float(np.ptp(distance[:, index])), 3),
                "close_frames": int(np.sum(distance[:, index] <= 25.)),
                "comparable_response_samples": int(eligible.sum()),
                "median_response_ratio_to_unlabeled_peers": round(ratio, 3) if ratio is not None else None,
                "spread_change": round(float(spread[-1, index] - spread[0, index]), 3),
                "median_outward_speed": round(float(np.median(outward[:, index])), 3),
                "fraction_near_observed_peer_speed_ceiling": round(float(np.mean(speed[:, index] >= .97 * ceiling)), 3),
            }
            rows.append({"focus_id": int(agent_id), "start_frame": start, "end_frame": start + 60,
                         **features, "diagnosis": blind_numeric_diagnosis({"features": features}),
                         "moderate_unsaturated_frames": int(np.sum(
                             (distance[:, index] >= 20.) & (distance[:, index] <= 30.)
                             & (speed[:, index] < .97 * ceiling)))})
    return rows


def write_mixed_observability_trial(output_root=None, source_root=None):
    """Offline evidence coverage across all sheep; private types only grade results."""
    source_root = Path(source_root or config.PROJECT_ROOT / "logs/anomaly_behavior/20261004_015855_871537")
    source_protocol = json.loads((source_root / "population_api_protocol.json").read_text(encoding="utf-8"))
    sources = pd.read_csv(source_root / "ratio_30/rule/summary.csv").sort_values("Seed")
    if sources.Seed.duplicated().any() or sources.empty:
        raise ValueError("Source seeds must be unique and nonempty")
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "api_calls": 0, "source_root": str(source_root), "seeds": sources.Seed.astype(int).tolist(),
        "selection": "All sheep, chronological nonoverlapping61-frame windows starting150; no incomplete trailing window.",
        "prediction": "Existing blind_numeric_diagnosis thresholds unchanged. Keep first definite diagnosis per sheep, or abstain if none. No hindsight best-window selection.",
        "inputs": ["Frame", "Agent_Type", "Agent_ID", "X", "Y"],
        "private_grading": "Status is checked for constancy and joined ONLY after geometry-based windows/predictions are computed.",
        "interpretation": "20 existing seed clusters, not new simulations or independent sheep trials; Normal prevalence70%; identity agreement is not repair effectiveness or LLM advantage.",
        "extra_measure": "Count20-30-distance samples below97% of observed peer95th-percentile speed, without changing prediction thresholds.",
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {str(int(row.Seed)): hashlib.sha256(Path(row.Data).read_bytes()).hexdigest()
                          for row in sources.itertuples()},
    }
    (output / "mixed_observability_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    mapping = dict(zip(("Normal", "A", "B", "C"), BLIND_DIAGNOSIS_CHOICES[:4]))
    all_windows, individuals = [], []
    for source in sources.itertuples():
        data = pd.read_parquet(source.Data)
        windows = mixed_observation_windows(data)
        cfg = source_protocol["base_config"]
        goal = np.random.RandomState(int(source.Seed)).uniform(cfg["GOAL_MARGIN"], cfg["SPACE_SIZE"] - cfg["GOAL_MARGIN"], 2)
        centers = data.loc[data.Agent_Type.eq("Sheep")].groupby("Frame")[["X", "Y"]].mean()
        measured = data.groupby("Frame").dist_to_goal.first().reindex(centers.index).to_numpy()
        if not np.isfinite(measured).all() or np.max(np.abs(np.linalg.norm(centers.to_numpy() - goal, axis=1) - measured)) > .03:
            raise ValueError("Source goal reconstruction does not match logged geometry")
        sheep = data.loc[data.Agent_Type.eq("Sheep")]
        if (sheep.groupby("Agent_ID").Status.nunique() != 1).any():
            raise ValueError("Private type must stay constant for identity grading")
        private = sheep.groupby("Agent_ID").Status.first()
        if not private.isin(mapping).all():
            raise ValueError("Unexpected private status")
        for agent_id, status in private.items():
            samples = [row for row in windows if row["focus_id"] == agent_id]
            definite = [row for row in samples if row["diagnosis"] != "insufficient_evidence"]
            first = definite[0] if definite else None
            diagnosis = first["diagnosis"] if first else "insufficient_evidence"
            individuals.append({"seed": int(source.Seed), "focus_id": int(agent_id), "true_status": status,
                                "windows": len(samples), "diagnosis": diagnosis,
                                "first_decision_frame": first["end_frame"] if first else None,
                                "correct": diagnosis == mapping[status],
                                "ever_moderate_unsaturated_6": any(row["moderate_unsaturated_frames"] >= 6 for row in samples),
                                "ever_comparable_response_6": any(row["comparable_response_samples"] >= 6 for row in samples),
                                "later_diagnosis_disagrees": any(row["diagnosis"] != diagnosis for row in definite)})
        all_windows.extend({"seed": int(source.Seed), **row} for row in windows)
        print(f"seed={source.Seed}: {len(windows)} geometry-only windows checked", flush=True)
    summary = {}
    for status in ("all", "Normal", "A", "B", "C"):
        rows = [row for row in individuals if status == "all" or row["true_status"] == status]
        definite = [row for row in rows if row["diagnosis"] != "insufficient_evidence"]
        summary[status] = {
            "individuals": len(rows), "identified": len(definite), "correct": sum(row["correct"] for row in rows),
            "coverage": len(definite) / len(rows) if rows else None,
            "selective_identity_accuracy": sum(row["correct"] for row in definite) / len(definite) if definite else None,
            "moderate_unsaturated_opportunity": sum(row["ever_moderate_unsaturated_6"] for row in rows),
            "comparable_response_opportunity": sum(row["ever_comparable_response_6"] for row in rows),
            "later_disagreement": sum(row["later_diagnosis_disagrees"] for row in rows),
            "confusion": dict(Counter(row["diagnosis"] for row in rows)),
        }
    report = {"api_calls": 0, "seed_count": len(sources), "windows": len(all_windows), "summary": summary,
              "always_normal_identity_hits": sum(row["true_status"] == "Normal" for row in individuals),
              "identity_precision_by_prediction": {
                  diagnosis: {
                      "answers": sum(row["diagnosis"] == diagnosis for row in individuals),
                      "hits": sum(row["diagnosis"] == diagnosis and row["correct"] for row in individuals),
                  } for diagnosis in BLIND_DIAGNOSIS_CHOICES[:4]},
              "source_hashes_unchanged": all(hashlib.sha256(Path(row.Data).read_bytes()).hexdigest()
                                            == protocol["source_sha256"][str(int(row.Seed))] for row in sources.itertuples()),
              "interpretation": protocol["interpretation"]}
    for values in report["identity_precision_by_prediction"].values():
        values["precision"] = values["hits"] / values["answers"] if values["answers"] else None
    _write_csv(output / "mixed_observation_windows.csv", all_windows)
    _write_csv(output / "mixed_observability_private_scores.csv", individuals)
    (output / "mixed_observability_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return output, report


def prepare_blind_diagnosis_cases(source_root):
    """Stratified private identities, label-independent closest-approach windows."""
    source_root = Path(source_root)
    protocol = json.loads((source_root / "population_api_protocol.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(source_root / "ratio_30/rule/summary.csv").set_index("Seed")
    rng = np.random.default_rng(9101)
    mapping = dict(zip(("Normal", "A", "B", "C"), BLIND_DIAGNOSIS_CHOICES[:4]))
    cases = []
    for seed in range(4701, 4706):
        source_path = Path(summary.loc[seed, "Data"])
        data = pd.read_parquet(source_path)
        initial = data.loc[data.Frame.eq(data.Frame.min()) & data.Agent_Type.eq("Sheep")]
        dog = data.loc[data.Agent_Type.eq("Dog")].set_index("Frame")
        centers = data.loc[data.Agent_Type.eq("Sheep")].groupby("Frame")[["X", "Y"]].mean()
        cfg = protocol["base_config"]
        goal = np.random.RandomState(seed).uniform(cfg["GOAL_MARGIN"], cfg["SPACE_SIZE"] - cfg["GOAL_MARGIN"], 2)
        measured = data.groupby("Frame").dist_to_goal.first().reindex(centers.index).to_numpy()
        if not np.isfinite(measured).all() or np.max(np.abs(np.linalg.norm(centers.to_numpy() - goal, axis=1) - measured)) > .03:
            raise ValueError("Source goal reconstruction does not match logged geometry")
        for status, truth in mapping.items():
            focus_id = int(rng.choice(sorted(initial.loc[initial.Status.eq(status), "Agent_ID"].tolist())))
            trajectory = data.loc[data.Agent_Type.eq("Sheep") & data.Agent_ID.eq(focus_id)].set_index("Frame")
            available = trajectory.index[(trajectory.index >= 150) & (trajectory.index <= trajectory.index.max() - 30)]
            distances = np.linalg.norm(trajectory.loc[available, ["X", "Y"]].to_numpy()
                                       - dog.loc[available, ["X", "Y"]].to_numpy(), axis=1)
            midpoint = int(available[int(np.argmin(distances))])
            start = midpoint - 30
            if trajectory.loc[start:start+60, "Status"].nunique() != 1:
                raise ValueError("This blind benchmark requires a constant private identity per clip")
            payload = blind_motion_payload(data, focus_id, start, goal)
            cases.append({"seed": seed, "focus_id": focus_id, "start_frame": start,
                          "true_status": status, "truth": truth, "source": str(source_path),
                          "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "payload": payload})
    rng.shuffle(cases)
    for index, case in enumerate(cases):
        case["clip_id"] = f"clip_{index+1:02d}"
    return cases


def write_blind_diagnosis_trial(output_root=None, source_root=None):
    """Twenty offline blind diagnoses, not a new control or repair experiment."""
    from responses_api_backend import DeepSeekBackend

    source_root = Path(source_root or config.PROJECT_ROOT / "logs/anomaly_behavior/20261004_015855_871537")
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"source_root": str(source_root), "seeds": list(range(4701, 4706)),
                "clips": 20, "private_strata": ["Normal", "A", "B", "C"], "selection_seed": 9101,
                "selection": "One random ID per private stratum/seed, closest observed dog approach after frame150 and before last30, centered61-frame window",
                "prompt": BLIND_DIAGNOSIS_PROMPT, "choices": BLIND_DIAGNOSIS_CHOICES,
                "numeric_rule": "Dispersing: dog farther25, spread gain>=5, median outward>=.08; otherwise need6 comparable response samples, distance span>=5 and saturation<.8; ratio<=.5 weak, >=1.5 enhanced, .75..1.25 and abs spread<3 normal; otherwise abstain",
                "interpretation": "Balanced, clustered20 clips from5 prior seeds, exploratory; latent identity is grading reference, not proof of observability or causal behavior in this clip; no repair success claim",
                "scoring": "Unconditional latent-identity accuracy counts abstention as nonidentification; also report coverage and accuracy among nonabstained clips",
                "source_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                                  for name in ("experiments/anomaly_behavior/behavior_characterization.py", "responses_api_backend.py")}}
    (output / "blind_diagnosis_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    cases = prepare_blind_diagnosis_cases(source_root)
    private = [{key: value for key, value in case.items() if key != "payload"} for case in cases]
    payloads = [{"clip_id": case["clip_id"], **case["payload"]} for case in cases]
    (output / "blind_diagnosis_truth_private.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
    (output / "blind_diagnosis_payloads.json").write_text(json.dumps(payloads, indent=2, allow_nan=False), encoding="utf-8")
    settings = SimpleNamespace(**{key: value for key, value in vars(config).items() if key.isupper()})
    settings.LLM_CACHE_ENABLED = False
    backend = DeepSeekBackend(settings=settings)
    protocol.update(model=backend.model, base_url=backend.base_url, reasoning_effort=backend.reasoning_effort)
    (output / "blind_diagnosis_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    schema = {"type": "json_schema", "name": "blind_sheep_diagnosis", "strict": True,
              "schema": {"type": "object", "properties": {
                  "diagnosis": {"type": "string", "enum": list(BLIND_DIAGNOSIS_CHOICES)},
                  "evidence_frames": {"type": "array", "maxItems": 3,
                                      "items": {"type": "integer", "enum": list(range(0, 61, 5))}},
                  "reason": {"type": "string", "maxLength": 240}},
                  "required": ["diagnosis", "evidence_frames", "reason"], "additionalProperties": False}}
    rows = []
    try:
        for case, payload in zip(cases, payloads):
            started = time.perf_counter()
            result, error, response_text = None, None, None
            try:
                options = {"model": backend.model, "instructions": BLIND_DIAGNOSIS_INSTRUCTIONS,
                           "input": json.dumps(payload, sort_keys=True, allow_nan=False),
                           "text": {"format": schema}, "max_output_tokens": 400, "store": False}
                if backend.reasoning_effort:
                    options["reasoning"] = {"effort": backend.reasoning_effort}
                response = backend.client.responses.create(**options)
                response_text = backend._extract_output_text(response)
                result = json.loads(response_text)
                if (not isinstance(result, dict) or set(result) != {"diagnosis", "evidence_frames", "reason"}
                        or result["diagnosis"] not in BLIND_DIAGNOSIS_CHOICES
                        or not isinstance(result["reason"], str) or len(result["reason"]) > 240
                        or not isinstance(result["evidence_frames"], list) or len(result["evidence_frames"]) > 3
                        or any(not isinstance(frame, int) or isinstance(frame, bool) or frame not in range(0, 61, 5)
                               for frame in result["evidence_frames"])):
                    raise ValueError("Invalid blind diagnosis schema")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            row = {"clip_id": case["clip_id"], "truth": case["truth"], "seed": case["seed"],
                   "numeric_diagnosis": blind_numeric_diagnosis(payload),
                   "llm_diagnosis": result["diagnosis"] if error is None else "invalid_response",
                   "latency_seconds": time.perf_counter() - started, "error": error}
            rows.append(row)
            (output / f"response_{case['clip_id']}.json").write_text(
                json.dumps({"result": result, "raw_text": response_text, "error": error}, indent=2), encoding="utf-8")
            _write_csv(output / "blind_diagnosis_runs.csv", rows)
            print(f"{case['clip_id']}: numeric={row['numeric_diagnosis']} llm={row['llm_diagnosis']} error={error is not None}", flush=True)
            if error and any(code in error for code in ("Error code: 401", "Error code: 402", "Error code: 403", "Error code: 404")):
                raise RuntimeError("API authorization/billing/model error; partial blind results retained")
    finally:
        backend.close()
    summary = {}
    for arm in ("numeric", "llm"):
        answered = [row for row in rows if row[f"{arm}_diagnosis"] not in ("insufficient_evidence", "invalid_response")]
        correct = sum(row[f"{arm}_diagnosis"] == row["truth"] for row in answered)
        summary[arm] = {"clips": len(rows), "identified_correctly": correct,
                        "unconditional_identity_accuracy": correct / len(rows), "coverage": len(answered) / len(rows),
                        "answered": len(answered), "selective_accuracy": correct / len(answered) if answered else None,
                        "abstained": sum(row[f"{arm}_diagnosis"] == "insufficient_evidence" for row in rows),
                        "invalid": sum(row[f"{arm}_diagnosis"] == "invalid_response" for row in rows),
                        "confusion": {truth: dict(Counter(row[f"{arm}_diagnosis"] for row in rows if row["truth"] == truth))
                                      for truth in BLIND_DIAGNOSIS_CHOICES[:4]}}
    report = {**protocol, "summary": summary,
              "observable_support": {"clips_with_six_comparable_response_samples": sum(payload["features"]["comparable_response_samples"] >= 6 for payload in payloads),
                                     "clips_near_speed_ceiling_at_least80percent": sum(payload["features"]["fraction_near_observed_peer_speed_ceiling"] >= .8 for payload in payloads)}}
    (output / "blind_diagnosis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report


def population_snapshot(env):
    distances = np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1)
    inside = distances < config.GOAL_RADIUS
    spread = np.linalg.norm(env.sheep_pos - env.sheep_pos.mean(axis=0), axis=1)
    outside = np.any((env.sheep_pos < 0) | (env.sheep_pos > config.SPACE_SIZE), axis=1)
    return {
        "frame": env.current_frame, "inside_goal": int(inside.sum()),
        "remaining": int((~inside).sum()),
        "mean_excess_goal_distance": float(np.maximum(distances - config.GOAL_RADIUS, 0).mean()),
        "centroid_distance": float(np.linalg.norm(env.sheep_pos.mean(axis=0) - env.goal_pos)),
        "max_spread": float(spread.max()), "mean_spread": float(spread.mean()),
        "outside_count": int(outside.sum()),
        **{f"remaining_{kind}": int(np.sum(~inside & (env.sheep_status == kind)))
           for kind in ("Normal", "A", "B", "C")},
    }


def population_scale_run(seed, count, scenario, primary_cap, extended_cap, *, anomaly_counts=None,
                         controller=None, pending_poll_seconds=0):
    # Largest-remainder allocation: A/B/C/Normal have approximately equal shares.
    groups = [count // 4 + int(index < count % 4) for index in range(4)]
    a, b, c = groups[:3] if scenario == "mixed" else (0, 0, 0)
    if anomaly_counts is not None:
        if (len(anomaly_counts) != 3 or any(not isinstance(value, int) or value < 0
                                           for value in anomaly_counts)
                or sum(anomaly_counts) > count):
            raise ValueError("Supply three nonnegative anomaly counts within flock size")
        a, b, c = anomaly_counts
    with temporary_config(NUM_SHEEP=count, NUM_ABNORMAL_A=a, NUM_ABNORMAL_B=b,
                          NUM_ABNORMAL_C=c, USE_FIXED_SEED=True, RANDOM_SEED=seed,
                          CONTROLLER_KIND="rule", REPAIR_ENABLED=True, MAX_STEPS=extended_cap):
        env, detector = SwarmEnv(controller=controller if controller is not None else LowLevelController()), make_detector()
        initial_hash = hashlib.sha256(env.sheep_pos.tobytes() + env.dog_pos.tobytes()
                                      + env.goal_pos.tobytes()).hexdigest()
        common = {"seed": seed, "num_sheep": count, "scenario": scenario}
        initial = population_snapshot(env)
        trace = [{**common, **initial, "alarm": "initial", "mode": env.current_mode}]
        tail = deque([initial], maxlen=config.DANGER_STAGNATION_FRAMES + 1)
        checkpoint = None
        success = False
        alarm_counts = {"danger": 0, "splitting": 0, "stagnation": 0, "dual": 0}
        mode_counts = {mode: 0 for mode in ("TARGET_TRACKING", "HARD_TO_GUIDE", "COHESION_PRIORITY")}
        goal_exits, outside_frames, peak_spread = 0, 0, initial["max_spread"]
        while env.current_frame < extended_cap:
            alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
            before = np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1) < config.GOAL_RADIUS
            env.record_frame_metrics(metrics)
            success = bool(env.step(alarms))
            if not success and getattr(env.controller, "pending_request", None) is not None:
                time.sleep(pending_poll_seconds)
            sample = population_snapshot(env)
            if not success:
                mode_counts[env.current_mode] += 1
                after = np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1) < config.GOAL_RADIUS
                goal_exits += int(np.sum(before & ~after))
                alarm_counts["danger"] += bool(alarms)
                alarm_counts["splitting"] += "flock_splitting" in alarms
                alarm_counts["stagnation"] += "stagnation" in alarms
                alarm_counts["dual"] += {"flock_splitting", "stagnation"}.issubset(alarms)
                outside_frames += sample["outside_count"] > 0
                peak_spread = max(peak_spread, sample["max_spread"])
            tail.append(sample)
            if env.current_frame == primary_cap or (success and env.current_frame <= primary_cap):
                checkpoint = {**sample, "success": success, "danger_frames": alarm_counts["danger"],
                              "goal_exits": goal_exits, "outside_frames": outside_frames,
                              "peak_spread": peak_spread, "mode_counts": mode_counts.copy()}
            if env.current_frame % 10 == 0 or env.current_frame == primary_cap or success or env.current_frame == extended_cap:
                trace.append({**common, **sample, "alarm": ";".join(alarms) or "none", "mode": env.current_mode})
            if success:
                break
        assert checkpoint is not None
        row = {
            **common, "A": a, "B": b, "C": c, "Normal": count - a - b - c,
            "initial_sha256": initial_hash, "goal_x": float(env.goal_pos[0]),
            "initial_status_sha256": hashlib.sha256(json.dumps(env.sheep_status.tolist()).encode()).hexdigest(),
            "goal_y": float(env.goal_pos[1]), "dog_start_x": float(env.dog_start_pos[0]),
            "dog_start_y": float(env.dog_start_pos[1]),
            "start_center_x": float(env.sheep_start_center[0]), "start_center_y": float(env.sheep_start_center[1]),
            "initial_max_spread": initial["max_spread"], "initial_mean_spread": initial["mean_spread"],
            "success_primary": checkpoint["success"], "capped_frames_primary": checkpoint["frame"],
            "inside_primary": checkpoint["inside_goal"], "remaining_primary": checkpoint["remaining"],
            "mean_excess_primary": checkpoint["mean_excess_goal_distance"],
            "danger_frames_primary": checkpoint["danger_frames"], "goal_exits_primary": checkpoint["goal_exits"],
            "outside_frames_primary": checkpoint["outside_frames"], "max_spread_primary": checkpoint["peak_spread"],
            **{f"{mode.lower()}_frames_primary": value for mode, value in checkpoint["mode_counts"].items()},
            **{f"remaining_primary_{kind}": checkpoint[f"remaining_{kind}"] for kind in ("Normal", "A", "B", "C")},
            "success_extended": success, "capped_frames_extended": env.current_frame,
            "inside_final": sample["inside_goal"], "remaining_final": sample["remaining"],
            "mean_excess_final": sample["mean_excess_goal_distance"],
            "tail_span_frames": sample["frame"] - tail[0]["frame"],
            "tail_mean_excess_reduction": tail[0]["mean_excess_goal_distance"] - sample["mean_excess_goal_distance"],
            "tail_inside_change": sample["inside_goal"] - tail[0]["inside_goal"],
            **{f"{name}_frames_extended": value for name, value in alarm_counts.items()},
        }
        print(f"seed={seed} N={count} {scenario}: "
              f"primary={checkpoint['success']} inside={checkpoint['inside_goal']}/{count}; "
              f"extended={success} frames={env.current_frame} inside={sample['inside_goal']}/{count}", flush=True)
        return row, trace


def write_population_scale_trial(output_root=None, first_seed=4601, seed_count=20,
                                 sheep_counts=(20, 50), primary_cap=2000, extended_cap=6000):
    if (first_seed < 0 or seed_count < 1 or not sheep_counts or min(sheep_counts) < 4
            or len(set(sheep_counts)) != len(sheep_counts)
            or primary_cap < 1 or extended_cap < primary_cap):
        raise ValueError("Use nonnegative seeds, positive counts, unique flock sizes >=4, and ordered positive caps")
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "first_seed": first_seed, "seed_count": seed_count, "sheep_counts": sheep_counts,
        "primary_cap": primary_cap, "extended_cap": extended_cap,
        "controller": "rule", "repair_enabled": True, "api_calls": 0,
        "allocation": "Largest remainder for equal A/B/C/Normal shares; normal arm has no anomalies",
        "base_config": {key: value for key, value in vars(config).items()
                        if key.isupper() and isinstance(value, (int, float, bool))},
        "source_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                          for name in ("config.py", "dynamics.py", "environment.py", "detector.py",
                                       "low_level_controller.py", "experiments/anomaly_behavior/behavior_characterization.py")},
        "interpretation": "Fixed environment, variable flock size (density also changes); exploratory rule-only scale test",
        "success_semantics": "Existing env success check before physics; caps count step calls",
    }
    (output / "population_scale_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, traces = [], []
    for seed in range(first_seed, first_seed + seed_count):
        anchors = []
        for count in sheep_counts:
            paired = []
            for scenario in ("normal", "mixed"):
                row, trace = population_scale_run(seed, count, scenario, primary_cap, extended_cap)
                rows.append(row)
                traces.extend(trace)
                paired.append(row["initial_sha256"])
                anchors.append(tuple(row[key] for key in ("goal_x", "goal_y", "dog_start_x", "dog_start_y",
                                                          "start_center_x", "start_center_y")))
            assert paired[0] == paired[1], "Normal/mixed initial geometry differs"
        assert len(set(anchors)) == 1, "Cross-size goal, dog, or sampling center differs"
    summary = {}
    for count in sheep_counts:
        for scenario in ("normal", "mixed"):
            group = [row for row in rows if row["num_sheep"] == count and row["scenario"] == scenario]
            summary[f"{count}_{scenario}"] = {
                "runs": len(group), "successes_primary": sum(row["success_primary"] for row in group),
                "mean_capped_frames_primary": statistics.mean(row["capped_frames_primary"] for row in group),
                "mean_inside_primary": statistics.mean(row["inside_primary"] for row in group),
                "mean_inside_fraction_primary": statistics.mean(row["inside_primary"] / count for row in group),
                "mean_remaining_primary": statistics.mean(row["remaining_primary"] for row in group),
                "mean_danger_frames_primary": statistics.mean(row["danger_frames_primary"] for row in group),
                "mean_max_spread_primary": statistics.mean(row["max_spread_primary"] for row in group),
                "mean_goal_exits_primary": statistics.mean(row["goal_exits_primary"] for row in group),
                "successes_extended": sum(row["success_extended"] for row in group),
                "mean_capped_frames_extended": statistics.mean(row["capped_frames_extended"] for row in group),
                "mean_remaining_final": statistics.mean(row["remaining_final"] for row in group),
                "failed_seeds_primary": [row["seed"] for row in group if not row["success_primary"]],
                "failed_seeds_extended": [row["seed"] for row in group if not row["success_extended"]],
            }
    _write_csv(output / "population_scale_runs.csv", rows)
    _write_csv(output / "population_scale_states.csv", traces)
    report = {**protocol, "summary": summary}
    (output / "population_scale_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report


REPAIR_DISTANCE_VARIANTS = {
    "baseline": (3.0, 10.0),
    "guide_near": (1.5, 10.0),
    "guide_far": (6.0, 10.0),
    "cohesion_near": (3.0, 5.0),
    "cohesion_far": (3.0, 20.0),
}


class RepairFeedbackTrialController(HighLevelController):
    """Experimental review of observed repair outcomes; not a default policy."""

    def __init__(self, backend, settings=config, review_frames=60, request_budget=6):
        super().__init__(backend=backend, settings=settings)
        self.review_frames = review_frames
        self.request_budget = request_budget
        self.previous_sample = None
        self.previous_decision = None
        self.repair_window = None
        self.repair_history = deque(maxlen=3)
        self.review_due = False
        self.events = []
        self.responses_seen = set()
        self.override_frames = 0

    def decide(self, observation):
        previous_request = self.pending_request
        decision = super().decide(observation)
        self.previous_decision = decision
        metadata = decision.metadata
        request_id = metadata.get("request_id")
        if metadata.get("high_level_status") in ("active", "rejected") and request_id not in self.responses_seen:
            self.responses_seen.add(request_id)
            self.events.append({"event": "response", "frame": observation.frame, **dict(metadata)})
        if (previous_request is not None and previous_request.request_id not in self.responses_seen
                and (self.pending_request is None or self.pending_request.request_id != previous_request.request_id)):
            self.events.append({"event": "cancel", "frame": observation.frame,
                                "request_id": previous_request.request_id, "reason": "trigger_changed_or_cleared"})
        if decision.mode != rule_mode_for_alarms(observation.alarms, self.settings.REPAIR_ENABLED):
            self.override_frames += 1
        return decision

    def _observe_state(self, observation, alarms):
        summary = super()._observe_state(observation, alarms)
        distances = np.linalg.norm(observation.sheep_positions - observation.goal_position, axis=1)
        center = observation.sheep_positions.mean(axis=0)
        sample = {"frame": observation.frame, "centroid_distance": float(np.linalg.norm(center - observation.goal_position)),
                  "max_spread": float(np.linalg.norm(observation.sheep_positions - center, axis=1).max()),
                  "inside_goal": summary["inside_goal"],
                  "inside_mask": distances < self.settings.GOAL_RADIUS}
        if self.previous_sample is not None and self.previous_decision is not None:
            mode = self.previous_decision.mode
            if self.repair_window is not None and self.repair_window["mode"] != mode:
                self._finish_window(self.previous_sample)
                self.review_due = False
            if self.repair_window is None:
                self.repair_window = {"mode": mode, "start": self.previous_sample,
                                      "targets": set(), "goal_exits": 0}
            target = self.previous_decision.target_sheep_id
            if target is not None:
                self.repair_window["targets"].add(int(target))
            self.repair_window["goal_exits"] += int(np.sum(self.previous_sample["inside_mask"] & ~sample["inside_mask"]))
            if sample["frame"] - self.repair_window["start"]["frame"] >= self.review_frames:
                self._finish_window(sample)
        self.previous_sample = sample
        if not alarms:
            self.review_due = False
        summary["repair_history"] = list(self.repair_history)
        summary["repair_review"] = {"window_frames": self.review_frames,
                                    "request_budget": self.request_budget,
                                    "requests_used": self.request_counter,
                                    "trigger": "Completed repair window without progress; not necessarily concurrent alarms",
                                    "progress_definition": "inside gain >0 OR guide centroid progress >=1 OR cohesion max-spread reduction >=1"}
        return summary

    def _finish_window(self, sample):
        window, self.repair_window = self.repair_window, None
        start, mode = window["start"], window["mode"]
        elapsed = sample["frame"] - start["frame"]
        if elapsed <= 0 or mode == "TARGET_TRACKING":
            return
        record = {"mode": mode, "start_frame": start["frame"], "end_frame": sample["frame"],
                  "elapsed_frames": elapsed, "target_ids": sorted(window["targets"]),
                  "centroid_progress": round(start["centroid_distance"] - sample["centroid_distance"], 3),
                  "max_spread_reduction": round(start["max_spread"] - sample["max_spread"], 3),
                  "inside_goal_change": sample["inside_goal"] - start["inside_goal"],
                  "goal_exits": window["goal_exits"]}
        progress = (record["inside_goal_change"] > 0
                    or (mode == "HARD_TO_GUIDE" and start["centroid_distance"] - sample["centroid_distance"] >= 1.)
                    or (mode == "COHESION_PRIORITY" and start["max_spread"] - sample["max_spread"] >= 1.))
        record["review_eligible"] = elapsed >= self.review_frames and not progress
        self.repair_history.append(record)
        self.review_due = record["review_eligible"]

    def _request_ready(self, observation, state_summary):
        return self.review_due and self.request_counter < self.request_budget

    def _make_request(self, observation, alarms, state_summary):
        request = super()._make_request(observation, alarms, state_summary)
        self.review_due = False
        if self.backend.source == "feedback_rule":
            last_mode = state_summary["repair_history"][-1]["mode"]
            alternatives = {"HARD_TO_GUIDE": ("flock_splitting", "COHESION_PRIORITY"),
                            "COHESION_PRIORITY": ("stagnation", "HARD_TO_GUIDE")}
            alarm, mode = alternatives[last_mode]
            choice = mode if alarm in alarms else state_summary["rule_mode"]
            self.backend.mode_mapping = {request.primary_alarm: choice}
        self.events.append({"event": "request", "frame": observation.frame,
                            "request_id": request.request_id, "state_summary": request.state_summary})
        return request


def write_repair_feedback_trial(output_root=None, first_seed=5001, seed_count=20, max_steps=2000):
    """Compare unchanged rule, delayed feedback rule and real feedback LLM."""
    if first_seed < 0 or seed_count < 1 or first_seed + seed_count > 2**32 or max_steps < 1:
        raise ValueError("Use valid seeds, positive run count and frame budget")
    from responses_api_backend import DeepSeekBackend

    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    arms = ("rule", "feedback_rule", "feedback_llm")
    with temporary_config(NUM_SHEEP=50, NUM_ABNORMAL_A=5, NUM_ABNORMAL_B=5,
                          NUM_ABNORMAL_C=5, SPACE_SIZE=200., SHEEP_START_RADIUS=15.,
                          GOAL_RADIUS=20., HARD_TO_GUIDE_DISTANCE=3., COHESION_REPAIR_DISTANCE=10.,
                          REPAIR_ENABLED=True):
        settings = SimpleNamespace(**{key: value for key, value in vars(config).items() if key.isupper()})
        settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = True
        settings.HIGH_LEVEL_MIN_HOLD_FRAMES = settings.HIGH_LEVEL_MAX_HOLD_FRAMES = 60
        settings.LLM_CACHE_ENABLED = False
        protocol = {"seeds": list(range(first_seed, first_seed + seed_count)), "arms": arms,
                    "num_sheep": 50, "anomaly_counts": [5, 5, 5], "max_steps": max_steps,
                    "review_frames": 60, "max_requests_per_run": 6, "hold_frames": 60,
                    "rule_feedback_latency_frames": 20, "pending_poll_seconds": .05,
                    "progress": "inside gain >0 OR guide centroid progress >=1 OR cohesion max-spread reduction >=1",
                    "feedback_rule": "If eligible failed repair has another alarm-admissible repair, switch; otherwise use rule_mode",
                    "target_selection": "Always low-level rule, for all arms",
                    "guard": "Unchanged alarm/schema/staleness/forecast guard shared by both feedback arms",
                    "interpretation": "Whole-policy exploratory comparison; API delay is real, rule delay fixed20, not a latency-isolated reasoning ablation",
                    "success_semantics": "Existing pre-physics success check; full-precision post-step inside counts",
                    "settings": {key: value for key, value in vars(settings).items()
                                 if isinstance(value, (str, int, float, bool))},
                    "source_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                                      for name in ("config.py", "dynamics.py", "environment.py", "detector.py",
                                                   "low_level_controller.py", "high_level_controller.py", "mode_forecast.py",
                                                   "responses_api_backend.py", "experiments/anomaly_behavior/behavior_characterization.py")}}
        (output / "repair_feedback_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        rows, traces = [], []
        for seed in protocol["seeds"]:
            fingerprints = []
            for arm in arms:
                controller = None
                if arm != "rule":
                    if arm == "feedback_rule":
                        backend = MockLLMBackend(latency_frames=20, hold_frames=60)
                        backend.source = "feedback_rule"
                    else:
                        backend = DeepSeekBackend(settings=settings)
                        backend.prompt_version += "-repair-feedback-v1"
                        backend.instructions += (
                            "\n\nREPAIR-FEEDBACK TRIAL: Override earlier trigger/hold suggestions. Calls follow an actual "
                            "60-frame repair window without progress, and can have a single alarm. "
                            "Use repair_history to interpret the measured consequences of the actually executed modes. "
                            "Target IDs can change within a window; outcomes are observations, not causal proof or forecasts. "
                            "A failed progress threshold does not prove a strategy cannot succeed later. "
                            "Decide whether to continue or change one existing alarm-admissible mode. "
                            "Balance actual progress, spread and goal exits with current state and forecasts; "
                            "do not switch merely because a review is requested. Return null target_sheep_id "
                            "and exactly 60 hold_frames. No new modes, distances or physical changes."
                        )
                    backend.trigger_on_concurrent_only = False
                    controller = RepairFeedbackTrialController(backend=backend, settings=settings)
                try:
                    row, trace = population_scale_run(seed, 50, "mixed", max_steps, max_steps,
                                                      anomaly_counts=(5, 5, 5), controller=controller,
                                                      pending_poll_seconds=.05)
                    events = controller.events if controller is not None else []
                    if controller is not None and controller.pending_request is not None:
                        events.append({"event": "cancel", "frame": row["capped_frames_primary"],
                                       "request_id": controller.pending_request.request_id, "reason": "run_end"})
                    responses = [event for event in events if event["event"] == "response"]
                    errors = [event.get("backend_error") for event in responses if event.get("backend_error")]
                    row.update(arm=arm, requests=controller.request_counter if controller else 0,
                               responses=len(responses), accepted=sum(event["high_level_status"] == "active" for event in responses),
                               rejected=sum(event["high_level_status"] == "rejected" for event in responses),
                               backend_errors=len(errors), override_frames=controller.override_frames if controller else 0)
                    (output / f"events_{arm}_{seed}.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
                    rows.append(row)
                    traces.extend({**sample, "arm": arm} for sample in trace)
                    fingerprints.append((row["initial_sha256"], row["initial_status_sha256"]))
                    _write_csv(output / "repair_feedback_runs.csv", rows)
                    print(f"feedback arm={arm} seed={seed} requests={row['requests']} responses={row['responses']} "
                          f"overrides={row['override_frames']} errors={len(errors)}", flush=True)
                    if any(any(code in error for code in ("Error code: 401", "Error code: 402", "Error code: 403", "Error code: 404")) for error in errors):
                        raise RuntimeError("API authorization/billing/model error; retained partial logs and stopped")
                finally:
                    if controller is not None:
                        controller.close()
            if len(set(fingerprints)) != 1:
                raise AssertionError("Feedback arms must have identical initial geometry and identities")
        summary = {}
        baseline = {row["seed"]: row for row in rows if row["arm"] == "rule"}
        for arm in arms:
            group = [row for row in rows if row["arm"] == arm]
            summary[arm] = {"successes": sum(row["success_primary"] for row in group),
                            **{f"mean_{field}": statistics.mean(row[field] for row in group)
                               for field in ("capped_frames_primary", "inside_primary", "goal_exits_primary", "danger_frames_primary")},
                            **{field: sum(row[field] for row in group)
                               for field in ("requests", "responses", "accepted", "rejected", "backend_errors", "override_frames")},
                            "rescued": [row["seed"] for row in group if row["success_primary"] and not baseline[row["seed"]]["success_primary"]],
                            "regressed": [row["seed"] for row in group if not row["success_primary"] and baseline[row["seed"]]["success_primary"]]}
        _write_csv(output / "repair_feedback_states.csv", traces)
        report = {**protocol, "matched_initial_groups": seed_count, "summary": summary}
        (output / "repair_feedback_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return output, report


def write_repair_distance_trial(output_root=None, first_seed=4801, seed_count=20,
                                sheep_counts=(20, 50), max_steps=2000, variants=None):
    """Single-factor rule-only repair-distance screen, with paired initial states."""
    variants = tuple(REPAIR_DISTANCE_VARIANTS) if variants is None else tuple(variants)
    if (first_seed < 0 or seed_count < 1 or first_seed + seed_count > 2**32
            or not sheep_counts or len(set(sheep_counts)) != len(sheep_counts)
            or any(not isinstance(count, int) or count < 15 for count in sheep_counts)
            or max_steps < 1 or not variants or variants[0] != "baseline"
            or len(set(variants)) != len(variants) or set(variants) - set(REPAIR_DISTANCE_VARIANTS)):
        raise ValueError("Use valid seeds, unique flock sizes >=15, positive cap, and baseline-first known variants")
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": list(range(first_seed, first_seed + seed_count)), "sheep_counts": sheep_counts,
        "anomaly_counts": [5, 5, 5], "max_steps": max_steps,
        "variants": {name: {"HARD_TO_GUIDE_DISTANCE": REPAIR_DISTANCE_VARIANTS[name][0],
                            "COHESION_REPAIR_DISTANCE": REPAIR_DISTANCE_VARIANTS[name][1]} for name in variants},
        "controller": "rule", "api_calls": 0,
        "fixed_environment": {"SPACE_SIZE": 200.0, "SHEEP_START_RADIUS": 15.0, "GOAL_RADIUS": 20.0},
        "base_config": {key: value for key, value in vars(config).items()
                        if key.isupper() and isinstance(value, (str, int, float, bool))},
        "source_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                          for name in ("config.py", "dynamics.py", "environment.py", "detector.py",
                                       "low_level_controller.py", "experiments/anomaly_behavior/behavior_characterization.py")},
        "success_semantics": "Existing pre-physics success check, capped step calls; inside counts use full precision post-step state",
        "selection": "Exploratory: rank nonbaseline variants on N=50 by successes, then mean capped frames; validate only if better than baseline",
        "validation": "Freeze one candidate before seeds 4901-4920; compare baseline/candidate at both sizes; do not change distances after validation",
        "interpretation": "Fixed distances, not adaptive control or an LLM result; 15 abnormal sheep at both sizes, different abnormal fractions",
    }
    (output / "repair_distance_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, traces = [], []
    with temporary_config(**protocol["fixed_environment"]):
        for seed in protocol["seeds"]:
            for count in sheep_counts:
                initial_states = []
                for name in variants:
                    hard, cohesion = REPAIR_DISTANCE_VARIANTS[name]
                    with temporary_config(HARD_TO_GUIDE_DISTANCE=hard, COHESION_REPAIR_DISTANCE=cohesion):
                        row, trace = population_scale_run(seed, count, "mixed", max_steps, max_steps,
                                                          anomaly_counts=(5, 5, 5))
                    row.update(variant=name, hard_distance=hard, cohesion_distance=cohesion)
                    rows.append(row)
                    traces.extend({**sample, "variant": name} for sample in trace)
                    initial_states.append((row["initial_sha256"], row["initial_status_sha256"]))
                    print(f"distance variant={name} seed={seed} N={count}", flush=True)
                    _write_csv(output / "repair_distance_runs.csv", rows)
                if len(set(initial_states)) != 1:
                    raise AssertionError("Distance arms must share full-precision geometry and sheep identities")
    summary = {}
    pairs = []
    for count in sheep_counts:
        baseline = {row["seed"]: row for row in rows if row["num_sheep"] == count and row["variant"] == "baseline"}
        for name in variants:
            group = [row for row in rows if row["num_sheep"] == count and row["variant"] == name]
            summary[f"{count}_{name}"] = {
                "runs": len(group), "successes": sum(row["success_primary"] for row in group),
                **{f"mean_{field}": statistics.mean(row[field] for row in group)
                   for field in ("capped_frames_primary", "inside_primary", "goal_exits_primary",
                                 "danger_frames_primary", "max_spread_primary", "hard_to_guide_frames_primary",
                                 "cohesion_priority_frames_primary")},
                "rescued": [row["seed"] for row in group if row["success_primary"] and not baseline[row["seed"]]["success_primary"]],
                "regressed": [row["seed"] for row in group if not row["success_primary"] and baseline[row["seed"]]["success_primary"]],
            }
            for row in group:
                ref = baseline[row["seed"]]
                pairs.append({"num_sheep": count, "seed": row["seed"], "variant": name,
                              "baseline_success": ref["success_primary"], "variant_success": row["success_primary"],
                              "delta_capped_frames": row["capped_frames_primary"] - ref["capped_frames_primary"],
                              "delta_inside": row["inside_primary"] - ref["inside_primary"],
                              "delta_goal_exits": row["goal_exits_primary"] - ref["goal_exits_primary"]})
    _write_csv(output / "repair_distance_states.csv", traces)
    _write_csv(output / "repair_distance_pairs.csv", pairs)
    report = {**protocol, "matched_initial_groups": seed_count * len(sheep_counts), "summary": summary}
    (output / "repair_distance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report


def write_permission_trial(output_root=None, first_seed=6101, seed_count=100):
    """Matched rule/mode/target/both arms; reuse the production batch runner."""
    if first_seed < 0 or seed_count < 1 or first_seed + seed_count > 2**32:
        raise ValueError("Use valid seed range and positive count")
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    variants = {"rule": ("rule", True, True), "mode_only": ("deepseek", True, False),
                "target_only": ("deepseek", False, True), "mode_target": ("deepseek", True, True)}
    seeds = list(range(first_seed, first_seed + seed_count))
    records = {name: [] for name in variants}
    with temporary_config(NUM_SHEEP=20, SHEEP_START_RADIUS=15., SPACE_SIZE=200., GOAL_RADIUS=20.,
                          HARD_TO_GUIDE_DISTANCE=3., COHESION_REPAIR_DISTANCE=10., LLM_CACHE_ENABLED=False):
        protocol = {"seeds": seeds, "variants": variants, "max_steps": 2000, "population": [5, 5, 5, 5],
                    "pairing": "Same seeds and recorded initial geometry/identities for all arms",
                    "interpretation": "100 distinct shared seeds, one API realization per arm/seed, not repeated identical-start API draws",
                    "base_config": {k: v for k, v in vars(config).items() if k.isupper() and isinstance(v, (str, int, float, bool))},
                    "source_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                                      for name in ("config.py", "dynamics.py", "detector.py", "environment.py", "control_types.py",
                                                   "low_level_controller.py", "high_level_controller.py", "mode_forecast.py",
                                                   "responses_api_backend.py", "run_experiments.py", "experiments/anomaly_behavior/behavior_characterization.py")}}
        (output / "permission_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        for index, seed in enumerate(seeds):
            starts = []
            # Rotate arm order to avoid putting every API policy in the same time slot.
            names = tuple(variants)
            names = names[index % len(names):] + names[:index % len(names)]
            for name in names:
                kind, mode, target = variants[name]
                arm = output / name
                with temporary_config(HIGH_LEVEL_MODE_SELECTION_ENABLED=mode, HIGH_LEVEL_TARGET_SELECTION_ENABLED=target,
                                      DATA_LOG_DIR=str(arm / "data"), CONTROL_LOG_DIR=str(arm / "control")):
                    options = ["--scenario", "custom", "--abnormal-a", "5", "--abnormal-b", "5", "--abnormal-c", "5",
                               "--repair", "--controller", kind, "--no-llm-cache", "--runs", str(seed_count),
                               "--seed-mode", "sequence", "--batch-seed", str(first_seed), "--max-steps", "2000"]
                    if kind == "deepseek" and not target:
                        options.append("--llm-mode-only")
                    elif kind == "deepseek" and not mode:
                        options.append("--llm-target-only")
                    args = parse_args(options)
                    case = select_cases(args)[0]
                    arm.mkdir(parents=True, exist_ok=True)
                    if not records[name]:
                        write_experiment_metadata(arm / "summary.csv", args, seeds, [case])
                    row = run_case(case, seed, 2000, 100000, controller_kind=kind, llm_cache_enabled=False)
                    records[name].append(row)
                    write_summary(records[name], arm / "summary.csv")
                    data = pd.read_parquet(row["Data"], columns=["Frame", "Agent_Type", "Agent_ID", "X", "Y", "Status"])
                    starts.append(data.loc[data.Frame.eq(data.Frame.min())].sort_values(["Agent_Type", "Agent_ID"]).reset_index(drop=True))
                    print(f"permissions seed={seed} {name}: success={row['Success']} frames={row['Frames']} "
                          f"requests={row['HighLevelRequests']} overrides={row['HighLevelOverrideFrames']}", flush=True)
                    if kind == "deepseek":
                        control = pd.read_parquet(row["Control"])
                        errors = control.llm_backend_error.dropna().astype(str)
                        if errors.str.contains(r"Error code: (?:401|402|403|404)|insufficient.*(?:balance|quota)",
                                               case=False, regex=True).any():
                            raise RuntimeError("DeepSeek authentication/billing/model error; partial logs retained")
            for start in starts[1:]:
                pd.testing.assert_frame_equal(starts[0], start)
    summary = {name: {"runs": len(rows), "successes": sum(row["Success"] for row in rows),
                      "mean_frames": statistics.mean(row["Frames"] for row in rows)} for name, rows in records.items()}
    report = {"protocol": protocol, "matched_initial_groups": seed_count, "summary": summary}
    (output / "permission_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report


def write_population_api_trial(output_root=None, first_seed=4601, seed_count=20,
                               ratios=(30, 60, 76)):
    """Compare unchanged rule/DeepSeek control on selected anomaly shares."""
    if first_seed < 0 or seed_count < 1 or first_seed + seed_count > 2**32:
        raise ValueError("Use a valid nonnegative seed sequence and positive run count")
    levels_by_ratio = {30: ("ratio_30", (5, 5, 5)),
                       60: ("ratio_60", (10, 10, 10)),
                       76: ("ratio_76", (13, 13, 12))}
    if not ratios or len(set(ratios)) != len(ratios) or set(ratios) - set(levels_by_ratio):
        raise ValueError("Choose unique anomaly ratios from 30, 60, 76")
    levels = tuple(levels_by_ratio[ratio] for ratio in ratios)
    output = Path(output_root or config.PROJECT_ROOT / "logs/anomaly_behavior") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(first_seed, first_seed + seed_count))
    records, comparisons = {}, {}
    with temporary_config(NUM_SHEEP=50, CONTROLLER_KIND="rule", LLM_CACHE_ENABLED=False):
        protocol = {
            "controller_arms": ["rule", "deepseek"], "num_sheep": 50,
            "levels": levels, "seeds": seeds, "max_steps": 2000,
            "llm_cache_enabled": False, "pairing": "same seed and population within each level",
            "base_config": {key: value for key, value in vars(config).items()
                            if key.isupper() and isinstance(value, (str, int, float, bool))},
            "source_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                              for name in ("config.py", "dynamics.py", "environment.py", "detector.py",
                                           "low_level_controller.py", "high_level_controller.py",
                                           "mode_forecast.py", "responses_api_backend.py", "run_experiments.py",
                                           "experiments/anomaly_behavior/behavior_characterization.py")},
            "interpretation": "Fixed-environment paired API trial; screening versus independent-seed validation depends on the seed cohort, not this runner",
        }
        (output / "population_api_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        for label, population in levels:
            records[label] = {"rule": [], "deepseek": []}
            for seed in seeds:
                for kind in ("rule", "deepseek"):
                    arm = output / label / kind
                    with temporary_config(DATA_LOG_DIR=str(arm / "data"),
                                          CONTROL_LOG_DIR=str(arm / "control")):
                        args = parse_args([
                            "--scenario", "custom", "--abnormal-a", str(population[0]),
                            "--abnormal-b", str(population[1]), "--abnormal-c", str(population[2]),
                            "--repair", "--controller", kind, "--no-llm-cache", "--max-steps", "2000",
                            "--runs", str(seed_count), "--seed-mode", "sequence", "--batch-seed", str(first_seed),
                        ])
                        case = select_cases(args)[0]
                        arm.mkdir(parents=True, exist_ok=True)
                        if not records[label][kind]:
                            write_experiment_metadata(arm / "summary.csv", args, seeds, [case])
                        row = run_case(case, seed, 2000, 100000, controller_kind=kind, llm_cache_enabled=False)
                        records[label][kind].append(row)
                        write_summary([{key: value for key, value in entry.items() if key != "BackendErrors"}
                                       for entry in records[label][kind]], arm / "summary.csv")
                        if kind == "deepseek":
                            control = pd.read_parquet(row["Control"])
                            responses = control.loc[control.llm_request_id.notna()
                                                    & control.llm_backend_error.notna()]
                            failures = responses.drop_duplicates("llm_request_id")
                            row["BackendErrors"] = len(failures)
                            # Abort systemic credential/endpoint failures, rather than
                            # silently completing a "dual-layer" batch on pure fallback.
                            fatal = failures.llm_backend_error.astype(str).str.contains(
                                r"Error code: (?:401|402|403|404)|insufficient.*(?:balance|quota)",
                                case=False, regex=True).any()
                            if fatal:
                                raise RuntimeError(f"DeepSeek authentication/billing/endpoint failure; partial logs retained in {arm}")
                        print(f"{label} seed={seed} {kind}: success={row['Success']} "
                              f"frames={row['Frames']} inside={row['FinalInsideGoal']}/50 "
                              f"requests={row['HighLevelRequests']} responses={row['HighLevelResponses']} "
                              f"override_frames={row['HighLevelOverrideFrames']}", flush=True)
            rule = pd.DataFrame(records[label]["rule"]).set_index("Seed")
            llm = pd.DataFrame(records[label]["deepseek"]).set_index("Seed").loc[rule.index]
            initial_columns = ["Frame", "Agent_Type", "Agent_ID", "X", "Y", "Status"]
            for seed in rule.index:
                starts = []
                for frame in (rule, llm):
                    data = pd.read_parquet(frame.loc[seed, "Data"], columns=initial_columns)
                    first = data.loc[data.Frame.eq(data.Frame.min())].sort_values(["Agent_Type", "Agent_ID"])
                    assert first.Agent_Type.eq("Sheep").sum() == 50
                    starts.append(first.reset_index(drop=True))
                pd.testing.assert_frame_equal(*starts)
            paired = pd.DataFrame({"seed": rule.index,
                                   "rule_success": rule.Success.to_numpy(), "llm_success": llm.Success.to_numpy()})
            for metric in ("Frames", "DangerFrames", "MaxDisp", "FinalInsideGoal", "GoalExits"):
                paired[f"rule_{metric}"] = rule[metric].to_numpy()
                paired[f"llm_{metric}"] = llm[metric].to_numpy()
                paired[f"delta_{metric}"] = llm[metric].to_numpy() - rule[metric].to_numpy()
            paired.to_csv(output / label / "paired.csv", index=False)
            comparisons[label] = {
                "pairs": seed_count, "matched_initial_states": seed_count,
                "rule_successes": int(rule.Success.sum()),
                "llm_successes": int(llm.Success.sum()),
                "rescued": int((~rule.Success & llm.Success).sum()),
                "regressed": int((rule.Success & ~llm.Success).sum()),
                "metrics": {metric: {"rule_mean": float(rule[metric].mean()),
                                     "llm_mean": float(llm[metric].mean()),
                                     "mean_delta": float((llm[metric] - rule[metric]).mean())}
                            for metric in ("Frames", "DangerFrames", "MaxDisp", "FinalInsideGoal", "GoalExits")},
                "api": {key: int(llm[key].sum()) for key in (
                    "HighLevelRequests", "HighLevelResponses", "HighLevelAccepted",
                    "HighLevelRejected", "HighLevelOverrideFrames", "HighLevelCacheHits", "BackendErrors")},
            }
            (output / "population_api_report.json").write_text(
                json.dumps({"protocol": protocol, "completed_levels": comparisons}, indent=2), encoding="utf-8")
    return output, {"protocol": protocol, "completed_levels": comparisons}
