"""Small deterministic trials that characterize the current anomaly model."""

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import config
from detector import Detector
from dynamics import advance_sheep
from environment import SwarmEnv
from run_experiments import temporary_config


DOG_RESPONSE_DISTANCES = (3.0, 10.0, 20.0, 40.0, 60.0)
DISPERSION_SAMPLE_FRAMES = (0, 1, 10, 30, 60)


def dog_response_curve(distances=DOG_RESPONSE_DISTANCES):
    """Measure one-frame motion caused only by a stationary dog."""
    rows = []
    for distance in distances:
        for status in ("Normal", "A"):
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
):
    """Compare the same compact flock with one Normal or one Type B sheep."""
    initial_positions = np.array([
        [96.0, 96.0], [100.0, 96.0], [104.0, 96.0],
        [96.0, 100.0], [100.0, 100.0], [104.0, 100.0],
        [96.0, 104.0], [100.0, 104.0], [104.0, 104.0],
    ])
    focus_id = 5
    requested_frames = sorted(set(sample_frames) | {0, frames})
    rows = []
    for focus_status in ("Normal", "B"):
        positions = initial_positions.copy()
        velocities = np.zeros_like(positions)
        statuses = np.array(["Normal"] * len(positions), dtype=object)
        statuses[focus_id] = focus_status
        with temporary_config(GOAL_RETENTION_ENABLED=False):
            for frame in range(frames + 1):
                if frame in requested_frames:
                    center = np.mean(positions, axis=0)
                    spread = np.linalg.norm(positions - center, axis=1)
                    rows.append({
                        "variant": "one_type_b" if focus_status == "B" else "all_normal",
                        "frame": frame,
                        "focus_distance_to_centroid": float(spread[focus_id]),
                        "mean_spread": float(np.mean(spread)),
                        "max_spread": float(np.max(spread)),
                        "focus_speed": float(np.linalg.norm(velocities[focus_id])),
                    })
                if frame == frames:
                    break
                positions, velocities = advance_sheep(
                    positions,
                    velocities,
                    statuses,
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
