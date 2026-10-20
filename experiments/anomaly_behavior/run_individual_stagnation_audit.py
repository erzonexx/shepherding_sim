"""Compare the current centroid stagnation alarm with per-sheep progress."""

import argparse
import csv
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment import SwarmEnv
from high_level_controller import build_controller
from run_experiments import CASES, make_detector, temporary_config


SCENARIOS = (
    ("normal", "C00R"),
    ("type_a", "C04"),
    ("type_b", "C06"),
    ("stress", "C08"),
)


def stalled_outside_goal(old_distances, current_distances, goal_radius,
                         progress_threshold):
    """Apply the detector's progress threshold separately to each sheep."""
    return (
        (current_distances >= goal_radius)
        & (old_distances - current_distances < progress_threshold)
    )


def run_case(case, seed, max_steps):
    with temporary_config(
        USE_FIXED_SEED=True,
        RANDOM_SEED=seed,
        NUM_ABNORMAL_A=case["num_abnormal_a"],
        NUM_ABNORMAL_B=case["num_abnormal_b"],
        REPAIR_ENABLED=case["repair_enabled"],
        MAX_STEPS=max_steps,
    ):
        env = SwarmEnv(controller=build_controller("rule"))
        detector = make_detector()
        history = deque(maxlen=config.DANGER_STAGNATION_FRAMES)
        counts = {
            "eligible_frames": 0,
            "centroid_stagnation_frames": 0,
            "individual_stagnation_frames": 0,
            "individual_only_frames": 0,
            "individual_only_type_a_frames": 0,
            "individual_only_type_b_frames": 0,
            "individual_only_normal_frames": 0,
            "first_individual_only_frame": None,
            "peak_individually_stagnant_sheep": 0,
        }
        success = False
        try:
            for frame in range(max_steps):
                positions = env.sheep_pos
                distances = np.linalg.norm(
                    positions - env.goal_pos, axis=1
                )
                history.append(distances.copy())
                alarms, _, _ = detector.detect(positions, env.goal_pos)
                if len(history) == config.DANGER_STAGNATION_FRAMES:
                    counts["eligible_frames"] += 1
                    stagnant = stalled_outside_goal(
                        history[0], distances, config.GOAL_RADIUS,
                        config.DANGER_STAGNATION_THRESHOLD,
                    )
                    any_individual = bool(np.any(stagnant))
                    centroid_stagnant = "stagnation" in alarms
                    counts["centroid_stagnation_frames"] += centroid_stagnant
                    counts["individual_stagnation_frames"] += any_individual
                    counts["peak_individually_stagnant_sheep"] = max(
                        counts["peak_individually_stagnant_sheep"],
                        int(np.sum(stagnant)),
                    )
                    if any_individual and not centroid_stagnant:
                        counts["individual_only_frames"] += 1
                        if counts["first_individual_only_frame"] is None:
                            counts["first_individual_only_frame"] = frame
                        for status, key in (
                            ("A", "individual_only_type_a_frames"),
                            ("B", "individual_only_type_b_frames"),
                            ("Normal", "individual_only_normal_frames"),
                        ):
                            counts[key] += bool(
                                np.any(stagnant & (env.sheep_status == status))
                            )
                success = bool(env.step(alarms))
                if success:
                    break
        finally:
            env.close()
        return {
            "seed": seed,
            "success": success,
            "frames": frame + 1,
            **counts,
        }


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario]
        if not group:
            continue
        result[scenario] = {
            "runs": len(group),
            "successes": sum(row["success"] for row in group),
            "runs_with_individual_only": sum(
                row["individual_only_frames"] > 0 for row in group
            ),
            "failed_runs_with_individual_only": sum(
                not row["success"] and row["individual_only_frames"] > 0
                for row in group
            ),
            "mean_eligible_frames": statistics.mean(
                row["eligible_frames"] for row in group
            ),
            "mean_individual_only_frames": statistics.mean(
                row["individual_only_frames"] for row in group
            ),
            "mean_centroid_stagnation_frames": statistics.mean(
                row["centroid_stagnation_frames"] for row in group
            ),
            "mean_individual_stagnation_frames": statistics.mean(
                row["individual_stagnation_frames"] for row in group
            ),
        }
    return result


def run_audit(first_seed=1160, seed_count=80, max_steps=2000):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            rows.append({
                "scenario": scenario,
                **run_case(cases[case_name], seed, max_steps),
            })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=1160)
    parser.add_argument("--seed-count", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    try:
        rows, summary = run_audit(
            args.first_seed, args.seed_count, args.max_steps
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "individual_stagnation_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "individual_stagnation_report.json").write_text(
        json.dumps({
            "controller": "rule",
            "first_seed": args.first_seed,
            "seed_count": args.seed_count,
            "max_steps": args.max_steps,
            "definition": "Outside-goal sheep with less than 1 unit goal-distance progress over 150 frames",
            "summary": summary,
        }, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, metrics in summary.items():
        print(
            f"{scenario}: {metrics['successes']}/{metrics['runs']} success; "
            f"individual-only in {metrics['runs_with_individual_only']} runs, "
            f"including {metrics['failed_runs_with_individual_only']} failures"
        )


if __name__ == "__main__":
    main()
