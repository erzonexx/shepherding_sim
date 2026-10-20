"""Summarize Type B repair trajectories without changing controller behavior."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from run_experiments import CASES
from tools.diagnose_failures import diagnose


def target_runs(actions, mode):
    runs = []
    current = []
    for action in actions:
        if action["mode"] != mode or action["target_id"] is None:
            if current:
                runs.append(current)
                current = []
            continue
        if current and action["target_id"] != current[-1]["target_id"]:
            runs.append(current)
            current = []
        current.append(action)
    if current:
        runs.append(current)
    return runs


def summarize_trace(report, states, actions):
    cohesion = [a for a in actions if a["mode"] == "COHESION_PRIORITY"]
    guide = [a for a in actions if a["mode"] == "HARD_TO_GUIDE"]
    concurrent_indices = [
        index for index, action in enumerate(actions)
        if "flock_splitting" in action.get("alarm", "")
        and "stagnation" in action.get("alarm", "")
    ]
    first_concurrent = concurrent_indices[0] if concurrent_indices else None
    guide_runs = [run for run in target_runs(actions, "HARD_TO_GUIDE") if len(run) >= 30]
    worsening_runs = sum(
        run[-1]["target_goal_distance_before"]
        > run[0]["target_goal_distance_before"] + 1.0
        for run in guide_runs
    )
    def share(rows, key):
        return sum(bool(row[key]) for row in rows) / len(rows) if rows else None

    return {
        "seed": report["seed"],
        "success": report["success_reported_by_env"],
        "frames": len(actions),
        "peak_inside_goal": report["peak_inside_goal"],
        "final_inside_goal": report["final_inside_goal"],
        "centroid_start": states[0]["centroid_distance"],
        "centroid_min": min(s["centroid_distance"] for s in states),
        "centroid_end": states[-1]["centroid_distance"],
        "goal_exits": report["individual_goal_exits"],
        "concurrent_alarm_frames": len(concurrent_indices),
        "first_concurrent_frame": first_concurrent,
        "frames_after_first_concurrent": (
            len(actions) - first_concurrent
            if first_concurrent is not None else 0
        ),
        "cohesion_frames": len(cohesion),
        "cohesion_nonfarthest_frames": sum(
            not a["target_is_farthest_from_center"] for a in cohesion
        ),
        "cohesion_behind_share": share(cohesion, "dog_behind_target"),
        "cohesion_outside_drive_share": share(cohesion, "drive_point_outside"),
        "guide_frames": len(guide),
        "guide_behind_share": share(guide, "dog_behind_target"),
        "guide_outside_drive_share": share(guide, "drive_point_outside"),
        "guide_long_target_runs": len(guide_runs),
        "guide_worsening_target_runs": worsening_runs,
        "guide_near_drive_point_share": share(guide, "near_drive_point"),
        "guide_outward_dog_response_share": sum(
            a["target_outward_dog_response"] > 0 for a in guide
            if a["target_outward_dog_response"] is not None
        ) / len(guide) if guide else None,
    }


def run_trace(first_seed=120, seed_count=40, max_steps=1000, case_name="C06"):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    if case_name not in cases or not cases[case_name]["repair_enabled"]:
        raise ValueError("choose a case with rule repair enabled")
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        report, states, actions = diagnose(cases[case_name], seed, max_steps)
        rows.append(summarize_trace(report, states, actions))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=120)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--case", choices=("C06", "C08"), default="C06")
    args = parser.parse_args()
    try:
        rows = run_trace(args.first_seed, args.seed_count, args.max_steps, args.case)
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "type_b_failure_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    by_outcome = {}
    for label, group in (
        ("success", [row for row in rows if row["success"]]),
        ("failure", [row for row in rows if not row["success"]]),
    ):
        by_outcome[label] = {
            "runs": len(group),
            "mean_peak_inside_goal": statistics.mean(
                row["peak_inside_goal"] for row in group
            ) if group else None,
            "mean_cohesion_frames": statistics.mean(
                row["cohesion_frames"] for row in group
            ) if group else None,
            "mean_guide_frames": statistics.mean(
                row["guide_frames"] for row in group
            ) if group else None,
            "cohesion_nonfarthest_frames": sum(
                row["cohesion_nonfarthest_frames"] for row in group
            ),
            "cohesion_frames": sum(row["cohesion_frames"] for row in group),
            "guide_worsening_target_runs": sum(
                row["guide_worsening_target_runs"] for row in group
            ),
            "guide_long_target_runs": sum(
                row["guide_long_target_runs"] for row in group
            ),
        }
    report = {
        "case": args.case,
        "controller": "rule",
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "max_steps": args.max_steps,
        "summary": by_outcome,
    }
    (output / "type_b_failure_trace.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
