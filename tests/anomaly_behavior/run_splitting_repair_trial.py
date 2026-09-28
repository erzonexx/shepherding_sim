"""Compare the rule controller under current and candidate splitting alarms."""

import argparse
from collections import Counter
import csv
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
from run_experiments import CASES, make_detector
from tests.anomaly_behavior.run_type_b_threshold_trials import POST_START_FRAME
from tools.diagnose_failures import diagnose


SCENARIOS = (("normal", "C00R"), ("type_b", "C06"), ("mixed", "C08"))
MAX_STREAK_FRAMES = 10


class SustainedMaxDetector:
    """Preserve early alarms, then require sustained maximum spread."""

    def __init__(self):
        self.base = make_detector()
        self.frame = 0
        self.max_streak = 0

    def detect(self, sheep_pos, goal_pos):
        alarms, metrics, report = self.base.detect(sheep_pos, goal_pos)
        if self.frame >= POST_START_FRAME:
            center = np.mean(sheep_pos, axis=0)
            max_spread = float(np.linalg.norm(sheep_pos - center, axis=1).max())
            self.max_streak = (
                self.max_streak + 1
                if max_spread > config.DANGER_DISPERSION_THRESHOLD else 0
            )
            splitting = self.max_streak >= MAX_STREAK_FRAMES
            alarms = (
                (["flock_splitting"] if splitting else [])
                + (["stagnation"] if "stagnation" in alarms else [])
            )
            metrics = {**metrics, "is_danger": int(bool(alarms))}
            report = {
                **report,
                "is_danger": bool(alarms),
                "danger_type": alarms[0] if alarms else "None",
                "danger_types": alarms,
                "description": "Candidate splitting rule used for offline trial.",
            }
        self.frame += 1
        return alarms, metrics, report


def run_trials(first_seed=60, seed_count=40, max_steps=1000, seeds=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected_seeds = list(seeds) if seeds is not None else list(
        range(first_seed, first_seed + seed_count)
    )
    if not selected_seeds or any(seed < 0 for seed in selected_seeds):
        raise ValueError("seeds must be nonnegative and nonempty")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in selected_seeds:
        for scenario, case_name in SCENARIOS:
            pair = {}
            for label, factory in (
                ("current", None), ("sustained_max_10", SustainedMaxDetector),
            ):
                result, states, actions = diagnose(
                    cases[case_name], seed, max_steps,
                    detector_factory=factory,
                )
                modes = Counter(action["mode"] for action in actions)
                cohesion_actions = [
                    action for action in actions
                    if action["mode"] == "COHESION_PRIORITY"
                ]
                cohesion_closer_to_goal = sum(
                    action["target_goal_distance_before"]
                    < states[action["from_state_frame"]]["centroid_distance"]
                    for action in cohesion_actions
                    if action["target_goal_distance_before"] is not None
                )
                pair[label] = {
                    "success": bool(result["success_reported_by_env"]),
                    "capped_frames": result["step_calls"],
                    "final_centroid_distance": states[-1]["centroid_distance"],
                    "peak_max_spread": max(state["max_spread"] for state in states),
                    "goal_exits": result["individual_goal_exits"],
                    "mode_switches": result["mode_switches"],
                    "target_switches": result["target_switches"],
                    "tracking_frames": modes["TARGET_TRACKING"],
                    "cohesion_frames": modes["COHESION_PRIORITY"],
                    "hard_to_guide_frames": modes["HARD_TO_GUIDE"],
                    "cohesion_targets_type_b": sum(
                        action["target_status"] == "B"
                        for action in cohesion_actions
                    ),
                    "cohesion_targets_normal": sum(
                        action["target_status"] == "Normal"
                        for action in cohesion_actions
                    ),
                    "cohesion_target_closer_to_goal_frames": cohesion_closer_to_goal,
                    "splitting_alarm_frames": sum(
                        "flock_splitting" in action["alarm"].split(";")
                        for action in actions
                    ),
                }
            rows.append({
                "seed": seed,
                "scenario": scenario,
                **{
                    f"{label}_{metric}": value
                    for label, result in pair.items()
                    for metric, value in result.items()
                },
            })
    return rows, summarize(rows)


def summarize(rows):
    summary = {}
    for scenario, _ in SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario]
        summary[scenario] = {
            "runs": len(group),
            "current_successes": sum(row["current_success"] for row in group),
            "candidate_successes": sum(
                row["sustained_max_10_success"] for row in group
            ),
            "rescued": sum(
                not row["current_success"] and row["sustained_max_10_success"]
                for row in group
            ),
            "regressed": sum(
                row["current_success"] and not row["sustained_max_10_success"]
                for row in group
            ),
            "current_mean_capped_frames": statistics.mean(
                row["current_capped_frames"] for row in group
            ),
            "candidate_mean_capped_frames": statistics.mean(
                row["sustained_max_10_capped_frames"] for row in group
            ),
            "current_mean_goal_exits": statistics.mean(
                row["current_goal_exits"] for row in group
            ),
            "candidate_mean_goal_exits": statistics.mean(
                row["sustained_max_10_goal_exits"] for row in group
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=60)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if (args.first_seed < 0 or args.seed_count < 1 or args.max_steps < 1
            or (args.seeds is not None and (not args.seeds or min(args.seeds) < 0))):
        parser.error("first seed must be nonnegative; counts must be positive")
    rows, summary = run_trials(
        args.first_seed, args.seed_count, args.max_steps, seeds=args.seeds
    )
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "splitting_repair_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "controller": "rule",
        "repair_enabled": True,
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "candidate": {
            "preserve_current_before_frame": POST_START_FRAME,
            "max_spread_threshold": config.DANGER_DISPERSION_THRESHOLD,
            "required_consecutive_frames": MAX_STREAK_FRAMES,
        },
        "summary": summary,
    }
    (output / "splitting_repair_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, result in summary.items():
        print(
            f"{scenario}: success {result['current_successes']} -> "
            f"{result['candidate_successes']}/{result['runs']}, "
            f"rescued {result['rescued']}, regressed {result['regressed']}, "
            "mean capped frames "
            f"{result['current_mean_capped_frames']:.1f} -> "
            f"{result['candidate_mean_capped_frames']:.1f}"
        )


if __name__ == "__main__":
    main()
