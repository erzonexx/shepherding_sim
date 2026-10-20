"""Paired trial of unit-length cohesion drive direction, without a production edit."""

import argparse
import csv
from dataclasses import replace
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
from low_level_controller import LowLevelController
from run_experiments import CASES
from tools.diagnose_failures import diagnose


class UnitCohesionController(LowLevelController):
    def decide(self, observation, mode_override=None):
        decision = super().decide(observation, mode_override=mode_override)
        if decision.mode != "COHESION_PRIORITY":
            return decision
        center = np.mean(observation.sheep_positions, axis=0)
        target = observation.sheep_positions[decision.target_sheep_id]
        outward = target - center
        point = target + outward / (
            np.linalg.norm(outward) + self.settings.EPSILON
        ) * self.settings.COHESION_REPAIR_DISTANCE
        return replace(decision, drive_point=point, navigation_point=point.copy())


SCENARIOS = (("normal", "C00"), ("type_b", "C06"), ("mixed", "C08"))


def summarize(rows):
    summary = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario and row["variant"] == "current"
        }
        candidate = [
            row for row in rows
            if row["scenario"] == scenario and row["variant"] == "unit_direction"
        ]
        summary[scenario] = {
            "runs": len(candidate),
            "baseline_successes": sum(row["success"] for row in baseline.values()),
            "candidate_successes": sum(row["success"] for row in candidate),
            "rescued": sum(
                row["success"] and not baseline[row["seed"]]["success"]
                for row in candidate
            ),
            "regressed": sum(
                not row["success"] and baseline[row["seed"]]["success"]
                for row in candidate
            ),
            "baseline_mean_capped_frames": statistics.mean(
                row["capped_frames"] for row in baseline.values()
            ),
            "candidate_mean_capped_frames": statistics.mean(
                row["capped_frames"] for row in candidate
            ),
            "baseline_mean_goal_exits": statistics.mean(
                row["goal_exits"] for row in baseline.values()
            ),
            "candidate_mean_goal_exits": statistics.mean(
                row["goal_exits"] for row in candidate
            ),
        }
    return summary


def run_trial(first_seed=210, seed_count=20, max_steps=1000):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, name in SCENARIOS:
            case = cases["C00R" if name == "C00" else name]
            for variant, factory in (
                ("current", None),
                ("unit_direction", UnitCohesionController),
            ):
                report, _, _ = diagnose(
                    case, seed, max_steps, controller_factory=factory
                )
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "goal_exits": report["individual_goal_exits"],
                    "peak_inside_goal": report["peak_inside_goal"],
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=210)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    try:
        rows, summary = run_trial(
            args.first_seed, args.seed_count, args.max_steps
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cohesion_direction_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "controller": "rule",
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "max_steps": args.max_steps,
        "candidate": "Normalize the selected cohesion target's outward vector",
        "summary": summary,
    }
    (output / "cohesion_direction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
