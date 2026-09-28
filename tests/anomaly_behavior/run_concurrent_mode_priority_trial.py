"""Compare rule-mode priority when splitting and stagnation overlap."""

import argparse
import copy
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
from low_level_controller import LowLevelController
from run_experiments import CASES
from tools.diagnose_failures import diagnose


SCENARIOS = (
    ("normal", "C00R"),
    ("type_a", "C04"),
    ("type_b", "C06"),
    ("stress", "C08"),
)


class GuideFirstOnConcurrent(LowLevelController):
    def _select_mode(self, alarms):
        if (
            self.settings.REPAIR_ENABLED
            and "flock_splitting" in alarms
            and "stagnation" in alarms
        ):
            return "HARD_TO_GUIDE"
        return super()._select_mode(alarms)


class CohesionFirstOnConcurrent(LowLevelController):
    def _select_mode(self, alarms):
        if not self.settings.REPAIR_ENABLED:
            return "TARGET_TRACKING"
        if "flock_splitting" in alarms:
            return "COHESION_PRIORITY"
        if "stagnation" in alarms:
            return "HARD_TO_GUIDE"
        return "TARGET_TRACKING"


class GuideFirstBoundaryGuard(LowLevelController):
    def __init__(self, margin, settings=config):
        super().__init__(settings)
        self.margin = margin

    def decide(self, observation, mode_override=None):
        if (
            mode_override is None
            and self.settings.REPAIR_ENABLED
            and "flock_splitting" in observation.alarms
            and "stagnation" in observation.alarms
        ):
            positions = observation.sheep_positions
            nearest_edge = np.min(np.minimum(
                positions, self.settings.SPACE_SIZE - positions
            ))
            if nearest_edge <= self.margin:
                return super().decide(
                    observation, mode_override="COHESION_PRIORITY"
                )
        return super().decide(observation, mode_override=mode_override)


class GuideFirstTargetBoundaryGuard(LowLevelController):
    """Use cohesion only when the current guide target is near the edge."""

    def __init__(self, margin, settings=config):
        super().__init__(settings)
        self.margin = margin

    def decide(self, observation, mode_override=None):
        if (
            mode_override is None
            and self.settings.REPAIR_ENABLED
            and "flock_splitting" in observation.alarms
            and "stagnation" in observation.alarms
        ):
            preview = LowLevelController.decide(
                copy.copy(self), observation, mode_override="HARD_TO_GUIDE"
            )
            target = observation.sheep_positions[preview.target_sheep_id]
            nearest_edge = np.min(np.minimum(
                target, self.settings.SPACE_SIZE - target
            ))
            if nearest_edge <= self.margin:
                return super().decide(
                    observation, mode_override="COHESION_PRIORITY"
                )
        return super().decide(observation, mode_override=mode_override)


VARIANTS = (
    ("cohesion_first", CohesionFirstOnConcurrent),
    ("guide_first", GuideFirstOnConcurrent),
    ("boundary_0", lambda: GuideFirstBoundaryGuard(0.0)),
    ("boundary_10", lambda: GuideFirstBoundaryGuard(10.0)),
    ("boundary_20", lambda: GuideFirstBoundaryGuard(20.0)),
    ("target_boundary_0", lambda: GuideFirstTargetBoundaryGuard(0.0)),
    ("target_boundary_5", lambda: GuideFirstTargetBoundaryGuard(5.0)),
    ("target_boundary_10", lambda: GuideFirstTargetBoundaryGuard(10.0)),
)


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario
            and row["variant"] == "cohesion_first"
        }
        result[scenario] = {}
        for variant, _ in VARIANTS:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["variant"] == variant
            ]
            if not group:
                continue
            result[scenario][variant] = {
                "runs": len(group),
                "successes": sum(row["success"] for row in group),
                "rescued": sum(
                    row["success"] and not baseline[row["seed"]]["success"]
                    for row in group
                ),
                "regressed": sum(
                    not row["success"] and baseline[row["seed"]]["success"]
                    for row in group
                ),
                "mean_capped_frames": statistics.mean(
                    row["capped_frames"] for row in group
                ),
                "mean_goal_exits": statistics.mean(
                    row["goal_exits"] for row in group
                ),
                "mean_peak_inside_goal": statistics.mean(
                    row["peak_inside_goal"] for row in group
                ),
                "mean_peak_max_spread": statistics.mean(
                    row["peak_max_spread"] for row in group
                ),
                "mean_outside_state_samples": statistics.mean(
                    row["outside_state_samples"] for row in group
                ),
                "mean_max_excursion": statistics.mean(
                    row["max_excursion"] for row in group
                ),
                "mean_alarm_frames": statistics.mean(
                    row["alarm_frames"] for row in group
                ),
                "runs_with_concurrent_alarm": sum(
                    row["concurrent_alarm_frames"] > 0 for row in group
                ),
                "mean_concurrent_alarm_frames": statistics.mean(
                    row["concurrent_alarm_frames"] for row in group
                ),
                "mean_concurrent_guide_frames": statistics.mean(
                    row["concurrent_guide_frames"] for row in group
                ),
            }
    return result


def run_trial(first_seed=650, seed_count=20, max_steps=1000,
              variant_names=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected = (
        VARIANTS[:2] if variant_names is None
        else tuple(item for item in VARIANTS if item[0] in variant_names)
    )
    if variant_names is not None and {name for name, _ in selected} != set(variant_names):
        raise ValueError("unknown concurrent-mode variant")
    if "cohesion_first" not in {name for name, _ in selected}:
        raise ValueError("include the cohesion-first baseline")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            for variant, factory in selected:
                report, states, actions = diagnose(
                    cases[case_name], seed, max_steps,
                    controller_factory=factory,
                )
                concurrent = [
                    action for action in actions
                    if "flock_splitting" in action["alarm"]
                    and "stagnation" in action["alarm"]
                ]
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "goal_exits": report["individual_goal_exits"],
                    "peak_inside_goal": report["peak_inside_goal"],
                    "peak_max_spread": max(
                        state["max_spread"] for state in states
                    ),
                    "outside_state_samples": report["outside_state_samples"],
                    "max_excursion": report["max_excursion"],
                    "alarm_frames": sum(
                        action["alarm"] != "none" for action in actions
                    ),
                    "concurrent_alarm_frames": len(concurrent),
                    "concurrent_guide_frames": sum(
                        action["mode"] == "HARD_TO_GUIDE"
                        for action in concurrent
                    ),
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=650)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", choices=[name for name, _ in VARIANTS])
    args = parser.parse_args()
    try:
        rows, summary = run_trial(
            args.first_seed, args.seed_count, args.max_steps,
            variant_names=args.variants,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "concurrent_mode_priority_runs.csv").open(
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
        "candidate": "HARD_TO_GUIDE rather than COHESION_PRIORITY on simultaneous alarms",
        "summary": summary,
    }
    (output / "concurrent_mode_priority_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, variants in summary.items():
        print(scenario)
        for name, result in variants.items():
            print(
                f"  {name}: {result['successes']}/{result['runs']} success, "
                f"rescued {result['rescued']}, regressed {result['regressed']}, "
                f"dual-alarm runs {result['runs_with_concurrent_alarm']}"
            )


if __name__ == "__main__":
    main()
