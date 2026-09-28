"""Test a lateral dog approach to the guide target, without production edits."""

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


SCENARIOS = (("normal", "C00"), ("type_b", "C06"), ("mixed", "C08"))
VARIANTS = (
    ("current", None, None, None),
    ("lateral_20", 20.0, None, None),
    ("lateral_40", 40.0, None, None),
    ("lateral_20_near", 20.0, config.DOG_SENSING_RANGE, None),
    ("near_goal_lateral_10", 10.0, None, 18),
    ("near_goal_lateral_20", 20.0, None, 18),
    ("near_goal_lateral_20_at15", 20.0, None, 15),
)


class LateralGuideController(LowLevelController):
    def __init__(self, radius, activation_distance=None,
                 min_inside_goal=None, settings=config):
        super().__init__(settings)
        self.radius = radius
        self.activation_distance = activation_distance
        self.min_inside_goal = min_inside_goal
        self.approach_target = None
        self.approach_side = 1.0

    def decide(self, observation, mode_override=None):
        decision = super().decide(observation, mode_override=mode_override)
        if decision.mode != "HARD_TO_GUIDE":
            self.approach_target = None
            return decision
        if self.min_inside_goal is not None:
            inside = np.sum(
                np.linalg.norm(
                    observation.sheep_positions - observation.goal_position,
                    axis=1,
                ) < self.settings.GOAL_RADIUS
            )
            if inside < self.min_inside_goal:
                self.approach_target = None
                return decision

        target = observation.sheep_positions[decision.target_sheep_id]
        goal_vector = observation.goal_position - target
        goal_distance = np.linalg.norm(goal_vector)
        if goal_distance < self.settings.EPSILON:
            return decision
        goal_direction = goal_vector / goal_distance
        lateral_direction = np.array([-goal_direction[1], goal_direction[0]])
        relative_dog = observation.dog_position - target
        forward = float(np.dot(relative_dog, goal_direction))
        lateral = float(np.dot(relative_dog, lateral_direction))

        if forward <= 0:
            return replace(decision, metadata={"approach_phase": "behind"})
        if (
            self.activation_distance is not None
            and np.linalg.norm(relative_dog) > self.activation_distance
        ):
            return replace(decision, metadata={"approach_phase": "far_direct"})
        if self.approach_target != decision.target_sheep_id:
            self.approach_target = decision.target_sheep_id
            self.approach_side = self._choose_side(
                target, lateral_direction, lateral
            )

        if self.approach_side * lateral < self.radius:
            waypoint = observation.dog_position + (
                self.approach_side * lateral_direction * self.radius
            )
            phase = "sidestep"
        else:
            waypoint = (
                target
                + self.approach_side * lateral_direction * self.radius
                - goal_direction
                * (self.radius + self.settings.HARD_TO_GUIDE_DISTANCE)
            )
            phase = "circle"
        waypoint = np.clip(waypoint, 0.0, self.settings.SPACE_SIZE)
        return replace(
            decision,
            navigation_point=waypoint,
            metadata={"approach_phase": phase},
        )

    def _choose_side(self, target, lateral_direction, lateral):
        def clearance(side):
            point = target + side * lateral_direction * self.radius
            return float(min(
                point[0], point[1],
                self.settings.SPACE_SIZE - point[0],
                self.settings.SPACE_SIZE - point[1],
            ))

        preferred = 1.0 if lateral >= 0 else -1.0
        if clearance(preferred) >= 0:
            return preferred
        return 1.0 if clearance(1.0) >= clearance(-1.0) else -1.0


def summarize(rows):
    output = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario and row["variant"] == "current"
        }
        output[scenario] = {}
        for variant, _, _, _ in VARIANTS:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["variant"] == variant
            ]
            if not group:
                continue
            guide_frames = sum(row["guide_frames"] for row in group)
            output[scenario][variant] = {
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
                "mean_peak_inside_goal": statistics.mean(
                    row["peak_inside_goal"] for row in group
                ),
                "guide_frames": guide_frames,
                "guide_outward_dog_response_share": (
                    sum(row["guide_outward_frames"] for row in group)
                    / guide_frames if guide_frames else None
                ),
                "guide_behind_dog_share": (
                    sum(row["guide_behind_frames"] for row in group)
                    / guide_frames if guide_frames else None
                ),
                "sidestep_frames": sum(row["sidestep_frames"] for row in group),
            }
    return output


def run_trial(first_seed=370, seed_count=20, max_steps=1000,
              variant_names=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected = (
        VARIANTS if variant_names is None
        else tuple(item for item in VARIANTS if item[0] in variant_names)
    )
    if {name for name, _, _, _ in selected} != (
        {name for name, _, _, _ in VARIANTS} if variant_names is None
        else set(variant_names)
    ):
        raise ValueError("unknown approach variant")
    if "current" not in {name for name, _, _, _ in selected}:
        raise ValueError("include the current baseline")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, name in SCENARIOS:
            case = cases["C00R" if name == "C00" else name]
            for variant, radius, activation_distance, min_inside_goal in selected:
                factory = (
                    None if radius is None
                    else lambda radius=radius, activation_distance=activation_distance, min_inside_goal=min_inside_goal: (
                        LateralGuideController(
                            radius, activation_distance, min_inside_goal
                        )
                    )
                )
                report, _, actions = diagnose(
                    case, seed, max_steps, controller_factory=factory
                )
                guide = [
                    action for action in actions
                    if action["mode"] == "HARD_TO_GUIDE"
                ]
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "peak_inside_goal": report["peak_inside_goal"],
                    "guide_frames": len(guide),
                    "guide_outward_frames": sum(
                        action["target_outward_dog_response"] > 0
                        for action in guide
                    ),
                    "guide_behind_frames": sum(
                        action["dog_behind_target"] for action in guide
                    ),
                    "sidestep_frames": sum(
                        action["approach_phase"] == "sidestep"
                        for action in guide
                    ),
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=370)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--variants", nargs="+",
        choices=[name for name, _, _, _ in VARIANTS],
    )
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
    with (output / "guide_approach_runs.csv").open(
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
        "candidate": "Guide target sidestep then circle from the goal-facing side",
        "summary": summary,
    }
    (output / "guide_approach_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, variants in summary.items():
        print(scenario)
        for name, result in variants.items():
            outward = result["guide_outward_dog_response_share"]
            outward_text = "n/a" if outward is None else f"{outward:.2f}"
            print(
                f"  {name}: {result['successes']}/{result['runs']} success, "
                f"rescued {result['rescued']}, regressed {result['regressed']}, "
                f"outward-response share {outward_text}"
            )


if __name__ == "__main__":
    main()
