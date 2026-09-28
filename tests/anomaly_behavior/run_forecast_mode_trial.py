"""Test a conservative, local forecast choice on simultaneous alarms."""

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
from high_level_controller import HighLevelController
from low_level_controller import LowLevelController
from mode_forecast import forecast_modes
from run_experiments import CASES
from tools.diagnose_failures import diagnose


SCENARIOS = (
    ("normal", "C00R"),
    ("type_a", "C04"),
    ("type_b", "C06"),
    ("stress", "C08"),
)


class ForecastDominanceController(LowLevelController):
    """Override guide-first only when cohesion's forecast strictly dominates."""

    def __init__(self, review_period=120, settings=config):
        super().__init__(settings)
        self.review_period = review_period
        self.next_review_frame = 0
        self.override_mode = None
        self.previous_concurrent = False
        self.forecast_calls = 0

    def decide(self, observation, mode_override=None):
        concurrent = (
            self.settings.REPAIR_ENABLED
            and "flock_splitting" in observation.alarms
            and "stagnation" in observation.alarms
        )
        if mode_override is not None or not concurrent:
            self.override_mode = None
            self.previous_concurrent = False
            return super().decide(observation, mode_override=mode_override)

        if not self.previous_concurrent or observation.frame >= self.next_review_frame:
            forecasts = forecast_modes(observation, self.settings)
            self.forecast_calls += 1
            self.override_mode = (
                "COHESION_PRIORITY"
                if HighLevelController._forecast_dominates(
                    forecasts["COHESION_PRIORITY"],
                    forecasts["HARD_TO_GUIDE"],
                ) else None
            )
            self.next_review_frame = observation.frame + self.review_period
        self.previous_concurrent = True
        return super().decide(observation, mode_override=self.override_mode)


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario and row["variant"] == "guide_first"
        }
        result[scenario] = {}
        for variant in ("guide_first", "forecast_dominance"):
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
                "mean_outside_state_samples": statistics.mean(
                    row["outside_state_samples"] for row in group
                ),
                "mean_forecast_calls": statistics.mean(
                    row["forecast_calls"] for row in group
                ),
                "mean_concurrent_cohesion_frames": statistics.mean(
                    row["concurrent_cohesion_frames"] for row in group
                ),
            }
    return result


def run_trial(first_seed=1320, seed_count=20, max_steps=2000,
              review_period=120):
    if first_seed < 0 or min(seed_count, max_steps, review_period) < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            for variant in ("guide_first", "forecast_dominance"):
                controller = (
                    LowLevelController() if variant == "guide_first"
                    else ForecastDominanceController(review_period)
                )
                report, _, actions = diagnose(
                    cases[case_name], seed, max_steps,
                    controller_factory=lambda: controller,
                )
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "outside_state_samples": report["outside_state_samples"],
                    "max_excursion": report["max_excursion"],
                    "forecast_calls": getattr(controller, "forecast_calls", 0),
                    "concurrent_cohesion_frames": sum(
                        action["mode"] == "COHESION_PRIORITY"
                        and "flock_splitting" in action["alarm"]
                        and "stagnation" in action["alarm"]
                        for action in actions
                    ),
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=1320)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--review-period", type=int, default=120)
    args = parser.parse_args()
    try:
        rows, summary = run_trial(
            args.first_seed, args.seed_count, args.max_steps,
            args.review_period,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "forecast_mode_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "forecast_mode_report.json").write_text(
        json.dumps({
            "controller": "rule/local_forecast",
            "first_seed": args.first_seed,
            "seed_count": args.seed_count,
            "max_steps": args.max_steps,
            "review_period": args.review_period,
            "summary": summary,
        }, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, variants in summary.items():
        for name, metrics in variants.items():
            print(
                f"{scenario} {name}: {metrics['successes']}/{metrics['runs']} "
                f"success; rescued {metrics['rescued']}, "
                f"regressed {metrics['regressed']}; "
                f"mean forecast calls {metrics['mean_forecast_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
