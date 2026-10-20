"""Screen one-at-a-time repair geometry changes with the current detector."""

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
from run_experiments import CASES, temporary_config
from tools.diagnose_failures import diagnose


SCENARIOS = (("normal", "C00R"), ("type_b", "C06"), ("mixed", "C08"))
VARIANTS = (
    ("current", {}),
    ("cohesion_15", {"COHESION_REPAIR_DISTANCE": 15.0}),
    ("cohesion_20", {"COHESION_REPAIR_DISTANCE": 20.0}),
    ("cohesion_25", {"COHESION_REPAIR_DISTANCE": 25.0}),
    ("guide_6", {"HARD_TO_GUIDE_DISTANCE": 6.0}),
    ("guide_10", {"HARD_TO_GUIDE_DISTANCE": 10.0}),
)


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        result[scenario] = {}
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario and row["variant"] == "current"
        }
        if not baseline:
            continue
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
                    not baseline[row["seed"]]["success"] and row["success"]
                    for row in group
                ),
                "regressed": sum(
                    baseline[row["seed"]]["success"] and not row["success"]
                    for row in group
                ),
                "mean_capped_frames": statistics.mean(
                    row["capped_frames"] for row in group
                ),
                "mean_goal_exits": statistics.mean(
                    row["goal_exits"] for row in group
                ),
            }
    return result


def run_screen(first_seed=100, seed_count=20, max_steps=1000, variant_names=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected = (
        VARIANTS if variant_names is None
        else tuple(variant for variant in VARIANTS if variant[0] in variant_names)
    )
    if variant_names is not None and {name for name, _ in selected} != set(variant_names):
        raise ValueError("unknown repair parameter variant")
    if "current" not in {name for name, _ in selected}:
        raise ValueError("current baseline must be included")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            for variant, overrides in selected:
                with temporary_config(**overrides):
                    report, _, _ = diagnose(cases[case_name], seed, max_steps)
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "goal_exits": report["individual_goal_exits"],
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", choices=[name for name, _ in VARIANTS])
    args = parser.parse_args()
    if args.first_seed < 0 or args.seed_count < 1 or args.max_steps < 1:
        parser.error("first seed must be nonnegative; counts must be positive")
    try:
        rows, summary = run_screen(
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
    with (output / "repair_parameter_runs.csv").open(
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
        "baseline_cohesion_distance": config.COHESION_REPAIR_DISTANCE,
        "baseline_guide_distance": config.HARD_TO_GUIDE_DISTANCE,
        "variants": {name: overrides for name, overrides in VARIANTS},
        "summary": summary,
    }
    (output / "repair_parameter_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, results in summary.items():
        print(f"{scenario}:")
        for variant, result in results.items():
            print(
                f"  {variant}: {result['successes']}/{result['runs']} success, "
                f"rescued {result['rescued']}, regressed {result['regressed']}, "
                f"mean capped {result['mean_capped_frames']:.1f}"
            )


if __name__ == "__main__":
    main()
