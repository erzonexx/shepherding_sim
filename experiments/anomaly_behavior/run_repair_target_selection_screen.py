"""Paired ablations of repair-target prediction and switching rules."""

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
    ("lookahead_0", {"REPAIR_TARGET_LOOKAHEAD_FRAMES": 0}),
    ("guide_hold_0", {"HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES": 0}),
    ("guide_hold_60", {"HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES": 60}),
    ("switch_margin_0", {"REPAIR_TARGET_SWITCH_MARGIN": 0.0}),
    ("switch_margin_6", {"REPAIR_TARGET_SWITCH_MARGIN": 6.0}),
)


def same_mode_switches(actions, mode):
    return sum(
        previous["mode"] == current["mode"] == mode
        and previous["target_id"] is not None
        and current["target_id"] is not None
        and previous["target_id"] != current["target_id"]
        for previous, current in zip(actions, actions[1:])
    )


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario and row["variant"] == "current"
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
                "mean_guide_target_switches": statistics.mean(
                    row["guide_target_switches"] for row in group
                ),
                "mean_cohesion_target_switches": statistics.mean(
                    row["cohesion_target_switches"] for row in group
                ),
            }
    return result


def run_screen(first_seed=470, seed_count=20, max_steps=1000,
               variant_names=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected = (
        VARIANTS if variant_names is None
        else tuple(item for item in VARIANTS if item[0] in variant_names)
    )
    if variant_names is not None and {name for name, _ in selected} != set(variant_names):
        raise ValueError("unknown target-selection variant")
    if "current" not in {name for name, _ in selected}:
        raise ValueError("include the current baseline")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            for variant, overrides in selected:
                with temporary_config(**overrides):
                    report, _, actions = diagnose(
                        cases[case_name], seed, max_steps
                    )
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "goal_exits": report["individual_goal_exits"],
                    "peak_inside_goal": report["peak_inside_goal"],
                    "guide_target_switches": same_mode_switches(
                        actions, "HARD_TO_GUIDE"
                    ),
                    "cohesion_target_switches": same_mode_switches(
                        actions, "COHESION_PRIORITY"
                    ),
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=470)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", choices=[name for name, _ in VARIANTS])
    args = parser.parse_args()
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
    with (output / "repair_target_selection_runs.csv").open(
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
        "baseline": {
            "lookahead_frames": config.REPAIR_TARGET_LOOKAHEAD_FRAMES,
            "guide_hold_frames": config.HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES,
            "cohesion_hold_frames": config.COHESION_TARGET_MIN_HOLD_FRAMES,
            "switch_margin": config.REPAIR_TARGET_SWITCH_MARGIN,
        },
        "variants": {name: values for name, values in VARIANTS},
        "summary": summary,
    }
    (output / "repair_target_selection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, variants in summary.items():
        print(scenario)
        for name, result in variants.items():
            print(
                f"  {name}: {result['successes']}/{result['runs']} success, "
                f"rescued {result['rescued']}, regressed {result['regressed']}, "
                f"guide switches {result['mean_guide_target_switches']:.1f}"
            )


if __name__ == "__main__":
    main()
