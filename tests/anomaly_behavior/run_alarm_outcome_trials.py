"""Compare alarm timing and rule-repair outcomes on matched simulation seeds."""

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


SCENARIOS = (
    ("normal", "C00", "C00R"),
    ("type_a", "C03", "C04"),
    ("type_b", "C05", "C06"),
    ("mixed", "C07", "C08"),
)
ALARM_NAMES = ("flock_splitting", "stagnation")


def profile_alarms(actions):
    counts = {
        "splitting_only_frames": 0,
        "stagnation_only_frames": 0,
        "concurrent_frames": 0,
    }
    first = {name: None for name in ALARM_NAMES}
    last = {name: None for name in ALARM_NAMES}
    for action in actions:
        alarms = set(action["alarm"].split(";"))
        frame = action["from_state_frame"]
        for name in ALARM_NAMES:
            if name in alarms:
                if first[name] is None:
                    first[name] = frame
                last[name] = frame
        if set(ALARM_NAMES) <= alarms:
            counts["concurrent_frames"] += 1
        elif "flock_splitting" in alarms:
            counts["splitting_only_frames"] += 1
        elif "stagnation" in alarms:
            counts["stagnation_only_frames"] += 1

    has_split = first["flock_splitting"] is not None
    has_stagnation = first["stagnation"] is not None
    if counts["concurrent_frames"]:
        category = "concurrent"
    elif has_split and has_stagnation:
        category = "sequential"
    elif has_split:
        category = "splitting_only"
    elif has_stagnation:
        category = "stagnation_only"
    else:
        category = "none"
    return {
        "alarm_category": category,
        **counts,
        "first_splitting_frame": first["flock_splitting"],
        "last_splitting_frame": last["flock_splitting"],
        "first_stagnation_frame": first["stagnation"],
        "last_stagnation_frame": last["stagnation"],
    }


def summarize_pairs(rows):
    summary = {}
    for scenario, _, _ in SCENARIOS:
        pairs = {}
        for row in rows:
            if row["scenario"] == scenario:
                pairs.setdefault(row["seed"], {})[row["repair_enabled"]] = row
        categories = {}
        for pair in pairs.values():
            baseline = pair[False]
            repaired = pair[True]
            bucket = categories.setdefault(baseline["alarm_category"], {
                "runs": 0,
                "baseline_successes": 0,
                "repair_successes": 0,
                "rescued": 0,
                "regressed": 0,
                "baseline_capped_frames": [],
                "repair_capped_frames": [],
            })
            bucket["runs"] += 1
            bucket["baseline_successes"] += baseline["success"]
            bucket["repair_successes"] += repaired["success"]
            bucket["rescued"] += not baseline["success"] and repaired["success"]
            bucket["regressed"] += baseline["success"] and not repaired["success"]
            bucket["baseline_capped_frames"].append(baseline["capped_frames"])
            bucket["repair_capped_frames"].append(repaired["capped_frames"])
        for bucket in categories.values():
            bucket["baseline_mean_capped_frames"] = statistics.mean(
                bucket.pop("baseline_capped_frames")
            )
            bucket["repair_mean_capped_frames"] = statistics.mean(
                bucket.pop("repair_capped_frames")
            )
        summary[scenario] = categories
    return summary


def run_trials(first_seed=0, seed_count=20, max_steps=1000):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, baseline_name, repair_name in SCENARIOS:
            for repair_enabled, case_name in ((False, baseline_name), (True, repair_name)):
                result, _, actions = diagnose(cases[case_name], seed, max_steps)
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "case": case_name,
                    "repair_enabled": repair_enabled,
                    "success": bool(result["success_reported_by_env"]),
                    "capped_frames": result["step_calls"],
                    "final_inside_goal": result["final_inside_goal"],
                    **profile_alarms(actions),
                })
    return rows, summarize_pairs(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.first_seed < 0 or args.seed_count < 1 or args.max_steps < 1:
        parser.error("first seed must be nonnegative; counts must be positive")

    rows, summary = run_trials(args.first_seed, args.seed_count, args.max_steps)
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "alarm_outcome_trials.csv").open(
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
        "start_radius": config.SHEEP_START_RADIUS,
        "alarm_thresholds": {
            "max_spread": config.DANGER_DISPERSION_THRESHOLD,
            "mean_spread": config.DANGER_MEAN_SPREAD_THRESHOLD,
            "stagnation_frames": config.DANGER_STAGNATION_FRAMES,
            "stagnation_progress": config.DANGER_STAGNATION_THRESHOLD,
        },
        "classification_basis": "no-repair trajectory for each paired seed",
        "summary": summary,
    }
    (output / "alarm_outcome_trials.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, categories in summary.items():
        print(f"{scenario}:")
        for category, bucket in categories.items():
            print(
                f"  {category}: {bucket['runs']} runs, success "
                f"{bucket['baseline_successes']} -> {bucket['repair_successes']}, "
                f"rescued {bucket['rescued']}, regressed {bucket['regressed']}"
            )


if __name__ == "__main__":
    main()
