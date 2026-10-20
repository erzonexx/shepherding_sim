"""Audit whether concurrent alarms cover rule-controller failures."""

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
    ("normal", "C00R"),
    ("type_a", "C04"),
    ("type_b", "C06"),
    ("stress", "C08"),
)


def alarm_category(action):
    alarms = set(action["alarm"].split(";"))
    splitting = "flock_splitting" in alarms
    stagnation = "stagnation" in alarms
    if splitting and stagnation:
        return "concurrent"
    if splitting:
        return "splitting_only"
    if stagnation:
        return "stagnation_only"
    return "none"


def concurrent_episode_lengths(categories):
    lengths = []
    current = 0
    for category in categories:
        if category == "concurrent":
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def summarize_run(report, states, actions):
    categories = [alarm_category(action) for action in actions]
    episodes = concurrent_episode_lengths(categories)
    first_concurrent = next(
        (index for index, category in enumerate(categories)
         if category == "concurrent"),
        None,
    )
    return {
        "seed": report["seed"],
        "success": report["success_reported_by_env"],
        "frames": len(actions),
        "none_frames": categories.count("none"),
        "splitting_only_frames": categories.count("splitting_only"),
        "stagnation_only_frames": categories.count("stagnation_only"),
        "concurrent_frames": categories.count("concurrent"),
        "concurrent_episodes": len(episodes),
        "longest_concurrent_episode": max(episodes, default=0),
        "first_concurrent_frame": first_concurrent,
        "last_concurrent_frame": (
            len(categories) - 1 - categories[::-1].index("concurrent")
            if first_concurrent is not None else None
        ),
        "peak_inside_goal": report["peak_inside_goal"],
        "final_inside_goal": report["final_inside_goal"],
        "max_spread": max(state["max_spread"] for state in states),
    }


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario]
        if not group:
            continue
        failures = [row for row in group if not row["success"]]
        with_dual = [row for row in group if row["concurrent_frames"] > 0]
        failure_with_dual = [
            row for row in failures if row["concurrent_frames"] > 0
        ]
        result[scenario] = {
            "runs": len(group),
            "successes": len(group) - len(failures),
            "failures": len(failures),
            "runs_with_concurrent_alarm": len(with_dual),
            "runs_with_3_consecutive_concurrent_frames": sum(
                row["longest_concurrent_episode"] >= 3 for row in group
            ),
            "runs_with_10_consecutive_concurrent_frames": sum(
                row["longest_concurrent_episode"] >= 10 for row in group
            ),
            "failures_with_concurrent_alarm": len(failure_with_dual),
            "failures_with_3_consecutive_concurrent_frames": sum(
                row["longest_concurrent_episode"] >= 3 for row in failures
            ),
            "failures_with_10_consecutive_concurrent_frames": sum(
                row["longest_concurrent_episode"] >= 10 for row in failures
            ),
            "failures_without_concurrent_alarm": len(failures) - len(failure_with_dual),
            "failures_with_at_least_30_concurrent_frames": sum(
                row["concurrent_frames"] >= 30 for row in failures
            ),
            "mean_concurrent_frames_in_dual_alarm_runs": (
                statistics.mean(row["concurrent_frames"] for row in with_dual)
                if with_dual else None
            ),
            "median_first_concurrent_frame_in_failed_dual_alarm_runs": (
                statistics.median(
                    row["first_concurrent_frame"] for row in failure_with_dual
                ) if failure_with_dual else None
            ),
            "mean_alarm_frame_counts": {
                category: statistics.mean(row[f"{category}_frames"] for row in group)
                for category in ("none", "splitting_only", "stagnation_only", "concurrent")
            },
        }
    return result


def run_audit(first_seed=1010, seed_count=80, max_steps=2000):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            report, states, actions = diagnose(cases[case_name], seed, max_steps)
            rows.append({
                "scenario": scenario,
                **summarize_run(report, states, actions),
            })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=1010)
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
    with (output / "alarm_coverage_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "alarm_coverage_report.json").write_text(
        json.dumps({
            "controller": "rule",
            "first_seed": args.first_seed,
            "seed_count": args.seed_count,
            "max_steps": args.max_steps,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Report saved to: {output}")
    for scenario, metrics in summary.items():
        print(
            f"{scenario}: {metrics['successes']}/{metrics['runs']} success; "
            f"failures with/without concurrent alarm "
            f"{metrics['failures_with_concurrent_alarm']}/"
            f"{metrics['failures_without_concurrent_alarm']}"
        )


if __name__ == "__main__":
    main()
