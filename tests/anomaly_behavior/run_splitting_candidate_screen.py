"""Screen splitting alarm candidates on untouched matched simulation seeds."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from run_experiments import CASES
from tests.anomaly_behavior.run_type_b_threshold_trials import POST_START_FRAME
from tools.diagnose_failures import diagnose


SCENARIOS = (("normal", "C00"), ("type_b", "C05"), ("mixed", "C07"))
CANDIDATES = ("current", "max_immediate", "max_5", "max_10", "max_20")


def candidate_masks(states, start_frame=POST_START_FRAME):
    masks = {name: [] for name in CANDIDATES}
    max_streak = 0
    for state in states:
        eligible = state["state_frame"] >= start_frame
        max_exceeded = state["max_spread"] > config.DANGER_DISPERSION_THRESHOLD
        mean_exceeded = state["mean_spread"] > config.DANGER_MEAN_SPREAD_THRESHOLD
        max_streak = max_streak + 1 if eligible and max_exceeded else 0
        masks["current"].append(eligible and max_exceeded and mean_exceeded)
        masks["max_immediate"].append(max_streak >= 1)
        for length in (5, 10, 20):
            masks[f"max_{length}"].append(max_streak >= length)
    return masks


def summarize_masks(seed, scenario, success, states, actions):
    masks = candidate_masks(states)
    stagnation = ["stagnation" in action["alarm"].split(";") for action in actions]
    rows = []
    for name, mask in masks.items():
        rows.append({
            "seed": seed,
            "scenario": scenario,
            "success": success,
            "candidate": name,
            "alert_frames": sum(mask),
            "alert_episodes": sum(
                active and (index == 0 or not mask[index - 1])
                for index, active in enumerate(mask)
            ),
            "first_alert_frame": next(
                (state["state_frame"] for state, active in zip(states, mask) if active),
                None,
            ),
            "concurrent_with_stagnation_frames": sum(
                active and stagnant for active, stagnant in zip(mask, stagnation)
            ),
        })
    return rows


def aggregate(rows):
    summary = {}
    for scenario, _ in SCENARIOS:
        summary[scenario] = {}
        for name in CANDIDATES:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["candidate"] == name
            ]
            summary[scenario][name] = {
                "runs": len(group),
                "runs_with_alert": sum(row["alert_frames"] > 0 for row in group),
                "successful_runs_with_alert": sum(
                    row["success"] and row["alert_frames"] > 0 for row in group
                ),
                "failed_runs_with_alert": sum(
                    not row["success"] and row["alert_frames"] > 0
                    for row in group
                ),
                "alert_frames": sum(row["alert_frames"] for row in group),
                "alert_episodes": sum(row["alert_episodes"] for row in group),
                "runs_with_concurrent_stagnation": sum(
                    row["concurrent_with_stagnation_frames"] > 0
                    for row in group
                ),
                "concurrent_stagnation_frames": sum(
                    row["concurrent_with_stagnation_frames"] for row in group
                ),
            }
    return summary


def run_screen(first_seed=20, seed_count=40, max_steps=1000):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            result, states, actions = diagnose(cases[case_name], seed, max_steps)
            rows.extend(summarize_masks(
                seed, scenario, bool(result["success_reported_by_env"]),
                states[:len(actions)], actions,
            ))
    return rows, aggregate(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.first_seed < 0 or args.seed_count < 1 or args.max_steps < 1:
        parser.error("first seed must be nonnegative; counts must be positive")
    rows, summary = run_screen(args.first_seed, args.seed_count, args.max_steps)
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "splitting_candidate_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "controller": "rule",
        "repair_enabled": False,
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "max_steps": args.max_steps,
        "post_start_frame": POST_START_FRAME,
        "max_spread_threshold": config.DANGER_DISPERSION_THRESHOLD,
        "mean_spread_threshold": config.DANGER_MEAN_SPREAD_THRESHOLD,
        "summary": summary,
    }
    (output / "splitting_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, candidates in summary.items():
        print(f"{scenario}:")
        for name, result in candidates.items():
            print(
                f"  {name}: {result['runs_with_alert']}/{result['runs']} "
                f"runs alerted, {result['successful_runs_with_alert']} "
                "successful runs alerted, "
                f"{result['runs_with_concurrent_stagnation']} runs overlap "
                "stagnation"
            )


if __name__ == "__main__":
    main()
