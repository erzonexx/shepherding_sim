"""Compare normal and Type B dispersion against each splitting threshold."""

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


POST_START_FRAME = 30


def threshold_counts(states, start_frame=POST_START_FRAME):
    counts = {"frames": 0, "max_only": 0, "mean_only": 0, "both": 0}
    for state in states:
        if state["state_frame"] < start_frame:
            continue
        counts["frames"] += 1
        max_exceeded = state["max_spread"] > config.DANGER_DISPERSION_THRESHOLD
        mean_exceeded = state["mean_spread"] > config.DANGER_MEAN_SPREAD_THRESHOLD
        if max_exceeded and mean_exceeded:
            counts["both"] += 1
        elif max_exceeded:
            counts["max_only"] += 1
        elif mean_exceeded:
            counts["mean_only"] += 1
    return counts


def compare_shared_frames(normal_states, type_b_states):
    paired = [
        (normal, type_b)
        for normal, type_b in zip(normal_states, type_b_states)
        if normal["state_frame"] >= POST_START_FRAME
    ]
    if not paired:
        return None
    return {
        "shared_post_start_frames": len(paired),
        "mean_max_spread_difference": statistics.mean(
            type_b["max_spread"] - normal["max_spread"]
            for normal, type_b in paired
        ),
        "mean_spread_difference": statistics.mean(
            type_b["mean_spread"] - normal["mean_spread"]
            for normal, type_b in paired
        ),
        "normal_split_frames": threshold_counts(
            (normal for normal, _ in paired), start_frame=0
        )["both"],
        "type_b_split_frames": threshold_counts(
            (type_b for _, type_b in paired), start_frame=0
        )["both"],
    }


def summarize(rows, pairs):
    summary = {}
    for variant in ("normal", "type_b"):
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "runs": len(selected),
            "successes": sum(row["success"] for row in selected),
            "initial_split_runs": sum(row["initial_split"] for row in selected),
            "post_start_split_runs": sum(row["post_start_both_frames"] > 0 for row in selected),
            "post_start_max_only_runs": sum(
                row["post_start_max_only_frames"] > 0 for row in selected
            ),
            "post_start_mean_only_runs": sum(
                row["post_start_mean_only_frames"] > 0 for row in selected
            ),
            "failures_without_post_start_split": sum(
                not row["success"] and row["post_start_both_frames"] == 0
                for row in selected
            ),
        }
    comparable = [pair for pair in pairs if pair["shared_post_start_frames"]]
    summary["paired"] = {
        "comparable_runs": len(comparable),
        "type_b_higher_mean_spread_runs": sum(
            pair["mean_spread_difference"] > 0 for pair in comparable
        ),
        "type_b_higher_max_spread_runs": sum(
            pair["mean_max_spread_difference"] > 0 for pair in comparable
        ),
        "mean_of_mean_spread_differences": statistics.mean(
            pair["mean_spread_difference"] for pair in comparable
        ) if comparable else None,
        "mean_of_max_spread_differences": statistics.mean(
            pair["mean_max_spread_difference"] for pair in comparable
        ) if comparable else None,
        "type_b_more_split_frames_runs": sum(
            pair["type_b_split_frames"] > pair["normal_split_frames"]
            for pair in comparable
        ),
    }
    return summary


def run_trials(first_seed=0, seed_count=20, max_steps=1000):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    rows = []
    pairs = []
    for seed in range(first_seed, first_seed + seed_count):
        trajectories = {}
        for variant, case_name in (("normal", "C00"), ("type_b", "C05")):
            result, states, actions = diagnose(cases[case_name], seed, max_steps)
            # Only states with an executed action have a corresponding alarm sample.
            observed_states = states[:len(actions)]
            counts = threshold_counts(observed_states)
            initial = threshold_counts(observed_states[:1], start_frame=0)
            rows.append({
                "seed": seed,
                "variant": variant,
                "success": bool(result["success_reported_by_env"]),
                "capped_frames": result["step_calls"],
                "initial_split": initial["both"] > 0,
                "post_start_observed_frames": counts["frames"],
                "post_start_max_only_frames": counts["max_only"],
                "post_start_mean_only_frames": counts["mean_only"],
                "post_start_both_frames": counts["both"],
            })
            trajectories[variant] = observed_states
        paired = compare_shared_frames(
            trajectories["normal"], trajectories["type_b"]
        )
        pairs.append({"seed": seed, **(paired or {
            "shared_post_start_frames": 0,
            "mean_max_spread_difference": None,
            "mean_spread_difference": None,
            "normal_split_frames": None,
            "type_b_split_frames": None,
        })})
    return rows, pairs, summarize(rows, pairs)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.first_seed < 0 or args.seed_count < 1 or args.max_steps < 1:
        parser.error("first seed must be nonnegative; counts must be positive")
    rows, pairs, summary = run_trials(
        args.first_seed, args.seed_count, args.max_steps
    )
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "type_b_threshold_runs.csv", rows)
    write_csv(output / "type_b_threshold_pairs.csv", pairs)
    report = {
        "controller": "rule",
        "repair_enabled": False,
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "max_steps": args.max_steps,
        "post_start_frame": POST_START_FRAME,
        "start_radius": config.SHEEP_START_RADIUS,
        "max_spread_threshold": config.DANGER_DISPERSION_THRESHOLD,
        "mean_spread_threshold": config.DANGER_MEAN_SPREAD_THRESHOLD,
        "summary": summary,
    }
    (output / "type_b_threshold_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
