"""Run paired rule-controller trials for normal and Type A populations."""

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


CASE_NAMES = ("C00", "C00R", "C03", "C04")


def run_trials(first_seed=0, seed_count=20, max_steps=1000):
    if seed_count < 1 or max_steps < 1:
        raise ValueError("seed_count and max_steps must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for name in CASE_NAMES:
            result, _, _ = diagnose(cases[name], seed, max_steps)
            rows.append({
                "seed": seed,
                "case": name,
                "success": bool(result["success_reported_by_env"]),
                "capped_frames": result["step_calls"],
                "final_inside_goal": result["final_inside_goal"],
            })
    summary = {}
    for name in CASE_NAMES:
        case_rows = [row for row in rows if row["case"] == name]
        summary[name] = {
            "success_count": sum(row["success"] for row in case_rows),
            "run_count": len(case_rows),
            "mean_capped_frames": statistics.mean(
                row["capped_frames"] for row in case_rows
            ),
            "failed_seeds": [
                row["seed"] for row in case_rows if not row["success"]
            ],
        }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dog-response-factor", type=float)
    args = parser.parse_args()
    if args.first_seed < 0 or args.seed_count < 1 or args.max_steps < 1:
        parser.error("first seed must be nonnegative; seed count and max steps must be positive")
    factor = (
        config.WEIGHT_DOG_REPULSION_A_FACTOR
        if args.dog_response_factor is None else args.dog_response_factor
    )
    if not 0 < factor <= 1:
        parser.error("dog response factor must be greater than zero and at most one")

    with temporary_config(WEIGHT_DOG_REPULSION_A_FACTOR=factor):
        rows, summary = run_trials(args.first_seed, args.seed_count, args.max_steps)
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "type_a_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "controller": "rule",
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "max_steps": args.max_steps,
        "type_a_dog_response_factor": factor,
        "summary": summary,
    }
    (output / "type_a_trials.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for name, result in summary.items():
        print(
            f"{name}: {result['success_count']}/{result['run_count']} success, "
            f"mean capped frames {result['mean_capped_frames']:.1f}"
        )


if __name__ == "__main__":
    main()
