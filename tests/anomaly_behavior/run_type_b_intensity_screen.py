"""Paired one-factor Type B behavior screen; never changes production defaults."""

import argparse
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
from dynamics import advance_sheep
from environment import SwarmEnv
from run_experiments import CASES, temporary_config
from tests.anomaly_behavior.run_type_b_threshold_trials import (
    POST_START_FRAME,
    compare_shared_frames,
    threshold_counts,
)
from tools.diagnose_failures import diagnose


VARIANTS = (
    ("current", {}),
    ("cohesion_0075", {"WEIGHT_COHESION_B_FACTOR": 0.075}),
    ("cohesion_030", {"WEIGHT_COHESION_B_FACTOR": 0.30}),
    ("cohesion_050", {"WEIGHT_COHESION_B_FACTOR": 0.50}),
    ("separation_100", {"WEIGHT_SEPARATION_B_FACTOR": 1.0}),
    ("separation_200", {"WEIGHT_SEPARATION_B_FACTOR": 2.0}),
    ("radius_3", {"SEPARATION_RADIUS_B": 3.0}),
    ("radius_9", {"SEPARATION_RADIUS_B": 9.0}),
)


def open_loop(seed, num_b, overrides, frames=200):
    """Measure flock spread with the dog beyond sensing range."""
    with temporary_config(
        USE_FIXED_SEED=True, RANDOM_SEED=seed,
        NUM_ABNORMAL_A=0, NUM_ABNORMAL_B=num_b, **overrides,
    ):
        env = SwarmEnv()
        positions = env.sheep_pos.copy()
        velocities = env.sheep_vel.copy()
        statuses = env.sheep_status.copy()
        goal = env.goal_pos.copy()
        dog = np.array([-1e6, -1e6])
        max_spreads = []
        mean_spreads = []
        for _ in range(frames):
            positions, velocities = advance_sheep(
                positions, velocities, statuses, dog, goal, config
            )
            spread = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
            max_spreads.append(float(spread.max()))
            mean_spreads.append(float(spread.mean()))
        env.close()
    return {
        "open_loop_final_max_spread": max_spreads[-1],
        "open_loop_final_mean_spread": mean_spreads[-1],
        "open_loop_mean_max_spread": statistics.mean(
            max_spreads[POST_START_FRAME - 1:]
        ),
        "open_loop_mean_spread": statistics.mean(
            mean_spreads[POST_START_FRAME - 1:]
        ),
    }


def summarize(rows):
    def mean_or_none(values):
        values = list(values)
        return statistics.mean(values) if values else None

    result = {}
    for repair in (False, True):
        group = [row for row in rows if row["repair"] == repair]
        current = {
            row["seed"]: row for row in group if row["variant"] == "current"
        }
        by_variant = {}
        for variant in ("normal", *(name for name, _ in VARIANTS)):
            selected = [row for row in group if row["variant"] == variant]
            if not selected:
                continue
            by_variant[variant] = {
                "runs": len(selected),
                "successes": sum(row["success"] for row in selected),
                "rescued_vs_current": sum(
                    row["success"] and not current[row["seed"]]["success"]
                    for row in selected
                ) if variant != "normal" else None,
                "regressed_vs_current": sum(
                    not row["success"] and current[row["seed"]]["success"]
                    for row in selected
                ) if variant != "normal" else None,
                "mean_capped_frames": statistics.mean(
                    row["capped_frames"] for row in selected
                ),
                "post_start_splitting_runs": sum(
                    row["post_start_split_frames"] > 0 for row in selected
                ),
                "mean_paired_max_spread_delta": mean_or_none(
                    row["paired_max_spread_delta"] for row in selected
                    if row["paired_max_spread_delta"] is not None
                ),
                "mean_paired_mean_spread_delta": mean_or_none(
                    row["paired_mean_spread_delta"] for row in selected
                    if row["paired_mean_spread_delta"] is not None
                ),
                "mean_open_loop_max_spread_delta": statistics.mean(
                    row["open_loop_max_spread_delta"] for row in selected
                ),
                "mean_open_loop_mean_spread_delta": statistics.mean(
                    row["open_loop_mean_spread_delta"] for row in selected
                ),
            }
        result["repair" if repair else "no_repair"] = by_variant
    return result


def run_screen(first_seed=270, seed_count=20, max_steps=1000,
               variant_names=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected = (
        VARIANTS if variant_names is None
        else tuple(item for item in VARIANTS if item[0] in variant_names)
    )
    if variant_names is not None and {name for name, _ in selected} != set(variant_names):
        raise ValueError("unknown Type B intensity variant")
    if "current" not in {name for name, _ in selected}:
        raise ValueError("include the current baseline")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        normal_open = open_loop(seed, num_b=0, overrides={})
        open_by_variant = {
            name: open_loop(seed, num_b=4, overrides=overrides)
            for name, overrides in selected
        }
        for repair in (False, True):
            normal_case = cases["C00R" if repair else "C00"]
            normal_report, normal_states, normal_actions = diagnose(
                normal_case, seed, max_steps
            )
            normal_observed = normal_states[:len(normal_actions)]
            normal_counts = threshold_counts(normal_observed)
            rows.append({
                "seed": seed, "variant": "normal", "repair": repair,
                "success": bool(normal_report["success_reported_by_env"]),
                "capped_frames": normal_report["step_calls"],
                "peak_inside_goal": normal_report["peak_inside_goal"],
                "post_start_split_frames": normal_counts["both"],
                "paired_max_spread_delta": 0.0,
                "paired_mean_spread_delta": 0.0,
                "open_loop_max_spread_delta": 0.0,
                "open_loop_mean_spread_delta": 0.0,
            })
            for name, overrides in selected:
                with temporary_config(**overrides):
                    report, states, actions = diagnose(
                        cases["C06" if repair else "C05"], seed, max_steps
                    )
                observed = states[:len(actions)]
                paired = compare_shared_frames(normal_observed, observed)
                counts = threshold_counts(observed)
                b_open = open_by_variant[name]
                rows.append({
                    "seed": seed, "variant": name, "repair": repair,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "peak_inside_goal": report["peak_inside_goal"],
                    "post_start_split_frames": counts["both"],
                    "paired_max_spread_delta": (
                        paired["mean_max_spread_difference"] if paired else None
                    ),
                    "paired_mean_spread_delta": (
                        paired["mean_spread_difference"] if paired else None
                    ),
                    "open_loop_max_spread_delta": (
                        b_open["open_loop_mean_max_spread"]
                        - normal_open["open_loop_mean_max_spread"]
                    ),
                    "open_loop_mean_spread_delta": (
                        b_open["open_loop_mean_spread"]
                        - normal_open["open_loop_mean_spread"]
                    ),
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=270)
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
    with (output / "type_b_intensity_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "controller": "rule",
        "dog_free_open_loop_frames": 200,
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "max_steps": args.max_steps,
        "variants": {name: values for name, values in selected_variants(args.variants)},
        "baseline_type_b_parameters": {
            "WEIGHT_COHESION_B_FACTOR": config.WEIGHT_COHESION_B_FACTOR,
            "WEIGHT_SEPARATION_B_FACTOR": config.WEIGHT_SEPARATION_B_FACTOR,
            "SEPARATION_RADIUS_B": config.SEPARATION_RADIUS_B,
        },
        "summary": summary,
    }
    (output / "type_b_intensity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for architecture, variants in summary.items():
        print(architecture)
        for name, result in variants.items():
            paired = result["mean_paired_max_spread_delta"]
            paired_text = "n/a" if paired is None else f"{paired:.1f}"
            print(
                f"  {name}: {result['successes']}/{result['runs']} success, "
                f"open-loop max-spread delta "
                f"{result['mean_open_loop_max_spread_delta']:.1f}, "
                f"paired max-spread delta {paired_text}"
            )


def selected_variants(names):
    return VARIANTS if names is None else tuple(
        item for item in VARIANTS if item[0] in names
    )


if __name__ == "__main__":
    main()
