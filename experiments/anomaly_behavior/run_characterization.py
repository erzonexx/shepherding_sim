"""Run the anomaly behavior characterization and save its report."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.anomaly_behavior.behavior_characterization import (
    write_characterization, write_population_scale_trial, write_population_api_trial,
    write_repair_distance_trial,
    write_repair_feedback_trial,
    write_blind_diagnosis_trial,
    write_controlled_diagnosis_trial,
    write_mixed_observability_trial,
    write_permission_trial,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-seeds", type=int, default=1000)
    parser.add_argument("--dispersion-frames", type=int, default=60)
    trials = parser.add_mutually_exclusive_group()
    trials.add_argument("--population-scale", action="store_true")
    trials.add_argument("--population-api", action="store_true")
    trials.add_argument("--repair-distance", action="store_true")
    trials.add_argument("--repair-feedback", action="store_true")
    trials.add_argument("--blind-diagnosis", action="store_true")
    trials.add_argument("--controlled-diagnosis", action="store_true")
    trials.add_argument("--mixed-observability", action="store_true")
    trials.add_argument("--permissions", action="store_true")
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--ratios", nargs="+", type=int, choices=(30, 60, 76),
                        help="Select anomaly ratios for --population-api; default: all three.")
    args = parser.parse_args()
    if args.ratios is not None and not args.population_api:
        parser.error("--ratios requires --population-api")
    if args.blind_diagnosis or args.controlled_diagnosis or args.mixed_observability:
        trial = (write_mixed_observability_trial if args.mixed_observability else
                 write_controlled_diagnosis_trial if args.controlled_diagnosis else write_blind_diagnosis_trial)
        output, report = trial()
        print(f"Report saved to: {output}")
        print(json.dumps(report["summary"], indent=2))
        return
    if args.population_scale or args.population_api or args.repair_distance or args.repair_feedback or args.permissions:
        try:
            trial = (write_permission_trial if args.permissions else
                     write_repair_feedback_trial if args.repair_feedback else
                     write_repair_distance_trial if args.repair_distance else
                     write_population_api_trial if args.population_api else write_population_scale_trial)
            options = {"ratios": tuple(args.ratios)} if args.ratios is not None else {}
            first_seed = args.first_seed if args.first_seed is not None else (6101 if args.permissions else 5001 if args.repair_feedback else
                                                                           4801 if args.repair_distance else 4601)
            output, report = trial(first_seed=first_seed, seed_count=args.seed_count, **options)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Report saved to: {output}")
        print(json.dumps(report.get("summary", report.get("completed_levels")), indent=2))
        return
    if args.initial_seeds < 1 or args.dispersion_frames < 1:
        parser.error("Both values must be positive.")

    output, report = write_characterization(
        initial_seed_count=args.initial_seeds,
        dispersion_frames=args.dispersion_frames,
    )
    type_a = report["type_a"]
    type_b = report["type_b"]
    initialization = report["normal_initialization"]
    print(f"Report saved to: {output}")
    print(
        "Type A / Normal dog-response ratio: "
        f"{type_a['type_a_to_normal_response_ratio']:.4f}"
    )
    print(
        "Type B / Normal final focus-distance ratio: "
        f"{type_b['type_b_to_normal_focus_distance_ratio']:.4f}"
    )
    print(
        "Normal frame-zero splitting rate: "
        f"{initialization['frame_zero_splitting_rate']:.1%}"
    )
    print(f"Review flags: {report['review_flags']}")


if __name__ == "__main__":
    main()
