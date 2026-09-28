"""Run the anomaly behavior characterization and save its report."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.anomaly_behavior.behavior_characterization import write_characterization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-seeds", type=int, default=1000)
    parser.add_argument("--dispersion-frames", type=int, default=60)
    args = parser.parse_args()
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
