"""Save matched normal, Type B, and mixed-stress alarm diagnostics."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from experiments.anomaly_behavior.behavior_characterization import (
    splitting_trajectory_screen,
    stress_alarm_screen,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-seeds", type=int, default=100)
    parser.add_argument("--trajectory-frames", type=int, default=60)
    parser.add_argument("--stress-seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if min(
        args.trajectory_seeds,
        args.trajectory_frames,
        args.stress_seeds,
        args.max_steps,
    ) <= 0:
        parser.error("all numeric arguments must be positive")

    report = {
        "start_radius": config.SHEEP_START_RADIUS,
        "trajectory_seeds": args.trajectory_seeds,
        "trajectory_frames": args.trajectory_frames,
        "stress_seeds": args.stress_seeds,
        "max_steps": args.max_steps,
        "normal_vs_type_b": splitting_trajectory_screen(
            args.trajectory_seeds,
            args.trajectory_frames,
        ),
        "mixed_stress": stress_alarm_screen(
            args.stress_seeds,
            args.max_steps,
        ),
    }
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "detector_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    print("Normal vs Type B:", report["normal_vs_type_b"])
    stress = report["mixed_stress"]
    print(
        "Mixed stress: splitting in "
        f"{stress['runs_with_splitting']}/{stress['run_count']} runs, "
        "stagnation in "
        f"{stress['runs_with_stagnation']}/{stress['run_count']} runs, "
        "concurrent alarms in "
        f"{stress['runs_with_concurrent_alarms']}/{stress['run_count']} runs"
    )


if __name__ == "__main__":
    main()
