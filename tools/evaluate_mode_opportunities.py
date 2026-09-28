"""Measure one fixed 30-frame intervention at the first dual alarm.

This is a counterfactual diagnostic, not an online policy or an LLM evaluation.
The intervention may continue after either alarm clears; it has no API latency.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment import SwarmEnv
from low_level_controller import LowLevelController, rule_mode_for_alarms
from run_experiments import make_detector, temporary_config


MODES = ("RULE", "TARGET_TRACKING", "COHESION_PRIORITY", "HARD_TO_GUIDE")


class FirstDualAlarmController:
    def __init__(self, selected_mode, hold_frames, delay_frames=0,
                 require_dual_at_start=False):
        self.low_level = LowLevelController()
        self.selected_mode = selected_mode
        self.hold_frames = hold_frames
        self.delay_frames = delay_frames
        self.require_dual_at_start = require_dual_at_start
        self.trigger_frame = None
        self.intervention_started_frame = None
        self.start_checked = False
        self.skipped = False
        self.override_frames = 0

    def decide(self, observation):
        if self.trigger_frame is None and {
            "flock_splitting", "stagnation"
        }.issubset(observation.alarms):
            self.trigger_frame = observation.frame
        start_frame = (
            None if self.trigger_frame is None
            else self.trigger_frame + self.delay_frames
        )
        if (
            self.selected_mode != "RULE"
            and start_frame is not None
            and observation.frame >= start_frame
            and not self.start_checked
        ):
            self.start_checked = True
            self.skipped = self.require_dual_at_start and not {
                "flock_splitting", "stagnation"
            }.issubset(observation.alarms)
            if not self.skipped:
                self.intervention_started_frame = observation.frame
        active = (
            self.selected_mode != "RULE"
            and self.intervention_started_frame is not None
            and observation.frame < self.intervention_started_frame + self.hold_frames
        )
        if active:
            rule_mode = rule_mode_for_alarms(observation.alarms, True)
            self.override_frames += self.selected_mode != rule_mode
        return self.low_level.decide(
            observation,
            mode_override=self.selected_mode if active else None,
        )


def run_trial(seed, mode, hold_frames, max_steps, delay_frames=0,
              require_dual_at_start=False):
    with temporary_config(
        USE_FIXED_SEED=True,
        RANDOM_SEED=seed,
        MAX_STEPS=max_steps,
        NUM_ABNORMAL_A=3,
        NUM_ABNORMAL_B=3,
        REPAIR_ENABLED=True,
    ):
        controller = FirstDualAlarmController(
            mode, hold_frames, delay_frames, require_dual_at_start
        )
        env = SwarmEnv(controller=controller)
        detector = make_detector()
        success = False
        for step in range(max_steps):
            alarms, _, _ = detector.detect(env.sheep_pos, env.goal_pos)
            success = env.step(alarms)
            if success:
                break
        return {
            "Seed": seed,
            "Mode": mode,
            "Success": success,
            "Frames": step + 1,
            "FirstDualAlarmFrame": controller.trigger_frame,
            "InterventionStartedFrame": controller.intervention_started_frame,
            "InterventionSkipped": controller.skipped,
            "OverrideFrames": controller.override_frames,
            "FinalCentroidDistance": round(
                float(np.linalg.norm(env.sheep_pos.mean(axis=0) - env.goal_pos)), 3
            ),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=2301)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument(
        "--timing-study", action="store_true",
        help="Compare immediate 30-frame cohesion with 30-frame delayed "
             "cohesion held for 30 or 15 frames, gated by a continuing dual alarm.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/experiments/first_dual_mode_opportunities.csv"),
    )
    args = parser.parse_args(argv)
    if args.runs < 1 or args.max_steps < 1 or args.hold_frames < 1:
        parser.error("--runs, --max-steps, and --hold-frames must be positive")
    if not 0 <= args.start_seed < 2**32 or args.start_seed + args.runs > 2**32:
        parser.error("seed range must stay within [0, 2**32)")
    arms = (
        (
            ("RULE", "RULE", args.hold_frames, 0, False),
            ("COHESION_NOW_30", "COHESION_PRIORITY", 30, 0, False),
            ("COHESION_WAIT_30_HOLD_30", "COHESION_PRIORITY", 30, 30, True),
            ("COHESION_WAIT_30_HOLD_15", "COHESION_PRIORITY", 15, 30, True),
        )
        if args.timing_study else
        tuple((mode, mode, args.hold_frames, 0, False) for mode in MODES)
    )
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.runs):
        for arm, mode, hold, delay, require_dual in arms:
            row = run_trial(
                seed, mode, hold, args.max_steps, delay, require_dual
            )
            rows.append({"Arm": arm, **row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for arm, _, _, _, _ in arms:
        group = [row for row in rows if row["Arm"] == arm]
        print(
            f"{arm}: success={sum(row['Success'] for row in group)}/{len(group)}, "
            f"mean capped frames={np.mean([row['Frames'] for row in group]):.1f}, "
            f"true override frames={sum(row['OverrideFrames'] for row in group)}, "
            f"skipped={sum(row['InterventionSkipped'] for row in group)}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
