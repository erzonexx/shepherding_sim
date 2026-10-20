"""Compare rule-mode priority when splitting and stagnation overlap."""

import argparse
import copy
import csv
import hashlib
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment import SwarmEnv
from low_level_controller import LowLevelController, rule_mode_for_alarms
from run_experiments import CASES, make_detector, temporary_config
from tools.diagnose_failures import diagnose


SCENARIOS = (
    ("normal", "C00R"),
    ("type_a", "C04"),
    ("type_b", "C06"),
    ("stress", "C08"),
)


class GuideFirstOnConcurrent(LowLevelController):
    def _select_mode(self, alarms):
        if (
            self.settings.REPAIR_ENABLED
            and "flock_splitting" in alarms
            and "stagnation" in alarms
        ):
            return "HARD_TO_GUIDE"
        return super()._select_mode(alarms)


class CohesionFirstOnConcurrent(LowLevelController):
    def _select_mode(self, alarms):
        if not self.settings.REPAIR_ENABLED:
            return "TARGET_TRACKING"
        if "flock_splitting" in alarms:
            return "COHESION_PRIORITY"
        if "stagnation" in alarms:
            return "HARD_TO_GUIDE"
        return "TARGET_TRACKING"


class GuideFirstBoundaryGuard(LowLevelController):
    def __init__(self, margin, settings=config):
        super().__init__(settings)
        self.margin = margin

    def decide(self, observation, mode_override=None):
        if (
            mode_override is None
            and self.settings.REPAIR_ENABLED
            and "flock_splitting" in observation.alarms
            and "stagnation" in observation.alarms
        ):
            positions = observation.sheep_positions
            nearest_edge = np.min(np.minimum(
                positions, self.settings.SPACE_SIZE - positions
            ))
            if nearest_edge <= self.margin:
                return super().decide(
                    observation, mode_override="COHESION_PRIORITY"
                )
        return super().decide(observation, mode_override=mode_override)


class GuideFirstTargetBoundaryGuard(LowLevelController):
    """Use cohesion only when the current guide target is near the edge."""

    def __init__(self, margin, settings=config):
        super().__init__(settings)
        self.margin = margin

    def decide(self, observation, mode_override=None):
        if (
            mode_override is None
            and self.settings.REPAIR_ENABLED
            and "flock_splitting" in observation.alarms
            and "stagnation" in observation.alarms
        ):
            preview = LowLevelController.decide(
                copy.copy(self), observation, mode_override="HARD_TO_GUIDE"
            )
            target = observation.sheep_positions[preview.target_sheep_id]
            nearest_edge = np.min(np.minimum(
                target, self.settings.SPACE_SIZE - target
            ))
            if nearest_edge <= self.margin:
                return super().decide(
                    observation, mode_override="COHESION_PRIORITY"
                )
        return super().decide(observation, mode_override=mode_override)


VARIANTS = (
    ("cohesion_first", CohesionFirstOnConcurrent),
    ("guide_first", GuideFirstOnConcurrent),
    ("boundary_0", lambda: GuideFirstBoundaryGuard(0.0)),
    ("boundary_10", lambda: GuideFirstBoundaryGuard(10.0)),
    ("boundary_20", lambda: GuideFirstBoundaryGuard(20.0)),
    ("target_boundary_0", lambda: GuideFirstTargetBoundaryGuard(0.0)),
    ("target_boundary_5", lambda: GuideFirstTargetBoundaryGuard(5.0)),
    ("target_boundary_10", lambda: GuideFirstTargetBoundaryGuard(10.0)),
)


SEQUENCES = {
    "collect_then_guide": ("COHESION_PRIORITY", "HARD_TO_GUIDE"),
    "guide_then_collect": ("HARD_TO_GUIDE", "COHESION_PRIORITY"),
    "collect_twice": ("COHESION_PRIORITY", "COHESION_PRIORITY"),
    "guide_twice": ("HARD_TO_GUIDE", "HARD_TO_GUIDE"),
}


class RepairSequenceTrial:
    """One bounded two-stage intervention, then ordinary rule control."""

    def __init__(self, low_level, sequence, stage_limit):
        if not sequence or any(mode not in ("COHESION_PRIORITY", "HARD_TO_GUIDE")
                               for mode in sequence) or stage_limit < 1:
            raise ValueError("Use existing repair modes and a positive stage limit")
        self.low_level = low_level
        self.sequence = tuple(sequence)
        self.stage_limit = stage_limit
        self.stage = 0
        self.started = None
        self.last_target = None
        self.events = []
        self.override_frames = 0

    def finish_stage(self, frame, reason):
        self.events.append({
            "stage": self.stage, "mode": self.sequence[self.stage],
            "start_frame": self.started, "end_frame": frame,
            "executed_frames": 0 if self.started is None else frame - self.started,
            "reason": reason, "last_target": self.last_target,
        })
        self.stage += 1
        self.started = None
        self.last_target = None

    def decide(self, observation):
        while self.stage < len(self.sequence):
            mode = self.sequence[self.stage]
            alarm = "flock_splitting" if mode == "COHESION_PRIORITY" else "stagnation"
            if not self.low_level.settings.REPAIR_ENABLED or alarm not in observation.alarms:
                self.finish_stage(observation.frame,
                                  "splitting_cleared" if mode == "COHESION_PRIORITY"
                                  and self.started is not None else "alarm_not_active")
                continue
            if (mode == "HARD_TO_GUIDE" and self.last_target is not None
                    and not self.low_level.repair_target_is_eligible(observation, self.last_target)):
                self.finish_stage(observation.frame, "target_arrived")
                continue
            if self.started is not None and observation.frame - self.started >= self.stage_limit:
                self.finish_stage(observation.frame, "stage_timeout")
                continue
            if self.started is None:
                self.started = observation.frame
            decision = self.low_level.decide(observation, mode_override=mode)
            self.last_target = decision.target_sheep_id
            self.override_frames += decision.mode != rule_mode_for_alarms(
                observation.alarms, self.low_level.settings.REPAIR_ENABLED
            )
            return decision
        return self.low_level.decide(observation)


def clone_rule_state(env, detector):
    """Fork full-precision physics, detector history, and target-selection memory."""
    clone = copy.copy(env)
    for name, value in vars(env).items():
        if isinstance(value, np.ndarray):
            setattr(clone, name, value.copy())
    clone.controller = copy.copy(env.controller)
    clone.history_data = list(env.history_data)
    clone.metrics_data = list(env.metrics_data)
    clone.control_data = list(env.control_data)
    clone.decision_metadata = dict(env.decision_metadata)
    return clone, copy.deepcopy(detector)


def state_fingerprint(env, detector):
    state = {
        name: getattr(env, name).tolist() for name in
        ("sheep_pos", "sheep_vel", "dog_pos", "dog_vel", "goal_pos", "sheep_status")
    }
    state.update(frame=env.current_frame, current_mode=env.current_mode,
                 detector_history=detector.history_dist_to_goal,
                 repair_mode=env.controller.repair_target_mode,
                 repair_target=env.controller.repair_target_id,
                 repair_age=env.controller.repair_target_age)
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def recorded_frame(env):
    agents = pd.DataFrame(env.history_data,
                          columns=["Frame", "Agent_Type", "Agent_ID", "X", "Y", "Status"])
    return agents.merge(pd.DataFrame(env.metrics_data), on="Frame", how="left")


def sequence_outcome(env, success):
    data = recorded_frame(env)
    sheep = data.loc[data.Agent_Type.eq("Sheep")].sort_values(["Agent_ID", "Frame"]).copy()
    sheep["inside"] = np.linalg.norm(sheep[["X", "Y"]].to_numpy() - env.goal_pos,
                                     axis=1) < config.GOAL_RADIUS
    exits = sum(int((group.inside.shift(fill_value=False) & ~group.inside).sum())
                for _, group in sheep.groupby("Agent_ID"))
    sheep["outside"] = ((sheep.X < 0) | (sheep.Y < 0)
                        | (sheep.X > config.SPACE_SIZE) | (sheep.Y > config.SPACE_SIZE))
    metrics = pd.DataFrame(env.metrics_data)
    return {
        "success": bool(success), "capped_frames": env.current_frame,
        "danger_frames": int(metrics.is_danger.sum()),
        "max_spread": float(metrics.dispersion.max()), "goal_exits": exits,
        "outside_frames": int(sheep.groupby("Frame").outside.any().sum()),
        "peak_outside_count": int(sheep.groupby("Frame").outside.sum().max()),
    }


def continue_rule_branch(env, detector, max_steps):
    success = False
    while env.current_frame < max_steps:
        alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
        env.record_frame_metrics(metrics)
        success = env.step(alarms)
        if success:
            break
    return sequence_outcome(env, success)


def summarize_sequences(rows):
    result = {}
    for cohort in ("all", "first_dual_alarm"):
        selected = [row for row in rows if cohort == "all" or row["eligible"]]
        baseline = {row["seed"]: row for row in selected if row["variant"] == "rule"}
        variants = {}
        for name in ("rule", *SEQUENCES):
            group = [row for row in selected if row["variant"] == name]
            if not group:
                continue
            differences = [row["capped_frames"] - baseline[row["seed"]]["capped_frames"]
                           for row in group]
            variants[name] = {
                "runs": len(group), "successes": sum(row["success"] for row in group),
                "mean_capped_frames": statistics.mean(row["capped_frames"] for row in group),
                "mean_danger_frames": statistics.mean(row["danger_frames"] for row in group),
                "mean_max_spread": statistics.mean(row["max_spread"] for row in group),
                "mean_goal_exits": statistics.mean(row["goal_exits"] for row in group),
                "mean_outside_frames": statistics.mean(row["outside_frames"] for row in group),
                "mean_frame_delta": statistics.mean(differences),
                "faster": sum(delta < 0 for delta in differences),
                "equal": sum(delta == 0 for delta in differences),
                "slower": sum(delta > 0 for delta in differences),
                "rescued": sum(row["success"] and not baseline[row["seed"]]["success"]
                               for row in group),
                "regressed": sum(not row["success"] and baseline[row["seed"]]["success"]
                                 for row in group),
            }
        result[cohort] = variants
    return result


def run_sequence_diagnostic(rule_summary):
    source = pd.read_csv(rule_summary).sort_values("Seed")
    metadata = json.loads(Path(str(rule_summary) + ".metadata.json").read_text(encoding="utf-8"))
    if source.empty or source.duplicated(["Case", "Seed"]).any() or not source.Controller.eq("rule").all():
        raise ValueError("Use a nonempty rule summary with unique Case/Seed pairs")
    for name in ("dynamics.py", "detector.py", "environment.py", "low_level_controller.py",
                 "control_types.py"):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != metadata["source_sha256"][name]:
            raise ValueError(f"Replay source differs from logged baseline: {name}")
    stage_limit = int(metadata["config"]["HIGH_LEVEL_MAX_HOLD_FRAMES"])
    max_steps = int(metadata["max_steps"])
    rows, events = [], []
    settings = {key: value for key, value in metadata["config"].items()
                if hasattr(config, key) and not key.endswith("PATH")}
    with temporary_config(**settings):
        for row in source.itertuples():
            with temporary_config(NUM_ABNORMAL_A=int(row.A), NUM_ABNORMAL_B=int(row.B),
                                  NUM_ABNORMAL_C=int(row.C), REPAIR_ENABLED=bool(row.Repair),
                                  USE_FIXED_SEED=True, RANDOM_SEED=int(row.Seed)):
                env = SwarmEnv()
                detector = make_detector()
                checkpoint = None
                success = False
                while env.current_frame < max_steps:
                    before_detector = copy.deepcopy(detector) if checkpoint is None else None
                    alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
                    if checkpoint is None and {"flock_splitting", "stagnation"}.issubset(alarms):
                        checkpoint = clone_rule_state(env, before_detector)
                    env.record_frame_metrics(metrics)
                    success = env.step(alarms)
                    if success:
                        break
                baseline = sequence_outcome(env, success)
                pd.testing.assert_frame_equal(recorded_frame(env), pd.read_parquet(row.Data))
                for key, column in (("success", "Success"), ("capped_frames", "Frames"),
                                    ("danger_frames", "DangerFrames"), ("goal_exits", "GoalExits"),
                                    ("max_spread", "MaxDisp")):
                    if baseline[key] != getattr(row, column):
                        raise ValueError(f"Baseline mismatch seed {row.Seed}: {key}")
                common = {"seed": int(row.Seed), "eligible": checkpoint is not None,
                          "checkpoint_frame": None if checkpoint is None else checkpoint[0].current_frame,
                          "checkpoint_sha256": None if checkpoint is None else state_fingerprint(*checkpoint)}
                rows.append({**common, "variant": "rule", **baseline, "override_frames": 0})
                for name, sequence in SEQUENCES.items():
                    if checkpoint is None:
                        outcome, override = baseline, 0
                    else:
                        branch, branch_detector = clone_rule_state(*checkpoint)
                        assert state_fingerprint(branch, branch_detector) == common["checkpoint_sha256"]
                        trial = RepairSequenceTrial(branch.controller, sequence, stage_limit)
                        branch.controller = trial
                        outcome = continue_rule_branch(branch, branch_detector, max_steps)
                        if trial.stage < len(sequence) and trial.started is not None:
                            trial.finish_stage(branch.current_frame,
                                               "task_complete" if outcome["success"] else "run_cap")
                        override = trial.override_frames
                        events.extend({"seed": int(row.Seed), "variant": name, **event}
                                      for event in trial.events)
                    rows.append({**common, "variant": name, **outcome, "override_frames": override})
                assert checkpoint is None or state_fingerprint(*checkpoint) == common["checkpoint_sha256"]
                print(f"seed={row.Seed} checkpoint={common['checkpoint_frame']} " + " ".join(
                    f"{entry['variant']}:{entry['capped_frames']}{'' if entry['success'] else '(timeout)'}"
                    for entry in rows[-5:]), flush=True)
    return rows, events, {
        "rule_summary": str(Path(rule_summary).resolve()), "max_steps": max_steps,
        "stage_limit": stage_limit, "sequences": SEQUENCES,
        "snapshot_selection": "first concurrent alarm in each logged rule trajectory; no outcome filtering",
        "baseline_agent_metric_replays_exact": len(source),
        "api_calls": 0, "summary": summarize_sequences(rows),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in (*metadata["source_sha256"], Path(__file__).relative_to(ROOT).as_posix())},
        "interpretation": "Offline branch diagnostic, not LLM performance or an online oracle policy.",
    }


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario
            and row["variant"] == "cohesion_first"
        }
        result[scenario] = {}
        for variant, _ in VARIANTS:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["variant"] == variant
            ]
            if not group:
                continue
            result[scenario][variant] = {
                "runs": len(group),
                "successes": sum(row["success"] for row in group),
                "rescued": sum(
                    row["success"] and not baseline[row["seed"]]["success"]
                    for row in group
                ),
                "regressed": sum(
                    not row["success"] and baseline[row["seed"]]["success"]
                    for row in group
                ),
                "mean_capped_frames": statistics.mean(
                    row["capped_frames"] for row in group
                ),
                "mean_goal_exits": statistics.mean(
                    row["goal_exits"] for row in group
                ),
                "mean_peak_inside_goal": statistics.mean(
                    row["peak_inside_goal"] for row in group
                ),
                "mean_peak_max_spread": statistics.mean(
                    row["peak_max_spread"] for row in group
                ),
                "mean_outside_state_samples": statistics.mean(
                    row["outside_state_samples"] for row in group
                ),
                "mean_max_excursion": statistics.mean(
                    row["max_excursion"] for row in group
                ),
                "mean_alarm_frames": statistics.mean(
                    row["alarm_frames"] for row in group
                ),
                "runs_with_concurrent_alarm": sum(
                    row["concurrent_alarm_frames"] > 0 for row in group
                ),
                "mean_concurrent_alarm_frames": statistics.mean(
                    row["concurrent_alarm_frames"] for row in group
                ),
                "mean_concurrent_guide_frames": statistics.mean(
                    row["concurrent_guide_frames"] for row in group
                ),
            }
    return result


def run_trial(first_seed=650, seed_count=20, max_steps=1000,
              variant_names=None):
    if first_seed < 0 or seed_count < 1 or max_steps < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    selected = (
        VARIANTS[:2] if variant_names is None
        else tuple(item for item in VARIANTS if item[0] in variant_names)
    )
    if variant_names is not None and {name for name, _ in selected} != set(variant_names):
        raise ValueError("unknown concurrent-mode variant")
    if "cohesion_first" not in {name for name, _ in selected}:
        raise ValueError("include the cohesion-first baseline")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            for variant, factory in selected:
                report, states, actions = diagnose(
                    cases[case_name], seed, max_steps,
                    controller_factory=factory,
                )
                concurrent = [
                    action for action in actions
                    if "flock_splitting" in action["alarm"]
                    and "stagnation" in action["alarm"]
                ]
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "goal_exits": report["individual_goal_exits"],
                    "peak_inside_goal": report["peak_inside_goal"],
                    "peak_max_spread": max(
                        state["max_spread"] for state in states
                    ),
                    "outside_state_samples": report["outside_state_samples"],
                    "max_excursion": report["max_excursion"],
                    "alarm_frames": sum(
                        action["alarm"] != "none" for action in actions
                    ),
                    "concurrent_alarm_frames": len(concurrent),
                    "concurrent_guide_frames": sum(
                        action["mode"] == "HARD_TO_GUIDE"
                        for action in concurrent
                    ),
                })
    return rows, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=650)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", choices=[name for name, _ in VARIANTS])
    parser.add_argument("--sequence-diagnostic", action="store_true",
                        help="Replay logged rule runs and fork two-stage order diagnostics without APIs.")
    parser.add_argument("--rule-summary", type=Path,
                        default=ROOT / "logs/experiments/target_selection_50_3301_3350/rule/summary.csv")
    args = parser.parse_args()
    if args.sequence_diagnostic:
        rows, events, report = run_sequence_diagnostic(args.rule_summary)
        output = config.PROJECT_ROOT / "logs/anomaly_behavior" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(rows).to_csv(output / "repair_sequence_runs.csv", index=False)
        pd.DataFrame(events).to_csv(output / "repair_sequence_events.csv", index=False)
        (output / "repair_sequence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to: {output}")
        print(json.dumps(report["summary"], indent=2))
        return
    try:
        rows, summary = run_trial(
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
    with (output / "concurrent_mode_priority_runs.csv").open(
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
        "candidate": "HARD_TO_GUIDE rather than COHESION_PRIORITY on simultaneous alarms",
        "summary": summary,
    }
    (output / "concurrent_mode_priority_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, variants in summary.items():
        print(scenario)
        for name, result in variants.items():
            print(
                f"  {name}: {result['successes']}/{result['runs']} success, "
                f"rescued {result['rescued']}, regressed {result['regressed']}, "
                f"dual-alarm runs {result['runs_with_concurrent_alarm']}"
            )


if __name__ == "__main__":
    main()
