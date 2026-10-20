"""Test a conservative, local forecast choice on simultaneous alarms."""

import argparse
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
from control_types import ControlObservation
from environment import SwarmEnv
from high_level_controller import HighLevelController
from low_level_controller import LowLevelController
from mode_forecast import forecast_modes
from run_experiments import CASES, make_detector, temporary_config
from experiments.anomaly_behavior.run_concurrent_mode_priority_trial import (
    RepairSequenceTrial, SEQUENCES, clone_rule_state, continue_rule_branch,
    sequence_outcome, state_fingerprint,
)
from tools.diagnose_failures import diagnose


SCENARIOS = (
    ("normal", "C00R"),
    ("type_a", "C04"),
    ("type_b", "C06"),
    ("stress", "C08"),
)


class ForecastDominanceController(LowLevelController):
    """Override guide-first only when cohesion's forecast strictly dominates."""

    def __init__(self, review_period=120, settings=config):
        super().__init__(settings)
        self.review_period = review_period
        self.next_review_frame = 0
        self.override_mode = None
        self.previous_concurrent = False
        self.forecast_calls = 0

    def decide(self, observation, mode_override=None):
        concurrent = (
            self.settings.REPAIR_ENABLED
            and "flock_splitting" in observation.alarms
            and "stagnation" in observation.alarms
        )
        if mode_override is not None or not concurrent:
            self.override_mode = None
            self.previous_concurrent = False
            return super().decide(observation, mode_override=mode_override)

        if not self.previous_concurrent or observation.frame >= self.next_review_frame:
            forecasts = forecast_modes(observation, self.settings)
            self.forecast_calls += 1
            self.override_mode = (
                "COHESION_PRIORITY"
                if HighLevelController._forecast_dominates(
                    forecasts["COHESION_PRIORITY"],
                    forecasts["HARD_TO_GUIDE"],
                ) else None
            )
            self.next_review_frame = observation.frame + self.review_period
        self.previous_concurrent = True
        return super().decide(observation, mode_override=self.override_mode)


def summarize(rows):
    result = {}
    for scenario, _ in SCENARIOS:
        baseline = {
            row["seed"]: row for row in rows
            if row["scenario"] == scenario and row["variant"] == "guide_first"
        }
        result[scenario] = {}
        for variant in ("guide_first", "forecast_dominance"):
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
                "mean_outside_state_samples": statistics.mean(
                    row["outside_state_samples"] for row in group
                ),
                "mean_forecast_calls": statistics.mean(
                    row["forecast_calls"] for row in group
                ),
                "mean_concurrent_cohesion_frames": statistics.mean(
                    row["concurrent_cohesion_frames"] for row in group
                ),
            }
    return result


def run_trial(first_seed=1320, seed_count=20, max_steps=2000,
              review_period=120):
    if first_seed < 0 or min(seed_count, max_steps, review_period) < 1:
        raise ValueError("first seed must be nonnegative; counts must be positive")
    cases = {case["case"]: case for case in CASES}
    cases["C00R"] = {**cases["C00"], "case": "C00R", "repair_enabled": True}
    rows = []
    for seed in range(first_seed, first_seed + seed_count):
        for scenario, case_name in SCENARIOS:
            for variant in ("guide_first", "forecast_dominance"):
                controller = (
                    LowLevelController() if variant == "guide_first"
                    else ForecastDominanceController(review_period)
                )
                report, _, actions = diagnose(
                    cases[case_name], seed, max_steps,
                    controller_factory=lambda: controller,
                )
                rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant,
                    "success": bool(report["success_reported_by_env"]),
                    "capped_frames": report["step_calls"],
                    "outside_state_samples": report["outside_state_samples"],
                    "max_excursion": report["max_excursion"],
                    "forecast_calls": getattr(controller, "forecast_calls", 0),
                    "concurrent_cohesion_frames": sum(
                        action["mode"] == "COHESION_PRIORITY"
                        and "flock_splitting" in action["alarm"]
                        and "stagnation" in action["alarm"]
                        for action in actions
                    ),
                })
    return rows, summarize(rows)


# Only these causal features may be screened; seed and eventual outcomes are not inputs.
SELECTOR_FEATURES = (
    "max_spread", "mean_spread", "centroid_distance", "inside_goal",
    "dog_to_centroid", "boundary_margin", "progress_advantage",
    "spread_advantage", "inside_advantage", "splitting_advantage",
)


def sequence_features(env, alarms):
    observation = ControlObservation(
        sheep_positions=env.sheep_pos.copy(), dog_position=env.dog_pos.copy(),
        goal_position=env.goal_pos.copy(), alarms=tuple(alarms),
        frame=env.current_frame, sheep_statuses=tuple(env.sheep_status.tolist()),
        current_mode=env.current_mode, sheep_velocities=env.sheep_vel.copy(),
        dog_velocity=env.dog_vel.copy(),
    )
    center = observation.sheep_positions.mean(axis=0)
    spread = np.linalg.norm(observation.sheep_positions - center, axis=1)
    # Preserve held-target memory in forecasts, without advancing the real controller.
    forecasts = forecast_modes(observation, config, env.controller)
    collect, guide = forecasts["COHESION_PRIORITY"], forecasts["HARD_TO_GUIDE"]
    return {
        "max_spread": float(spread.max()), "mean_spread": float(spread.mean()),
        "centroid_distance": float(np.linalg.norm(center - env.goal_pos)),
        "inside_goal": int(np.sum(np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1)
                                  < config.GOAL_RADIUS)),
        "dog_to_centroid": float(np.linalg.norm(env.dog_pos - center)),
        "boundary_margin": float(np.minimum(env.sheep_pos, config.SPACE_SIZE - env.sheep_pos).min()),
        "progress_advantage": collect["centroid_progress"] - guide["centroid_progress"],
        "spread_advantage": guide["final_max_spread"] - collect["final_max_spread"],
        "inside_advantage": collect["inside_goal_change"] - guide["inside_goal_change"],
        "splitting_advantage": guide["predicted_splitting_frames"] - collect["predicted_splitting_frames"],
        "forecast_dominates": HighLevelController._forecast_dominates(collect, guide),
        "forecasts": forecasts,
    }


def selector_collects(row, selector):
    if not row["eligible"] or selector["kind"] == "rule":
        return False
    if selector["kind"] == "dominance":
        return bool(row["forecast_dominates"])
    value = row[selector["feature"]]
    return value <= selector["threshold"] if selector["side"] == "le" else value > selector["threshold"]


def selector_summary(rows, selector):
    selected = [row["collect"] if selector_collects(row, selector) else row["rule"] for row in rows]
    deltas = [outcome["capped_frames"] - row["rule"]["capped_frames"]
              for row, outcome in zip(rows, selected)]
    return {
        "runs": len(rows), "eligible": sum(row["eligible"] for row in rows),
        "interventions": sum(selector_collects(row, selector) for row in rows),
        "successes": sum(outcome["success"] for outcome in selected),
        "mean_capped_frames": statistics.mean(outcome["capped_frames"] for outcome in selected),
        "mean_danger_frames": statistics.mean(outcome["danger_frames"] for outcome in selected),
        "mean_max_spread": statistics.mean(outcome["max_spread"] for outcome in selected),
        "mean_goal_exits": statistics.mean(outcome["goal_exits"] for outcome in selected),
        "mean_frame_delta": statistics.mean(deltas),
        "faster": sum(delta < 0 for delta in deltas), "slower": sum(delta > 0 for delta in deltas),
        "rescued": sum(outcome["success"] and not row["rule"]["success"]
                       for row, outcome in zip(rows, selected)),
        "regressed": sum(not outcome["success"] and row["rule"]["success"]
                         for row, outcome in zip(rows, selected)),
    }


def fit_sequence_selector(rows):
    """Screen a single threshold on exploration data only; no nested decision tree."""
    candidates = [{"kind": "rule"}, {"kind": "dominance"}]
    eligible = [row for row in rows if row["eligible"]]
    for feature in SELECTOR_FEATURES:
        values = sorted({row[feature] for row in eligible})
        for left, right in zip(values, values[1:]):
            threshold = left + (right - left) / 2
            for side in ("le", "gt"):
                candidate = {"kind": "threshold", "feature": feature,
                             "threshold": threshold, "side": side}
                count = sum(selector_collects(row, candidate) for row in eligible)
                if min(count, len(eligible) - count) >= 5:
                    candidates.append(candidate)
    screening = [{"selector": candidate, "metrics": selector_summary(rows, candidate)}
                 for candidate in candidates]
    # Success count takes precedence over completion time. Exact ties keep rules.
    best = max(screening, key=lambda item: (item["metrics"]["successes"],
                                           -item["metrics"]["mean_capped_frames"]))
    return best["selector"], screening


def replay_sequence_features(seed, max_steps, stage_limit, population, expected=None):
    with temporary_config(USE_FIXED_SEED=True, RANDOM_SEED=seed, CONTROLLER_KIND="rule",
                          NUM_ABNORMAL_A=population[0], NUM_ABNORMAL_B=population[1],
                          NUM_ABNORMAL_C=population[2], REPAIR_ENABLED=True):
        env, detector = SwarmEnv(), make_detector()
        checkpoint, features = None, {}
        success = False
        while env.current_frame < max_steps:
            alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
            if checkpoint is None and {"flock_splitting", "stagnation"}.issubset(alarms):
                # This detector has consumed the current sample; retain it and execute
                # this same alarm once before the branch's next detector update.
                checkpoint = clone_rule_state(env, detector)
                fingerprint = state_fingerprint(*checkpoint)
                features = sequence_features(env, alarms)
                assert state_fingerprint(*checkpoint) == fingerprint
                first_metrics, first_alarms = metrics.copy(), tuple(alarms)
            env.record_frame_metrics(metrics)
            success = env.step(alarms)
            if success:
                break
        baseline = sequence_outcome(env, success)
        common = {"seed": seed, "eligible": checkpoint is not None,
                  "checkpoint_frame": None if checkpoint is None else checkpoint[0].current_frame,
                  **features}
        if expected is not None:
            for variant, outcome in (("rule", baseline),):
                logged = expected[variant]
                for key, value in outcome.items():
                    if isinstance(value, float):
                        matches = np.isclose(value, logged[key], rtol=0, atol=1e-10)
                    else:
                        matches = value == logged[key]
                    if not matches:
                        raise ValueError(f"Exploration replay mismatch seed {seed}: {key}")
            if common["checkpoint_frame"] != expected["rule"]["checkpoint_frame"]:
                if checkpoint is not None or not pd.isna(expected["rule"]["checkpoint_frame"]):
                    raise ValueError(f"Checkpoint mismatch seed {seed}")
            collect = {key: expected["collect_then_guide"][key] for key in baseline}
        elif checkpoint is None:
            collect = baseline.copy()
        else:
            branch, branch_detector = clone_rule_state(*checkpoint)
            trial = RepairSequenceTrial(branch.controller, SEQUENCES["collect_then_guide"], stage_limit)
            branch.controller = trial
            branch.record_frame_metrics(first_metrics)
            branch_success = branch.step(first_alarms)
            collect = (sequence_outcome(branch, True) if branch_success else
                       continue_rule_branch(branch, branch_detector, max_steps))
            assert state_fingerprint(*checkpoint) == fingerprint
        print(f"seed={seed} checkpoint={common['checkpoint_frame']} "
              f"rule={baseline['capped_frames']} collect={collect['capped_frames']}", flush=True)
        return {**common, "rule": baseline, "collect": collect}


def run_sequence_selector_diagnostic(sequence_runs, output):
    sequence_runs = Path(sequence_runs)
    source_report = json.loads(sequence_runs.with_name("repair_sequence_report.json").read_text(encoding="utf-8"))
    metadata = json.loads(Path(source_report["rule_summary"] + ".metadata.json").read_text(encoding="utf-8"))
    for name, digest in source_report["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Exploration source changed: {name}")
    source = pd.read_csv(sequence_runs)
    if source.duplicated(["seed", "variant"]).any():
        raise ValueError("Exploration rows must have unique seed/variant pairs")
    settings = {key: value for key, value in metadata["config"].items()
                if hasattr(config, key) and not key.endswith("PATH")}
    population = tuple(metadata["case_settings"][0][f"num_abnormal_{kind}"] for kind in "abc")
    max_steps, stage_limit = source_report["max_steps"], source_report["stage_limit"]
    with temporary_config(**settings):
        exploration = []
        for seed, group in source.groupby("seed", sort=True):
            expected = {row["variant"]: row for row in group.to_dict("records")}
            exploration.append(replay_sequence_features(int(seed), max_steps, stage_limit, population, expected))
        selector, screening = fit_sequence_selector(exploration)
        protocol = {
            "selector": selector, "exploration_seeds": [row["seed"] for row in exploration],
            "validation_seeds": list(range(4401, 4451)), "features": SELECTOR_FEATURES,
            "minimum_exploration_leaf": 5, "objective": "maximize success count, then minimize capped frames",
            "sequence_runs": str(sequence_runs.resolve()),
            "sequence_runs_sha256": hashlib.sha256(sequence_runs.read_bytes()).hexdigest(),
            "population": population, "max_steps": max_steps, "stage_limit": stage_limit,
            "config": metadata["config"],
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in (*metadata["source_sha256"],
                                           Path(__file__).relative_to(ROOT).as_posix(),
                                           "experiments/anomaly_behavior/run_concurrent_mode_priority_trial.py")},
            "limitation": "One first-dual-alarm intervention; no API, no production policy change.",
        }
        if set(protocol["exploration_seeds"]) & set(protocol["validation_seeds"]):
            raise ValueError("Exploration and validation seeds overlap")
        # Freeze the choice on disk BEFORE generating any validation outcomes.
        frozen = json.dumps(protocol, indent=2)
        (output / "frozen_selector.json").write_text(frozen, encoding="utf-8")
        print(f"Frozen selector: {selector}", flush=True)
        validation = [replay_sequence_features(seed, max_steps, stage_limit, population)
                      for seed in protocol["validation_seeds"]]
    assert (output / "frozen_selector.json").read_text(encoding="utf-8") == frozen
    all_rows = [{"cohort": name, **row} for name, rows in
                (("exploration", exploration), ("validation", validation)) for row in rows]
    (output / "selector_states.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    report = {
        "api_calls": 0, "protocol_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
        "screened_candidates": len(screening), "selector": selector,
        "summary": {name: {label: selector_summary(rows, policy) for label, policy in
                           (("rule", {"kind": "rule"}), ("forecast_dominance", {"kind": "dominance"}),
                            ("fitted_selector", selector))}
                    for name, rows in (("exploration", exploration), ("validation", validation))},
        "screening": screening,
        "interpretation": "Exploration fit followed by frozen held-out evaluation, not evidence of LLM superiority.",
    }
    (output / "sequence_selector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selector", "summary", "api_calls")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=1320)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--review-period", type=int, default=120)
    parser.add_argument("--sequence-selector-diagnostic", type=Path,
                        help="Explore a saved two-stage cohort, freeze one simple selector, validate on 50 other seeds.")
    args = parser.parse_args()
    if args.sequence_selector_diagnostic:
        output = config.PROJECT_ROOT / "logs/anomaly_behavior" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        run_sequence_selector_diagnostic(args.sequence_selector_diagnostic, output)
        print(f"Report saved to: {output}")
        return
    try:
        rows, summary = run_trial(
            args.first_seed, args.seed_count, args.max_steps,
            args.review_period,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        config.PROJECT_ROOT / "logs" / "anomaly_behavior"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "forecast_mode_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "forecast_mode_report.json").write_text(
        json.dumps({
            "controller": "rule/local_forecast",
            "first_seed": args.first_seed,
            "seed_count": args.seed_count,
            "max_steps": args.max_steps,
            "review_period": args.review_period,
            "summary": summary,
        }, indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {output}")
    for scenario, variants in summary.items():
        for name, metrics in variants.items():
            print(
                f"{scenario} {name}: {metrics['successes']}/{metrics['runs']} "
                f"success; rescued {metrics['rescued']}, "
                f"regressed {metrics['regressed']}; "
                f"mean forecast calls {metrics['mean_forecast_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
