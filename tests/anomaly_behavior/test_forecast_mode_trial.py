"""Tests for the experimental local forecast selector."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from control_types import ControlObservation
from environment import SwarmEnv
from run_experiments import make_detector, temporary_config
from experiments.anomaly_behavior.run_forecast_mode_trial import (
    ForecastDominanceController, SELECTOR_FEATURES, fit_sequence_selector,
    replay_sequence_features, selector_collects, selector_summary, sequence_features,
    run_sequence_selector_diagnostic,
)
from experiments.anomaly_behavior.run_concurrent_mode_priority_trial import state_fingerprint


class ForecastModeTrialTests(unittest.TestCase):
    def observation(self, frame, alarms=("flock_splitting", "stagnation")):
        return ControlObservation(
            sheep_positions=np.array([[20., 20.], [40., 40.]]),
            dog_position=np.array([10., 10.]),
            goal_position=np.array([180., 180.]),
            alarms=alarms,
            frame=frame,
        )

    def test_dominant_cohesion_is_held_until_review(self):
        guide = {
            "centroid_progress": 1, "inside_goal_change": 0,
            "final_max_spread": 40, "predicted_goal_exits": 0,
            "predicted_splitting_frames": 10, "peak_out_of_bounds": 0,
        }
        cohesion = {**guide, "final_max_spread": 30}
        controller = ForecastDominanceController(review_period=10)
        with patch(
            "experiments.anomaly_behavior.run_forecast_mode_trial.forecast_modes",
            return_value={
                "COHESION_PRIORITY": cohesion,
                "HARD_TO_GUIDE": guide,
            },
        ) as forecast:
            self.assertEqual(
                controller.decide(self.observation(1)).mode,
                "COHESION_PRIORITY",
            )
            self.assertEqual(
                controller.decide(self.observation(2)).mode,
                "COHESION_PRIORITY",
            )
            self.assertEqual(forecast.call_count, 1)
            self.assertEqual(
                controller.decide(self.observation(3, ("stagnation",))).mode,
                "HARD_TO_GUIDE",
            )
            self.assertEqual(
                controller.decide(self.observation(4)).mode,
                "COHESION_PRIORITY",
            )
            self.assertEqual(forecast.call_count, 2)

    def selector_row(self, index, collect_frames=200, eligible=True):
        outcome = {"success": True, "capped_frames": 100, "danger_frames": 20,
                   "max_spread": 50., "goal_exits": 1}
        return {"seed": index, "eligible": eligible, "forecast_dominates": False,
                **{key: index for key in SELECTOR_FEATURES},
                "rule": outcome, "collect": {**outcome, "capped_frames": collect_frames}}

    def test_threshold_fits_exploration_and_does_not_refit_on_new_rows(self):
        training = [self.selector_row(index, 50 if index < 5 else 200) for index in range(10)]
        policy, _ = fit_sequence_selector(training)
        self.assertEqual(policy, {"kind": "threshold", "feature": "max_spread",
                                  "threshold": 4.5, "side": "le"})
        validation = [self.selector_row(index, 200) for index in range(10)]
        self.assertEqual(selector_summary(validation, policy)["mean_capped_frames"], 150)
        self.assertEqual(policy["threshold"], 4.5)
        self.assertFalse(selector_collects(self.selector_row(0, eligible=False), policy))

    def test_success_precedes_speed_and_ties_keep_rule(self):
        rows = [self.selector_row(index, 50) for index in range(10)]
        for row in rows:
            row["collect"]["success"] = False
        self.assertEqual(fit_sequence_selector(rows)[0], {"kind": "rule"})
        rows = [self.selector_row(index, 100) for index in range(10)]
        self.assertEqual(fit_sequence_selector(rows)[0], {"kind": "rule"})

    def test_forecasts_receive_target_memory_without_mutating_real_state(self):
        with temporary_config(USE_FIXED_SEED=True, RANDOM_SEED=3302, CONTROLLER_KIND="rule"):
            env, detector = SwarmEnv(), make_detector()
            env.controller.repair_target_id = 1
            env.controller.repair_target_age = 17
            before = state_fingerprint(env, detector)
            from mode_forecast import forecast_modes
            with patch("experiments.anomaly_behavior.run_forecast_mode_trial.forecast_modes",
                       wraps=forecast_modes) as forecast:
                features = sequence_features(env, ("flock_splitting", "stagnation"))
            self.assertIs(forecast.call_args.args[2], env.controller)
            self.assertEqual(state_fingerprint(env, detector), before)
            self.assertTrue(set(SELECTOR_FEATURES).issubset(features))

    def test_no_alarm_case_skips_intervention_and_api(self):
        with patch("high_level_controller.build_controller", side_effect=RuntimeError("No API controller")):
            row = replay_sequence_features(70, 2, 60, (0, 0, 0))
        self.assertFalse(row["eligible"])
        self.assertEqual(row["collect"], row["rule"])

    def test_protocol_is_frozen_before_validation_and_fit_sees_only_exploration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "repair_sequence_runs.csv"
            source.write_text("seed,variant\n" + "".join(
                f"{seed},{variant}\n" for seed in range(10)
                for variant in ("rule", "collect_then_guide")), encoding="utf-8")
            rule_summary = output / "rule.csv"
            metadata = {"source_sha256": {}, "config": {}, "case_settings": [
                {"num_abnormal_a": 5, "num_abnormal_b": 5, "num_abnormal_c": 5}]}
            Path(str(rule_summary) + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            source.with_name("repair_sequence_report.json").write_text(json.dumps({
                "source_sha256": {}, "rule_summary": str(rule_summary),
                "max_steps": 2000, "stage_limit": 60}), encoding="utf-8")
            seen_frozen = []

            def replay(seed, *args):
                if seed >= 4401:
                    frozen = json.loads((output / "frozen_selector.json").read_text(encoding="utf-8"))
                    self.assertEqual(frozen["selector"]["threshold"], 4.5)
                    seen_frozen.append(seed)
                return self.selector_row(seed, 50 if seed < 5 else 200)

            with patch("experiments.anomaly_behavior.run_forecast_mode_trial.replay_sequence_features",
                       side_effect=replay), patch(
                           "experiments.anomaly_behavior.run_forecast_mode_trial.fit_sequence_selector",
                           wraps=fit_sequence_selector) as fit, patch("builtins.print"):
                report = run_sequence_selector_diagnostic(source, output)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual([row["seed"] for row in fit.call_args.args[0]], list(range(10)))
            self.assertEqual(seen_frozen, list(range(4401, 4451)))
            self.assertEqual(report["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
