import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from control_types import ControlObservation
from environment import SwarmEnv
from high_level_controller import HighLevelController, MockLLMBackend
from low_level_controller import LowLevelController
from mode_forecast import forecast_mode, forecast_modes
from run_experiments import count_high_level_override_frames, temporary_config


class TargetBackend(MockLLMBackend):
    trigger_on_concurrent_only = True

    def __init__(self, target=1, mode="HARD_TO_GUIDE"):
        super().__init__(latency_frames=1, hold_frames=4)
        self.target = target
        self.mode = mode

    def poll(self, request_id, frame):
        response = super().poll(request_id, frame)
        if response is not None:
            response.update(mode=self.mode, target_sheep_id=self.target)
        return response


class HighLevelTargetSelectionTests(unittest.TestCase):
    def settings(self):
        values = {key: value for key, value in vars(config).items() if key.isupper()}
        values.update(HIGH_LEVEL_MIN_HOLD_FRAMES=1, HIGH_LEVEL_MIN_DWELL_FRAMES=1,
                      HIGH_LEVEL_FORECAST_FRAMES=2)
        return SimpleNamespace(**values)

    def observation(self, frame=0):
        return ControlObservation(
            sheep_positions=np.array([[40., 100.], [80., 100.], [120., 100.]]),
            dog_position=np.array([65., 110.]),
            goal_position=np.array([180., 100.]),
            alarms=("flock_splitting", "stagnation"), frame=frame,
            sheep_statuses=("A", "B", "C"), sheep_velocities=np.zeros((3, 2)),
            dog_velocity=np.zeros(2),
        )

    def controller(self, target=1, mode="HARD_TO_GUIDE"):
        settings = self.settings()
        return HighLevelController(settings=settings, backend=TargetBackend(target, mode))

    def accept_target(self, controller, observation):
        # Isolate interface behavior, not a claim that this target is preferable.
        with patch.object(controller, "_forecast_dominates", return_value=False), \
             patch.object(controller, "_forecast_preference_reversed", return_value=False):
            return controller.decide(observation)

    def test_mode_only_rejects_explicit_target_and_keeps_candidate_summary(self):
        controller = self.controller()
        controller.settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        controller.decide(self.observation())
        self.assertIn("repair_candidates", controller.pending_request.state_summary)
        decision = controller.decide(self.observation(1))
        self.assertEqual(decision.metadata["rejection_reason"], "target_selection_disabled")
        self.assertIsNone(controller.active_target_id)
        controller.close()

    def test_mode_only_accepts_null_target(self):
        controller = self.controller(target=None)
        controller.settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        controller.decide(self.observation())
        decision = self.accept_target(controller, self.observation(1))
        self.assertEqual(decision.source, "mock_llm")
        self.assertEqual(decision.target_sheep_id, 0)
        self.assertFalse(decision.metadata["target_override"])
        controller.close()

    def test_target_only_selects_target_without_overriding_rule_mode(self):
        controller = self.controller()
        controller.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        controller.decide(self.observation())
        decision = self.accept_target(controller, self.observation(1))
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(decision.target_sheep_id, 1)
        self.assertTrue(decision.metadata["target_override"])
        controller.close()

    def test_target_only_rejects_mode_selection(self):
        controller = self.controller(target=None, mode="TARGET_TRACKING")
        controller.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        controller.decide(self.observation())
        decision = controller.decide(self.observation(1))
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.metadata["rejection_reason"], "mode_selection_disabled")
        controller.close()

    def test_target_only_rejects_response_if_rule_mode_changed_during_inference(self):
        controller = self.controller()
        controller.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        controller.decide(self.observation())
        changed = replace(self.observation(1), alarms=("flock_splitting",))
        decision = controller.decide(changed)
        self.assertEqual(decision.mode, "COHESION_PRIORITY")
        self.assertEqual(decision.metadata["rejection_reason"], "rule_mode_changed")
        controller.close()

    def test_target_only_releases_before_minimum_dwell_when_rule_mode_changes(self):
        for target in (1, None):
            with self.subTest(target=target):
                controller = self.controller(target=target)
                controller.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
                controller.settings.HIGH_LEVEL_MIN_DWELL_FRAMES = 15
                controller.decide(self.observation())
                self.accept_target(controller, self.observation(1))
                changed = replace(self.observation(2), alarms=("flock_splitting",))
                decision = controller.decide(changed)
                self.assertEqual(decision.mode, "COHESION_PRIORITY")
                self.assertEqual(decision.source, "rule")
                self.assertEqual(decision.metadata["release_reason"], "rule_mode_changed")
                self.assertIsNone(controller.active_target_id)
                self.assertIsNone(controller.active_mode)
                controller.close()

    def test_target_only_keeps_target_when_rule_mode_is_unchanged(self):
        controller = self.controller()
        controller.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        controller.decide(self.observation())
        self.accept_target(controller, self.observation(1))
        changed = replace(self.observation(2), alarms=("stagnation",))
        decision = controller.decide(changed)
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(decision.target_sheep_id, 1)
        controller.close()

    def test_target_only_releases_arrived_target(self):
        controller = self.controller()
        controller.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        controller.decide(self.observation())
        self.accept_target(controller, self.observation(1))
        positions = self.observation(2).sheep_positions.copy()
        positions[1] = self.observation(2).goal_position
        decision = controller.decide(replace(self.observation(2), sheep_positions=positions))
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(decision.metadata["release_reason"], "target_no_longer_eligible")
        self.assertIsNone(controller.active_target_id)
        controller.close()

    def test_low_level_override_changes_target_not_motion_rule(self):
        low = LowLevelController()
        observation = self.observation()
        self.assertEqual(low.decide(observation).target_sheep_id, 0)
        chosen = low.decide(observation, mode_override="HARD_TO_GUIDE", target_override=1)
        self.assertEqual(chosen.target_sheep_id, 1)
        np.testing.assert_allclose(chosen.drive_point, [77., 100.], atol=1e-6)
        collected = low.decide(observation, mode_override="COHESION_PRIORITY", target_override=2)
        np.testing.assert_allclose(collected.drive_point, [130., 100.], atol=1e-6)
        self.assertEqual(low.repair_target_id, 2)

    def test_low_level_rejects_invalid_or_arrived_targets(self):
        low = LowLevelController()
        observation = self.observation()
        for target in (True, -1, 3, 1.0, "1"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                low.decide(observation, mode_override="HARD_TO_GUIDE", target_override=target)
        with self.assertRaises(ValueError):
            low.decide(observation, mode_override="TARGET_TRACKING", target_override=1)
        positions = observation.sheep_positions.copy()
        positions[1] = observation.goal_position
        with self.assertRaises(ValueError):
            low.decide(replace(observation, sheep_positions=positions),
                       mode_override="HARD_TO_GUIDE", target_override=1)

    def test_candidate_list_is_bounded_includes_held_target_and_excludes_arrivals(self):
        low = LowLevelController()
        observation = replace(self.observation(), sheep_positions=np.array(
            [[20., 100.], [40., 100.], [60., 100.], [80., 100.], [100., 100.], [180., 100.]]
        ), sheep_velocities=np.zeros((6, 2)))
        low.repair_target_mode, low.repair_target_id, low.repair_target_age = "HARD_TO_GUIDE", 4, 1
        before = vars(low).copy()
        candidates = low.repair_candidates(observation, "HARD_TO_GUIDE")
        self.assertEqual(candidates, [4, 0, 1])
        self.assertNotIn(5, candidates)
        self.assertEqual(vars(low), before)

    def test_target_forecast_matches_fixed_target_motion_and_preserves_live_controller(self):
        settings = self.settings()
        observation = self.observation()
        low = LowLevelController(settings)
        low.decide(observation)
        before = vars(low).copy()
        chosen = forecast_mode(observation, "HARD_TO_GUIDE", 2, settings, low, target_sheep_id=1)
        default = forecast_modes(observation, settings, low)["HARD_TO_GUIDE"]
        self.assertNotEqual(chosen, default)
        self.assertEqual(vars(low), before)
        self.assertEqual(chosen["horizon_frames"], 2)

    def test_forecast_releases_a_target_that_enters_goal(self):
        observation = replace(self.observation(),
            sheep_positions=np.array([[159.5, 100.], [100., 100.]]),
            sheep_statuses=("Normal", "Normal"),
            sheep_velocities=np.array([[0.7, 0.], [0., 0.]]),
            dog_position=np.array([140., 100.]))
        result = forecast_mode(observation, "HARD_TO_GUIDE", 10, self.settings(),
                               target_sheep_id=0)
        self.assertEqual(result["horizon_frames"], 10)
        self.assertGreaterEqual(result["final_inside_goal"], 1)

    def test_request_offers_per_target_geometry_and_forecasts(self):
        controller = self.controller()
        pending = controller.decide(self.observation())
        self.assertEqual(pending.source, "rule")
        summary = controller.pending_request.state_summary
        for mode, candidates in summary["repair_candidates"].items():
            self.assertEqual(len(candidates), 3)
            for candidate in candidates:
                self.assertIn("target_status", candidate)
                self.assertIn("dog_to_drive_point", candidate)
                self.assertEqual(candidate["forecast"], forecast_mode(
                    self.observation(), mode, 2, controller.settings,
                    controller.low_level_controller, target_sheep_id=candidate["target_id"],
                ))
        controller.close()

    def test_selected_target_reaches_execution_and_is_held(self):
        controller = self.controller()
        controller.decide(self.observation())
        chosen = self.accept_target(controller, self.observation(1))
        self.assertEqual(chosen.source, "mock_llm")
        self.assertEqual(chosen.mode, "HARD_TO_GUIDE")
        self.assertEqual(chosen.target_sheep_id, 1)
        self.assertEqual(chosen.metadata["selected_target_sheep_id"], 1)
        self.assertEqual(chosen.metadata["rule_target_sheep_id"], 0)
        self.assertTrue(chosen.metadata["target_override"])
        self.assertEqual(controller.decide(self.observation(2)).target_sheep_id, 1)
        controller.close()

    def test_null_target_delegates_to_existing_rule(self):
        controller = self.controller(target=None)
        controller.decide(self.observation())
        chosen = self.accept_target(controller, self.observation(1))
        self.assertEqual(chosen.target_sheep_id, 0)
        self.assertFalse(chosen.metadata["target_override"])
        controller.close()

    def test_invalid_targets_fall_back_without_crashing(self):
        for target, reason in ((True, "target_not_integer"), (1.0, "target_not_integer"),
                               ("1", "target_not_integer"), (-1, "target_not_offered"),
                               (3, "target_not_offered")):
            with self.subTest(target=target):
                controller = self.controller(target=target)
                controller.decide(self.observation())
                chosen = controller.decide(self.observation(1))
                self.assertEqual(chosen.source, "rule")
                self.assertEqual(chosen.metadata["rejection_reason"], reason)
                controller.close()

    def test_target_outside_offered_shortlist_is_rejected(self):
        controller = self.controller(target=3)
        observation = replace(self.observation(), sheep_positions=np.array(
            [[40., 100.], [60., 100.], [80., 100.], [120., 100.]]
        ), sheep_velocities=np.zeros((4, 2)))
        controller.decide(observation)
        chosen = controller.decide(replace(observation, frame=1))
        self.assertEqual(chosen.metadata["rejection_reason"], "target_not_offered")
        controller.close()

    def test_normal_mode_cannot_force_a_sheep_target(self):
        controller = self.controller(mode="TARGET_TRACKING")
        controller.decide(self.observation())
        chosen = controller.decide(self.observation(1))
        self.assertEqual(chosen.metadata["rejection_reason"], "target_requires_repair_mode")
        controller.close()

    def test_target_that_arrives_during_inference_is_rejected(self):
        controller = self.controller()
        controller.decide(self.observation())
        positions = self.observation().sheep_positions.copy()
        positions[1] = self.observation().goal_position
        chosen = controller.decide(replace(self.observation(1), sheep_positions=positions))
        self.assertEqual(chosen.metadata["rejection_reason"], "target_no_longer_eligible")
        controller.close()

    def test_offered_target_remains_valid_after_dropping_out_of_current_shortlist(self):
        controller = self.controller(target=2)
        observation = replace(self.observation(), sheep_positions=np.array(
            [[40., 100.], [60., 100.], [80., 100.], [120., 100.]]
        ), sheep_velocities=np.zeros((4, 2)))
        controller.decide(observation)
        positions = observation.sheep_positions.copy()
        positions[2] = [155., 100.]
        current = replace(observation, frame=1, sheep_positions=positions)
        self.assertNotIn(2, controller.low_level_controller.repair_candidates(current, "HARD_TO_GUIDE"))
        chosen = self.accept_target(controller, current)
        self.assertEqual(chosen.target_sheep_id, 2)
        self.assertEqual(chosen.source, "mock_llm")
        controller.close()

    def test_malformed_mode_with_target_falls_back(self):
        controller = self.controller(mode=[])
        controller.decide(self.observation())
        chosen = controller.decide(self.observation(1))
        self.assertEqual(chosen.metadata["rejection_reason"], "mode_not_allowed")
        controller.close()

    def test_active_target_is_released_on_arrival(self):
        controller = self.controller()
        controller.decide(self.observation())
        self.accept_target(controller, self.observation(1))
        positions = self.observation().sheep_positions.copy()
        positions[1] = self.observation().goal_position
        chosen = controller.decide(replace(self.observation(2), sheep_positions=positions))
        self.assertEqual(chosen.source, "rule")
        self.assertNotEqual(chosen.target_sheep_id, 1)
        self.assertEqual(chosen.metadata["release_reason"], "target_no_longer_eligible")
        controller.close()

    def test_boundary_guard_checks_chosen_target_not_rule_target(self):
        controller = self.controller()
        observation = replace(self.observation(), sheep_positions=np.array(
            [[40., 100.], [1., 100.], [120., 100.]]
        ))
        controller.decide(observation)
        chosen = controller.decide(replace(observation, frame=1))
        self.assertEqual(chosen.metadata["rejection_reason"], "candidate_drive_point_out_of_bounds")
        controller.close()

    def test_guard_compares_forecast_for_the_chosen_target(self):
        controller = self.controller()
        controller.decide(self.observation())
        response = controller.backend.poll("high-1", 1)
        summary = controller._with_mode_forecasts(self.observation(1),
                    controller.pending_request.state_summary, response=response)
        selected = summary["selected_target_candidate"]["forecast"]
        for key in ("centroid_progress", "inside_goal_change"):
            selected[key] = -1000
        for key in ("final_max_spread", "predicted_goal_exits",
                    "predicted_splitting_frames", "peak_out_of_bounds"):
            selected[key] = 1000
        mode, hold, reason = controller._validate_response(response, 1,
                                self.observation().alarms, summary)
        self.assertIsNone(mode)
        self.assertTrue(reason.startswith("forecast_dominated_by_"))
        controller.close()

    def test_target_only_change_is_logged_and_counted_as_an_override(self):
        controller = self.controller()
        observation = self.observation()
        with temporary_config(NUM_ABNORMAL_A=0, NUM_ABNORMAL_B=0, NUM_ABNORMAL_C=0):
            env = SwarmEnv(controller=controller)
            env.sheep_pos = observation.sheep_positions.copy()
            env.sheep_vel = observation.sheep_velocities.copy()
            env.sheep_status = np.array(observation.sheep_statuses, dtype=object)
            env.dog_pos = observation.dog_position.copy()
            env.goal_pos = observation.goal_position.copy()
            env.step(observation.alarms)
            with patch.object(controller, "_forecast_dominates", return_value=False), \
                 patch.object(controller, "_forecast_preference_reversed", return_value=False):
                env.step(observation.alarms)
            row = env.control_data[-1]
            self.assertEqual(row["llm_selected_target_sheep_id"], 1)
            self.assertEqual(row["llm_rule_target_sheep_id"], 0)
            self.assertEqual(json.loads(row["llm_response_target_sheep_id"]), 1)
            self.assertTrue(row["llm_target_override"])
            self.assertEqual(count_high_level_override_frames(pd.DataFrame(env.control_data), True), 1)
            env.close()


if __name__ == "__main__":
    unittest.main()
