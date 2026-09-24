import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
from control_types import ControlObservation
from environment import SwarmEnv
from high_level_controller import HighLevelController, MockLLMBackend, build_controller
from low_level_controller import LowLevelController
from run_experiments import temporary_config


class ErrorBackend:
    source = "openai"

    def __init__(self):
        self.request = None

    def submit(self, request):
        self.request = request

    def poll(self, request_id, frame):
        if self.request is None or frame <= self.request.trigger_frame:
            return None
        return {
            "request_id": request_id,
            "trigger_frame": self.request.trigger_frame,
            "backend_error": "TimeoutError: test timeout",
            "latency_seconds": 1.0,
            "model": "test-model",
        }

    def cancel(self, request_id):
        del request_id
        self.request = None


class ConcurrentOnlyBackend(MockLLMBackend):
    trigger_on_concurrent_only = True


class HighLevelControllerTests(unittest.TestCase):
    def settings(self, **overrides):
        values = {
            name: getattr(config, name)
            for name in dir(config)
            if name.isupper()
        }
        values.update(
            HIGH_LEVEL_MIN_HOLD_FRAMES=1,
            HIGH_LEVEL_MAX_HOLD_FRAMES=300,
            HIGH_LEVEL_MIN_DWELL_FRAMES=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def observation(
        self,
        frame,
        alarms=("flock_splitting",),
        sheep_positions=None,
        statuses=(),
        sheep_velocities=None,
        dog_velocity=None,
    ):
        return ControlObservation(
            sheep_positions=(
                np.array([[20., 20.], [60., 20.]])
                if sheep_positions is None
                else np.asarray(sheep_positions, dtype=float)
            ),
            dog_position=np.array([10., 20.]),
            goal_position=np.array([180., 180.]),
            alarms=alarms,
            frame=frame,
            sheep_statuses=statuses,
            sheep_velocities=(
                None
                if sheep_velocities is None
                else np.asarray(sheep_velocities, dtype=float)
            ),
            dog_velocity=(
                None
                if dog_velocity is None
                else np.asarray(dog_velocity, dtype=float)
            ),
        )

    def controller(self, latency=2, hold=3, mapping=None):
        settings = self.settings()
        backend = MockLLMBackend(
            latency_frames=latency,
            hold_frames=hold,
            mode_mapping=mapping,
        )
        return HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        ), backend

    def test_pending_keeps_low_level_rule_control_running(self):
        controller, _ = self.controller()
        decision = controller.decide(self.observation(0))
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.mode, "COHESION_PRIORITY")
        self.assertEqual(decision.metadata["high_level_status"], "pending")
        self.assertEqual(controller.request_counter, 1)
        controller.decide(self.observation(1))
        self.assertEqual(controller.request_counter, 1)

    def test_valid_response_activates_then_expires(self):
        controller, _ = self.controller()
        controller.decide(self.observation(0))
        decision = controller.decide(self.observation(2))
        self.assertEqual(decision.source, "mock_llm")
        self.assertEqual(decision.metadata["high_level_status"], "active")
        self.assertEqual(decision.metadata["latency_frames"], 2)
        self.assertEqual(decision.metadata["hold_frames"], 3)
        self.assertEqual(decision.metadata["selected_mode"], "COHESION_PRIORITY")
        self.assertEqual(decision.metadata["reference_mode"], "COHESION_PRIORITY")
        self.assertTrue(decision.metadata["selection_correct"])
        decision = controller.decide(self.observation(4))
        self.assertEqual(decision.metadata["latency_frames"], 2)
        decision = controller.decide(self.observation(5))
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.metadata["high_level_status"], "pending")
        self.assertEqual(controller.request_counter, 2)

    def test_alarm_clear_cancels_pending_request(self):
        controller, backend = self.controller(latency=10)
        controller.decide(self.observation(0))
        request_id = controller.pending_request.request_id
        decision = controller.decide(self.observation(1, ()))
        self.assertEqual(decision.metadata["high_level_status"], "idle")
        self.assertNotIn(request_id, backend.pending)
        self.assertIsNone(controller.pending_request)

    def test_alarm_change_replaces_pending_request(self):
        controller, backend = self.controller(latency=10)
        controller.decide(self.observation(0))
        first = controller.pending_request.request_id
        controller.decide(self.observation(1, ("stagnation",)))
        second = controller.pending_request.request_id
        self.assertNotEqual(first, second)
        self.assertNotIn(first, backend.pending)
        self.assertIn(second, backend.pending)

    def test_concurrent_alarm_context_replaces_single_alarm_request(self):
        controller, backend = self.controller(latency=10)
        controller.decide(self.observation(0))
        first = controller.pending_request.request_id
        controller.decide(
            self.observation(1, ("flock_splitting", "stagnation"))
        )
        request = controller.pending_request
        self.assertNotEqual(first, request.request_id)
        self.assertEqual(
            request.alarms, ("flock_splitting", "stagnation")
        )
        self.assertNotIn(first, backend.pending)

    def test_request_summary_contains_trends_and_sheep_types(self):
        controller, _ = self.controller(latency=10)
        controller.decide(
            self.observation(
                0,
                (),
                sheep_positions=[[40., 40.], [42., 40.]],
                statuses=("A", "B"),
            )
        )
        controller.decide(
            self.observation(
                5,
                ("flock_splitting",),
                sheep_positions=[[20., 40.], [70., 40.]],
                statuses=("A", "B"),
                sheep_velocities=[[-0.2, 0.0], [0.4, 0.0]],
                dog_velocity=[1.0, 0.0],
            )
        )
        summary = controller.pending_request.state_summary
        self.assertEqual(summary["type_a_count"], 1)
        self.assertEqual(summary["type_b_count"], 1)
        self.assertEqual(summary["context_span_frames"], 5)
        self.assertGreater(summary["max_spread_change"], 0)
        self.assertEqual(summary["dog_speed"], 1.0)
        self.assertEqual(summary["cohesion_candidate"]["target_status"], "A")
        self.assertEqual(summary["hard_to_guide_candidate"]["target_status"], "A")
        self.assertEqual(summary["type_geometry"]["B"]["count"], 1)
        self.assertIn("estimated_travel_frames", summary["cohesion_candidate"])

    def test_request_summary_distinguishes_repair_candidate_geometry(self):
        controller, _ = self.controller(latency=10)
        positions = [[20., 100.], [90., 100.], [100., 100.]]
        observation = self.observation(
            0,
            ("flock_splitting", "stagnation"),
            sheep_positions=positions,
            statuses=("A", "B", "Normal"),
            sheep_velocities=[[-0.4, 0.0], [0.3, 0.0], [0.0, 0.0]],
            dog_velocity=[1.5, 0.0],
        )
        controller.decide(observation)
        summary = controller.pending_request.state_summary

        self.assertEqual(summary["cohesion_candidate"]["target_id"], 0)
        self.assertEqual(summary["hard_to_guide_candidate"]["target_id"], 0)
        self.assertEqual(summary["hard_to_guide_candidate"]["target_status"], "A")
        self.assertLess(summary["hard_to_guide_candidate"]["target_goalward_speed"], 0)
        self.assertGreaterEqual(summary["moving_away_from_goal_count"], 1)
        self.assertEqual(summary["outside_goal"], 3)

        for mode, key in (
            ("COHESION_PRIORITY", "cohesion_candidate"),
            ("HARD_TO_GUIDE", "hard_to_guide_candidate"),
        ):
            decision = controller.low_level_controller.decide(
                observation, mode_override=mode
            )
            expected_distance = np.linalg.norm(
                observation.dog_position - decision.drive_point
            )
            self.assertAlmostEqual(
                summary[key]["dog_to_drive_point"], expected_distance, places=3
            )

    def test_invalid_mode_is_rejected_with_rule_fallback(self):
        controller, _ = self.controller(
            latency=1, mapping={"flock_splitting": "TELEPORT"}
        )
        controller.decide(self.observation(0))
        decision = controller.decide(self.observation(1))
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.mode, "COHESION_PRIORITY")
        self.assertEqual(decision.metadata["high_level_status"], "rejected")
        self.assertEqual(decision.metadata["request_id"], "high-1")
        self.assertEqual(decision.metadata["rejection_reason"], "mode_not_allowed")

    def test_stale_response_is_rejected(self):
        with temporary_config(HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES=2):
            controller, _ = self.controller(latency=3)
            controller.decide(self.observation(0))
            decision = controller.decide(self.observation(3))
        self.assertEqual(decision.metadata["rejection_reason"], "response_stale")
        self.assertEqual(decision.source, "rule")

    def test_out_of_range_hold_is_rejected(self):
        controller, _ = self.controller(latency=1, hold=1000)
        controller.decide(self.observation(0))
        decision = controller.decide(self.observation(1))
        self.assertEqual(
            decision.metadata["rejection_reason"], "hold_frames_out_of_range"
        )

    def test_backend_error_falls_back_and_keeps_error_metadata(self):
        controller = HighLevelController(
            low_level_controller=LowLevelController(), backend=ErrorBackend()
        )
        controller.decide(self.observation(0))
        decision = controller.decide(self.observation(1))
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.metadata["high_level_status"], "rejected")
        self.assertEqual(decision.metadata["rejection_reason"], "backend_error")
        self.assertIn("TimeoutError", decision.metadata["backend_error"])

    def test_repair_disabled_never_submits(self):
        with temporary_config(REPAIR_ENABLED=False):
            controller, backend = self.controller()
            decision = controller.decide(self.observation(0))
        self.assertEqual(decision.metadata["high_level_status"], "disabled")
        self.assertEqual(controller.request_counter, 0)
        self.assertEqual(backend.pending, {})

    def test_api_style_backend_only_requests_concurrent_alarm_context(self):
        backend = ConcurrentOnlyBackend(latency_frames=10)
        controller = HighLevelController(
            low_level_controller=LowLevelController(), backend=backend
        )

        single = controller.decide(self.observation(0, ("stagnation",)))
        concurrent = controller.decide(
            self.observation(1, ("flock_splitting", "stagnation"))
        )

        self.assertEqual(single.metadata["high_level_status"], "rule_only")
        self.assertEqual(concurrent.metadata["high_level_status"], "pending")
        self.assertEqual(controller.request_counter, 1)

    def test_concurrent_request_survives_while_selected_alarm_remains(self):
        backend = ConcurrentOnlyBackend(
            latency_frames=2,
            hold_frames=3,
            mode_mapping={"flock_splitting": "HARD_TO_GUIDE"},
        )
        settings = self.settings()
        controller = HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        )
        controller._dominating_forecast_mode = lambda *args: None

        controller.decide(
            self.observation(0, ("flock_splitting", "stagnation"))
        )
        pending = controller.decide(self.observation(1, ("stagnation",)))
        active = controller.decide(self.observation(2, ("stagnation",)))

        self.assertEqual(pending.metadata["high_level_status"], "pending")
        self.assertEqual(active.metadata["high_level_status"], "active")
        self.assertEqual(active.mode, "HARD_TO_GUIDE")
        self.assertEqual(active.source, "mock_llm")

    def test_response_is_rejected_when_selected_alarm_has_cleared(self):
        backend = ConcurrentOnlyBackend(
            latency_frames=1,
            hold_frames=10,
            mode_mapping={"flock_splitting": "COHESION_PRIORITY"},
        )
        settings = self.settings()
        controller = HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        )

        controller.decide(
            self.observation(0, ("flock_splitting", "stagnation"))
        )
        decision = controller.decide(self.observation(1, ("stagnation",)))

        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(
            decision.metadata["rejection_reason"], "selected_alarm_not_active"
        )

    def test_response_is_rejected_for_out_of_bounds_candidate_drive_point(self):
        backend = ConcurrentOnlyBackend(latency_frames=1, hold_frames=10)
        settings = self.settings()
        controller = HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        )
        observation = self.observation(
            0,
            ("flock_splitting", "stagnation"),
            sheep_positions=[[1., 100.], [80., 100.], [90., 100.]],
        )

        controller.decide(observation)
        decision = controller.decide(
            self.observation(
                1,
                ("flock_splitting", "stagnation"),
                sheep_positions=[[1., 100.], [80., 100.], [90., 100.]],
            )
        )

        self.assertEqual(decision.source, "rule")
        self.assertEqual(
            decision.metadata["rejection_reason"],
            "candidate_drive_point_out_of_bounds",
        )

    def test_concurrent_event_mode_holds_after_alarm_becomes_single(self):
        backend = ConcurrentOnlyBackend(
            latency_frames=2,
            hold_frames=10,
            mode_mapping={"flock_splitting": "COHESION_PRIORITY"},
        )
        settings = self.settings(HIGH_LEVEL_MIN_DWELL_FRAMES=3)
        controller = HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        )

        controller.decide(
            self.observation(0, ("flock_splitting", "stagnation"))
        )
        controller.decide(
            self.observation(1, ("flock_splitting", "stagnation"))
        )
        decision = controller.decide(
            self.observation(2, ("flock_splitting", "stagnation"))
        )

        self.assertEqual(decision.source, "mock_llm")
        self.assertEqual(decision.metadata["high_level_status"], "active")
        self.assertEqual(decision.mode, "COHESION_PRIORITY")

        still_active = controller.decide(self.observation(3, ("stagnation",)))
        self.assertEqual(still_active.source, "mock_llm")
        self.assertEqual(still_active.mode, "COHESION_PRIORITY")

    def test_selected_alarm_clear_releases_after_minimum_dwell(self):
        backend = ConcurrentOnlyBackend(
            latency_frames=1,
            hold_frames=10,
            mode_mapping={"flock_splitting": "COHESION_PRIORITY"},
        )
        settings = self.settings(HIGH_LEVEL_MIN_DWELL_FRAMES=3)
        controller = HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        )

        controller.decide(
            self.observation(0, ("flock_splitting", "stagnation"))
        )
        controller.decide(
            self.observation(1, ("flock_splitting", "stagnation"))
        )
        self.assertEqual(
            controller.decide(self.observation(3, ("stagnation",))).source,
            "mock_llm",
        )
        released = controller.decide(self.observation(4, ("stagnation",)))

        self.assertEqual(released.source, "rule")
        self.assertEqual(released.mode, "HARD_TO_GUIDE")
        self.assertEqual(released.metadata["high_level_status"], "rule_only")
        self.assertEqual(
            released.metadata["release_reason"], "selected_alarm_cleared"
        )
        self.assertEqual(released.metadata["released_active_frames"], 3)

    def test_all_alarms_clear_releases_before_minimum_dwell(self):
        backend = ConcurrentOnlyBackend(
            latency_frames=1,
            hold_frames=10,
            mode_mapping={"flock_splitting": "COHESION_PRIORITY"},
        )
        settings = self.settings(HIGH_LEVEL_MIN_DWELL_FRAMES=5)
        controller = HighLevelController(
            low_level_controller=LowLevelController(settings),
            backend=backend,
            settings=settings,
        )

        both = ("flock_splitting", "stagnation")
        controller.decide(self.observation(0, both))
        controller.decide(self.observation(1, both))
        released = controller.decide(self.observation(2, ()))

        self.assertEqual(released.metadata["high_level_status"], "idle")
        self.assertEqual(
            released.metadata["release_reason"], "all_alarms_cleared"
        )
        self.assertEqual(released.metadata["released_active_frames"], 1)

    def test_factory_defaults(self):
        self.assertIsInstance(build_controller("rule"), LowLevelController)
        self.assertIsInstance(build_controller("mock-llm"), HighLevelController)
        with self.assertRaises(ValueError):
            build_controller("unknown")

    def test_factory_builds_deepseek_controller_without_network_request(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            controller = build_controller("deepseek")
        try:
            self.assertIsInstance(controller, HighLevelController)
            self.assertEqual(controller.backend.source, "deepseek")
        finally:
            controller.close()

    def test_environment_logs_pending_and_active_provenance(self):
        controller, _ = self.controller(latency=1, hold=2)
        env = SwarmEnv(controller=controller)
        env.goal_pos = np.array([180., 180.])
        env.sheep_pos = np.array([[20., 20.], [60., 20.]])
        env.sheep_vel = np.zeros((2, 2))
        env.sheep_status = np.array(["Normal", "Normal"], dtype=object)
        env.step(["flock_splitting"])
        env.step(["flock_splitting"])
        first, second = env.control_data
        self.assertEqual(first["high_level_status"], "pending")
        self.assertEqual(first["decision_source"], "rule")
        self.assertEqual(second["high_level_status"], "active")
        self.assertEqual(second["decision_source"], "mock_llm")
        self.assertEqual(second["llm_latency_frames"], 1)
        self.assertEqual(second["llm_hold_frames"], 2)
        self.assertEqual(second["llm_response_hold_frames"], 2)


if __name__ == "__main__":
    unittest.main()
