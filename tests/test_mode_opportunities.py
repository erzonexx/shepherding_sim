import unittest

import numpy as np

from control_types import ControlObservation
from tools.evaluate_mode_opportunities import FirstDualAlarmController, run_trial


def observation(frame, alarms):
    return ControlObservation(
        sheep_positions=np.array([[40.0, 80.0], [55.0, 85.0]]),
        dog_position=np.array([20.0, 80.0]),
        goal_position=np.array([150.0, 150.0]),
        sheep_statuses=("Normal", "Normal"),
        alarms=alarms,
        frame=frame,
    )


class ModeOpportunityTests(unittest.TestCase):
    def test_one_bounded_intervention_at_first_dual_alarm(self):
        controller = FirstDualAlarmController("COHESION_PRIORITY", 3)
        self.assertEqual(
            controller.decide(observation(6, ("stagnation",))).mode,
            "HARD_TO_GUIDE",
        )
        self.assertIsNone(controller.trigger_frame)
        self.assertEqual(
            controller.decide(observation(7, ("stagnation", "flock_splitting"))).mode,
            "COHESION_PRIORITY",
        )
        self.assertEqual(controller.trigger_frame, 7)
        self.assertEqual(
            controller.decide(observation(8, ())).mode,
            "COHESION_PRIORITY",
        )
        self.assertEqual(
            controller.decide(observation(10, ("stagnation", "flock_splitting"))).mode,
            "HARD_TO_GUIDE",
        )
        self.assertEqual(controller.override_frames, 2)

    def test_rule_arm_has_no_override(self):
        controller = FirstDualAlarmController("RULE", 30)
        decision = controller.decide(
            observation(0, ("stagnation", "flock_splitting"))
        )
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(controller.trigger_frame, 0)
        self.assertEqual(controller.override_frames, 0)

    def test_delayed_intervention_requires_dual_alarm_at_start(self):
        controller = FirstDualAlarmController(
            "COHESION_PRIORITY", 3, delay_frames=2,
            require_dual_at_start=True,
        )
        controller.decide(observation(7, ("stagnation", "flock_splitting")))
        controller.decide(observation(8, ("stagnation", "flock_splitting")))
        self.assertEqual(
            controller.decide(observation(9, ("stagnation",))).mode,
            "HARD_TO_GUIDE",
        )
        self.assertTrue(controller.skipped)
        self.assertIsNone(controller.intervention_started_frame)
        self.assertEqual(
            controller.decide(observation(10, ("stagnation", "flock_splitting"))).mode,
            "HARD_TO_GUIDE",
        )

    def test_delayed_intervention_starts_once_and_expires(self):
        controller = FirstDualAlarmController(
            "COHESION_PRIORITY", 2, delay_frames=2,
            require_dual_at_start=True,
        )
        controller.decide(observation(7, ("stagnation", "flock_splitting")))
        controller.decide(observation(8, ("stagnation", "flock_splitting")))
        self.assertEqual(
            controller.decide(observation(9, ("stagnation", "flock_splitting"))).mode,
            "COHESION_PRIORITY",
        )
        self.assertEqual(controller.intervention_started_frame, 9)
        self.assertEqual(
            controller.decide(observation(11, ("stagnation", "flock_splitting"))).mode,
            "HARD_TO_GUIDE",
        )

    def test_rule_arm_matches_recorded_baseline(self):
        result = run_trial(2203, "RULE", 30, 2000)
        self.assertTrue(result["Success"])
        self.assertEqual(result["Frames"], 644)
        self.assertEqual(result["FirstDualAlarmFrame"], 229)


if __name__ == "__main__":
    unittest.main()
