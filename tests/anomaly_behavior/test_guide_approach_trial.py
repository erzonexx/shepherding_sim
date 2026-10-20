import unittest

import numpy as np

from control_types import ControlObservation
from experiments.anomaly_behavior.run_guide_approach_trial import (
    LateralGuideController,
)


class GuideApproachTrialTests(unittest.TestCase):
    def observation(self, dog, alarms=("stagnation",)):
        return ControlObservation(
            sheep_positions=np.array([[50., 50.], [80., 80.]]),
            dog_position=np.array(dog),
            goal_position=np.array([100., 50.]),
            alarms=alarms,
        )

    def test_goal_side_dog_sidesteps_before_circling(self):
        controller = LateralGuideController(radius=20.0)
        first = controller.decide(self.observation([60., 50.]))
        self.assertEqual(first.mode, "HARD_TO_GUIDE")
        self.assertEqual(first.target_sheep_id, 0)
        self.assertEqual(first.metadata["approach_phase"], "sidestep")
        np.testing.assert_allclose(first.navigation_point, [60., 70.])

        second = controller.decide(self.observation([60., 75.]))
        self.assertEqual(second.metadata["approach_phase"], "circle")
        self.assertLess(second.navigation_point[0], 50.)
        self.assertGreaterEqual(second.navigation_point[1], 70.)

        behind = controller.decide(self.observation([40., 50.]))
        self.assertEqual(behind.metadata["approach_phase"], "behind")
        np.testing.assert_allclose(
            behind.navigation_point, behind.drive_point
        )

    def test_other_modes_use_existing_navigation(self):
        controller = LateralGuideController(radius=20.0)
        decision = controller.decide(self.observation([60., 50.], alarms=()))
        self.assertEqual(decision.mode, "TARGET_TRACKING")
        np.testing.assert_allclose(
            decision.navigation_point, decision.drive_point
        )

    def test_near_gate_does_not_detour_from_far_away(self):
        controller = LateralGuideController(
            radius=20.0, activation_distance=50.0
        )
        far = controller.decide(self.observation([110., 50.]))
        self.assertEqual(far.metadata["approach_phase"], "far_direct")
        np.testing.assert_allclose(far.navigation_point, far.drive_point)
        near = controller.decide(self.observation([90., 50.]))
        self.assertEqual(near.metadata["approach_phase"], "sidestep")

    def test_near_goal_gate_only_detours_with_enough_sheep_inside(self):
        controller = LateralGuideController(
            radius=10.0, min_inside_goal=2
        )
        outside = controller.decide(self.observation([60., 50.]))
        self.assertNotIn("approach_phase", outside.metadata)
        np.testing.assert_allclose(outside.navigation_point, outside.drive_point)

        observation = self.observation([85., 50.])
        observation = ControlObservation(
            sheep_positions=np.array([[82., 50.], [85., 55.], [60., 50.]]),
            dog_position=observation.dog_position,
            goal_position=observation.goal_position,
            alarms=observation.alarms,
        )
        controller = LateralGuideController(
            radius=10.0, min_inside_goal=2
        )
        near_goal = controller.decide(observation)
        self.assertEqual(near_goal.metadata["approach_phase"], "sidestep")


if __name__ == "__main__":
    unittest.main()
