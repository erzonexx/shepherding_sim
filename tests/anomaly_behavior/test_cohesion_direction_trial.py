import unittest

import numpy as np

import config
from control_types import ControlObservation
from low_level_controller import LowLevelController
from run_experiments import temporary_config
from experiments.anomaly_behavior.run_cohesion_direction_trial import (
    UnitCohesionController,
)


class CohesionDirectionTrialTests(unittest.TestCase):
    def test_candidate_keeps_full_standoff_for_predictive_nonfarthest_target(self):
        positions = np.array([[0., 0.], [10., 0.], [-9., 0.]])
        observation = ControlObservation(
            sheep_positions=positions,
            sheep_velocities=np.array([[0., 0.], [0., 0.], [-0.1, 0.]]),
            dog_position=np.array([0., -10.]),
            goal_position=np.array([100., 0.]),
            alarms=("flock_splitting",),
        )
        with temporary_config(
            REPAIR_TARGET_SELECTION_ENABLED=True,
            COHESION_REPAIR_DISTANCE=10.0,
            REPAIR_TARGET_LOOKAHEAD_FRAMES=20,
        ):
            original = LowLevelController().decide(observation)
            candidate = UnitCohesionController().decide(observation)
        self.assertEqual(original.target_sheep_id, 2)
        self.assertEqual(candidate.target_sheep_id, 2)
        self.assertLess(
            np.linalg.norm(original.drive_point - positions[2]),
            config.COHESION_REPAIR_DISTANCE,
        )
        self.assertAlmostEqual(
            np.linalg.norm(candidate.drive_point - positions[2]),
            config.COHESION_REPAIR_DISTANCE,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
