import unittest

import numpy as np

from environment import SwarmEnv
from run_experiments import CASES, temporary_config
from tools.diagnose_failures import diagnose


class GoalRetentionTests(unittest.TestCase):
    def test_light_retention_is_the_default(self):
        import config

        self.assertTrue(config.GOAL_RETENTION_ENABLED)
        self.assertEqual(config.GOAL_RETENTION_INNER_FRACTION, 0.75)
        self.assertEqual(config.GOAL_RETENTION_WEIGHT, 0.5)
        self.assertEqual(config.GOAL_RETENTION_VELOCITY_DAMPING, 0.90)

    def make_env(self):
        env = SwarmEnv()
        env.goal_pos = np.array([100.0, 100.0])
        env.sheep_pos = np.array([[119.0, 100.0], [150.0, 100.0]])
        env.sheep_vel = np.array([[0.7, 0.0], [0.0, 0.0]])
        env.sheep_status = np.array(["Normal", "Normal"], dtype=object)
        env.dog_pos = np.array([0.0, 0.0])
        return env

    def test_retention_reduces_outward_motion_near_goal_edge(self):
        with temporary_config(
            GOAL_RETENTION_ENABLED=False,
            REPAIR_ENABLED=False,
        ):
            baseline = self.make_env()
            baseline.step([])
            baseline_x = baseline.sheep_pos[0, 0]

        with temporary_config(
            GOAL_RETENTION_ENABLED=True,
            REPAIR_ENABLED=False,
        ):
            retained = self.make_env()
            retained.step([])

        self.assertLess(retained.sheep_pos[0, 0], baseline_x)
        self.assertEqual(retained.sheep_status.tolist(), ["Normal", "Normal"])

    def test_disabled_option_preserves_baseline_trajectory(self):
        with temporary_config(GOAL_RETENTION_ENABLED=False):
            first, _, _ = diagnose(CASES[8], 70, 200)
            second, _, _ = diagnose(CASES[8], 70, 200)

        self.assertEqual(first["trajectory_sha256"], second["trajectory_sha256"])


if __name__ == "__main__":
    unittest.main()
