"""Tests for the read-only individual stagnation comparison."""

import unittest

import numpy as np

from experiments.anomaly_behavior.run_individual_stagnation_audit import (
    stalled_outside_goal,
)


class IndividualStagnationAuditTests(unittest.TestCase):
    def test_only_outside_sheep_below_progress_threshold_are_stalled(self):
        actual = stalled_outside_goal(
            np.array([30.0, 40.0, 18.0, 32.0]),
            np.array([29.5, 38.0, 18.2, 34.0]),
            goal_radius=20.0,
            progress_threshold=1.0,
        )
        np.testing.assert_array_equal(actual, [True, False, False, True])

    def test_strict_progress_cutoff_matches_detector(self):
        actual = stalled_outside_goal(
            np.array([30.0]), np.array([29.0]), 20.0, 1.0
        )
        np.testing.assert_array_equal(actual, [False])


if __name__ == "__main__":
    unittest.main()
