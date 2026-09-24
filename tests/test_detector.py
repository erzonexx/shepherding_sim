import unittest

import numpy as np

from detector import Detector


class DetectorTests(unittest.TestCase):
    def test_splitting_and_stagnation_can_be_reported_together(self):
        detector = Detector(
            dispersion_threshold=40.0,
            mean_spread_threshold=15.0,
            stagnation_frames=3,
            stagnation_threshold=1.0,
        )
        sheep = np.array([[-50.0, 0.0], [50.0, 0.0]])
        goal = np.array([100.0, 0.0])

        detector.detect(sheep, goal)
        detector.detect(sheep, goal)
        alarms, metrics, report = detector.detect(sheep, goal)

        self.assertEqual(alarms, ["flock_splitting", "stagnation"])
        self.assertEqual(report["danger_types"], alarms)
        self.assertEqual(report["danger_type"], "flock_splitting")
        self.assertEqual(metrics["is_danger"], 1)

    def test_normal_output_remains_compatible(self):
        detector = Detector(40.0, 15.0, 3, 1.0)
        compact = np.array([[0.0, 0.0], [1.0, 0.0]])
        goal = np.array([100.0, 0.0])

        alarms, _, report = detector.detect(compact, goal)

        self.assertEqual(alarms, [])
        self.assertEqual(report["danger_type"], "None")
        self.assertEqual(report["danger_types"], [])


if __name__ == "__main__":
    unittest.main()
