import unittest

import numpy as np

from experiments.anomaly_behavior.run_splitting_repair_trial import SustainedMaxDetector


class SplittingRepairTrialTests(unittest.TestCase):
    def test_candidate_only_triggers_after_warmup_and_ten_consecutive_frames(self):
        detector = SustainedMaxDetector()
        positions = np.zeros((20, 2))
        positions[-1, 0] = 100.0
        goal = np.array([180.0, 180.0])
        for frame in range(40):
            alarms, metrics, report = detector.detect(positions, goal)
            self.assertEqual("flock_splitting" in alarms, frame == 39)
            self.assertEqual(metrics["is_danger"], int(bool(alarms)))
            self.assertEqual(report["danger_types"], alarms)


if __name__ == "__main__":
    unittest.main()
