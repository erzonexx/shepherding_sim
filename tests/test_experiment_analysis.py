import unittest

import pandas as pd

from run_experiments import count_high_level_override_frames
from tools.analyze_experiment_pair import compare_frames, exact_signed_mean_pvalue


class ExperimentAnalysisTests(unittest.TestCase):
    def row(self, seed, success, frames, max_disp):
        return {
            "Case": "C08",
            "Seed": seed,
            "Success": success,
            "Frames": frames,
            "FinalDist": 10.0,
            "MinDist": 5.0,
            "MaxDisp": max_disp,
            "DangerFrames": 100,
            "PeakInsideGoal": 19,
            "FinalInsideGoal": 10,
            "GoalExits": 2,
        }

    def test_comparison_counts_success_flip_and_metric_delta(self):
        baseline = pd.DataFrame([self.row(1, False, 2000, 100.0)])
        candidate = pd.DataFrame([self.row(1, True, 1200, 70.0)])

        paired, summary = compare_frames(baseline, candidate)

        self.assertEqual(summary["failure_to_success"], 1)
        self.assertEqual(summary["success_to_failure"], 0)
        self.assertEqual(summary["metrics"]["Frames"]["mean_delta"], -800.0)
        self.assertEqual(float(paired.iloc[0]["Delta_MaxDisp"]), -30.0)

    def test_mismatched_keys_are_rejected(self):
        baseline = pd.DataFrame([self.row(1, False, 2000, 100.0)])
        candidate = pd.DataFrame([self.row(2, True, 1200, 70.0)])
        with self.assertRaises(ValueError):
            compare_frames(baseline, candidate)

    def test_zero_differences_have_unit_pvalue(self):
        self.assertEqual(exact_signed_mean_pvalue([0, 0, 0]), 1.0)

    def test_override_frames_exclude_llm_choices_matching_rule_fallback(self):
        control = pd.DataFrame([
            {"decision_source": "rule", "alarms": "flock_splitting;stagnation",
             "current_mode": "HARD_TO_GUIDE"},
            {"decision_source": "deepseek", "alarms": "flock_splitting;stagnation",
             "current_mode": "HARD_TO_GUIDE"},
            {"decision_source": "deepseek", "alarms": "flock_splitting;stagnation",
             "current_mode": "COHESION_PRIORITY"},
            {"decision_source": "deepseek", "alarms": "stagnation",
             "current_mode": "HARD_TO_GUIDE"},
            {"decision_source": "deepseek", "alarms": "stagnation",
             "current_mode": "COHESION_PRIORITY"},
        ])

        self.assertEqual(count_high_level_override_frames(control, True), 2)
        self.assertEqual(count_high_level_override_frames(control, False), 4)


if __name__ == "__main__":
    unittest.main()
