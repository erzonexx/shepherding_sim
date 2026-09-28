import unittest

from anomaly_behavior.run_alarm_outcome_trials import profile_alarms, summarize_pairs


def action(frame, alarm):
    return {"from_state_frame": frame, "alarm": alarm}


class AlarmOutcomeTrialsTests(unittest.TestCase):
    def test_sequential_alarms_are_not_concurrent(self):
        result = profile_alarms([
            action(0, "flock_splitting"),
            action(1, "none"),
            action(149, "stagnation"),
        ])
        self.assertEqual(result["alarm_category"], "sequential")
        self.assertEqual(result["concurrent_frames"], 0)
        self.assertEqual(result["first_stagnation_frame"], 149)

    def test_concurrent_alarm_is_classified_separately(self):
        result = profile_alarms([
            action(0, "flock_splitting"),
            action(150, "flock_splitting;stagnation"),
        ])
        self.assertEqual(result["alarm_category"], "concurrent")
        self.assertEqual(result["splitting_only_frames"], 1)
        self.assertEqual(result["concurrent_frames"], 1)

    def test_outcomes_use_baseline_alarm_category(self):
        rows = [
            {
                "seed": 0, "scenario": "mixed", "repair_enabled": False,
                "alarm_category": "sequential", "success": False,
                "capped_frames": 1000,
            },
            {
                "seed": 0, "scenario": "mixed", "repair_enabled": True,
                "alarm_category": "none", "success": True,
                "capped_frames": 300,
            },
        ]
        bucket = summarize_pairs(rows)["mixed"]["sequential"]
        self.assertEqual(bucket["runs"], 1)
        self.assertEqual(bucket["rescued"], 1)
        self.assertEqual(bucket["repair_mean_capped_frames"], 300)


if __name__ == "__main__":
    unittest.main()
