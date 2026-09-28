import unittest

from anomaly_behavior.run_repair_parameter_screen import summarize


class RepairParameterScreenTests(unittest.TestCase):
    def test_summary_counts_paired_rescues_and_regressions(self):
        rows = [
            {"seed": 0, "scenario": "type_b", "variant": "current",
             "success": False, "capped_frames": 100, "goal_exits": 1},
            {"seed": 1, "scenario": "type_b", "variant": "current",
             "success": True, "capped_frames": 40, "goal_exits": 0},
            {"seed": 0, "scenario": "type_b", "variant": "guide_6",
             "success": True, "capped_frames": 30, "goal_exits": 0},
            {"seed": 1, "scenario": "type_b", "variant": "guide_6",
             "success": False, "capped_frames": 100, "goal_exits": 2},
        ]
        result = summarize(rows)["type_b"]["guide_6"]
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["mean_capped_frames"], 65)


if __name__ == "__main__":
    unittest.main()
