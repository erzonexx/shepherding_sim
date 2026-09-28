import unittest

from tests.anomaly_behavior.run_type_b_failure_trace import (
    summarize_trace,
    target_runs,
)


class TypeBFailureTraceTests(unittest.TestCase):
    def test_target_runs_split_on_mode_and_target_change(self):
        actions = [
            {"mode": mode, "target_id": target}
            for mode, target in (
                ("HARD_TO_GUIDE", 1),
                ("HARD_TO_GUIDE", 1),
                ("HARD_TO_GUIDE", 2),
                ("TARGET_TRACKING", None),
                ("HARD_TO_GUIDE", 2),
            )
        ]
        self.assertEqual(
            [len(run) for run in target_runs(actions, "HARD_TO_GUIDE")],
            [2, 1, 1],
        )

    def test_summary_counts_long_worsening_target_run(self):
        guide = [{
            "mode": "HARD_TO_GUIDE",
            "target_id": 2,
            "target_goal_distance_before": 40.0 + index,
            "dog_behind_target": False,
            "near_drive_point": False,
            "target_outward_dog_response": 0.1,
            "drive_point_outside": True,
        } for index in range(30)]
        guide[0]["alarm"] = "flock_splitting;stagnation"
        cohesion = [{
            "mode": "COHESION_PRIORITY",
            "target_id": 1,
            "target_is_farthest_from_center": False,
            "dog_behind_target": True,
            "drive_point_outside": False,
        }]
        report = {
            "seed": 7,
            "success_reported_by_env": False,
            "peak_inside_goal": 4,
            "final_inside_goal": 2,
            "individual_goal_exits": 1,
        }
        states = [
            {"centroid_distance": 50.0},
            {"centroid_distance": 60.0},
        ]
        result = summarize_trace(report, states, cohesion + guide)
        self.assertEqual(result["cohesion_nonfarthest_frames"], 1)
        self.assertEqual(result["guide_long_target_runs"], 1)
        self.assertEqual(result["guide_worsening_target_runs"], 1)
        self.assertEqual(result["guide_outward_dog_response_share"], 1.0)
        self.assertEqual(result["guide_outside_drive_share"], 1.0)
        self.assertEqual(result["cohesion_outside_drive_share"], 0.0)
        self.assertEqual(result["first_concurrent_frame"], 1)
        self.assertEqual(result["frames_after_first_concurrent"], 30)


if __name__ == "__main__":
    unittest.main()
