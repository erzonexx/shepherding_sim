"""Tests for the read-only alarm coverage audit."""

import unittest

from tests.anomaly_behavior.run_alarm_coverage_audit import (
    alarm_category,
    concurrent_episode_lengths,
    summarize,
    summarize_run,
)


class AlarmCoverageAuditTests(unittest.TestCase):
    def test_alarm_categories_are_mutually_exclusive(self):
        expected = (
            ("none", "none"),
            ("flock_splitting", "splitting_only"),
            ("stagnation", "stagnation_only"),
            ("flock_splitting;stagnation", "concurrent"),
        )
        for alarm, category in expected:
            with self.subTest(alarm=alarm):
                self.assertEqual(alarm_category({"alarm": alarm}), category)

    def test_run_summary_records_first_and_last_dual_alarm(self):
        report = {
            "seed": 1,
            "success_reported_by_env": False,
            "peak_inside_goal": 10,
            "final_inside_goal": 7,
        }
        states = [{"max_spread": 2}, {"max_spread": 3}]
        actions = [
            {"alarm": "none"},
            {"alarm": "flock_splitting;stagnation"},
            {"alarm": "stagnation"},
            {"alarm": "flock_splitting;stagnation"},
        ]
        row = summarize_run(report, states, actions)
        self.assertEqual(row["first_concurrent_frame"], 1)
        self.assertEqual(row["last_concurrent_frame"], 3)
        self.assertEqual(row["concurrent_frames"], 2)
        self.assertEqual(row["concurrent_episodes"], 2)
        self.assertEqual(row["longest_concurrent_episode"], 1)
        self.assertEqual(row["max_spread"], 3)

    def test_concurrent_episodes_include_final_run(self):
        self.assertEqual(
            concurrent_episode_lengths([
                "none", "concurrent", "concurrent", "stagnation_only",
                "concurrent", "concurrent", "concurrent",
            ]),
            [2, 3],
        )

    def test_aggregate_distinguishes_unreachable_failures(self):
        rows = [
            {"scenario": "type_b", "seed": 1, "success": False,
             "concurrent_frames": 0, "first_concurrent_frame": None,
             "longest_concurrent_episode": 0,
             "none_frames": 1, "splitting_only_frames": 2,
             "stagnation_only_frames": 3},
            {"scenario": "type_b", "seed": 2, "success": False,
             "concurrent_frames": 30, "first_concurrent_frame": 42,
             "longest_concurrent_episode": 10,
             "none_frames": 1, "splitting_only_frames": 2,
             "stagnation_only_frames": 3},
            {"scenario": "type_b", "seed": 3, "success": True,
             "concurrent_frames": 4, "first_concurrent_frame": 10,
             "longest_concurrent_episode": 1,
             "none_frames": 1, "splitting_only_frames": 2,
             "stagnation_only_frames": 3},
        ]
        result = summarize(rows)["type_b"]
        self.assertEqual(result["failures_with_concurrent_alarm"], 1)
        self.assertEqual(result["failures_without_concurrent_alarm"], 1)
        self.assertEqual(result["failures_with_at_least_30_concurrent_frames"], 1)
        self.assertEqual(result["failures_with_10_consecutive_concurrent_frames"], 1)
        self.assertEqual(
            result["median_first_concurrent_frame_in_failed_dual_alarm_runs"],
            42,
        )


if __name__ == "__main__":
    unittest.main()
