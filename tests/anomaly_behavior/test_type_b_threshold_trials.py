import unittest

from experiments.anomaly_behavior.run_type_b_threshold_trials import (
    compare_shared_frames,
    threshold_counts,
)


def state(frame, max_spread, mean_spread):
    return {
        "state_frame": frame,
        "max_spread": max_spread,
        "mean_spread": mean_spread,
    }


class TypeBThresholdTrialsTests(unittest.TestCase):
    def test_threshold_components_are_counted_independently_after_start(self):
        counts = threshold_counts([
            state(0, 50, 20),
            state(30, 41, 14),
            state(31, 39, 16),
            state(32, 41, 16),
            state(33, 40, 15),
        ])
        self.assertEqual(counts, {
            "frames": 4,
            "max_only": 1,
            "mean_only": 1,
            "both": 1,
        })

    def test_paired_comparison_uses_only_shared_frames(self):
        normal = [state(0, 10, 5), state(30, 30, 10), state(31, 35, 12)]
        type_b = [
            state(0, 10, 5), state(30, 45, 18),
            state(31, 42, 17), state(32, 60, 25),
        ]
        result = compare_shared_frames(normal, type_b)
        self.assertEqual(result["shared_post_start_frames"], 2)
        self.assertEqual(result["normal_split_frames"], 0)
        self.assertEqual(result["type_b_split_frames"], 2)
        self.assertAlmostEqual(result["mean_spread_difference"], 6.5)


if __name__ == "__main__":
    unittest.main()
