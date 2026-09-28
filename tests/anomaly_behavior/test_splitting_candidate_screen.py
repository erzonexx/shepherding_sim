import unittest

from anomaly_behavior.run_splitting_candidate_screen import (
    candidate_masks,
    summarize_masks,
)


def state(frame, max_spread, mean_spread):
    return {
        "state_frame": frame,
        "max_spread": max_spread,
        "mean_spread": mean_spread,
    }


class SplittingCandidateScreenTests(unittest.TestCase):
    def test_streak_resets_and_does_not_count_prestart_frames(self):
        states = [state(frame, 41, 10) for frame in range(27, 35)]
        states.extend([state(35, 39, 10)])
        states.extend(state(frame, 41, 10) for frame in range(36, 46))
        masks = candidate_masks(states)
        self.assertEqual(sum(masks["max_immediate"]), 15)
        self.assertEqual(sum(masks["max_5"]), 7)
        self.assertEqual(sum(masks["max_10"]), 1)
        self.assertEqual(sum(masks["current"]), 0)

    def test_concurrent_frames_use_existing_stagnation_alarm(self):
        states = [state(30, 41, 16), state(31, 41, 10)]
        actions = [
            {"alarm": "flock_splitting;stagnation"},
            {"alarm": "stagnation"},
        ]
        by_name = {
            row["candidate"]: row
            for row in summarize_masks(0, "mixed", False, states, actions)
        }
        self.assertEqual(by_name["current"]["concurrent_with_stagnation_frames"], 1)
        self.assertEqual(by_name["max_immediate"]["concurrent_with_stagnation_frames"], 2)


if __name__ == "__main__":
    unittest.main()
