import unittest

import config
from experiments.anomaly_behavior.run_repair_target_selection_screen import (
    run_screen,
    same_mode_switches,
    summarize,
)


class RepairTargetSelectionScreenTests(unittest.TestCase):
    def test_switches_count_only_within_same_repair_mode(self):
        actions = [
            {"mode": mode, "target_id": target}
            for mode, target in (
                ("HARD_TO_GUIDE", 0),
                ("HARD_TO_GUIDE", 1),
                ("COHESION_PRIORITY", 2),
                ("COHESION_PRIORITY", 3),
                ("TARGET_TRACKING", None),
                ("HARD_TO_GUIDE", 2),
            )
        ]
        self.assertEqual(same_mode_switches(actions, "HARD_TO_GUIDE"), 1)
        self.assertEqual(same_mode_switches(actions, "COHESION_PRIORITY"), 1)

    def test_summary_uses_paired_seed_outcomes(self):
        rows = [
            {"seed": seed, "scenario": "type_b", "variant": variant,
             "success": success, "capped_frames": 100,
             "goal_exits": 0, "peak_inside_goal": 2,
             "guide_target_switches": 1, "cohesion_target_switches": 0}
            for seed, variant, success in (
                (0, "current", False), (1, "current", True),
                (0, "guide_hold_0", True), (1, "guide_hold_0", False),
            )
        ]
        result = summarize(rows)["type_b"]["guide_hold_0"]
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["regressed"], 1)

    def test_small_trial_restores_parameter_and_keeps_cases_separate(self):
        original = config.HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES
        rows, summary = run_screen(
            first_seed=470, seed_count=1, max_steps=40,
            variant_names=("current", "guide_hold_0"),
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(config.HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES, original)
        self.assertEqual(summary["type_b"]["current"]["runs"], 1)
        self.assertEqual(summary["mixed"]["guide_hold_0"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
