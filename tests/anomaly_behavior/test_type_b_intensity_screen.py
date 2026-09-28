import unittest

import config
from tests.anomaly_behavior.run_type_b_intensity_screen import (
    open_loop,
    run_screen,
)


class TypeBIntensityScreenTests(unittest.TestCase):
    def test_open_loop_is_repeatable_and_restores_settings(self):
        original = (
            config.RANDOM_SEED,
            config.NUM_ABNORMAL_B,
            config.WEIGHT_COHESION_B_FACTOR,
        )
        first = open_loop(
            270, num_b=4, overrides={"WEIGHT_COHESION_B_FACTOR": 0.30},
            frames=35,
        )
        second = open_loop(
            270, num_b=4, overrides={"WEIGHT_COHESION_B_FACTOR": 0.30},
            frames=35,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            original,
            (config.RANDOM_SEED, config.NUM_ABNORMAL_B,
             config.WEIGHT_COHESION_B_FACTOR),
        )

    def test_paired_screen_keeps_architecture_and_behavior_independent(self):
        rows, summary = run_screen(
            first_seed=270, seed_count=1, max_steps=40,
            variant_names=("current", "cohesion_030"),
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {(row["variant"], row["repair"]) for row in rows},
            {(name, repair) for name in ("normal", "current", "cohesion_030")
             for repair in (False, True)},
        )
        self.assertEqual(
            summary["no_repair"]["current"]["runs"], 1
        )
        self.assertEqual(summary["repair"]["cohesion_030"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
