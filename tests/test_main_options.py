import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

import config
from environment import SwarmEnv
from high_level_controller import HighLevelController, MockLLMBackend, build_controller
from low_level_controller import LowLevelController
from main import SCENARIOS, apply_run_settings, parse_args
from run_experiments import temporary_config


class MainOptionsTests(unittest.TestCase):
    def test_architecture_switch_preserves_scenario_and_initial_state(self):
        for scenario, counts in SCENARIOS.items():
            snapshots = []
            for kind in ("rule", "deepseek"):
                with self.subTest(scenario=scenario, controller=kind), temporary_config():
                    args = parse_args([
                        "--controller", kind, "--scenario", scenario,
                        "--seed", "141", "--repair", "--no-video",
                    ])
                    apply_run_settings(args)
                    with patch("responses_api_backend.DeepSeekBackend", return_value=MockLLMBackend()):
                        controller = build_controller(kind)
                    self.assertIsInstance(
                        controller, LowLevelController if kind == "rule" else HighLevelController
                    )
                    env = SwarmEnv(controller=controller)
                    self.assertEqual(int(np.sum(env.sheep_status == "A")), counts[0])
                    self.assertEqual(int(np.sum(env.sheep_status == "B")), counts[1])
                    self.assertEqual(int(np.sum(env.sheep_status == "C")), counts[2])
                    snapshots.append((env.sheep_pos, env.dog_pos, env.goal_pos, env.sheep_status))
                    env.close()
            for rule_state, llm_state in zip(*snapshots):
                np.testing.assert_array_equal(rule_state, llm_state)

    def test_default_run_preserves_config_settings(self):
        with temporary_config(
            NUM_ABNORMAL_A=1, NUM_ABNORMAL_B=5, REPAIR_ENABLED=False,
            NUM_ABNORMAL_C=2,
            CONTROLLER_KIND="mock-llm", MAX_STEPS=123, USE_FIXED_SEED=False,
            SAVE_ANIMATION_MP4=True,
        ):
            apply_run_settings(parse_args([]))
            self.assertEqual((config.NUM_ABNORMAL_A, config.NUM_ABNORMAL_B), (1, 5))
            self.assertEqual(config.NUM_ABNORMAL_C, 2)
            self.assertFalse(config.REPAIR_ENABLED)
            self.assertFalse(config.USE_FIXED_SEED)
            self.assertTrue(config.SAVE_ANIMATION_MP4)
            self.assertEqual(config.CONTROLLER_KIND, "mock-llm")
            self.assertEqual(config.MAX_STEPS, 123)

    def test_custom_population_overrides_preset_without_changing_repair(self):
        with temporary_config():
            apply_run_settings(parse_args([
                "--scenario", "stress", "--abnormal-a", "1", "--abnormal-b", "6", "--abnormal-c", "4",
                "--no-repair", "--max-steps", "100", "--no-video",
            ]))
            self.assertEqual((config.NUM_ABNORMAL_A, config.NUM_ABNORMAL_B), (1, 6))
            self.assertEqual(config.NUM_ABNORMAL_C, 4)
            self.assertFalse(config.REPAIR_ENABLED)
            self.assertEqual(config.MAX_STEPS, 100)
            self.assertFalse(config.SAVE_ANIMATION_MP4)

    def test_invalid_conditions_rejected_before_simulation(self):
        for argv in (
            ["--abnormal-a", "-1"],
            ["--abnormal-c", "-1"],
            ["--abnormal-a", "7", "--abnormal-b", "7", "--abnormal-c", "7"],
            ["--abnormal-a", "20", "--abnormal-b", "1"],
            ["--max-steps", "0"],
            ["--seed", "-1"],
            ["--seed", "4294967296"],
            ["--seed", "70", "--random-seed"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_explicit_seed_overrides_random_initialization(self):
        with temporary_config(USE_FIXED_SEED=False):
            apply_run_settings(parse_args(["--seed", "42"]))
            self.assertTrue(config.USE_FIXED_SEED)
            self.assertEqual(config.RANDOM_SEED, 42)
            apply_run_settings(parse_args(["--random-seed"]))
            self.assertFalse(config.USE_FIXED_SEED)


if __name__ == "__main__":
    unittest.main()
