import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
from main import SCENARIOS
from run_experiments import (
    CASES, main, parse_args, select_cases, select_seeds,
    write_experiment_metadata,
)


class BatchOptionsTests(unittest.TestCase):
    def test_seed_plan_and_population_are_independent_of_architecture(self):
        for scenario, counts in SCENARIOS.items():
            rule = parse_args([
                "--scenario", scenario, "--controller", "rule", "--runs", "10", "--repair",
            ])
            llm = parse_args([
                "--scenario", scenario, "--controller", "deepseek", "--runs", "10", "--repair",
            ])
            with self.subTest(scenario=scenario):
                self.assertEqual(select_seeds(rule), select_seeds(llm))
                self.assertEqual(len(set(select_seeds(rule))), 10)
                self.assertEqual(select_cases(rule), select_cases(llm))
                case = select_cases(rule)[0]
                self.assertEqual((case["num_abnormal_a"], case["num_abnormal_b"], case["num_abnormal_c"]), counts)
                self.assertTrue(case["repair_enabled"])

    def test_mode_only_permission_is_scoped_and_recorded(self):
        original = config.HIGH_LEVEL_TARGET_SELECTION_ENABLED
        def inspect_batch(args):
            with tempfile.TemporaryDirectory() as directory:
                path = write_experiment_metadata(Path(directory) / "summary.csv", args,
                                                 select_seeds(args), select_cases(args))
                return json.loads(path.read_text(encoding="utf-8"))
        with patch("run_experiments.run_batch", side_effect=inspect_batch):
            metadata = main(["--controller", "deepseek", "--llm-mode-only"])
        self.assertFalse(metadata["config"]["HIGH_LEVEL_TARGET_SELECTION_ENABLED"])
        self.assertEqual(config.HIGH_LEVEL_TARGET_SELECTION_ENABLED, original)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--controller", "rule", "--llm-mode-only"])

    def test_target_only_permission_is_scoped_and_independent_of_population(self):
        original = (config.HIGH_LEVEL_MODE_SELECTION_ENABLED,
                    config.HIGH_LEVEL_TARGET_SELECTION_ENABLED)
        with patch("run_experiments.run_batch", side_effect=lambda args: (
            config.HIGH_LEVEL_MODE_SELECTION_ENABLED,
            config.HIGH_LEVEL_TARGET_SELECTION_ENABLED,
            select_seeds(args), select_cases(args),
        )):
            observed = main(["--controller", "deepseek", "--llm-target-only",
                             "--scenario", "custom", "--abnormal-a", "5",
                             "--abnormal-b", "5", "--abnormal-c", "5",
                             "--runs", "50", "--seed-mode", "sequence", "--batch-seed", "3301"])
        self.assertEqual(observed[:2], (False, True))
        self.assertEqual(observed[2], list(range(3301, 3351)))
        self.assertEqual(observed[3][0]["num_abnormal_c"], 5)
        self.assertEqual((config.HIGH_LEVEL_MODE_SELECTION_ENABLED,
                          config.HIGH_LEVEL_TARGET_SELECTION_ENABLED), original)
        for args in (["--llm-target-only"],
                     ["--controller", "deepseek", "--llm-mode-only", "--llm-target-only"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_random_groups_are_repeatable_and_sequence_is_explicit(self):
        first = select_seeds(parse_args(["--runs", "10", "--batch-seed", "141"]))
        self.assertEqual(first, select_seeds(parse_args(["--runs", "10", "--batch-seed", "141"])))
        self.assertNotEqual(first, select_seeds(parse_args(["--runs", "10", "--batch-seed", "142"])))
        self.assertEqual(
            select_seeds(parse_args(["--runs", "3", "--seed-mode", "sequence", "--batch-seed", "141"])),
            [141, 142, 143],
        )

    def test_original_batch_defaults_and_cases_are_preserved(self):
        args = parse_args([])
        self.assertEqual(select_seeds(args), [70])
        self.assertEqual(select_cases(args), CASES)
        self.assertEqual(select_cases(parse_args(["--cases", "C08"])), [CASES[8]])
        self.assertEqual(select_seeds(parse_args(["--seeds", "141", "142", "141"])), [141, 142])

    def test_custom_counts_and_repair_off(self):
        args = parse_args([
            "--scenario", "custom", "--abnormal-a", "1", "--abnormal-b", "6", "--abnormal-c", "4",
            "--controller", "deepseek", "--no-repair", "--export", "video",
        ])
        case = select_cases(args)[0]
        self.assertEqual((case["num_abnormal_a"], case["num_abnormal_b"]), (1, 6))
        self.assertEqual(case["num_abnormal_c"], 4)
        self.assertFalse(case["repair_enabled"])
        self.assertTrue(args.visuals and args.analysis and args.video)

    def test_invalid_batch_plans_are_rejected(self):
        for argv in (
            ["--runs", "0"], ["--max-steps", "0"], ["--max-log-files", "0"],
            ["--runs", "2", "--seed", "70"], ["--seeds", "-1"],
            ["--batch-seed", "4294967296"],
            ["--runs", "2", "--seed-mode", "sequence", "--batch-seed", "4294967295"],
            ["--scenario", "stress", "--cases", "C08"],
            ["--scenario", "custom", "--abnormal-a", "-1"],
            ["--scenario", "custom", "--abnormal-c", "-1"],
            ["--scenario", "custom", "--abnormal-a", "7", "--abnormal-b", "7", "--abnormal-c", "7"],
            ["--cases", "C08", "--abnormal-c", "1"],
            ["--scenario", "custom", "--abnormal-a", "20", "--abnormal-b", "1"],
            ["--cases", "C08", "--no-repair"],
            ["--concurrent-min-frames", "0"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_metadata_records_actual_population_and_generated_seeds(self):
        args = parse_args(["--scenario", "stress", "--runs", "10", "--repair"])
        seeds, cases = select_seeds(args), select_cases(args)
        with tempfile.TemporaryDirectory() as directory:
            path = write_experiment_metadata(Path(directory) / "summary.csv", args, seeds, cases)
            metadata = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["seeds"], seeds)
        self.assertEqual(metadata["case_settings"], cases)
        self.assertEqual(metadata["seed_plan"], {"mode": "random", "batch_seed": 70})
        self.assertIn("run_experiments.py", metadata["source_sha256"])

    def test_trigger_controls_are_independent_of_scenario(self):
        args = parse_args([
            "--scenario", "stress", "--controller", "deepseek",
            "--concurrent-min-frames", "3",
            "--one-request-per-concurrent-episode",
        ])
        self.assertEqual(args.concurrent_min_frames, 3)
        self.assertTrue(args.one_request_per_concurrent_episode)
        self.assertEqual(select_cases(args)[0]["num_abnormal_a"], 3)
        self.assertEqual(select_cases(args)[0]["num_abnormal_b"], 3)

    def test_trigger_overrides_are_scoped_to_the_batch(self):
        original = (
            config.HIGH_LEVEL_CONCURRENT_MIN_FRAMES,
            config.HIGH_LEVEL_ONE_REQUEST_PER_EPISODE,
        )
        with patch("run_experiments.run_batch", side_effect=lambda _: (
            config.HIGH_LEVEL_CONCURRENT_MIN_FRAMES,
            config.HIGH_LEVEL_ONE_REQUEST_PER_EPISODE,
        )):
            observed = main([
                "--concurrent-min-frames", "3",
                "--one-request-per-concurrent-episode",
            ])
        self.assertEqual(observed, (3, True))
        self.assertEqual(
            (config.HIGH_LEVEL_CONCURRENT_MIN_FRAMES,
             config.HIGH_LEVEL_ONE_REQUEST_PER_EPISODE),
            original,
        )


if __name__ == "__main__":
    unittest.main()
