import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from dynamics import advance_sheep
from run_experiments import temporary_config
from anomaly_behavior.behavior_characterization import (
    build_report,
    dog_response_curve,
    normal_initial_split_screen,
    type_b_dispersion_trial,
    write_characterization,
)


class AnomalyBehaviorCharacterizationTests(unittest.TestCase):
    def test_temporary_anomaly_parameters_are_restored(self):
        initial_response = config.WEIGHT_DOG_REPULSION_A_FACTOR
        with temporary_config(
            WEIGHT_DOG_REPULSION_A_FACTOR=initial_response / 2,
        ):
            self.assertEqual(
                config.WEIGHT_DOG_REPULSION_A_FACTOR,
                initial_response / 2,
            )
        self.assertEqual(config.WEIGHT_DOG_REPULSION_A_FACTOR, initial_response)

    def test_dog_response_trial_is_deterministic_and_respects_sensing_range(self):
        first = dog_response_curve((20.0, 60.0))
        second = dog_response_curve((20.0, 60.0))
        self.assertEqual(first, second)
        rows = {(row["status"], row["dog_distance"]): row for row in first}
        self.assertGreater(rows[("Normal", 20.0)]["speed"], 0.0)
        self.assertGreater(rows[("A", 20.0)]["speed"], 0.0)
        self.assertLess(rows[("A", 20.0)]["speed"], rows[("Normal", 20.0)]["speed"])
        self.assertEqual(rows[("Normal", 60.0)]["speed"], 0.0)
        self.assertEqual(rows[("A", 60.0)]["speed"], 0.0)
        self.assertAlmostEqual(
            rows[("A", 20.0)]["speed"] / rows[("Normal", 20.0)]["speed"],
            config.WEIGHT_DOG_REPULSION_A_FACTOR,
        )

    def test_type_a_follows_the_same_flock_dynamics_without_a_dog(self):
        initial = np.array([[2.0, 100.0], [8.0, 101.0], [12.0, 99.0]])
        trajectories = []
        with temporary_config(GOAL_RETENTION_ENABLED=False):
            for focus_status in ("Normal", "A"):
                positions = initial.copy()
                velocities = np.zeros_like(positions)
                statuses = np.array([focus_status, "Normal", "Normal"])
                for _ in range(30):
                    positions, velocities = advance_sheep(
                        positions,
                        velocities,
                        statuses,
                        np.array([-1000.0, -1000.0]),
                        np.array([180.0, 180.0]),
                        config,
                    )
                trajectories.append((positions, velocities))
        for normal, type_a in zip(*trajectories):
            np.testing.assert_allclose(type_a, normal)

    def test_type_b_trial_starts_from_identical_geometry(self):
        rows = type_b_dispersion_trial(frames=10, sample_frames=(0, 10))
        initial = [row for row in rows if row["frame"] == 0]
        self.assertEqual(len(initial), 2)
        for metric in ("focus_distance_to_centroid", "mean_spread", "max_spread"):
            self.assertEqual(initial[0][metric], initial[1][metric])
        self.assertEqual(rows, type_b_dispersion_trial(frames=10, sample_frames=(0, 10)))

    def test_normal_initialization_screen_is_repeatable(self):
        first = normal_initial_split_screen(10)
        second = normal_initial_split_screen(10)
        self.assertEqual(first, second)
        self.assertEqual([row["seed"] for row in first], list(range(10)))
        self.assertTrue(all(isinstance(row["splitting_alarm"], bool) for row in first))

    def test_report_flags_are_derived_from_measured_values(self):
        report, _, _, _ = build_report(initial_seed_count=20, dispersion_frames=10)
        ratio = report["type_a"]["type_a_to_normal_response_ratio"]
        split_rate = report["normal_initialization"]["frame_zero_splitting_rate"]
        self.assertEqual(
            report["review_flags"]["type_a_response_below_5_percent_of_normal"],
            ratio < 0.05,
        )
        self.assertEqual(
            report["review_flags"]["normal_initial_split_rate_above_5_percent"],
            split_rate > 0.05,
        )

    def test_outputs_stay_inside_the_requested_report_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = write_characterization(
                output_root=directory,
                initial_seed_count=5,
                dispersion_frames=3,
            )
            self.assertEqual(output.parent, Path(directory))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "report.json",
                    "type_a_dog_response.csv",
                    "type_b_dispersion.csv",
                    "normal_initial_split.csv",
                },
            )


if __name__ == "__main__":
    unittest.main()
