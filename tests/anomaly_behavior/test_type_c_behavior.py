import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import config
from control_types import ControlObservation
from dynamics import advance_sheep
from environment import SwarmEnv
from high_level_controller import HighLevelController, MockLLMBackend
from mode_forecast import forecast_modes
from run_experiments import temporary_config
from visualizer import COLOR_MAP, LEGEND_LABELS, Visualizer
from experiments.anomaly_behavior.behavior_characterization import dog_response_curve


class TypeCBehaviorTests(unittest.TestCase):
    def test_near_dog_response_is_stronger_and_far_response_is_normal(self):
        rows = dog_response_curve((20.0, 30.0, 40.0, 60.0), ("Normal", "C"))
        speeds = {(row["status"], row["dog_distance"]): row["speed"] for row in rows}
        self.assertGreater(speeds[("C", 20.0)], speeds[("Normal", 20.0)])
        for distance in (30.0, 40.0, 60.0):
            self.assertEqual(speeds[("C", distance)], speeds[("Normal", distance)])
        self.assertEqual(speeds[("C", 60.0)], 0.0)

    def test_force_response_is_continuous_and_speed_remains_bounded(self):
        distances = (3.0, 20.0, 30.0 - 1e-5, 30.0, 30.0 + 1e-5)
        with temporary_config(SHEEP_MAX_SPEED=100.0):
            rows = dog_response_curve(distances, ("Normal", "C"))
        speeds = {(row["status"], row["dog_distance"]): row["speed"] for row in rows}
        self.assertAlmostEqual(speeds[("C", 20.0)] / speeds[("Normal", 20.0)], 5.0 / 3.0)
        self.assertGreater(speeds[("C", 3.0)] / speeds[("Normal", 3.0)], 2.5)
        self.assertLess(abs(speeds[("C", 30.0 - 1e-5)] - speeds[("C", 30.0 + 1e-5)]), 1e-5)
        bounded = dog_response_curve((0.0, 3.0, 10.0, 20.0), ("C",))
        for row in bounded:
            self.assertTrue(np.isfinite(row["speed"]))
            self.assertLessEqual(row["speed"], config.SHEEP_MAX_SPEED)

    def test_type_c_preserves_flocking_and_boundary_response_without_dog(self):
        initial = np.array([[2.0, 100.0], [8.0, 101.0], [12.0, 99.0]])
        trajectories = []
        for status in ("Normal", "C"):
            positions = initial.copy()
            velocities = np.zeros_like(positions)
            for _ in range(30):
                positions, velocities = advance_sheep(
                    positions, velocities, np.array([status, "Normal", "Normal"]),
                    np.array([-1000.0, -1000.0]), np.array([180.0, 180.0]), config,
                )
            trajectories.append((positions, velocities))
        for normal, sensitive in zip(*trajectories):
            np.testing.assert_array_equal(normal, sensitive)

    def test_type_c_preserves_goal_retention_without_dog(self):
        positions = np.array([[118.0, 100.0], [100.0, 100.0]])
        velocities = np.array([[0.6, 0.0], [0.0, 0.0]])
        outcomes = [advance_sheep(
            positions, velocities, np.array([status, "Normal"]),
            np.array([-1000.0, -1000.0]), np.array([100.0, 100.0]), config,
        ) for status in ("Normal", "C")]
        for normal, sensitive in zip(*outcomes):
            np.testing.assert_array_equal(normal, sensitive)

    def test_assignment_preserves_geometry_and_existing_a_b_identities(self):
        with temporary_config(NUM_ABNORMAL_A=3, NUM_ABNORMAL_B=3, NUM_ABNORMAL_C=0,
                              RANDOM_SEED=3101, USE_FIXED_SEED=True):
            baseline = SwarmEnv()
            with temporary_config(NUM_ABNORMAL_C=4):
                first = SwarmEnv()
                second = SwarmEnv()
            for field in ("sheep_pos", "dog_pos", "goal_pos"):
                np.testing.assert_array_equal(getattr(baseline, field), getattr(first, field))
            np.testing.assert_array_equal(first.sheep_status, second.sheep_status)
            for status, count in (("A", 3), ("B", 3), ("C", 4), ("Normal", 10)):
                self.assertEqual(int(np.sum(first.sheep_status == status)), count)
            for status in ("A", "B"):
                np.testing.assert_array_equal(baseline.sheep_status == status,
                                              first.sheep_status == status)
            for env in (baseline, first, second):
                env.close()

    def test_invalid_population_and_sensitive_parameters_are_rejected(self):
        for overrides in (
            {"NUM_ABNORMAL_A": 7, "NUM_ABNORMAL_B": 7, "NUM_ABNORMAL_C": 7},
            {"NUM_ABNORMAL_C": -1},
            {"NUM_ABNORMAL_C": 1, "TYPE_C_SENSITIVITY_RADIUS": 0.0},
            {"NUM_ABNORMAL_C": 1, "TYPE_C_SENSITIVITY_RADIUS": 60.0},
            {"NUM_ABNORMAL_C": 1, "WEIGHT_DOG_REPULSION_C_FACTOR": 0.5},
            {"NUM_ABNORMAL_C": 1, "WEIGHT_DOG_REPULSION_C_FACTOR": float("nan")},
            {"NUM_ABNORMAL_C": 1, "WEIGHT_DOG_REPULSION_C_FACTOR": float("inf")},
        ):
            with self.subTest(overrides=overrides), temporary_config(**overrides):
                with self.assertRaises(ValueError):
                    SwarmEnv()

    def observation(self):
        return ControlObservation(
            sheep_positions=np.array([[100.0, 100.0], [110.0, 100.0]]),
            dog_position=np.array([75.0, 100.0]),
            goal_position=np.array([180.0, 100.0]),
            alarms=("flock_splitting", "stagnation"),
            sheep_statuses=("C", "Normal"),
            sheep_velocities=np.zeros((2, 2)),
        )

    def test_forecast_uses_type_c_dynamics_without_mutating_state(self):
        observation = self.observation()
        before = observation.sheep_positions.copy()
        with temporary_config(HIGH_LEVEL_FORECAST_FRAMES=1):
            sensitive = forecast_modes(observation, config)
            normal = forecast_modes(replace(observation, sheep_statuses=("Normal", "Normal")), config)
        self.assertGreater(sensitive["TARGET_TRACKING"]["centroid_progress"],
                           normal["TARGET_TRACKING"]["centroid_progress"])
        np.testing.assert_array_equal(observation.sheep_positions, before)

    def test_summary_describes_type_c_only_when_present(self):
        controller = HighLevelController(backend=MockLLMBackend(latency_frames=10))
        controller.decide(self.observation())
        summary = controller.pending_request.state_summary
        self.assertEqual(summary["type_c_count"], 1)
        self.assertEqual(summary["type_geometry"]["C"]["count"], 1)
        self.assertEqual(summary["type_c_response"]["sensitivity_radius"], 30.0)
        controller.close()
        normal = HighLevelController(backend=MockLLMBackend(latency_frames=10))
        normal.decide(replace(self.observation(), sheep_statuses=("Normal", "Normal")))
        self.assertNotIn("type_c_count", normal.pending_request.state_summary)
        self.assertNotIn("C", normal.pending_request.state_summary["type_geometry"])
        normal.close()

    def test_visualizer_renders_type_c_and_labels_it(self):
        with temporary_config(NUM_ABNORMAL_A=0, NUM_ABNORMAL_B=0, NUM_ABNORMAL_C=4):
            env = SwarmEnv()
            figure = Figure(figsize=(7, 7))
            FigureCanvasAgg(figure)
            with patch("visualizer.plt.subplots", return_value=(figure, figure.subplots())):
                vis = Visualizer(env)
            try:
                with patch("matplotlib.pyplot.pause"):
                    vis.render(env)
                vis.fig.canvas.draw()
                self.assertIn(LEGEND_LABELS["C"], [text.get_text() for text in vis.ax.get_legend().get_texts()])
                self.assertTrue(COLOR_MAP["C"])
                self.assertEqual(len(vis.scat_sheep.get_facecolors()), config.NUM_SHEEP)
            finally:
                plt.close(vis.fig)
                plt.ioff()
                env.close()


if __name__ == "__main__":
    unittest.main()
