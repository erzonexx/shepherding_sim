import unittest
from types import SimpleNamespace

import numpy as np

import config
from control_types import ControlObservation
from high_level_controller import HighLevelController
from low_level_controller import LowLevelController
from mode_forecast import FORECAST_MODES, forecast_modes


class ModeForecastTests(unittest.TestCase):
    def settings(self, **overrides):
        values = {
            name: getattr(config, name)
            for name in dir(config)
            if name.isupper()
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def observation(self):
        return ControlObservation(
            sheep_positions=np.array(
                [[20., 100.], [75., 100.], [90., 105.]], dtype=float
            ),
            dog_position=np.array([55., 75.], dtype=float),
            goal_position=np.array([180., 100.], dtype=float),
            alarms=("flock_splitting", "stagnation"),
            frame=100,
            sheep_statuses=("A", "B", "Normal"),
            sheep_velocities=np.array(
                [[-0.2, 0.0], [0.4, 0.1], [0.1, -0.1]], dtype=float
            ),
            dog_velocity=np.array([1.0, 0.0], dtype=float),
        )

    def test_forecasts_all_modes_without_mutating_observation(self):
        observation = self.observation()
        sheep_before = observation.sheep_positions.copy()
        velocity_before = observation.sheep_velocities.copy()

        forecasts = forecast_modes(
            observation,
            self.settings(HIGH_LEVEL_FORECAST_FRAMES=5),
        )

        self.assertEqual(tuple(forecasts), FORECAST_MODES)
        for forecast in forecasts.values():
            self.assertEqual(forecast["horizon_frames"], 5)
            self.assertIn("centroid_progress", forecast)
            self.assertIn("predicted_goal_exits", forecast)
            self.assertIn("peak_out_of_bounds", forecast)
        np.testing.assert_array_equal(
            observation.sheep_positions,
            sheep_before,
        )
        np.testing.assert_array_equal(
            observation.sheep_velocities,
            velocity_before,
        )

    def test_forecast_modes_produce_distinct_consequences(self):
        forecasts = forecast_modes(
            self.observation(),
            self.settings(HIGH_LEVEL_FORECAST_FRAMES=10),
        )
        signatures = {
            (
                forecast["centroid_progress"],
                forecast["final_max_spread"],
                forecast["dog_travel"],
            )
            for forecast in forecasts.values()
        }
        self.assertGreater(len(signatures), 1)

    def test_forecast_carries_target_hold_without_mutating_live_controller(self):
        settings = self.settings(HIGH_LEVEL_FORECAST_FRAMES=3)
        observation = ControlObservation(
            sheep_positions=np.array([[20., 100.], [30., 100.]]),
            dog_position=np.array([25., 100.]),
            goal_position=np.array([180., 100.]),
            alarms=("stagnation",),
            sheep_velocities=np.array([[0.7, 0.], [-0.7, 0.]]),
        )
        live = LowLevelController(settings)
        live.repair_target_mode = "HARD_TO_GUIDE"
        live.repair_target_id = 0
        live.repair_target_age = 5

        held = forecast_modes(observation, settings, live)
        fresh = forecast_modes(observation, settings)

        self.assertNotEqual(
            held["HARD_TO_GUIDE"], fresh["HARD_TO_GUIDE"]
        )
        self.assertEqual(live.repair_target_id, 0)
        self.assertEqual(live.repair_target_age, 5)

    def test_pareto_dominance_requires_no_worse_on_every_metric(self):
        selected = {
            "centroid_progress": 1.0,
            "inside_goal_change": 0,
            "final_max_spread": 45.0,
            "predicted_goal_exits": 2,
            "predicted_splitting_frames": 10,
            "peak_out_of_bounds": 1,
        }
        better = {
            "centroid_progress": 2.0,
            "inside_goal_change": 1,
            "final_max_spread": 40.0,
            "predicted_goal_exits": 1,
            "predicted_splitting_frames": 8,
            "peak_out_of_bounds": 0,
        }
        tradeoff = dict(better, final_max_spread=50.0)

        self.assertTrue(
            HighLevelController._forecast_dominates(better, selected)
        )
        self.assertFalse(
            HighLevelController._forecast_dominates(tradeoff, selected)
        )


if __name__ == "__main__":
    unittest.main()
