"""Tests for the experimental local forecast selector."""

import unittest
from unittest.mock import patch

import numpy as np

from control_types import ControlObservation
from tests.anomaly_behavior.run_forecast_mode_trial import (
    ForecastDominanceController,
)


class ForecastModeTrialTests(unittest.TestCase):
    def observation(self, frame, alarms=("flock_splitting", "stagnation")):
        return ControlObservation(
            sheep_positions=np.array([[20., 20.], [40., 40.]]),
            dog_position=np.array([10., 10.]),
            goal_position=np.array([180., 180.]),
            alarms=alarms,
            frame=frame,
        )

    def test_dominant_cohesion_is_held_until_review(self):
        guide = {
            "centroid_progress": 1, "inside_goal_change": 0,
            "final_max_spread": 40, "predicted_goal_exits": 0,
            "predicted_splitting_frames": 10, "peak_out_of_bounds": 0,
        }
        cohesion = {**guide, "final_max_spread": 30}
        controller = ForecastDominanceController(review_period=10)
        with patch(
            "tests.anomaly_behavior.run_forecast_mode_trial.forecast_modes",
            return_value={
                "COHESION_PRIORITY": cohesion,
                "HARD_TO_GUIDE": guide,
            },
        ) as forecast:
            self.assertEqual(
                controller.decide(self.observation(1)).mode,
                "COHESION_PRIORITY",
            )
            self.assertEqual(
                controller.decide(self.observation(2)).mode,
                "COHESION_PRIORITY",
            )
            self.assertEqual(forecast.call_count, 1)
            self.assertEqual(
                controller.decide(self.observation(3, ("stagnation",))).mode,
                "HARD_TO_GUIDE",
            )
            self.assertEqual(
                controller.decide(self.observation(4)).mode,
                "COHESION_PRIORITY",
            )
            self.assertEqual(forecast.call_count, 2)


if __name__ == "__main__":
    unittest.main()
