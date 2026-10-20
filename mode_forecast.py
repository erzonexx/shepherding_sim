"""Short-horizon counterfactual forecasts for existing shepherd modes."""

import copy
import numpy as np

from control_types import ControlObservation
from dynamics import advance_dog, advance_sheep
from low_level_controller import LowLevelController


FORECAST_MODES = (
    "TARGET_TRACKING",
    "COHESION_PRIORITY",
    "HARD_TO_GUIDE",
)


def forecast_modes(observation, settings, low_level_controller=None):
    horizon = max(1, int(settings.HIGH_LEVEL_FORECAST_FRAMES))
    return {
        mode: forecast_mode(
            observation, mode, horizon, settings, low_level_controller
        )
        for mode in FORECAST_MODES
    }


def forecast_mode(observation, mode, horizon, settings,
                  low_level_controller=None, target_sheep_id=None):
    positions = np.array(observation.sheep_positions, dtype=float, copy=True)
    velocities = observation.sheep_velocities
    if velocities is None or np.shape(velocities) != np.shape(positions):
        velocities = np.zeros_like(positions)
    else:
        velocities = np.array(velocities, dtype=float, copy=True)
    dog_position = np.array(observation.dog_position, dtype=float, copy=True)
    dog_velocity = observation.dog_velocity
    if dog_velocity is None or np.shape(dog_velocity) != (2,):
        dog_velocity = np.zeros(2, dtype=float)
    else:
        dog_velocity = np.array(dog_velocity, dtype=float, copy=True)
    goal_position = np.array(observation.goal_position, dtype=float, copy=True)
    statuses = list(observation.sheep_statuses[: len(positions)])
    statuses.extend(["Normal"] * (len(positions) - len(statuses)))
    statuses = np.asarray(statuses)

    initial = _flock_metrics(positions, goal_position, settings)
    previous_inside = _inside_goal(positions, goal_position, settings)
    goal_exits = 0
    peak_out_of_bounds = initial["out_of_bounds"]
    minimum_boundary_margin = initial["boundary_margin"]
    splitting_frames = 0
    dog_travel = 0.0
    target_finished = False
    controller = (
        copy.copy(low_level_controller)
        if low_level_controller is not None
        else LowLevelController(settings)
    )

    for offset in range(horizon):
        forecast_observation = ControlObservation(
            sheep_positions=positions,
            dog_position=dog_position,
            goal_position=goal_position,
            alarms=observation.alarms,
            frame=observation.frame + offset,
            sheep_statuses=tuple(statuses.tolist()),
            current_mode=mode,
            sheep_velocities=velocities,
            dog_velocity=dog_velocity,
        )
        decision = controller.decide(
            forecast_observation,
            mode_override=None if target_finished else mode,
            target_override=target_sheep_id,
        )
        next_dog_position, dog_velocity = advance_dog(
            dog_position,
            decision.navigation_point,
            settings,
        )
        dog_travel += float(np.linalg.norm(next_dog_position - dog_position))
        dog_position = next_dog_position
        positions, velocities = advance_sheep(
            positions,
            velocities,
            statuses,
            dog_position,
            goal_position,
            settings,
        )

        metrics = _flock_metrics(positions, goal_position, settings)
        current_inside = _inside_goal(positions, goal_position, settings)
        goal_exits += int(np.sum(previous_inside & ~current_inside))
        previous_inside = current_inside
        if target_sheep_id is not None and current_inside[target_sheep_id]:
            target_sheep_id = None
            target_finished = True
        peak_out_of_bounds = max(
            peak_out_of_bounds,
            metrics["out_of_bounds"],
        )
        minimum_boundary_margin = min(
            minimum_boundary_margin,
            metrics["boundary_margin"],
        )
        if (
            metrics["max_spread"] > settings.DANGER_DISPERSION_THRESHOLD
            and metrics["mean_spread"]
            > settings.DANGER_MEAN_SPREAD_THRESHOLD
        ):
            splitting_frames += 1

    final = _flock_metrics(positions, goal_position, settings)
    return {
        "horizon_frames": horizon,
        "centroid_progress": round(
            initial["centroid_distance"] - final["centroid_distance"],
            3,
        ),
        "final_centroid_distance": round(final["centroid_distance"], 3),
        "max_spread_change": round(
            final["max_spread"] - initial["max_spread"],
            3,
        ),
        "final_max_spread": round(final["max_spread"], 3),
        "mean_spread_change": round(
            final["mean_spread"] - initial["mean_spread"],
            3,
        ),
        "final_mean_spread": round(final["mean_spread"], 3),
        "inside_goal_change": final["inside_goal"] - initial["inside_goal"],
        "final_inside_goal": final["inside_goal"],
        "predicted_goal_exits": goal_exits,
        "predicted_splitting_frames": splitting_frames,
        "peak_out_of_bounds": peak_out_of_bounds,
        "minimum_boundary_margin": round(minimum_boundary_margin, 3),
        "dog_travel": round(dog_travel, 3),
        "predicted_success": final["inside_goal"] == len(positions),
    }


def _inside_goal(positions, goal_position, settings):
    return np.linalg.norm(positions - goal_position, axis=1) < settings.GOAL_RADIUS


def _flock_metrics(positions, goal_position, settings):
    center = np.mean(positions, axis=0)
    spread = np.linalg.norm(positions - center, axis=1)
    boundary_margins = np.minimum.reduce(
        (
            positions[:, 0],
            positions[:, 1],
            settings.SPACE_SIZE - positions[:, 0],
            settings.SPACE_SIZE - positions[:, 1],
        )
    )
    return {
        "centroid_distance": float(np.linalg.norm(center - goal_position)),
        "max_spread": float(np.max(spread)),
        "mean_spread": float(np.mean(spread)),
        "inside_goal": int(np.sum(_inside_goal(positions, goal_position, settings))),
        "out_of_bounds": int(np.sum(boundary_margins < 0.0)),
        "boundary_margin": float(np.min(boundary_margins)),
    }
