"""Shared kinematic state update used by live and shadow simulation."""

import numpy as np


def limit_speed(velocity, max_speed):
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        return velocity / speed * max_speed
    return np.copy(velocity)


def limit_speed_array(velocities, max_speed, epsilon):
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_speed / (speeds + epsilon))
    return velocities * scale


def advance_dog(dog_position, navigation_point, settings):
    velocity = limit_speed(
        navigation_point - dog_position,
        settings.DOG_MAX_SPEED,
    )
    position = np.clip(
        dog_position + velocity,
        0.0,
        settings.SPACE_SIZE,
    )
    return position, velocity


def advance_sheep(
    sheep_positions,
    sheep_velocities,
    sheep_statuses,
    dog_position,
    goal_position,
    settings,
):
    positions = np.asarray(sheep_positions, dtype=float)
    velocities = np.asarray(sheep_velocities, dtype=float)
    statuses = np.asarray(sheep_statuses)
    center = np.mean(positions, axis=0)
    distances_to_goal = np.linalg.norm(positions - goal_position, axis=1)

    cohesion_weights = np.where(
        statuses == "B",
        settings.WEIGHT_COHESION * settings.WEIGHT_COHESION_B_FACTOR,
        settings.WEIGHT_COHESION,
    )[:, np.newaxis]
    dog_repulsion_weights = np.where(
        statuses == "A",
        settings.WEIGHT_DOG_REPULSION * settings.WEIGHT_DOG_REPULSION_A_FACTOR,
        settings.WEIGHT_DOG_REPULSION,
    )[:, np.newaxis]
    separation_weights = np.where(
        statuses == "B",
        settings.WEIGHT_SEPARATION * settings.WEIGHT_SEPARATION_B_FACTOR,
        settings.WEIGHT_SEPARATION,
    )[:, np.newaxis]

    cohesion = (center - positions) * cohesion_weights

    vectors_from_dog = positions - dog_position
    distances_from_dog = np.linalg.norm(
        vectors_from_dog, axis=1, keepdims=True
    )
    directions_from_dog = vectors_from_dog / (
        distances_from_dog + settings.EPSILON
    )
    dog_repulsion = directions_from_dog * (
        dog_repulsion_weights / (distances_from_dog + settings.EPSILON)
    )
    dog_repulsion[
        distances_from_dog[:, 0] > settings.DOG_SENSING_RANGE
    ] = 0.0

    boundary_force = np.zeros_like(positions)
    boundary_force[positions < settings.BOUNDARY_MARGIN] += settings.WEIGHT_BOUNDARY
    boundary_force[
        positions > settings.SPACE_SIZE - settings.BOUNDARY_MARGIN
    ] -= settings.WEIGHT_BOUNDARY

    goal_retention = np.zeros_like(positions)
    goal_retention_mask = np.zeros(len(positions), dtype=bool)
    if settings.GOAL_RETENTION_ENABLED:
        inner_radius = (
            settings.GOAL_RADIUS * settings.GOAL_RETENTION_INNER_FRACTION
        )
        goal_retention_mask = (
            (distances_to_goal >= inner_radius)
            & (distances_to_goal < settings.GOAL_RADIUS)
        )
        radial_width = max(
            settings.GOAL_RADIUS - inner_radius,
            settings.EPSILON,
        )
        strength = np.clip(
            (distances_to_goal - inner_radius) / radial_width,
            0.0,
            1.0,
        )
        directions_to_goal = (goal_position - positions) / (
            distances_to_goal[:, np.newaxis] + settings.EPSILON
        )
        goal_retention[goal_retention_mask] = (
            directions_to_goal[goal_retention_mask]
            * strength[goal_retention_mask, np.newaxis]
            * settings.GOAL_RETENTION_WEIGHT
        )

    differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    separation_radius = np.where(
        statuses == "B",
        settings.SEPARATION_RADIUS_B,
        settings.SEPARATION_RADIUS_NORMAL,
    )[:, np.newaxis]
    separation_mask = (distances > 0.0) & (distances < separation_radius)
    repulsion_forces = differences / (
        distances[..., np.newaxis] ** 2 + settings.EPSILON
    )
    repulsion_forces[~separation_mask] = 0.0
    separation = np.sum(repulsion_forces, axis=1) * separation_weights

    total_force = (
        cohesion
        + dog_repulsion
        + boundary_force
        + separation
        + goal_retention
    )
    next_velocities = velocities + total_force
    if settings.GOAL_RETENTION_ENABLED:
        next_velocities[goal_retention_mask] *= (
            settings.GOAL_RETENTION_VELOCITY_DAMPING
        )
    next_velocities = limit_speed_array(
        next_velocities,
        settings.SHEEP_MAX_SPEED,
        settings.EPSILON,
    )
    return positions + next_velocities, next_velocities
