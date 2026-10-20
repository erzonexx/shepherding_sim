"""Low-level kinematic vector controller for shepherd motion."""

import copy
from typing import Optional

import numpy as np

import config
from control_types import ControlDecision, ControlObservation


def rule_mode_for_alarms(alarms, repair_enabled=True):
    """Choose the deterministic fallback mode from the active alarms."""
    if not repair_enabled:
        return "TARGET_TRACKING"
    if "flock_splitting" in alarms and "stagnation" in alarms:
        return "HARD_TO_GUIDE"
    if "flock_splitting" in alarms:
        return "COHESION_PRIORITY"
    if "stagnation" in alarms:
        return "HARD_TO_GUIDE"
    return "TARGET_TRACKING"


class LowLevelController:
    """Compute kinematic commands, with the original rule mapping as fallback."""

    def __init__(self, settings=config):
        self.settings = settings
        self.repair_target_mode = None
        self.repair_target_id = None
        self.repair_target_age = 0
        self.repair_target_hold_released = False

    def decide(
        self, observation: ControlObservation, mode_override: Optional[str] = None,
        target_override: Optional[int] = None,
    ) -> ControlDecision:
        cfg = self.settings
        sheep_pos = observation.sheep_positions
        goal_pos = observation.goal_position
        alarms = observation.alarms

        center = np.mean(sheep_pos, axis=0)
        to_goal = goal_pos - center
        centroid_distance = np.linalg.norm(to_goal)
        distances_to_goal = np.linalg.norm(sheep_pos - goal_pos, axis=1)

        mode = mode_override if mode_override is not None else self._select_mode(alarms)
        if target_override is not None:
            if mode not in ("HARD_TO_GUIDE", "COHESION_PRIORITY"):
                raise ValueError("Only repair modes accept a target override.")
            if not self.repair_target_is_eligible(observation, target_override):
                raise ValueError("Repair target must be a valid sheep outside the goal.")
            if self.repair_target_mode == mode and self.repair_target_id == target_override:
                self.repair_target_age += 1
            else:
                self.repair_target_age = 1
            self.repair_target_mode = mode
            self.repair_target_id = target_override
            self.repair_target_hold_released = False
        vectors_to_center = sheep_pos - center
        distances_to_center = np.linalg.norm(vectors_to_center, axis=1)
        max_distance_to_center = np.max(distances_to_center)
        target_id = None

        if mode == "TARGET_TRACKING":
            if max_distance_to_center > cfg.COHESION_THRESHOLD:
                target_id = int(np.argmax(distances_to_center))
                action = "collect"
                target_pos = sheep_pos[target_id]
                direction = vectors_to_center[target_id] / (
                    max_distance_to_center + cfg.EPSILON
                )
                drive_point = target_pos + direction * cfg.DOG_DRIVE_DISTANCE
            else:
                action = "drive"
                direction = to_goal / (centroid_distance + cfg.EPSILON)
                drive_point = center - direction * cfg.DOG_DRIVE_DISTANCE
        elif mode == "HARD_TO_GUIDE":
            if target_override is not None:
                target_id = target_override
            elif cfg.REPAIR_TARGET_SELECTION_ENABLED:
                target_id = self._select_repair_target(
                    mode,
                    self._hard_to_guide_scores(observation, distances_to_goal),
                    distances_to_goal >= cfg.GOAL_RADIUS,
                )
            else:
                target_id = int(np.argmax(distances_to_goal))
            action = "guide_individual"
            target_pos = sheep_pos[target_id]
            direction = (goal_pos - target_pos) / (
                distances_to_goal[target_id] + cfg.EPSILON
            )
            drive_point = target_pos - direction * cfg.HARD_TO_GUIDE_DISTANCE
        elif mode == "COHESION_PRIORITY":
            if target_override is not None:
                target_id = target_override
            elif cfg.REPAIR_TARGET_SELECTION_ENABLED:
                target_id = self._select_repair_target(
                    mode,
                    self._cohesion_scores(observation, vectors_to_center),
                    distances_to_goal >= cfg.GOAL_RADIUS,
                )
            else:
                target_id = int(np.argmax(distances_to_center))
            action = "collect"
            target_pos = sheep_pos[target_id]
            direction = vectors_to_center[target_id] / (
                max_distance_to_center + cfg.EPSILON
            )
            drive_point = target_pos + direction * cfg.COHESION_REPAIR_DISTANCE
        else:
            raise ValueError(f"Unknown shepherd mode: {mode}")

        if mode == "TARGET_TRACKING":
            self._clear_repair_target()

        drive_point = np.copy(drive_point)
        navigation_point = np.copy(drive_point)

        return ControlDecision(
            mode=mode,
            action=action,
            target_sheep_id=target_id,
            drive_point=drive_point,
            navigation_point=np.copy(navigation_point),
            issued_for_frame=observation.frame,
            source="rule",
        )

    def repair_target_is_eligible(self, observation, target_id):
        return (
            isinstance(target_id, int)
            and not isinstance(target_id, bool)
            and 0 <= target_id < len(observation.sheep_positions)
            and np.linalg.norm(
                observation.sheep_positions[target_id] - observation.goal_position
            ) >= self.settings.GOAL_RADIUS
        )

    def repair_candidates(self, observation, mode, limit=3):
        positions = observation.sheep_positions
        distances = np.linalg.norm(positions - observation.goal_position, axis=1)
        if mode == "HARD_TO_GUIDE":
            scores = self._hard_to_guide_scores(observation, distances)
        elif mode == "COHESION_PRIORITY":
            scores = self._cohesion_scores(observation, positions - np.mean(positions, axis=0))
        else:
            return []
        eligible = np.flatnonzero(distances >= self.settings.GOAL_RADIUS)
        ranked = sorted(eligible.tolist(), key=lambda target: (-scores[target], target))
        # Include the rule's held target, not just the highest instantaneous scores.
        default = copy.copy(self).decide(observation, mode_override=mode).target_sheep_id
        if default in ranked:
            ranked.remove(default)
            ranked.insert(0, default)
        return ranked[:limit]

    def _cohesion_scores(self, observation, vectors_to_center):
        """Estimate which sheep will remain the strongest outward outlier."""
        cfg = self.settings
        distances = np.linalg.norm(vectors_to_center, axis=1)
        velocities = self._velocities(observation)
        relative_velocities = velocities - np.mean(velocities, axis=0)
        outward_directions = vectors_to_center / (
            distances[:, np.newaxis] + cfg.EPSILON
        )
        outward_speeds = np.sum(relative_velocities * outward_directions, axis=1)
        return distances + cfg.REPAIR_TARGET_LOOKAHEAD_FRAMES * outward_speeds

    def _hard_to_guide_scores(self, observation, distances_to_goal):
        """Estimate each sheep's remaining goal distance after a short lookahead."""
        cfg = self.settings
        velocities = self._velocities(observation)
        directions_to_goal = (
            observation.goal_position - observation.sheep_positions
        ) / (distances_to_goal[:, np.newaxis] + cfg.EPSILON)
        goalward_speeds = np.sum(velocities * directions_to_goal, axis=1)
        return (
            distances_to_goal
            - cfg.REPAIR_TARGET_LOOKAHEAD_FRAMES * goalward_speeds
        )

    def _velocities(self, observation):
        velocities = observation.sheep_velocities
        if velocities is None or np.shape(velocities) != np.shape(
            observation.sheep_positions
        ):
            return np.zeros_like(observation.sheep_positions)
        return np.asarray(velocities)

    def _select_repair_target(self, mode, scores, outside_goal):
        cfg = self.settings
        eligible = np.ones(len(scores), dtype=bool)
        if cfg.REPAIR_TARGET_OUTSIDE_GOAL_FIRST and np.any(outside_goal):
            eligible = np.asarray(outside_goal, dtype=bool)

        ranked_scores = np.where(eligible, scores, -np.inf)
        best = int(np.argmax(ranked_scores))
        current = self.repair_target_id
        current_is_eligible = (
            self.repair_target_mode == mode
            and current is not None
            and 0 <= current < len(scores)
            and eligible[current]
        )

        if current_is_eligible:
            keep_for_minimum = (
                not self.repair_target_hold_released
                and self.repair_target_age < self._repair_target_minimum_hold(mode)
            )
            current_is_competitive = (
                scores[best]
                <= scores[current] + cfg.REPAIR_TARGET_SWITCH_MARGIN
            )
            if keep_for_minimum or current_is_competitive:
                self.repair_target_age += 1
                return current

        self.repair_target_mode = mode
        self.repair_target_id = best
        self.repair_target_age = 1
        self.repair_target_hold_released = False
        return best

    def release_repair_target_hold(self):
        """Allow immediate rule reassessment, retaining age and score hysteresis."""
        self.repair_target_hold_released = True

    def _repair_target_minimum_hold(self, mode):
        if mode == "COHESION_PRIORITY":
            return self.settings.COHESION_TARGET_MIN_HOLD_FRAMES
        return self.settings.HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES

    def _clear_repair_target(self):
        self.repair_target_mode = None
        self.repair_target_id = None
        self.repair_target_age = 0
        self.repair_target_hold_released = False

    def _select_mode(self, alarms):
        return rule_mode_for_alarms(alarms, self.settings.REPAIR_ENABLED)

