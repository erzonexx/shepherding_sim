"""Event-triggered high-level mode selection for shepherd control."""

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol

import numpy as np

import config
from control_types import ControlDecision, ControlObservation
from low_level_controller import LowLevelController
from mode_forecast import forecast_modes


ALLOWED_MODES = {
    "TARGET_TRACKING",
    "COHESION_PRIORITY",
    "HARD_TO_GUIDE",
}


@dataclass(frozen=True)
class HighLevelRequest:
    request_id: str
    trigger_frame: int
    alarms: tuple[str, ...]
    primary_alarm: str
    state_summary: dict[str, Any]


class HighLevelBackend(Protocol):
    source: str

    def submit(self, request: HighLevelRequest) -> None:
        """Start one non-blocking request."""

    def poll(self, request_id: str, frame: int) -> Optional[Mapping[str, Any]]:
        """Return a response when ready, otherwise None."""

    def cancel(self, request_id: str) -> None:
        """Discard a pending request when its triggering state disappears."""


class MockLLMBackend:
    """Frame-delayed backend for integration tests; it is not an LLM."""

    source = "mock_llm"

    def __init__(self, latency_frames=30, hold_frames=120, mode_mapping=None):
        self.latency_frames = latency_frames
        self.hold_frames = hold_frames
        self.mode_mapping = mode_mapping or {
            "flock_splitting": "COHESION_PRIORITY",
            "stagnation": "HARD_TO_GUIDE",
        }
        self.pending = {}

    def submit(self, request):
        self.pending[request.request_id] = request

    def poll(self, request_id, frame):
        request = self.pending.get(request_id)
        if request is None or frame < request.trigger_frame + self.latency_frames:
            return None
        self.pending.pop(request_id, None)
        return {
            "request_id": request.request_id,
            "trigger_frame": request.trigger_frame,
            "mode": self.mode_mapping.get(request.primary_alarm, "TARGET_TRACKING"),
            "hold_frames": self.hold_frames,
            "reason": "deterministic mock mapping",
            "latency_seconds": None,
            "cache_hit": False,
            "model": "mock",
        }

    def cancel(self, request_id):
        self.pending.pop(request_id, None)


class HighLevelController:
    """Keep low-level control active while a mode request is pending."""

    def __init__(self, low_level_controller=None, backend=None, settings=config):
        self.settings = settings
        self.low_level_controller = low_level_controller or LowLevelController(settings)
        self.backend = backend or MockLLMBackend(
            latency_frames=settings.MOCK_LLM_LATENCY_FRAMES,
            hold_frames=settings.HIGH_LEVEL_HOLD_FRAMES,
        )
        self.pending_request = None
        self.active_mode = None
        self.active_alarms = ()
        self.active_started_frame = None
        self.active_until_frame = None
        self.active_hold_frames = None
        self.active_request = None
        self.active_latency_frames = None
        self.active_response_metadata = {}
        self.request_counter = 0
        self.retry_after_frame = 0
        self.state_history = deque(
            maxlen=max(2, int(settings.HIGH_LEVEL_CONTEXT_FRAMES))
        )
        self.observed_alarms = ()
        self.alarm_started_frame = None

    def decide(self, observation: ControlObservation) -> ControlDecision:
        alarms = self._known_alarms(observation.alarms)
        state_summary = self._observe_state(observation, alarms)
        concurrent_only = getattr(
            self.backend, "trigger_on_concurrent_only", False
        )
        if not self.settings.REPAIR_ENABLED:
            self._clear_pending()
            release_event = self._release_active(
                "repair_disabled", observation.frame
            )
            return self._low_level_decision(
                observation,
                status="disabled",
                release_event=release_event,
            )

        if not alarms:
            self._clear_pending()
            release_event = self._release_active(
                "all_alarms_cleared", observation.frame
            )
            return self._low_level_decision(
                observation,
                status="idle",
                release_event=release_event,
            )

        if (
            self.pending_request is not None
            and not self._request_is_relevant(self.pending_request, alarms)
        ):
            self._clear_pending()
        release_event = None
        if self.active_mode is not None:
            release_reason = self._active_release_reason(
                alarms, observation.frame
            )
            if release_reason is not None:
                release_event = self._release_active(
                    release_reason, observation.frame
                )

        rejection_reason = None
        completed_request = None
        if self.pending_request is not None:
            response = self.backend.poll(
                self.pending_request.request_id, observation.frame
            )
            if response is not None:
                validation_summary = self._with_mode_forecasts(
                    observation, state_summary
                )
                mode, hold_frames, rejection_reason = self._validate_response(
                    response,
                    observation.frame,
                    alarms,
                    validation_summary,
                )
                request = self.pending_request
                completed_request = request
                self.pending_request = None
                if rejection_reason is None:
                    self.active_mode = mode
                    self.active_alarms = request.alarms
                    self.active_started_frame = observation.frame
                    self.active_until_frame = observation.frame + hold_frames
                    self.active_hold_frames = hold_frames
                    self.active_request = request
                    self.active_latency_frames = (
                        observation.frame - request.trigger_frame
                    )
                    self.active_response_metadata = self._response_metadata(response)
                else:
                    self.retry_after_frame = (
                        observation.frame + self.settings.HIGH_LEVEL_RETRY_FRAMES
                    )

        if (
            self.active_mode is not None
            and observation.frame < self.active_until_frame
        ):
            metadata = self._metadata(
                "active",
                request=self.active_request,
                selected_mode=self.active_mode,
                latency_frames=self.active_latency_frames,
                hold_frames=self.active_hold_frames,
                response_metadata=self.active_response_metadata,
                release_event=release_event,
            )
            decision = self.low_level_controller.decide(
                observation, mode_override=self.active_mode
            )
            return replace(
                decision,
                source=self.backend.source,
                metadata=metadata,
            )

        if self.active_mode is not None:
            release_event = self._release_active(
                "max_hold_reached", observation.frame
            )

        if (
            self.pending_request is None
            and observation.frame >= self.retry_after_frame
            and (not concurrent_only or len(alarms) >= 2)
        ):
            self.pending_request = self._make_request(
                observation, alarms, state_summary
            )
            self.backend.submit(self.pending_request)

        if rejection_reason is not None:
            return self._low_level_decision(
                observation,
                status="rejected",
                request=completed_request,
                rejection_reason=rejection_reason,
                response_metadata=(
                    self._response_metadata(response) if completed_request else None
                ),
                release_event=release_event,
            )
        fallback_status = (
            "pending"
            if self.pending_request is not None
            else "rule_only"
            if concurrent_only
            else "pending"
        )
        return self._low_level_decision(
            observation,
            status=fallback_status,
            request=self.pending_request,
            release_event=release_event,
        )

    def _low_level_decision(
        self,
        observation,
        status,
        request=None,
        rejection_reason=None,
        response_metadata=None,
        release_event=None,
    ):
        decision = self.low_level_controller.decide(observation)
        return replace(
            decision,
            metadata=self._metadata(
                status,
                request=request,
                rejection_reason=rejection_reason,
                response_metadata=response_metadata,
                release_event=release_event,
            ),
        )

    def _make_request(self, observation, alarms, state_summary):
        self.request_counter += 1
        return HighLevelRequest(
            request_id=f"high-{self.request_counter}",
            trigger_frame=observation.frame,
            alarms=alarms,
            primary_alarm=self._primary_alarm(alarms),
            state_summary=self._with_mode_forecasts(
                observation, state_summary
            ),
        )

    def _validate_response(self, response, frame, alarms, current_summary):
        if not isinstance(response, Mapping):
            return None, None, "response_not_an_object"
        request = self.pending_request
        if response.get("backend_error"):
            return None, None, "backend_error"
        if response.get("request_id") != request.request_id:
            return None, None, "request_id_mismatch"
        if response.get("trigger_frame") != request.trigger_frame:
            return None, None, "trigger_frame_mismatch"
        if not self._request_is_relevant(request, alarms):
            return None, None, "trigger_no_longer_relevant"
        if frame - request.trigger_frame > self.settings.HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES:
            return None, None, "response_stale"
        mode = response.get("mode")
        if mode not in ALLOWED_MODES:
            return None, None, "mode_not_allowed"
        required_alarm = {
            "COHESION_PRIORITY": "flock_splitting",
            "HARD_TO_GUIDE": "stagnation",
        }.get(mode)
        if required_alarm is not None and required_alarm not in alarms:
            return None, None, "selected_alarm_not_active"
        candidate_key = {
            "COHESION_PRIORITY": "cohesion_candidate",
            "HARD_TO_GUIDE": "hard_to_guide_candidate",
        }.get(mode)
        candidate = current_summary.get(candidate_key, {}) if candidate_key else {}
        boundary_margin = candidate.get("drive_point_boundary_margin")
        if (
            isinstance(boundary_margin, (int, float))
            and boundary_margin < 0.0
        ):
            return None, None, "candidate_drive_point_out_of_bounds"
        dominating_mode = self._dominating_forecast_mode(
            mode,
            alarms,
            current_summary.get("mode_forecasts", {}),
        )
        if dominating_mode is not None:
            return None, None, f"forecast_dominated_by_{dominating_mode.lower()}"
        hold_frames = response.get("hold_frames")
        if isinstance(hold_frames, bool) or not isinstance(hold_frames, int):
            return None, None, "hold_frames_not_integer"
        if not (
            self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES
            <= hold_frames
            <= self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES
        ):
            return None, None, "hold_frames_out_of_range"
        return mode, hold_frames, None

    def _with_mode_forecasts(self, observation, state_summary):
        summary = dict(state_summary)
        summary["mode_forecasts"] = forecast_modes(
            observation,
            self.settings,
        )
        return summary

    def _dominating_forecast_mode(self, selected_mode, alarms, forecasts):
        selected = forecasts.get(selected_mode)
        if not isinstance(selected, Mapping):
            return None
        admissible = {"TARGET_TRACKING"}
        if "flock_splitting" in alarms:
            admissible.add("COHESION_PRIORITY")
        if "stagnation" in alarms:
            admissible.add("HARD_TO_GUIDE")
        for mode in sorted(admissible):
            alternative = forecasts.get(mode)
            if mode == selected_mode or not isinstance(alternative, Mapping):
                continue
            if self._forecast_dominates(alternative, selected):
                return mode
        return None

    @staticmethod
    def _forecast_dominates(alternative, selected):
        maximize = ("centroid_progress", "inside_goal_change")
        minimize = (
            "final_max_spread",
            "predicted_goal_exits",
            "predicted_splitting_frames",
            "peak_out_of_bounds",
        )
        keys = maximize + minimize
        if any(key not in alternative or key not in selected for key in keys):
            return False
        no_worse = all(
            alternative[key] >= selected[key] for key in maximize
        ) and all(alternative[key] <= selected[key] for key in minimize)
        strictly_better = any(
            alternative[key] > selected[key] for key in maximize
        ) or any(alternative[key] < selected[key] for key in minimize)
        return no_worse and strictly_better

    def _request_is_relevant(self, request, alarms):
        if getattr(self.backend, "trigger_on_concurrent_only", False):
            return bool(set(request.alarms).intersection(alarms))
        return request.alarms == alarms

    def _active_release_reason(self, alarms, frame):
        if frame >= self.active_until_frame:
            return "max_hold_reached"
        if not getattr(self.backend, "trigger_on_concurrent_only", False):
            return None if self.active_alarms == alarms else "trigger_changed"
        if not set(self.active_alarms).intersection(alarms):
            return "trigger_no_longer_relevant"
        active_frames = frame - self.active_started_frame
        if active_frames < self.settings.HIGH_LEVEL_MIN_DWELL_FRAMES:
            return None
        if (
            self.active_mode == "COHESION_PRIORITY"
            and "flock_splitting" not in alarms
        ):
            return "selected_alarm_cleared"
        if (
            self.active_mode == "HARD_TO_GUIDE"
            and "stagnation" not in alarms
        ):
            return "selected_alarm_cleared"
        if self.active_mode == "TARGET_TRACKING" and alarms != self.active_alarms:
            return "trigger_context_changed"
        return None

    def _known_alarms(self, alarms):
        return tuple(
            alarm
            for alarm in ("flock_splitting", "stagnation")
            if alarm in alarms
        )

    def _primary_alarm(self, alarms):
        if "flock_splitting" in alarms:
            return "flock_splitting"
        if "stagnation" in alarms:
            return "stagnation"
        return None

    def _observe_state(self, observation, alarms):
        positions = observation.sheep_positions
        center = np.mean(positions, axis=0)
        vectors_from_center = positions - center
        vectors_to_goal = observation.goal_position - positions
        distances_to_center = np.linalg.norm(vectors_from_center, axis=1)
        distances_to_goal = np.linalg.norm(vectors_to_goal, axis=1)
        velocities = observation.sheep_velocities
        if velocities is None or np.shape(velocities) != np.shape(positions):
            velocities = np.zeros_like(positions)
        else:
            velocities = np.asarray(velocities, dtype=float)
        dog_velocity = observation.dog_velocity
        if dog_velocity is None or np.shape(dog_velocity) != (2,):
            dog_velocity = np.zeros(2, dtype=float)
        else:
            dog_velocity = np.asarray(dog_velocity, dtype=float)

        goal_directions = vectors_to_goal / (
            distances_to_goal[:, np.newaxis] + self.settings.EPSILON
        )
        outward_directions = vectors_from_center / (
            distances_to_center[:, np.newaxis] + self.settings.EPSILON
        )
        goalward_speeds = np.sum(velocities * goal_directions, axis=1)
        outward_speeds = np.sum(velocities * outward_directions, axis=1)
        centroid_velocity = np.mean(velocities, axis=0)
        centroid_to_goal = observation.goal_position - center
        centroid_distance = np.linalg.norm(centroid_to_goal)
        centroid_goalward_speed = float(
            np.dot(
                centroid_velocity,
                centroid_to_goal / (centroid_distance + self.settings.EPSILON),
            )
        )

        cohesion_id = int(np.argmax(distances_to_center))
        hard_id = int(np.argmax(distances_to_goal))
        cohesion_drive_point = (
            positions[cohesion_id]
            + outward_directions[cohesion_id]
            * self.settings.COHESION_REPAIR_DISTANCE
        )
        hard_drive_point = (
            positions[hard_id]
            - goal_directions[hard_id] * self.settings.HARD_TO_GUIDE_DISTANCE
        )
        statuses = self._normalized_statuses(
            observation.sheep_statuses, positions.shape[0]
        )
        current = {
            "frame": observation.frame,
            "centroid_distance": float(centroid_distance),
            "max_spread": float(np.max(distances_to_center)),
            "mean_spread": float(np.mean(distances_to_center)),
        }
        self.state_history.append(current)

        if alarms != self.observed_alarms:
            self.observed_alarms = alarms
            self.alarm_started_frame = observation.frame if alarms else None

        oldest = self.state_history[0]
        return {
            "alarms": list(alarms),
            "primary_alarm": self._primary_alarm(alarms),
            "alarm_duration_frames": (
                observation.frame - self.alarm_started_frame + 1 if alarms else 0
            ),
            "context_span_frames": observation.frame - oldest["frame"],
            "current_mode": observation.current_mode,
            "centroid_distance": round(current["centroid_distance"], 3),
            "centroid_progress": round(
                oldest["centroid_distance"] - current["centroid_distance"], 3
            ),
            "max_spread": round(current["max_spread"], 3),
            "max_spread_change": round(
                current["max_spread"] - oldest["max_spread"], 3
            ),
            "mean_spread": round(current["mean_spread"], 3),
            "max_spread_threshold": self.settings.DANGER_DISPERSION_THRESHOLD,
            "mean_spread_threshold": self.settings.DANGER_MEAN_SPREAD_THRESHOLD,
            "mean_spread_change": round(
                current["mean_spread"] - oldest["mean_spread"], 3
            ),
            "inside_goal": int(
                np.sum(distances_to_goal < self.settings.GOAL_RADIUS)
            ),
            "outside_goal": int(
                np.sum(distances_to_goal >= self.settings.GOAL_RADIUS)
            ),
            "num_sheep": int(positions.shape[0]),
            "type_a_count": statuses.count("A"),
            "type_b_count": statuses.count("B"),
            "centroid_goalward_speed": round(centroid_goalward_speed, 3),
            "moving_away_from_goal_count": int(
                np.sum(goalward_speeds < -self.settings.EPSILON)
            ),
            "moving_outward_from_flock_count": int(
                np.sum(outward_speeds > self.settings.EPSILON)
            ),
            "nearest_boundary_margin": round(
                min(self._boundary_margin(point) for point in positions), 3
            ),
            "out_of_bounds_count": int(
                np.sum(
                    np.any(
                        (positions < 0.0)
                        | (positions > self.settings.SPACE_SIZE),
                        axis=1,
                    )
                )
            ),
            "dog_to_centroid": round(
                float(np.linalg.norm(observation.dog_position - center)), 3
            ),
            "dog_speed": round(float(np.linalg.norm(dog_velocity)), 3),
            "type_geometry": self._status_geometry(
                statuses,
                distances_to_goal,
                distances_to_center,
                goalward_speeds,
                outward_speeds,
            ),
            "cohesion_candidate": self._candidate_summary(
                cohesion_id,
                statuses,
                positions,
                observation.dog_position,
                cohesion_drive_point,
                distances_to_goal,
                distances_to_center,
                goalward_speeds,
                outward_speeds,
            ),
            "hard_to_guide_candidate": self._candidate_summary(
                hard_id,
                statuses,
                positions,
                observation.dog_position,
                hard_drive_point,
                distances_to_goal,
                distances_to_center,
                goalward_speeds,
                outward_speeds,
            ),
        }

    def _normalized_statuses(self, statuses, count):
        normalized = list(statuses[:count])
        normalized.extend(["Normal"] * (count - len(normalized)))
        return tuple(normalized)

    def _boundary_margin(self, point):
        return float(
            min(
                point[0],
                point[1],
                self.settings.SPACE_SIZE - point[0],
                self.settings.SPACE_SIZE - point[1],
            )
        )

    def _candidate_summary(
        self,
        target_id,
        statuses,
        positions,
        dog_position,
        drive_point,
        distances_to_goal,
        distances_to_center,
        goalward_speeds,
        outward_speeds,
    ):
        target = positions[target_id]
        desired_side = drive_point - target
        dog_side = dog_position - target
        alignment = float(
            np.dot(desired_side, dog_side)
            / (
                (np.linalg.norm(desired_side) + self.settings.EPSILON)
                * (np.linalg.norm(dog_side) + self.settings.EPSILON)
            )
        )
        dog_travel = float(np.linalg.norm(dog_position - drive_point))
        return {
            "target_id": target_id,
            "target_status": statuses[target_id],
            "target_goal_distance": round(float(distances_to_goal[target_id]), 3),
            "target_spread": round(float(distances_to_center[target_id]), 3),
            "target_goalward_speed": round(float(goalward_speeds[target_id]), 3),
            "target_outward_speed": round(float(outward_speeds[target_id]), 3),
            "dog_to_target": round(float(np.linalg.norm(dog_side)), 3),
            "dog_to_drive_point": round(dog_travel, 3),
            "estimated_travel_frames": round(
                dog_travel / max(self.settings.DOG_MAX_SPEED, self.settings.EPSILON),
                1,
            ),
            "dog_approach_alignment": round(alignment, 3),
            "drive_point_boundary_margin": round(
                self._boundary_margin(drive_point), 3
            ),
        }

    def _status_geometry(
        self,
        statuses,
        distances_to_goal,
        distances_to_center,
        goalward_speeds,
        outward_speeds,
    ):
        summaries = {}
        status_array = np.asarray(statuses)
        for status in ("A", "B"):
            mask = status_array == status
            count = int(np.sum(mask))
            if count == 0:
                summaries[status] = {"count": 0}
                continue
            summaries[status] = {
                "count": count,
                "mean_goal_distance": round(float(np.mean(distances_to_goal[mask])), 3),
                "max_goal_distance": round(float(np.max(distances_to_goal[mask])), 3),
                "mean_spread": round(float(np.mean(distances_to_center[mask])), 3),
                "mean_goalward_speed": round(float(np.mean(goalward_speeds[mask])), 3),
                "mean_outward_speed": round(float(np.mean(outward_speeds[mask])), 3),
            }
        return summaries

    def _response_metadata(self, response):
        keys = (
            "reason",
            "latency_seconds",
            "cache_hit",
            "model",
            "prompt_version",
            "response_id",
            "response_status",
            "incomplete_reason",
            "input_tokens",
            "output_tokens",
            "backend_error",
        )
        metadata = {
            key: response.get(key)
            for key in keys
            if response.get(key) is not None
        }
        if response.get("mode") is not None:
            metadata["response_mode"] = response.get("mode")
        if response.get("hold_frames") is not None:
            metadata["response_hold_frames"] = response.get("hold_frames")
        return metadata

    def _reference_mode(self, alarms):
        if alarms == ("flock_splitting",):
            return "COHESION_PRIORITY"
        if alarms == ("stagnation",):
            return "HARD_TO_GUIDE"
        return None

    def _metadata(
        self,
        status,
        request=None,
        selected_mode=None,
        latency_frames=None,
        hold_frames=None,
        rejection_reason=None,
        response_metadata=None,
        release_event=None,
    ):
        reference_mode = self._reference_mode(request.alarms) if request else None
        metadata = {
            "high_level_status": status,
            "request_id": request.request_id if request else None,
            "trigger_frame": request.trigger_frame if request else None,
            "trigger_alarm": request.primary_alarm if request else None,
            "trigger_alarms": list(request.alarms) if request else None,
            "state_summary": request.state_summary if request else None,
            "selected_mode": selected_mode,
            "reference_mode": reference_mode,
            "selection_correct": (
                selected_mode == reference_mode
                if selected_mode is not None and reference_mode is not None
                else None
            ),
            "latency_frames": latency_frames,
            "hold_frames": hold_frames,
            "rejection_reason": rejection_reason,
            "release_reason": (
                release_event.get("reason") if release_event else None
            ),
            "released_request_id": (
                release_event.get("request_id") if release_event else None
            ),
            "released_mode": (
                release_event.get("mode") if release_event else None
            ),
            "released_active_frames": (
                release_event.get("active_frames") if release_event else None
            ),
        }
        metadata.update(response_metadata or {})
        return metadata

    def _clear_pending(self):
        if self.pending_request is not None:
            self.backend.cancel(self.pending_request.request_id)
            self.pending_request = None

    def _clear_active(self):
        self.active_mode = None
        self.active_alarms = ()
        self.active_started_frame = None
        self.active_until_frame = None
        self.active_hold_frames = None
        self.active_request = None
        self.active_latency_frames = None
        self.active_response_metadata = {}

    def _release_active(self, reason, frame):
        if self.active_mode is None:
            return None
        event = {
            "reason": reason,
            "request_id": (
                self.active_request.request_id if self.active_request else None
            ),
            "mode": self.active_mode,
            "active_frames": max(0, frame - self.active_started_frame),
        }
        self._clear_active()
        return event

    def close(self):
        self._clear_pending()
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()


def build_controller(kind="rule", settings=config):
    if kind == "rule":
        return LowLevelController(settings)
    if kind == "mock-llm":
        return HighLevelController(settings=settings)
    if kind in ("openai", "deepseek"):
        from responses_api_backend import DeepSeekBackend, OpenAIBackend

        backend_class = OpenAIBackend if kind == "openai" else DeepSeekBackend
        return HighLevelController(
            backend=backend_class(settings=settings), settings=settings
        )
    raise ValueError(f"Unknown controller kind: {kind}")
