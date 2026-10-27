"""Event-triggered mode and repair-target selection for shepherd control."""

import copy
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol
import time

import numpy as np

import config
from control_types import ControlDecision, ControlObservation
from low_level_controller import LowLevelController, rule_mode_for_alarms
from mode_forecast import forecast_mode, forecast_modes
from experiment_audit import TIMING_FIELDS, utc_now
import target_handoff


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
    observed_at_utc: Optional[str] = None
    observed_perf_counter: Optional[float] = None


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
    """Keep low-level control active while a coordination request is pending."""

    def __init__(self, low_level_controller=None, backend=None, settings=config):
        self.settings = settings
        target_handoff.validate_settings(settings)
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
        self.active_target_id = None
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
        self.concurrent_alarm_frames = 0
        self.concurrent_request_submitted = False
        self.alarm_distance_history = deque(maxlen=int(settings.DANGER_STAGNATION_FRAMES))

    def decide(self, observation: ControlObservation) -> ControlDecision:
        self._observation_clock = (utc_now(), time.perf_counter())
        alarms = self._known_alarms(observation.alarms)
        state_summary = self._observe_state(observation, alarms)
        concurrent_only = getattr(
            self.backend, "trigger_on_concurrent_only", False
        )
        if concurrent_only and len(alarms) >= 2:
            self.concurrent_alarm_frames += 1
        else:
            self.concurrent_alarm_frames = 0
            self.concurrent_request_submitted = False
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
            release_reason = (
                "target_no_longer_eligible"
                if self.active_target_id is not None
                and not self.low_level_controller.repair_target_is_eligible(
                    observation, self.active_target_id
                )
                else self._active_release_reason(alarms, observation.frame)
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
                response = dict(response) if isinstance(response, Mapping) else response
                if isinstance(response, dict):
                    response.setdefault('consumed_at_utc', utc_now())
                    response.setdefault('consumed_frame', observation.frame)
                validation_summary = self._with_mode_forecasts(
                    observation, state_summary, response=response
                )
                mode, hold_frames, rejection_reason = self._validate_response(
                    response,
                    observation.frame,
                    alarms,
                    validation_summary,
                )
                request = self.pending_request
                if isinstance(response, dict):
                    response.update(validated_at_utc=utc_now(),
                                    latency_frames=observation.frame - request.trigger_frame)
                    audit = getattr(self.backend, 'record_event', None)
                    if audit is not None:
                        audit('validated', request.request_id, frame=observation.frame,
                              accepted=rejection_reason is None,
                              rejection_reason=rejection_reason)
                completed_request = request
                self.pending_request = None
                if rejection_reason is None:
                    if self._response_delegates_to_rule(response, validation_summary):
                        self._clear_active()
                        self.retry_after_frame = observation.frame + self.settings.HIGH_LEVEL_RETRY_FRAMES
                        return self._low_level_decision(
                            observation, status='delegated', request=request,
                            response_metadata=self._response_metadata(response),
                            release_event=release_event)
                    self.active_mode = mode
                    self.active_alarms = request.alarms
                    self.active_started_frame = observation.frame
                    self.active_until_frame = observation.frame + hold_frames
                    self.active_hold_frames = hold_frames
                    self.active_target_id = response.get("target_sheep_id")
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
                selected_target_id=self.active_target_id,
            )
            rule_decision = copy.copy(self.low_level_controller).decide(observation)
            decision = self.low_level_controller.decide(
                observation, mode_override=(
                    self.active_mode
                    if getattr(self.settings, "HIGH_LEVEL_MODE_SELECTION_ENABLED", True)
                    else None
                ),
                target_override=self.active_target_id,
            )
            metadata["rule_target_sheep_id"] = rule_decision.target_sheep_id
            metadata["target_override"] = (
                decision.mode == rule_decision.mode
                and decision.target_sheep_id != rule_decision.target_sheep_id
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
            and self._request_ready(observation, state_summary)
            and (
                not concurrent_only
                or (
                    len(alarms) >= 2
                    and self.concurrent_alarm_frames
                    >= self.settings.HIGH_LEVEL_CONCURRENT_MIN_FRAMES
                    and (
                        not self.settings.HIGH_LEVEL_ONE_REQUEST_PER_EPISODE
                        or not self.concurrent_request_submitted
                    )
                )
            )
        ):
            self.pending_request = self._make_request(
                observation, alarms, state_summary
            )
            self.backend.submit(self.pending_request)
            if concurrent_only:
                self.concurrent_request_submitted = True

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

    def _request_ready(self, observation, state_summary):
        return True

    def _make_request(self, observation, alarms, state_summary):
        self.request_counter += 1
        return HighLevelRequest(
            request_id=f"high-{self.request_counter}",
            trigger_frame=observation.frame,
            alarms=alarms,
            primary_alarm=self._primary_alarm(alarms),
            state_summary=self._with_mode_forecasts(
                observation, state_summary, include_targets=True
            ),
            observed_at_utc=self._observation_clock[0],
            observed_perf_counter=self._observation_clock[1],
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
        if not isinstance(mode, str) or mode not in ALLOWED_MODES:
            return None, None, "mode_not_allowed"
        if getattr(self.settings, 'HIGH_LEVEL_EXPLICIT_EXECUTION', False):
            from execution_contract import POLICIES
            policy = response.get('execution_policy')
            if policy not in POLICIES:
                return None, None, 'execution_policy_not_allowed'
            if policy == 'FOLLOW_RULE':
                if mode != request.state_summary.get('rule_mode') or response.get('target_sheep_id') is not None:
                    return None, None, 'invalid_rule_delegation'
                hold = response.get('hold_frames')
                if type(hold) is not int or not self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES <= hold <= self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES:
                    return None, None, 'hold_frames_out_of_range'
                return current_summary['rule_mode'], hold, None
        if not getattr(self.settings, "HIGH_LEVEL_MODE_SELECTION_ENABLED", True):
            if mode != request.state_summary.get("rule_mode"):
                return None, None, "mode_selection_disabled"
            if mode != current_summary.get("rule_mode"):
                return None, None, "rule_mode_changed"
        target = response.get("target_sheep_id")
        if target is not None:
            if not getattr(self.settings, "HIGH_LEVEL_TARGET_SELECTION_ENABLED", True):
                return None, None, "target_selection_disabled"
            if mode == "TARGET_TRACKING":
                return None, None, "target_requires_repair_mode"
            if not isinstance(target, int) or isinstance(target, bool):
                return None, None, "target_not_integer"
            offered = request.state_summary.get("repair_candidates", {}).get(mode, [])
            if not any(item["target_id"] == target for item in offered):
                return None, None, "target_not_offered"
        if target_handoff.enabled(self.settings):
            if not {'flock_splitting', 'stagnation'} <= set(alarms):
                return None, None, 'concurrent_alarms_cleared'
            if target_handoff.delegates(target, current_summary['delivery_rule_target_sheep_id']):
                hold = response.get('hold_frames')
                if type(hold) is not int:
                    return None, None, 'hold_frames_not_integer'
                if not self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES <= hold <= self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES:
                    return None, None, 'hold_frames_out_of_range'
                return current_summary['rule_mode'], hold, None
        if current_summary.get("target_selection_error"):
            return None, None, current_summary["target_selection_error"]
        if (
            target is not None
            and current_summary.get("selected_target_candidate", {}).get("target_id") != target
        ):
            return None, None, "target_forecast_missing"
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
        selected_target = current_summary.get("selected_target_candidate")
        candidate = current_summary.get(candidate_key, {}) if candidate_key else {}
        if selected_target is not None:
            candidate = selected_target
        boundary_margin = candidate.get("drive_point_boundary_margin")
        if (
            isinstance(boundary_margin, (int, float))
            and boundary_margin < 0.0
        ):
            return None, None, "candidate_drive_point_out_of_bounds"
        request_summary = request.state_summary
        selection_summary = current_summary
        if selected_target is not None:
            request_candidate = next(
                item for item in request_summary["repair_candidates"][mode]
                if item["target_id"] == response["target_sheep_id"]
            )
            request_summary = dict(request_summary)
            request_summary["mode_forecasts"] = {
                **request_summary["mode_forecasts"], mode: request_candidate["forecast"]
            }
            selection_summary = dict(current_summary)
            selection_summary["mode_forecasts"] = {
                **current_summary["mode_forecasts"], mode: selected_target["forecast"]
            }
        if self._forecast_preference_reversed(
            mode, request_summary, selection_summary
        ):
            return None, None, "forecast_preference_reversed"
        dominating_mode = self._dominating_forecast_mode(
            mode,
            alarms,
            selection_summary.get("mode_forecasts", {}),
        )
        if dominating_mode is not None:
            return None, None, f"forecast_dominated_by_{dominating_mode.lower()}"
        if selected_target is not None:
            fallback_forecast = current_summary["mode_forecasts"][mode]
            if self._forecast_dominates(fallback_forecast, selected_target["forecast"]):
                return None, None, "forecast_dominated_by_rule_target"
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

    def _response_delegates_to_rule(self, response, summary):
        return (
            target_handoff.enabled(self.settings)
            and target_handoff.delegates(response.get('target_sheep_id'),
                summary['delivery_rule_target_sheep_id'])
        ) or (
            getattr(self.settings, 'HIGH_LEVEL_EXPLICIT_EXECUTION', False)
            and response.get('execution_policy') == 'FOLLOW_RULE'
        )

    @staticmethod
    def _forecast_preference_reversed(mode, request_summary, current_summary):
        request_rule = request_summary.get("rule_mode")
        current_rule = current_summary.get("rule_mode")
        if mode == current_rule or request_rule != current_rule:
            return False
        request_forecasts = request_summary.get("mode_forecasts", {})
        current_forecasts = current_summary.get("mode_forecasts", {})
        forecasts = (
            request_forecasts.get(mode),
            request_forecasts.get(request_rule),
            current_forecasts.get(mode),
            current_forecasts.get(current_rule),
        )
        if not all(isinstance(forecast, Mapping) for forecast in forecasts):
            return False
        try:
            request_choice, request_fallback, current_choice, current_fallback = (
                (float(forecast["centroid_progress"]),
                 float(forecast["final_max_spread"]))
                for forecast in forecasts
            )
        except (KeyError, TypeError, ValueError):
            return False
        return (
            request_choice[0] > request_fallback[0]
            and request_choice[1] < request_fallback[1]
            and current_choice[0] < current_fallback[0]
            and current_choice[1] > current_fallback[1]
        )

    def _with_mode_forecasts(
        self, observation, state_summary, include_targets=False, response=None
    ):
        summary = dict(state_summary)
        if target_handoff.enabled(self.settings):
            return self._with_target_handoff_forecasts(observation, summary, include_targets, response)
        if getattr(self.settings, 'HIGH_LEVEL_EXPLICIT_EXECUTION', False):
            return self._with_execution_forecasts(observation, summary, include_targets, response)
        summary["mode_forecasts"] = forecast_modes(
            observation,
            self.settings,
            self.low_level_controller,
        )
        if include_targets:
            summary["repair_candidates"] = {
                mode: [
                    self._target_candidate(observation, mode, target)
                    for target in self.low_level_controller.repair_candidates(observation, mode)
                ]
                for mode in ("HARD_TO_GUIDE", "COHESION_PRIORITY")
            }
        self._add_selected_target_forecast(observation, summary, response)
        return summary

    def _with_target_handoff_forecasts(self, observation, summary, include_targets, response):
        hold = response.get('hold_frames') if isinstance(response, Mapping) else None
        if type(hold) is not int or not self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES <= hold <= self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES:
            hold = int(self.settings.HIGH_LEVEL_FORECAST_FRAMES)
        horizon = max(1, int(self.settings.HIGH_LEVEL_FORECAST_FRAMES))
        rule = copy.copy(self.low_level_controller).decide(observation)
        def predict(mode, target=None):
            return target_handoff.forecast_action(observation, mode, hold, horizon, self.settings,
                self.low_level_controller, self.alarm_distance_history, target)
        summary['target_handoff'] = dict(version=target_handoff.VERSION, forecast_hold_frames=hold,
            admission_requires_both_alarms=True, same_target_delegates=True,
            release_waives_inherited_minimum_hold=True)
        summary['delivery_rule_target_sheep_id'] = rule.target_sheep_id
        summary['rule_forecast'] = predict(summary['rule_mode'])
        summary['mode_forecasts'] = {mode: predict(mode) for mode in ALLOWED_MODES}
        if include_targets:
            summary['repair_candidates'] = {
                mode: [self._target_candidate(observation, mode, target, hold)
                    for target in self.low_level_controller.repair_candidates(observation, mode)]
                for mode in ('HARD_TO_GUIDE', 'COHESION_PRIORITY')}
        self._add_selected_target_forecast(observation, summary, response, hold, exact_integer=True)
        return summary

    def _with_execution_forecasts(self, observation, summary, include_targets, response):
        from execution_contract import forecast_execution, VERSION
        # Delegation needs identity/age checks, not intervention-geometry rejection.
        if isinstance(response, Mapping) and response.get('execution_policy') == 'FOLLOW_RULE':
            return summary
        hold = response.get('hold_frames') if isinstance(response, Mapping) else None
        if type(hold) is not int or not self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES <= hold <= self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES:
            hold = int(self.settings.HIGH_LEVEL_FORECAST_FRAMES)
        horizon = max(1, int(self.settings.HIGH_LEVEL_FORECAST_FRAMES))
        def predict(mode, target=None, policy='COMMIT'):
            return forecast_execution(observation, mode, hold, horizon, self.settings,
                self.low_level_controller, self.alarm_distance_history, target, policy,
                self.pending_request.alarms if self.pending_request is not None else observation.alarms)
        summary['execution_contract'] = dict(version=VERSION, forecast_hold_frames=hold,
            follow_rule_cooldown_frames=self.settings.HIGH_LEVEL_RETRY_FRAMES)
        summary['rule_forecast'] = predict(summary['rule_mode'], policy='FOLLOW_RULE')
        summary['mode_forecasts'] = {mode: predict(mode) for mode in ALLOWED_MODES}
        if include_targets:
            summary['repair_candidates'] = {
                mode: [self._target_candidate(observation, mode, target, hold)
                       for target in self.low_level_controller.repair_candidates(observation, mode)]
                for mode in ('HARD_TO_GUIDE', 'COHESION_PRIORITY')}
        self._add_selected_target_forecast(observation, summary, response, hold, exact_integer=True)
        return summary

    def _add_selected_target_forecast(
        self, observation, summary, response, hold_frames=None, *, exact_integer=False
    ):
        """Share target checks across forecast contracts without changing their semantics."""
        if not isinstance(response, Mapping) or response.get('target_sheep_id') is None:
            return
        mode, target = response.get('mode'), response['target_sheep_id']
        offered = (
            self.pending_request.state_summary.get('repair_candidates', {}).get(mode, [])
            if isinstance(mode, str) else []
        )
        integer = type(target) is int if exact_integer else isinstance(target, int) and not isinstance(target, bool)
        if mode not in ('HARD_TO_GUIDE', 'COHESION_PRIORITY'):
            summary['target_selection_error'] = 'target_requires_repair_mode'
        elif not integer:
            summary['target_selection_error'] = 'target_not_integer'
        elif not any(item['target_id'] == target for item in offered):
            summary['target_selection_error'] = 'target_not_offered'
        elif not self.low_level_controller.repair_target_is_eligible(observation, target):
            summary['target_selection_error'] = 'target_no_longer_eligible'
        else:
            summary['selected_target_candidate'] = self._target_candidate(observation, mode, target, hold_frames)

    def _target_candidate(self, observation, mode, target_id, hold_frames=None, *, include_forecast=True):
        positions = observation.sheep_positions
        center = np.mean(positions, axis=0)
        goal_vectors = observation.goal_position - positions
        center_vectors = positions - center
        goal_distances = np.linalg.norm(goal_vectors, axis=1)
        center_distances = np.linalg.norm(center_vectors, axis=1)
        velocities = self.low_level_controller._velocities(observation)
        goalward = np.sum(
            velocities * goal_vectors / (goal_distances[:, None] + self.settings.EPSILON),
            axis=1,
        )
        outward = np.sum(
            (velocities - np.mean(velocities, axis=0)) * center_vectors
            / (center_distances[:, None] + self.settings.EPSILON),
            axis=1,
        )
        decision = copy.copy(self.low_level_controller).decide(
            observation, mode_override=mode, target_override=target_id
        )
        summary = self._candidate_summary(
            target_id, self._normalized_statuses(observation.sheep_statuses, len(positions)),
            positions, observation.dog_position, decision.drive_point,
            goal_distances, center_distances, goalward, outward,
        )
        if not include_forecast:
            return summary
        if target_handoff.enabled(self.settings):
            summary['forecast'] = target_handoff.forecast_action(observation, mode,
                int(self.settings.HIGH_LEVEL_FORECAST_FRAMES) if hold_frames is None else hold_frames,
                max(1, int(self.settings.HIGH_LEVEL_FORECAST_FRAMES)), self.settings,
                self.low_level_controller, self.alarm_distance_history, target_id)
        elif getattr(self.settings, 'HIGH_LEVEL_EXPLICIT_EXECUTION', False):
            from execution_contract import forecast_execution
            summary['forecast'] = forecast_execution(observation, mode,
                int(self.settings.HIGH_LEVEL_FORECAST_FRAMES) if hold_frames is None else hold_frames,
                max(1, int(self.settings.HIGH_LEVEL_FORECAST_FRAMES)), self.settings,
                self.low_level_controller, self.alarm_distance_history, target_id,
                trigger_alarms=self.pending_request.alarms if self.pending_request is not None else observation.alarms)
        else:
            summary["forecast"] = forecast_mode(
                observation, mode, max(1, int(self.settings.HIGH_LEVEL_FORECAST_FRAMES)),
                self.settings, self.low_level_controller, target_sheep_id=target_id,
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
        if (
            not getattr(self.settings, "HIGH_LEVEL_MODE_SELECTION_ENABLED", True)
            and self.active_mode != rule_mode_for_alarms(alarms, self.settings.REPAIR_ENABLED)
        ):
            return "rule_mode_changed"
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

    def _rule_mode(self, alarms):
        return rule_mode_for_alarms(alarms, self.settings.REPAIR_ENABLED)

    def _observe_state(self, observation, alarms):
        positions = observation.sheep_positions
        center = np.mean(positions, axis=0)
        self.alarm_distance_history.append(float(np.linalg.norm(observation.goal_position - center)))
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
        centroid_velocity = np.mean(velocities, axis=0)
        # A translating flock is not dispersing: measure radial motion relative
        # to its moving centroid, exactly as in _target_candidate.
        outward_speeds = np.sum((velocities - centroid_velocity) * outward_directions, axis=1)
        centroid_to_goal = observation.goal_position - center
        centroid_distance = np.linalg.norm(centroid_to_goal)
        centroid_goalward_speed = float(
            np.dot(
                centroid_velocity,
                centroid_to_goal / (centroid_distance + self.settings.EPSILON),
            )
        )

        cohesion_decision = copy.copy(self.low_level_controller).decide(
            observation, mode_override="COHESION_PRIORITY"
        )
        hard_decision = copy.copy(self.low_level_controller).decide(
            observation, mode_override="HARD_TO_GUIDE"
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
        summary = {
            "alarms": list(alarms),
            "primary_alarm": self._primary_alarm(alarms),
            "rule_mode": self._rule_mode(alarms),
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
                cohesion_decision.target_sheep_id,
                statuses,
                positions,
                observation.dog_position,
                cohesion_decision.drive_point,
                distances_to_goal,
                distances_to_center,
                goalward_speeds,
                outward_speeds,
            ),
            "hard_to_guide_candidate": self._candidate_summary(
                hard_decision.target_sheep_id,
                statuses,
                positions,
                observation.dog_position,
                hard_decision.drive_point,
                distances_to_goal,
                distances_to_center,
                goalward_speeds,
                outward_speeds,
            ),
        }
        if "C" in statuses:
            summary["type_c_count"] = statuses.count("C")
            summary["type_c_response"] = {
                "sensitivity_radius": self.settings.TYPE_C_SENSITIVITY_RADIUS,
                "max_dog_repulsion_factor": self.settings.WEIGHT_DOG_REPULSION_C_FACTOR,
                "behavior": "Normal flocking; dog repulsion rises linearly as distance falls below sensitivity_radius; normal speed cap.",
            }
        return summary

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
        for status in ("A", "B", "C"):
            mask = status_array == status
            count = int(np.sum(mask))
            if status == "C" and count == 0:
                continue
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
            "served_model",
            "execution_policy",
            "prompt_version",
            "response_id",
            "response_status",
            "incomplete_reason",
            "input_tokens",
            "output_tokens",
            "backend_error",
            "latency_frames",
            *TIMING_FIELDS,
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
        metadata["response_target_sheep_id"] = response.get("target_sheep_id")
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
        selected_target_id=None,
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
            "selected_target_sheep_id": selected_target_id,
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
        self.active_target_id = None
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
        if target_handoff.enabled(self.settings) and self.active_target_id is not None:
            self.low_level_controller.release_repair_target_hold()
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
    if kind in ("openai", "deepseek", "vllm"):
        from responses_api_backend import DeepSeekBackend, OpenAIBackend
        from vllm_backend import VLLMBackend

        backend_class = {"openai": OpenAIBackend, "deepseek": DeepSeekBackend, "vllm": VLLMBackend}[kind]
        return HighLevelController(
            backend=backend_class(settings=settings), settings=settings
        )
    raise ValueError(f"Unknown controller kind: {kind}")
