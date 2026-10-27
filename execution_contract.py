"""Explicit delegation and forecasts that run the real release/hold machinery."""
import copy
import numpy as np

from control_types import ControlObservation
from detector import Detector
from dynamics import advance_dog, advance_sheep
from mode_forecast import _flock_metrics, _inside_goal

VERSION = "explicit-delegation-v1"
POLICIES = ("FOLLOW_RULE", "COMMIT")
INSTRUCTIONS = """

EXPLICIT EXECUTION CONTRACT (overrides older hold/forecast descriptions above)
Return execution_policy as FOLLOW_RULE or COMMIT. FOLLOW_RULE means no high-level
mode or target lock: the ordinary rule controller keeps reconsidering both every
frame, with its original internal target hysteresis. Copy the request's rule_mode,
set target_sheep_id=null, and return a schema-valid hold_frames (ignored for this
policy). A fresh request is suppressed only for the configured retry interval.
COMMIT means intentionally hold the chosen mode and optional target for hold_frames,
subject to the real early-release rules: all alarms clearing, an ineligible target,
selected alarm clearing after minimum dwell, or changed trigger context. A null
target under COMMIT delegates target selection only; the mode is still held.
Choosing the current rule target under COMMIT deliberately commits to that ID.
Use FOLLOW_RULE when approving continued rule control; equality of the selected
mode/target alone does not mean delegation.
rule_forecast follows the changing rules. mode_forecasts and candidate forecasts
simulate COMMIT, with changing alarms and actual early releases, then follow rules;
they assume the execution_contract.forecast_hold_frames stated in the request.
Arrival validation recomputes forecasts for your actual hold_frames. Forecasts
exclude future model answers. They are short-horizon estimates, not safety proofs.
"""


def forecast_execution(observation, mode, hold_frames, horizon, settings,
                       low_level_controller, alarm_history, target_sheep_id=None,
                       execution_policy="COMMIT", trigger_alarms=None, return_trace=False):
    """No API calls; clone low-level memory and execute the actual high-level class.

    alarm_history includes the detector sample at observation.frame. The first
    observation uses its already-computed alarms; later ones use Detector.
    """
    from high_level_controller import HighLevelController, HighLevelRequest

    class NoRequests:
        source = "forecast"
        trigger_on_concurrent_only = True

    class ShadowController(HighLevelController):
        def _observe_state(self, observation, alarms):
            # No pending replies or future requests exist in this shadow run.
            # Prompt/report summaries cannot affect its decisions. Detector below
            # maintains the alarm history; real hold/release and motion still run.
            return {}

        def _request_ready(self, observation, state_summary):
            return False

    controller = ShadowController(copy.copy(low_level_controller), NoRequests(), settings)
    if execution_policy == "COMMIT":
        controller.active_mode = mode
        controller.active_alarms = tuple(observation.alarms if trigger_alarms is None else trigger_alarms)
        controller.active_started_frame = observation.frame
        controller.active_until_frame = observation.frame + hold_frames
        controller.active_hold_frames = hold_frames
        controller.active_target_id = target_sheep_id
        controller.active_request = HighLevelRequest("forecast", observation.frame,
            tuple(observation.alarms), None, {})
    elif execution_policy != "FOLLOW_RULE":
        raise ValueError("Unknown execution_policy")

    detector = Detector(settings.DANGER_DISPERSION_THRESHOLD,
        settings.DANGER_MEAN_SPREAD_THRESHOLD, settings.DANGER_STAGNATION_FRAMES,
        settings.DANGER_STAGNATION_THRESHOLD)
    detector.history_dist_to_goal = list(alarm_history)
    positions = np.array(observation.sheep_positions, dtype=float, copy=True)
    velocities = controller.low_level_controller._velocities(observation).copy()
    dog = np.array(observation.dog_position, dtype=float, copy=True)
    dog_velocity = np.zeros(2) if observation.dog_velocity is None else np.array(observation.dog_velocity, copy=True)
    goal = np.array(observation.goal_position, copy=True)
    statuses = tuple(controller._normalized_statuses(observation.sheep_statuses, len(positions)))
    initial = _flock_metrics(positions, goal, settings)
    inside = _inside_goal(positions, goal, settings)
    alarms = tuple(observation.alarms)
    exits = splitting = committed = simulated = 0
    peak = initial['out_of_bounds']
    margin = initial['boundary_margin']
    travel = 0.
    release_offset = release_reason = None
    current_mode = observation.current_mode
    trace = [] if return_trace else None
    for offset in range(horizon):
        if inside.all():
            break
        obs = ControlObservation(positions, dog, goal, alarms,
            observation.frame + offset, statuses, current_mode, velocities, dog_velocity)
        decision = controller.decide(obs)
        if decision.source == 'forecast':
            committed += 1
        elif execution_policy == 'COMMIT' and release_offset is None:
            release_offset = offset
            release_reason = decision.metadata.get('release_reason')
        current_mode = decision.mode
        next_dog, dog_velocity = advance_dog(dog, decision.navigation_point, settings)
        travel += float(np.linalg.norm(next_dog - dog))
        dog = next_dog
        positions, velocities = advance_sheep(positions, velocities,
            np.asarray(statuses), dog, goal, settings)
        metrics = _flock_metrics(positions, goal, settings)
        next_inside = _inside_goal(positions, goal, settings)
        exits += int(np.sum(inside & ~next_inside))
        inside = next_inside
        peak = max(peak, metrics['out_of_bounds'])
        margin = min(margin, metrics['boundary_margin'])
        splitting += int(metrics['max_spread'] > settings.DANGER_DISPERSION_THRESHOLD
                         and metrics['mean_spread'] > settings.DANGER_MEAN_SPREAD_THRESHOLD)
        simulated += 1
        if trace is not None:
            trace.append(dict(sheep_positions=positions.tolist(),sheep_velocities=velocities.tolist(),
                dog_position=dog.tolist(),dog_velocity=dog_velocity.tolist(),
                mode=decision.mode,target_sheep_id=decision.target_sheep_id))
        alarms = tuple(detector.detect(positions, goal)[0])
    final = _flock_metrics(positions, goal, settings)
    result = dict(
        horizon_frames=horizon,
        centroid_progress=round(initial['centroid_distance'] - final['centroid_distance'], 3),
        final_centroid_distance=round(final['centroid_distance'], 3),
        max_spread_change=round(final['max_spread'] - initial['max_spread'], 3),
        final_max_spread=round(final['max_spread'], 3),
        mean_spread_change=round(final['mean_spread'] - initial['mean_spread'], 3),
        final_mean_spread=round(final['mean_spread'], 3),
        inside_goal_change=final['inside_goal'] - initial['inside_goal'],
        final_inside_goal=final['inside_goal'], predicted_goal_exits=exits,
        predicted_splitting_frames=splitting, peak_out_of_bounds=peak,
        minimum_boundary_margin=round(margin, 3), dog_travel=round(travel, 3),
        predicted_success=bool(inside.all()), execution_policy=execution_policy,
        forecast_hold_frames=hold_frames if execution_policy == 'COMMIT' else None,
        simulated_frames=simulated, committed_frames=committed,
        release_offset=release_offset, release_reason=release_reason,
    )
    if trace is not None:
        result['trace'] = trace
    return result
