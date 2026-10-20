"""Opt-in target-only admission and execution-aligned forecast contract."""
import copy

from low_level_controller import rule_mode_for_alarms

VERSION = 'target-handoff-v1'
INSTRUCTIONS = """

TARGET HANDOFF CONTRACT (overrides older hold/forecast descriptions above)
The rule chooses every mode. A null target, or the rule's current target at
response arrival, delegates to the changing rules without a target lock.
A different target can start an intervention only if BOTH splitting and
stagnation alarms are still active at arrival. Existing interventions retain
their usual early-release conditions; losing one alarm alone need not end one.
On release, the rule immediately reassesses urgency without a remaining minimum
hold inherited from the intervention. It may keep the same target if competitive.
rule_forecast follows dynamic rules. Candidate forecasts simulate this exact
admission, hold, dynamic-alarm release and handoff, with no future model replies.
They use target_handoff.forecast_hold_frames; arrival validation uses your actual
hold_frames. These short-horizon estimates do not establish eventual success.
"""


def enabled(settings):
    return bool(getattr(settings, 'HIGH_LEVEL_TARGET_HANDOFF_ENABLED', False))


def validate_settings(settings):
    if enabled(settings) and (
        getattr(settings, 'HIGH_LEVEL_MODE_SELECTION_ENABLED', True)
        or not getattr(settings, 'HIGH_LEVEL_TARGET_SELECTION_ENABLED', True)
        or getattr(settings, 'HIGH_LEVEL_EXPLICIT_EXECUTION', False)
    ):
        raise ValueError('Target handoff requires target-only permissions and the original decision schema.')


def delegates(target, rule_target):
    return target is None or target == rule_target


def forecast_action(observation, mode, hold, horizon, settings, low, history, target=None):
    from execution_contract import forecast_execution
    rule = copy.copy(low).decide(observation)
    admissible = (
        {'flock_splitting', 'stagnation'} <= set(observation.alarms)
        and mode == rule_mode_for_alarms(observation.alarms, settings.REPAIR_ENABLED)
        and not delegates(target, rule.target_sheep_id)
    )
    return forecast_execution(observation, mode, hold, horizon, settings, low,
        history, target_sheep_id=target, execution_policy='COMMIT' if admissible else 'FOLLOW_RULE')
