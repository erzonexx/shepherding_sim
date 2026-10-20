"""Deterministic baseline: select only from the forecasts supplied to the LLM."""
import math

VERSION = 'forecast-lexicographic-v1'
ORDER = (
    ('peak_out_of_bounds', 1), ('predicted_goal_exits', 1),
    ('inside_goal_change', -1), ('centroid_progress', -1),
    ('final_max_spread', 1), ('predicted_splitting_frames', 1),
)


def choose(summary):
    mode = summary['rule_mode']
    rule_target = summary.get('delivery_rule_target_sheep_id')
    options = [(None, summary['rule_forecast'])] + [
        (c['target_id'], c['forecast']) for c in summary['repair_candidates'].get(mode, [])]
    if len({t for t, _ in options}) != len(options):
        raise ValueError('Duplicate candidate target')
    ranked = []
    for target, forecast in options:
        key = []
        for name, sign in ORDER:
            value = float(forecast[name])
            if not math.isfinite(value):
                raise ValueError('Nonfinite forecast metric')
            key.append(sign * value)
        tie = (0 if target is None else 1 if target == rule_target else 2,
               -1 if target is None else target)
        ranked.append(dict(target=target, metric_key=key, tie_key=list(tie)))
    ranked.sort(key=lambda r: (r['metric_key'], r['tie_key']))
    return ranked[0]['target'], ranked
