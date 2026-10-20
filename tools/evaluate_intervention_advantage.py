"""Offline intervention gate: all historical outcomes, seed-separated validation.

This first study deliberately retains legacy execution and a fixed 30-frame hold.
It does not relabel old outcomes as results of the new explicit-execution profile.
No model client, downloads, paid requests, or deployment are used here.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from detector import Detector
from environment import SwarmEnv
from experiment_audit import source_hashes, write_json_exclusive
from high_level_controller import HighLevelController
from low_level_controller import LowLevelController, rule_mode_for_alarms
from run_experiments import temporary_config

STUDY = ROOT / 'logs/experiments/label_feasibility_20261013'
DELAYS = (20, 50)
ALPHAS = (1., 10., 100.)
THRESHOLDS = (20., 50., 100.)  # Predicted reduction in 600-frame capped cost.
HOLD = 30
STATE_KEYS = ('centroid_distance', 'centroid_progress', 'max_spread', 'max_spread_change',
    'mean_spread', 'mean_spread_change', 'inside_goal', 'outside_goal', 'num_sheep',
    'centroid_goalward_speed', 'moving_away_from_goal_count', 'nearest_boundary_margin',
    'out_of_bounds_count', 'dog_to_centroid', 'dog_speed')
GEOMETRY_KEYS = ('target_goal_distance', 'target_spread', 'target_goalward_speed',
    'dog_to_target', 'dog_to_drive_point', 'estimated_travel_frames',
    'dog_approach_alignment', 'drive_point_boundary_margin')
FORECAST_KEYS = ('centroid_progress', 'final_centroid_distance', 'final_max_spread',
    'final_mean_spread', 'inside_goal_change', 'predicted_goal_exits',
    'predicted_splitting_frames', 'peak_out_of_bounds', 'minimum_boundary_margin', 'dog_travel')
MODES = ('TARGET_TRACKING', 'COHESION_PRIORITY', 'HARD_TO_GUIDE')
FEATURE_NAMES = (['commit'] + list(STATE_KEYS) + list(GEOMETRY_KEYS)
    + ['forecast_' + k for k in FORECAST_KEYS] + ['forecast_delta_' + k for k in FORECAST_KEYS]
    + ['mode_' + k for k in MODES] + ['explicit_target', 'same_rule_mode', 'same_default_target']
    + ['type_' + k for k in ('A', 'B', 'C', 'Normal')])


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'))


def action_forecast(summary, action):
    mode, target = action['mode'], action['target_sheep_id']
    if target is None:
        return summary['mode_forecasts'][mode], {}
    candidate = next(c for c in summary['repair_candidates'][mode] if c['target_id'] == target)
    return candidate['forecast'], candidate


def features(summary, action):
    if action is None:
        return np.zeros(len(FEATURE_NAMES))
    forecast, candidate = action_forecast(summary, action)
    baseline = summary['mode_forecasts'][summary['rule_mode']]
    default_key = {'COHESION_PRIORITY': 'cohesion_candidate', 'HARD_TO_GUIDE': 'hard_to_guide_candidate'}.get(action['mode'])
    if not candidate and default_key:
        candidate = summary[default_key]
    row = ([1.] + [float(summary.get(k, 0.)) for k in STATE_KEYS]
        + [float(candidate.get(k, 0.)) for k in GEOMETRY_KEYS]
        + [float(forecast[k]) for k in FORECAST_KEYS]
        + [float(forecast[k]) - float(baseline[k]) for k in FORECAST_KEYS]
        + [float(action['mode'] == mode) for mode in MODES]
        + [float(action['target_sheep_id'] is not None), float(action['mode'] == summary['rule_mode']),
           float(default_key is not None and action['target_sheep_id'] == summary[default_key].get('target_id'))]
        + [float(candidate.get('target_status') == status) for status in ('A', 'B', 'C', 'Normal')])
    assert len(row) == len(FEATURE_NAMES) and np.isfinite(row).all()
    return np.asarray(row)


def load_records(study=STUDY):
    manifest = json.loads((study / 'states.json').read_text())
    records = []
    for entry in manifest:
        data = json.loads((study / 'outcomes' / (entry['id'] + '.json')).read_text())
        assert (data['seed'], data['split']) == (entry['seed'], entry['split'])
        for delay in DELAYS:
            candidates = [o for o in data['delays'][str(delay)]['outcomes'] if o['action']['hold_frames'] == HOLD]
            assert len({action_key(o['action']) for o in candidates}) == len(candidates)
            records.append(dict(id=entry['id'], seed=entry['seed'], split=entry['split'], delay=delay,
                manifest=entry, summary=data['input_summary'], baseline=data['baseline'], outcomes=candidates))
    train_seeds = {r['seed'] for r in records if r['split'] == 'train'}
    validation_seeds = {r['seed'] for r in records if r['split'] == 'validation'}
    assert not train_seeds & validation_seeds
    return records


def capped_cost(score, limit=600):
    return score['elapsed_frames'] if score['success'] else limit


def training_rows(records):
    counts = {seed: sum(r['seed'] == seed for r in records) for seed in {r['seed'] for r in records}}
    x, y, weights = [], [], []
    for record in records:
        baseline = record['baseline']['600']
        choices = [(None, baseline)] + [(o['action'], o['scores']['600']) for o in record['outcomes']]
        for action, score in choices:
            x.append(features(record['summary'], action))
            y.append((capped_cost(baseline) - capped_cost(score)) / 600.)
            weights.append(1. / (counts[record['seed']] * len(choices)))
    return np.asarray(x), np.asarray(y), np.asarray(weights)


def fit(records, alpha):
    x, y, weights = training_rows(records)
    scale = np.sqrt(np.average(x*x, weights=weights, axis=0))
    scale[scale < 1e-9] = 1.
    z = x / scale
    weighted = z * np.sqrt(weights[:, None])
    coefficients = np.linalg.solve(weighted.T @ weighted + alpha * np.eye(z.shape[1]),
                                   weighted.T @ (y * np.sqrt(weights)))
    return dict(alpha=alpha, scale=scale.tolist(), coefficients=coefficients.tolist(),
                feature_names=FEATURE_NAMES, hold_frames=HOLD)


def predictions(model, record):
    x = np.asarray([features(record['summary'], o['action']) for o in record['outcomes']])
    return (x / np.asarray(model['scale'])) @ np.asarray(model['coefficients']) * 600.


def choose(record, values, threshold):
    best = int(np.argmax(values))
    return record['outcomes'][best]['action'] if values[best] > threshold else None


def short_forecast_choice(record):
    # Same available submission-time numbers; no outcome labels/arrival state.
    def rank(forecast):
        return (int(forecast['predicted_success']), forecast['final_inside_goal'],
                -forecast['peak_out_of_bounds'], -forecast['predicted_goal_exits'],
                forecast['centroid_progress'], -forecast['final_max_spread'])
    baseline = rank(record['summary']['mode_forecasts'][record['summary']['rule_mode']])
    candidates = [(rank(action_forecast(record['summary'], o['action'])[0]), o['action']) for o in record['outcomes']]
    best_score, best_action = max(candidates, key=lambda x: x[0])
    return best_action if best_score > baseline else None


def lookup(record, action):
    if action is None:
        return record['baseline']
    return next(o['scores'] for o in record['outcomes'] if action_key(o['action']) == action_key(action))


def select_model(records):
    assert records and all(r['split'] == 'train' for r in records), 'Validation must not tune the model'
    seeds = sorted({r['seed'] for r in records})
    np.random.default_rng(20261016).shuffle(seeds)
    folds = {seed: i % 5 for i, seed in enumerate(seeds)}
    trials = []
    for alpha in ALPHAS:
        oof = {}
        for fold in range(5):
            train = [r for r in records if folds[r['seed']] != fold]
            model = fit(train, alpha)
            for record in (r for r in records if folds[r['seed']] == fold):
                oof[record['id']] = predictions(model, record)
        for threshold in THRESHOLDS:
            chosen = [lookup(r, choose(r, oof[r['id']], threshold))['600'] for r in records]
            baseline = [r['baseline']['600'] for r in records]
            # Tune on train-only conditional outcomes; prefer safety and staying
            # with rules on ties. This is not full-task held-out evaluation.
            rescues = sum(a['success'] and not b['success'] for a,b in zip(chosen, baseline))
            regressions = sum(b['success'] and not a['success'] for a,b in zip(chosen, baseline))
            delta_cost = float(np.mean([capped_cost(a)-capped_cost(b) for a,b in zip(chosen,baseline)]))
            boundary_worse = sum(a['peak_out_of_bounds'] > b['peak_out_of_bounds'] for a,b in zip(chosen,baseline))
            interventions = sum(choose(r, oof[r['id']], threshold) is not None for r in records)
            trials.append(dict(alpha=alpha, threshold=threshold, rescues=rescues, regressions=regressions,
                mean_capped_frame_delta=delta_cost, boundary_worse=boundary_worse, interventions=interventions))
    selected = min(trials, key=lambda t: (t['regressions'], t['boundary_worse'],
        -(t['rescues']-t['regressions']), t['mean_capped_frame_delta'], t['interventions'], -t['threshold']))
    model = fit(records, selected['alpha'])
    model['threshold_frames'] = selected['threshold']
    return model, dict(selection=selected, trials=trials, seed_folds=folds)


class ReplayBackend:
    source = 'offline_advantage'
    trigger_on_concurrent_only = True

    def __init__(self, request, delay, action):
        self.request, self.delay, self.action = request, delay, action
        self.canceled = self.delivered = False

    def poll(self, identifier, frame):
        if self.action is None or self.canceled or self.delivered or frame < self.request.trigger_frame + self.delay:
            return None
        self.delivered = True
        return dict(request_id=identifier, trigger_frame=self.request.trigger_frame,
                    **self.action, reason='offline selected action')

    def cancel(self, identifier):
        self.canceled = True


class OneIntervention(HighLevelController):
    def _request_ready(self, observation, state_summary):
        return False


class LeanEnv(SwarmEnv):
    def _record_state(self):
        pass

    def _record_control_state(self, *args):
        pass


def rollout(entry, delay, action, expected):
    payload = Path(entry['path']).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == entry['sha256']
    state = pickle.loads(payload)  # Trusted, hash-verified local checkpoints only.
    env = LeanEnv.__new__(LeanEnv)
    for key, value in state['env'].items():
        setattr(env, key, copy.deepcopy(value))
    env.history_data, env.control_data, env.metrics_data = [], [], []
    env.decision_source, env.decision_frame, env.decision_metadata = 'rule', None, {}
    low = LowLevelController()
    for key, value in state['low'].items():
        setattr(low, key, value)
    backend = ReplayBackend(state['request'], delay, action)
    env.controller = OneIntervention(low, backend)
    for key, value in state['high'].items():
        setattr(env.controller, key, copy.deepcopy(value))
    env.controller.pending_request = copy.deepcopy(state['request'])
    detector = Detector(config.DANGER_DISPERSION_THRESHOLD, config.DANGER_MEAN_SPREAD_THRESHOLD,
                        config.DANGER_STAGNATION_FRAMES, config.DANGER_STAGNATION_THRESHOLD)
    detector.history_dist_to_goal = list(state['detector_history'])
    steps = danger = exits = overrides = 0
    inside = np.linalg.norm(env.sheep_pos - env.goal_pos, axis=1) < config.GOAL_RADIUS
    peak = int(np.any((env.sheep_pos < 0) | (env.sheep_pos > config.SPACE_SIZE), axis=1).sum())
    checkpoints = {}
    success = False
    def score():
        return dict(success=bool(success), elapsed_frames=steps, inside_goal=int(inside.sum()),
            danger_frames=danger, goal_exits=exits, peak_out_of_bounds=peak,
            centroid_distance=round(float(np.linalg.norm(env.sheep_pos.mean(axis=0)-env.goal_pos)), 3),
            override_frames=overrides)
    while env.current_frame < 2000 and not success:
        alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
        success = bool(env.step(alarms))
        steps += 1
        danger += int(metrics['is_danger'])
        current_inside = np.linalg.norm(env.sheep_pos-env.goal_pos, axis=1) < config.GOAL_RADIUS
        exits += int(np.sum(inside & ~current_inside))
        inside = current_inside
        peak = max(peak, int(np.any((env.sheep_pos < 0) | (env.sheep_pos > config.SPACE_SIZE), axis=1).sum()))
        overrides += int(env.decision_source == backend.source and (
            env.decision_metadata.get('target_override', False) or env.current_mode != rule_mode_for_alarms(alarms, True)))
        if steps in (300, 600):
            checkpoints[str(steps)] = score()
    for horizon in ('300', '600'):
        if horizon not in checkpoints:
            assert success
            checkpoints[horizon] = score()
    assert checkpoints == expected, 'Historical 300/600 outcomes must replay exactly before final evaluation'
    result = score()
    result['global_completion_or_stop_frame'] = env.current_frame
    return result


def evaluate_job(job):
    with temporary_config(NUM_SHEEP=20, NUM_ABNORMAL_A=5, NUM_ABNORMAL_B=5, NUM_ABNORMAL_C=5,
                          REPAIR_ENABLED=True, HIGH_LEVEL_MODE_SELECTION_ENABLED=True,
                          HIGH_LEVEL_TARGET_SELECTION_ENABLED=True, HIGH_LEVEL_EXPLICIT_EXECUTION=False):
        result = rollout(job['manifest'], job['delay'], job['action'], job['expected'])
    return {**{k:v for k,v in job.items() if k not in ('manifest', 'expected')}, 'final': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = load_records()
    protocol = dict(source_sha256=source_hashes(ROOT), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution_profile='legacy-v12; NOT explicit-delegation-v1', holds=[HOLD], delays=list(DELAYS),
        input_states=100, train_states=80, validation_states=20, seed_group_split=True,
        target='reduction in 600-frame capped completion cost; includes harmful, neutral and no-intervention rows',
        evaluation='freeze train-only group-CV choices, then replay one intervention followed by rules to global frame 2000',
        comparator='fixed lexicographic 30-frame numerical selector using same submission-time inputs',
        no_future_response_splicing=True, new_model_calls=0, llm_training=False,
        deployment=False, alpha_grid=ALPHAS, threshold_grid_frames=THRESHOLDS)
    write_json_exclusive(out / 'protocol.json', protocol)
    models, cv, jobs, dataset = {}, {}, [], []
    for delay in DELAYS:
        train = [r for r in records if r['delay']==delay and r['split']=='train']
        models[str(delay)], cv[str(delay)] = select_model(train)
        model = models[str(delay)]
        for record in (r for r in records if r['delay']==delay):
            dataset.append(dict(id=record['id'], seed=record['seed'], split=record['split'], delay=delay,
                action=None, benefit_600_frames=0, label='no_intervention'))
            for outcome in record['outcomes']:
                benefit = capped_cost(record['baseline']['600']) - capped_cost(outcome['scores']['600'])
                dataset.append(dict(id=record['id'], seed=record['seed'], split=record['split'], delay=delay,
                    action=outcome['action'], benefit_600_frames=benefit,
                    label='helpful' if benefit>0 else 'harmful' if benefit<0 else 'neutral',
                    accepted_at_arrival=outcome['status']=='active'))
            actions = {'rule': None, 'learned_gate': choose(record, predictions(model,record), model['threshold_frames']),
                       'short_forecast': short_forecast_choice(record)}
            for arm, action in actions.items():
                jobs.append(dict(id=record['id'], seed=record['seed'], split=record['split'], delay=delay,
                    arm=arm, action=action, manifest=record['manifest'], expected=lookup(record, action)))
    write_json_exclusive(out / 'frozen_models.json', models)
    write_json_exclusive(out / 'train_only_selection.json', cv)
    with (out / 'training_examples.jsonl').open('x', encoding='utf-8') as handle:
        for row in dataset:
            handle.write(json.dumps(row) + '\n')
    # Freeze every selected action before computing ANY new final outcomes.
    write_json_exclusive(out / 'frozen_actions.json', [{k:v for k,v in j.items() if k not in ('manifest','expected')} for j in jobs])
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_job, job) for job in jobs]
        with (out / 'full_horizon_results.jsonl').open('x', encoding='utf-8') as handle:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                handle.write(json.dumps(result) + '\n')
                handle.flush()
                if len(results) % 50 == 0:
                    print(json.dumps(dict(completed=len(results), total=len(jobs))), flush=True)
    summary = dict(new_model_calls=0, live_model_or_closed_loop_policy_test=False, arms={})
    for split in ('train','validation'):
        for delay in DELAYS:
            baseline = {r['id']:r for r in results if r['split']==split and r['delay']==delay and r['arm']=='rule'}
            for arm in ('rule','learned_gate','short_forecast'):
                rows = [r for r in results if r['split']==split and r['delay']==delay and r['arm']==arm]
                summary['arms'][f'{split}_{delay}_{arm}'] = dict(states=len(rows), seeds=len({r['seed'] for r in rows}),
                    chosen_interventions=sum(r['action'] is not None for r in rows),
                    successes=sum(r['final']['success'] for r in rows),
                    rescues=sum(r['final']['success'] and not baseline[r['id']]['final']['success'] for r in rows),
                    regressions=sum(not r['final']['success'] and baseline[r['id']]['final']['success'] for r in rows),
                    mean_capped_frames=float(np.mean([r['final']['global_completion_or_stop_frame'] for r in rows])),
                    mean_frame_delta=float(np.mean([r['final']['global_completion_or_stop_frame']-baseline[r['id']]['final']['global_completion_or_stop_frame'] for r in rows])),
                    boundary_worse=sum(r['final']['peak_out_of_bounds']>baseline[r['id']]['final']['peak_out_of_bounds'] for r in rows))
    summary['labels'] = {name:sum(r['label']==name for r in dataset) for name in ('helpful','harmful','neutral','no_intervention')}
    summary['limitations'] = ['Only 20 held-out states in 12 seed groups; not 20 independent seed trials.',
        'A selected action is applied once, then rules; not a multi-intervention closed-loop policy.',
        'Training labels use 600 frames; final evaluation uses the unchanged global 2000-frame cap.',
        'Legacy outcomes cannot certify the new explicit-execution forecasts/gates.',
        'Training-set final results are descriptive; held-out outcomes never tune or refit the model.']
    write_json_exclusive(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
