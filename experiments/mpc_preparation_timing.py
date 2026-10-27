"""Offline before/after timing with exact historical inputs and forecast traces."""
import argparse
import copy
import cProfile
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import pickle
import pstats
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import config
from control_types import ControlObservation
from detector import Detector
from environment import SwarmEnv
from experiment_audit import source_hashes, utc_now
from high_level_controller import HighLevelController, HighLevelRequest
from realtime_controller import NoRequests, public_settings, snapshot, predictor
from run_experiments import temporary_config
from tools.spark.benchmark_session import write
from experiments.mpc_latency_diagnostic import digest, read, canonical
import mpc_controller as current
import execution_contract as execution

OUT = ROOT/'logs/experiments/mpc_preparation_optimization_20261021'
REFERENCE = ROOT/'logs/experiments/mpc_pilot_10hz_20261020_v2'
DIAGNOSTIC = ROOT/'logs/experiments/mpc_latency_diagnostic_20261021_v2'
SCRIPT = 'experiments/mpc_preparation_timing.py'
REPEATS = 4
CHANGED = {'high_level_controller.py', 'execution_contract.py', 'mpc_controller.py'}


def request(r):
    return HighLevelRequest(**dict(r, alarms=tuple(r['alarms'])))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline():
    old_execution = load_module('_preparation_old_execution', OUT/'source_before/execution_contract.py')
    old = load_module('_preparation_old_mpc', OUT/'source_before/mpc_controller.py')
    old.forecast_execution = old_execution.forecast_execution
    return old, old_execution


def restore_rank_state(item):
    s = item['result']['prediction_state']
    obs = ControlObservation(np.array(s['positions']), np.array(s['dog']), np.array(s['goal']),
        tuple(s['alarms']), s['frame'], tuple(s['statuses']), s['current_mode'],
        np.array(s['velocities']), np.array(s['dog_velocity']))
    return dict(observation=obs, low=s['low'], history=s['history'],
                summary=dict(rule_mode=item['result']['candidates'][0]['mode']))


def prepare():
    assert not OUT.exists(), 'Preserve existing preparation evidence'
    before = dict(source_hashes(ROOT), **{'mpc_controller.py':digest(ROOT/'mpc_controller.py'),
        'realtime_controller.py':digest(ROOT/'realtime_controller.py')})
    old_manifest = read(DIAGNOSTIC/'protocol.json')['source_sha256']
    assert all(before[k] == old_manifest[k] for k in before)
    OUT.mkdir(parents=True)
    for name in before:
        target = OUT/'source_before'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    values = {**public_settings(config), **read(REFERENCE/'protocol.json')['config']}
    originals = read(DIAGNOSTIC/'inputs.json')
    captures = {}
    for seed in sorted({i['seed'] for i in originals}):
        wanted = {i['request']['trigger_frame'] for i in originals if i['seed'] == seed}
        class Capture(HighLevelController):
            def _request_ready(self, *args):
                return False

            def _observe_state(self, obs, alarms):
                summary = super()._observe_state(obs, alarms)
                if obs.frame in wanted:
                    captures[(seed, obs.frame)] = snapshot(self, obs, summary)
                return summary
        with temporary_config(**values):
            env = SwarmEnv(Capture(backend=NoRequests()))
            initial = read(REFERENCE/'mpc'/str(seed)/'initial.json')
            env.sheep_pos = np.array(initial['positions'])
            env.sheep_vel = np.array(initial['velocities'])
            env.dog_pos = np.array(initial['dog'])
            env.dog_vel = np.array(initial['dog_velocity'])
            env.goal_pos = np.array(initial['goal'])
            env.sheep_status = np.array(initial['statuses'], dtype=object)
            detector = Detector(config.DANGER_DISPERSION_THRESHOLD, config.DANGER_MEAN_SPREAD_THRESHOLD,
                                config.DANGER_STAGNATION_FRAMES, config.DANGER_STAGNATION_THRESHOLD)
            for _ in range(max(wanted)+1):
                alarms = detector.detect(env.sheep_pos, env.goal_pos)[0]
                env.step(alarms)
            # Offline reconstruction only: no close/save of a new efficacy run.
    records = []
    for item in originals:
        r = request(item['request'])
        state = captures[(item['seed'], r.trigger_frame)]
        r = replace(r, state_summary=state['summary'])
        prepared = current.proposal_input(state, r, values)
        assert canonical(prepared.state_summary) == canonical(item['request']['state_summary']), \
            f'Historical input not reproduced: {item["seed"]}:{r.trigger_frame}'
        records.append(dict(seed=item['seed'], state=state, request=r,
                            historical_sha256=item['input_sha256']))
    with (OUT/'replay_inputs.pickle').open('xb') as f:
        pickle.dump(records, f)
    write(OUT/'protocol.json', dict(at_utc=utc_now(), source_before_sha256=before,
        benchmark_sha256=digest(ROOT/SCRIPT), replay_inputs_sha256=digest(OUT/'replay_inputs.pickle'),
        historical_inputs_sha256=digest(DIAGNOSTIC/'inputs.json'), config=values,
        inputs=len(records), paired_repetitions=REPEATS, allowed_changed_sources=sorted(CHANGED),
        api_calls=0, model_started=False, full_real_time_simulations=0,
        validation='Exact archived17 prompts; paired before/after times; full forecast trace and low-level memory identity.',
        limits=['Offline preparation and ranking timing only; not actual API/delivery or control efficacy.',
                'Four alternating repetitions per input on current host; not a broad hardware benchmark.']))
    profiler = cProfile.Profile()
    for i in (0, 7, 16):
        item = records[i]
        profiler.runcall(current.proposal_input, item['state'], item['request'], values)
    profiler.dump_stats(OUT/'preparation_before.prof')
    with (OUT/'profile_before.txt').open('w', encoding='utf-8') as f:
        pstats.Stats(profiler, stream=f).sort_stats('cumulative').print_stats(35)
    write(OUT/'baseline_prepared.json', dict(at_utc=utc_now(), exact_historical_prompts=len(records),
        replay_inputs_sha256=digest(OUT/'replay_inputs.pickle'), api_calls=0))
    print('Prepared17 exact historical snapshots and profile; no model requests.', flush=True)


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values,95)), maximum=float(max(values)))


def compare():
    assert not (OUT/'comparison_started.json').exists(), 'Do not repeat or overwrite comparison'
    p = read(OUT/'protocol.json')
    assert digest(ROOT/SCRIPT) == p['benchmark_sha256']
    assert digest(OUT/'replay_inputs.pickle') == p['replay_inputs_sha256']
    for name, h in p['source_before_sha256'].items():
        if name not in CHANGED:
            assert digest(ROOT/name) == h
    write(OUT/'comparison_started.json', dict(at_utc=utc_now()))
    with (OUT/'replay_inputs.pickle').open('rb') as f:
        records = pickle.load(f)
    values = p['config']
    old, old_execution = baseline()
    # Baseline must use the saved candidate method, including its legacy forecast.
    old_high = load_module('_preparation_old_high', OUT/'source_before/high_level_controller.py')
    def old_predictor(state, settings):
        controller = predictor(state, settings)
        controller._target_candidate = old_high.HighLevelController._target_candidate.__get__(controller)
        return controller
    old.predictor = old_predictor
    rows, trace_checks = [], 0
    for i, item in enumerate(records):
        state, r = item['state'], item['request']
        before_state = pickle.dumps(state)
        # Warm both implementations separately, excluded from timed rows.
        a, b = old.proposal_input(state, r, values), current.proposal_input(state, r, values)
        assert canonical(a.state_summary) == canonical(b.state_summary)
        assert digest_bytes(canonical(b.state_summary).encode()) == item['historical_sha256']
        for rep in range(REPEATS):
            timings = {}
            for name, fn in ((('before',old.proposal_input),('after',current.proposal_input))
                             if (i+rep)%2==0 else (('after',current.proposal_input),('before',old.proposal_input))):
                t = time.perf_counter()
                enriched = fn(state,r,values)
                timings[name] = time.perf_counter()-t
                assert canonical(enriched.state_summary) == canonical(a.state_summary)
            rows.append(dict(input_index=i, seed=item['seed'], frame=r.trigger_frame,
                             repetition=rep, **timings))
        # Every offered COMMIT branch, null-target mode and changing-rule branch.
        branches = [(m,None,'COMMIT') for m in sorted(current.ALLOWED_MODES)]
        branches += [(m,e['target_id'],'COMMIT') for m,entries in a.state_summary['repair_candidates'].items() for e in entries]
        branches.append((state['summary']['rule_mode'],None,'FOLLOW_RULE'))
        for mode,target,policy in branches:
            c = predictor(state,values)
            args = (state['observation'], mode, current.HOLD, current.HORIZON, c.settings,
                    c.low_level_controller, state['history'], target)
            kwargs = dict(execution_policy=policy,return_trace=True)
            expected = old_execution.forecast_execution(*args,**kwargs)
            actual = execution.forecast_execution(*args,**kwargs)
            assert actual == expected, f'Forecast changed at {item["seed"]}:{r.trigger_frame}, {mode}:{target}:{policy}'
            trace_checks += 1
        assert pickle.dumps(state) == before_state, 'Input snapshot mutated'
        print(json.dumps(dict(event='preparation_checked', inputs=i+1, total=17)),flush=True)
    rankings = []
    for item in read(DIAGNOSTIC/'ranking_inputs.json'):
        state = restore_rank_state(item)
        expected = old.rank_candidates(state,request(item['bundle']['request']),item['bundle']['response'],values)
        actual = current.rank_candidates(state,request(item['bundle']['request']),item['bundle']['response'],values)
        assert actual['candidates'] == expected['candidates'] == item['result']['candidates']
        assert actual['winner_id'] == expected['winner_id'] == item['result']['winner_id']
        rankings.append(dict(seed=item['seed'],before=expected['validation_seconds'],after=actual['validation_seconds']))
    write(OUT/'paired_timings.json', rows)
    write(OUT/'summary.json', dict(at_utc=utc_now(), inputs=17,repeats=REPEATS,
        before=stats([r['before'] for r in rows]),after=stats([r['after'] for r in rows]),
        mean_saved_seconds=float(np.mean([r['before']-r['after'] for r in rows])),
        exact_historical_prompts=17,exact_full_forecast_traces=trace_checks,
        exact_rankings=rankings,snapshot_mutations=0,api_calls=0,full_delivery_measured=False,limits=p['limits']))
    after = {n:digest(ROOT/n) for n in p['source_before_sha256']}
    for name in after:
        target = OUT/'source_after'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    write(OUT/'comparison_complete.json',dict(at_utc=utc_now(),comparison_passed=True,
        source_after_sha256=after, summary_sha256=digest(OUT/'summary.json'),
        exact_full_forecast_traces=trace_checks,model_started=False,api_calls=0))


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--compare',action='store_true')
    args=parser.parse_args()
    if args.prepare==args.compare:
        parser.error('Choose --prepare before edits, or --compare after edits')
    try:
        prepare() if args.prepare else compare()
    except BaseException as exc:
        if OUT.exists():
            write(OUT/('prepare_failure.json' if args.prepare else 'compare_failure.json'),
                dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1000:]))
        raise


if __name__=='__main__':
    main()
