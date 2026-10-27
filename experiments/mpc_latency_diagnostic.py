"""Replay saved MPC inputs: single/pair requests and fresh/reused clients only."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import config
from control_types import ControlObservation
from experiment_audit import utc_now, source_hashes
from high_level_controller import HighLevelRequest
from low_level_controller import rule_mode_for_alarms
from mpc_controller import (ROLES, PROPOSAL_INSTRUCTIONS, VERSION, rank_candidates)
from realtime_controller import Hold30VLLM, public_settings
from tools.spark.benchmark_session import ModelSession, MODEL, write
from experiments.qwen_parallel_timing import metric_values

REFERENCE = ROOT/'logs/experiments/mpc_pilot_10hz_20261020_v2'
OUT = ROOT/'logs/experiments/mpc_latency_diagnostic_20261021_v2'
PORT = 18038
BASE = f'http://127.0.0.1:{PORT}/v1'
CONDITIONS = ('single_fresh', 'single_reuse', 'pair_fresh', 'pair_reuse')
# Balanced four-condition orders; the seventeenth input adds one partial block.
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
SCRIPT = 'experiments/mpc_latency_diagnostic.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def request(record):
    return HighLevelRequest(**dict(record, alarms=tuple(record['alarms'])))


def sources():
    return dict(source_hashes(ROOT), **{n:digest(ROOT/n) for n in (
        SCRIPT, 'mpc_controller.py', 'realtime_controller.py',
        'realtime_joint_controller.py', 'tools/spark/benchmark_session.py')})


def inputs():
    records, files, ranked = [], {}, []
    assert read(REFERENCE/'completion.json')['core_audit_passed']
    for path in sorted((REFERENCE/'mpc').glob('*/run.requests.jsonl')):
        files[str(path.relative_to(ROOT))] = digest(path)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        bundles = {}
        for e in events:
            if e['event'] == 'selection_ready' or (e['event'] == 'discarded_background_result'
                                                   and e.get('stage') == 'selection'):
                result = e['result']
                item = dict(seed=int(path.parent.name), request=result['request'],
                    input_sha256=hashlib.sha256(canonical(result['request']['state_summary']).encode()).hexdigest(),
                    historical_preparation_seconds=result['prediction_seconds'],
                    historical_pair_seconds=result['pair_seconds'])
                records.append(item)
                bundles[e['request_id']] = result
        for e in events:
            if e['event'] == 'mpc_ranked':
                ranked.append(dict(seed=int(path.parent.name), result=e['result'],
                                   bundle=bundles[e['request_id']]))
    assert len(records) == 17 and len(ranked) == 3
    return records, files, ranked


def prepare():
    assert not OUT.exists(), 'Existing diagnostic must not be overwritten or rerun'
    records, files, ranked = inputs()
    previous = read(REFERENCE/'protocol.json')
    current = sources()
    assert digest(ROOT/'mpc_controller.py') == previous['extra_source_sha256']['mpc_controller.py']
    for name, expected in previous['source_sha256'].items():
        assert digest(ROOT/name) == expected, f'Control source changed: {name}'
    schedule = [dict(input_index=i, condition=CONDITIONS[j])
                for i in range(len(records)) for j in ORDERS[i % 4]]
    OUT.mkdir(parents=True)
    write(OUT/'inputs.json', records)
    write(OUT/'ranking_inputs.json', ranked)
    initial_attempt = ROOT/'logs/experiments/mpc_latency_diagnostic_20261021'
    assert read(initial_attempt/'cleanup.json')['state_restored']
    assert not list(initial_attempt.glob('warmup*.json')) and not (initial_attempt/'rounds').exists()
    values = {**public_settings(config), **previous['config'], 'VLLM_BASE_URL':BASE, 'VLLM_MODEL':MODEL}
    write(OUT/'protocol.json', dict(at_utc=utc_now(), source_sha256=current,
        historical_files_sha256=files, reference_protocol_sha256=digest(REFERENCE/'protocol.json'),
        inputs_sha256=digest(OUT/'inputs.json'), ranking_inputs_sha256=digest(OUT/'ranking_inputs.json'),
        model=MODEL, model_instances=1, config=values,
        preparation_correction=dict(previous=str(initial_attempt),
            previous_failure_sha256=digest(initial_attempt/'execution_failure.json'),
            reason='Historical config stores overrides only; merge public defaults for VLLM_CACHE_PATH before constructing clients.',
            previous_api_calls=0, previous_service_state_restored=True, no_control_policy_change=True),
        conditions=list(CONDITIONS), schedule=schedule, formal_rounds=68, formal_api_calls=102,
        warmup='One fresh pair then one reused pair on input0; four calls, excluded but retained.',
        single_role=ROLES[0][0], pair_roles=[r for r,g in ROLES],
        request_timeout_seconds=8., temperature=0, client_retries=0,
        server_recipe_unchanged=True, production_control_unchanged=True,
        simulations=0, full_delivery_measured=False,
        limits=['Open-loop fixed-input latency, not closed-loop control efficacy or actual delivery.',
            'Warm prefix cache is shared, not reset. Counterbalanced order reduces but does not eliminate cache effects.',
            'Seventeen rounds per condition: P95 is a small-sample tail estimate.',
            'Single uses the same progress prompt as one member of the pair; no prompt compression.',
            'Server queue/prefill/decode are per-round aggregate metric deltas, not per-request traces.',
            'Historical preparation times are reused; saved three ranking snapshots are replayed separately.',
            'No modified cancellation, ranking, horizon, hold, model, service batching, or controller policy.']))
    for name in current:
        target = OUT/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return read(OUT/'protocol.json')


def check_sources(p):
    assert sources() == p['source_sha256'], 'Frozen sources changed'
    assert digest(OUT/'inputs.json') == p['inputs_sha256']
    assert digest(OUT/'ranking_inputs.json') == p['ranking_inputs_sha256']


class MeasuredBackend(Hold30VLLM):
    def _create_response(self, options):
        # Same non-streaming production request; instrument without changing payload.
        self.wire_sha256 = hashlib.sha256(canonical(options).encode()).hexdigest()
        started = time.perf_counter()
        response = super()._create_response(options)
        self.create_seconds = time.perf_counter()-started
        usage = self._field(response, 'usage')
        details = self._field(usage, 'prompt_tokens_details')
        self.cached_prompt_tokens = self._field(details, 'cached_tokens')
        return response


def new_backend(values, role, guidance):
    started = time.perf_counter()
    settings = SimpleNamespace(**dict(values, LLM_API_TIMEOUT_SECONDS=8., LLM_CACHE_ENABLED=False))
    backend = MeasuredBackend(settings=settings, base_url=BASE, model=MODEL,
        compact_output=False, prefix_priming=False, input_format='json')
    backend.instructions += PROPOSAL_INSTRUCTIONS+'\nProposal preference: '+guidance
    backend.prompt_version += '-'+VERSION+'-'+role
    assert backend.model == MODEL and backend.base_url == BASE and not backend.cache_enabled
    return backend, time.perf_counter()-started


def close(backend):
    backend.close()
    backend.client.close()


def call_one(req, values, role, guidance, backend, barrier, submitted):
    entered = time.perf_counter()
    fresh = backend is None
    initialized = 0.
    if fresh:
        backend, initialized = new_backend(values, role, guidance)
    try:
        waiting = time.perf_counter()
        barrier.wait(timeout=15)
        released = time.perf_counter()
        response = backend._call_api(req, 'diagnostic-no-cache')
        ready = time.perf_counter()
        record = dict(role=role, client_initialization_seconds=initialized,
            thread_start_offset_seconds=entered-submitted, barrier_wait_seconds=released-waiting,
            call_start_offset_seconds=released-submitted, ready_offset_seconds=ready-submitted,
            client_call_seconds=ready-released, create_response_seconds=getattr(backend, 'create_seconds', None),
            cached_prompt_tokens=getattr(backend, 'cached_prompt_tokens', None),
            wire_sha256=getattr(backend, 'wire_sha256', None), response=response,
            instructions_sha256=hashlib.sha256(backend.instructions.encode()).hexdigest())
        assert not response.get('backend_error'), response.get('backend_error')
        assert response['served_model'] == MODEL and response['hold_frames'] == 30
        assert not response.get('cache_hit')
        return record
    finally:
        if fresh:
            close(backend)


def server_metrics():
    with urlopen(BASE.removesuffix('/v1')+'/metrics', timeout=5) as response:
        return response.read().decode()


def run_round(condition, req, values, reused, pool):
    roles = ROLES if condition.startswith('pair_') else ROLES[:1]
    barrier = threading.Barrier(len(roles))
    submitted = time.perf_counter()
    futures = [pool.submit(call_one, req, values, role, guidance,
        reused[role] if condition.endswith('_reuse') else None, barrier, submitted)
        for role,guidance in roles]
    # All futures must finish even after a failure; never retry a request.
    responses, errors = [], []
    for f in futures:
        try:
            responses.append(f.result())
        except Exception as exc:
            errors.append(str(exc))
    assert not errors, str(errors)
    finished = time.perf_counter()
    return dict(condition=condition, seconds=finished-submitted,
        answers_ready_seconds=max(r['ready_offset_seconds'] for r in responses),
        calls=responses, measured_concurrent_calls=len(roles),
        call_start_skew_seconds=max(r['call_start_offset_seconds'] for r in responses)
            -min(r['call_start_offset_seconds'] for r in responses))


def ranking_replays(values):
    results = []
    for item in read(OUT/'ranking_inputs.json'):
        s = item['result']['prediction_state']
        obs = ControlObservation(np.array(s['positions']), np.array(s['dog']), np.array(s['goal']),
            tuple(s['alarms']), s['frame'], tuple(s['statuses']), s['current_mode'],
            np.array(s['velocities']), np.array(s['dog_velocity']))
        state = dict(observation=obs, low=s['low'], history=s['history'],
                     summary=dict(rule_mode=rule_mode_for_alarms(s['alarms'])))
        r = rank_candidates(state, request(item['bundle']['request']), item['bundle']['response'], values)
        assert r['winner_id'] == item['result']['winner_id']
        assert r['candidates'] == item['result']['candidates']
        results.append(dict(seed=item['seed'], frame=s['frame'],
            seconds=r['validation_seconds'], distinct_candidates=len(r['candidates']),
            ranking_and_scores_reproduced=True))
    write(OUT/'ranking_replays.json', results)


def measure(p):
    values, reused = p['config'], {}
    records = read(OUT/'inputs.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            for role,guidance in ROLES:
                reused[role], seconds = new_backend(values, role, guidance)
                write(OUT/(role+'_reused_initialization.json'), dict(seconds=seconds))
            for i,c in enumerate(('pair_fresh', 'pair_reuse')):
                before = server_metrics()
                r = run_round(c, request(records[0]['request']), values, reused, pool)
                r['at_utc'] = utc_now()
                write(OUT/f'warmup{i}.json', r)
                (OUT/f'warmup{i}_before.txt').write_text(before)
                (OUT/f'warmup{i}_after.txt').write_text(server_metrics())
            for i,entry in enumerate(p['schedule']):
                check_sources(p)
                assert not (OUT/'stop_requested').exists(), 'Stop requested'
                folder = OUT/'rounds'/f'{i:03d}'
                folder.mkdir(parents=True)
                item = records[entry['input_index']]
                (folder/'metrics_before.txt').write_text(server_metrics())
                r = run_round(entry['condition'], request(item['request']), values, reused, pool)
                r.update(at_utc=utc_now(), **entry, input_sha256=item['input_sha256'])
                write(folder/'result.json', r)
                (folder/'metrics_after.txt').write_text(server_metrics())
                write(OUT/'progress.json', dict(at_utc=utc_now(), completed_rounds=i+1,
                    planned_rounds=68, completed_api_calls=sum(1 if e['condition'].startswith('single') else 2
                                                            for e in p['schedule'][:i+1])))
                if (i+1) % 4 == 0:
                    print(json.dumps(dict(event='input_block_completed', rounds=i+1, planned=68)), flush=True)
            ranking_replays(values)
        finally:
            for backend in reused.values():
                close(backend)


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(max(values)))


def audit(p):
    check_sources(p)
    assert read(OUT/'cleanup.json')['state_restored'] and not read(OUT/'cleanup.json')['fp8_started']
    assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
    assert not list(OUT.glob('*failure.json')) and not (OUT/'stop_requested').exists()
    for name,expected in p['historical_files_sha256'].items():
        assert digest(ROOT/name) == expected
    rows = [read(path) for path in sorted((OUT/'rounds').glob('*/result.json'))]
    assert len(rows) == 68 and sum(len(r['calls']) for r in rows) == 102
    wires, ids, summaries = {}, set(), []
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        assert sorted(r['input_index'] for r in selected) == list(range(17))
        for r in selected:
            assert len(r['calls']) == (2 if condition.startswith('pair') else 1)
            intervals = [(c['call_start_offset_seconds'], c['ready_offset_seconds']) for c in r['calls']]
            if len(intervals) == 2:
                assert max(a for a,b in intervals) < min(b for a,b in intervals), 'Calls did not overlap'
            for c in r['calls']:
                key = (r['input_index'], c['role'])
                assert wires.setdefault(key, c['wire_sha256']) == c['wire_sha256'], 'Payload changed between conditions'
                response = c['response']
                assert response['response_id'] not in ids
                ids.add(response['response_id'])
        calls = [c for r in selected for c in r['calls']]
        metric_deltas = {}
        for i,r in enumerate(rows):
            if r['condition'] != condition:
                continue
            folder = OUT/'rounds'/f'{i:03d}'
            before = metric_values((folder/'metrics_before.txt').read_text())
            after = metric_values((folder/'metrics_after.txt').read_text())
            for key,v in after.items():
                if key.endswith(('_sum', '_count', '_total')):
                    metric_deltas[key] = metric_deltas.get(key, 0.)+v-before.get(key, 0.)
        summaries.append(dict(condition=condition, rounds=17, calls=len(calls),
            answers_ready_seconds=stats([r['answers_ready_seconds'] for r in selected]),
            full_round_seconds=stats([r['seconds'] for r in selected]),
            api_seconds=stats([c['response']['api_seconds'] for c in calls]),
            client_initialization_seconds=stats([c['client_initialization_seconds'] for c in calls]),
            input_tokens=stats([c['response']['input_tokens'] for c in calls]),
            output_tokens=stats([c['response']['output_tokens'] for c in calls]),
            start_skew_seconds=stats([r['call_start_skew_seconds'] for r in selected]),
            server_metric_deltas=metric_deltas))
    paired = []
    for a,b in (('single_fresh','single_reuse'), ('pair_fresh','pair_reuse'),
                ('single_reuse','pair_reuse'), ('single_fresh','pair_fresh')):
        av = {r['input_index']:r['answers_ready_seconds'] for r in rows if r['condition']==a}
        bv = {r['input_index']:r['answers_ready_seconds'] for r in rows if r['condition']==b}
        diffs = [bv[i]-av[i] for i in range(17)]
        paired.append(dict(reference=a, treatment=b, treatment_minus_reference_mean_seconds=float(np.mean(diffs)),
                           treatment_faster_inputs=sum(v<0 for v in diffs), inputs=17))
    write(OUT/'summary.json', dict(at_utc=utc_now(), conditions=summaries, paired=paired,
        historical_preparation_seconds=stats([x['historical_preparation_seconds'] for x in read(OUT/'inputs.json')]),
        ranking_replays=read(OUT/'ranking_replays.json'), full_delivery_measured=False, limits=p['limits']))
    write(OUT/'completion.json', dict(at_utc=utc_now(), complete=True, core_audit_passed=True,
        formal_rounds=68, formal_api_calls=102, warmup_api_calls=4, simulations=0, paid_calls=0,
        service_state_restored=True, identical_payloads_verified=True, concurrent_pairs_verified=True,
        evidence_sha256={n:digest(OUT/n) for n in ('protocol.json','inputs.json','ranking_inputs.json',
            'summary.json','cleanup.json','model_identity.json')}))
    print(json.dumps(dict(event='audited_complete', summary=str(OUT/'summary.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Use --run to execute one new diagnostic; existing evidence is never overwritten')
    p = prepare()
    try:
        # Construct the actual adapter before loading a model; no network request.
        for role,guidance in ROLES:
            backend, _ = new_backend(p['config'], role, guidance)
            close(backend)
        write(OUT/'client_preflight.json', dict(at_utc=utc_now(), actual_adapters_constructed=2, api_calls=0))
        with ModelSession(OUT, 'mpc-latency-diagnostic-20261021-v2', PORT):
            measure(p)
        audit(p)
    except BaseException as exc:
        write(OUT/'execution_failure.json', dict(at_utc=utc_now(), type=type(exc).__name__, reason=str(exc)[-1400:]))
        raise


if __name__ == '__main__':
    main()
