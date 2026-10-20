"""Measure joint Qwen decisions at10Hz with1/2/4/6 concurrent physical scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from experiment_audit import utc_now
from experiments import rule_timing
from tools import run_realtime_joint_comparison as joint
from tools.spark.benchmark_session import ModelSession, MODEL

OUT = ROOT/'logs/experiments/qwen_parallel_10hz_20261020'
REFERENCE = ROOT/'logs/experiments/realtime_joint_new100_30hz_20261019/protocol.json'
BASE = 'http://127.0.0.1:18036/v1'
WORKERS = (1, 2, 4, 6)
SEEDS = list(range(10001,10007))
BENCHMARK = ('experiments/qwen_parallel_timing.py','experiments/rule_timing.py','tools/spark/benchmark_session.py')
read, save, digest = joint.read, joint.save, joint.digest


def configure(workers=None):
    joint.OUT = OUT if workers is None else OUT/f'workers{workers}'
    joint.BASE, joint.ARMS = BASE, ('qwen',)
    joint.configure()


def protocol():
    p = read(OUT/'protocol.json')
    joint.check_frozen(p)
    assert {name:digest(ROOT/name) for name in BENCHMARK} == p['benchmark_sha256']
    return p


def prepare():
    assert not OUT.exists(), 'Preserve existing benchmark; no restart or overwrite'
    previous = read(REFERENCE)
    p = dict(previous)
    p.update(at_utc=utc_now(), name='10Hz joint Qwen concurrency latency benchmark',
        seeds=SEEDS, workers=list(WORKERS), simulations=24, arms=['qwen'], hz=10,
        period_seconds=.1, source_sha256=joint.hashes(),
        benchmark_sha256={name:digest(ROOT/name) for name in BENCHMARK},
        config=dict(previous['config'], VLLM_BASE_URL=BASE, CONTROLLER_KIND='vllm'),
        validation_snapshot_max_age_seconds=3.,
        pipeline='Original joint mode-target policy, asynchronous measured-time forecasts/API/revalidation; no MPC.',
        execution='Fixed1/2/4/6 scene workers, same six historical seeds,2000frames each, actual10Hz.',
        model_instances=1, serving_recipe_unchanged=True,
        reference=str(REFERENCE), reference_sha256=digest(REFERENCE),
        limits=['Performance benchmark on six historical scenes, not independent efficacy evidence.',
            'Scene concurrency is not API concurrency; alarms determine actual request overlap.',
            'Single fixed1/2/4/6 pass; shared prefix cache and OS load may affect results.',
            'Model recipe remains max-num-seqs4; it is not increased for six scene workers.',
            'Per-frame policies unchanged:30-frame hold is up to3seconds at10Hz.',
            'API means include completed canceled/discarded replies; delivered results form a different subset.',
            'Server metrics and CPU/memory samples are diagnostics, not hard realtime guarantees.'])
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json', p)
    for n in WORKERS:
        (OUT/f'workers{n}'/'workers').mkdir(parents=True)
    for name in [*p['source_sha256'], *BENCHMARK]:
        target = OUT/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)


def worker_init(n):
    configure(n)
    save(OUT/f'workers{n}'/'workers'/f'{os.getpid()}.json', dict(pid=os.getpid()))


def stop_all():
    (OUT/'stop_requested').touch()
    for n in WORKERS:
        (OUT/f'workers{n}'/'stop_requested').touch()


def run_case(n, seed):
    configure(n)
    p = protocol()
    assert not (OUT/f'workers{n}'/'qwen'/str(seed)).exists()
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
    try:
        row = joint.runner.run_one(p, seed, 'qwen', executor=pool)
        print(json.dumps(dict(event='scene_finished', workers=n, seed=seed, frames=row['Frames'])), flush=True)
        return row
    except BaseException as exc:
        stop_all()
        save(OUT/f'case_failure_{n}_{seed}.json', dict(type=type(exc).__name__, reason=str(exc)[-1200:]))
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def metrics_text():
    with urlopen(BASE.removesuffix('/v1')+'/metrics', timeout=5) as response:
        return response.read().decode()


def metric_values(text):
    values = {}
    for line in text.splitlines():
        match = re.match(r'^([\w:]+)(?:\{.*\})?\s+([0-9.eE+-]+)(?:\s|$)', line)
        if match:
            key, value = match.groups()
            values[key] = values.get(key, 0.)+float(value)
    return values


def model_monitor(folder, stopped, errors):
    try:
        with (folder/'server_activity.jsonl').open('x', encoding='utf-8') as stream:
            while not stopped.is_set():
                values = metric_values(metrics_text())
                stream.write(json.dumps(dict(at_utc=utc_now(), running=values.get('vllm:num_requests_running'),
                                             waiting=values.get('vllm:num_requests_waiting')))+'\n')
                stream.flush()
                stopped.wait(1.)
    except BaseException as exc:
        errors.append('Server metrics sampling failed: '+str(exc))
        stop_all()


def host_monitor(folder, stopped, errors):
    rule_timing.monitor(folder, stopped, errors)
    if errors:
        stop_all()


def run_stages():
    p = protocol()
    progress = dict(at_utc=utc_now(), completed=0, planned=24)
    for n in WORKERS:
        protocol()
        folder = OUT/f'workers{n}'
        errors, stopped = [], threading.Event()
        with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_init, initargs=(n,)) as pool:
            starters = [pool.submit(time.monotonic) for _ in range(n)]
            for f in starters:
                f.result(timeout=60)
            deadline = time.monotonic()+60
            while len(list((folder/'workers').glob('*.json'))) < n:
                assert time.monotonic()<deadline, 'Scene worker startup timed out'
                time.sleep(.05)
            (folder/'metrics_before.txt').write_text(metrics_text(), encoding='utf-8')
            threads = [threading.Thread(target=fn, args=(folder, stopped, errors))
                       for fn in (host_monitor, model_monitor)]
            for t in threads:
                t.start()
            started = time.perf_counter()
            save(folder/'started.json', dict(at_utc=utc_now(), workers=n, hz=10))
            print(json.dumps(dict(event='stage_started', workers=n, hz=10)), flush=True)
            rows = []
            try:
                futures = [pool.submit(run_case, n, s) for s in p['seeds']]
                for f in as_completed(futures):
                    rows.append(f.result())
                    progress.update(at_utc=utc_now(), completed=progress['completed']+1,
                                    current_workers=n, current_stage_completed=len(rows))
                    temporary = OUT/'progress.tmp'
                    temporary.write_text(json.dumps(progress), encoding='utf-8')
                    temporary.replace(OUT/'progress.json')
            finally:
                elapsed = time.perf_counter()-started
                stopped.set()
                for t in threads:
                    t.join()
            assert not errors and not (OUT/'stop_requested').exists(), str(errors)
            (folder/'metrics_after.txt').write_text(metrics_text(), encoding='utf-8')
        save(folder/'stage_complete.json', dict(at_utc=utc_now(), seconds=elapsed, rows=rows))
        print(json.dumps(dict(event='stage_complete', workers=n, seconds=elapsed)), flush=True)
    save(OUT/'runs_complete.json', dict(at_utc=utc_now(), simulations=24))


def audit():
    p = protocol()
    assert read(OUT/'runs_complete.json')['simulations']==24
    assert not list(OUT.glob('*failure*.json')) and not (OUT/'stop_requested').exists()
    assert read(OUT/'cleanup.json')['state_restored'] and not read(OUT/'cleanup.json')['fp8_started']
    assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
    baseline, records = {}, []
    for n in WORKERS:
        stage = read(OUT/f'workers{n}'/'stage_complete.json')
        assert sorted(r['Seed'] for r in stage['rows'])==SEEDS
        api, predictions, deliveries, validation, ages, intervals = [], [], [], [], [], []
        late = over = ticks_total = canceled_count = 0
        for row in stage['rows']:
            meta = read(row['RunMetadata'])
            assert meta['status']=='completed' and meta['control_hz']==10
            assert meta['config']['HIGH_LEVEL_MODE_SELECTION_ENABLED'] and meta['config']['HIGH_LEVEL_TARGET_SELECTION_ENABLED']
            assert not meta['config']['HIGH_LEVEL_TARGET_HANDOFF_ENABLED'] and not meta['config']['HIGH_LEVEL_EXPLICIT_EXECUTION']
            assert meta['source_sha256']=={key:p['source_sha256'][key] for key in meta['source_sha256']}
            data, control = pd.read_parquet(row['Data']), pd.read_parquet(row['Control'])
            initial = data.loc[data.Frame==0].reset_index(drop=True)
            if n==1:
                baseline[row['Seed']] = initial
            else:
                pd.testing.assert_frame_equal(initial, baseline[row['Seed']], check_exact=True)
            assert control.Frame.tolist()==list(range(row['Frames']-int(row['Success'])))
            clock = np.load(row['Timing'])
            ticks, steps, lateness = clock['tick_perf_counter'], clock['step_seconds'], clock['lateness_seconds']
            origin = float(clock['origin_perf_counter'])
            assert len(ticks)==len(steps)==len(lateness)==row['Frames']
            assert np.all(ticks>=origin+np.arange(len(ticks))*.1-.001)
            assert np.allclose(lateness, np.maximum(0,ticks-origin-np.arange(len(ticks))*.1), atol=1e-6)
            ticks_total += len(ticks)
            late += int((lateness>.1).sum())
            over += int((steps>.1).sum())
            active = control.loc[control.decision_source!='rule']
            assert set(control.decision_source)<={'rule','vllm'}
            assert not control.llm_backend_error.notna().any()
            assert active.llm_hold_frames.eq(30).all()
            assert all(x<=30 for x in active.groupby('llm_request_id').size())
            assert (active.current_mode==active.llm_selected_mode).all()
            explicit = active.loc[active.llm_selected_target_sheep_id.notna()]
            assert (explicit.target_sheep_id==explicit.llm_selected_target_sheep_id).all()
            events = [json.loads(line) for line in Path(row['RequestsAudit']).read_text().splitlines()]
            canceled = {e['request_id'] for e in events if e['event']=='pipeline_canceled'}
            canceled_count += len(canceled)
            assert not set(active.llm_request_id)&canceled
            ready = {e['request_id']:e for e in events if e['event']=='selection_ready'}
            finished = [i for i,e in enumerate(events) if e['event']=='physical_scene_finished']
            assert len(finished)==1 and not any(e['event']=='pipeline_delivered' for e in events[finished[0]+1:])
            for e in events:
                if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e['stage']=='selection'):
                    result = e['result'];response = result['response']
                    assert response['served_model']==MODEL and response['hold_frames']==30
                    assert not response.get('backend_error') and not response.get('cache_hit')
                    api.append(response['api_seconds']);predictions.append(result['prediction_seconds'])
                    intervals.append(tuple(datetime.fromisoformat(response[k]).timestamp()
                                           for k in ('api_started_at_utc','api_returned_at_utc')))
                if e['event']=='pipeline_delivered':
                    rid, frame = e['request_id'], e['frame']
                    request = ready[rid]['result']['request']
                    assert rid not in canceled and 0<=frame-e['validation_frame']<=30
                    assert frame-request['trigger_frame']<=p['response_age_limit_frames']
                    assert control.loc[(control.Frame>=request['trigger_frame'])&(control.Frame<frame),'decision_source'].eq('rule').all()
                    deliveries.append(e['end_to_end_seconds']);validation.append(e['validation_seconds'])
                    ages.append(frame-request['trigger_frame'])
        assert api, 'No model decisions exercised'
        points = sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals])
        live = peak = 0
        for _,delta in points:
            live += delta;peak = max(peak,live)
        resources = [json.loads(x) for x in (OUT/f'workers{n}'/'resources.jsonl').read_text().splitlines()]
        server = [json.loads(x) for x in (OUT/f'workers{n}'/'server_activity.jsonl').read_text().splitlines()]
        before = metric_values((OUT/f'workers{n}'/'metrics_before.txt').read_text())
        after = metric_values((OUT/f'workers{n}'/'metrics_after.txt').read_text())
        measured = {}
        for label in ('queue','prefill','decode','inference'):
            key = 'vllm:request_'+label+'_time_seconds'
            count = after.get(key+'_count',0)-before.get(key+'_count',0)
            measured[label+'_mean'] = (after[key+'_sum']-before[key+'_sum'])/count if count>0 else None
        avg = lambda values:float(np.mean(values)) if values else None
        records.append(dict(workers=n,runs=6,model_calls=len(api),api_mean=avg(api),
            api_p95=float(np.percentile(api,95)),api_max=max(api),prediction_mean=avg(predictions),
            delivery_mean=avg(deliveries),delivery_p95=float(np.percentile(deliveries,95)) if deliveries else None,
            validation_mean=avg(validation),delivery_mean_age_frames=avg(ages),deliveries=len(deliveries),
            canceled=canceled_count,peak_overlapping_api_requests=peak,
            server_sampled_running_peak=max([v['running'] for v in server if v['running'] is not None], default=None),
            server_sampled_waiting_peak=max([v['waiting'] for v in server if v['waiting'] is not None], default=None),
            server_times=measured,batch_seconds=stage['seconds'],ticks=ticks_total,
            step_overruns=over,late_ticks_over_period=late,
            max_step_seconds=max(r['MaxStepSeconds'] for r in stage['rows']),
            max_lateness_seconds=max(r['MaxLatenessSeconds'] for r in stage['rows']),
            min_available_gib=min(v['available_bytes'] for v in resources)/2**30,
            cpu_mean=avg([v['cpu_percent'] for v in resources if v['cpu_percent'] is not None]),
            successes=sum(r['Success'] for r in stage['rows'])))
    save(OUT/'summary.json', dict(at_utc=utc_now(), stages=records, limits=p['limits'], paid_calls=0))
    save(OUT/'completion.json', dict(at_utc=utc_now(), complete=True, core_audit_passed=True,
        simulations=24, control_hz=10, service_state_restored=True, paid_calls=0,
        evidence_sha256={n:digest(OUT/n) for n in ('protocol.json','summary.json','model_identity.json','cleanup.json')}))
    print(json.dumps(dict(event='audited_complete', stages=records)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Start one new24-run benchmark and restore the original model state.')
    args = parser.parse_args()
    if not args.run:
        parser.error('Use --run to execute this benchmark')
    configure();prepare()
    save(OUT/'manager_started.json', dict(at_utc=utc_now(), pid=os.getpid(), single_manager=True))
    try:
        with ModelSession(OUT, 'qwen-parallel-10hz-20261020', 18036):
            save(OUT/'formal_started.json', dict(at_utc=utc_now(), simulations=24, hz=10, workers=list(WORKERS)))
            run_stages()
        audit()
    except BaseException as exc:
        stop_all()
        save(OUT/'execution_failure.json', dict(at_utc=utc_now(), type=type(exc).__name__, reason=str(exc)[-1600:]))
        raise


if __name__ == '__main__':
    main()
