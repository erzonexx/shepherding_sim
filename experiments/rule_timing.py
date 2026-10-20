"""Rule-only pacing and six-process benchmark. No model service is used."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from ctypes import wintypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import config
from detector import Detector
from environment import SwarmEnv
from experiment_audit import source_hashes, utc_now
from low_level_controller import LowLevelController
from realtime_clock import RealtimeClock

STAGES = (('30hz_serial', 30, 1), ('30hz_parallel6', 30, 6), ('10hz_parallel6', 10, 6))


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (key, ctypes.c_ulonglong) for key in (
            'total', 'available', 'total_page', 'available_page',
            'total_virtual', 'available_virtual', 'extended')]


def host_memory():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot sample Windows host memory')
    return dict(total_bytes=status.total, available_bytes=status.available)


def cpu_times():
    values = [wintypes.FILETIME() for _ in range(3)]
    if not ctypes.windll.kernel32.GetSystemTimes(*[ctypes.byref(v) for v in values]):
        raise OSError('Cannot sample Windows CPU time')
    return tuple((v.dwHighDateTime << 32) | v.dwLowDateTime for v in values)


def monitor(folder, stopped, errors):
    previous = cpu_times()
    try:
        with (folder / 'resources.jsonl').open('x', encoding='utf-8') as stream:
            while not stopped.is_set():
                current = cpu_times()
                idle, kernel, user = [x-y for x, y in zip(current, previous)]
                previous = current
                memory = host_memory()
                stream.write(json.dumps(dict(
                    at_utc=utc_now(), **memory,
                    cpu_percent=100*(1-idle/(kernel+user)) if kernel+user else None))+'\n')
                stream.flush()
                if memory['available_bytes'] < 2*2**30:
                    raise MemoryError('Available host memory below2GiB; stop without retry')
                stopped.wait(.25)
    except BaseException as exc:
        errors.append(str(exc))
        (folder.parent / 'stop_requested').touch()


def worker_ready(folder):
    write(Path(folder) / 'workers' / f'{os.getpid()}.json', dict(pid=os.getpid(), at_utc=utc_now()))


def check_source(protocol):
    actual = dict(source_hashes(ROOT), realtime_clock=digest(ROOT/'realtime_clock.py'), runner=digest(__file__))
    if actual != protocol['source_sha256']:
        raise RuntimeError('Source changed during benchmark; preserve evidence and stop')


def run_scene(output, stage, hz, seed):
    output = Path(output)
    protocol = read(output/'protocol.json')
    check_source(protocol)
    folder = output/stage/str(seed)
    folder.mkdir(exist_ok=False)
    # Each job has its own process/config state; rules use only per-frame inputs.
    for key, value in protocol['config'].items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.RANDOM_SEED = seed
    config.USE_FIXED_SEED = True
    config.CONTROLLER_KIND = 'rule'
    config.LOG_RETENTION_ENABLED = False
    controller = LowLevelController()
    env = SwarmEnv(controller)
    detector = Detector(config.DANGER_DISPERSION_THRESHOLD, config.DANGER_MEAN_SPREAD_THRESHOLD,
                        config.DANGER_STAGNATION_FRAMES, config.DANGER_STAGNATION_THRESHOLD)
    initial = dict(sheep=env.sheep_pos.tolist(), dog=env.dog_pos.tolist(), goal=env.goal_pos.tolist(),
                   statuses=env.sheep_status.tolist())
    write(folder/'started.json', dict(at_utc=utc_now(), seed=seed, hz=hz, pid=os.getpid(),
                                     controller='rule', model_calls=0, initial=initial))
    states, velocities, dogs, dog_velocities, controls = [], [], [], [], []
    initial_hash = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
    clock = RealtimeClock(hz)
    ticks = []
    reached = False
    try:
        for _ in range(protocol['max_steps']):
            if (output/'stop_requested').exists():
                raise RuntimeError('Benchmark stopped; no automatic retry')
            tick = clock.tick(env.current_frame)
            ticks.append(tick)
            alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
            env.record_frame_metrics(metrics)
            reached = bool(env.step(alarms))
            states.append(env.sheep_pos.copy())
            velocities.append(env.sheep_vel.copy())
            dogs.append(env.dog_pos.copy())
            dog_velocities.append(env.dog_vel.copy())
            controls.append(dict(mode=env.current_mode,
                                 target=None if env.target_sheep_id is None else int(env.target_sheep_id),
                                 drive_point=env.drive_point.tolist(),
                                 navigation_point=env.navigation_point.tolist()))
            clock.record_step(tick)
            if reached:
                break
    finally:
        physical_end = time.perf_counter()
        env.close()
        np.savez_compressed(folder/'clock.npz', tick=np.array(ticks), step=np.array(clock.step_seconds),
                            lateness=np.array(clock.lateness), origin=np.array(clock.start))
    arrays = dict(sheep=np.array(states), sheep_velocity=np.array(velocities), dog=np.array(dogs),
                  dog_velocity=np.array(dog_velocities))
    np.savez_compressed(folder/'trajectory.npz', **arrays)
    write(folder/'controls.json', controls)
    trace = hashlib.sha256()
    for name in sorted(arrays):
        trace.update(arrays[name].tobytes())
    trace.update(json.dumps(controls, sort_keys=True).encode())
    row = dict(seed=seed, stage=stage, hz=hz, frames=env.current_frame, success=reached,
               physical_seconds=physical_end-clock.start,
               origin=clock.start, physical_end=physical_end,
               step_overruns=sum(x>clock.period for x in clock.step_seconds),
               late_ticks_over_period=sum(x>clock.period for x in clock.lateness),
               max_step_seconds=max(clock.step_seconds), max_lateness_seconds=max(clock.lateness),
               initial_sha256=initial_hash, trajectory_sha256=trace.hexdigest(), model_calls=0)
    write(folder/'completed.json', row)
    print(json.dumps(dict(event='scene_complete', stage=stage, seed=seed, frames=row['frames'])), flush=True)
    return row


def prepare(output, reference, seeds):
    if output.exists():
        raise FileExistsError('Never overwrite a completed or partial benchmark')
    if os.name != 'nt':
        raise RuntimeError('This benchmark uses Windows host resource sampling')
    previous = read(reference)
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError('Provide exactly six distinct seeds')
    output.mkdir(parents=True)
    sources = dict(source_hashes(ROOT), realtime_clock=digest(ROOT/'realtime_clock.py'), runner=digest(__file__))
    write(output/'protocol.json', dict(
        at_utc=utc_now(), purpose='Rule-only pacing and six-process capability test; not new efficacy validation',
        seeds=seeds, stages=[dict(name=s, hz=h, workers=n) for s, h, n in STAGES], simulations=18,
        config=previous['config'], max_steps=2000, source_sha256=sources,
        reference=str(reference), reference_sha256=digest(reference), model_calls=0,
        motion='Per-frame motion, alarms, target holds and all control parameters unchanged; only wall-clock pacing differs.',
        order='One fixed pass:30Hz serial,30Hz six-way,10Hz six-way; no outcome-based selection or retry.',
        limits=['Six historical scenarios; not a success-rate generalization experiment.',
                '10Hz slows the entire simulated environment, not only the controller updates.',
                'Windows soft realtime; all deadline lateness recorded.',
                'Resource samples every0.25s; stage timing excludes initial process import/warmup.',
                'No claim about LLM latency or real-robot speed.']))
    write(output/'host.json', dict(at_utc=utc_now(), cpu_logical=os.cpu_count(), **host_memory()))
    snapshot = output/'source'
    for name in sources:
        source = Path(__file__) if name == 'runner' else ROOT/('realtime_clock.py' if name == 'realtime_clock' else name)
        target = snapshot/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def run(output):
    protocol = read(output/'protocol.json')
    for stage, hz, workers in STAGES:
        check_source(protocol)
        folder = output/stage
        folder.mkdir(exist_ok=False)
        (folder/'workers').mkdir()
        rows = []
        stopped, errors = threading.Event(), []
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_ready, initargs=(str(folder),)) as pool:
            # Launch all processes before the measured stage starts.
            warmed = [pool.submit(time.monotonic) for _ in range(workers)]
            for future in warmed:
                future.result(timeout=60)
            deadline = time.monotonic()+60
            while len(list((folder/'workers').glob('*.json'))) < workers:
                if time.monotonic()>deadline:
                    raise TimeoutError('Worker startup timed out')
                time.sleep(.05)
            thread = threading.Thread(target=monitor, args=(folder, stopped, errors))
            thread.start()
            started = time.perf_counter()
            print(json.dumps(dict(event='stage_started', stage=stage, hz=hz, workers=workers)), flush=True)
            try:
                jobs = [pool.submit(run_scene, str(output), stage, hz, seed) for seed in protocol['seeds']]
                for future in as_completed(jobs):
                    rows.append(future.result())
            finally:
                elapsed = time.perf_counter()-started
                stopped.set()
                thread.join()
            if errors:
                raise RuntimeError(errors[0])
        write(folder/'stage_complete.json', dict(at_utc=utc_now(), seconds=elapsed, rows=rows))
        print(json.dumps(dict(event='stage_complete', stage=stage, seconds=elapsed)), flush=True)
    audit(output)


def audit(output):
    protocol = read(output/'protocol.json')
    check_source(protocol)
    baseline, summaries = {}, []
    if (output/'stop_requested').exists():
        raise RuntimeError('Stopped benchmark cannot pass audit')
    for stage, hz, workers in STAGES:
        record = read(output/stage/'stage_complete.json')
        rows = sorted(record['rows'], key=lambda r: r['seed'])
        if [r['seed'] for r in rows] != sorted(protocol['seeds']):
            raise AssertionError('Incomplete stage')
        clock_totals = dict(ticks=0, step_overruns=0, late_ticks_over_period=0)
        for row in rows:
            folder = output/stage/str(row['seed'])
            t = np.load(folder/'clock.npz')
            n = row['frames']
            assert len(t['tick']) == len(t['step']) == len(t['lateness']) == n
            assert np.all(t['tick'] >= float(t['origin'])+np.arange(n)/hz-.001)
            assert np.allclose(t['lateness'], np.maximum(0, t['tick']-float(t['origin'])-np.arange(n)/hz), atol=1e-6)
            assert row['physical_seconds'] >= (n-1)/hz-.001
            assert row['model_calls'] == 0
            clock_totals['ticks'] += n
            clock_totals['step_overruns'] += row['step_overruns']
            clock_totals['late_ticks_over_period'] += row['late_ticks_over_period']
            if stage == STAGES[0][0]:
                baseline[row['seed']] = row
            else:
                old = baseline[row['seed']]
                assert all(old[k] == row[k] for k in ('initial_sha256', 'trajectory_sha256', 'frames', 'success'))
                with np.load(output/STAGES[0][0]/str(row['seed'])/'trajectory.npz') as a, np.load(folder/'trajectory.npz') as b:
                    assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
                assert read(folder/'controls.json') == read(output/STAGES[0][0]/str(row['seed'])/'controls.json')
        resources = [json.loads(x) for x in (output/stage/'resources.jsonl').read_text().splitlines()]
        activity = sorted([(r['origin'], 1) for r in rows]+[(r['physical_end'], -1) for r in rows])
        active = peak = 0
        for _, delta in activity:
            active += delta
            peak = max(peak, active)
        assert peak == workers
        summaries.append(dict(stage=stage, hz=hz, workers=workers, runs=6,
            successes=sum(r['success'] for r in rows), mean_capped_frames=sum(r['frames'] for r in rows)/6,
            batch_seconds=record['seconds'], sum_scene_physical_seconds=sum(r['physical_seconds'] for r in rows),
            peak_simultaneous_scenes=peak, **clock_totals,
            max_step_seconds=max(r['max_step_seconds'] for r in rows),
            max_lateness_seconds=max(r['max_lateness_seconds'] for r in rows),
            min_available_gib=min(r['available_bytes'] for r in resources)/2**30,
            cpu_mean=float(np.mean([r['cpu_percent'] for r in resources if r['cpu_percent'] is not None]))))
    write(output/'summary.json', dict(at_utc=utc_now(), stages=summaries, all_trajectories_identical=True,
                                    model_calls=0, paid_calls=0))
    write(output/'completion.json', dict(at_utc=utc_now(), complete=True, simulations=18,
        core_audit_passed=True, all_trajectories_identical=True, model_calls=0,
        evidence_sha256={n:digest(output/n) for n in ('protocol.json', 'summary.json')}))
    print(json.dumps(dict(event='audited_complete', stages=summaries)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Prepare and execute one new benchmark; never resumes old output.')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/experiments/rule_timing_10hz6_20261020')
    parser.add_argument('--reference', type=Path, default=ROOT/'logs/experiments/realtime_joint_new100_30hz_20261019/protocol.json')
    parser.add_argument('--seeds', nargs=6, type=int, default=list(range(10001,10007)))
    args = parser.parse_args()
    if not args.run:
        parser.error('Use --run to start the rule-only benchmark')
    prepare(args.output, args.reference, args.seeds)
    try:
        run(args.output)
    except BaseException as exc:
        (args.output/'stop_requested').touch()
        write(args.output/'execution_failure.json', dict(at_utc=utc_now(), type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == '__main__':
    main()
