"""One-intervention, full-tail oracle bounds for mode/target permissions.

No live LLM, model server, real-time sleeps, or modifications to production code.
Ground-truth labels belong to the physical world; only explicitly named oracle
rollouts expose them to a counterfactual predictor. Ordinary rules see no labels.
"""
import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from control_types import ControlObservation
from detector import Detector
from dynamics import advance_dog, advance_sheep
from execution_contract import forecast_execution
from experiment_audit import utc_now
from low_level_controller import LowLevelController, rule_mode_for_alarms

MODES = ('TARGET_TRACKING', 'COHESION_PRIORITY', 'HARD_TO_GUIDE')
ARMS = ('rule', 'oracle_target', 'oracle_mode', 'oracle_joint')
HOLD = 30
BOTH = {'flock_splitting', 'stagnation'}
SOURCE_FILES = ('config.py', 'control_types.py', 'detector.py', 'dynamics.py',
    'low_level_controller.py', 'mode_forecast.py', 'execution_contract.py',
    'high_level_controller.py', 'target_handoff.py', 'experiment_audit.py',
    'environment.py', 'experiments/oracle_permissions.py',
    'tests/test_oracle_permissions.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def detector_for(cfg):
    return Detector(cfg.DANGER_DISPERSION_THRESHOLD, cfg.DANGER_MEAN_SPREAD_THRESHOLD,
                    cfg.DANGER_STAGNATION_FRAMES, cfg.DANGER_STAGNATION_THRESHOLD)


def initial_world(seed, values):
    """Physical initializer: identical RandomState draw order to SwarmEnv."""
    cfg = SimpleNamespace(**values)
    rng = np.random.RandomState(seed)
    goal = rng.uniform(cfg.GOAL_MARGIN, cfg.SPACE_SIZE - cfg.GOAL_MARGIN, 2)
    center = rng.uniform(cfg.SHEEP_START_MARGIN, cfg.SPACE_SIZE - cfg.SHEEP_START_MARGIN, 2)
    dog = rng.uniform(cfg.DOG_START_MARGIN, cfg.SPACE_SIZE - cfg.DOG_START_MARGIN, 2)
    positions = center + rng.randn(cfg.NUM_SHEEP, 2) * cfg.SHEEP_START_RADIUS
    indices = np.arange(cfg.NUM_SHEEP)
    rng.shuffle(indices)
    # PHYSICS ONLY: hidden types generate the actual motion, never rule choices.
    labels = ['Normal'] * cfg.NUM_SHEEP
    for index in indices[:cfg.NUM_ABNORMAL_A]:
        labels[int(index)] = 'A'
    assert cfg.NUM_ABNORMAL_B == cfg.NUM_ABNORMAL_C == 0
    return dict(positions=positions.tolist(), velocities=np.zeros_like(positions).tolist(),
                dog=dog.tolist(), dog_velocity=[0., 0.], goal=goal.tolist(),
                physical_types=labels)


def public_observation(world, alarms, frame, mode):
    """The rule/geometry interface deliberately has an empty type field."""
    return ControlObservation(np.asarray(world['positions']), np.asarray(world['dog']),
        np.asarray(world['goal']), tuple(alarms), frame, (), mode,
        np.asarray(world['velocities']), np.asarray(world['dog_velocity']))


def world_hash(world):
    return hashlib.sha256(json.dumps(world, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rule_run(seed, values, *, world=None, keep_trace=False):
    """Unpaced physical world. Only physical integration accesses type labels."""
    cfg = SimpleNamespace(**values)
    world = initial_world(seed, values) if world is None else copy.deepcopy(world)
    initial = copy.deepcopy(world)
    low, detector = LowLevelController(cfg), detector_for(cfg)
    first = None
    mode = 'TARGET_TRACKING'
    trace = [] if keep_trace else None
    success = False
    for frame in range(cfg.MAX_STEPS):
        positions, goal = np.asarray(world['positions']), np.asarray(world['goal'])
        alarms = tuple(detector.detect(positions, goal)[0])
        if np.all(np.linalg.norm(positions - goal, axis=1) < cfg.GOAL_RADIUS):
            success = True
            break
        if first is None and BOTH <= set(alarms):
            first = dict(frame=frame, alarms=list(alarms), current_mode=mode,
                world=copy.deepcopy(world), history=list(detector.history_dist_to_goal),
                low=copy.deepcopy({k: v for k, v in vars(low).items() if k != 'settings'}))
        obs = public_observation(world, alarms, frame, mode)
        assert not obs.sheep_statuses
        decision = low.decide(obs)
        dog, dog_velocity = advance_dog(obs.dog_position, decision.navigation_point, cfg)
        # PHYSICS ONLY: the controller above never receives physical_types.
        positions, velocities = advance_sheep(positions, obs.sheep_velocities,
            np.asarray(world['physical_types']), dog, goal, cfg)
        world.update(positions=positions.tolist(), velocities=velocities.tolist(),
                     dog=dog.tolist(), dog_velocity=dog_velocity.tolist())
        mode = decision.mode
        if trace is not None:
            trace.append(dict(frame=frame, mode=mode, target=decision.target_sheep_id,
                              positions=positions.tolist(), dog=dog.tolist()))
    result = dict(seed=seed, success=bool(success), frames=frame + 1,
        interventions=0, initial_sha256=world_hash(initial), initial=initial,
        first_trigger=first, final_world=world)
    if trace is not None:
        result['trace'] = trace
    return result


def action_key(mode, target, follow_mode=False):
    return f'{mode}:{target}:mode_follows_rule={follow_mode}'


def candidate_actions(first, values):
    """Label-blind candidate generation, reusing the production top-three pool.

    Target-only branches let the rule choose the mode on every frame, including
    its existing mode-change release. Joint includes those exact branches too.
    """
    cfg = SimpleNamespace(**values)
    low = LowLevelController(cfg)
    low.__dict__.update(copy.deepcopy(first['low']))
    obs = public_observation(first['world'], first['alarms'], first['frame'], first['current_mode'])
    assert not obs.sheep_statuses
    rule_mode = rule_mode_for_alarms(obs.alarms, cfg.REPAIR_ENABLED)
    actions = [dict(id='rule', policy='FOLLOW_RULE', mode=rule_mode, target=None,
                    mode_follows_rule=False, arms=list(ARMS))]
    rejected = []

    def add(mode, target, follow_mode, arms):
        command = copy.copy(low).decide(obs, mode_override=mode, target_override=target)
        valid = bool(np.all((command.drive_point >= 0.) & (command.drive_point <= cfg.SPACE_SIZE)))
        item = dict(id=action_key(mode, target, follow_mode), policy='COMMIT', mode=mode,
                    target=target, mode_follows_rule=follow_mode, arms=arms)
        (actions if valid else rejected).append(item)

    for mode in MODES:
        add(mode, None, False, ['oracle_mode', 'oracle_joint'])
        for target in low.repair_candidates(obs, mode):
            add(mode, target, False, ['oracle_joint'])
            if mode == rule_mode:
                add(mode, target, True, ['oracle_target', 'oracle_joint'])
    assert len({a['id'] for a in actions}) == len(actions)
    return actions, rejected


def oracle_rollout(first, action, values, *, trace=False):
    """ORACLE ONLY: exact hidden-type dynamics, complete remaining frame budget.

    The existing execution contract supplies the unchanged hold, early-release,
    handback and rule-hysteresis behavior. No requests or prompt construction run.
    """
    cfg = SimpleNamespace(**dict(values,
        HIGH_LEVEL_MODE_SELECTION_ENABLED=not action['mode_follows_rule']))
    low = LowLevelController(cfg)
    low.__dict__.update(copy.deepcopy(first['low']))
    world = first['world']
    obs = public_observation(world, first['alarms'], first['frame'], first['current_mode'])
    # ORACLE PRIVILEGE: hidden labels are supplied only to this counterfactual.
    from dataclasses import replace
    obs = replace(obs, sheep_statuses=tuple(world['physical_types']))
    horizon = cfg.MAX_STEPS - first['frame']
    f = forecast_execution(obs, action['mode'], HOLD, horizon, cfg, low,
        first['history'], action['target'], execution_policy=action['policy'],
        trigger_alarms=first['alarms'], return_trace=trace)
    raw_finish = first['frame'] + f['simulated_frames'] + 1
    # SwarmEnv tests success BEFORE movement. Entering on the final movement
    # would require observation MAX_STEPS+1 and is not counted as success.
    success = bool(f['predicted_success'] and raw_finish <= cfg.MAX_STEPS)
    result = dict(action=action, success=success,
        frames=raw_finish if success else cfg.MAX_STEPS,
        interventions=int(f['committed_frames'] > 0),
        committed_frames=f['committed_frames'], forecast=f)
    return result


def branch_rank(branch):
    # Whole-task success first; successful completion time next. On failed or
    # equally fast branches, prefer no intervention, then deterministic action ID.
    return (not branch['success'], branch['frames'],
            branch['action']['id'] != 'rule', branch['action']['id'])


def formal_case(seed, values):
    started = time.perf_counter()
    baseline = rule_run(seed, values)
    first = baseline['first_trigger']
    branches, rejected = [], []
    if first is not None:
        actions, rejected = candidate_actions(first, values)
        for action in actions:
            branches.append(oracle_rollout(first, action, values))
        reference = next(b for b in branches if b['action']['id'] == 'rule')
        assert (reference['success'], reference['frames']) == (baseline['success'], baseline['frames']), 'Rule tail mismatch'
    selected = {}
    for arm in ARMS:
        if first is None:
            selected[arm] = dict(success=baseline['success'], frames=baseline['frames'],
                                 interventions=0, selected_action='rule')
        else:
            best = min((b for b in branches if arm in b['action']['arms']), key=branch_rank)
            selected[arm] = dict(success=best['success'], frames=best['frames'],
                interventions=best['interventions'], selected_action=best['action']['id'])
    for arm in ARMS[1:]:
        assert selected[arm]['success'] >= selected['rule']['success']
        assert branch_rank(dict(selected['oracle_joint'], action={'id':selected['oracle_joint']['selected_action']})) <= branch_rank(dict(selected[arm], action={'id':selected[arm]['selected_action']}))
    return dict(seed=seed, initial_sha256=baseline['initial_sha256'], initial=baseline['initial'],
        first_trigger=first, baseline=dict(success=baseline['success'],frames=baseline['frames']),
        branches=branches, rejected_actions=rejected, selected=selected,
        seconds=time.perf_counter()-started)


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [center-radius, center+radius]


def exact_mcnemar(rescues, regressions):
    n = rescues + regressions
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(rescues, regressions)+1)) / 2**n) if n else 1.


def aggregate(cases):
    rows = []
    for arm in ARMS:
        results = [c['selected'][arm] for c in cases]
        successes = sum(r['success'] for r in results)
        finish = [r['frames'] for r in results if r['success']]
        rescues = [c['seed'] for c in cases if not c['selected']['rule']['success'] and c['selected'][arm]['success']]
        regressions = [c['seed'] for c in cases if c['selected']['rule']['success'] and not c['selected'][arm]['success']]
        rows.append(dict(arm=arm, scenes=len(cases), successes=successes,
            success_rate=successes/len(cases), wilson95=wilson(successes,len(cases)),
            mean_success_frames=float(np.mean(finish)) if finish else None,
            mean_capped_frames=float(np.mean([r['frames'] for r in results])),
            mean_interventions=float(np.mean([r['interventions'] for r in results])),
            rescues=len(rescues), regressions=len(regressions),
            paired_difference=(len(rescues)-len(regressions))/len(cases),
            mcnemar_exact_p=exact_mcnemar(len(rescues),len(regressions)),
            rescue_seeds=rescues, regression_seeds=regressions))
    return rows


def collect_jobs(function, jobs, workers, output, phase):
    results = []
    stage = Path(output)/phase
    stage.mkdir(exist_ok=False)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, *args):args[0] for args in jobs}
        for future in as_completed(pending):
            if (Path(output)/'stop_requested').exists():
                for other in pending:
                    other.cancel()
                raise RuntimeError('User stop requested; preserve completed results')
            record = future.result()
            save(stage/f'{pending[future]}.json', record)
            results.append(record)
            save(Path(output)/'progress.json',dict(at_utc=utc_now(),phase=phase,
                completed=len(results),total=len(jobs),workers=workers,
                seconds=time.perf_counter()-started))
    return sorted(results,key=lambda r:r['seed'])


def verify_sources(protocol):
    for name, expected in protocol['source_sha256'].items():
        assert digest(ROOT/name) == expected, f'Frozen source changed: {name}'


def prepare(output):
    """Register disjoint seeds, a finite calibration search, and frozen sources."""
    output=Path(output)
    assert not output.exists(),'Preserve prior evidence; do not overwrite'
    reference=ROOT/'logs/experiments/mpc_single_new100_10hz6_20261022/protocol.json'
    baseline=read(reference)
    used=set();scanned=0;malformed=[]
    def collect(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key.lower() in ('seed','random_seed') and type(value) is int:
                    used.add(value)
                elif key.lower().endswith('seeds') and isinstance(value,list):
                    used.update(x for x in value if type(x) is int)
                elif isinstance(value,(list,dict)):
                    collect(value)
        elif isinstance(obj,list):
            for value in obj:
                if isinstance(value,(list,dict)):
                    collect(value)
    for base in (ROOT/'logs',ROOT.parent/'shepherding_sim_archive'):
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ('.json','.csv'):
                continue
            if any(p in ('.venv','__pycache__','.git','node_modules') for p in path.parts):
                continue
            scanned+=1
            if path.suffix.lower()=='.json':
                raw=path.read_text(encoding='utf-8-sig')
                try:
                    collect(json.loads(raw))
                except ValueError:
                    used.update(int(x) for x in re.findall(r'\d+',raw))
                    malformed.append(dict(path=str(path),sha256=digest(path),method='conservatively exclude all integer substrings'))
            else:
                with path.open(encoding='utf-8-sig',newline='') as stream:
                    reader=csv.DictReader(stream)
                    keys=[k for k in reader.fieldnames or [] if k.lower() in ('seed','random_seed')]
                    for row in reader:
                        for key in keys:
                            if row.get(key):
                                used.add(int(float(row[key])))
    seeds=[];seed=40001
    while len(seeds)<300:
        if seed not in used:
            seeds.append(seed)
        seed+=1
    values=dict(baseline['config'],NUM_ABNORMAL_A=5,NUM_ABNORMAL_B=0,NUM_ABNORMAL_C=0,
                CONTROLLER_KIND='rule',WEIGHT_DOG_REPULSION_A_FACTOR=.15)
    p=dict(at_utc=utc_now(),name='A-only first-intervention oracle permission bounds',
        config=values,calibration_seeds=seeds[:100],formal_seeds=seeds[100:],
        calibration_factors=[.15,.10,.05,.02,.01,0.],
        calibration_rule='Search this registered order; stop at first 50%-70% rule success. If none, stop before formal evaluation.',
        arms=list(ARMS),max_workers=12,hold_frames=HOLD,
        oracle_definition='First simultaneous splitting/stagnation event only. Exhaustively evaluate admissible existing actions to the full remaining2000-frame budget; commit at most30 frames with existing early release, then rules forever. Choose whole-task success then completion frames; include no intervention in every arm. Joint includes both restricted executable policy sets.',
        source_sha256={n:digest(ROOT/n) for n in SOURCE_FILES},
        reference_protocol_sha256=digest(reference),
        assumptions=[
            '20 sheep:5A+15Normal; B=C=0. Counts fixed before calibration. This differs from previous5A+5B+5C+5Normal experiments.',
            'Baseline retains current alarm-driven rule repairs and goal retention; it is not an exact reproduction of every2014 paper detail.',
            'Only A response coefficient varies in difficulty calibration. No rule, spatial distribution, alarm, frame budget, or scoring retuning.',
            'Intervention opportunity is the first concurrent alarm, not every alarm frame. Later interventions are deliberately excluded.',
            'Target pools reuse each repair mode\'s production top3 eligible sheep, including held rule target. Out-of-field initial drive points are excluded as in current MPC admission.',
            'Mode-only commits the mode but delegates target choice dynamically. Target-only follows rule modes and retains original mode-change release. Joint contains both and mode+target commits.',
            'Physical engine always uses hidden types. Rule and candidate-generation observations have empty type fields. Full-tail forecasting and best-branch selection are explicitly privileged oracles.',
            'Existing motion, high-level release/handoff code, and LLM files are unmodified. Existing forecast_execution is reused without requests or input summaries.',
            'No short-window risk gate ranks candidates. Ranking is final official success then completion frames, preferring rule on ties. This intentionally differs from the earlier MPC safety-first score.',
            'Existing forecast reports entry after a final movement as predicted_success. Wrapper enforces SwarmEnv pre-movement success timing: no success after the2000th movement.',
            'Unpaced CPU simulation, noLLM latency;10Hz is only a nominal frame-to-time scale. These are privileged single-intervention bounds, not online achievable performance or learning results.',
            'Parallelism checked on12 calibration worlds at1/6/12 workers; BLAS threads fixed to1. Formal seeds excluded from speed/difficulty checks.',
            'Wilson95 intervals per success rate; exact two-sided McNemar versus baseline; exploratory p-values without multiplicity adjustment.',
            'Oracle may decline intervention; therefore no regressions are expected by construction, not evidence of a deployable safe selector.'
        ])
    output.mkdir(parents=True)
    save(output/'protocol.json',p)
    save(output/'seed_preflight.json',dict(at_utc=utc_now(),records_scanned=scanned,
        calibration_seeds=seeds[:100],formal_seeds=seeds[100:],historical_overlap=[],
        calibration_formal_overlap=[],conservative_parse=malformed))
    for name in SOURCE_FILES:
        target=output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    print(json.dumps(dict(prepared=str(output),records_scanned=scanned,
        calibration_seeds=[seeds[0],seeds[99]],formal_seeds=[seeds[100],seeds[-1]])),flush=True)


def run(output):
    output = Path(output)
    p = read(output/'protocol.json')
    assert not (output/'manager_started.json').exists(), 'Do not restart an existing batch'
    verify_sources(p)
    save(output/'manager_started.json',dict(at_utc=utc_now(),pid=os.getpid(),host=platform.node(),
        python=sys.version,numpy=np.__version__,models_started=False))
    started = time.perf_counter()
    try:
        # Performance/consistency check uses calibration seeds only, no formal outcomes.
        speed_seeds = p['calibration_seeds'][:12]
        benchmarks = []
        canonical = None
        for workers in (1,6,12):
            then = time.perf_counter()
            records = collect_jobs(rule_run,[(s,p['config']) for s in speed_seeds],
                                   workers,output,f'parallel_check_{workers}')
            fingerprints = [{k:r[k] for k in ('seed','success','frames','initial_sha256','final_world')} for r in records]
            if canonical is None:
                canonical = fingerprints
            assert fingerprints == canonical, 'Parallel execution changed physical results'
            benchmarks.append(dict(workers=workers,seconds=time.perf_counter()-then,
                                   exact_physical_results_equal=True))
        # Keep six when twelve has no measured throughput benefit.
        workers = 12 if benchmarks[2]['seconds'] < benchmarks[1]['seconds'] else 6
        save(output/'parallel_check.json',dict(benchmarks=benchmarks,workers=workers,
            thread_limits={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
        calibration = []
        for factor in p['calibration_factors']:
            values = dict(p['config'],WEIGHT_DOG_REPULSION_A_FACTOR=factor)
            records = collect_jobs(rule_run,[(s,values) for s in p['calibration_seeds']],
                workers,output,'calibration_'+str(factor).replace('.','p'))
            calibration.append(dict(factor=factor,successes=sum(r['success'] for r in records),
                scenes=len(records),success_rate=sum(r['success'] for r in records)/len(records)))
            save(output/'calibration.json',dict(rows=calibration,target_interval=[.5,.7],
                selection_rule='First factor in preregistered order reaching 50%-70%; no other parameters changed.'))
            if .5 <= calibration[-1]['success_rate'] <= .7:
                break
        chosen = calibration[-1]
        if not .5 <= chosen['success_rate'] <= .7:
            save(output/'calibration_blocked.json',dict(at_utc=utc_now(),rows=calibration,
                reason='No registered A-response factor attained 50%-70%. No formal cohort run; do not change population or rule parameters without a revised plan.'))
            return
        values = dict(p['config'],WEIGHT_DOG_REPULSION_A_FACTOR=chosen['factor'])
        save(output/'frozen_config.json',dict(config=values,calibration=chosen,hold_frames=HOLD,
            formal_seeds=p['formal_seeds'],oracle_definition=p['oracle_definition']))
        # These worlds are saved before any formal result is computed.
        worlds = [dict(seed=s,world=initial_world(s,values)) for s in p['formal_seeds']]
        save(output/'formal_scenarios.json',worlds)
        save(output/'formal_started.json',dict(at_utc=utc_now(),scenes=len(worlds),workers=workers))
        cases = collect_jobs(formal_case,[(s,values) for s in p['formal_seeds']],workers,output,'formal')
        for case,world in zip(cases,worlds):
            assert case['seed']==world['seed'] and case['initial_sha256']==world_hash(world['world'])
        summary = dict(at_utc=utc_now(),rows=aggregate(cases),calibration=chosen,
            scenes=len(cases),triggered_scenes=sum(c['first_trigger'] is not None for c in cases),
            full_tail_branches=sum(len(c['branches']) for c in cases),
            seconds=time.perf_counter()-started,definition=p['oracle_definition'],assumptions=p['assumptions'])
        save(output/'summary.json',summary)
        with (output/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            columns=['scene_id','seed','first_trigger_frame','initial_sha256']+[a+'_'+k for a in ARMS for k in ('success','frames','interventions','selected_action')]
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for index,c in enumerate(cases):
                row=dict(scene_id=index+1,seed=c['seed'],initial_sha256=c['initial_sha256'],
                         first_trigger_frame=c['first_trigger']['frame'] if c['first_trigger'] else '')
                row.update({a+'_'+k:c['selected'][a][k] for a in ARMS for k in ('success','frames','interventions','selected_action')})
                writer.writerow(row)
        verify_sources(p)
        # Replay every selected nontrivial branch; independent selection/permissions checks.
        checked = 0
        for c in cases:
            if not c['first_trigger']:
                continue
            for arm in ARMS:
                eligible=[b for b in c['branches'] if arm in b['action']['arms']]
                expected=min(eligible,key=branch_rank)
                assert expected['action']['id']==c['selected'][arm]['selected_action']
                if arm=='oracle_target' and expected['action']['id']!='rule':
                    assert expected['action']['mode_follows_rule']
                if arm=='oracle_mode':
                    assert expected['action']['target'] is None
                assert expected['committed_frames']<=HOLD
            for selected_id in set(r['selected_action'] for r in c['selected'].values()):
                original=next(b for b in c['branches'] if b['action']['id']==selected_id)
                replay=oracle_rollout(c['first_trigger'],original['action'],values)
                assert replay==original,'Selected action replay mismatch'
                checked+=1
        verify_sources(p)
        save(output/'audit.json',dict(at_utc=utc_now(),passed=True,paired_initial_worlds=len(cases),
            baseline_tails_reproduced=sum(c['first_trigger'] is not None for c in cases),
            selected_branches_replayed=checked,permission_sets_nested=True,hold_bound_checked=True,
            labels_absent_from_rule_observations=True,no_model_calls=True,frozen_sources_unchanged=True))
        evidence=('protocol.json','frozen_config.json','formal_scenarios.json','calibration.json',
                  'parallel_check.json','summary.json','paired_results.csv','audit.json')
        save(output/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,
            scenes=len(cases),models_started=False,evidence_sha256={n:digest(output/n) for n in evidence}))
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        save(output/'failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path,help='Existing registered evidence directory; refuses restart')
    group.add_argument('--prepare',type=Path,help='New local registration directory with frozen sources')
    args=parser.parse_args()
    prepare(args.prepare) if args.prepare else run(args.run)


if __name__=='__main__':
    main()
