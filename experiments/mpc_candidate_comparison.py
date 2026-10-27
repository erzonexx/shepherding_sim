"""Four-arm six-scene10Hz mechanism pilot; one versus two MPC model proposals."""
import argparse
from dataclasses import asdict,is_dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from experiment_audit import utc_now
from experiments.mpc_pilot import RecordedInitialEnv, replay_ranked, events
from experiments.rule_timing import monitor as host_monitor
from experiments.qwen_parallel_timing import metric_values
from mpc_controller import (MPCBackend,MPCController,proposal_job,ROLES,VERSION,HORIZON,HOLD,
                            PAIR_DEADLINE_SECONDS,VALIDATION_MAX_AGE_FRAMES)
from realtime_controller import AsyncPredictionBackend,public_settings
from realtime_joint_controller import AsyncJointController,VERSION as JOINT_VERSION
from tools import run_realtime_joint_comparison as joint
from tools.spark.benchmark_session import ModelSession,MODEL,write as update_json
import config

OUT=ROOT/'logs/experiments/mpc_candidate_comparison_10hz6_20261021_v2'
BASE='http://127.0.0.1:18039/v1'
SEEDS=list(range(10001,10007))
ARMS=('rule','qwen','mpc_single','mpc_pair')
COUNTS={'mpc_single':1,'mpc_pair':2}
REFERENCE=ROOT/'logs/experiments/mpc_pilot_10hz_20261020_v2'
PREPARATION=ROOT/'logs/experiments/mpc_preparation_optimization_20261021'
PREFLIGHT=ROOT.parent/'shepherding_sim_archive/mpc_candidate_preflight_20261021'
EXTRA=('mpc_controller.py','experiments/mpc_candidate_comparison.py','tests/test_mpc_controller.py',
       'experiments/mpc_pilot.py','experiments/rule_timing.py','experiments/qwen_parallel_timing.py',
       'tools/spark/benchmark_session.py')
read,save,digest=joint.read,joint.save,joint.digest


def backend_factory(settings,kind,base_url,executor):
    if kind in COUNTS:
        return MPCBackend(settings,base_url,executor,proposal_count=COUNTS[kind])
    return AsyncPredictionBackend(settings,kind=kind,base_url=base_url,executor=executor)


def controller_factory(backend,settings,validation_max_age_frames):
    if isinstance(backend,MPCBackend):return MPCController(backend,settings=settings)
    return AsyncJointController(backend,settings=settings,validation_max_age_frames=validation_max_age_frames)


def configure(arm):
    joint.OUT,joint.BASE,joint.ARMS=OUT,BASE,ARMS
    joint.configure()
    joint.runner.AsyncPredictionBackend=backend_factory
    joint.runner.AsyncTargetController=controller_factory
    joint.runner.SwarmEnv=RecordedInitialEnv
    joint.runner.VERSION=VERSION if arm in COUNTS else JOINT_VERSION


def frozen():
    p=read(OUT/'protocol.json')
    joint.check_frozen(p)
    assert p['extra_source_sha256']=={n:digest(ROOT/n) for n in EXTRA},'Pilot source changed'
    return p


def prepare():
    assert not OUT.exists(),'Preserve existing pilot; no resume or overwrite'
    assert read(PREPARATION/'completion.json')['core_audit_passed']
    ref=read(REFERENCE/'protocol.json')
    values={**public_settings(config),**ref['config'],'VLLM_BASE_URL':BASE,'VLLM_MODEL':MODEL}
    p=dict(ref,at_utc=utc_now(),name='Four-arm one-versus-two MPC proposal mechanism pilot',
        config=values,arms=list(ARMS),simulations=24,seeds=SEEDS,workers_per_arm=6,hz=10,period_seconds=.1,
        source_sha256=joint.hashes(),extra_source_sha256={n:digest(ROOT/n) for n in EXTRA},
        mpc_proposal_counts=COUNTS,reference=str(REFERENCE),
        reference_protocol_sha256=digest(REFERENCE/'protocol.json'),
        optimized_preparation_completion_sha256=digest(PREPARATION/'completion.json'),
        execution='Four fresh groups in fixed order, six concurrent worlds per group. Never overlap groups.',
        seed_policy='Same six historical mechanism cases, not independent generalization.',
        pipeline='Actual10Hz; forecasts/model/revalidation all background; rule control continues.',
        preflight='Same saved state in one spawned worker, one real proposal then two; three calls, separate from formal runs.',
        limits=['Six known scenes are a mechanism test, not evidence of population-level success rate.',
            'Fixed group order, cold worker imports and shared prefix cache are not fully controlled.',
            'Primary MPC single/pair comparison changes proposal count only; progress role remains identical.',
            'Original Qwen policy differs from MPC input, admission/validation and scoring; not a one-switch MPC ablation.',
            'Actual latency and model output variation can change closed-loop trajectories.',
            'Cancelled rounds are retained; no positive LLM takeover or successful ranking is required to declare mechanism records complete.',
            'Short-horizon ranking is not proof of full-task improvement or real-robot timing.',
            'Control overruns and model concurrency are measured rather than assumed.'])
    p.pop('smoke',None)
    p.pop('preflight_gate_correction',None)
    p['mpc']=dict(ref['mpc'],proposal_counts=COUNTS,proposal_roles=[r for r,g in ROLES])
    first=ROOT/'logs/experiments/mpc_candidate_comparison_10hz6_20261021'
    assert not (first/'formal_started.json').exists() and read(first/'cleanup.json')['state_restored']
    old=read(first/'protocol.json')
    assert old['source_sha256']==p['source_sha256']
    assert old['extra_source_sha256']['mpc_controller.py']==p['extra_source_sha256']['mpc_controller.py']
    p['entry_recording_correction']=dict(previous=str(first),
        previous_failure_sha256=digest(first/'execution_failure.json'),
        reason='Serialize preflight request dataclasses; atomically replace progress snapshots instead of exclusive creation.',
        control_sources_unchanged=True,formal_scenes_previously_started=0,previous_preflight_api_calls=3)
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json',p)
    for name in {*p['source_sha256'],*EXTRA}:
        target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(OUT/'manager_started.json',dict(at_utc=utc_now(),pid=os.getpid(),single_manager=True))


def validate_model_results(row,arm):
    outputs=[]
    for e in events(row):
        if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e.get('stage')=='selection'):
            result=e['result']
            if arm in COUNTS:
                proposals=result['response']['proposals']
                assert result['backend_info']['proposal_count']==COUNTS[arm]
                assert [x['candidate_id'] for x in proposals]==[r for r,g in ROLES[:COUNTS[arm]]]
                outputs.extend(x['response'] for x in proposals)
            else:outputs.append(result['response'])
    for r in outputs:
        assert not r.get('backend_error'),r.get('backend_error')
        assert r['served_model']==MODEL and r['hold_frames']==HOLD and not r.get('cache_hit')
    return outputs


def run_case(arm,seed):
    configure(arm);p=frozen()
    assert not (OUT/arm/str(seed)).exists(),'Existing case must not be rerun'
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) if arm!='rule' else None
    try:
        row=joint.runner.run_one(p,seed,arm,executor=pool)
        if arm!='rule':validate_model_results(row,arm)
        print(json.dumps(dict(event='scene_finished',arm=arm,seed=seed,frames=row['Frames'],success=row['Success'])),flush=True)
        return row
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        save(OUT/f'failure_{arm}_{seed}.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1200:]))
        raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)


def preflight():
    p=frozen()
    with (PREPARATION/'replay_inputs.pickle').open('rb') as f:item=pickle.load(f)[0]
    results=[]
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        for count in (1,2):
            r=pool.submit(proposal_job,item['state'],item['request'],p['config'],'qwen',BASE,proposal_count=count).result(timeout=60)
            proposals=r['response']['proposals']
            assert len(proposals)==count
            for proposal in proposals:
                response=proposal['response']
                assert not response.get('backend_error'),response.get('backend_error')
                assert response['served_model']==MODEL and response['hold_frames']==HOLD
            results.append(r)
    assert results[0]['request'].state_summary==results[1]['request'].state_summary
    assert results[0]['response']['proposals'][0]['instructions_sha256']==results[1]['response']['proposals'][0]['instructions_sha256']
    save(OUT/'model_preflight.json',json_safe(dict(at_utc=utc_now(),passed=True,api_calls=3,results=results)))
    print(json.dumps(dict(event='model_preflight_passed',api_calls=3)),flush=True)


def json_safe(record):
    def default(value):
        if is_dataclass(value):return asdict(value)
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    return json.loads(json.dumps(record,default=default,allow_nan=False))


def metrics_text():
    with urlopen(BASE.removesuffix('/v1')+'/metrics',timeout=5) as r:return r.read().decode()


def model_monitor(folder,stopped,errors):
    try:
        with (folder/'server_activity.jsonl').open('x') as stream:
            while not stopped.is_set():
                m=metric_values(metrics_text())
                stream.write(json.dumps(dict(at_utc=utc_now(),running=m.get('vllm:num_requests_running'),waiting=m.get('vllm:num_requests_waiting')))+'\n')
                stream.flush();stopped.wait(1.)
    except BaseException as exc:
        errors.append(str(exc));(OUT/'stop_requested').touch()


def stages():
    done=0
    for arm in ARMS:
        frozen();folder=OUT/(arm+'_stage');folder.mkdir()
        (folder/'metrics_before.txt').write_text(metrics_text())
        stopped,errors=threading.Event(),[]
        monitors=[threading.Thread(target=f,args=(folder,stopped,errors)) for f in (host_monitor,model_monitor)]
        for t in monitors:t.start()
        started=time.perf_counter();rows=[]
        print(json.dumps(dict(event='stage_started',arm=arm,workers=6)),flush=True)
        try:
            with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
                for f in as_completed([pool.submit(run_case,arm,s) for s in SEEDS]):
                    rows.append(f.result());done+=1
                    update_json(OUT/'progress.json',dict(at_utc=utc_now(),completed=done,planned=24,arm=arm,arm_completed=len(rows)))
        finally:
            elapsed=time.perf_counter()-started;stopped.set()
            for t in monitors:t.join()
        assert not errors and not (OUT/'stop_requested').exists(),str(errors)
        (folder/'metrics_after.txt').write_text(metrics_text())
        save(folder/'stage_complete.json',dict(at_utc=utc_now(),seconds=elapsed,rows=rows))
        print(json.dumps(dict(event='stage_complete',arm=arm,seconds=elapsed)),flush=True)
    save(OUT/'runs_complete.json',dict(at_utc=utc_now(),simulations=24))


def mean(a):return float(np.mean(a)) if a else None


def peak(intervals):
    live=maximum=0
    for _,delta in sorted(intervals):live+=delta;maximum=max(maximum,live)
    return maximum


def audit():
    p=frozen()
    assert read(OUT/'runs_complete.json')['simulations']==24
    assert not list(OUT.glob('*failure*.json')) and not (OUT/'stop_requested').exists()
    assert read(OUT/'cleanup.json')['state_restored'] and not read(OUT/'cleanup.json')['fp8_started']
    assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
    initials,all_rows,stats={},{},[]
    replayed=0
    for arm in ARMS:
        stage=read(OUT/(arm+'_stage')/'stage_complete.json');rows=stage['rows']
        assert sorted(r['Seed'] for r in rows)==SEEDS
        all_rows[arm]={r['Seed']:r for r in rows}
        calls,delays,prep,proposal_times,rank_times,physical,intervals=[],[],[],[],[],[],[]
        canceled=ranked=duplicates=duplicate_pairs=proposal_rounds=submitted=active_frames=0
        late=over=ticks_total=deadline=0
        chosen={}
        for row in rows:
            folder=Path(row['RunMetadata']).parent;meta=read(row['RunMetadata'])
            assert meta['status']=='completed' and meta['control_hz']==10 and meta['config']['MAX_STEPS']==2000
            assert meta['source_sha256']=={k:p['source_sha256'][k] for k in meta['source_sha256']}
            initial=read(folder/'initial.json')
            assert initial==read(REFERENCE/'mpc'/str(row['Seed'])/'initial.json')
            if arm=='rule':initials[row['Seed']]=initial
            else:assert initial==initials[row['Seed']]
            ctrl=pd.read_parquet(row['Control'])
            assert ctrl.Frame.tolist()==list(range(row['Frames']-int(row['Success'])))
            clock=np.load(row['Timing']);ticks=clock['tick_perf_counter'];origin=float(clock['origin_perf_counter'])
            assert len(ticks)==row['Frames']==len(clock['step_seconds'])
            assert np.all(ticks>=origin+np.arange(len(ticks))*.1-.001)
            assert np.allclose(clock['lateness_seconds'],np.maximum(0,ticks-origin-np.arange(len(ticks))*.1),atol=1e-6)
            ticks_total+=len(ticks);over+=int((clock['step_seconds']>.1).sum());late+=int((clock['lateness_seconds']>.1).sum())
            physical.extend([(float(ticks[0]),1),(float(ticks[-1]),-1)])
            if arm=='rule':
                assert set(ctrl.decision_source)=={'rule'}
                continue
            assert meta['config']['HIGH_LEVEL_MODE_SELECTION_ENABLED'] and meta['config']['HIGH_LEVEL_TARGET_SELECTION_ENABLED']
            assert not meta['config']['HIGH_LEVEL_TARGET_HANDOFF_ENABLED'] and not meta['config']['HIGH_LEVEL_EXPLICIT_EXECUTION']
            active=ctrl.loc[ctrl.decision_source!='rule'];active_frames+=len(active)
            assert set(ctrl.decision_source)<={'rule','vllm'} and not ctrl.llm_backend_error.notna().any()
            assert active.llm_hold_frames.eq(HOLD).all() and all(n<=HOLD for n in active.groupby('llm_request_id').size())
            assert (active.current_mode==active.llm_selected_mode).all()
            explicit=active.loc[active.llm_selected_target_sheep_id.notna()]
            assert (explicit.target_sheep_id==explicit.llm_selected_target_sheep_id).all()
            ev=events(row);canceled_ids={e['request_id'] for e in ev if e['event']=='pipeline_canceled'}
            canceled+=len(canceled_ids);assert not set(active.llm_request_id)&canceled_ids
            end=[i for i,e in enumerate(ev) if e['event']=='physical_scene_finished']
            assert len(end)==1 and not any(e['event']=='pipeline_delivered' for e in ev[end[0]+1:])
            ready={e['request_id']:e['result'] for e in ev if e['event']=='selection_ready'}
            responses=validate_model_results(row,arm)
            calls.extend(r['api_seconds'] for r in responses)
            for r in responses:
                intervals.extend([(datetime.fromisoformat(r['api_started_at_utc']).timestamp(),1),
                                  (datetime.fromisoformat(r['api_returned_at_utc']).timestamp(),-1)])
            for e in ev:
                if e['event']=='pipeline_submitted':submitted+=1
                if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e.get('stage')=='selection'):
                    r=e['result'];prep.append(r['prediction_seconds'])
                    if arm in COUNTS:
                        proposal_rounds+=1;proposal_times.append(r['pair_seconds'])
                        a=r['response']['proposals']
                        if len(a)==2 and (a[0]['response']['mode'],a[0]['response']['target_sheep_id'])==(a[1]['response']['mode'],a[1]['response']['target_sheep_id']):duplicate_pairs+=1
                if e['event']=='mpc_ranked':
                    r=e['result'];assert r['horizon_frames']==HORIZON and r['hold_frames']==HOLD
                    assert r['candidates'][0]['candidate_id']=='rule'
                    replay_ranked(r,p['config'],ready[e['request_id']]['request']);replayed+=len(r['candidates'])
                    assert r['winner_id'] in {'rule',*[key for key,guidance in ROLES[:COUNTS[arm]]]}
                    ranked+=1;rank_times.append(r['validation_seconds'])
                    duplicates+=sum(x['reason']=='duplicate_proposal' for x in r['rejected'])
                    chosen[r['winner_id']]=chosen.get(r['winner_id'],0)+1
                if e['event']=='mpc_deadline_fallback':deadline+=1
                if e['event']=='pipeline_delivered':
                    r=ready[e['request_id']]['request']
                    assert e['request_id'] not in canceled_ids
                    assert 0<=e['frame']-e['validation_frame']<=(VALIDATION_MAX_AGE_FRAMES if arm in COUNTS else 30)
                    assert e['frame']-r['trigger_frame']<=p['response_age_limit_frames']
                    assert ctrl.loc[(ctrl.Frame>=r['trigger_frame'])&(ctrl.Frame<e['frame']),'decision_source'].eq('rule').all()
                    delays.append(e['end_to_end_seconds'])
        resources=[json.loads(x) for x in (OUT/(arm+'_stage')/'resources.jsonl').read_text().splitlines()]
        assert peak(physical)==6
        stats.append(dict(arm=arm,runs=6,successes=sum(r['Success'] for r in rows),capped_frames_mean=mean([r['Frames'] for r in rows]),
            batch_seconds=stage['seconds'],api_calls=len(calls),api_mean=mean(calls),api_p95=float(np.percentile(calls,95)) if calls else None,
            preparation_mean=mean(prep),proposal_phase_mean=mean(proposal_times),delivery_mean=mean(delays),deliveries=len(delays),
            fresh_rollout_mean=mean(rank_times),submitted_rounds=submitted,completed_proposal_rounds=proposal_rounds,
            canceled=canceled,ranked_rounds=ranked,duplicate_pairs=duplicate_pairs,deduplicated_proposals=duplicates,
            chosen=chosen,active_llm_frames=active_frames,deadline_fallbacks=deadline,
            scene_overlap_peak=peak(physical),api_overlap_peak=peak(intervals),ticks=ticks_total,step_overruns=over,late_ticks_over_period=late,
            max_step_seconds=max(r['MaxStepSeconds'] for r in rows),max_lateness_seconds=max(r['MaxLatenessSeconds'] for r in rows),
            min_available_gib=min(v['available_bytes'] for v in resources)/2**30))
    comparisons=[]
    for a,b in (('rule','qwen'),('rule','mpc_single'),('rule','mpc_pair'),('mpc_single','mpc_pair')):
        comparisons.append(dict(reference=a,treatment=b,
            rescues=[s for s in SEEDS if not all_rows[a][s]['Success'] and all_rows[b][s]['Success']],
            regressions=[s for s in SEEDS if all_rows[a][s]['Success'] and not all_rows[b][s]['Success']]))
    save(OUT/'summary.json',dict(at_utc=utc_now(),stages=stats,comparisons=comparisons,offline_forecast_replays=replayed,limits=p['limits']))
    save(OUT/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,simulations=24,paid_calls=0,
        service_state_restored=True,evidence_sha256={n:digest(OUT/n) for n in ('protocol.json','summary.json','cleanup.json','model_identity.json','model_preflight.json','offline_tests.log')}))
    print(json.dumps(dict(event='audited_complete',stages=stats,comparisons=comparisons)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:parser.error('Use --run for one new frozen pilot; existing evidence is never overwritten')
    configure('rule');prepare()
    try:
        tested=read(PREFLIGHT/'validated_sources_v2.json')
        assert tested['source_sha256']==joint.hashes()
        assert tested['extra_source_sha256']=={n:digest(ROOT/n) for n in EXTRA}
        assert digest(PREFLIGHT/'tests.log')==tested['tests_log_sha256']
        shutil.copyfile(PREFLIGHT/'tests.log',OUT/'offline_tests.log')
        save(OUT/'offline_tests_passed.json',tested)
        with ModelSession(OUT,'mpc-candidate-comparison-10hz6-20261021-v2',18039):
            preflight()
            save(OUT/'formal_started.json',dict(at_utc=utc_now(),simulations=24,hz=10,workers_per_arm=6))
            stages()
        audit()
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        save(OUT/'execution_failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1600:]))
        raise


if __name__=='__main__':main()
