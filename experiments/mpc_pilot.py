"""Frozen six-scene, three-arm MPC pilot at actual10Hz; no paid API or service retuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import config
from control_types import ControlObservation
from environment import SwarmEnv
from experiment_audit import utc_now
from experiments.rule_timing import monitor as host_monitor
from experiments.qwen_parallel_timing import metric_values
from mpc_controller import (MPCBackend,MPCController,VERSION,HORIZON,HOLD,
    PAIR_DEADLINE_SECONDS,VALIDATION_MAX_AGE_FRAMES,dynamic_forecast)
from realtime_controller import AsyncPredictionBackend
from realtime_joint_controller import AsyncJointController,VERSION as JOINT_VERSION
from tools import run_realtime_joint_comparison as joint
from tools.spark.benchmark_session import ModelSession,MODEL

OUT=ROOT/'logs/experiments/mpc_pilot_10hz_20261020_v2'
BASE='http://127.0.0.1:18037/v1'
SEEDS=list(range(10001,10007))
ARMS=('rule','qwen','mpc')
EXTRA=('mpc_controller.py','experiments/mpc_pilot.py','tests/test_mpc_controller.py',
       'tools/spark/benchmark_session.py','experiments/rule_timing.py','experiments/qwen_parallel_timing.py')
read,save,digest=joint.read,joint.save,joint.digest


class RecordedInitialEnv(SwarmEnv):
    def __init__(self,controller=None):
        super().__init__(controller)
        save(Path(config.DATA_LOG_DIR).parent/'initial.json',dict(
            positions=self.sheep_pos.tolist(),velocities=self.sheep_vel.tolist(),
            dog=self.dog_pos.tolist(),dog_velocity=self.dog_vel.tolist(),
            goal=self.goal_pos.tolist(),statuses=self.sheep_status.tolist()))


def backend_factory(settings,kind,base_url,executor):
    if kind=='mpc':return MPCBackend(settings,base_url,executor)
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
    joint.runner.VERSION=VERSION if arm=='mpc' else JOINT_VERSION


def frozen():
    p=read(OUT/'protocol.json')
    joint.check_frozen(p)
    assert p['extra_source_sha256']=={name:digest(ROOT/name) for name in EXTRA},'Pilot sources changed'
    return p


def prepare():
    assert not OUT.exists(),'Do not overwrite or resume a pilot'
    ref=ROOT/'logs/experiments/qwen_parallel_10hz_20261020/protocol.json'
    p=dict(read(ref),at_utc=utc_now(),name='Three-source event MPC mechanism pilot',
        arms=list(ARMS),simulations=18,seeds=SEEDS,workers_per_arm=6,hz=10,
        source_sha256=joint.hashes(),extra_source_sha256={n:digest(ROOT/n) for n in EXTRA},
        config=dict(read(ref)['config'],VLLM_BASE_URL=BASE,VLLM_MODEL=MODEL),
        reference=str(ref),reference_sha256=digest(ref),
        checkpoint_before_mpc='430c06c2df7907c6fa92cf00795fcc575045ff91',
        mpc=dict(version=VERSION,horizon_frames=HORIZON,hold_frames=HOLD,
            pair_deadline_seconds=PAIR_DEADLINE_SECONDS,validation_max_age_frames=VALIDATION_MAX_AGE_FRAMES,
            proposal_preferences=['safe progress','safe cohesion'],
            score_order=['peak boundary violations','goal exits','more inside goal',
                'centroid distance','maximum spread','splitting frames'],
            safety_gate='No greater predicted boundary peak or goal exits than rules',
            tie_break='Dynamic rules first, then fixed proposal identifier',
            rolling='Replan on original dual-alarm admission after hold/release or retry; no model calls inside forecasts'),
        smoke=dict(arm='mpc',seed=10006,frame_limit=400,
            reason='Known dual alarm at178 frames; fixed mechanism check, not efficacy data'),
        execution='One400-frame MPC smoke, then rule6/Qwen6/MPC6 stages, six scenes concurrently per stage; no twelve-scene combined load.',
        limits=['Six previously used seeds are a mechanism pilot, not generalization evidence.',
            'Actual latency and stochastic model outputs may change trajectories between runs.',
            'Two preferences, fresh dynamic scoring and a stricter snapshot age are a combined treatment, not an isolated ablation.',
            'The same model service is shared by all scenes, max-num-seqs remains4.',
            'Cold-start/cache and fixed stage order are not separately controlled.',
            'Current-state checks bound prediction age; exact same-frame heavy forecasts are not claimed.',
            'Predicted improvement does not guarantee full-scene success.'])
    p.pop('workers',None)
    p.pop('benchmark_sha256',None)
    first=ROOT/'logs/experiments/mpc_pilot_10hz_20261020'
    old=read(first/'protocol.json')
    assert old['source_sha256']==p['source_sha256']
    assert old['extra_source_sha256']['mpc_controller.py']==digest(ROOT/'mpc_controller.py')
    assert not (first/'formal_started.json').exists() and read(first/'cleanup.json')['state_restored']
    p['preflight_gate_correction']=dict(previous=str(first),
        previous_protocol_sha256=digest(first/'protocol.json'),
        previous_failure_sha256=digest(first/'execution_failure.json'),
        reason='A completed prediction is not required when dual alarms correctly clear before model return. Accept and audit cancellation instead.',
        control_sources_unchanged=True,no_formal_scene_previously_started=True)
    OUT.mkdir(parents=True);save(OUT/'protocol.json',p)
    for name in {*p['source_sha256'],*EXTRA}:
        target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(OUT/'manager_started.json',dict(at_utc=utc_now(),pid=os.getpid(),single_manager=True))


def run_case(arm,seed,smoke=False):
    configure(arm);p=frozen()
    folder=(OUT/'smoke'/arm/str(seed)) if smoke else (OUT/arm/str(seed))
    assert not folder.exists(),'Existing scene must not be rerun'
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) if arm!='rule' else None
    try:
        row=joint.runner.run_one(p,seed,arm,executor=pool,smoke=smoke,frame_limit=400 if smoke else None)
        print(json.dumps(dict(event='scene_finished',arm=arm,seed=seed,smoke=smoke,frames=row['Frames'],success=row['Success'])),flush=True)
        return row
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        save(OUT/f'failure_{arm}_{seed}.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1200:]))
        raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)


def events(row):
    return [json.loads(x) for x in Path(row['RequestsAudit']).read_text().splitlines()]


def metrics_text():
    with urlopen(BASE.removesuffix('/v1')+'/metrics',timeout=5) as response:return response.read().decode()


def model_monitor(folder,stopped,errors):
    try:
        with (folder/'server_activity.jsonl').open('x') as stream:
            while not stopped.is_set():
                v=metric_values(metrics_text())
                stream.write(json.dumps(dict(at_utc=utc_now(),running=v.get('vllm:num_requests_running'),waiting=v.get('vllm:num_requests_waiting')))+'\n');stream.flush()
                stopped.wait(1.)
    except BaseException as exc:
        errors.append(str(exc));(OUT/'stop_requested').touch()


def smoke_check(row):
    e=events(row)
    pairs=[x for x in e if x['event']=='selection_ready' or
           (x['event']=='discarded_background_result' and x.get('stage')=='selection')]
    ranked=[x for x in e if x['event']=='mpc_ranked']
    assert pairs,'No model proposal pair exercised in smoke'
    assert all(len(x['result']['response']['proposals'])==2 for x in pairs)
    for x in pairs:
        for proposal in x['result']['response']['proposals']:
            response=proposal['response']
            assert not response.get('backend_error') and response['served_model']==MODEL,response.get('backend_error')
    clock=np.load(row['Timing'])
    assert len(clock['tick_perf_counter'])==400
    canceled={x['request_id'] for x in e if x['event']=='pipeline_canceled'}
    control=pd.read_parquet(row['Control'])
    assert not set(control.loc[control.decision_source!='rule','llm_request_id'])&canceled
    assert ranked or canceled,'Neither ranking nor valid cancellation exercised'
    if not ranked:
        assert all(x['request_id'] in canceled for x in pairs)
    save(OUT/'smoke_passed.json',dict(at_utc=utc_now(),frames=row['Frames'],pairs=len(pairs),
        rankings=len(ranked),cancellations=len(canceled),path='ranking' if ranked else 'cancellation'))


def stages():
    completed=0
    for arm in ARMS:
        frozen();folder=OUT/(arm+'_stage');folder.mkdir()
        (folder/'metrics_before.txt').write_text(metrics_text())
        stopped,errors=threading.Event(),[]
        threads=[threading.Thread(target=f,args=(folder,stopped,errors)) for f in (host_monitor,model_monitor)]
        for thread in threads:thread.start()
        start=time.perf_counter();rows=[]
        print(json.dumps(dict(event='stage_started',arm=arm,workers=6,hz=10)),flush=True)
        try:
            with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(run_case,arm,s) for s in SEEDS]
                for f in as_completed(futures):
                    rows.append(f.result());completed+=1
                    temporary=OUT/'progress.tmp';temporary.write_text(json.dumps(dict(at_utc=utc_now(),
                        completed=completed,planned=18,arm=arm,arm_completed=len(rows))))
                    temporary.replace(OUT/'progress.json')
        finally:
            elapsed=time.perf_counter()-start;stopped.set()
            for thread in threads:thread.join()
        assert not errors and not (OUT/'stop_requested').exists(),str(errors)
        (folder/'metrics_after.txt').write_text(metrics_text())
        save(folder/'stage_complete.json',dict(at_utc=utc_now(),seconds=elapsed,rows=rows))
        print(json.dumps(dict(event='stage_complete',arm=arm,seconds=elapsed)),flush=True)
    save(OUT/'runs_complete.json',dict(at_utc=utc_now(),simulations=18))


def replay_ranked(result,values,request):
    s=result['prediction_state']
    obs=ControlObservation(np.array(s['positions']),np.array(s['dog']),np.array(s['goal']),tuple(s['alarms']),
        s['frame'],tuple(s['statuses']),s['current_mode'],np.array(s['velocities']),np.array(s['dog_velocity']))
    state=dict(observation=obs,low=s['low'],history=s['history'])
    for candidate in result['candidates']:
        is_rule=candidate['candidate_id']=='rule'
        f=dynamic_forecast(state,values,candidate['mode'],candidate['target_sheep_id'],
            policy='FOLLOW_RULE' if is_rule else 'COMMIT',trigger_alarms=request['alarms'])
        assert f==candidate['forecast'],'Saved fresh prediction not reproducible'


def audit():
    p=frozen();assert read(OUT/'runs_complete.json')['simulations']==18
    assert not list(OUT.glob('*failure*.json')) and not (OUT/'stop_requested').exists()
    assert read(OUT/'cleanup.json')['state_restored'] and not read(OUT/'cleanup.json')['fp8_started']
    assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
    all_rows,initial,stats={}, {},[]
    replayed=0
    for arm in ARMS:
        stage=read(OUT/(arm+'_stage')/'stage_complete.json');rows=stage['rows'];all_rows[arm]={r['Seed']:r for r in rows}
        assert sorted(all_rows[arm])==SEEDS
        calls,delays,pairs,rollouts,intervals=[],[],[],[],[]
        canceled=duplicate=ranked_count=deadline_count=0;choices={}
        total_ticks=step_over=late_ticks=0;physical=[]
        for row in rows:
            folder=Path(row['RunMetadata']).parent;meta=read(row['RunMetadata'])
            assert meta['status']=='completed' and meta['control_hz']==10 and meta['config']['MAX_STEPS']==2000
            init=read(folder/'initial.json')
            if arm=='rule':initial[row['Seed']]=init
            else:assert init==initial[row['Seed']]
            ctrl=pd.read_parquet(row['Control']);assert ctrl.Frame.tolist()==list(range(row['Frames']-int(row['Success'])))
            clock=np.load(row['Timing']);ticks=clock['tick_perf_counter'];origin=float(clock['origin_perf_counter'])
            assert len(ticks)==row['Frames']
            assert np.all(ticks>=origin+np.arange(len(ticks))*.1-.001)
            assert np.allclose(clock['lateness_seconds'],np.maximum(0,ticks-origin-np.arange(len(ticks))*.1),atol=1e-6)
            total_ticks+=len(ticks);step_over+=int((clock['step_seconds']>.1).sum());late_ticks+=int((clock['lateness_seconds']>.1).sum())
            physical.extend([(float(ticks[0]),1),(float(ticks[-1]),-1)])
            if arm=='rule':assert set(ctrl.decision_source)=={'rule'};continue
            assert meta['config']['HIGH_LEVEL_MODE_SELECTION_ENABLED'] and meta['config']['HIGH_LEVEL_TARGET_SELECTION_ENABLED']
            assert not meta['config']['HIGH_LEVEL_TARGET_HANDOFF_ENABLED'] and not meta['config']['HIGH_LEVEL_EXPLICIT_EXECUTION']
            active=ctrl.loc[ctrl.decision_source!='rule'];assert set(ctrl.decision_source)<={'rule','vllm'}
            assert not ctrl.llm_backend_error.notna().any() and active.llm_hold_frames.eq(HOLD).all()
            assert all(v<=HOLD for v in active.groupby('llm_request_id').size())
            assert (active.current_mode==active.llm_selected_mode).all()
            targeted=active.loc[active.llm_selected_target_sheep_id.notna()]
            assert (targeted.target_sheep_id==targeted.llm_selected_target_sheep_id).all()
            ev=events(row);canceled_ids={x['request_id'] for x in ev if x['event']=='pipeline_canceled'}
            canceled+=len(canceled_ids);assert not set(active.llm_request_id)&canceled_ids
            finished=[i for i,x in enumerate(ev) if x['event']=='physical_scene_finished'];assert len(finished)==1
            assert not any(x['event']=='pipeline_delivered' for x in ev[finished[0]+1:])
            ready={x['request_id']:x['result'] for x in ev if x['event']=='selection_ready'}
            for e in ev:
                if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e.get('stage')=='selection'):
                    result=e['result']
                    responses=[v['response'] for v in result['response']['proposals']] if arm=='mpc' else [result['response']]
                    if arm=='mpc':
                        assert len(responses)==2;pairs.append(result['pair_seconds'])
                    for response in responses:
                        assert not response.get('backend_error'),response.get('backend_error')
                        assert response['served_model']==MODEL and response['hold_frames']==HOLD and not response.get('cache_hit')
                        calls.append(response['api_seconds'])
                        intervals.extend([(datetime.fromisoformat(response['api_started_at_utc']).timestamp(),1),
                                          (datetime.fromisoformat(response['api_returned_at_utc']).timestamp(),-1)])
                if e['event']=='mpc_ranked':
                    result=e['result'];assert result['horizon_frames']==HORIZON and result['hold_frames']==HOLD
                    assert result['snapshot_frame']==result['frame'] and result['frame']>=ready[e['request_id']]['request']['trigger_frame']
                    assert result['candidates'][0]['candidate_id']=='rule'
                    replay_ranked(result,p['config'],ready[e['request_id']]['request']);replayed+=len(result['candidates'])
                    ranked_count+=1;rollouts.append(result['validation_seconds'])
                    duplicate+=sum(x['reason']=='duplicate_proposal' for x in result['rejected'])
                    choices[result['winner_id']]=choices.get(result['winner_id'],0)+1
                if e['event']=='mpc_deadline_fallback':deadline_count+=1
                if e['event']=='pipeline_delivered':
                    request=ready[e['request_id']]['request']
                    assert e['request_id'] not in canceled_ids
                    assert 0<=e['frame']-e['validation_frame']<=(VALIDATION_MAX_AGE_FRAMES if arm=='mpc' else 30)
                    assert e['frame']-request['trigger_frame']<=p['response_age_limit_frames']
                    assert ctrl.loc[(ctrl.Frame>=request['trigger_frame'])&(ctrl.Frame<e['frame']),'decision_source'].eq('rule').all()
                    delays.append(e['end_to_end_seconds'])
        live=scene_peak=0
        for _,delta in sorted(physical):live+=delta;scene_peak=max(scene_peak,live)
        assert scene_peak==6
        live=api_peak=0
        for _,delta in sorted(intervals):live+=delta;api_peak=max(api_peak,live)
        resources=[json.loads(x) for x in (OUT/(arm+'_stage')/'resources.jsonl').read_text().splitlines()]
        mean=lambda a:float(np.mean(a)) if a else None
        stats.append(dict(arm=arm,successes=sum(r['Success'] for r in rows),runs=6,
            capped_frames_mean=mean([r['Frames'] for r in rows]),batch_seconds=stage['seconds'],
            api_calls=len(calls),api_mean=mean(calls),api_p95=float(np.percentile(calls,95)) if calls else None,
            pair_mean=mean(pairs),delivery_mean=mean(delays),fresh_rollout_mean=mean(rollouts),
            canceled=canceled,duplicate_proposals=duplicate,ranked_rounds=ranked_count,
            chosen=choices,deadline_fallbacks=deadline_count,api_overlap_peak=api_peak,scene_overlap_peak=scene_peak,
            ticks=total_ticks,step_overruns=step_over,late_ticks_over_period=late_ticks,
            max_step_seconds=max(r['MaxStepSeconds'] for r in rows),max_lateness_seconds=max(r['MaxLatenessSeconds'] for r in rows),
            min_available_gib=min(x['available_bytes'] for x in resources)/2**30))
    assert next(s for s in stats if s['arm']=='mpc')['ranked_rounds']>0
    comparisons=[]
    for a,b in [('rule','qwen'),('rule','mpc'),('qwen','mpc')]:
        comparisons.append(dict(reference=a,treatment=b,
            rescues=[s for s in SEEDS if not all_rows[a][s]['Success'] and all_rows[b][s]['Success']],
            regressions=[s for s in SEEDS if all_rows[a][s]['Success'] and not all_rows[b][s]['Success']]))
    save(OUT/'summary.json',dict(at_utc=utc_now(),stages=stats,comparisons=comparisons,
        offline_prediction_replays=replayed,limits=p['limits']))
    save(OUT/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,
        simulations=18,smoke_frames=400,paid_calls=0,service_state_restored=True,
        evidence_sha256={n:digest(OUT/n) for n in ('protocol.json','summary.json','model_identity.json','cleanup.json','smoke_passed.json')}))
    print(json.dumps(dict(event='audited_complete',stages=stats,comparisons=comparisons)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:parser.error('Use --run to start one new frozen pilot')
    configure('mpc');prepare()
    try:
        with (OUT/'offline_tests.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests'],stdout=log,stderr=log)
        assert result.returncode==0,'Offline regression failed; model not started'
        save(OUT/'offline_tests_passed.json',dict(at_utc=utc_now(),returncode=result.returncode))
        with ModelSession(OUT,'mpc-pilot-10hz-20261020-v2',18037):
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
                smoke_check(pool.submit(run_case,'mpc',10006,True).result())
            save(OUT/'formal_started.json',dict(at_utc=utc_now(),simulations=18,hz=10,workers_per_arm=6))
            stages()
        audit()
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        save(OUT/'execution_failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1600:]))
        raise


if __name__=='__main__':main()
