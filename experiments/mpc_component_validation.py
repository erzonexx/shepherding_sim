"""Five-arm candidate-source x MPC-filter mechanism ablation.

Runs each100-scene arm sequentially, with six actual10Hz worlds per arm.
Only the completion backend replaces validation/ranking. All model proposal
inputs, motion, authority, holds, alarms, cancellation and freshness checks stay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import csv
import json
import multiprocessing
import os
from pathlib import Path
import pickle
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
from detector import Detector
from environment import SwarmEnv
from execution_contract import forecast_execution
from experiment_audit import utc_now
from high_level_controller import HighLevelRequest
from low_level_controller import rule_mode_for_alarms
from mpc_controller import MPCBackend,MPCController,MODEL,HOLD,rank_candidates,proposal_job
from mpc_completion_controller import (CompletionMPCBackend,rank_completion_candidates,
    full_rule_tail_forecast,VERSION as COMPLETION_VERSION)
from mpc_component_ablation import (AblationBackend,AblationController,rank_ablation_candidates,
    heuristic_action,arm_factors,ARMS,VERSION as ABLATION_VERSION)
from realtime_clock import RealtimeClock
from realtime_controller import predictor,public_settings
from tools import run_realtime_joint_comparison as joint
from tools.spark.benchmark_session import write as update
from experiments.mpc_pilot import RecordedInitialEnv
from experiments.mpc_candidate_comparison import events,metric_values,peak
from experiments.oracle_permissions import wilson,exact_mcnemar

OUT=Path(os.environ.get('SHEPHERD_COMPLETION_OUTPUT',ROOT/'logs/experiments/mpc_component_new100_10hz6_20261024'))
BASE='http://127.0.0.1:8002/v1'
EXTRA=('mpc_controller.py','mpc_completion_controller.py',
    'experiments/mpc_component_validation.py','mpc_component_ablation.py','tests/test_mpc_component_ablation.py','experiments/mpc_pilot.py',
    'experiments/mpc_candidate_comparison.py','experiments/qwen_parallel_timing.py',
    'experiments/rule_timing.py','experiments/oracle_permissions.py',
    'tests/test_mpc_controller.py','tests/test_mpc_completion.py',
    'tools/spark/benchmark_session.py','plot_analysis.py')
read,save,digest=joint.read,joint.save,joint.digest


def backend_factory(settings,kind,base_url,executor):
    return AblationBackend(settings,kind,base_url,executor)


def controller_factory(backend,settings,validation_max_age_frames):
    return AblationController(backend,settings=settings)


def configure(arm):
    joint.OUT,joint.BASE,joint.ARMS=OUT,BASE,ARMS
    joint.configure()
    joint.runner.AsyncPredictionBackend=backend_factory
    joint.runner.AsyncTargetController=controller_factory
    joint.runner.SwarmEnv=RecordedInitialEnv
    joint.runner.VERSION=ABLATION_VERSION


def frozen():
    p=read(OUT/'protocol.json')
    assert len(p['seeds'])==100 and len(set(p['seeds']))==100
    assert p['arms']==list(ARMS) and p['simulations']==500
    joint.check_frozen(p)
    assert p['extra_source_sha256']=={n:digest(ROOT/n) for n in EXTRA},'Frozen trial source changed'
    return p


def model_metrics():
    with urlopen(BASE.removesuffix('/v1')+'/metrics',timeout=5) as r:return r.read().decode()


def model_monitor(folder,stopped,errors):
    try:
        with (folder/'server_activity.jsonl').open('x') as stream:
            while not stopped.is_set():
                m=metric_values(model_metrics())
                stream.write(json.dumps(dict(at_utc=utc_now(),running=m.get('vllm:num_requests_running'),waiting=m.get('vllm:num_requests_waiting')))+'\n')
                stream.flush();stopped.wait(1.)
    except BaseException as exc:
        errors.append(str(exc));(OUT/'stop_requested').touch()


def host_monitor(folder,stopped,errors):
    """Linux monitor; has no role in physical/control computations."""
    previous=None
    try:
        with (folder/'resources.jsonl').open('x') as stream:
            while not stopped.is_set():
                memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
                cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
                current=(sum(cpu[:8]),cpu[3]+cpu[4]);percent=None
                if previous and current[0]>previous[0]:percent=100*(1-(current[1]-previous[1])/(current[0]-previous[0]))
                previous=current
                stream.write(json.dumps(dict(at_utc=utc_now(),total_bytes=memory['MemTotal'],available_bytes=memory['MemAvailable'],cpu_percent=percent))+'\n')
                stream.flush()
                if memory['MemAvailable']<2*2**30:raise MemoryError('Available host memory below2GiB')
                stopped.wait(.5)
    except BaseException as exc:
        errors.append(str(exc));(OUT/'stop_requested').touch()


def run_case(arm,seed):
    configure(arm);p=frozen()
    assert not (OUT/arm/str(seed)).exists(),'Do not retry a formal scene'
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) if arm!='rule' else None
    try:
        row=joint.runner.run_one(p,seed,arm,executor=pool)
        if arm!='rule':
            for event in events(row):
                if event['event']=='selection_ready' or (event['event']=='discarded_background_result' and event.get('stage')=='selection'):
                    result=event['result'];assert result['backend_info']['proposal_count']==1
                    assert len(result['response']['proposals'])==1
                    clock=result['response']['clock_proposal']['response']
                    assert not clock.get('backend_error') and clock['served_model']==MODEL
                    response=result['response']['proposals'][0]['response']
                    assert response['hold_frames']==HOLD and not response.get('cache_hit')
                    assert result['backend_info']['action_source']==arm_factors(arm)[0]
        print(json.dumps(dict(event='scene_finished',arm=arm,seed=seed,frames=row['Frames'],success=row['Success'])),flush=True)
        return row
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        update(OUT/f'failure_{arm}_{seed}.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1500:]))
        raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)


def smoke_case(index,item,arm):
    """Historical snapshot,80 real10Hz ticks; separate from the formal seed set."""
    p=frozen();state=item[0];saved=state['observation']
    folder=OUT/'smoke'/arm/str(index);folder.mkdir(parents=True)
    # Real classes need an actual configuration module, using the existing scope.
    with joint.joint_config(**{k:v for k,v in p['config'].items() if hasattr(config,k)}):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as worker:
            backend=backend_factory(config,arm,BASE,worker)
            controller=AblationController(backend,settings=config)
            controller.low_level_controller.__dict__.update(copy_low(state['low']))
            controller.alarm_distance_history.extend(state['history'][:-1])
            backend.set_audit_path(folder/'pipeline.jsonl');backend.warm()
            env=SwarmEnv(controller)
            env.sheep_pos=saved.sheep_positions.copy();env.sheep_vel=saved.sheep_velocities.copy()
            env.dog_pos=saved.dog_position.copy();env.dog_vel=saved.dog_velocity.copy()
            env.goal_pos=saved.goal_position.copy();env.sheep_status=np.array(saved.sheep_statuses,dtype=object)
            env.current_frame=saved.frame;env.current_mode=saved.current_mode
            detector=Detector(config.DANGER_DISPERSION_THRESHOLD,config.DANGER_MEAN_SPREAD_THRESHOLD,
                config.DANGER_STAGNATION_FRAMES,config.DANGER_STAGNATION_THRESHOLD)
            detector.history_dist_to_goal=list(state['history'])
            clock=RealtimeClock(10);timestamps=[]
            try:
                for offset in range(80):
                    tick=clock.tick(offset);timestamps.append(tick)
                    alarms=saved.alarms if offset==0 else detector.detect(env.sheep_pos,env.goal_pos)[0]
                    done=env.step(alarms);clock.record_step(tick)
                    if done:break
                backend.record_event('physical_scene_finished',None,frame=env.current_frame)
            finally:env.close()
            logs=[json.loads(line) for line in (folder/'pipeline.jsonl').read_text().splitlines()]
            ranked=[e['result'] for e in logs if e['event']=='mpc_ranked']
            result=dict(index=index,arm=arm,ticks=len(timestamps),submitted=sum(e['event']=='pipeline_submitted' for e in logs),
                ranked=len(ranked),max_validation_seconds=max([r['validation_seconds'] for r in ranked] or [0]),
                max_step_seconds=max(clock.step_seconds),max_lateness_seconds=max(clock.lateness),
                actual_active_frames=sum(r['decision_source']!='rule' for r in env.control_data))
            update(folder/'result.json',result);return result


def copy_low(low):
    import copy
    return copy.deepcopy(low)


def preflight():
    p=frozen()
    with (OUT/'preflight_inputs.pickle').open('rb') as stream:items=pickle.load(stream)
    # Exactly one live progress proposal; identical prompt path for both MPC arms.
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        live=pool.submit(proposal_job,items[0][0],items[0][1],p['config'],'qwen',BASE,proposal_count=1).result(timeout=60)
    proposal=live['response']['proposals'][0]['response']
    assert not proposal.get('backend_error') and proposal['served_model']==MODEL and proposal['hold_frames']==30
    # Same-platform exact comparison against the retained execution contract.
    parity=0
    for state,request,bundle,_ in items:
        c=predictor(state,p['config']);obs=state['observation']
        for mode,target,policy in ((state['summary']['rule_mode'],None,'FOLLOW_RULE'),
            (bundle['proposals'][0]['response']['mode'],bundle['proposals'][0]['response'].get('target_sheep_id'),'COMMIT')):
            args=(obs,mode,30,p['config']['MAX_STEPS']-obs.frame,c.settings,c.low_level_controller,state['history'],target)
            old=forecast_execution(*args,execution_policy=policy,trigger_alarms=request.alarms)
            new=full_rule_tail_forecast(*args,policy=policy,trigger_alarms=request.alarms)
            assert old==new,'Accelerated full-tail forecast differs from retained execution'
            parity+=1
    results=[]
    for arm in ARMS[1:]:
        with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
            results.extend(pool.map(smoke_case,range(6),items,[arm]*6))
    assert all(r['max_step_seconds']<.1 and r['max_lateness_seconds']<.1 for r in results),'Smoke has a full-period control overrun'
    assert all(sum(r['submitted'] for r in results if r['arm']==arm)>0 for arm in ARMS[1:])
    assert all(sum(r['ranked'] for r in results if r['arm']==arm)>0 for arm in ARMS[1:])
    update(OUT/'model_preflight.json',dict(at_utc=utc_now(),passed=True,live_proposal=asdict(live['request']),
        response=live['response'],exact_original_tail_comparisons=parity,smoke=results))


def stages():
    p=frozen();done=0
    for arm in ARMS:
        frozen();folder=OUT/(arm+'_stage');folder.mkdir()
        (folder/'metrics_before.txt').write_text(model_metrics())
        stopped,errors=threading.Event(),[]
        monitors=[threading.Thread(target=f,args=(folder,stopped,errors)) for f in (host_monitor,model_monitor)]
        for thread in monitors:thread.start()
        start=time.perf_counter();rows=[]
        print(json.dumps(dict(event='stage_started',arm=arm,workers=6)),flush=True)
        try:
            with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
                pending=[pool.submit(run_case,arm,s) for s in p['seeds']]
                for future in as_completed(pending):
                    rows.append(future.result());done+=1
                    update(OUT/'progress.json',dict(at_utc=utc_now(),completed=done,planned=500,arm=arm,arm_completed=len(rows)))
        finally:
            stopped.set()
            for thread in monitors:thread.join()
        assert not errors and not (OUT/'stop_requested').exists(),str(errors)
        (folder/'metrics_after.txt').write_text(model_metrics())
        save(folder/'stage_complete.json',dict(at_utc=utc_now(),seconds=time.perf_counter()-start,rows=rows))
    save(OUT/'runs_complete.json',dict(at_utc=utc_now(),simulations=500))


def replay_ranked(result,ready,values,arm):
    s=result['prediction_state']
    obs=ControlObservation(np.array(s['positions']),np.array(s['dog']),np.array(s['goal']),tuple(s['alarms']),
        s['frame'],tuple(s['statuses']),s['current_mode'],np.array(s['velocities']),np.array(s['dog_velocity']))
    state=dict(observation=obs,low=s['low'],history=s['history'],summary=dict(rule_mode=rule_mode_for_alarms(s['alarms'])))
    request=HighLevelRequest(**dict(ready['request'],alarms=tuple(ready['request']['alarms'])))
    actual=rank_ablation_candidates(state,request,ready['response'],values,arm=arm)
    for key in ('winner_id','shadow_winner_id','response','candidates','rejected','horizon_frames','hold_frames','snapshot_frame'):
        assert actual[key]==result[key],key


def audit():
    p=frozen();assert read(OUT/'runs_complete.json')['simulations']==500
    assert not list(OUT.glob('*failure*.json')) and not (OUT/'stop_requested').exists()
    cleanup=read(OUT/'cleanup.json');assert cleanup['state_restored'] and not cleanup['fp8_started']
    assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
    registered={row['seed']:{k:v for k,v in row.items() if k!='seed'} for row in read(OUT/'registered_scenarios_spark.json')}
    initials,all_rows,stats={},{},[];replayed=0;instruction_hash=None
    for arm in ARMS:
        stage=read(OUT/(arm+'_stage')/'stage_complete.json');rows=stage['rows']
        assert sorted(r['Seed'] for r in rows)==p['seeds'];all_rows[arm]={r['Seed']:r for r in rows}
        calls=[];delivery=[];ranking=[];preparation=[];physical=[];api_intervals=[]
        active_requests=active_frames=submitted=canceled=ranked=deadline=over=late=ticks_total=0
        chosen={};boundary_cases=boundary_frames=boundary_peak=0;stale_rejections=0
        for row in rows:
            folder=Path(row['RunMetadata']).parent;meta=read(row['RunMetadata'])
            assert meta['status']=='completed' and meta['control_hz']==10 and meta['config']['MAX_STEPS']==2000
            assert meta['seed']==row['Seed'] and meta['source_sha256']=={k:p['source_sha256'][k] for k in meta['source_sha256']}
            initial=read(folder/'initial.json')
            assert all(initial['statuses'].count(t)==5 for t in ('A','B','C','Normal'))
            if arm=='rule':
                assert initial==registered[row['Seed']]
                initials[row['Seed']]=initial
            else:assert initial==initials[row['Seed']]
            ctrl=pd.read_parquet(row['Control'])
            assert ctrl.Frame.tolist()==list(range(row['Frames']-int(row['Success'])))
            clock=np.load(row['Timing']);ticks=clock['tick_perf_counter'];origin=float(clock['origin_perf_counter'])
            assert len(ticks)==len(clock['step_seconds'])==row['Frames']
            assert np.all(ticks>=origin+np.arange(len(ticks))*.1-.001)
            assert np.allclose(clock['lateness_seconds'],np.maximum(0,ticks-origin-np.arange(len(ticks))*.1),atol=1e-6)
            ticks_total+=len(ticks);over+=int((clock['step_seconds']>.1).sum());late+=int((clock['lateness_seconds']>.1).sum())
            physical.extend([(float(ticks[0]),1),(float(ticks[-1]),-1)])
            data=pd.read_parquet(row['Data'],columns=['Frame','Agent_Type','X','Y']);sheep=data.loc[data.Agent_Type.str.lower().eq('sheep')]
            outside=((sheep.X<0)|(sheep.X>p['config']['SPACE_SIZE'])|(sheep.Y<0)|(sheep.Y>p['config']['SPACE_SIZE'])).groupby(sheep.Frame).sum()
            boundary_peak=max(boundary_peak,int(outside.max()));boundary_cases+=int(outside.max()>0);boundary_frames+=int(outside.gt(0).sum())
            if arm=='rule':assert set(ctrl.decision_source)=={'rule'};continue
            assert meta['config']['HIGH_LEVEL_MODE_SELECTION_ENABLED'] and meta['config']['HIGH_LEVEL_TARGET_SELECTION_ENABLED']
            assert not meta['config']['HIGH_LEVEL_TARGET_HANDOFF_ENABLED'] and not meta['config']['HIGH_LEVEL_EXPLICIT_EXECUTION']
            active=ctrl.loc[ctrl.decision_source!='rule'];active_frames+=len(active);active_requests+=active.llm_request_id.nunique()
            assert set(ctrl.decision_source)<={'rule','vllm' if arm_factors(arm)[0]=='llm' else 'heuristic'} and not ctrl.llm_backend_error.notna().any()
            assert active.llm_hold_frames.eq(HOLD).all() and all(n<=HOLD for n in active.groupby('llm_request_id').size())
            assert (active.current_mode==active.llm_selected_mode).all()
            explicit=active.loc[active.llm_selected_target_sheep_id.notna()]
            assert (explicit.target_sheep_id==explicit.llm_selected_target_sheep_id).all()
            logs=events(row);end=[i for i,e in enumerate(logs) if e['event']=='physical_scene_finished'];assert len(end)==1
            assert not any(e['event']=='pipeline_delivered' for e in logs[end[0]+1:])
            cancel_ids={e['request_id'] for e in logs if e['event']=='pipeline_canceled'};canceled+=len(cancel_ids)
            assert not set(active.llm_request_id)&cancel_ids
            ready={e['request_id']:e['result'] for e in logs if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e.get('stage')=='selection')}
            for e in logs:
                rid=e['request_id']
                if e['event']=='pipeline_submitted':submitted+=1;assert {'flock_splitting','stagnation'}<=set(e['state_summary']['alarms'])
                if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e.get('stage')=='selection'):
                    r=e['result'];preparation.append(r['prediction_seconds'])
                    assert r['backend_info']['proposal_count']==1
                    proposal=r['response']['clock_proposal'];h=proposal['instructions_sha256']
                    if instruction_hash is None:instruction_hash=h
                    assert h==instruction_hash,'Proposal prompt differs between MPC arms'
                    response=proposal['response'];assert response['served_model']==MODEL and not response.get('backend_error')
                    assert r['backend_info']['action_source']==arm_factors(arm)[0]
                    if arm_factors(arm)[0]=='heuristic':
                        geometry=r['response']['heuristic_input']
                        expected=heuristic_action(r['request']['state_summary'],np.array(geometry['positions']),np.array(geometry['goal']))
                        action=r['response']['proposals'][0]['response']
                        assert all(action[k]==expected[k] for k in ('mode','target_sheep_id'))
                        assert 'served_model' not in action
                    calls.append(response['api_seconds'])
                    from datetime import datetime
                    api_intervals.extend([(datetime.fromisoformat(response['api_started_at_utc']).timestamp(),1),
                                          (datetime.fromisoformat(response['api_returned_at_utc']).timestamp(),-1)])
                if e['event']=='mpc_ranked':
                    r=e['result'];assert r['hold_frames']==30
                    assert r['horizon_frames']==2000-r['frame']
                    replay_ranked(r,ready[rid],p['config'],arm);replayed+=1;ranked+=1
                    ranking.append(r['validation_seconds']);chosen[r['winner_id']]=chosen.get(r['winner_id'],0)+1
                if e['event']=='mpc_deadline_fallback':deadline+=1
                if e['event']=='pipeline_delivered':
                    assert rid not in cancel_ids
                    # Delivered results may be rejected for age; only applied results
                    # must satisfy the existing <=5-frame current-state test.
                    if rid in set(active.llm_request_id):assert 0<=e['frame']-e['validation_frame']<=5
                    trigger=ready[rid]['request']['trigger_frame']
                    assert ctrl.loc[(ctrl.Frame>=trigger)&(ctrl.Frame<e['frame']),'decision_source'].eq('rule').all()
                    delivery.append(e['end_to_end_seconds'])
            if 'llm_rejection_reason' in ctrl:
                stale_rejections+=int(ctrl.llm_rejection_reason.eq('background_validation_stale').sum())
        assert peak(physical)==6
        resources=[json.loads(x) for x in (OUT/(arm+'_stage')/'resources.jsonl').read_text().splitlines()]
        successes=sum(r['Success'] for r in rows);completed=[r['Frames'] for r in rows if r['Success']]
        stats.append(dict(arm=arm,runs=len(p['seeds']),successes=successes,success_rate=successes/len(p['seeds']),wilson95=wilson(successes,len(p['seeds'])),
            capped_frames_mean=float(np.mean([r['Frames'] for r in rows])),completion_frames_mean=float(np.mean(completed)) if completed else None,
            batch_seconds=stage['seconds'],goal_exits_total=sum(r['GoalExits'] for r in rows),recorded_boundary_cases=boundary_cases,
            recorded_boundary_frames=boundary_frames,recorded_boundary_peak=boundary_peak,
            api_calls=len(calls),api_mean=float(np.mean(calls)) if calls else None,api_p95=float(np.percentile(calls,95)) if calls else None,
            delivery_mean=float(np.mean(delivery)) if delivery else None,delivery_p95=float(np.percentile(delivery,95)) if delivery else None,
            ranking_mean=float(np.mean(ranking)) if ranking else None,ranking_p95=float(np.percentile(ranking,95)) if ranking else None,
            preparation_mean=float(np.mean(preparation)) if preparation else None,submitted=submitted,canceled=canceled,ranked=ranked,
            actual_interventions=int(active_requests),actual_active_frames=active_frames,mean_interventions=active_requests/len(p['seeds']),
            chosen=chosen,deadline_fallbacks=deadline,stale_rejections=stale_rejections,api_overlap_peak=peak(api_intervals),
            ticks=ticks_total,step_overruns=over,late_ticks_over_period=late,
            max_step_seconds=max(r['MaxStepSeconds'] for r in rows),max_lateness_seconds=max(r['MaxLatenessSeconds'] for r in rows),
            min_available_gib=min(x['available_bytes'] for x in resources)/2**30))
    pairs=[dict(seed=s,**{a:dict(success=all_rows[a][s]['Success'],frames=all_rows[a][s]['Frames']) for a in ARMS}) for s in p['seeds']]
    comparisons=[]
    contrasts=(('llm_mpc','heuristic_mpc'),('llm_mpc','llm_direct'),('heuristic_mpc','heuristic_direct'),('llm_direct','heuristic_direct'))
    for treatment,reference in contrasts+tuple((arm,'rule') for arm in ARMS[1:]):
        rescues=[r['seed'] for r in pairs if r[treatment]['success'] and not r[reference]['success']]
        regressions=[r['seed'] for r in pairs if not r[treatment]['success'] and r[reference]['success']]
        comparisons.append(dict(treatment=treatment,reference=reference,rescues=rescues,regressions=regressions,
            net_successes=len(rescues)-len(regressions),difference=(len(rescues)-len(regressions))/len(p['seeds']),
            exact_mcnemar_p=exact_mcnemar(len(rescues),len(regressions)),
            faster=sum(r[treatment]['frames']<r[reference]['frames'] for r in pairs),
            slower=sum(r[treatment]['frames']>r[reference]['frames'] for r in pairs)))
    ordered=sorted(range(4),key=lambda i:comparisons[i]['exact_mcnemar_p']);previous=0.
    for rank,index in enumerate(ordered):
        previous=max(previous,min(1.,(4-rank)*comparisons[index]['exact_mcnemar_p']))
        comparisons[index]['holm_p']=previous
    for item in comparisons:item['primary_factor_contrast']=(item['treatment'],item['reference']) in contrasts
    rates={item['arm']:item['success_rate'] for item in stats}
    interaction=(rates['llm_mpc']-rates['heuristic_mpc'])-(rates['llm_direct']-rates['heuristic_direct'])
    update(OUT/'summary.json',dict(at_utc=utc_now(),stages=stats,comparisons=comparisons,
        paired_scenes=pairs,rankings_replayed=replayed,descriptive_interaction=interaction,limits=p['limits']))
    flat=[]
    for row in pairs:
        item=dict(seed=row['seed'])
        item.update({a+'_'+k:row[a][k] for a in ARMS for k in ('success','frames')});flat.append(item)
    pd.DataFrame(flat).to_csv(OUT/'paired_results.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'summary.csv',index=False)
    evidence=('protocol.json','summary.json','summary.csv','paired_results.csv','cleanup.json',
        'model_identity.json','model_preflight.json','offline_tests.log','seed_preflight.json','runtime_packages.txt')
    update(OUT/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,
        simulations=500,scenes=100,paid_calls=0,service_state_restored=True,
        evidence_sha256={n:digest(OUT/n) for n in evidence}))
    print(json.dumps(dict(event='audited_complete',stages=stats,comparisons=comparisons)),flush=True)


def model_start():
    p=frozen()
    lease=Path(OUT.parent/'model_lease_path.txt').read_text().strip()
    lease=Path(lease)
    assert lease.parent==Path.home()/'shepherding/experiments'
    lease.mkdir(exist_ok=False)
    (lease/'lease_seconds').write_text(str(p['model_lease_seconds']))
    with (lease/'lease.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/spark/benchmark_session.py'),'--lease',str(lease)],stdout=log,stderr=log,start_new_session=True)
    (lease/'lease.pid').write_text(str(process.pid))
    deadline=time.monotonic()+960
    while not (lease/'ready.json').exists():
        if (lease/'error.json').exists():raise RuntimeError('Model guardian: '+str(read(lease/'error.json')))
        assert time.monotonic()<deadline,'Model readiness timeout'
        time.sleep(2)
    ready=read(lease/'ready.json')
    with urlopen(BASE+'/models',timeout=5) as response:assert [v['id'] for v in json.load(response)['data']]==[MODEL]
    update(OUT/'model_identity.json',dict(exact_retained_recipe_verified=True,ready=ready))
    return lease


def model_stop(lease):
    (lease/'finish').touch();deadline=time.monotonic()+150
    while not (lease/'cleanup.json').exists():
        assert time.monotonic()<deadline,'Model cleanup timeout'
        time.sleep(2)
    cleanup=read(lease/'cleanup.json');update(OUT/'cleanup.json',cleanup)
    assert cleanup['state_restored'] and not cleanup['fp8_started']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('Use --run for the single registered500-run trial')
    configure('rule');p=frozen()
    assert not (OUT/'manager_started.json').exists(),'Do not restart the batch'
    update(OUT/'manager_started.json',dict(at_utc=utc_now(),pid=os.getpid(),platform=sys.platform))
    lease=None
    try:
        package=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (OUT/'runtime_packages.txt').write_text(package.stdout)
        with (OUT/'offline_tests.log').open('w') as log:
            tested=subprocess.run([sys.executable,'-m','unittest','tests.test_mpc_controller','tests.test_mpc_completion',
                'tests.test_realtime_controller','tests.test_realtime_joint_controller','tests.test_execution_contract',
                'tests.test_mpc_component_ablation'],stdout=log,stderr=log)
        assert tested.returncode==0,'Offline mechanism regression failed before model startup'
        lease=model_start()
        preflight();update(OUT/'formal_started.json',dict(at_utc=utc_now(),simulations=500,hz=10,workers=6))
        stages()
        model_stop(lease);lease=None
        audit()
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        update(OUT/'execution_failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-2000:]))
        if lease is None:
            possible=Path(Path(OUT.parent/'model_lease_path.txt').read_text().strip())
            if possible.exists() and not (possible/'cleanup.json').exists():lease=possible
        if lease is not None:
            try:model_stop(lease)
            except BaseException as cleanup_error:update(OUT/'cleanup_failure.json',dict(reason=str(cleanup_error)))
        raise


if __name__=='__main__':main()
