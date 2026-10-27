"""Fresh100 paired validation: rules versus original single-proposal MPC at10Hz."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import config
from control_types import ControlObservation
from experiment_audit import utc_now
from high_level_controller import HighLevelRequest
from low_level_controller import rule_mode_for_alarms
from mpc_controller import BOTH,HOLD,HORIZON,MODEL,forecast_rank,rank_candidates
from realtime_controller import public_settings
from tools.spark.benchmark_session import ModelSession,write as update
from experiments import mpc_candidate_comparison as shared

OUT=ROOT/'logs/experiments/mpc_single_new100_10hz6_20261022'
REFERENCE=ROOT/'logs/experiments/mpc_simplification_new50_10hz6_20261021'
RETIREMENT=ROOT.parent/'shepherding_sim_archive/retired_mpc_simplification_20261022'
BASE='http://127.0.0.1:18042/v1'
ARMS=('rule','mpc_single')
COUNTS={'mpc_single':1}
EXTRA=(*shared.EXTRA,'experiments/mpc_validation.py')
read,save,digest=shared.read,shared.save,shared.digest


def configure(arm):
    shared.OUT,shared.BASE,shared.ARMS,shared.COUNTS=OUT,BASE,ARMS,COUNTS
    shared.EXTRA=EXTRA
    shared.configure(arm)


def frozen():
    return shared.frozen()


def scan_seeds():
    used=set();records=[];conservative=[]
    def collect(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k.lower() in ('seed','random_seed') and type(v) is int:used.add(v)
                elif k.lower()=='seeds' and isinstance(v,list):used.update(x for x in v if type(x) is int)
                elif isinstance(v,(dict,list)):collect(v)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):collect(v)
    roots=(ROOT/'logs',ROOT.parent/'shepherding_sim_archive')
    for base in roots:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ('.json','.csv'):continue
            if any(p in ('node_modules','.git','.venv','__pycache__') for p in path.parts):continue
            if path.suffix.lower()=='.json':
                raw=path.read_text(encoding='utf-8-sig')
                try:collect(json.loads(raw))
                except ValueError:
                    used.update(int(x) for x in re.findall(r'\d+',raw))
                    conservative.append(dict(path=str(path),sha256=digest(path),method='Exclude every integer substring'))
            else:
                with path.open(encoding='utf-8-sig',newline='') as stream:
                    reader=csv.DictReader(stream)
                    keys=[k for k in reader.fieldnames or [] if k.lower() in ('seed','random_seed')]
                    if keys:
                        for row in reader:
                            for k in keys:
                                if row.get(k):used.add(int(float(row[k])))
            records.append(str(path))
    seeds=[];n=30001
    while len(seeds)<100:
        if n not in used:seeds.append(n)
        n+=1
    return dict(at_utc=utc_now(),seeds=seeds,overlap=[],unreadable=[],records_scanned=len(records),
        records=records,conservative_parse=conservative,scan_roots=list(map(str,roots)))


def prepare():
    assert not OUT.exists(),'Existing evidence must not be restarted or overwritten'
    previous=read(REFERENCE/'completion.json')
    assert previous['complete'] and previous['core_audit_passed']
    for n,h in previous['evidence_sha256'].items():assert digest(REFERENCE/n)==h
    proof=read(RETIREMENT/'original_policy_replay.json')
    assert proof['original_rankings_replayed']==46 and proof['all_candidates_scores_winners_responses_equal']
    ref=read(REFERENCE/'protocol.json');scan=scan_seeds()
    values={**public_settings(config),**ref['config'],'VLLM_BASE_URL':BASE,'VLLM_MODEL':MODEL}
    p=dict(at_utc=utc_now(),name='Fresh100 rules versus original single-proposal MPC',
        seeds=scan['seeds'],arms=list(ARMS),simulations=200,workers_per_arm=6,hz=10,period_seconds=.1,
        case=ref['case'],config=values,source_sha256=shared.joint.hashes(),
        extra_source_sha256={n:digest(ROOT/n) for n in EXTRA},model=MODEL,
        reference=str(REFERENCE),reference_completion_sha256=digest(REFERENCE/'completion.json'),
        retirement_replay_sha256=digest(RETIREMENT/'original_policy_replay.json'),
        source_note='Simplified code removed; original policy replayed identically on46 saved rankings. Current sources frozen separately.',
        mpc_proposal_counts=COUNTS,mpc=dict(ref['mpc'],proposal_counts=COUNTS,proposal_roles=['llm_progress'],
            safety_gate='No greater predicted boundary peak or goal exits than dynamic rules'),
        hold_frames=HOLD,response_age_limit_frames=values['HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES'],
        validation_snapshot_max_age_frames=5,validation_snapshot_max_age_seconds=.5,
        artificial_response_age=None,paid_calls=0,model_lease_seconds=10800,
        seed_policy='100 fresh seeds scanned against all historical JSON/CSV; no outcome-based selection.',
        execution='rule100 then mpc_single100; up to6 worlds per group, refill immediately, never overlap groups.',
        pipeline='Actual10Hz; forecasts/model/revalidation in background; rule control continues.',
        preflight='One real progress-role request on a saved historical input; separate from formal cases.',
        primary_metric='All20 sheep reach goal within2000frames',
        secondary_metrics=['Capped frames','Paired rescues and regressions','Goal exits and recorded boundary violations',
            'Actual preparation/API/validation/delivery times','Cancellations and actual takeovers','Control computation and scheduling overruns'],
        stopping_rule='Complete all registered200 runs regardless of scores. Infrastructure failures stop without automatic retry; preserve evidence and restore prior service.',
        limits=['100 fresh cases at one flock size/composition, not all task distributions.',
            'Fixed group order and stochastic model responses/actual arrival times affect trajectories.',
            'Original rule versus MPC adds model input/selection and scoring; does not isolate LLM necessity from non-LLM candidate search.',
            'Short forecasts assume later dynamic rules, not future unknown LLM replies.',
            '10Hz is declared simulation pacing, not a calibrated real-robot guarantee.',
            'Keep this cohort separate from the prior50 and known six-case pilots.'])
    OUT.mkdir(parents=True);save(OUT/'protocol.json',p);save(OUT/'seed_preflight.json',scan)
    for n in {*p['source_sha256'],*EXTRA}:
        target=OUT/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
    save(OUT/'manager_started.json',dict(at_utc=utc_now(),pid=os.getpid(),single_manager=True))
    print(json.dumps(dict(event='registered',seeds=[scan['seeds'][0],scan['seeds'][-1]],simulations=200,records_scanned=scan['records_scanned'])),flush=True)


def run_case(arm,seed):
    configure(arm)
    return shared.run_case(arm,seed)


def preflight():
    p=frozen()
    with (shared.PREPARATION/'replay_inputs.pickle').open('rb') as stream:item=pickle.load(stream)[0]
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        backend=shared.backend_factory(config,'mpc_single',BASE,pool)
        result=pool.submit(backend.select_fn,item['state'],item['request'],p['config'],'qwen',BASE).result(timeout=60)
        backend.close()
    proposals=result['response']['proposals'];assert len(proposals)==1 and proposals[0]['candidate_id']=='llm_progress'
    r=proposals[0]['response'];assert not r.get('backend_error') and r['served_model']==MODEL and r['hold_frames']==HOLD
    save(OUT/'model_preflight.json',shared.json_safe(dict(at_utc=utc_now(),passed=True,api_calls=1,result=result)))
    print(json.dumps(dict(event='model_preflight_passed',api_calls=1)),flush=True)


def stages():
    p=frozen();done=0
    for arm in ARMS:
        frozen();folder=OUT/(arm+'_stage');folder.mkdir()
        (folder/'metrics_before.txt').write_text(shared.metrics_text())
        stopped,errors=threading.Event(),[]
        monitors=[threading.Thread(target=f,args=(folder,stopped,errors)) for f in (shared.host_monitor,shared.model_monitor)]
        for t in monitors:t.start()
        start=time.perf_counter();rows=[]
        print(json.dumps(dict(event='stage_started',arm=arm,workers=6)),flush=True)
        try:
            with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(run_case,arm,s) for s in p['seeds']]
                try:
                    for future in as_completed(futures):
                        rows.append(future.result());done+=1
                        update(OUT/'progress.json',dict(at_utc=utc_now(),completed=done,planned=200,arm=arm,arm_completed=len(rows)))
                except BaseException:
                    for future in futures:future.cancel()
                    raise
        finally:
            elapsed=time.perf_counter()-start;stopped.set()
            for t in monitors:t.join()
        assert not errors and not (OUT/'stop_requested').exists(),str(errors)
        (folder/'metrics_after.txt').write_text(shared.metrics_text())
        save(folder/'stage_complete.json',dict(at_utc=utc_now(),seconds=elapsed,rows=rows))
        print(json.dumps(dict(event='stage_complete',arm=arm,seconds=elapsed)),flush=True)
    save(OUT/'runs_complete.json',dict(at_utc=utc_now(),simulations=200))


def replay(r,bundle,values):
    s=r['prediction_state']
    obs=ControlObservation(np.array(s['positions']),np.array(s['dog']),np.array(s['goal']),tuple(s['alarms']),
        s['frame'],tuple(s['statuses']),s['current_mode'],np.array(s['velocities']),np.array(s['dog_velocity']))
    state=dict(observation=obs,low=s['low'],history=s['history'],summary=dict(rule_mode=rule_mode_for_alarms(s['alarms'])))
    request=HighLevelRequest(**dict(bundle['request'],alarms=tuple(bundle['request']['alarms'])))
    actual=rank_candidates(state,request,bundle['response'],values)
    for key in ('winner_id','response','candidates','rejected','horizon_frames','hold_frames','snapshot_frame'):assert actual[key]==r[key],key


def audit():
    p=frozen();seeds=p['seeds'];assert read(OUT/'runs_complete.json')['simulations']==200
    assert not list(OUT.glob('*failure*.json')) and not (OUT/'stop_requested').exists()
    cleanup=read(OUT/'cleanup.json');assert cleanup['state_restored'] and not cleanup['fp8_started']
    assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
    initials,all_rows,stats={},{},[];replayed=0;instruction_hash=None
    for arm in ARMS:
        stage=read(OUT/(arm+'_stage')/'stage_complete.json');rows=stage['rows']
        assert sorted(r['Seed'] for r in rows)==seeds;all_rows[arm]={r['Seed']:r for r in rows}
        calls,prep,phase,delivery,rank_times,physical,api_intervals=[],[],[],[],[],[],[]
        submitted=canceled=ranked=active_requests=active_frames=deadline=over=late=ticks_total=0
        chosen={};rejections={};boundary_frames=boundary_cases=boundary_peak=0
        for row in rows:
            folder=Path(row['RunMetadata']).parent;meta=read(row['RunMetadata'])
            assert meta['status']=='completed' and meta['control_hz']==10 and meta['config']['MAX_STEPS']==2000
            assert meta['seed']==row['Seed']==meta['config']['RANDOM_SEED']
            assert meta['source_sha256']=={k:p['source_sha256'][k] for k in meta['source_sha256']}
            initial=read(folder/'initial.json')
            if arm=='rule':initials[row['Seed']]=initial
            else:assert initial==initials[row['Seed']]
            ctrl=pd.read_parquet(row['Control']);assert ctrl.Frame.tolist()==list(range(row['Frames']-int(row['Success'])))
            clock=np.load(row['Timing']);ticks=clock['tick_perf_counter'];origin=float(clock['origin_perf_counter'])
            assert len(ticks)==len(clock['step_seconds'])==row['Frames']
            assert np.all(ticks>=origin+np.arange(len(ticks))*.1-.001)
            assert np.allclose(clock['lateness_seconds'],np.maximum(0,ticks-origin-np.arange(len(ticks))*.1),atol=1e-6)
            ticks_total+=len(ticks);over+=int((clock['step_seconds']>.1).sum());late+=int((clock['lateness_seconds']>.1).sum())
            physical.extend([(float(ticks[0]),1),(float(ticks[-1]),-1)])
            data=pd.read_parquet(row['Data'],columns=['Frame','Agent_Type','X','Y']);sheep=data.loc[data.Agent_Type.str.lower().eq('sheep')]
            assert len(sheep)>0
            outside=((sheep.X<0)|(sheep.X>p['config']['SPACE_SIZE'])|(sheep.Y<0)|(sheep.Y>p['config']['SPACE_SIZE'])).groupby(sheep.Frame).sum()
            peak=int(outside.max());boundary_peak=max(boundary_peak,peak);boundary_cases+=int(peak>0);boundary_frames+=int(outside.gt(0).sum())
            if arm=='rule':assert set(ctrl.decision_source)=={'rule'};continue
            assert meta['config']['HIGH_LEVEL_MODE_SELECTION_ENABLED'] and meta['config']['HIGH_LEVEL_TARGET_SELECTION_ENABLED']
            assert not meta['config']['HIGH_LEVEL_TARGET_HANDOFF_ENABLED'] and not meta['config']['HIGH_LEVEL_EXPLICIT_EXECUTION']
            active=ctrl.loc[ctrl.decision_source!='rule'];active_frames+=len(active);active_requests+=active.llm_request_id.nunique()
            assert set(ctrl.decision_source)<={'rule','vllm'} and not ctrl.llm_backend_error.notna().any()
            assert active.llm_hold_frames.eq(HOLD).all() and all(n<=HOLD for n in active.groupby('llm_request_id').size())
            assert (active.current_mode==active.llm_selected_mode).all()
            explicit=active.loc[active.llm_selected_target_sheep_id.notna()];assert (explicit.target_sheep_id==explicit.llm_selected_target_sheep_id).all()
            events=shared.events(row);end=[i for i,e in enumerate(events) if e['event']=='physical_scene_finished'];assert len(end)==1
            assert not any(e['event']=='pipeline_delivered' for e in events[end[0]+1:])
            cancel_ids={e['request_id'] for e in events if e['event']=='pipeline_canceled'}
            canceled+=len(cancel_ids);assert not set(active.llm_request_id)&cancel_ids
            ready={e['request_id']:e['result'] for e in events if e['event']=='selection_ready'}
            responses=shared.validate_model_results(row,arm);calls.extend(r['api_seconds'] for r in responses)
            for r in responses:api_intervals.extend([(datetime.fromisoformat(r['api_started_at_utc']).timestamp(),1),(datetime.fromisoformat(r['api_returned_at_utc']).timestamp(),-1)])
            for e in events:
                rid=e['request_id']
                if e['event']=='pipeline_submitted':submitted+=1;assert BOTH<=set(e['state_summary']['alarms'])
                if e['event']=='selection_ready' or (e['event']=='discarded_background_result' and e.get('stage')=='selection'):
                    r=e['result'];prep.append(r['prediction_seconds']);phase.append(r['pair_seconds'])
                    h=r['response']['proposals'][0]['instructions_sha256']
                    if instruction_hash is None:instruction_hash=h
                    assert h==instruction_hash
                if e['event']=='mpc_ranked':
                    r=e['result'];assert r['horizon_frames']==HORIZON and r['hold_frames']==HOLD
                    assert BOTH<=set(r['prediction_state']['alarms'])
                    replay(r,ready[rid],p['config']);replayed+=1;ranked+=1;rank_times.append(r['validation_seconds'])
                    chosen[r['winner_id']]=chosen.get(r['winner_id'],0)+1
                    for rejection in r['rejected']:rejections[rejection['reason']]=rejections.get(rejection['reason'],0)+1
                if e['event']=='mpc_deadline_fallback':deadline+=1
                if e['event']=='pipeline_delivered':
                    assert rid not in cancel_ids and 0<=e['frame']-e['validation_frame']<=5
                    trigger=ready[rid]['request']['trigger_frame'];assert e['frame']-trigger<=p['response_age_limit_frames']
                    assert ctrl.loc[(ctrl.Frame>=trigger)&(ctrl.Frame<e['frame']),'decision_source'].eq('rule').all()
                    delivery.append(e['end_to_end_seconds'])
        assert shared.peak(physical)==6
        resources=[json.loads(x) for x in (OUT/(arm+'_stage')/'resources.jsonl').read_text().splitlines()]
        stats.append(dict(arm=arm,runs=100,successes=sum(r['Success'] for r in rows),capped_frames_mean=shared.mean([r['Frames'] for r in rows]),
            batch_seconds=stage['seconds'],goal_exits_total=sum(r['GoalExits'] for r in rows),recorded_boundary_frames=boundary_frames,
            recorded_boundary_cases=boundary_cases,recorded_boundary_peak=boundary_peak,
            api_calls=len(calls),api_mean=shared.mean(calls),api_p95=float(np.percentile(calls,95)) if calls else None,
            preparation_mean=shared.mean(prep),proposal_phase_mean=shared.mean(phase),delivery_mean=shared.mean(delivery),deliveries=len(delivery),
            fresh_rollout_mean=shared.mean(rank_times),submitted_rounds=submitted,canceled=canceled,ranked_rounds=ranked,
            chosen=chosen,candidate_rejections=rejections,active_llm_requests=int(active_requests),active_llm_frames=active_frames,deadline_fallbacks=deadline,
            scene_overlap_peak=shared.peak(physical),api_overlap_peak=shared.peak(api_intervals),ticks=ticks_total,step_overruns=over,late_ticks_over_period=late,
            max_step_seconds=max(r['MaxStepSeconds'] for r in rows),max_lateness_seconds=max(r['MaxLatenessSeconds'] for r in rows),min_available_gib=min(x['available_bytes'] for x in resources)/2**30))
    pairs=[dict(seed=s,**{a:dict(success=all_rows[a][s]['Success'],frames=all_rows[a][s]['Frames']) for a in ARMS}) for s in seeds]
    comparison=dict(reference='rule',treatment='mpc_single',rescues=[x['seed'] for x in pairs if not x['rule']['success'] and x['mpc_single']['success']],
        regressions=[x['seed'] for x in pairs if x['rule']['success'] and not x['mpc_single']['success']],
        faster=sum(x['mpc_single']['frames']<x['rule']['frames'] for x in pairs),
        same=sum(x['mpc_single']['frames']==x['rule']['frames'] for x in pairs),slower=sum(x['mpc_single']['frames']>x['rule']['frames'] for x in pairs))
    scan=read(OUT/'seed_preflight.json');assert scan['seeds']==seeds and not scan['overlap'] and not scan['unreadable']
    save(OUT/'summary.json',dict(at_utc=utc_now(),stages=stats,comparison=comparison,paired_scenes=pairs,offline_rankings_replayed=replayed,limits=p['limits']))
    evidence=('protocol.json','summary.json','cleanup.json','model_identity.json','model_preflight.json','offline_tests.log','seed_preflight.json')
    save(OUT/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,simulations=200,paid_calls=0,service_state_restored=True,
        evidence_sha256={n:digest(OUT/n) for n in evidence}))
    print(json.dumps(dict(event='audited_complete',stages=stats,comparison=comparison)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('Use --run for one frozen200-run validation; completed evidence is never overwritten')
    configure('rule');prepare()
    try:
        temp=OUT/'test_temp';temp.mkdir()
        with (OUT/'offline_tests.log').open('w',encoding='utf-8') as log:
            tested=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests'],stdout=log,stderr=log,
                env=dict(os.environ,TEMP=str(temp),TMP=str(temp)))
        assert tested.returncode==0,'Offline regression tests failed; model not started'
        with ModelSession(OUT,'mpc-single-new100-10hz6-20261022',18042,lease_seconds=10800):
            preflight();save(OUT/'formal_started.json',dict(at_utc=utc_now(),simulations=200,hz=10,workers_per_arm=6));stages()
        audit()
    except BaseException as exc:
        (OUT/'stop_requested').touch()
        save(OUT/'execution_failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1600:]))
        raise


if __name__=='__main__':main()
