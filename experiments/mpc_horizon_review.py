"""Offline review of two accepted MPC choices; fixed actions, varying lookahead."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from control_types import ControlObservation
from detector import Detector
from dynamics import advance_dog,advance_sheep
from execution_contract import forecast_execution
from experiment_audit import utc_now
from low_level_controller import rule_mode_for_alarms
from mpc_controller import forecast_rank,HOLD
from realtime_controller import predictor
from experiments import mpc_candidate_comparison as comparison

OUT=ROOT/'logs/experiments/mpc_horizon_review_20261021'
REFERENCE=comparison.OUT
SEEDS=(10005,10006)
HORIZONS=(30,60,120,240)
read,save,digest=comparison.read,comparison.save,comparison.digest


def snapshot(record):
    s=record['prediction_state']
    obs=ControlObservation(np.array(s['positions']),np.array(s['dog']),np.array(s['goal']),
        tuple(s['alarms']),s['frame'],tuple(s['statuses']),s['current_mode'],
        np.array(s['velocities']),np.array(s['dog_velocity']))
    return dict(observation=obs,low=copy.deepcopy(s['low']),history=list(s['history']),
                summary=dict(rule_mode=rule_mode_for_alarms(s['alarms'])))


def advance_rule_frame(state,values):
    """Bridge the one physical rule tick between validation snapshot and adoption."""
    c=predictor(state,values);obs=state['observation']
    decision=c.low_level_controller.decide(obs)
    dog,dog_velocity=advance_dog(obs.dog_position,decision.navigation_point,c.settings)
    positions,velocities=advance_sheep(obs.sheep_positions,obs.sheep_velocities,
        np.asarray(obs.sheep_statuses),dog,obs.goal_position,c.settings)
    detector=Detector(c.settings.DANGER_DISPERSION_THRESHOLD,c.settings.DANGER_MEAN_SPREAD_THRESHOLD,
        c.settings.DANGER_STAGNATION_FRAMES,c.settings.DANGER_STAGNATION_THRESHOLD)
    detector.history_dist_to_goal=list(state['history'])
    alarms=tuple(detector.detect(positions,obs.goal_position)[0])
    new_obs=ControlObservation(positions,dog,obs.goal_position.copy(),alarms,obs.frame+1,
        obs.sheep_statuses,decision.mode,velocities,dog_velocity)
    return dict(observation=new_obs,low={k:copy.deepcopy(v) for k,v in vars(c.low_level_controller).items() if k!='settings'},
        history=list(detector.history_dist_to_goal),summary=dict(rule_mode=rule_mode_for_alarms(alarms)))


def forecast(state,values,candidate,horizon,trigger_alarms,trace=False):
    c=predictor(state,values)
    started=time.perf_counter()
    result=forecast_execution(state['observation'],candidate['mode'],HOLD,horizon,c.settings,
        c.low_level_controller,state['history'],candidate['target_sheep_id'],
        execution_policy='FOLLOW_RULE' if candidate['candidate_id']=='rule' else 'COMMIT',
        trigger_alarms=trigger_alarms,return_trace=trace)
    return result,time.perf_counter()-started


def select(candidates):
    rule=candidates[0]
    eligible=[rule]
    for c in candidates[1:]:
        f,r=c['forecast'],rule['forecast']
        if f['peak_out_of_bounds']<=r['peak_out_of_bounds'] and f['predicted_goal_exits']<=r['predicted_goal_exits']:
            eligible.append(c)
    return min(eligible,key=lambda c:(forecast_rank(c['forecast']),c['candidate_id']!='rule',c['candidate_id']))['candidate_id']


def verify_recorded_trace(result,frame,data,control):
    """Recorded positions have two decimals; modes and targets remain exact."""
    sheep=data.loc[data.Agent_Type=='Sheep'].sort_values(['Frame','Agent_ID'])
    dogs=data.loc[data.Agent_Type=='Dog'].sort_values('Frame')
    recorded={int(f):g[['X','Y']].to_numpy() for f,g in sheep.groupby('Frame')}
    dog_recorded={int(r.Frame):np.array([r.X,r.Y]) for r in dogs.itertuples()}
    commands=control.set_index('Frame')
    checked=0
    for offset,step in enumerate(result['trace']):
        actual_frame=frame+offset
        if actual_frame in commands.index:
            command=commands.loc[actual_frame]
            assert step['mode']==command.current_mode
            target=None if pd.isna(command.target_sheep_id) else int(command.target_sheep_id)
            assert step['target_sheep_id']==target
        next_frame=actual_frame+1
        if next_frame in recorded:
            assert np.array_equal(np.round(step['sheep_positions'],2),recorded[next_frame])
            assert np.array_equal(np.round(step['dog_position'],2),dog_recorded[next_frame])
            checked+=1
    return checked


def run():
    assert not OUT.exists(),'Preserve existing review; no overwrite or retuned repetition'
    completion=read(REFERENCE/'completion.json')
    assert completion['complete'] and completion['core_audit_passed']
    for name,h in completion['evidence_sha256'].items():assert digest(REFERENCE/name)==h
    p=comparison.frozen();values=p['config']
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json',dict(at_utc=utc_now(),seeds=list(SEEDS),horizons=list(HORIZONS),
        remaining_budget=2000,hold_frames=HOLD,states=['validation_snapshot','actual_adoption_state'],
        source_sha256=p['source_sha256'],extra_source_sha256=p['extra_source_sha256'],
        review_source_sha256=digest(__file__),reference_completion_sha256=digest(REFERENCE/'completion.json'),
        api_calls=0,model_started=False,live_controller_changed=False,
        limits=['Two selected known outcomes; diagnostic only, not selection of a new runtime horizon.',
            'All branches keep original30-frame maximum hold and real early releases, then rule control.',
            'No future model answers are inserted; delivered-rule choices do not change physical motion.',
            'Saved position logs have two decimals; forecast reproduction at saved snapshot is exact.',
            'Elapsed forecast time is a local diagnostic, not a deadline guarantee.']))
    for name in {*p['source_sha256'],*p['extra_source_sha256'],'experiments/mpc_horizon_review.py'}:
        dst=OUT/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    results=[]
    for seed in SEEDS:
        row=read(REFERENCE/'mpc_single'/str(seed)/'completed_row.json')
        events=comparison.events(row)
        ranked=[e for e in events if e['event']=='mpc_ranked' and e['result']['winner_id']!='rule']
        assert len(ranked)==1
        event=ranked[0];original=event['result'];rid=event['request_id']
        delivered=next(e for e in events if e['event']=='pipeline_delivered' and e['request_id']==rid)
        ready=next(e for e in events if e['event']=='selection_ready' and e['request_id']==rid)
        trigger=ready['result']['request']['alarms']
        state=snapshot(original)
        assert delivered['frame']==state['observation'].frame+1
        adopted=advance_rule_frame(state,values)
        assert adopted['observation'].frame==delivered['frame']
        control=pd.read_parquet(row['Control']);data=pd.read_parquet(row['Data'])
        first=int(control.loc[control.decision_source!='rule','Frame'].min())
        assert first==adopted['observation'].frame
        branch_definitions=[dict(candidate_id=c['candidate_id'],mode=c['mode'],target_sheep_id=c['target_sheep_id']) for c in original['candidates']]
        for label,s in (('validation_snapshot',state),('actual_adoption_state',adopted)):
            frame=s['observation'].frame
            queries=[]
            for horizon in (*HORIZONS,2000-frame):
                candidates=[];elapsed=0.
                for candidate in branch_definitions:
                    f,t=forecast(s,values,candidate,horizon,trigger,trace=horizon==2000-frame)
                    elapsed+=t
                    if label=='validation_snapshot' and horizon==60:
                        expected=next(c['forecast'] for c in original['candidates'] if c['candidate_id']==candidate['candidate_id'])
                        assert f==expected,'Original60-frame scores changed'
                    checked=None
                    if label=='actual_adoption_state' and horizon==2000-frame:
                        if candidate['candidate_id']!='rule':
                            checked=verify_recorded_trace(f,frame,data,control)
                            predicted_frames=frame+f['simulated_frames']+int(f['predicted_success'])
                            assert predicted_frames==row['Frames'] and f['predicted_success']==row['Success']
                        else:
                            rule_row=read(REFERENCE/'rule'/str(seed)/'completed_row.json')
                            checked=verify_recorded_trace(f,frame,pd.read_parquet(rule_row['Data']),pd.read_parquet(rule_row['Control']))
                            predicted_frames=frame+f['simulated_frames']+int(f['predicted_success'])
                            assert predicted_frames==rule_row['Frames'] and f['predicted_success']==rule_row['Success']
                    trace=f.pop('trace',None)
                    if trace is not None:
                        save(OUT/f'{seed}_{label}_{candidate["candidate_id"]}_full_trace.json',trace)
                    candidates.append(dict(**candidate,forecast=f,forecast_seconds=t,recorded_trace_frames_checked=checked,
                        absolute_finish_frame=frame+f['simulated_frames']+1 if f['predicted_success'] else None))
                queries.append(dict(horizon_frames=horizon,prediction_seconds=elapsed,winner=select(candidates),candidates=candidates))
            results.append(dict(seed=seed,state=label,frame=frame,actual_success=row['Success'],actual_frames=row['Frames'],queries=queries))
        print(json.dumps(dict(event='case_reviewed',seed=seed)),flush=True)
    comparison.frozen()
    save(OUT/'summary.json',dict(at_utc=utc_now(),cases=results,api_calls=0,live_controller_changed=False))
    save(OUT/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,api_calls=0,model_started=False,
        original60_frame_scores_reproduced=True,actual_and_rule_tail_recordings_reproduced=True,
        reference_completion_unchanged=digest(REFERENCE/'completion.json')==read(OUT/'protocol.json')['reference_completion_sha256'],
        evidence_sha256={n:digest(OUT/n) for n in ('protocol.json','summary.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--validation100',action='store_true',help='Review all rescue/regression episodes from the completed new100 cohort, offline only')
    args=parser.parse_args()
    if not args.run:parser.error('Use --run for the fixed offline review')
    if args.validation100:
        return review_validation100()
    try:run()
    except BaseException as exc:
        if OUT.exists():save(OUT/'failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1200:]))
        raise


def review_validation100():
    from experiments import mpc_validation as validation
    reference=validation.OUT
    output=ROOT/'logs/experiments/mpc_review_new100_20261022'
    assert not output.exists(),'Preserve existing offline review'
    validation.configure('rule');p=validation.frozen();values=p['config']
    done=read(reference/'completion.json');assert done['complete'] and done['core_audit_passed']
    for n,h in done['evidence_sha256'].items():assert digest(reference/n)==h
    summary=read(reference/'summary.json');rescues=summary['comparison']['rescues'];losses=summary['comparison']['regressions']
    seeds=sorted(rescues+losses)
    output.mkdir()
    save(output/'protocol.json',dict(at_utc=utc_now(),reference=str(reference),reference_completion_sha256=digest(reference/'completion.json'),
        seeds=seeds,rescues=rescues,regressions=losses,horizons=list(HORIZONS),hold_frames=HOLD,budget=2000,
        source_sha256=p['source_sha256'],extra_source_sha256=p['extra_source_sha256'],review_source_sha256=digest(__file__),
        api_calls=0,model_started=False,live_controller_changed=False,
        branch_definition='At each actual adoption: current selected action with original early releases then rules, versus rules immediately. No future LLM answers in either branch.',
        checks=['Original validation60 metrics exactly reproduced','Bridge validation snapshot to real adoption with intervening rule ticks',
            'Actual action branch reproduces recorded control/positions until the next real LLM adoption',
            'First adoption rule-only branch reproduces the original pure-rule tail',
            'Last adoption action branch reproduces the recorded whole-scene outcome'],
        limits=['11 cases selected by known paired outcomes, not a new effectiveness sample.',
            'Later branches condition on the actual earlier intervention history.',
            'Branch outcomes exclude future LLM calls, so they need not match a multi-intervention online episode.',
            'Recorded position comparisons use the stored two-decimal precision.',
            'Long-window choices are diagnostics, not proposed or tested runtime changes.']))
    target=output/'source/experiments/mpc_horizon_review.py';target.parent.mkdir(parents=True);shutil.copyfile(__file__,target)
    results=[];reproduced=prefix_ticks=0
    try:
        for seed in seeds:
            row=read(reference/'mpc_single'/str(seed)/'completed_row.json');rule_row=read(reference/'rule'/str(seed)/'completed_row.json')
            data=pd.read_parquet(row['Data']);control=pd.read_parquet(row['Control'])
            rule_data=pd.read_parquet(rule_row['Data']);rule_control=pd.read_parquet(rule_row['Control'])
            events=comparison.events(row)
            rankings={e['request_id']:e['result'] for e in events if e['event']=='mpc_ranked'}
            ready={e['request_id']:e['result'] for e in events if e['event']=='selection_ready'}
            deliveries={e['request_id']:e for e in events if e['event']=='pipeline_delivered'}
            active=control.loc[control.decision_source!='rule']
            groups=sorted(active.groupby('llm_request_id'),key=lambda item:int(item[1].Frame.min()))
            assert groups
            episodes=[]
            for index,(rid,g) in enumerate(groups):
                ranked=rankings[rid];state=snapshot(ranked);trigger=ready[rid]['request']['alarms']
                definitions=[{k:c[k] for k in ('candidate_id','mode','target_sheep_id')} for c in ranked['candidates']]
                assert len(definitions)==2 and ranked['winner_id']=='llm_progress'
                validation_scores=[]
                for candidate in definitions:
                    f,_=forecast(state,values,candidate,60,trigger)
                    expected=next(c['forecast'] for c in ranked['candidates'] if c['candidate_id']==candidate['candidate_id'])
                    assert f==expected,'Original validation forecast changed'
                    validation_scores.append(dict(**candidate,forecast=f));reproduced+=1
                frame=int(g.Frame.min());assert deliveries[rid]['frame']==frame
                bridge=frame-state['observation'].frame;assert 0<=bridge<=5
                while state['observation'].frame<frame:state=advance_rule_frame(state,values)
                initial_sheep=data.loc[(data.Frame==frame)&(data.Agent_Type=='Sheep')].sort_values('Agent_ID')[['X','Y']].to_numpy()
                initial_dog=data.loc[(data.Frame==frame)&(data.Agent_Type=='Dog')][['X','Y']].to_numpy()[0]
                assert np.array_equal(np.round(state['observation'].sheep_positions,2),initial_sheep)
                assert np.array_equal(np.round(state['observation'].dog_position,2),initial_dog)
                queries=[];tails={};remaining=2000-frame
                for horizon in (*HORIZONS,remaining):
                    candidates=[];elapsed=0.
                    for candidate in definitions:
                        full=horizon==remaining
                        f,seconds=forecast(state,values,candidate,horizon,trigger,trace=full);elapsed+=seconds
                        verified=0
                        if full:
                            predicted_frames=frame+f['simulated_frames']+int(f['predicted_success'])
                            is_rule=candidate['candidate_id']=='rule'
                            if not is_rule:
                                next_frame=int(groups[index+1][1].Frame.min()) if index+1<len(groups) else 2000
                                prefix=dict(trace=f['trace'][:max(0,next_frame-frame)])
                                verified=verify_recorded_trace(prefix,frame,data,control);prefix_ticks+=verified
                                assert f['committed_frames']==len(g)
                                if index+1==len(groups):assert predicted_frames==row['Frames'] and f['predicted_success']==row['Success']
                            elif index==0:
                                verified=verify_recorded_trace(f,frame,rule_data,rule_control)
                                assert predicted_frames==rule_row['Frames'] and f['predicted_success']==rule_row['Success']
                            trace=f.pop('trace')
                            tails[candidate['candidate_id']]=dict(success=f['predicted_success'],frames=predicted_frames,
                                forecast=f,recorded_trace_frames_checked=verified)
                        candidates.append(dict(**candidate,forecast=f,forecast_seconds=seconds))
                    queries.append(dict(horizon_frames=horizon,winner=select(candidates),candidates=candidates,prediction_seconds=elapsed))
                r,l=tails['rule'],tails['llm_progress']
                effect=('rescues' if l['success'] and not r['success'] else 'harms' if r['success'] and not l['success']
                    else 'both_succeed_faster' if l['success'] and l['frames']<r['frames'] else 'both_succeed_slower' if l['success'] and l['frames']>r['frames']
                    else 'both_succeed_equal' if l['success'] else 'both_fail')
                rf=validation_scores[0]['forecast'];lf=validation_scores[1]['forecast']
                keys=('peak_out_of_bounds','predicted_goal_exits','final_inside_goal','final_centroid_distance','final_max_spread','predicted_splitting_frames')
                first_metric=next(k for k in keys if rf[k]!=lf[k])
                episodes.append(dict(request_id=rid,index=index,validation_frame=ranked['frame'],adoption_frame=frame,
                    snapshot_age_frames=bridge,mode=str(g.llm_selected_mode.iloc[0]),target=None if pd.isna(g.llm_selected_target_sheep_id.iloc[0]) else int(g.llm_selected_target_sheep_id.iloc[0]),
                    actual_active_frames=len(g),first_rank_difference=dict(metric=first_metric,rule=rf[first_metric],llm=lf[first_metric]),
                    original_validation_candidates=validation_scores,adoption_queries=queries,rule_tail=r,llm_tail=l,isolated_effect=effect))
            results.append(dict(seed=seed,outcome='rescue' if seed in rescues else 'regression',actual_success=row['Success'],actual_frames=row['Frames'],
                rule_success=rule_row['Success'],rule_frames=rule_row['Frames'],episodes=episodes))
            print(json.dumps(dict(event='case_reviewed',seed=seed,episodes=len(episodes),effects=[e['isolated_effect'] for e in episodes])),flush=True)
        validation.frozen();assert digest(__file__)==read(output/'protocol.json')['review_source_sha256']
        save(output/'summary.json',dict(at_utc=utc_now(),cases=results,original_forecasts_reproduced=reproduced,actual_prefix_position_frames_checked=prefix_ticks,
            api_calls=0,live_controller_changed=False))
        save(output/'completion.json',dict(at_utc=utc_now(),complete=True,core_audit_passed=True,api_calls=0,model_started=False,
            all_first_rule_tails_reproduced=True,all_last_action_tails_reproduced=True,all_actual_prefixes_reproduced=True,
            reference_completion_unchanged=digest(reference/'completion.json')==read(output/'protocol.json')['reference_completion_sha256'],
            evidence_sha256={n:digest(output/n) for n in ('protocol.json','summary.json')}))
    except BaseException as exc:
        save(output/'failure.json',dict(at_utc=utc_now(),type=type(exc).__name__,reason=str(exc)[-1200:]));raise


if __name__=='__main__':main()
