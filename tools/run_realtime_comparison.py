"""Wall-clock-paced three-arm comparison; no planner waits in a physical tick.

One lane per arm, one active scene per lane. Qwen requests and both forecasting
stages execute in a separate worker process. All timing is measured rather than
injected. Run --prepare, a smoke check, then --run after the retained model is ready.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import config
from detector import Detector
from environment import SwarmEnv
from experiment_audit import source_hashes,config_snapshot,utc_now,begin_run_record,finish_run_record,write_json_exclusive
from low_level_controller import LowLevelController
from realtime_clock import RealtimeClock
from realtime_controller import AsyncPredictionBackend,AsyncTargetController,MODEL,VERSION
from run_experiments import temporary_config,summarize_case

OUT=ROOT/'logs/experiments/realtime200_30hz_20261018'
REFERENCE=ROOT/'logs/experiments/selector_validation100_age23_handoffv1_20261018'
EXTRA=('forecast_selector.py','realtime_controller.py','realtime_clock.py',
       'tools/run_realtime_comparison.py')
ARMS=('rule','selector','qwen')
BASE='http://127.0.0.1:18032/v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):write_json_exclusive(path,obj)
def hashes():return dict(source_hashes(ROOT),**{n:digest(ROOT/n) for n in EXTRA})


def prepare():
    assert not OUT.exists(),'Preserve an existing cohort'
    ref=read(REFERENCE/'protocol.json')
    assert read(REFERENCE/'completion.json')['complete']
    # Preserve the existing motion model and target-only handoff implementation.
    assert source_hashes(ROOT)==ref['source_sha256']
    used=set();records=[]
    def collect(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k.lower() in ('seed','random_seed') and type(v) is int:used.add(v)
                elif k.lower()=='seeds' and isinstance(v,list):used.update(x for x in v if type(x) is int)
                elif isinstance(v,(dict,list)):collect(v)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):collect(v)
    for path in (ROOT/'logs/experiments').rglob('*.json'):
        try:collect(read(path));records.append(str(path.relative_to(ROOT)))
        except (OSError,ValueError,UnicodeError):continue
    for path in (ROOT/'logs/experiments').rglob('*.csv'):
        try:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream)
                key=next((k for k in reader.fieldnames or [] if k.lower()=='seed'),None)
                if key:used.update(int(float(row[key])) for row in reader if row.get(key))
            records.append(str(path.relative_to(ROOT)))
        except (OSError,ValueError,UnicodeError):continue
    seeds=[];n=6701
    while len(seeds)<200:
        if n not in used:seeds.append(n)
        n+=1
    cfg=dict(ref['config'],NUM_SHEEP=20,NUM_ABNORMAL_A=5,NUM_ABNORMAL_B=5,NUM_ABNORMAL_C=5,
        VLLM_BASE_URL=BASE,VLLM_MODEL=MODEL,HIGH_LEVEL_TARGET_HANDOFF_ENABLED=True,
        HIGH_LEVEL_MODE_SELECTION_ENABLED=False,HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,
        LLM_PENDING_POLL_SECONDS=0.0,LLM_CACHE_ENABLED=False,LOG_RETENTION_ENABLED=False)
    OUT.mkdir(parents=True)
    protocol=dict(at_utc=utc_now(),name='Measured asynchronous realtime comparison200 at30Hz',
        seeds=seeds,arms=list(ARMS),simulations=600,hz=30,period_seconds=1/30,
        case=ref['case'],config=cfg,source_sha256=hashes(),model=MODEL,
        recipe='Retained NVFP4; MTP3; main Marlin/draft Triton',hold_frames=30,
        artificial_response_age=None,response_age_limit_frames=cfg['HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES'],
        validation_snapshot_max_age_frames=30,validation_snapshot_max_age_seconds=1.0,
        execution='Independent arm lanes; one scene per arm; one Qwen scene and one API request at a time. Separate process for each intervention lane background worker.',
        pipeline='Snapshot -> background request forecasts -> selector or local Qwen -> background revalidation -> current fast checks.',
        current_fast_checks=['both alarms for admission','current rule mode','target still eligible',
            'current rule delegation','current drive-point boundary','request age','validation snapshot age'],
        original_handoff='Hold/release and forecast filters retained; expensive revalidation uses an explicitly bounded-age background snapshot.',
        lateness_policy='Never wait for a planner or skip a physical step; keep absolute tick deadlines and record all scheduler lateness and step overruns.',
        primary_metric='All20 sheep reach goal by2000 frames',
        secondary_metrics=['Capped frames, failed2000','Paired rescues/regressions','Goal exits',
            'Raw per-tick timing and lateness','Prediction/API/revalidation/end-to-end durations',
            'Expired/canceled/discarded replies'],paid_calls=0,cache=False,thinking=False,
        stopping_rule='Complete all200 scenes per arm regardless of scores; stop only for provenance, worker, model, IO or transport failure. No selective retry or retuning.',
        limits=['30Hz is a declared simulated time scale, not a calibrated real flock model.',
            'Windows scheduling is soft realtime; every deadline overrun is reported, not hidden.',
            'No exact same-frame expensive forecast: current fast checks plus background validation age<=30frames.',
            'Actual timing causes run-to-run variability even with the same seed.',
            'Parallel independent arm lanes share the Windows host; one GPU model only.',
            'Old fixed-age/unpaced results are not pooled with this experiment.'])
    save(OUT/'protocol.json',protocol)
    save(OUT/'seed_preflight.json',dict(seeds=seeds,overlap=[],records_scanned=len(records),records=records))
    source=OUT/'source';source.mkdir()
    for name in EXTRA:shutil.copyfile(ROOT/name,source/Path(name).name)
    print(json.dumps(dict(event='registered',first_seed=seeds[0],last_seed=seeds[-1],simulations=600,hz=30)),flush=True)


def check_frozen(protocol):
    assert hashes()==protocol['source_sha256'],'Experiment source changed'


def run_one(protocol,seed,arm,executor=None,smoke=False,frame_limit=None):
    check_frozen(protocol)
    folder=(OUT/'smoke'/arm/str(seed)) if smoke else (OUT/arm/str(seed))
    if (folder/'completed_row.json').exists():return read(folder/'completed_row.json')
    folder.mkdir(parents=True,exist_ok=False)
    scoped={k:v for k,v in protocol['config'].items() if hasattr(config,k)}
    scoped.update(USE_FIXED_SEED=True,RANDOM_SEED=seed,MAX_STEPS=frame_limit or 2000,
        HIGH_LEVEL_TARGET_HANDOFF_ENABLED=arm!='rule',CONTROLLER_KIND='rule' if arm=='rule' else 'vllm',
        SAVE_TRAJECTORY_PNG=False,SAVE_ANIMATION_MP4=False,
        DATA_LOG_DIR=str(folder/'data'),CONTROL_LOG_DIR=str(folder/'control'),
        ANALYSIS_LOG_DIR=str(folder/'analysis'),VISUAL_LOG_DIR=str(folder/'visuals'),VIDEO_LOG_DIR=str(folder/'video'))
    backend=None;env=None;clock=None
    with temporary_config(**scoped):
        if arm=='rule':controller=LowLevelController()
        else:
            backend=AsyncPredictionBackend(config,kind=arm,base_url=BASE,executor=executor)
            controller=AsyncTargetController(backend,settings=config,
                validation_max_age_frames=protocol['validation_snapshot_max_age_frames'])
            backend.warm()
        env=SwarmEnv(controller)
        stamp='realtime_'+str(seed)+'_'+arm
        manifest_path=folder/'run.json'
        manifest=begin_run_record(manifest_path,controller,config,case=protocol['case'],seed=seed,
            controller_kind=arm,execution_version=VERSION,control_hz=protocol['hz'],
            validation_snapshot_age_limit=protocol['validation_snapshot_max_age_frames'])
        detector=Detector(config.DANGER_DISPERSION_THRESHOLD,config.DANGER_MEAN_SPREAD_THRESHOLD,
            config.DANGER_STAGNATION_FRAMES,config.DANGER_STAGNATION_THRESHOLD)
        reached=False;frame=0;wall=[]
        try:
            clock=RealtimeClock(protocol['hz'])
            for frame in range(frame_limit or 2000):
                if not smoke and (OUT/'stop_requested').exists():
                    raise RuntimeError('Another lane failed; stopping all arms')
                tick=clock.tick(frame);wall.append(tick)
                alarms,metrics,_=detector.detect(env.sheep_pos,env.goal_pos)
                env.record_frame_metrics(metrics)
                reached=bool(env.step(alarms))
                clock.record_step(tick)
                if reached:break
            physical_end=time.perf_counter()
            if backend:backend.record_event('physical_scene_finished',None,frame=env.current_frame,
                elapsed_seconds=physical_end-clock.start,success=reached)
        finally:
            # First mark physical time ended, then cancel/drain; late answers cannot apply.
            if clock is not None:
                np.savez_compressed(folder/'clock.npz',tick_perf_counter=np.array(wall),
                    lateness_seconds=np.array(clock.lateness),step_seconds=np.array(clock.step_seconds),
                    origin_perf_counter=np.array(clock.start))
            if env is not None:env.close()
        env.save_data_to_parquet(stamp);env.save_control_to_parquet(stamp)
        data=Path(config.DATA_LOG_DIR)/('data_'+stamp+'.parquet')
        control=Path(config.CONTROL_LOG_DIR)/('control_'+stamp+'.parquet')
        row=summarize_case(protocol['case'],seed,stamp,data,control,Path('unused_video'),Path('unused_png'),
            reached,env.current_frame,env.goal_pos)
        duration=float(wall[-1]-clock.start+clock.step_seconds[-1])
        row.update(Arm=arm,Controller=arm,RunMetadata=str(manifest_path),
            RequestsAudit=manifest.get('requests_audit',''),PhysicalWallSeconds=duration,
            ControlHz=protocol['hz'],MaxStepSeconds=max(clock.step_seconds),
            StepOverruns=sum(t>clock.period for t in clock.step_seconds),
            MaxLatenessSeconds=max(clock.lateness),LateTicksOverPeriod=sum(t>clock.period for t in clock.lateness),
            Timing=str(folder/'clock.npz'))
        finish_run_record(manifest_path,manifest,success=reached,frames=env.current_frame,
            physical_wall_seconds=duration,data=str(data),control=str(control))
        save(folder/'completed_row.json',row)
        return row


def lane(arm):
    protocol=read(OUT/'protocol.json');rows=[]
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) if arm!='rule' else None
    try:
        for seed in protocol['seeds']:
            try:
                row=run_one(protocol,seed,arm,executor=pool);rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/(arm+'_runs.csv'),index=False)
                progress=dict(at_utc=utc_now(),arm=arm,completed=len(rows),planned=200,
                    successes=sum(r['Success'] for r in rows),last_seed=seed)
                temp=OUT/(arm+'_progress.tmp');temp.write_text(json.dumps(progress),encoding='utf-8')
                temp.replace(OUT/(arm+'_progress.json'))
                print(json.dumps(dict(event='scene_completed',**progress)),flush=True)
            except Exception as exc:
                save(OUT/(arm+'_failure.json'),dict(seed=seed,arm=arm,type=type(exc).__name__,
                    reason='Stopped; partial evidence preserved; no selective retry.'))
                (OUT/'stop_requested').touch()
                raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
    return arm


def summarize():
    protocol=read(OUT/'protocol.json');check_frozen(protocol)
    rows=[read(OUT/a/str(s)/'completed_row.json') for s in protocol['seeds'] for a in ARMS]
    pd.DataFrame(rows).to_csv(OUT/'runs600.csv',index=False)
    totals={};comparisons=[]
    for arm in ARMS:
        selected=[r for r in rows if r['Arm']==arm]
        totals[arm]=dict(runs=200,successes=sum(r['Success'] for r in selected),
            mean_capped_frames=sum(r['Frames'] for r in selected)/200,
            mean_goal_exits=sum(r['GoalExits'] for r in selected)/200,
            step_overruns=sum(r['StepOverruns'] for r in selected),
            late_ticks_over_period=sum(r['LateTicksOverPeriod'] for r in selected),
            max_step_seconds=max(r['MaxStepSeconds'] for r in selected))
    lookup={(r['Seed'],r['Arm']):r for r in rows}
    for seed in protocol['seeds']:
        c=dict(Seed=seed)
        for arm in ARMS:c[arm+'_success']=lookup[seed,arm]['Success'];c[arm+'_frames']=lookup[seed,arm]['Frames']
        comparisons.append(c)
    paired={}
    for a,b in (('selector','rule'),('qwen','rule'),('qwen','selector')):
        paired[a+'_vs_'+b]=dict(rescues=[c['Seed'] for c in comparisons if c[a+'_success'] and not c[b+'_success']],
            regressions=[c['Seed'] for c in comparisons if not c[a+'_success'] and c[b+'_success']])
    pd.DataFrame(comparisons).to_csv(OUT/'comparison200.csv',index=False)
    save(OUT/'summary.json',dict(at_utc=utc_now(),totals=totals,paired=paired,paid_calls=0,limits=protocol['limits']))
    save(OUT/'runs_complete.json',dict(at_utc=utc_now(),simulations=600,source_frozen=True,paid_calls=0))
    print(json.dumps(dict(event='all600_complete',totals=totals,paired=paired)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--lane',choices=ARMS)
    parser.add_argument('--smoke',choices=('selector','qwen'))
    args=parser.parse_args()
    if args.prepare:prepare();return
    if args.lane:lane(args.lane);return
    if args.smoke:
        p=read(OUT/'protocol.json')
        row=run_one(p,8001,args.smoke,smoke=True,frame_limit=350)
        print(json.dumps(dict(event='smoke_complete',arm=args.smoke,frames=row['Frames'],
            requests=row['HighLevelRequests'],responses=row['HighLevelResponses'],
            max_step_seconds=row['MaxStepSeconds'],wall_seconds=row['PhysicalWallSeconds'])),flush=True)
        return
    if args.run:
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(lane,arm) for arm in ARMS]
            for future in futures:future.result()
        summarize()
        return
    parser.error('Choose --prepare, --run, --lane or --smoke')


if __name__=='__main__':main()
