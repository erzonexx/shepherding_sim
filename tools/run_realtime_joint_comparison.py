"""Paired rule/joint-Qwen supplement on the early DeepSeek100 scenes at actual30Hz.

No change to the completed cohort or its frozen sources. This module configures
the existing runner in its own process; no global default configuration is edited.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import config
from experiment_audit import utc_now
from run_experiments import temporary_config
from realtime_joint_controller import AsyncJointController, VERSION
from tools import run_realtime_comparison as runner

REFERENCE = ROOT/'logs/experiments/realtime200_30hz_20261018'
OUT = ROOT/'logs/experiments/realtime_joint100_30hz_20261019'
HISTORICAL = ROOT/'logs/experiments/paired_100_20261008/3_1_permissions/20261008_191455_311061'
ARMS = ('rule', 'qwen')
BASE = 'http://127.0.0.1:18033/v1'
ADDITIONS = ('realtime_joint_controller.py', 'tools/run_realtime_joint_comparison.py')
read, save, digest = runner.read, runner.save, runner.digest
original_hashes = runner.hashes


def hashes():
    return dict(original_hashes(), **{name: digest(ROOT/name) for name in ADDITIONS})


def check_frozen(protocol):
    assert hashes() == protocol['source_sha256'], 'Frozen source changed; preserve partial evidence'


@contextmanager
def joint_config(**values):
    # The reused runner has a target-only default. Explicitly choose the existing
    # original joint contract for this supplement, including the stored manifest.
    values['HIGH_LEVEL_TARGET_HANDOFF_ENABLED'] = False
    with temporary_config(**values):
        yield


def configure():
    runner.OUT, runner.BASE, runner.ARMS = OUT, BASE, ARMS
    runner.AsyncTargetController = AsyncJointController
    runner.VERSION = VERSION
    runner.temporary_config = joint_config
    runner.check_frozen = check_frozen


def prepare():
    assert not OUT.exists(), 'Do not overwrite or restart a cohort'
    reference = read(REFERENCE/'protocol.json')
    assert read(REFERENCE/'completion.json')['complete']
    assert original_hashes() == reference['source_sha256'], 'Prior core sources differ'
    historical = read(HISTORICAL/'permission_protocol.json')
    assert historical['seeds'] == list(range(6101, 6201))
    assert historical['population'] == [5,5,5,5] and historical['max_steps']==2000
    cfg = dict(reference['config'], HIGH_LEVEL_MODE_SELECTION_ENABLED=True,
               HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,
               HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False, VLLM_BASE_URL=BASE)
    protocol = dict(reference, at_utc=utc_now(),
        name='Measured original joint mode-target versus rule supplement',
        arms=list(ARMS), simulations=200, seeds=historical['seeds'], config=cfg, source_sha256=hashes(),
        execution_version=VERSION, reference=str(REFERENCE),
        reference_protocol_sha256=digest(REFERENCE/'protocol.json'),
        reference_completion_sha256=digest(REFERENCE/'completion.json'),
        seed_policy='Intentional replication of early DeepSeek100 initial scenes, not new generalization',
        historical_reference=str(HISTORICAL),
        historical_protocol_sha256=digest(HISTORICAL/'permission_protocol.json'),
        primary_comparison='New joint mode-target versus newly rerun rule, both30Hz',
        execution='Two independent lanes; one scene per lane, one local API request at a time',
        original_handoff='Original joint mode-target holds/releases; target-only handoff-v1 disabled',
        current_fast_checks=['both alarms for admission', 'offered target and current eligibility',
                             'current drive-point geometry', 'response and validation age'],
        authority_note='Joint versus historical target-only handoff is not a one-switch ablation; original joint forecast and execution semantics are retained.',
        limits=reference['limits'] + [
            'Previously tested seeds6101-6200; this is a missing-authority supplement, not independent generalization.',
            'Early DeepSeek100 did not use this measured30Hz pipeline; historical model results are contextual, not a controlled backend-only comparison.',
            'Historical target-only policy used handoff-v1, this joint policy retains original joint execution.'])
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json', protocol)
    source = OUT/'source'
    source.mkdir()
    for name in protocol['source_sha256']:
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    save(OUT/'preparation.json', dict(at_utc=utc_now(), state='awaiting_spark_connection',
        formal_scenes_started=0, model_calls=0, planned_simulations=200,
        planned_seeds=[6101,6200], paid_calls=0))
    print('Prepared rule100 and joint-Qwen100; no formal scenes or model calls started.', flush=True)


def lane(arm):
    configure()
    protocol=read(OUT/'protocol.json')
    rows=[]
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) if arm!='rule' else None
    try:
        for seed in protocol['seeds']:
            try:
                row=runner.run_one(protocol,seed,arm,executor=pool)
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/(arm+'_runs.csv'),index=False)
                progress=dict(at_utc=utc_now(),arm=arm,completed=len(rows),planned=len(protocol['seeds']),
                              successes=sum(r['Success'] for r in rows),last_seed=seed)
                temp=OUT/(arm+'_progress.tmp')
                temp.write_text(__import__('json').dumps(progress),encoding='utf-8')
                temp.replace(OUT/(arm+'_progress.json'))
                print(__import__('json').dumps(dict(event='scene_completed',**progress)),flush=True)
            except Exception as exc:
                save(OUT/(arm+'_failure.json'),dict(seed=seed,arm=arm,type=type(exc).__name__,
                    reason='Stopped; evidence retained; no automatic retry'))
                (OUT/'stop_requested').touch()
                raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
    return arm


def summarize():
    protocol = read(OUT/'protocol.json')
    check_frozen(protocol)
    rows = [read(OUT/a/str(s)/'completed_row.json') for s in protocol['seeds'] for a in ARMS]
    totals = {}
    for arm in ARMS:
        selected = [r for r in rows if r['Arm']==arm]
        totals[arm] = dict(runs=len(selected), successes=sum(r['Success'] for r in selected),
            mean_capped_frames=sum(r['Frames'] for r in selected)/len(selected),
            step_overruns=sum(r['StepOverruns'] for r in selected),
            late_ticks_over_period=sum(r['LateTicksOverPeriod'] for r in selected))
    index = {(r['Seed'],r['Arm']):r for r in rows}
    pairs = [dict(Seed=s, **{a+'_success':index[s,a]['Success'] for a in ARMS},
                  **{a+'_frames':index[s,a]['Frames'] for a in ARMS}) for s in protocol['seeds']]
    paired = dict(rescues=[r['Seed'] for r in pairs if r['qwen_success'] and not r['rule_success']],
                  regressions=[r['Seed'] for r in pairs if not r['qwen_success'] and r['rule_success']])
    pd.DataFrame(rows).to_csv(OUT/'runs200.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUT/'comparison100.csv', index=False)
    save(OUT/'summary.json', dict(at_utc=utc_now(), totals=totals, paired=paired, paid_calls=0))
    save(OUT/'runs_complete.json', dict(at_utc=utc_now(), simulations=200, source_frozen=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--lane', choices=ARMS)
    args = parser.parse_args()
    configure()
    if args.prepare:
        prepare()
    elif args.lane:
        lane(args.lane)
    elif args.smoke:
        row = runner.run_one(read(OUT/'protocol.json'),8001,'qwen',smoke=True,frame_limit=350)
        assert row['HighLevelRequests'] > 0
        assert row['MaxStepSeconds'] < 1/30
        events = [__import__('json').loads(line) for line in Path(row['RequestsAudit']).read_text().splitlines()]
        answers = [e['result']['response'] for e in events
                   if e['event'] in ('selection_ready','discarded_background_result')
                   and e.get('result',{}).get('response')]
        assert answers and all(r.get('served_model')==runner.MODEL and not r.get('backend_error') for r in answers)
        save(OUT/'smoke_passed.json', dict(joint_pipeline_exercised=True, actual_model_verified=True))
    elif args.run:
        assert read(OUT/'model_identity.json')['exact_retained_recipe_verified']
        assert read(OUT/'smoke_passed.json')['joint_pipeline_exercised']
        assert not (OUT/'stop_requested').exists()
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(lane,a) for a in ARMS]
            for future in futures:
                future.result()
        summarize()
    else:
        parser.error('Choose --prepare, --smoke, --run or --lane')


if __name__ == '__main__':
    main()
