"""Matched local execution diagnostics: equal virtual frame age, fresh responses.

This is a diagnostic cohort selected from previous outcomes, not a held-out trial.
Only an explicitly supplied local vLLM endpoint/model is used. No cloud clients.
"""
import argparse
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from experiment_audit import source_hashes, utc_now, write_json_exclusive
from high_level_controller import HighLevelController
from responses_api_backend import DeepSeekBackend, OpenAIBackend
from run_experiments import run_case, temporary_config, write_summary
from vllm_backend import VLLMBackend

SEEDS = (6102, 6124, 6125, 6129, 6134, 6135, 6140)
HISTORICAL_RESCUES = (6102, 6129, 6134, 6135)
HISTORICAL_REGRESSIONS = (6124, 6125, 6140)
MODEL = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
REFERENCE = ROOT / 'logs/experiments/qwen36_paired50_20261016'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class FixedAgeVLLMBackend(VLLMBackend):
    """Hold results until due; pause virtual time at due if the API is slower.

    Retain backend's real-time accounting. This is a decision-quality experiment,
    not a claim that the service actually responds within `frame_age` frames.
    """
    def __init__(self, *args, frame_age=23, wait_seconds=120., **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_age = frame_age
        self.wait_seconds = wait_seconds
        self.delivery_frames = {}
        self.recorded_results = []

    def submit(self, request):
        self.delivery_frames[request.request_id] = request.trigger_frame + self.frame_age
        return super().submit(request)

    def poll(self, request_id, frame):
        due = self.delivery_frames.get(request_id)
        if due is None or frame < due:
            return None
        if frame != due:
            raise RuntimeError('Fixed-age delivery frame was skipped')
        with self._lock:
            future = self._pending.get(request_id)
        if future is None:
            return None
        started = time.perf_counter()
        try:
            future.result(timeout=self.wait_seconds)
        except FutureTimeout:
            self.record_event('diagnostic_wait_timeout', request_id, frame=frame)
            raise RuntimeError('Local request exceeded diagnostic timeout; run excluded') from None
        result = super().poll(request_id, frame)
        assert result is not None
        self.record_event('fixed_age_delivery', request_id, frame=frame, due_frame=due,
                          paused_wall_seconds=time.perf_counter()-started)
        self.delivery_frames.pop(request_id, None)
        if result.get('backend_error'):
            # Keep detailed sanitized result in its request journal; do not label
            # a transport/schema failure as a model-quality failure.
            raise RuntimeError('Local backend error; partial simulation excluded; inspect request journal')
        return result

    def _execute_timed(self, *args):
        result = super()._execute_timed(*args)
        self.recorded_results.append(result)
        return result

    def cancel(self, request_id):
        self.delivery_frames.pop(request_id, None)
        return super().cancel(request_id)

    def close(self):
        super().close()
        # Drain canceled-but-running requests between runs. No cross-arm queues.
        self.executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self.client, 'close', None)
        if close:
            close()
        if any(r.get('backend_error') for r in self.recorded_results):
            raise RuntimeError('At least one local request failed; exclude this run')


def initial_state(path):
    df = pd.read_parquet(path)
    return df.loc[df.Frame == 0, ['Agent_Type','Agent_ID','X','Y','Status']].reset_index(drop=True)


def cloud_forbidden(*args, **kwargs):
    raise AssertionError('Cloud APIs are disabled for this diagnostic')


def check_source(protocol):
    assert source_hashes(ROOT) == protocol['source_sha256'], 'Source changed during frozen diagnostics'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol['runner_sha256']


def analyze(out, rows, rule, seeds=SEEDS):
    results = {}
    for seed in seeds:
        arms = {r['Arm']:r for r in rows if r['Seed'] == seed}
        baseline = rule.loc[seed]
        results[str(seed)] = dict(
            historical_role=('rescue' if seed in HISTORICAL_RESCUES else
                             'regression' if seed in HISTORICAL_REGRESSIONS else 'other_benchmark_seed'),
            rule=dict(success=bool(baseline.Success), frames=int(baseline.Frames)),
            **{name:dict(success=bool(r['Success']),frames=int(r['Frames']),
                final_inside=int(r['FinalInsideGoal']),goal_exits=int(r['GoalExits']),
                requests=int(r['HighLevelRequests']),accepted=int(r['HighLevelAccepted']),
                delegated=int(r['HighLevelDelegated']),rejected=int(r['HighLevelRejected']))
               for name,r in arms.items()},
        )
    totals = {}
    for name in ('legacy','explicit'):
        selected = [r for r in rows if r['Arm'] == name]
        totals[name] = dict(successes=sum(r['Success'] for r in selected),runs=len(selected),
            mean_capped_frames=float(np.mean([r['Frames'] for r in selected])),
            rescued_rule_failures=[r['Seed'] for r in selected if r['Success'] and not rule.loc[r['Seed'],'Success']],
            lost_rule_successes=[r['Seed'] for r in selected if not r['Success'] and rule.loc[r['Seed'],'Success']],
            delegated_responses=sum(r['HighLevelDelegated'] for r in selected))
    better = [seed for seed in seeds if results[str(seed)]['explicit']['success'] and not results[str(seed)]['legacy']['success']]
    worse = [seed for seed in seeds if results[str(seed)]['legacy']['success'] and not results[str(seed)]['explicit']['success']]
    summary = dict(totals=totals, explicit_rescues_vs_fresh_legacy=better,
        explicit_regressions_vs_fresh_legacy=worse, seeds=results, paid_api_calls=0,
        limitation='Previously examined benchmark seeds, one fresh realization per arm. Two changes bundled: explicit intent and release-aware forecasts. Fixed virtual age23; real API time is separately recorded. No generalization or real-time success claim.')
    atomic_json(out/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--frame-age', type=int, default=23)
    parser.add_argument('--full50', action='store_true',
                        help='Fresh paired runs on every seed 6101-6150; never reuse the prior seven.')
    parser.add_argument('--order-offset', type=int, choices=(0, 1), default=0,
                        help='Reverse which arm runs first for repeated diagnostic rounds.')
    args = parser.parse_args(argv)
    from urllib.parse import urlsplit
    url = urlsplit(args.base_url)
    if url.hostname not in ('127.0.0.1','localhost') or url.username or url.password or url.query or url.fragment:
        parser.error('Only an uncredentialed localhost vLLM URL is accepted')
    if args.frame_age != 23:
        parser.error('This frozen diagnostic requires frame age23')
    seeds = tuple(range(6101,6151)) if args.full50 else SEEDS
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with urlopen(args.base_url.rstrip('/')+'/models', timeout=5) as response:
        available = [m['id'] for m in json.load(response)['data']]
    assert available == [MODEL], 'Use the exact retained model from the paired50 study'
    audit = json.loads((REFERENCE/'compatibility.json').read_text())
    historical_meta = json.loads((Path(audit['historical_source'])/'mode_target/summary.csv.metadata.json').read_text())
    case = historical_meta['case_settings'][0]
    rule = pd.read_csv(REFERENCE/'rule_reference.csv').set_index('Seed')
    rule.loc[list(seeds)].reset_index().to_csv(out/'rule_reference.csv', index=False)
    if args.full50:
        pd.read_csv(REFERENCE/'deepseek_reference.csv').to_csv(out/'historical_deepseek_reference.csv',index=False)
        pd.read_csv(REFERENCE/'spark/summary.csv').to_csv(out/'historical_spark_reference.csv',index=False)
    overrides = {k:v for k,v in audit['frozen_config'].items() if hasattr(config,k)}
    overrides.update(LLM_CACHE_ENABLED=False, LOG_RETENTION_ENABLED=False,
        VLLM_MODEL=MODEL,VLLM_BASE_URL=args.base_url, VLLM_INPUT_FORMAT='json',
        VLLM_COMPACT_OUTPUT=False,VLLM_PREFIX_PRIMING=False,VLLM_PREFIX_TOKEN_FILE='',
        VLLM_MAX_OUTPUT_TOKENS=audit['frozen_config']['LLM_MAX_OUTPUT_TOKENS'],
        HIGH_LEVEL_MODE_SELECTION_ENABLED=True,HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,
        SAVE_TRAJECTORY_PNG=False,SAVE_ANIMATION_MP4=False,
        LLM_PENDING_POLL_SECONDS=0.)
    protocol = dict(created_at_utc=utc_now(),seeds=list(seeds),frame_age=23,max_steps=2000,
        model=MODEL,weight_and_serving_recipe='retained NVFP4, MTP3, main Marlin/draft Triton',
        case=case, source_sha256=source_hashes(ROOT),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fresh_arms=['legacy','explicit'],rule_reference='existing exact-replayed rule results',
        temperature=0,thinking=False,input_format='json',compact_output=False,
        arm_order_offset=args.order_offset,
        prefix_priming=False,decision_cache=False,config=overrides,paid_api_calls=0,
        limitations=['Previously examined benchmark seeds, not held-out evaluation.',
                    'One fresh provider realization in each arm; sampling variation remains.',
                    'Controlled virtual frame age, not real-time deployment performance.',
                    'Explicit intent plus execution-aware forecasts evaluated as one package.'])
    write_json_exclusive(out/'protocol.json',protocol)
    source = out/'source'
    source.mkdir()
    for name in protocol['source_sha256']:
        (source/name).write_bytes((ROOT/name).read_bytes())
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows = []
    # Alternate which arm runs first across seeds; never run GPU requests in parallel.
    order = [(seed, arm) for index,seed in enumerate(seeds)
             for arm in (('legacy','explicit') if (index+args.order_offset)%2 == 0 else ('explicit','legacy'))]
    for seed,arm in order:
        check_source(protocol)
        folder=out/arm/str(seed)
        folder.mkdir(parents=True)
        instances=[]
        def factory(kind='rule', settings=config):
            assert kind == 'vllm'
            backend=FixedAgeVLLMBackend(settings=settings,base_url=args.base_url,model=MODEL,
                compact_output=False,prefix_priming=False,input_format='json',frame_age=23,
                cache_path=folder/'unused_cache.jsonl')
            instances.append(backend)
            return HighLevelController(backend=backend,settings=settings)
        scoped = dict(overrides, HIGH_LEVEL_EXPLICIT_EXECUTION=(arm=='explicit'),
            DATA_LOG_DIR=str(folder/'data'),CONTROL_LOG_DIR=str(folder/'control'),
            VISUAL_LOG_DIR=str(folder/'visuals'),VIDEO_LOG_DIR=str(folder/'videos'),
            ANALYSIS_LOG_DIR=str(folder/'analysis'))
        started=time.perf_counter()
        print(json.dumps(dict(event='start',seed=seed,arm=arm)),flush=True)
        try:
            with temporary_config(**scoped), ExitStack() as stack:
                stack.enter_context(patch('run_experiments.build_controller',factory))
                stack.enter_context(patch.object(DeepSeekBackend,'_make_client',cloud_forbidden))
                stack.enter_context(patch.object(OpenAIBackend,'_make_client',cloud_forbidden))
                row=run_case(case,seed,2000,100000,controller_kind='vllm',llm_cache_enabled=False)
            pd.testing.assert_frame_equal(initial_state(row['Data']),initial_state(rule.loc[seed,'Data']),check_exact=True)
            control=pd.read_parquet(row['Control'])
            consumed=control.loc[control.llm_latency_frames.notna()].drop_duplicates('llm_request_id')
            assert set(consumed.llm_latency_frames)=={23} or consumed.empty
            assert row['HighLevelCacheHits']==0
            assert set(control.llm_model.dropna()) <= {MODEL}
            # Count/journal ALL calls, including those canceled before consumption.
            requests=[r for backend in instances for r in backend.recorded_results]
            assert not any(r.get('backend_error') for r in requests)
            atomic_json(folder/'request_statistics.json',dict(calls=len(requests),
                consumed=len(consumed),cache_hits=sum(r.get('cache_hit',False) for r in requests),
                mean_worker_seconds=float(np.mean([r['latency_seconds'] for r in requests])) if requests else None,
                mean_queue_seconds=float(np.mean([r['queue_seconds'] for r in requests])) if requests else None,
                mean_submit_to_ready_seconds=float(np.mean([r['submit_to_ready_seconds'] for r in requests])) if requests else None,
                input_tokens=sum(r.get('input_tokens') or 0 for r in requests),
                output_tokens=sum(r.get('output_tokens') or 0 for r in requests),
                all_consumed_at_fixed_frame_age=True))
            row['Arm']=arm
            row['WallSeconds']=time.perf_counter()-started
            # write_summary retains the project's schema; extra arm fields live
            # in the study CSV rather than changing production summary behavior.
            rows.append(row)
            pd.DataFrame(rows).to_csv(out/'runs.csv',index=False)
            atomic_json(out/'progress.json',dict(completed=len(rows),planned=len(order),last_seed=seed,last_arm=arm,
                paid_api_calls=0,source_frozen=True))
            print(json.dumps(dict(event='completed',seed=seed,arm=arm,success=row['Success'],frames=row['Frames'],
                accepted=row['HighLevelAccepted'],delegated=row['HighLevelDelegated'],
                rejected=row['HighLevelRejected'],calls=len(requests))),flush=True)
        except Exception as exc:
            atomic_json(out/'failure.json',dict(seed=seed,arm=arm,type=type(exc).__name__,
                diagnostic='Partial run excluded; inspect its request journal. Do not mix with completed results.'))
            raise
    check_source(protocol)
    summary=analyze(out,rows,rule,seeds)
    print(json.dumps(dict(event='complete',totals=summary['totals'],
        rescues=summary['explicit_rescues_vs_fresh_legacy'],regressions=summary['explicit_regressions_vs_fresh_legacy'])),flush=True)


if __name__=='__main__':
    main()
