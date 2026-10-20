"""Background prediction/selection/revalidation; the fast loop never waits on jobs.

The legacy controller remains the owner of admission, holds and handoff. Heavy
forecasts execute in a separate process on immutable snapshots. Main-loop checks
use the current mode, alarms, eligibility, drive geometry and a bounded-age
validation snapshot. An exact same-frame expensive forecast is not claimed.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from control_types import ControlObservation
from experiment_audit import utc_now
from forecast_selector import choose, VERSION as SELECTOR_VERSION
from high_level_controller import HighLevelController, HighLevelRequest
from low_level_controller import LowLevelController
from vllm_backend import VLLMBackend

MODEL = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
VERSION = 'async-target-handoff-v1'
_MODEL_BACKEND = None
_MODEL_KEY = None


class RealtimePipelineError(RuntimeError):
    pass


class Hold30VLLM(VLLMBackend):
    def _response_format(self, *args, **kwargs):
        fmt = copy.deepcopy(super()._response_format(*args, **kwargs))
        fmt['schema']['properties']['hold_frames'].update(minimum=30, maximum=30)
        return fmt


class NoRequests:
    source = 'background_forecast'
    trigger_on_concurrent_only = True


def public_settings(settings):
    # No environment or credentials are serialized into snapshots or journals.
    values = {}
    for key, value in vars(settings).items():
        if not key.isupper() or any(x in key for x in ('API_KEY', 'SECRET', 'PASSWORD', 'TOKEN')):
            # Output budgets are public configuration, unlike access tokens.
            if key not in ('VLLM_MAX_OUTPUT_TOKENS', 'LLM_MAX_OUTPUT_TOKENS'):
                continue
        if isinstance(value, (str, int, float, bool, type(None))):
            values[key] = value
    return values


def warm_worker():
    return multiprocessing.current_process().pid


def snapshot(controller, observation, summary):
    obs = ControlObservation(
        np.array(observation.sheep_positions, copy=True),
        np.array(observation.dog_position, copy=True),
        np.array(observation.goal_position, copy=True), tuple(observation.alarms),
        observation.frame, tuple(observation.sheep_statuses), observation.current_mode,
        None if observation.sheep_velocities is None else np.array(observation.sheep_velocities, copy=True),
        None if observation.dog_velocity is None else np.array(observation.dog_velocity, copy=True))
    return dict(observation=obs, summary=copy.deepcopy(summary),
        low={k:copy.deepcopy(v) for k,v in vars(controller.low_level_controller).items() if k!='settings'},
        history=list(controller.alarm_distance_history))


def predictor(state, values):
    settings = SimpleNamespace(**values)
    low = LowLevelController(settings)
    for key, value in state['low'].items():
        setattr(low, key, copy.deepcopy(value))
    controller = HighLevelController(low, NoRequests(), settings)
    controller.alarm_distance_history.extend(state['history'])
    return controller


def selection_job(state, request, values, kind, base_url):
    global _MODEL_BACKEND, _MODEL_KEY
    started = time.perf_counter()
    controller = predictor(state, values)
    rich = controller._with_mode_forecasts(state['observation'], state['summary'], include_targets=True)
    enriched = replace(request, state_summary=rich)
    predicted = time.perf_counter()
    ranking = None
    if kind == 'selector':
        target, ranking = choose(rich)
        response = dict(request_id=request.request_id, trigger_frame=request.trigger_frame,
            mode=rich['rule_mode'], target_sheep_id=target, hold_frames=30,
            reason=SELECTOR_VERSION, cache_hit=False, prompt_version=SELECTOR_VERSION)
        backend_info = dict(source='forecast_selector')
    else:
        if kind != 'qwen':
            raise ValueError('Only selector and local Qwen are supported')
        key = (base_url, MODEL, json.dumps(values, sort_keys=True))
        if key != _MODEL_KEY:
            if _MODEL_BACKEND is not None:
                _MODEL_BACKEND.close()
                _MODEL_BACKEND.client.close()
            _MODEL_BACKEND = Hold30VLLM(settings=controller.settings, base_url=base_url,
                model=MODEL, compact_output=False, prefix_priming=False, input_format='json')
            _MODEL_KEY = key
        # Synchronous in the worker process only. No Future.result() in the fast loop.
        response = _MODEL_BACKEND._call_api(enriched, 'async-no-cache')
        backend_info = dict(source='vllm', model=MODEL,
            instructions_sha256=hashlib.sha256(_MODEL_BACKEND.instructions.encode()).hexdigest())
        if response.get('backend_error'):
            return dict(request=enriched, response=response, backend_error=True,
                        prediction_seconds=predicted-started, total_worker_seconds=time.perf_counter()-started)
        if response.get('served_model') != MODEL or response.get('hold_frames') != 30:
            raise RealtimePipelineError('Unexpected served model or hold value')
    return dict(request=enriched, response=response, ranking=ranking, backend_info=backend_info,
        prediction_seconds=predicted-started, selection_seconds=time.perf_counter()-predicted,
        total_worker_seconds=time.perf_counter()-started, worker_pid=multiprocessing.current_process().pid)


def validation_job(state, request, response, values):
    started = time.perf_counter()
    controller = predictor(state, values)
    controller.pending_request = request
    summary = controller._with_mode_forecasts(state['observation'], state['summary'], response=response)
    return dict(summary=summary, frame=state['observation'].frame,
                validation_seconds=time.perf_counter()-started,
                worker_pid=multiprocessing.current_process().pid)


class AsyncPredictionBackend:
    trigger_on_concurrent_only = True
    target_handoff_enabled = True
    cache_enabled = False
    prompt_version = VERSION

    def __init__(self, settings, kind='selector', base_url=None, executor=None,
                 select_fn=selection_job, validate_fn=validation_job):
        if kind not in ('selector', 'qwen'):
            raise ValueError('Unknown asynchronous selector')
        self.kind, self.settings = kind, settings
        self.source = 'forecast_selector' if kind=='selector' else 'vllm'
        self.model = MODEL if kind=='qwen' else None
        self.base_url = base_url
        self.values = public_settings(settings)
        self.executor = executor or ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'))
        self.owns_executor = executor is None
        self.select_fn, self.validate_fn = select_fn, validate_fn
        self.controller = None
        self.job = None
        self.audit_path = None
        self.events = []
        self.closed = False
        self.validation_result = None

    def warm(self):
        # Initialization occurs before a scene's physical clock starts.
        return self.executor.submit(warm_worker).result(timeout=30)

    def set_audit_path(self, path):
        self.audit_path = Path(path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open('x', encoding='utf-8'):
            pass

    def record_event(self, event, request_id, **fields):
        # Buffer only: no file IO or JSON serialization in the control loop.
        self.events.append(dict(event=event, request_id=request_id, at_utc=utc_now(), **fields))

    def _finish_discarded(self):
        if self.job and self.job['canceled'] and self.job['future'].done():
            job = self.job
            if not job['future'].cancelled():
                result = self._result(job['future'])
                self.record_event('discarded_background_result', job['request'].request_id,
                    stage=job['stage'], result=result)
            self.job = None

    @staticmethod
    def _result(future):
        try:
            result = future.result()  # Caller must have observed done().
        except Exception as exc:
            raise RealtimePipelineError('Background worker failed: '+type(exc).__name__) from None
        if result.get('backend_error'):
            raise RealtimePipelineError('Local model transport or schema failure; inspect sanitized journal')
        return result

    def can_submit(self):
        self._finish_discarded()
        return not self.closed and self.job is None

    def submit(self, request):
        if not self.can_submit():
            raise RealtimePipelineError('Bounded background slot is occupied')
        state = snapshot(self.controller, self.controller.latest_observation, request.state_summary)
        submitted = time.perf_counter()
        future = self.executor.submit(self.select_fn, state, request, self.values, self.kind, self.base_url)
        self.job = dict(request=request, future=future, stage='selection', canceled=False,
                        submitted=submitted, submitted_frame=request.trigger_frame)
        self.record_event('pipeline_submitted', request.request_id, trigger_frame=request.trigger_frame,
            observation_at_utc=request.observed_at_utc, submitted_perf_counter=submitted,
            state_summary=request.state_summary)

    def poll(self, request_id, frame):
        self._finish_discarded()
        job = self.job
        if job is None or job['canceled'] or job['request'].request_id!=request_id:
            return None
        if not job['future'].done():
            return None
        result = self._result(job['future'])
        if job['stage']=='selection':
            enriched, response = result['request'], result['response']
            self.controller.pending_request = enriched
            job.update(request=enriched, response=response)
            self.record_event('selection_ready', request_id, consumed_frame=frame, result=result)
            state = snapshot(self.controller, self.controller.latest_observation, self.controller.latest_summary)
            job.update(stage='validation', validation_frame=frame,
                future=self.executor.submit(self.validate_fn, state, enriched, response, self.values))
            return None
        self.validation_result = result
        response = dict(job['response'])
        response.update(consumed_at_utc=utc_now(), consumed_frame=frame,
            latency_frames=frame-job['submitted_frame'],
            latency_seconds=time.perf_counter()-(job['request'].observed_perf_counter or job['submitted']),
            observation_at_utc=job['request'].observed_at_utc)
        self.record_event('pipeline_delivered', request_id, frame=frame,
            validation_frame=result['frame'], validation_seconds=result['validation_seconds'],
            end_to_end_seconds=response['latency_seconds'], response=response)
        self.job = None
        return response

    def cancel(self, request_id):
        if self.job and self.job['request'].request_id==request_id:
            self.job['canceled'] = True
            canceled = self.job['future'].cancel()
            self.record_event('pipeline_canceled', request_id, stage=self.job['stage'],
                              canceled_before_start=canceled)
            self._finish_discarded()

    def close(self):
        self.closed = True
        try:
            if self.job:
                self.cancel(self.job['request'].request_id)
                if self.job:
                    # Physical run has ended. Drain for auditing, never apply this result.
                    self.job['future'].result(timeout=90)
                    self._finish_discarded()
        finally:
            if self.owns_executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            if self.audit_path:
                with self.audit_path.open('a', encoding='utf-8') as stream:
                    for event in self.events:
                        stream.write(json.dumps(event, default=lambda v: vars(v) if isinstance(v,HighLevelRequest) else v.item(), allow_nan=False)+'\n')
            self.events.clear()


class AsyncTargetController(HighLevelController):
    def __init__(self, backend, low_level_controller=None, settings=None,
                 validation_max_age_frames=30):
        if settings is None:
            import config
            settings = config
        if (not settings.HIGH_LEVEL_TARGET_HANDOFF_ENABLED or settings.HIGH_LEVEL_MODE_SELECTION_ENABLED
                or not settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED):
            raise ValueError('Async target controller requires target-only handoff settings')
        super().__init__(low_level_controller, backend, settings)
        backend.controller = self
        self.validation_max_age_frames = validation_max_age_frames
        self.latest_observation = self.latest_summary = None

    def decide(self, observation):
        self.latest_observation = observation
        self.backend._finish_discarded()
        return super().decide(observation)

    def _observe_state(self, observation, alarms):
        summary = super()._observe_state(observation, alarms)
        self.latest_summary = summary
        return summary

    def _request_ready(self, observation, summary):
        return self.backend.can_submit()

    def _make_request(self, observation, alarms, summary):
        self.request_counter += 1
        return HighLevelRequest(f'high-{self.request_counter}', observation.frame, alarms,
            self._primary_alarm(alarms), copy.deepcopy(summary),
            self._observation_clock[0], self._observation_clock[1])

    def _with_mode_forecasts(self, observation, summary, include_targets=False, response=None):
        if include_targets:
            raise RealtimePipelineError('Heavy request forecasts must run in the worker')
        result = self.backend.validation_result
        if result is None:
            raise RealtimePipelineError('No completed background validation')
        cached = copy.deepcopy(result['summary'])
        cached.update(summary)
        cached['delivery_rule_target_sheep_id'] = copy.copy(self.low_level_controller).decide(observation).target_sheep_id
        target = response.get('target_sheep_id') if isinstance(response,dict) else None
        mode = response.get('mode') if isinstance(response,dict) else None
        if type(target) is int and mode in ('HARD_TO_GUIDE','COHESION_PRIORITY'):
            if self.low_level_controller.repair_target_is_eligible(observation,target):
                d = copy.copy(self.low_level_controller).decide(observation,mode_override=mode,target_override=target)
                if 'selected_target_candidate' in cached:
                    cached['selected_target_candidate']['drive_point_boundary_margin'] = self._boundary_margin(d.drive_point)
            else:
                cached['target_selection_error'] = 'target_no_longer_eligible'
        return cached

    def _validate_response(self, response, frame, alarms, current_summary):
        result = self.backend.validation_result
        if result is None or not 0<=frame-result['frame']<=self.validation_max_age_frames:
            return None,None,'background_validation_stale'
        return super()._validate_response(response,frame,alarms,current_summary)
