"""Event-triggered finite-candidate MPC: changing rules versus two Qwen proposals.

Only the background worker predicts. Proposals share an initial input; ranking
uses one fresh post-response snapshot and the real hold/release machinery.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
import hashlib
import time
from types import SimpleNamespace

from execution_contract import forecast_execution
from high_level_controller import ALLOWED_MODES
from realtime_controller import (AsyncPredictionBackend, Hold30VLLM, MODEL,
    predictor, snapshot)
from realtime_joint_controller import AsyncJointController
from experiment_audit import utc_now

VERSION = 'event-three-candidate-mpc-v1'
HORIZON = 60
HOLD = 30
PAIR_DEADLINE_SECONDS = 8.
VALIDATION_MAX_AGE_FRAMES = 5
BOTH = {'flock_splitting', 'stagnation'}
ROLES = (
    ('llm_progress', 'Favor safe goalward progress when several choices are reasonable.'),
    ('llm_cohesion', 'Favor safe cohesion recovery when several choices are reasonable.'),
)
PROPOSAL_INSTRUCTIONS = """

MPC PROPOSAL CONTRACT (overrides older forecast-horizon descriptions):
You propose one mode and optional offered target, with hold_frames=30.
The supplied forecasts simulate at most 30 committed frames, including actual
early releases, then dynamic rules up to 60 frames, with evolving alarms.
After both proposals return, an external controller compares them with dynamic
rules using a NEW state snapshot. Your answer is a proposal, not a command.
All three modes remain admissible according to the schema. Do not invent a
different action merely to differ from another proposal. Keep reason concise.
"""


def dynamic_forecast(state, values, mode, target=None, policy='COMMIT', trigger_alarms=None):
    c = predictor(state, values)
    return forecast_execution(state['observation'], mode, HOLD, HORIZON, c.settings,
        c.low_level_controller, state['history'], target,
        execution_policy=policy, trigger_alarms=trigger_alarms)


def proposal_input(state, request, values):
    c = predictor(state, values)
    obs = state['observation']
    rich = dict(state['summary'])
    rich['mode_forecasts'] = {m:dynamic_forecast(state, values, m) for m in sorted(ALLOWED_MODES)}
    rich['rule_forecast'] = dynamic_forecast(state, values, rich['rule_mode'], policy='FOLLOW_RULE')
    rich['repair_candidates'] = {}
    for mode in ('HARD_TO_GUIDE', 'COHESION_PRIORITY'):
        entries = []
        for target in c.low_level_controller.repair_candidates(obs, mode):
            # MPC supplies its own60-frame forecast; avoid computing a discarded legacy one.
            entry = c._target_candidate(obs, mode, target, include_forecast=False)
            entry['forecast'] = dynamic_forecast(state, values, mode, target)
            entries.append(entry)
        rich['repair_candidates'][mode] = entries
    rich['mpc_proposal_contract'] = dict(horizon_frames=HORIZON, max_hold_frames=HOLD,
        execution='COMMIT with real early releases, then dynamic rules; no future model answers')
    return replace(request, state_summary=rich)


def call_proposal(request, values, base_url, role, guidance, deadline):
    local = dict(values, LLM_API_TIMEOUT_SECONDS=max(.05, deadline-time.perf_counter()),
                 LLM_CACHE_ENABLED=False)
    backend = Hold30VLLM(settings=SimpleNamespace(**local), base_url=base_url, model=MODEL,
        compact_output=False, prefix_priming=False, input_format='json')
    backend.instructions += PROPOSAL_INSTRUCTIONS+'\nProposal preference: '+guidance
    backend.prompt_version += '-'+VERSION+'-'+role
    try:
        response = backend._call_api(request, 'mpc-no-cache')
        return dict(candidate_id=role, response=response,
                    instructions_sha256=hashlib.sha256(backend.instructions.encode()).hexdigest())
    finally:
        backend.close()
        backend.client.close()


def proposal_job(state, request, values, kind, base_url, *, proposal_count=2):
    if type(proposal_count) is not int or proposal_count not in (1,2):
        raise ValueError('MPC requires one or two LLM proposals')
    started = time.perf_counter()
    enriched = proposal_input(state, request, values)
    prepared = time.perf_counter()
    deadline = started+PAIR_DEADLINE_SECONDS
    with ThreadPoolExecutor(max_workers=proposal_count) as threads:
        futures = [threads.submit(call_proposal, enriched, values, base_url, role, guidance, deadline)
                   for role,guidance in ROLES[:proposal_count]]
        proposals = [f.result() for f in futures]
    finished = time.perf_counter()
    return dict(request=enriched,
        response=dict(request_id=request.request_id, trigger_frame=request.trigger_frame,
                      proposals=proposals),
        prediction_seconds=prepared-started, pair_seconds=finished-prepared,
        total_worker_seconds=finished-started,
        proposal_deadline_exceeded=finished>deadline,
        backend_info=dict(source='vllm', model=MODEL, version=VERSION, proposal_count=proposal_count))


def candidate_error(candidate, request, state, values):
    response = candidate['response']
    obs = state['observation']
    c = predictor(state, values)
    if response.get('backend_error'):
        return 'model_request_failed'
    if response.get('request_id')!=request.request_id or response.get('trigger_frame')!=request.trigger_frame:
        return 'request_identity_mismatch'
    if response.get('served_model')!=MODEL:
        return 'served_model_mismatch'
    if type(response.get('hold_frames')) is not int or response['hold_frames']!=HOLD:
        return 'invalid_hold'
    mode,target = response.get('mode'),response.get('target_sheep_id')
    if not isinstance(mode,str) or mode not in ALLOWED_MODES:
        return 'mode_not_allowed'
    if target is not None:
        if type(target) is not int or mode=='TARGET_TRACKING':
            return 'invalid_target'
        if not any(x['target_id']==target for x in request.state_summary['repair_candidates'].get(mode, [])):
            return 'target_not_offered'
        if not c.low_level_controller.repair_target_is_eligible(obs,target):
            return 'target_no_longer_eligible'
    decision = copy.copy(c.low_level_controller).decide(obs, mode_override=mode, target_override=target)
    if c._boundary_margin(decision.drive_point)<0:
        return 'drive_point_out_of_bounds'
    return None


def forecast_rank(forecast):
    # Fixed before running: prioritize boundary/goal safety, then task progress.
    return (forecast['peak_out_of_bounds'],forecast['predicted_goal_exits'],
        -forecast['final_inside_goal'],forecast['final_centroid_distance'],
        forecast['final_max_spread'],forecast['predicted_splitting_frames'])


def rank_candidates(state, request, bundle, values):
    started = time.perf_counter()
    obs = state['observation']
    rule_mode = state['summary']['rule_mode']
    rule = dict(candidate_id='rule', mode=rule_mode, target_sheep_id=None,
        forecast=dynamic_forecast(state,values,rule_mode,policy='FOLLOW_RULE'))
    candidates,rejected,seen = [rule],[],set()
    relevant = BOTH<=set(obs.alarms) and obs.frame-request.trigger_frame<=values['HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES']
    for proposal in bundle['proposals']:
        error = candidate_error(proposal,request,state,values) if relevant else 'trigger_cleared_or_stale'
        response = proposal['response']
        key = (response.get('mode'),response.get('target_sheep_id'))
        if not error and key in seen:
            error = 'duplicate_proposal'
        if error:
            rejected.append(dict(candidate_id=proposal['candidate_id'],reason=error))
            continue
        seen.add(key)
        f = dynamic_forecast(state,values,key[0],key[1],trigger_alarms=request.alarms)
        candidate = dict(candidate_id=proposal['candidate_id'],mode=key[0],target_sheep_id=key[1],forecast=f)
        candidate['eligible'] = (f['peak_out_of_bounds']<=rule['forecast']['peak_out_of_bounds']
            and f['predicted_goal_exits']<=rule['forecast']['predicted_goal_exits'])
        candidates.append(candidate)
    eligible = [x for x in candidates if x.get('eligible',True)]
    winner = min(eligible,key=lambda x:(forecast_rank(x['forecast']),x['candidate_id']!='rule',x['candidate_id']))
    response = dict(request_id=request.request_id,trigger_frame=request.trigger_frame,
        mode=winner['mode'],target_sheep_id=winner['target_sheep_id'],hold_frames=HOLD,
        mpc_candidate_id=winner['candidate_id'],mpc_policy='FOLLOW_RULE' if winner['candidate_id']=='rule' else 'COMMIT',
        model=MODEL,prompt_version=VERSION,cache_hit=False,
        reason='MPC selected '+winner['candidate_id'])
    return dict(summary={},frame=obs.frame,validation_seconds=time.perf_counter()-started,
        winner_id=winner['candidate_id'],response=response,candidates=candidates,rejected=rejected,
        snapshot_frame=obs.frame,horizon_frames=HORIZON,hold_frames=HOLD,
        prediction_state=dict(frame=obs.frame,positions=obs.sheep_positions.tolist(),
            velocities=obs.sheep_velocities.tolist(),dog=obs.dog_position.tolist(),
            dog_velocity=obs.dog_velocity.tolist(),goal=obs.goal_position.tolist(),
            statuses=list(obs.sheep_statuses),alarms=list(obs.alarms),current_mode=obs.current_mode,
            low=copy.deepcopy(state['low']),history=list(state['history'])))


class MPCBackend(AsyncPredictionBackend):
    def __init__(self, settings, base_url=None, executor=None, *, proposal_count=2):
        if type(proposal_count) is not int or proposal_count not in (1,2):
            raise ValueError('MPC requires one or two LLM proposals')
        self.proposal_count = proposal_count
        super().__init__(settings,kind='qwen',base_url=base_url,executor=executor,
                         select_fn=partial(proposal_job,proposal_count=proposal_count),
                         validate_fn=rank_candidates)
        self.prompt_version = VERSION

    def poll(self, request_id, frame):
        job = self.job
        if job and not job['canceled'] and job['request'].request_id==request_id:
            if job['stage']=='selection' and time.perf_counter()-job['submitted']>PAIR_DEADLINE_SECONDS:
                request = job['request']
                self.cancel(request_id)
                response = dict(request_id=request_id,trigger_frame=request.trigger_frame,
                    mode=self.controller.latest_summary['rule_mode'],target_sheep_id=None,hold_frames=HOLD,
                    mpc_candidate_id='rule',mpc_policy='FOLLOW_RULE',reason='MPC pair deadline',
                    model=MODEL,prompt_version=VERSION,cache_hit=False)
                self.validation_result = dict(frame=frame,summary={},response=response,timed_out=True)
                self.record_event('mpc_deadline_fallback',request_id,frame=frame,response=response)
                return response
            if job['stage']=='validation' and job['future'].done():
                result = self._result(job['future'])
                job['response'] = result['response']
                self.record_event('mpc_ranked',request_id,consumed_frame=frame,result=result)
        return super().poll(request_id,frame)


class MPCController(AsyncJointController):
    def __init__(self, backend, low_level_controller=None, settings=None, **unused):
        super().__init__(backend,low_level_controller,settings,
                         validation_max_age_frames=VALIDATION_MAX_AGE_FRAMES)
        backend.prompt_version = VERSION

    def _request_is_relevant(self, request, alarms):
        return BOTH<=set(alarms)

    def _with_mode_forecasts(self, observation, summary, include_targets=False, response=None):
        if include_targets:
            raise RuntimeError('MPC forecasts must remain in the background worker')
        return dict(summary)

    def _response_delegates_to_rule(self, response, summary):
        return response.get('mpc_policy')=='FOLLOW_RULE'

    def _validate_response(self, response, frame, alarms, current_summary):
        result = self.backend.validation_result
        request = self.pending_request
        if not isinstance(response,dict) or not request:
            return None,None,'missing_mpc_round'
        if response.get('request_id')!=request.request_id or response.get('trigger_frame')!=request.trigger_frame:
            return None,None,'request_identity_mismatch'
        if not BOTH<=set(alarms):
            return None,None,'concurrent_alarms_cleared'
        if frame-request.trigger_frame>self.settings.HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES:
            return None,None,'response_stale'
        if result is None or not 0<=frame-result['frame']<=self.validation_max_age_frames:
            return None,None,'background_validation_stale'
        if response!=result['response'] and any(response.get(k)!=result['response'].get(k)
                for k in ('request_id','trigger_frame','mode','target_sheep_id','hold_frames','mpc_candidate_id','mpc_policy')):
            return None,None,'mpc_winner_mismatch'
        if response.get('mpc_policy')=='FOLLOW_RULE':
            return current_summary['rule_mode'],HOLD,None
        if result.get('winner_id') not in {r for r,g in ROLES[:self.backend.proposal_count]} or response.get('mpc_policy')!='COMMIT':
            return None,None,'invalid_mpc_winner'
        target,mode = response.get('target_sheep_id'),response.get('mode')
        if mode not in ALLOWED_MODES or type(response.get('hold_frames')) is not int or response['hold_frames']!=HOLD:
            return None,None,'invalid_mpc_action'
        obs = self.latest_observation
        if target is not None and not self.low_level_controller.repair_target_is_eligible(obs,target):
            return None,None,'target_no_longer_eligible'
        decision = copy.copy(self.low_level_controller).decide(obs,mode_override=mode,target_override=target)
        if self._boundary_margin(decision.drive_point)<0:
            return None,None,'drive_point_out_of_bounds'
        return mode,HOLD,None

    def _response_metadata(self, response):
        return dict(super()._response_metadata(response),mpc_candidate_id=response.get('mpc_candidate_id'),
                    mpc_policy=response.get('mpc_policy'))
