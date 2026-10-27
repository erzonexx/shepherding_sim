"""Latency-matched candidate-source x MPC-filter ablation, independent of retained MPC.

All four intervention arms call the SAME retained LLM proposal pipeline and run
the SAME pair of full-tail forecasts. Heuristic arms ignore the LLM's action and
use only its completion timing; direct arms record but ignore forecast ranking.
Thus these are mechanism ablations, not benchmarks of a model-free deployment.
"""
import copy
from functools import partial
import time
import numpy as np

from high_level_controller import ALLOWED_MODES
from mpc_controller import (MPCBackend,MPCController,BOTH,HOLD,MODEL,VERSION as PROPOSAL_VERSION,
    candidate_error,proposal_job)
from mpc_completion_controller import full_rule_tail_forecast,completion_outcome,completion_key
from realtime_controller import predictor

VERSION='candidate-source-filter-factorial-v1'
ARMS=('rule','heuristic_direct','llm_direct','heuristic_mpc','llm_mpc')


def arm_factors(arm):
    if arm not in ARMS[1:]:raise ValueError('Unknown intervention arm')
    return arm.split('_')[0],arm.endswith('_mpc')


def heuristic_action(summary,positions,goal):
    """Geometry-only: rule mode, farthest offered target from goal, smallest-ID tie.

    Deliberately receives no types, velocities, predictions or model output.
    The offered IDs are the existing candidate screen; no extra targets added.
    """
    mode=summary['rule_mode']
    offered=[row['target_id'] for row in summary.get('repair_candidates',{}).get(mode,[])]
    target=min(offered,key=lambda i:(-float(np.linalg.norm(positions[i]-goal)),i)) if offered else None
    return dict(mode=mode,target_sheep_id=target)


def ablation_proposal_job(state,request,values,kind,base_url,*,arm):
    source,filtered=arm_factors(arm)
    result=proposal_job(state,request,values,kind,base_url,proposal_count=1)
    clock=copy.deepcopy(result['response']['proposals'][0])
    response=clock['response']
    if response.get('backend_error') or response.get('served_model')!=MODEL:
        raise RuntimeError('Retained clock/model proposal failed')
    if source=='heuristic':
        action=heuristic_action(result['request'].state_summary,state['observation'].sheep_positions,
            state['observation'].goal_position)
        result['response']['heuristic_input']=dict(positions=state['observation'].sheep_positions.tolist(),
            goal=state['observation'].goal_position.tolist())
        proposal=dict(candidate_id='heuristic',generator='geometry-heuristic-v1',response=dict(
            request_id=request.request_id,trigger_frame=request.trigger_frame,hold_frames=HOLD,
            **action,model='geometry-heuristic-v1',cache_hit=False))
    else:
        proposal=copy.deepcopy(clock);proposal['generator']='llm'
    result['response']['clock_proposal']=clock
    result['response']['proposals']=[proposal]
    result['response']['ablation_arm']=arm
    result['backend_info'].update(version=VERSION,action_source=source,mpc_filter_enabled=filtered,
        clock_service_model=MODEL,clock_only=source=='heuristic')
    return result


def ablation_candidate_error(proposal,request,state,values):
    if proposal.get('generator')=='llm':return candidate_error(proposal,request,state,values)
    if proposal.get('generator')!='geometry-heuristic-v1':return 'unknown_candidate_generator'
    r=proposal['response'];obs=state['observation'];c=predictor(state,values)
    if r.get('request_id')!=request.request_id or r.get('trigger_frame')!=request.trigger_frame:
        return 'request_identity_mismatch'
    if type(r.get('hold_frames')) is not int or r['hold_frames']!=HOLD:return 'invalid_hold'
    mode,target=r.get('mode'),r.get('target_sheep_id')
    if mode not in ALLOWED_MODES:return 'mode_not_allowed'
    if target is not None:
        if type(target) is not int or mode=='TARGET_TRACKING':return 'invalid_target'
        if not any(x['target_id']==target for x in request.state_summary['repair_candidates'].get(mode,[])):
            return 'target_not_offered'
        if not c.low_level_controller.repair_target_is_eligible(obs,target):return 'target_no_longer_eligible'
    decision=copy.copy(c.low_level_controller).decide(obs,mode_override=mode,target_override=target)
    if c._boundary_margin(decision.drive_point)<0:return 'drive_point_out_of_bounds'
    return None


def rank_ablation_candidates(state,request,bundle,values,*,arm):
    """Retained full-tail scoring, with one explicit source/filter factor switch."""
    source,filtered=arm_factors(arm)
    assert bundle['ablation_arm']==arm and len(bundle['proposals'])==1
    started=time.perf_counter();obs=state['observation'];controller=predictor(state,values)
    rule_mode=state['summary']['rule_mode'];horizon=max(0,values['MAX_STEPS']-obs.frame)
    def forecast(mode,target=None,policy='COMMIT'):
        return full_rule_tail_forecast(obs,mode,HOLD,horizon,controller.settings,
            controller.low_level_controller,state['history'],target,policy=policy,trigger_alarms=request.alarms)
    rule=dict(candidate_id='rule',mode=rule_mode,target_sheep_id=None,
        forecast=forecast(rule_mode,policy='FOLLOW_RULE'))
    candidates,rejected=[rule],[]
    relevant=BOTH<=set(obs.alarms) and obs.frame-request.trigger_frame<=values['HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES']
    proposal=bundle['proposals'][0];r=proposal['response']
    error=ablation_candidate_error(proposal,request,state,values) if relevant else 'trigger_cleared_or_stale'
    if error:rejected.append(dict(candidate_id=proposal['candidate_id'],reason=error))
    else:candidates.append(dict(candidate_id=proposal['candidate_id'],mode=r['mode'],
        target_sheep_id=r.get('target_sheep_id'),eligible=True,
        forecast=forecast(r['mode'],r.get('target_sheep_id'))))
    for candidate in candidates:
        candidate['task_outcome']=completion_outcome(candidate['forecast'],obs.frame,values['MAX_STEPS'])
    scored_winner=min(candidates,key=completion_key)
    winner=scored_winner if filtered else candidates[-1]
    response=dict(request_id=request.request_id,trigger_frame=request.trigger_frame,
        mode=winner['mode'],target_sheep_id=winner['target_sheep_id'],hold_frames=HOLD,
        mpc_candidate_id=winner['candidate_id'],mpc_policy='FOLLOW_RULE' if winner['candidate_id']=='rule' else 'COMMIT',
        model=MODEL if source=='llm' else 'geometry-heuristic-v1',prompt_version=PROPOSAL_VERSION,
        cache_hit=False,candidate_generator=source,mpc_filter_enabled=filtered,
        reason='Completion filter' if filtered else 'Direct valid proposal; prediction is shadow-only')
    return dict(summary={},frame=obs.frame,validation_seconds=time.perf_counter()-started,
        winner_id=winner['candidate_id'],shadow_winner_id=scored_winner['candidate_id'],response=response,
        candidates=candidates,rejected=rejected,snapshot_frame=obs.frame,horizon_frames=horizon,
        hold_frames=HOLD,ranking_version=VERSION,remaining_budget=horizon,ablation_arm=arm,
        prediction_state=dict(frame=obs.frame,positions=obs.sheep_positions.tolist(),
            velocities=obs.sheep_velocities.tolist(),dog=obs.dog_position.tolist(),
            dog_velocity=obs.dog_velocity.tolist(),goal=obs.goal_position.tolist(),
            statuses=list(obs.sheep_statuses),alarms=list(obs.alarms),current_mode=obs.current_mode,
            low=copy.deepcopy(state['low']),history=list(state['history'])))


class AblationBackend(MPCBackend):
    def __init__(self,settings,arm,base_url=None,executor=None):
        source,self.filter_enabled=arm_factors(arm);self.arm=arm
        super().__init__(settings,base_url,executor,proposal_count=1)
        self.source='vllm' if source=='llm' else 'heuristic'
        self.model=MODEL if source=='llm' else 'geometry-heuristic-v1'
        self.allowed_candidate_ids={'llm_progress'} if source=='llm' else {'heuristic'}
        self.select_fn=partial(ablation_proposal_job,arm=arm)
        self.validate_fn=partial(rank_ablation_candidates,arm=arm)


class AblationController(MPCController):
    # Same fast checks as retained MPCController, replacing only its hardcoded
    # candidate-ID whitelist. No forecast/score computation runs in this class.
    def _validate_response(self,response,frame,alarms,current_summary):
        result=self.backend.validation_result;request=self.pending_request
        if not isinstance(response,dict) or not request:return None,None,'missing_mpc_round'
        if response.get('request_id')!=request.request_id or response.get('trigger_frame')!=request.trigger_frame:
            return None,None,'request_identity_mismatch'
        if not BOTH<=set(alarms):return None,None,'concurrent_alarms_cleared'
        if frame-request.trigger_frame>self.settings.HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES:
            return None,None,'response_stale'
        if result is None or not 0<=frame-result['frame']<=self.validation_max_age_frames:
            return None,None,'background_validation_stale'
        if response!=result['response'] and any(response.get(k)!=result['response'].get(k)
            for k in ('request_id','trigger_frame','mode','target_sheep_id','hold_frames','mpc_candidate_id','mpc_policy')):
            return None,None,'mpc_winner_mismatch'
        if response.get('mpc_policy')=='FOLLOW_RULE':return current_summary['rule_mode'],HOLD,None
        if result.get('winner_id') not in self.backend.allowed_candidate_ids or response.get('mpc_policy')!='COMMIT':
            return None,None,'invalid_mpc_winner'
        target,mode=response.get('target_sheep_id'),response.get('mode')
        if mode not in ALLOWED_MODES or type(response.get('hold_frames')) is not int or response['hold_frames']!=HOLD:
            return None,None,'invalid_mpc_action'
        obs=self.latest_observation
        if target is not None and not self.low_level_controller.repair_target_is_eligible(obs,target):
            return None,None,'target_no_longer_eligible'
        decision=copy.copy(self.low_level_controller).decide(obs,mode_override=mode,target_override=target)
        if self._boundary_margin(decision.drive_point)<0:return None,None,'drive_point_out_of_bounds'
        return mode,HOLD,None

    def _response_metadata(self,response):
        return dict(super()._response_metadata(response),candidate_generator=response.get('candidate_generator'),
            mpc_filter_enabled=response.get('mpc_filter_enabled'))
