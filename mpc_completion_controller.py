"""Task-completion MPC variant; legacy MPC and proposal prompts stay intact.

The physical model is known, as in the existing MPC. Counterfactual tails assume
no future LLM replies; asynchronous admission and actual execution can diverge.
"""
import copy
import time
import numpy as np

from control_types import ControlObservation
from detector import Detector
from dynamics import advance_dog, advance_sheep
from high_level_controller import HighLevelController, HighLevelRequest
from mode_forecast import _flock_metrics, _inside_goal
from mpc_controller import (BOTH, HOLD, MODEL, VERSION as ORIGINAL_VERSION,
    MPCBackend, candidate_error)
from realtime_controller import predictor

VERSION = 'mpc-rule-tail-completion-v1'


def full_rule_tail_forecast(observation,mode,hold,horizon,settings,low,history,
                           target=None,policy='COMMIT',trigger_alarms=None,trace=False):
    """Exact existing execution for the intervention; direct rules after release.

    Once no request or active intervention exists, HighLevelController only calls
    low.decide. Skip its metadata/UTC bookkeeping for the rest of the rule tail.
    Motion, detector history, target hysteresis and successful first entry remain
    identical. Tests compare every recorded physical frame and forecast metric.
    """
    class NoRequests:
        source='forecast'
        trigger_on_concurrent_only=True
    class Shadow(HighLevelController):
        def _observe_state(self,obs,alarms):return {}
        def _request_ready(self,obs,summary):return False
    controller=Shadow(copy.copy(low),NoRequests(),settings)
    if policy=='COMMIT':
        controller.active_mode=mode
        controller.active_alarms=tuple(observation.alarms if trigger_alarms is None else trigger_alarms)
        controller.active_started_frame=observation.frame
        controller.active_until_frame=observation.frame+hold
        controller.active_hold_frames=hold
        controller.active_target_id=target
        controller.active_request=HighLevelRequest('forecast',observation.frame,tuple(observation.alarms),None,{})
    elif policy!='FOLLOW_RULE':
        raise ValueError('Unknown execution policy')
    detector=Detector(settings.DANGER_DISPERSION_THRESHOLD,settings.DANGER_MEAN_SPREAD_THRESHOLD,
        settings.DANGER_STAGNATION_FRAMES,settings.DANGER_STAGNATION_THRESHOLD)
    detector.history_dist_to_goal=list(history)
    positions=np.array(observation.sheep_positions,dtype=float,copy=True)
    velocities=controller.low_level_controller._velocities(observation).copy()
    dog=np.array(observation.dog_position,dtype=float,copy=True)
    dog_velocity=np.zeros(2) if observation.dog_velocity is None else np.array(observation.dog_velocity,copy=True)
    goal=np.array(observation.goal_position,copy=True)
    statuses=tuple(controller._normalized_statuses(observation.sheep_statuses,len(positions)))
    initial=_flock_metrics(positions,goal,settings)
    inside=_inside_goal(positions,goal,settings)
    alarms=tuple(observation.alarms)
    exits=splitting=committed=simulated=0
    peak=initial['out_of_bounds'];margin=initial['boundary_margin'];travel=0.
    release_offset=release_reason=None
    current_mode=observation.current_mode
    steps=[] if trace else None
    direct=policy=='FOLLOW_RULE'
    for offset in range(horizon):
        if inside.all():break
        obs=ControlObservation(positions,dog,goal,alarms,observation.frame+offset,
            statuses,current_mode,velocities,dog_velocity)
        if direct:
            decision=controller.low_level_controller.decide(obs)
        else:
            decision=controller.decide(obs)
            direct=controller.active_mode is None
        if decision.source=='forecast':committed+=1
        elif policy=='COMMIT' and release_offset is None:
            release_offset=offset;release_reason=decision.metadata.get('release_reason')
        current_mode=decision.mode
        next_dog,dog_velocity=advance_dog(dog,decision.navigation_point,settings)
        travel+=float(np.linalg.norm(next_dog-dog));dog=next_dog
        positions,velocities=advance_sheep(positions,velocities,np.asarray(statuses),dog,goal,settings)
        next_inside=_inside_goal(positions,goal,settings)
        exits+=int(np.sum(inside & ~next_inside));inside=next_inside
        outside=int(np.sum(np.any((positions<0.) | (positions>settings.SPACE_SIZE),axis=1)))
        peak=max(peak,outside)
        boundary_margin=float(min(np.min(positions),np.min(settings.SPACE_SIZE-positions)))
        margin=min(margin,boundary_margin)
        alarms=tuple(detector.detect(positions,goal)[0])
        splitting+=int('flock_splitting' in alarms)
        simulated+=1
        if steps is not None:
            steps.append(dict(sheep_positions=positions.tolist(),sheep_velocities=velocities.tolist(),
                dog_position=dog.tolist(),dog_velocity=dog_velocity.tolist(),mode=decision.mode,
                target_sheep_id=decision.target_sheep_id))
    final=_flock_metrics(positions,goal,settings)
    result=dict(horizon_frames=horizon,
        centroid_progress=round(initial['centroid_distance']-final['centroid_distance'],3),
        final_centroid_distance=round(final['centroid_distance'],3),
        max_spread_change=round(final['max_spread']-initial['max_spread'],3),
        final_max_spread=round(final['max_spread'],3),
        mean_spread_change=round(final['mean_spread']-initial['mean_spread'],3),
        final_mean_spread=round(final['mean_spread'],3),
        inside_goal_change=final['inside_goal']-initial['inside_goal'],final_inside_goal=final['inside_goal'],
        predicted_goal_exits=exits,predicted_splitting_frames=splitting,peak_out_of_bounds=peak,
        minimum_boundary_margin=round(margin,3),dog_travel=round(travel,3),
        predicted_success=bool(inside.all()),execution_policy=policy,
        forecast_hold_frames=hold if policy=='COMMIT' else None,simulated_frames=simulated,
        committed_frames=committed,release_offset=release_offset,release_reason=release_reason)
    if steps is not None:result['trace']=steps
    return result


def completion_outcome(forecast, frame, max_steps):
    raw_finish = frame + forecast['simulated_frames'] + 1
    success = bool(forecast['predicted_success'] and raw_finish <= max_steps)
    return dict(success=success, frames=raw_finish if success else max_steps)


def completion_key(candidate):
    result = candidate['task_outcome']
    return (not result['success'], result['frames'],
            candidate['candidate_id'] != 'rule', candidate['candidate_id'])


def rank_completion_candidates(state, request, bundle, values):
    """Same offered candidates and current validity tests; replace ranking only.

    Compare a complete remaining-budget rule tail to <=30 committed frames then
    a rule tail. Success and completion frame override the old risk-first order.
    When neither branch succeeds, or finishes equally fast, keep dynamic rules.
    Risk and goal exits remain measured; the legacy componentwise gate is absent.
    """
    started=time.perf_counter()
    obs=state['observation']
    controller=predictor(state,values)
    rule_mode=state['summary']['rule_mode']
    horizon=max(0,values['MAX_STEPS']-obs.frame)

    def forecast(mode,target=None,policy='COMMIT'):
        return full_rule_tail_forecast(obs,mode,HOLD,horizon,controller.settings,
            controller.low_level_controller,state['history'],target,
            policy=policy,trigger_alarms=request.alarms)

    rule=dict(candidate_id='rule',mode=rule_mode,target_sheep_id=None,
              forecast=forecast(rule_mode,policy='FOLLOW_RULE'))
    candidates,rejected,seen=[rule],[],set()
    relevant=BOTH<=set(obs.alarms) and obs.frame-request.trigger_frame<=values['HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES']
    for proposal in bundle['proposals']:
        error=candidate_error(proposal,request,state,values) if relevant else 'trigger_cleared_or_stale'
        response=proposal['response']
        key=(response.get('mode'),response.get('target_sheep_id'))
        if not error and key in seen:
            error='duplicate_proposal'
        if error:
            rejected.append(dict(candidate_id=proposal['candidate_id'],reason=error))
            continue
        seen.add(key)
        candidates.append(dict(candidate_id=proposal['candidate_id'],mode=key[0],
            target_sheep_id=key[1],eligible=True,forecast=forecast(*key)))
    for candidate in candidates:
        candidate['task_outcome']=completion_outcome(candidate['forecast'],obs.frame,values['MAX_STEPS'])
    winner=min(candidates,key=completion_key)
    response=dict(request_id=request.request_id,trigger_frame=request.trigger_frame,
        mode=winner['mode'],target_sheep_id=winner['target_sheep_id'],hold_frames=HOLD,
        mpc_candidate_id=winner['candidate_id'],
        mpc_policy='FOLLOW_RULE' if winner['candidate_id']=='rule' else 'COMMIT',
        model=MODEL,prompt_version=ORIGINAL_VERSION,cache_hit=False,
        reason='Task-completion rule-tail MPC selected '+winner['candidate_id'])
    return dict(summary={},frame=obs.frame,validation_seconds=time.perf_counter()-started,
        winner_id=winner['candidate_id'],response=response,candidates=candidates,rejected=rejected,
        snapshot_frame=obs.frame,horizon_frames=horizon,hold_frames=HOLD,
        ranking_version=VERSION,remaining_budget=values['MAX_STEPS']-obs.frame,
        prediction_state=dict(frame=obs.frame,positions=obs.sheep_positions.tolist(),
            velocities=obs.sheep_velocities.tolist(),dog=obs.dog_position.tolist(),
            dog_velocity=obs.dog_velocity.tolist(),goal=obs.goal_position.tolist(),
            statuses=list(obs.sheep_statuses),alarms=list(obs.alarms),current_mode=obs.current_mode,
            low=copy.deepcopy(state['low']),history=list(state['history'])))


class CompletionMPCBackend(MPCBackend):
    def __init__(self,settings,base_url=None,executor=None):
        super().__init__(settings,base_url,executor,proposal_count=1)
        self.validate_fn=rank_completion_candidates
