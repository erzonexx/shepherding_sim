"""Candidate MPC must preserve physical execution, bounded async jobs and rule fallback."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import config
from control_types import ControlObservation
from detector import Detector
from environment import SwarmEnv
from execution_contract import forecast_execution
from high_level_controller import HighLevelRequest
from low_level_controller import LowLevelController
from mpc_controller import (MPCBackend,MPCController,MODEL,HOLD,HORIZON,
    proposal_job,rank_candidates,proposal_input)
from realtime_controller import public_settings,snapshot
from run_experiments import temporary_config


def settings(**extra):
    values={k:v for k,v in vars(config).items() if k.isupper()}
    values.update(HIGH_LEVEL_MODE_SELECTION_ENABLED=True,HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,
        HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False,HIGH_LEVEL_EXPLICIT_EXECUTION=False,LLM_CACHE_ENABLED=False)
    values.update(extra)
    return SimpleNamespace(**values)


def observation(frame=200):
    return ControlObservation(np.array([[38.,100.],[80.,100.],[122.,100.]]),np.array([65.,110.]),
        np.array([180.,100.]),('flock_splitting','stagnation'),frame,('A','B','C'),
        sheep_velocities=np.zeros((3,2)),dog_velocity=np.zeros(2))


def metric(distance=100,peak=0,exits=0):
    return dict(peak_out_of_bounds=peak,predicted_goal_exits=exits,final_inside_goal=0,
                final_centroid_distance=distance,final_max_spread=50,predicted_splitting_frames=20)


class MPCTests(unittest.TestCase):
    def fixture(self):
        s=settings()
        b=MPCBackend(s,executor=ThreadPoolExecutor(1))
        c=MPCController(b,settings=s)
        obs=observation()
        c.latest_observation=obs
        summary=c._observe_state(obs,obs.alarms)
        state=snapshot(c,obs,summary)
        request=HighLevelRequest('one',obs.frame,obs.alarms,None,dict(summary,
            repair_candidates={m:[{'target_id':0},{'target_id':2}] for m in ('HARD_TO_GUIDE','COHESION_PRIORITY')}))
        return c,b,state,request

    def finish(self,c,b):
        c.close();b.executor.shutdown()

    def proposals(self,r,second_target=2):
        return dict(proposals=[dict(candidate_id=key,response=dict(request_id=r.request_id,
            trigger_frame=r.trigger_frame,mode='HARD_TO_GUIDE',target_sheep_id=target,
            hold_frames=30,served_model=MODEL)) for key,target in [('llm_progress',0),('llm_cohesion',second_target)]])

    def test_two_model_calls_overlap_and_use_separate_clients(self):
        barrier=threading.Barrier(2);seen=[]
        class FakeBackend:
            def __init__(self,**kw):
                self.instructions='shared';self.prompt_version='base';self.client=SimpleNamespace(close=lambda:None)
            def _call_api(self,r,key):
                seen.append(self)
                barrier.wait(timeout=2)
                return dict(mode='HARD_TO_GUIDE',target_sheep_id=None,hold_frames=30)
            def close(self):pass
        with patch('mpc_controller.proposal_input',side_effect=lambda s,r,v:r),patch('mpc_controller.Hold30VLLM',FakeBackend):
            r=HighLevelRequest('one',200,observation().alarms,None,{})
            result=proposal_job({},r,public_settings(settings()),'qwen','http://unused')
        self.assertEqual(len(seen),2);self.assertIsNot(seen[0],seen[1])
        self.assertEqual(len(result['response']['proposals']),2)

    def test_geometry_only_preserves_candidate_fields_without_discarded_forecast(self):
        c,b,state,r=self.fixture()
        try:
            with patch('high_level_controller.forecast_mode',return_value=metric()) as forecast:
                full=c._target_candidate(state['observation'],'HARD_TO_GUIDE',0)
                geometry=c._target_candidate(state['observation'],'HARD_TO_GUIDE',0,include_forecast=False)
            self.assertEqual(geometry,{k:v for k,v in full.items() if k!='forecast'})
            forecast.assert_called_once()
        finally:self.finish(c,b)

    def test_single_proposal_retains_same_progress_input_and_does_not_call_cohesion(self):
        r=HighLevelRequest('one',200,observation().alarms,None,{'unchanged':'input'})
        calls=[]
        def answer(req,values,base,role,guidance,deadline):
            calls.append((req,role,guidance))
            return dict(candidate_id=role,response=dict(mode='HARD_TO_GUIDE',target_sheep_id=None,hold_frames=30))
        with patch('mpc_controller.proposal_input',return_value=r),patch('mpc_controller.call_proposal',side_effect=answer):
            result=proposal_job({},r,{},'qwen','http://unused',proposal_count=1)
            self.assertEqual([p['candidate_id'] for p in result['response']['proposals']],['llm_progress'])
            self.assertEqual(len(calls),1)
            self.assertIs(calls[0][0],r)
            self.assertEqual(result['backend_info']['proposal_count'],1)
            pair=proposal_job({},r,{},'qwen','http://unused')
            self.assertEqual({p['candidate_id'] for p in pair['response']['proposals']},{'llm_progress','llm_cohesion'})
            self.assertTrue(all(req is r for req,role,guidance in calls))

    def test_backend_rejects_invalid_proposal_count_before_allocating_workers(self):
        for count in (0,3,True,'1'):
            with self.assertRaises(ValueError):MPCBackend(settings(),proposal_count=count)

    def test_rule_wins_ties_and_predicted_safety_regressions(self):
        c,b,state,r=self.fixture()
        try:
            for target_forecast in (metric(),metric(50,peak=1),metric(50,exits=1)):
                with patch('mpc_controller.dynamic_forecast',side_effect=lambda *a,**kw:
                    metric() if kw.get('policy')=='FOLLOW_RULE' else target_forecast):
                    result=rank_candidates(state,r,self.proposals(r),public_settings(c.settings))
                self.assertEqual(result['winner_id'],'rule')
        finally:self.finish(c,b)

    def test_valid_better_candidate_can_win_and_duplicates_are_recorded(self):
        c,b,state,r=self.fixture()
        try:
            with patch('mpc_controller.dynamic_forecast',side_effect=lambda *a,**kw:
                metric() if kw.get('policy')=='FOLLOW_RULE' else metric(90)):
                result=rank_candidates(state,r,self.proposals(r,0),public_settings(c.settings))
            self.assertEqual(result['winner_id'],'llm_progress')
            self.assertEqual(len(result['candidates']),2)
            self.assertEqual(result['rejected'][0]['reason'],'duplicate_proposal')
        finally:self.finish(c,b)

    def test_unoffered_and_ineligible_targets_do_not_enter_prediction(self):
        c,b,state,r=self.fixture()
        try:
            bundle=self.proposals(r)
            bundle['proposals'][0]['response']['target_sheep_id']=1
            bundle['proposals'][1]['response']['target_sheep_id']=1000
            result=rank_candidates(state,r,bundle,public_settings(c.settings))
            self.assertEqual(result['winner_id'],'rule');self.assertEqual(len(result['candidates']),1)
        finally:self.finish(c,b)

    def test_no_model_work_without_both_alarms_and_wait_is_nonblocking(self):
        c,b,state,r=self.fixture();release=threading.Event()
        def slow(*args):
            release.wait(timeout=3)
            return dict(request=args[1],response=dict(proposals=[]))
        b.select_fn=slow
        try:
            c.decide(replace(observation(),alarms=('stagnation',)))
            self.assertIsNone(b.job)
            start=time.perf_counter();c.decide(observation())
            for frame in range(201,216):self.assertEqual(c.decide(observation(frame)).source,'rule')
            self.assertLess(time.perf_counter()-start,.3)
            self.assertIsNotNone(b.job)
            c.decide(replace(observation(216),alarms=('stagnation',)))
            self.assertTrue(b.job['canceled'])
            quiet=c.decide(replace(observation(217),alarms=()))
            self.assertEqual(quiet.source,'rule')
            self.assertIsNone(c.pending_request)
        finally:self.finish(c,b)

    def test_deadline_returns_rule_without_applying_late_answers(self):
        c,b,state,r=self.fixture();release=threading.Event()
        def slow(*args):
            release.wait(timeout=3)
            return dict(request=args[1],response=dict(proposals=[]))
        b.select_fn=slow
        try:
            c.decide(observation());b.job['submitted']-=9
            actual=c.decide(observation(201))
            self.assertEqual(actual.source,'rule');self.assertIsNone(c.active_mode)
            self.assertEqual(actual.metadata['high_level_status'],'delegated')
            self.assertTrue(b.job['canceled'])
        finally:release.set();self.finish(c,b)

    def test_fresh_winner_is_checked_against_delivery_state(self):
        c,b,state,r=self.fixture()
        try:
            c.pending_request=r
            with patch('mpc_controller.dynamic_forecast',side_effect=lambda *a,**kw:
                metric() if kw.get('policy')=='FOLLOW_RULE' else metric(90)):
                result=rank_candidates(state,r,self.proposals(r),public_settings(c.settings))
            b.validation_result=result
            self.assertIsNone(c._validate_response(result['response'],201,observation().alarms,state['summary'])[2])
            self.assertEqual(c._validate_response(result['response'],201,('stagnation',),state['summary'])[2],
                             'concurrent_alarms_cleared')
            self.assertEqual(c._validate_response(result['response'],206,observation().alarms,state['summary'])[2],
                             'background_validation_stale')
            tampered=dict(result['response'],mode='COHESION_PRIORITY')
            self.assertEqual(c._validate_response(tampered,201,observation().alarms,state['summary'])[2],'mpc_winner_mismatch')
            altered=observation(201);altered.sheep_positions[result['response']['target_sheep_id']]=altered.goal_position
            c.latest_observation=altered
            self.assertEqual(c._validate_response(result['response'],201,observation().alarms,state['summary'])[2],
                             'target_no_longer_eligible')
        finally:self.finish(c,b)

    def test_60_frame_forecast_matches_every_real_step_and_early_release(self):
        class QuietMPC(MPCController):
            def _request_ready(self,*args):return False
        with temporary_config(NUM_SHEEP=3,NUM_ABNORMAL_A=1,NUM_ABNORMAL_B=1,NUM_ABNORMAL_C=1,
            HIGH_LEVEL_MODE_SELECTION_ENABLED=True,HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,
            HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False,HIGH_LEVEL_EXPLICIT_EXECUTION=False,
            DANGER_DISPERSION_THRESHOLD=1000.,DANGER_STAGNATION_THRESHOLD=1000.):
            obs=observation();history=[100.]*config.DANGER_STAGNATION_FRAMES
            low=LowLevelController();low.decide(obs)
            before=copy.deepcopy({k:v for k,v in vars(low).items() if k!='settings'})
            forecast=forecast_execution(obs,'COHESION_PRIORITY',HOLD,HORIZON,config,low,history,
                                        target_sheep_id=0,return_trace=True)
            self.assertEqual(low.repair_target_age,before['repair_target_age'])
            backend=MPCBackend(config,executor=ThreadPoolExecutor(1))
            c=QuietMPC(backend,copy.copy(low),config)
            c.active_mode='COHESION_PRIORITY';c.active_alarms=obs.alarms;c.active_started_frame=obs.frame
            c.active_until_frame=obs.frame+HOLD;c.active_hold_frames=HOLD;c.active_target_id=0
            c.active_request=HighLevelRequest('shadow',obs.frame,obs.alarms,None,{})
            env=SwarmEnv(c);env.sheep_pos=obs.sheep_positions.copy();env.sheep_vel=obs.sheep_velocities.copy()
            env.dog_pos=obs.dog_position.copy();env.dog_vel=obs.dog_velocity.copy();env.goal_pos=obs.goal_position.copy()
            env.current_frame=obs.frame;env.sheep_status=np.array(obs.sheep_statuses)
            detector=Detector(config.DANGER_DISPERSION_THRESHOLD,config.DANGER_MEAN_SPREAD_THRESHOLD,
                config.DANGER_STAGNATION_FRAMES,config.DANGER_STAGNATION_THRESHOLD)
            detector.history_dist_to_goal=history.copy();alarms=obs.alarms
            try:
                for expected in forecast['trace']:
                    env.step(alarms)
                    np.testing.assert_array_equal(env.sheep_pos,np.array(expected['sheep_positions']))
                    np.testing.assert_array_equal(env.sheep_vel,np.array(expected['sheep_velocities']))
                    np.testing.assert_array_equal(env.dog_pos,np.array(expected['dog_position']))
                    np.testing.assert_array_equal(env.dog_vel,np.array(expected['dog_velocity']))
                    self.assertEqual((env.current_mode,env.target_sheep_id),(expected['mode'],expected['target_sheep_id']))
                    alarms=detector.detect(env.sheep_pos,env.goal_pos)[0]
                self.assertEqual(forecast['committed_frames'],15)
                self.assertEqual(forecast['release_reason'],'selected_alarm_cleared')
            finally:env.close();backend.executor.shutdown()


if __name__=='__main__':unittest.main()
