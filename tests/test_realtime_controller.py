"""Real worker delays must not stop physics or resurrect canceled suggestions."""
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import multiprocessing
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import config
from environment import SwarmEnv
from high_level_controller import HighLevelController, HighLevelRequest
from realtime_clock import RealtimeClock
from realtime_controller import (AsyncPredictionBackend, AsyncTargetController,
    public_settings, selection_job, snapshot, validation_job)
from run_experiments import temporary_config
from tests.anomaly_behavior import test_high_level_target_selection as target_fixture


def slow_selection(*args):
    time.sleep(.5)
    return selection_job(*args)


class RealtimeControllerTests(unittest.TestCase):
    def settings(self):
        values={k:v for k,v in vars(config).items() if k.isupper()}
        values.update(HIGH_LEVEL_TARGET_HANDOFF_ENABLED=True,HIGH_LEVEL_MODE_SELECTION_ENABLED=False,
            HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,HIGH_LEVEL_EXPLICIT_EXECUTION=False,
            HIGH_LEVEL_FORECAST_FRAMES=1,LLM_CACHE_ENABLED=False)
        return SimpleNamespace(**values)

    def observation(self,frame=0):
        return target_fixture.HighLevelTargetSelectionTests().observation(frame)

    def make(self,executor=None,fn=slow_selection):
        settings=self.settings()
        b=AsyncPredictionBackend(settings,executor=executor,select_fn=fn)
        return AsyncTargetController(b,settings=settings)

    def test_actual_process_worker_does_not_block_control_or_mutate_snapshot(self):
        c=self.make();c.backend.warm()
        obs=self.observation();before=obs.sheep_positions.copy()
        try:
            start=time.perf_counter()
            with patch.object(HighLevelController,'_with_mode_forecasts',side_effect=AssertionError('Main-thread forecast')):
                # Child process has its own class; the patched main method must never be used.
                for frame in range(20):
                    d=c.decide(replace(obs,frame=frame))
                    self.assertEqual(d.source,'rule')
            self.assertLess(time.perf_counter()-start,.3)
            self.assertFalse(c.backend.job['future'].done())
            np.testing.assert_array_equal(obs.sheep_positions,before)
        finally:c.close()

    def test_physics_advances_while_slow_worker_is_running(self):
        c=self.make();c.backend.warm();obs=self.observation()
        with temporary_config(NUM_SHEEP=3,NUM_ABNORMAL_A=1,NUM_ABNORMAL_B=1,NUM_ABNORMAL_C=1):
            env=SwarmEnv(c)
            env.sheep_pos=obs.sheep_positions.copy();env.sheep_vel=obs.sheep_velocities.copy()
            env.dog_pos=obs.dog_position.copy();env.dog_vel=obs.dog_velocity.copy()
            env.goal_pos=obs.goal_position.copy();env.sheep_status=np.array(obs.sheep_statuses,dtype=object)
            initial=env.sheep_pos.copy()
            try:
                clock=RealtimeClock(100)
                for frame in range(20):
                    clock.tick(frame);env.step(obs.alarms)
                self.assertEqual(env.current_frame,20)
                self.assertFalse(np.array_equal(env.sheep_pos,initial))
                self.assertFalse(c.backend.job['future'].done())
                self.assertTrue(all(row['decision_source']=='rule' for row in env.control_data))
            finally:env.close()

    def test_cancel_does_not_queue_or_apply_an_obsolete_answer(self):
        c=self.make();c.backend.warm()
        try:
            c.decide(self.observation(0))
            deadline=time.perf_counter()+2
            while not c.backend.job['future'].running():
                self.assertLess(time.perf_counter(),deadline)
                time.sleep(.001)
            c.decide(replace(self.observation(1),alarms=()))
            self.assertIsNone(c.pending_request)
            c.decide(self.observation(2))
            self.assertEqual(c.request_counter,1)
            self.assertIsNone(c.active_mode)
            self.assertFalse(c.backend.can_submit())
        finally:c.close()

    def test_arrival_validation_also_runs_off_main_and_can_deliver(self):
        c=self.make(fn=selection_job);c.backend.warm()
        try:
            c.decide(self.observation(0))
            limit=time.perf_counter()+5
            frame=1
            while not any(e['event']=='pipeline_delivered' for e in c.backend.events):
                self.assertLess(time.perf_counter(),limit)
                c.decide(self.observation(frame));frame+=1
                time.sleep(.005)
            deliveries=[e for e in c.backend.events if e['event']=='pipeline_delivered']
            self.assertGreaterEqual(deliveries[0]['frame'],2)
            self.assertNotEqual(c.backend.validation_result['worker_pid'],multiprocessing.current_process().pid)
            self.assertEqual(deliveries[0]['response']['hold_frames'],30)
        finally:c.close()

    def test_stale_validation_is_rejected_without_waiting_or_repredicting(self):
        executor=ThreadPoolExecutor(1);c=self.make(executor)
        try:
            c.backend.validation_result={'frame':0,'summary':{}}
            mode,hold,reason=c._validate_response({},31,(),{})
            self.assertEqual(reason,'background_validation_stale')
            self.assertIsNone(mode)
        finally:c.close();executor.shutdown()

    def test_target_arriving_in_goal_during_validation_cannot_be_taken_over(self):
        executor=ThreadPoolExecutor(1);c=self.make(executor,selection_job)
        try:
            obs=self.observation();summary=c._observe_state(obs,tuple(obs.alarms))
            c.latest_observation=obs
            request=HighLevelRequest('one',0,tuple(obs.alarms),'flock_splitting',summary)
            state=snapshot(c,obs,summary)
            rich=selection_job(state,request,public_settings(c.settings),'selector',None)['request']
            response=dict(request_id='one',trigger_frame=0,mode='HARD_TO_GUIDE',target_sheep_id=1,hold_frames=30)
            validation=validation_job(state,rich,response,public_settings(c.settings))
            c.pending_request=rich;c.backend.validation_result=validation
            positions=obs.sheep_positions.copy();positions[1]=obs.goal_position
            current=replace(obs,frame=1,sheep_positions=positions)
            base=c._observe_state(current,tuple(current.alarms))
            merged=c._with_mode_forecasts(current,base,response=response)
            self.assertEqual(merged['target_selection_error'],'target_no_longer_eligible')
            self.assertEqual(c._validate_response(response,1,current.alarms,merged)[2],'target_no_longer_eligible')
        finally:c.close();executor.shutdown()

    def test_close_discards_pending_results_and_flushes_credential_free_journal(self):
        c=self.make();c.backend.warm()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events.jsonl';c.backend.set_audit_path(path)
            c.decide(self.observation(0));c.close()
            text=path.read_text()
            self.assertIn('pipeline_canceled',text)
            self.assertTrue('discarded_background_result' in text or '"canceled_before_start": true' in text)
            self.assertNotIn('pipeline_delivered',text)


if __name__=='__main__':unittest.main()
