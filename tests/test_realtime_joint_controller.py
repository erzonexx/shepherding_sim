"""Joint authority must work without blocking physics or accepting stale replies."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time
from unittest.mock import patch
import unittest

from high_level_controller import HighLevelController
from realtime_controller import AsyncPredictionBackend, selection_job
from realtime_joint_controller import AsyncJointController
from tests.anomaly_behavior.test_high_level_target_selection import HighLevelTargetSelectionTests


def slow_selection(state,request,*args):
    time.sleep(.5)
    return {'request':request,'response':{'mode':'HARD_TO_GUIDE','hold_frames':30}}


class JointControllerTests(unittest.TestCase):
    def make(self, select_fn=slow_selection):
        fixture=HighLevelTargetSelectionTests()
        settings=fixture.settings()
        settings.HIGH_LEVEL_MODE_SELECTION_ENABLED=True
        settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED=True
        settings.HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False
        settings.HIGH_LEVEL_EXPLICIT_EXECUTION=False
        settings.LLM_CACHE_ENABLED=False
        backend=AsyncPredictionBackend(settings,kind='qwen',executor=ThreadPoolExecutor(1),select_fn=select_fn)
        return AsyncJointController(backend,settings=settings),fixture.observation()

    def test_joint_authority_can_apply_mode_and_explicit_target(self):
        c,obs=self.make()
        try:
            c.active_mode='COHESION_PRIORITY';c.active_alarms=obs.alarms
            c.active_started_frame=0;c.active_until_frame=30;c.active_hold_frames=30;c.active_target_id=2
            d=c.decide(replace(obs,frame=1))
            self.assertEqual(d.source,'vllm')
            self.assertEqual(d.mode,'COHESION_PRIORITY')
            self.assertEqual(d.target_sheep_id,2)
            self.assertFalse(c.settings.HIGH_LEVEL_TARGET_HANDOFF_ENABLED)
        finally:c.close();c.backend.executor.shutdown()

    def test_slow_worker_leaves_the_fast_loop_running(self):
        c,obs=self.make()
        # Use the deterministic selector only inside this offline pipeline test.
        c.backend.kind='selector'
        try:
            start=time.perf_counter()
            c.decide(obs)
            for frame in range(1,20):
                self.assertEqual(c.decide(replace(obs,frame=frame)).source,'rule')
            self.assertLess(time.perf_counter()-start,.3)
            self.assertFalse(c.backend.job['future'].done())
        finally:c.close();c.backend.executor.shutdown()

    def test_joint_admission_rejects_lost_alarm_and_old_validation(self):
        c,obs=self.make()
        try:
            self.assertEqual(c._validate_response({},1,('stagnation',),{})[2],'concurrent_alarms_cleared')
            c.backend.validation_result={'frame':0,'summary':{}}
            self.assertEqual(c._validate_response({},31,obs.alarms,{})[2],'background_validation_stale')
        finally:c.close();c.backend.executor.shutdown()


if __name__=='__main__':unittest.main()
