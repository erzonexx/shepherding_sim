"""Task completion ranking and exact accelerated full-tail prediction."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import config
from control_types import ControlObservation
from execution_contract import forecast_execution
from low_level_controller import LowLevelController
from mpc_completion_controller import (completion_key,completion_outcome,full_rule_tail_forecast)


class CompletionTests(unittest.TestCase):
    def test_terminal_success_outweighs_failure(self):
        rule=dict(candidate_id='rule',task_outcome=dict(success=False,frames=2000))
        llm=dict(candidate_id='llm_progress',task_outcome=dict(success=True,frames=1999))
        self.assertLess(completion_key(llm),completion_key(rule))

    def test_failed_and_equal_finish_candidates_delegate_to_rule(self):
        for success in (False,True):
            outcome=dict(success=success,frames=1800 if success else 2000)
            rule=dict(candidate_id='rule',task_outcome=outcome)
            llm=dict(candidate_id='llm_progress',task_outcome=outcome)
            self.assertLess(completion_key(rule),completion_key(llm))

    def test_success_uses_actual_environment_budget(self):
        self.assertEqual(completion_outcome(dict(predicted_success=True,simulated_frames=2),1998,2000),dict(success=False,frames=2000))
        self.assertEqual(completion_outcome(dict(predicted_success=True,simulated_frames=1),1998,2000),dict(success=True,frames=2000))

    def test_accelerated_forecast_all_frames_equal_original(self):
        values={k:v for k,v in vars(config).items() if k.isupper()}
        values.update(HIGH_LEVEL_MODE_SELECTION_ENABLED=True,HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,
            HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False,HIGH_LEVEL_EXPLICIT_EXECUTION=False)
        settings=SimpleNamespace(**values)
        for seed in (21,43):
            rng=np.random.RandomState(seed)
            positions=rng.normal([70.,100.],28.,(20,2))
            obs=ControlObservation(positions,np.array([65.,110.]),np.array([180.,100.]),
                ('flock_splitting','stagnation'),200,tuple(['A']*5+['B']*5+['C']*5+['Normal']*5),
                'HARD_TO_GUIDE',np.zeros_like(positions),np.zeros(2))
            for mode,target,policy in (('HARD_TO_GUIDE',0,'COMMIT'),('COHESION_PRIORITY',1,'COMMIT'),
                                      ('TARGET_TRACKING',None,'COMMIT'),('HARD_TO_GUIDE',None,'FOLLOW_RULE')):
                low=LowLevelController(settings)
                history=[float(np.linalg.norm(obs.goal_position-positions.mean(axis=0)))]*150
                expected=forecast_execution(obs,mode,30,500,settings,low,history,target,
                    execution_policy=policy,trigger_alarms=obs.alarms,return_trace=True)
                actual=full_rule_tail_forecast(obs,mode,30,500,settings,low,history,target,
                    policy=policy,trigger_alarms=obs.alarms,trace=True)
                self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
