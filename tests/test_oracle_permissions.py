"""Physics equivalence, oracle permission sets, and whole-task statistics."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import config
from detector import Detector
from environment import SwarmEnv
from experiments import oracle_permissions as experiment


def values(**overrides):
    result={k:v for k,v in vars(config).items() if k.isupper() and isinstance(v,(str,int,float,bool,type(None)))}
    result.update(NUM_SHEEP=20,NUM_ABNORMAL_A=5,NUM_ABNORMAL_B=0,NUM_ABNORMAL_C=0,
        REPAIR_ENABLED=True,HIGH_LEVEL_MODE_SELECTION_ENABLED=True,
        HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False,HIGH_LEVEL_EXPLICIT_EXECUTION=False,
        MAX_STEPS=180,DANGER_DISPERSION_THRESHOLD=0.,DANGER_MEAN_SPREAD_THRESHOLD=0.,
        DANGER_STAGNATION_FRAMES=2,DANGER_STAGNATION_THRESHOLD=1e9)
    result.update(overrides)
    return result


class OraclePermissionsTests(unittest.TestCase):
    def test_initialization_and_entire_rule_motion_equal_environment(self):
        for seed in (40001,40002):
            v=values(RANDOM_SEED=seed,USE_FIXED_SEED=True)
            expected=experiment.rule_run(seed,v,keep_trace=True)
            with patch.multiple(config,**v):
                env=SwarmEnv()
                np.testing.assert_array_equal(env.sheep_pos,expected['initial']['positions'])
                np.testing.assert_array_equal(env.goal_pos,expected['initial']['goal'])
                self.assertEqual(env.sheep_status.tolist(),expected['initial']['physical_types'])
                detector=experiment.detector_for(config)
                for step in expected['trace']:
                    alarms=detector.detect(env.sheep_pos,env.goal_pos)[0]
                    done=env.step(alarms)
                    self.assertFalse(done)
                    np.testing.assert_array_equal(env.sheep_pos,step['positions'])
                    np.testing.assert_array_equal(env.dog_pos,step['dog'])
                    self.assertEqual(env.current_mode,step['mode'])
                    self.assertEqual(env.target_sheep_id,step['target'])

    def test_rule_inputs_and_candidates_are_label_blind(self):
        v=values()
        baseline=experiment.rule_run(40001,v)
        first=baseline['first_trigger']
        obs=experiment.public_observation(first['world'],first['alarms'],first['frame'],first['current_mode'])
        self.assertEqual(obs.sheep_statuses,())
        expected=experiment.candidate_actions(first,v)
        changed=copy.deepcopy(first)
        changed['world']['physical_types']=['DO NOT READ']*20
        self.assertEqual(experiment.candidate_actions(changed,v),expected)

    def test_oracle_rule_tail_matches_every_physical_frame(self):
        v=values()
        baseline=experiment.rule_run(40001,v,keep_trace=True)
        first=baseline['first_trigger']
        action=experiment.candidate_actions(first,v)[0][0]
        result=experiment.oracle_rollout(first,action,v,trace=True)
        self.assertEqual((result['success'],result['frames']),(baseline['success'],baseline['frames']))
        for predicted,actual in zip(result['forecast']['trace'],baseline['trace'][first['frame']:]):
            np.testing.assert_array_equal(predicted['sheep_positions'],actual['positions'])
            np.testing.assert_array_equal(predicted['dog_position'],actual['dog'])
            self.assertEqual(predicted['mode'],actual['mode'])
            self.assertEqual(predicted['target_sheep_id'],actual['target'])

    def test_joint_contains_both_restricted_permission_sets(self):
        v=values()
        first=experiment.rule_run(40001,v)['first_trigger']
        actions,_=experiment.candidate_actions(first,v)
        sets={arm:{a['id'] for a in actions if arm in a['arms']} for arm in experiment.ARMS}
        self.assertLessEqual(sets['oracle_target'],sets['oracle_joint'])
        self.assertLessEqual(sets['oracle_mode'],sets['oracle_joint'])
        for action in actions:
            if action['id']=='rule':continue
            if 'oracle_target' in action['arms']:
                self.assertTrue(action['mode_follows_rule'])
            if 'oracle_mode' in action['arms']:
                self.assertIsNone(action['target'])

    def test_complete_case_can_always_decline_intervention(self):
        case=experiment.formal_case(40001,values())
        for result in case['selected'].values():
            self.assertGreaterEqual(result['success'],case['baseline']['success'])
            self.assertLessEqual(result['interventions'],1)
        for branch in case['branches']:
            self.assertLessEqual(branch['committed_frames'],30)

    def test_final_movement_does_not_add_success_observation(self):
        v=values(MAX_STEPS=2)
        first=experiment.rule_run(40001,v)['first_trigger']
        action=experiment.candidate_actions(first,v)[0][0]
        f=dict(predicted_success=True,simulated_frames=1,committed_frames=0)
        with patch.object(experiment,'forecast_execution',return_value=f):
            result=experiment.oracle_rollout(first,action,v)
        self.assertFalse(result['success'])
        self.assertEqual(result['frames'],2)

    def test_success_at_start_has_environment_frame_count(self):
        v=values()
        world=experiment.initial_world(40001,v)
        world['positions']=[world['goal']]*20
        result=experiment.rule_run(40001,v,world=world)
        self.assertTrue(result['success'])
        self.assertEqual(result['frames'],1)
        self.assertIsNone(result['first_trigger'])

    def test_paired_statistics_known_values(self):
        self.assertEqual(experiment.exact_mcnemar(0,0),1.)
        self.assertEqual(experiment.exact_mcnemar(6,0),.03125)
        lo,hi=experiment.wilson(50,100)
        self.assertAlmostEqual(lo,1-hi)
        self.assertLess(lo,.5)
        self.assertGreater(hi,.5)


if __name__=='__main__':
    unittest.main()
