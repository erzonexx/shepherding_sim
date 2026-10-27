"""Test the factorial switches, label-free heuristic and retained ranking parity."""
import copy
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest
import numpy as np

from mpc_controller import MODEL
from mpc_completion_controller import rank_completion_candidates
from mpc_component_ablation import (heuristic_action,arm_factors,ablation_proposal_job,
    rank_ablation_candidates,ablation_candidate_error,AblationBackend,AblationController)
from realtime_controller import public_settings
from tests import test_mpc_controller as existing_tests


class ComponentAblationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=existing_tests.MPCTests();self.c,self.b,self.state,self.request=self.fixture.fixture()
        self.values=public_settings(self.c.settings)

    def tearDown(self):self.fixture.finish(self.c,self.b)

    def bundle(self,arm='llm_mpc'):
        bundle=self.fixture.proposals(self.request)
        bundle['proposals']=bundle['proposals'][:1]
        bundle['proposals'][0]['generator']='llm'
        bundle['ablation_arm']=arm
        return bundle

    def test_retained_llm_filter_ranking_is_exact(self):
        bundle=self.bundle();expected=rank_completion_candidates(self.state,self.request,bundle,self.values)
        actual=rank_ablation_candidates(self.state,self.request,bundle,self.values,arm='llm_mpc')
        for key in ('winner_id','candidates','rejected','horizon_frames','hold_frames','snapshot_frame'):
            self.assertEqual(actual[key],expected[key],key)
        for key in ('mode','target_sheep_id','mpc_candidate_id','mpc_policy'):
            self.assertEqual(actual['response'][key],expected['response'][key])

    def test_direct_ignores_score_but_computes_identical_forecasts(self):
        def forecast(obs,mode,hold,horizon,settings,low,history,target=None,**kwargs):
            return dict(predicted_success=target is None,simulated_frames=20)
        with patch('mpc_component_ablation.full_rule_tail_forecast',side_effect=forecast) as f:
            filtered=rank_ablation_candidates(self.state,self.request,self.bundle(),self.values,arm='llm_mpc')
            direct=rank_ablation_candidates(self.state,self.request,self.bundle('llm_direct'),self.values,arm='llm_direct')
        self.assertEqual(f.call_count,4)
        self.assertEqual(filtered['winner_id'],'rule');self.assertEqual(direct['winner_id'],'llm_progress')
        self.assertEqual(direct['shadow_winner_id'],'rule');self.assertEqual(filtered['candidates'],direct['candidates'])

    def test_invalid_direct_proposal_still_falls_back(self):
        bundle=self.bundle('llm_direct');bundle['proposals'][0]['response']['target_sheep_id']=999
        result=rank_ablation_candidates(self.state,self.request,bundle,self.values,arm='llm_direct')
        self.assertEqual(result['winner_id'],'rule');self.assertEqual(result['rejected'][0]['reason'],'target_not_offered')

    def test_heuristic_uses_only_offered_geometry_and_deterministic_tie(self):
        class Guard(dict):
            def __getitem__(self,key):
                if key not in ('rule_mode',):raise AssertionError('Unexpected hidden input')
                return super().__getitem__(key)
            def get(self,key,default=None):
                if key!='repair_candidates':raise AssertionError('Unexpected hidden input')
                return super().get(key,default)
        summary=Guard(rule_mode='HARD_TO_GUIDE',repair_candidates={'HARD_TO_GUIDE':[{'target_id':2},{'target_id':1}]},hidden_types=object())
        positions=np.array([[99.,99.],[0.,0.],[0.,0.]])
        self.assertEqual(heuristic_action(summary,positions,np.array([10.,0.])),dict(mode='HARD_TO_GUIDE',target_sheep_id=1))

    def test_heuristic_does_not_claim_to_be_model_generated(self):
        raw=dict(request=self.request,response=self.bundle('heuristic_mpc'),backend_info={'proposal_count':1})
        raw['response']['proposals'][0]['response'].update(served_model=MODEL)
        with patch('mpc_component_ablation.proposal_job',return_value=copy.deepcopy(raw)) as call:
            result=ablation_proposal_job(self.state,self.request,self.values,'qwen','unused',arm='heuristic_mpc')
        p=result['response']['proposals'][0]
        self.assertEqual(p['candidate_id'],'heuristic');self.assertNotIn('served_model',p['response'])
        self.assertEqual(result['response']['clock_proposal'],raw['response']['proposals'][0])
        self.assertTrue(result['backend_info']['clock_only']);call.assert_called_once()

    def test_heuristic_validity_matches_original_geometry_checks(self):
        proposal=dict(generator='geometry-heuristic-v1',response=self.bundle()['proposals'][0]['response'])
        expected=ablation_candidate_error(proposal,self.request,self.state,self.values)
        self.assertIsNone(expected)
        proposal['response']['target_sheep_id']=999
        self.assertEqual(ablation_candidate_error(proposal,self.request,self.state,self.values),'target_not_offered')

    def test_factor_map(self):
        self.assertEqual(arm_factors('heuristic_direct'),('heuristic',False))
        self.assertEqual(arm_factors('llm_mpc'),('llm',True))
        with self.assertRaises(ValueError):arm_factors('rule')

    def test_heuristic_fast_acceptance_and_freshness_without_forecasting(self):
        with ThreadPoolExecutor(1) as executor:
            b=AblationBackend(self.c.settings,'heuristic_direct',executor=executor)
            c=AblationController(b,settings=self.c.settings)
            c.latest_observation=self.state['observation'];c.pending_request=self.request
            response=dict(request_id=self.request.request_id,trigger_frame=self.request.trigger_frame,
                mode='HARD_TO_GUIDE',target_sheep_id=0,hold_frames=30,
                mpc_candidate_id='heuristic',mpc_policy='COMMIT')
            b.validation_result=dict(frame=200,winner_id='heuristic',response=response)
            with patch('mpc_component_ablation.full_rule_tail_forecast',side_effect=AssertionError('Main-loop forecast')):
                self.assertEqual(c._validate_response(response,200,self.request.alarms,self.state['summary']),('HARD_TO_GUIDE',30,None))
                self.assertEqual(c._validate_response(response,206,self.request.alarms,self.state['summary'])[2],'background_validation_stale')
            c.pending_request=None;c.close()


if __name__=='__main__':unittest.main()
