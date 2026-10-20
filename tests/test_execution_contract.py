import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import config
from control_types import ControlObservation
from detector import Detector
from environment import SwarmEnv
from execution_contract import forecast_execution
from high_level_controller import HighLevelController, HighLevelRequest, MockLLMBackend
from low_level_controller import LowLevelController
from mode_forecast import _flock_metrics
from responses_api_backend import OpenAIBackend
from run_experiments import temporary_config
from vllm_backend import VLLMBackend, VLLMOutputError


class QuietController(HighLevelController):
    def _request_ready(self, observation, state_summary):
        return False


def active_controller(observation, settings, low, mode, target, hold):
    backend = MockLLMBackend()
    backend.trigger_on_concurrent_only = True
    controller = QuietController(copy.copy(low), backend, settings)
    controller.active_mode = mode
    controller.active_alarms = tuple(observation.alarms)
    controller.active_started_frame = observation.frame
    controller.active_until_frame = observation.frame + hold
    controller.active_hold_frames = hold
    controller.active_target_id = target
    controller.active_request = HighLevelRequest('test', observation.frame, tuple(observation.alarms), None, {})
    return controller


class ExecutionContractTests(unittest.TestCase):
    def settings(self, **extra):
        values = {k: v for k, v in vars(config).items() if k.isupper()}
        values['HIGH_LEVEL_EXPLICIT_EXECUTION'] = True
        values.update(extra)
        return SimpleNamespace(**values)

    def observation(self):
        return ControlObservation(np.array([[38., 100.], [80., 100.], [122., 100.]]),
            np.array([65., 110.]), np.array([180., 100.]), ('flock_splitting', 'stagnation'),
            frame=200, sheep_statuses=('A', 'B', 'C'), sheep_velocities=np.zeros((3, 2)), dog_velocity=np.zeros(2))

    def test_follow_rule_matches_changing_rule_actions_and_internal_memory(self):
        settings = self.settings()
        obs = self.observation()
        low = LowLevelController(settings)
        low.decide(obs)
        reference = copy.copy(low)
        controller = QuietController(low, MockLLMBackend(latency_frames=1), settings)
        request = HighLevelRequest('one', 200, obs.alarms, 'flock_splitting', {'rule_mode': 'HARD_TO_GUIDE'})
        controller.pending_request = request
        controller.backend.pending['one'] = request
        old_poll = controller.backend.poll
        def poll(identifier, frame):
            answer = old_poll(identifier, frame)
            answer.update(mode='HARD_TO_GUIDE', target_sheep_id=None,
                          execution_policy='FOLLOW_RULE', hold_frames=30)
            return answer
        controller.backend.poll = poll
        from dataclasses import replace
        for offset in range(1, 40):
            current = replace(obs, frame=200 + offset,
                              alarms=obs.alarms if offset < 5 else ('flock_splitting',))
            expected, actual = reference.decide(current), controller.decide(current)
            self.assertEqual((actual.mode, actual.target_sheep_id), (expected.mode, expected.target_sheep_id))
            np.testing.assert_array_equal(actual.navigation_point, expected.navigation_point)
            self.assertEqual(low.repair_target_age, reference.repair_target_age)
            if offset == 1:
                self.assertEqual(actual.metadata['high_level_status'], 'delegated')
            self.assertIsNone(controller.active_mode)

    def test_commit_current_target_is_still_an_intentional_hold(self):
        settings = self.settings()
        observation = self.observation()
        low = LowLevelController(settings)
        target = copy.copy(low).decide(observation, mode_override='HARD_TO_GUIDE').target_sheep_id
        controller = active_controller(observation, settings, low, 'HARD_TO_GUIDE', target, 30)
        decision = controller.decide(observation)
        self.assertEqual(decision.target_sheep_id, target)
        self.assertEqual(controller.active_until_frame, observation.frame + 30)

    def test_forecast_matches_real_early_release_and_does_not_mutate_low_state(self):
        with temporary_config(NUM_SHEEP=3, NUM_ABNORMAL_A=1, NUM_ABNORMAL_B=1, NUM_ABNORMAL_C=1,
                              HIGH_LEVEL_EXPLICIT_EXECUTION=True, DANGER_DISPERSION_THRESHOLD=1000.,
                              DANGER_STAGNATION_THRESHOLD=1000.):
            observation = self.observation()
            low = LowLevelController()
            low.decide(observation)
            before = (low.repair_target_mode, low.repair_target_id, low.repair_target_age)
            history = [100.] * config.DANGER_STAGNATION_FRAMES
            forecast = forecast_execution(observation, 'COHESION_PRIORITY', 30, 30, config, low, history)
            self.assertEqual(before, (low.repair_target_mode, low.repair_target_id, low.repair_target_age))
            controller = active_controller(observation, config, low, 'COHESION_PRIORITY', None, 30)
            env = SwarmEnv(controller)
            env.sheep_pos, env.sheep_vel = observation.sheep_positions.copy(), observation.sheep_velocities.copy()
            env.dog_pos, env.dog_vel = observation.dog_position.copy(), observation.dog_velocity.copy()
            env.goal_pos, env.current_frame = observation.goal_position.copy(), observation.frame
            env.sheep_status = np.array(observation.sheep_statuses)
            detector = Detector(config.DANGER_DISPERSION_THRESHOLD, config.DANGER_MEAN_SPREAD_THRESHOLD,
                                config.DANGER_STAGNATION_FRAMES, config.DANGER_STAGNATION_THRESHOLD)
            detector.history_dist_to_goal = history.copy()
            alarms = observation.alarms
            for _ in range(30):
                env.step(alarms)
                alarms = detector.detect(env.sheep_pos, env.goal_pos)[0]
            final = _flock_metrics(env.sheep_pos, env.goal_pos, config)
            self.assertEqual(forecast['final_max_spread'], round(final['max_spread'], 3))
            self.assertEqual(forecast['final_centroid_distance'], round(final['centroid_distance'], 3))
            self.assertEqual(forecast['committed_frames'], 15)
            self.assertEqual(forecast['release_offset'], 15)
            self.assertEqual(forecast['release_reason'], 'selected_alarm_cleared')

    def test_follow_rule_forecast_and_shorter_commit_are_different_policies(self):
        settings, observation = self.settings(), self.observation()
        low = LowLevelController(settings)
        history = [100.] * settings.DANGER_STAGNATION_FRAMES
        follow = forecast_execution(observation, 'HARD_TO_GUIDE', 30, 30, settings, low, history,
                                    execution_policy='FOLLOW_RULE')
        short = forecast_execution(observation, 'HARD_TO_GUIDE', 15, 30, settings, low, history)
        self.assertEqual(follow['committed_frames'], 0)
        self.assertLessEqual(short['committed_frames'], 15)
        self.assertEqual(short['forecast_hold_frames'], 15)

    def test_missing_policy_and_invalid_delegation_are_rejected(self):
        settings, obs = self.settings(), self.observation()
        controller = QuietController(settings=settings)
        controller.pending_request = HighLevelRequest('one', 200, obs.alarms, None, {'rule_mode': 'HARD_TO_GUIDE'})
        response = dict(request_id='one', trigger_frame=200, mode='HARD_TO_GUIDE', target_sheep_id=None, hold_frames=30)
        self.assertEqual(controller._validate_response(response, 201, obs.alarms, {})[2], 'execution_policy_not_allowed')
        response.update(execution_policy='FOLLOW_RULE', target_sheep_id=0)
        self.assertEqual(controller._validate_response(response, 201, obs.alarms, {})[2], 'invalid_rule_delegation')

    def test_schema_cache_namespace_and_original_schema_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = OpenAIBackend(settings=self.settings(HIGH_LEVEL_EXPLICIT_EXECUTION=False),
                                   client=object(), cache_path=Path(directory) / 'legacy')
            explicit = OpenAIBackend(settings=self.settings(), client=object(), cache_path=Path(directory) / 'explicit')
            try:
                self.assertNotIn('execution_policy', legacy._response_format()['schema']['required'])
                self.assertIn('execution_policy', explicit._response_format()['schema']['required'])
                self.assertNotEqual(legacy._cache_key({}), explicit._cache_key({}))
            finally:
                legacy.close()
                explicit.close()

    def test_compact_input_cannot_silently_drop_execution_policy(self):
        with self.assertRaisesRegex(RuntimeError, 'original JSON'):
            VLLMBackend(settings=self.settings(), client=object(), input_format='tables')

    def test_responses_and_chat_adapters_preserve_delegation_through_cache(self):
        request = HighLevelRequest('one', 200, ('flock_splitting','stagnation'), None,
                                   {'rule_mode':'HARD_TO_GUIDE'})
        decision = dict(mode='HARD_TO_GUIDE', target_sheep_id=None, hold_frames=30,
                        reason='continue dynamic rules', execution_policy='FOLLOW_RULE')
        from dataclasses import replace
        for adapter in (OpenAIBackend,VLLMBackend):
            calls=[]
            def create(**kwargs):
                calls.append(kwargs)
                if adapter is OpenAIBackend:
                    self.assertIn('execution_policy', kwargs['text']['format']['schema']['required'])
                    return SimpleNamespace(output_text=json.dumps(decision))
                self.assertIn('execution_policy', kwargs['response_format']['json_schema']['schema']['required'])
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=json.dumps(decision)))])
            client=SimpleNamespace(responses=SimpleNamespace(create=create),
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with tempfile.TemporaryDirectory() as directory:
                backend=adapter(settings=self.settings(LLM_CACHE_ENABLED=True), client=client,
                                cache_path=Path(directory)/'cache')
                try:
                    for identifier in ('one','two'):
                        backend.submit(replace(request,request_id=identifier))
                        backend._pending[identifier].result(timeout=2)
                        result=backend.poll(identifier,201)
                        self.assertEqual(result.get('execution_policy'),'FOLLOW_RULE',result)
                    self.assertTrue(result['cache_hit'])
                    self.assertEqual(len(calls),1)
                finally:
                    backend.close()


if __name__ == '__main__':
    unittest.main()
