"""Admission, handoff, and forecast/real-physics parity; no external clients."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import config
from control_types import ControlObservation
from detector import Detector
from environment import SwarmEnv
from high_level_controller import HighLevelController, HighLevelRequest
from low_level_controller import LowLevelController
from mode_forecast import _flock_metrics
from run_experiments import temporary_config
from target_handoff import forecast_action
from tests.anomaly_behavior import test_high_level_target_selection as fixtures
from vllm_backend import VLLMBackend


class QuietController(HighLevelController):
    def _request_ready(self, observation, state_summary):
        return False


class TargetHandoffTests(unittest.TestCase):
    def settings(self, **extra):
        values = {k:v for k,v in vars(config).items() if k.isupper()}
        values.update(HIGH_LEVEL_TARGET_HANDOFF_ENABLED=True, HIGH_LEVEL_MODE_SELECTION_ENABLED=False,
            HIGH_LEVEL_TARGET_SELECTION_ENABLED=True, HIGH_LEVEL_EXPLICIT_EXECUTION=False)
        values.update(extra)
        return SimpleNamespace(**values)

    def obs(self, frame=0):
        return fixtures.HighLevelTargetSelectionTests().observation(frame)

    def controller(self, target):
        backend = fixtures.TargetBackend(target)
        backend.hold_frames = 30
        return HighLevelController(backend=backend, settings=self.settings())

    def active(self, settings=None, hold=15):
        settings = settings or self.settings()
        backend = fixtures.TargetBackend(1)
        c = QuietController(backend=backend, settings=settings)
        obs = self.obs()
        c.active_mode, c.active_target_id = 'HARD_TO_GUIDE', 1
        c.active_alarms = tuple(obs.alarms)
        c.active_started_frame, c.active_until_frame, c.active_hold_frames = 0, hold, hold
        c.active_request = HighLevelRequest('one', 0, tuple(obs.alarms), 'flock_splitting', {})
        return c

    def test_same_target_delegates_and_rule_can_change_before_llm_hold_expires(self):
        c = self.controller(0)
        c.decide(self.obs())
        c.low_level_controller.repair_target_age = 40
        expected_low = copy.copy(c.low_level_controller)
        expected = expected_low.decide(self.obs(1))
        actual = c.decide(self.obs(1))
        self.assertEqual(actual.metadata['high_level_status'], 'delegated')
        self.assertIsNone(c.active_mode)
        np.testing.assert_array_equal(actual.navigation_point, expected.navigation_point)
        changed = replace(self.obs(2), sheep_positions=np.array([[150.,100.],[80.,100.],[-50.,100.]]))
        actual, expected = c.decide(changed), expected_low.decide(changed)
        self.assertEqual(actual.target_sheep_id, 2)
        self.assertEqual(actual.target_sheep_id, expected.target_sheep_id)
        np.testing.assert_array_equal(actual.navigation_point, expected.navigation_point)
        c.close()

    def test_single_remaining_alarm_rejects_new_takeover(self):
        c = self.controller(1)
        c.decide(self.obs())
        d = c.decide(replace(self.obs(1), alarms=('stagnation',)))
        self.assertEqual(d.metadata['rejection_reason'], 'concurrent_alarms_cleared')
        self.assertEqual(d.source, 'rule')
        self.assertEqual(d.target_sheep_id, 0)
        self.assertIsNone(c.active_mode)
        c.close()

    def test_fifteen_frame_handoff_can_reconsider_without_waiting_for_thirty(self):
        new = self.active()
        old = self.active(self.settings(HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False))
        for frame in range(15):
            self.assertEqual(new.decide(self.obs(frame)).target_sheep_id, 1)
            self.assertEqual(old.decide(self.obs(frame)).target_sheep_id, 1)
        self.assertEqual(new.low_level_controller.repair_target_age, 15)
        self.assertEqual(new.decide(self.obs(15)).target_sheep_id, 0)
        self.assertEqual(old.decide(self.obs(15)).target_sheep_id, 1)
        self.assertEqual(new.low_level_controller.repair_target_age, 1)
        new.close(); old.close()

    def test_handoff_does_not_force_switch_when_target_remains_competitive(self):
        c = self.active()
        positions = np.array([[150.,100.],[40.,100.],[120.,100.]])
        for frame in range(16):
            d = c.decide(replace(self.obs(frame), sheep_positions=positions))
            self.assertEqual(d.target_sheep_id, 1)
        self.assertEqual(d.source, 'rule')
        self.assertEqual(c.low_level_controller.repair_target_age, 16)
        c.close()

    def test_early_releases_still_work(self):
        for reason in ('mode', 'arrival', 'alarms'):
            with self.subTest(reason=reason):
                c = self.active(hold=60)
                c.decide(self.obs())
                obs = self.obs(1)
                if reason == 'mode':
                    obs = replace(obs, alarms=('flock_splitting',))
                elif reason == 'alarms':
                    obs = replace(obs, alarms=())
                else:
                    pos = obs.sheep_positions.copy(); pos[1] = obs.goal_position
                    obs = replace(obs, sheep_positions=pos)
                self.assertEqual(c.decide(obs).source, 'rule')
                self.assertIsNone(c.active_mode)
                c.close()

    def test_forecast_matches_real_dynamic_alarms_and_clean_handoff(self):
        with temporary_config(NUM_SHEEP=3,NUM_ABNORMAL_A=1,NUM_ABNORMAL_B=1,NUM_ABNORMAL_C=1,
            HIGH_LEVEL_TARGET_HANDOFF_ENABLED=True,HIGH_LEVEL_MODE_SELECTION_ENABLED=False,
            HIGH_LEVEL_TARGET_SELECTION_ENABLED=True,HIGH_LEVEL_EXPLICIT_EXECUTION=False,
            DANGER_DISPERSION_THRESHOLD=1000.,DANGER_STAGNATION_THRESHOLD=1000.):
            obs = self.obs()
            low = LowLevelController()
            history = [100.] * config.DANGER_STAGNATION_FRAMES
            before = vars(low).copy()
            f = forecast_action(obs,'HARD_TO_GUIDE',15,30,config,low,history,1)
            self.assertEqual(before, vars(low))
            c = self.active(config)
            env = SwarmEnv(c)
            env.sheep_pos,env.sheep_vel = obs.sheep_positions.copy(),obs.sheep_velocities.copy()
            env.dog_pos,env.dog_vel = obs.dog_position.copy(),obs.dog_velocity.copy()
            env.goal_pos,env.sheep_status = obs.goal_position.copy(),np.array(obs.sheep_statuses)
            detector = Detector(config.DANGER_DISPERSION_THRESHOLD,config.DANGER_MEAN_SPREAD_THRESHOLD,
                config.DANGER_STAGNATION_FRAMES,config.DANGER_STAGNATION_THRESHOLD)
            detector.history_dist_to_goal = history.copy()
            alarms=obs.alarms
            active=0
            for _ in range(30):
                env.step(alarms)
                active += env.decision_source == 'mock_llm'
                alarms=detector.detect(env.sheep_pos,env.goal_pos)[0]
            metrics=_flock_metrics(env.sheep_pos,env.goal_pos,config)
            self.assertEqual(f['committed_frames'],active)
            self.assertEqual(f['committed_frames'],15)
            self.assertEqual(f['final_centroid_distance'],round(metrics['centroid_distance'],3))
            self.assertEqual(f['final_max_spread'],round(metrics['max_spread'],3))
            env.close()

    def test_forecast_same_target_or_single_alarm_is_dynamic_rule_delegation(self):
        settings,low=self.settings(),LowLevelController(self.settings())
        for obs,target in ((self.obs(),0),(replace(self.obs(),alarms=('stagnation',)),1)):
            f=forecast_action(obs,'HARD_TO_GUIDE',30,30,settings,low,[100.]*settings.DANGER_STAGNATION_FRAMES,target)
            self.assertEqual(f['execution_policy'],'FOLLOW_RULE')
            self.assertEqual(f['committed_frames'],0)

    def test_backend_namespaces_and_original_schema_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            old=VLLMBackend(settings=self.settings(HIGH_LEVEL_TARGET_HANDOFF_ENABLED=False),
                client=object(),cache_path=Path(folder)/'old')
            new=VLLMBackend(settings=self.settings(),client=object(),cache_path=Path(folder)/'new')
            try:
                self.assertEqual(hashlib.sha256(old.instructions.encode()).hexdigest(),
                    'bc274cabb56a8ba3f00988a3f7eb936d3bab4c7823909390dfe2b8419892fdcf')
                self.assertNotEqual(old._cache_key({}),new._cache_key({}))
                self.assertIn('target-handoff-v1',new.prompt_version)
                self.assertNotIn('execution_policy',new._response_format(rule_mode='HARD_TO_GUIDE')['schema']['required'])
            finally:
                old.close();new.close()

    def test_invalid_permission_combination_is_not_silently_enabled(self):
        with self.assertRaisesRegex(ValueError,'target-only'):
            HighLevelController(settings=self.settings(HIGH_LEVEL_MODE_SELECTION_ENABLED=True))


if __name__=='__main__':
    unittest.main()
