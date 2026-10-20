"""Offline checks for preservation, real queue accounting, and input semantics."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from control_types import ControlObservation
from environment import SwarmEnv
from experiment_audit import config_snapshot, begin_run_record
from high_level_controller import HighLevelController, HighLevelRequest, MockLLMBackend
from responses_api_backend import OpenAIBackend
from run_experiments import main, temporary_config
from vllm_backend import VLLMBackend


class ExperimentRecordingTests(unittest.TestCase):
    def test_separate_batches_preserve_raw_data_and_refuse_overwrite(self):
        original = (config.DATA_LOG_DIR, config.CONTROL_LOG_DIR, config.LOG_RETENTION_ENABLED)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ('first', 'second'):
                output = root / arm / 'summary.csv'
                with contextlib.redirect_stdout(io.StringIO()):
                    main(['--scenario', 'normal', '--seeds', '70', '71',
                          '--max-steps', '3', '--max-log-files', '1', '--output', str(output)])
                rows = pd.read_csv(output)
                self.assertEqual(len(rows), 2)
                metadata = json.loads(output.with_suffix('.csv.metadata.json').read_text())
                self.assertFalse(metadata['config']['LOG_RETENTION_ENABLED'])
                source = Path(metadata['source_snapshot'])
                for name, digest in metadata['source_sha256'].items():
                    self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
                for row in rows.itertuples():
                    self.assertTrue(Path(row.Data).is_relative_to(output.parent))
                    self.assertEqual(len(pd.read_parquet(row.Control)), 3)
                    run = json.loads(Path(row.RunMetadata).read_text())
                    self.assertEqual(run['status'], 'completed')
                    self.assertEqual(run['config']['NUM_ABNORMAL_A'], 0)
                    self.assertEqual(run['seed'], row.Seed)
                if arm == 'first':
                    protected = {Path(p): Path(p).read_bytes() for p in rows.Data}
            for path, data in protected.items():
                self.assertEqual(path.read_bytes(), data)
            before = output.read_bytes()
            with self.assertRaises(SystemExit):
                main(['--output', str(output), '--scenario', 'normal', '--max-steps', '3'])
            self.assertEqual(output.read_bytes(), before)
        self.assertEqual((config.DATA_LOG_DIR, config.CONTROL_LOG_DIR, config.LOG_RETENTION_ENABLED), original)

    def test_retention_disabled_preserves_more_than_limit(self):
        with tempfile.TemporaryDirectory() as directory, temporary_config(
            DATA_LOG_DIR=directory, CONTROL_LOG_DIR=directory,
            LOG_RETENTION_ENABLED=False, MAX_LOG_FILES=1,
        ):
            env = SwarmEnv()
            env.step()
            for stamp in ('first', 'second'):
                env.save_data_to_parquet(stamp)
                env.save_control_to_parquet(stamp)
            self.assertEqual(len(list(Path(directory).glob('*.parquet'))), 4)

    def test_actual_backend_overrides_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'VLLM_MODEL': 'actual-local-model'}):
            backend = VLLMBackend(client=object(), cache_path=Path(directory) / 'cache')
            controller = HighLevelController(backend=backend)
            try:
                path = Path(directory) / 'run.json'
                begin_run_record(path, controller, config)
                record = json.loads(path.read_text())
                self.assertEqual(record['backend']['model'], 'actual-local-model')
                self.assertEqual(record['backend']['input_format'], config.VLLM_INPUT_FORMAT)
                self.assertIn('v12-relative-outward', record['backend']['prompt_version'])
            finally:
                controller.close()
        safe = config_snapshot(SimpleNamespace(
            API_KEY='never-record', ACCESS_TOKEN='never-record',
            VLLM_BASE_URL='https://user:password@example.org/v1?api_key=never-record',
            LLM_MAX_OUTPUT_TOKENS=100,
        ))
        self.assertNotIn('never-record', json.dumps(safe))
        self.assertEqual(safe['VLLM_BASE_URL'], 'https://example.org/v1')
        self.assertEqual(safe['LLM_MAX_OUTPUT_TOKENS'], 100)

    def test_queue_and_canceled_running_call_remain_visible(self):
        entered, release = threading.Event(), threading.Event()
        class Provider:
            def create(self, **kwargs):
                if not entered.is_set():
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('test gate')
                return SimpleNamespace(output_text=json.dumps({
                    'mode': 'HARD_TO_GUIDE', 'target_sheep_id': None,
                    'hold_frames': 30, 'reason': 'test',
                }), model='server-model')
        request = HighLevelRequest('one', 10, ('stagnation',), 'stagnation', {'rule_mode': 'HARD_TO_GUIDE'})
        with tempfile.TemporaryDirectory() as directory, temporary_config(LLM_CACHE_ENABLED=False):
            executor = ThreadPoolExecutor(max_workers=1)
            backend = OpenAIBackend(client=SimpleNamespace(responses=Provider()), executor=executor,
                                    cache_path=Path(directory) / 'cache')
            path = Path(directory) / 'requests.jsonl'
            backend.set_audit_path(path)
            try:
                backend.submit(request)
                self.assertTrue(entered.wait(2))
                backend.cancel('one')
                backend.submit(replace(request, request_id='two'))
                release.set()
                executor.shutdown(wait=True)
                result = backend.poll('two', 20)
                self.assertIsNone(backend.poll('one', 20))
                self.assertGreater(result['queue_seconds'], 0)
                self.assertAlmostEqual(result['submit_to_ready_seconds'],
                                       result['queue_seconds'] + result['worker_seconds'])
                self.assertGreaterEqual(result['worker_seconds'], result['latency_seconds'])
                self.assertGreaterEqual(result['submit_to_consume_seconds'], result['submit_to_ready_seconds'])
                self.assertEqual(result['served_model'], 'server-model')
                events = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertTrue(any(e['event'] == 'response_ready' and e['request_id'] == 'one' for e in events))
                canceled = next(e for e in events if e['event'] == 'canceled')
                self.assertFalse(canceled['canceled_before_start'])
            finally:
                release.set()
                executor.shutdown(wait=True)

    def test_application_frame_is_first_use_and_rejection_has_frame_age(self):
        backend = MockLLMBackend(latency_frames=1, hold_frames=30)
        controller = HighLevelController(backend=backend)
        controller._validate_response = lambda *args: ('HARD_TO_GUIDE', 30, None)
        with temporary_config(NUM_SHEEP=3, NUM_ABNORMAL_A=0, NUM_ABNORMAL_B=0, NUM_ABNORMAL_C=0):
            env = SwarmEnv(controller=controller)
            for _ in range(3):
                env.step(('stagnation',))
            self.assertEqual(env.control_data[1]['llm_first_applied_frame'], 1)
            self.assertEqual(env.control_data[2]['llm_first_applied_frame'], 1)
            self.assertEqual(env.control_data[1]['llm_latency_frames'], 1)
            self.assertIsNotNone(env.control_data[1]['llm_observation_to_apply_seconds'])
            env.close()
            controller = HighLevelController(backend=MockLLMBackend(latency_frames=1))
            controller._validate_response = lambda *args: (None, None, 'test_rejection')
            env = SwarmEnv(controller=controller)
            env.step(('stagnation',))
            env.step(('stagnation',))
            self.assertEqual(env.control_data[1]['llm_latency_frames'], 1)
            self.assertIsNone(env.control_data[1]['llm_first_applied_frame'])
            env.close()

    def test_cache_and_error_paths_keep_end_to_end_timings(self):
        class Provider:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls > 1:
                    raise TimeoutError('not saved in journal')
                return SimpleNamespace(output_text=json.dumps({
                    'mode': 'HARD_TO_GUIDE', 'target_sheep_id': None,
                    'hold_frames': 30, 'reason': 'test',
                }))
        request = HighLevelRequest('one', 10, ('stagnation',), 'stagnation', {'rule_mode': 'HARD_TO_GUIDE'})
        with tempfile.TemporaryDirectory() as directory, temporary_config(LLM_CACHE_ENABLED=True):
            executor = ThreadPoolExecutor(max_workers=1)
            provider = Provider()
            backend = OpenAIBackend(client=SimpleNamespace(responses=provider), executor=executor,
                                    cache_path=Path(directory) / 'cache')
            journal = Path(directory) / 'requests.jsonl'
            backend.set_audit_path(journal)
            try:
                for item in (request, replace(request, request_id='cached'),
                             replace(request, request_id='error', state_summary={'rule_mode': 'HARD_TO_GUIDE', 'x': 1})):
                    backend.submit(item)
                    backend._pending[item.request_id].result(timeout=2)
                    result = backend.poll(item.request_id, 11)
                    self.assertGreaterEqual(result['submit_to_ready_seconds'], 0)
                    if item.request_id == 'cached':
                        self.assertTrue(result['cache_hit'])
                        self.assertEqual(result['latency_seconds'], 0)
                        self.assertNotIn('api_seconds', result)
                    if item.request_id == 'error':
                        self.assertIn('backend_error', result)
                        self.assertGreaterEqual(result['api_seconds'], 0)
                self.assertEqual(provider.calls, 2)
                self.assertNotIn('not saved in journal', journal.read_text())
                ready = [json.loads(line) for line in journal.read_text().splitlines()
                         if json.loads(line)['event'] == 'response_ready']
                self.assertTrue(ready[-1]['has_backend_error'])
            finally:
                executor.shutdown(wait=True)


class OutwardSemanticsTests(unittest.TestCase):
    def observation(self, velocities):
        return ControlObservation(
            sheep_positions=np.array([[40., 100.], [80., 100.], [120., 100.]]),
            sheep_velocities=np.asarray(velocities), sheep_statuses=('A', 'B', 'C'),
            dog_position=np.array([65., 110.]), goal_position=np.array([180., 100.]),
            alarms=('flock_splitting', 'stagnation'),
        )

    def summary(self, observation):
        controller = HighLevelController()
        return controller, controller._observe_state(observation, observation.alarms)

    def test_rigid_translation_is_not_dispersion(self):
        _, summary = self.summary(self.observation([[.4, .2]] * 3))
        self.assertEqual(summary['moving_outward_from_flock_count'], 0)
        for candidate in ('cohesion_candidate', 'hard_to_guide_candidate'):
            self.assertEqual(summary[candidate]['target_outward_speed'], 0)
        for geometry in summary['type_geometry'].values():
            self.assertEqual(geometry['mean_outward_speed'], 0)

    def test_outward_quantities_are_invariant_under_common_velocity(self):
        observation = self.observation([[-.2, .1], [0., -.2], [.2, .1]])
        _, first = self.summary(observation)
        _, shifted = self.summary(replace(observation, sheep_velocities=observation.sheep_velocities + [.35, .17]))
        self.assertEqual(first['moving_outward_from_flock_count'], 2)
        self.assertEqual(first['moving_outward_from_flock_count'], shifted['moving_outward_from_flock_count'])
        for status in first['type_geometry']:
            self.assertEqual(first['type_geometry'][status]['mean_outward_speed'],
                             shifted['type_geometry'][status]['mean_outward_speed'])

    def test_default_and_explicit_same_target_agree_on_speed(self):
        observation = self.observation([[.6, .1], [.3, .1], [0., .1]])
        controller, summary = self.summary(observation)
        for name, mode in (('cohesion_candidate', 'COHESION_PRIORITY'),
                           ('hard_to_guide_candidate', 'HARD_TO_GUIDE')):
            candidate = summary[name]
            explicit = controller._target_candidate(observation, mode, candidate['target_id'])
            self.assertEqual(candidate['target_outward_speed'], explicit['target_outward_speed'])


if __name__ == '__main__':
    unittest.main()
