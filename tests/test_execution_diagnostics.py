import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import config
from high_level_controller import HighLevelRequest
from tools.run_execution_diagnostics import FixedAgeVLLMBackend, MODEL


class FixedAgeDiagnosticsTests(unittest.TestCase):
    def backend(self, folder, gate=None, started=None, fail=False):
        class Provider:
            def create(self, **kwargs):
                if started is not None:
                    started.set()
                if gate is not None and not gate.wait(2):
                    raise TimeoutError('test gate')
                if fail:
                    raise TimeoutError('test failure')
                decision=dict(mode='HARD_TO_GUIDE',target_sheep_id=None,hold_frames=30,reason='test')
                return SimpleNamespace(model=MODEL,choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=json.dumps(decision)))])
        settings=SimpleNamespace(**{k:v for k,v in vars(config).items() if k.isupper()})
        settings.LLM_CACHE_ENABLED=False
        settings.HIGH_LEVEL_EXPLICIT_EXECUTION=False
        client=SimpleNamespace(chat=SimpleNamespace(completions=Provider()),close=lambda:None)
        return FixedAgeVLLMBackend(settings=settings,client=client,model=MODEL,
                                   cache_path=Path(folder)/'unused',frame_age=23)

    def request(self):
        return HighLevelRequest('one',10,('stagnation',),'stagnation',{'rule_mode':'HARD_TO_GUIDE'})

    def test_early_answer_is_withheld_until_frame23(self):
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder)
            try:
                backend.submit(self.request())
                backend._pending['one'].result(timeout=2)
                self.assertIsNone(backend.poll('one',32))
                result=backend.poll('one',33)
                self.assertEqual(result['consumed_frame']-result['trigger_frame'],23)
                self.assertEqual(len(backend.recorded_results),1)
            finally:
                backend.close()

    def test_slow_api_pauses_wall_time_at_due_frame(self):
        entered,release=threading.Event(),threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder,release,entered)
            timer=None
            try:
                backend.submit(self.request())
                self.assertTrue(entered.wait(1))
                self.assertFalse(backend._pending['one'].done())
                timer=threading.Timer(.03,release.set)
                timer.start()
                result=backend.poll('one',33)
                self.assertTrue(release.is_set())
                self.assertEqual(result['consumed_frame'],33)
            finally:
                release.set()
                if timer:
                    timer.join()
                backend.close()

    def test_canceled_running_call_is_drained_without_late_application(self):
        entered,release=threading.Event(),threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder,release,entered)
            backend.submit(self.request())
            self.assertTrue(entered.wait(1))
            backend.cancel('one')
            release.set()
            backend.close()
            self.assertIsNone(backend.poll('one',33))
            self.assertEqual(len(backend.recorded_results),1)

    def test_wrong_delivery_age_and_backend_errors_do_not_become_quality_results(self):
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder)
            try:
                backend.submit(self.request())
                with self.assertRaisesRegex(RuntimeError,'skipped'):
                    backend.poll('one',34)
            finally:
                backend.close()
            backend=self.backend(folder,fail=True)
            backend.submit(self.request())
            with self.assertRaisesRegex(RuntimeError,'partial simulation'):
                backend.poll('one',33)
            with self.assertRaisesRegex(RuntimeError,'exclude'):
                backend.close()


if __name__=='__main__':
    unittest.main()
