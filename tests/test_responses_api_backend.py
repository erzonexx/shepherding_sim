from concurrent.futures import Future
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI

from high_level_controller import HighLevelRequest
from run_experiments import temporary_config
from responses_api_backend import DeepSeekBackend, OpenAIBackend


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        del kwargs


class FakeResponses:
    def __init__(
        self,
        output=None,
        error=None,
        gate=None,
        started=None,
        output_text=None,
        status="completed",
        incomplete_reason=None,
    ):
        self.output = output or {
            "mode": "COHESION_PRIORITY",
            "hold_frames": 60,
            "reason": "dispersion is increasing",
        }
        self.error = error
        self.gate = gate
        self.started = started
        self.output_text = output_text
        self.status = status
        self.incomplete_reason = incomplete_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=2)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            id="resp_test",
            output_text=(
                json.dumps(self.output)
                if self.output_text is None
                else self.output_text
            ),
            output=[],
            status=self.status,
            incomplete_details=(
                SimpleNamespace(reason=self.incomplete_reason)
                if self.incomplete_reason
                else None
            ),
            usage=SimpleNamespace(input_tokens=40, output_tokens=12),
        )


class FakeClient:
    def __init__(self, output=None, error=None, gate=None, started=None, **kwargs):
        self.responses = FakeResponses(
            output=output, error=error, gate=gate, started=started, **kwargs
        )


class ResponsesAPIBackendTests(unittest.TestCase):
    def request(self, request_id="high-1", alarms=("flock_splitting",)):
        return HighLevelRequest(
            request_id=request_id,
            trigger_frame=10,
            alarms=alarms,
            primary_alarm=alarms[0] if alarms else "none",
            state_summary={
                "alarms": list(alarms),
                "max_spread": 45.0,
                "max_spread_change": 5.0,
            },
        )

    def backend(self, directory, client):
        return OpenAIBackend(
            client=client,
            executor=ImmediateExecutor(),
            model="test-model",
            cache_path=Path(directory) / "cache.jsonl",
        )

    def test_structured_response_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            backend = self.backend(directory, client)
            backend.submit(self.request())
            result = backend.poll("high-1", 11)

        self.assertEqual(result["mode"], "COHESION_PRIORITY")
        self.assertEqual(result["request_id"], "high-1")
        self.assertEqual(result["input_tokens"], 40)
        self.assertFalse(result["cache_hit"])
        call = client.responses.calls[0]
        self.assertFalse(call["store"])
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertEqual(call["text"]["format"]["type"], "json_schema")

    def test_identical_state_uses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            backend = self.backend(directory, client)
            backend.submit(self.request("high-1"))
            backend.poll("high-1", 11)
            backend.submit(self.request("high-2"))
            cached = backend.poll("high-2", 12)

        self.assertEqual(len(client.responses.calls), 1)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(cached["request_id"], "high-2")

    def test_target_is_preserved_in_api_result_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(output={"mode": "HARD_TO_GUIDE", "target_sheep_id": 3,
                                        "hold_frames": 30, "reason": "nearby lagging sheep"})
            backend = self.backend(directory, client)
            backend.submit(self.request("high-1", alarms=("stagnation",)))
            response = backend.poll("high-1", 11)
            backend.submit(self.request("high-2", alarms=("stagnation",)))
            cached = backend.poll("high-2", 12)
        self.assertEqual(response["target_sheep_id"], 3)
        self.assertEqual(cached["target_sheep_id"], 3)
        self.assertTrue(cached["cache_hit"])
        schema = client.responses.calls[0]["text"]["format"]["schema"]
        self.assertIn("target_sheep_id", schema["required"])
        self.assertEqual(schema["properties"]["target_sheep_id"]["type"], ["integer", "null"])

    def test_mode_only_preserves_state_and_explicitly_restricts_target_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            full = self.backend(directory, client)
            full.submit(self.request())
            full.poll("high-1", 11)
            with temporary_config(HIGH_LEVEL_TARGET_SELECTION_ENABLED=False):
                restricted = self.backend(directory, client)
            self.assertNotEqual(full._cache_key(self.request().state_summary),
                                restricted._cache_key(self.request().state_summary))
            restricted.submit(self.request("high-2"))
            result = restricted.poll("high-2", 11)
            self.assertFalse(result["cache_hit"])
            self.assertTrue(result["prompt_version"].endswith("-mode-only-v2"))
            first, second = client.responses.calls
            self.assertTrue(second["instructions"].startswith(first["instructions"]))
            self.assertIn("Always return target_sheep_id as JSON null", second["instructions"])
            self.assertEqual(first["input"], second["input"])
            schema = second["text"]["format"]["schema"]
            self.assertEqual(schema["properties"]["target_sheep_id"], {"type": "null"})

    def test_target_only_restricts_mode_schema_and_separates_cache(self):
        request = self.request(alarms=("stagnation",))
        request.state_summary["rule_mode"] = "HARD_TO_GUIDE"
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(output={"mode": "HARD_TO_GUIDE", "target_sheep_id": 3,
                                        "hold_frames": 30, "reason": "nearby target"})
            full = self.backend(directory, client)
            full.submit(request)
            full.poll("high-1", 11)
            with temporary_config(HIGH_LEVEL_MODE_SELECTION_ENABLED=False):
                restricted = self.backend(directory, client)
            self.assertNotEqual(full._cache_key(request.state_summary),
                                restricted._cache_key(request.state_summary))
            restricted.submit(request)
            result = restricted.poll("high-1", 11)
            self.assertFalse(result["cache_hit"])
            self.assertEqual(result["target_sheep_id"], 3)
            self.assertTrue(result["prompt_version"].endswith("-target-only-v1"))
            first, second = client.responses.calls
            self.assertEqual(first["input"], second["input"])
            self.assertTrue(second["instructions"].startswith(first["instructions"]))
            self.assertIn("Always copy rule_mode exactly", second["instructions"])
            schema = second["text"]["format"]["schema"]["properties"]
            self.assertEqual(schema["mode"]["enum"], ["HARD_TO_GUIDE"])
            self.assertEqual(schema["target_sheep_id"]["type"], ["integer", "null"])
            self.assertEqual(restricted._response_format("COHESION_PRIORITY")
                             ["schema"]["properties"]["mode"]["enum"], ["COHESION_PRIORITY"])
            with self.assertRaises(ValueError):
                restricted._response_format()

    def test_mode_schema_matches_each_request_alarm_combination(self):
        cases = (((), ["TARGET_TRACKING"]),
                 (("flock_splitting",), ["TARGET_TRACKING", "COHESION_PRIORITY"]),
                 (("stagnation",), ["TARGET_TRACKING", "HARD_TO_GUIDE"]),
                 (("flock_splitting", "stagnation"), ["TARGET_TRACKING", "COHESION_PRIORITY", "HARD_TO_GUIDE"]))
        with tempfile.TemporaryDirectory() as directory, temporary_config(LLM_CACHE_ENABLED=False):
            for alarms, expected in cases:
                with self.subTest(alarms=alarms):
                    client = FakeClient(output={"mode": "TARGET_TRACKING", "hold_frames": 30})
                    backend = self.backend(directory, client)
                    request = self.request(alarms=alarms)
                    request.state_summary["alarms"] = ["unknown_summary_alarm"]
                    backend.submit(request)
                    result = backend.poll("high-1", 11)
                    self.assertNotIn("backend_error", result)
                    schema = client.responses.calls[0]["text"]["format"]["schema"]
                    self.assertEqual(schema["properties"]["mode"]["enum"], expected)
                    self.assertIn("forecasts for other modes are context", client.responses.calls[0]["instructions"])
                    backend.close()

    def test_provider_ignoring_mode_enum_is_rejected_and_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(output={"mode": "HARD_TO_GUIDE", "hold_frames": 30})
            backend = self.backend(directory, client)
            for request_id in ("high-1", "high-2"):
                backend.submit(self.request(request_id))
                result = backend.poll(request_id, 11)
                self.assertIn("not admissible", result["backend_error"])
            self.assertEqual(len(client.responses.calls), 2)
            self.assertFalse(backend.cache_path.exists())

    def test_cache_respects_authoritative_request_alarms(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(output={"mode": "TARGET_TRACKING", "hold_frames": 30})
            backend = self.backend(directory, client)
            for request_id, alarms in (("high-1", ("flock_splitting",)),
                                       ("high-2", ("stagnation",))):
                request = self.request(request_id, alarms=alarms)
                request.state_summary["alarms"] = []
                backend.submit(request)
                result = backend.poll(request_id, 11)
                self.assertFalse(result["cache_hit"])
            self.assertEqual(len(client.responses.calls), 2)

    def test_target_only_cannot_request_a_rule_mode_without_its_alarm(self):
        with tempfile.TemporaryDirectory() as directory, temporary_config(HIGH_LEVEL_MODE_SELECTION_ENABLED=False):
            backend = self.backend(directory, FakeClient())
            with self.assertRaisesRegex(ValueError, "admissible rule_mode"):
                backend._response_format("HARD_TO_GUIDE", alarms=("flock_splitting",))

    def test_api_error_becomes_backend_result(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = self.backend(
                directory, FakeClient(error=TimeoutError("test timeout"))
            )
            backend.submit(self.request())
            result = backend.poll("high-1", 11)

        self.assertIn("TimeoutError", result["backend_error"])
        self.assertEqual(result["model"], "test-model")

    def test_submit_does_not_wait_for_api_response(self):
        gate = threading.Event()
        started = threading.Event()
        client = FakeClient(gate=gate, started=started)
        with tempfile.TemporaryDirectory() as directory:
            backend = OpenAIBackend(
                client=client,
                model="test-model",
                cache_path=Path(directory) / "cache.jsonl",
            )
            before = time.perf_counter()
            backend.submit(self.request())
            elapsed = time.perf_counter() - before
            self.assertTrue(started.wait(timeout=1))
            self.assertLess(elapsed, 0.25)
            self.assertIsNone(backend.poll("high-1", 10))
            gate.set()
            for _ in range(100):
                result = backend.poll("high-1", 11)
                if result is not None:
                    break
                time.sleep(0.005)
            backend.close()

        self.assertEqual(result["mode"], "COHESION_PRIORITY")

    def test_official_sdk_serializes_responses_request(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx2.Response(
                200,
                json={
                    "id": "resp_sdk_test",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "max_output_tokens": 200,
                    "model": "test-model",
                    "output": [
                        {
                            "id": "msg_test",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": json.dumps(
                                        {
                                            "mode": "COHESION_PRIORITY",
                                            "hold_frames": 60,
                                            "reason": "spread is increasing",
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                    "parallel_tool_calls": True,
                    "previous_response_id": None,
                    "reasoning": {"effort": None, "summary": None},
                    "store": False,
                    "temperature": 1.0,
                    "text": {"format": {"type": "text"}},
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": 1.0,
                    "truncation": "disabled",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 15,
                    },
                    "metadata": {},
                },
            )

        http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
        client = OpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = self.backend(directory, client)
            backend.submit(self.request())
            result = backend.poll("high-1", 11)

        self.assertEqual(result["response_id"], "resp_sdk_test")
        self.assertEqual(
            captured["body"]["text"]["format"]["type"], "json_schema"
        )

    def test_deepseek_backend_uses_separate_provider_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekBackend(
                client=FakeClient(),
                executor=ImmediateExecutor(),
                model="deepseek-test",
                cache_path=Path(directory) / "deepseek-cache.jsonl",
            )
            backend.submit(self.request())
            result = backend.poll("high-1", 11)

        self.assertEqual(backend.source, "deepseek")
        self.assertEqual(result["model"], "deepseek-test")
        self.assertEqual(
            backend.client.responses.calls[0]["reasoning"], {"effort": "none"}
        )
        self.assertTrue(backend.trigger_on_concurrent_only)

    def test_empty_incomplete_response_preserves_diagnostic_details(self):
        client = FakeClient(
            output_text="",
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = self.backend(directory, client)
            backend.submit(self.request())
            result = backend.poll("high-1", 11)

        self.assertIn("empty response output", result["backend_error"])
        self.assertEqual(result["response_id"], "resp_test")
        self.assertEqual(result["response_status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "max_output_tokens")
        self.assertEqual(result["output_tokens"], 12)

    def test_deepseek_defaults_to_official_endpoint_and_configured_model(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            backend = DeepSeekBackend()
        try:
            self.assertEqual(backend.model, "deepseek-v4-flash")
            self.assertEqual(backend.base_url, "https://api.deepseek.com")
        finally:
            backend.close()

    def test_deepseek_requires_its_own_api_key(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": "openai-test-key"},
        ):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                DeepSeekBackend()


if __name__ == "__main__":
    unittest.main()
