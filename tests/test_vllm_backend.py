from concurrent.futures import Future
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI

import config
from high_level_controller import HighLevelRequest, build_controller
from main import parse_args as parse_main
from run_experiments import parse_args as parse_batch
from vllm_backend import VLLMBackend


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class VLLMBackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = SimpleNamespace(**{key: value for key, value in vars(config).items() if key.isupper()})
        self.settings.LLM_CACHE_ENABLED = False
        self.settings.VLLM_COMPACT_OUTPUT = False
        self.settings.VLLM_INPUT_FORMAT = "json"
        self.settings.VLLM_PREFIX_PRIMING = False
        self.decision = {"mode": "HARD_TO_GUIDE", "target_sheep_id": 2,
                         "hold_frames": 30, "reason": "lagging candidate"}
        self.calls = []
        self.finish = "stop"
        self.content = None
        self.error = None

    def handler(self, request):
        self.calls.append((request.url.path, json.loads(request.content)))
        if self.error:
            return httpx2.Response(400, json={"error": {"message": self.error}})
        if request.url.path == "/v1/completions":
            count = len(self.calls[-1][1]["prompt"])
            return httpx2.Response(200, json={
                "id": "primer", "object": "text_completion", "created": 0, "model": "test-qwen",
                "choices": [{"index": 0, "finish_reason": "length", "text": "ignored"}],
                "usage": {"prompt_tokens": count, "completion_tokens": 1, "total_tokens": count + 1},
            })
        return httpx2.Response(200, json={
            "id": "chat_test", "object": "chat.completion", "created": 0,
            "model": "test-qwen", "choices": [{"index": 0, "finish_reason": self.finish,
                "message": {"role": "assistant", "content": self.content if self.content is not None else json.dumps(self.decision)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
        })

    def backend(self):
        client = OpenAI(api_key="dummy-local-test", base_url="http://127.0.0.1:8000/v1",
                        max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(self.handler)))
        self.addCleanup(client.close)
        return VLLMBackend(settings=self.settings, client=client, executor=ImmediateExecutor(),
                           cache_path=Path(self.directory.name) / "cache.jsonl")

    def request(self, identifier="high-1", alarms=("flock_splitting", "stagnation")):
        return HighLevelRequest(identifier, 10, alarms, "stagnation", {
            "rule_mode": "HARD_TO_GUIDE", "alarms": list(alarms),
            "repair_candidates": {"HARD_TO_GUIDE": [{"target_id": 2}], "COHESION_PRIORITY": [{"target_id": 3}]},
        })

    def result(self, backend, request=None):
        request = request or self.request()
        backend.submit(request)
        return backend.poll(request.request_id, 11)

    def test_sdk_serializes_chat_schema_and_maps_metadata(self):
        result = self.result(self.backend())
        self.assertNotIn("backend_error", result)
        self.assertEqual(result["target_sheep_id"], 2)
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (100, 25))
        path, body = self.calls[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual([item["role"] for item in body["messages"]], ["system", "user"])
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["max_tokens"], 512)
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        for field in ("input", "instructions", "text", "store", "reasoning", "max_output_tokens"):
            self.assertNotIn(field, body)

    def test_bad_output_is_rejected_before_cache(self):
        original = copy.deepcopy(self.decision)
        invalid = [dict(original, hold_frames=True), dict(original, hold_frames=61),
                   dict(original, target_sheep_id=True), dict(original, target_sheep_id=9),
                   dict(original, mode="TARGET_TRACKING"), dict(original, reason="x" * 161),
                   dict(original, extra="unexpected"), {key: value for key, value in original.items() if key != "target_sheep_id"}]
        backend = self.backend()
        for index, decision in enumerate(invalid):
            with self.subTest(decision=decision):
                self.decision = decision
                self.assertIn("backend_error", self.result(backend, self.request(str(index))))
        for content in ("<think>reason</think>{}", "[]", "", "not JSON"):
            self.content = content
            self.assertIn("backend_error", self.result(backend))
        self.assertFalse(backend.cache_path.exists())

    def test_truncated_valid_json_is_not_accepted(self):
        self.finish = "length"
        backend = self.backend()
        self.assertIn("truncated", self.result(backend)["backend_error"])
        self.assertFalse(backend.cache_path.exists())

    def test_alarm_mode_enum_is_authoritative(self):
        backend = self.backend()
        result = self.result(backend, self.request(alarms=("flock_splitting",)))
        self.assertIn("backend_error", result)
        enum = self.calls[0][1]["response_format"]["json_schema"]["schema"]["properties"]["mode"]["enum"]
        self.assertNotIn("HARD_TO_GUIDE", enum)

    def test_mode_only_requires_null_target(self):
        self.settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        backend = self.backend()
        self.assertIn("backend_error", self.result(backend))
        self.decision["target_sheep_id"] = None
        self.assertNotIn("backend_error", self.result(backend))
        self.assertIn("MODE-ONLY ABLATION", self.calls[-1][1]["messages"][0]["content"])

    def test_target_only_preserves_rule_mode(self):
        self.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        backend = self.backend()
        self.assertNotIn("backend_error", self.result(backend))
        schema = self.calls[0][1]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["mode"]["enum"], ["HARD_TO_GUIDE"])
        self.decision.update(mode="COHESION_PRIORITY", target_sheep_id=3)
        self.assertIn("backend_error", self.result(backend, self.request("high-2")))

    def test_error_body_never_enters_control_log(self):
        self.error = "Bearer private-token-from-server"
        backend = self.backend()
        result = self.result(backend)
        self.assertIn("HTTP 400", result["backend_error"])
        self.assertNotIn("private-token", json.dumps(result))
        self.assertFalse(backend.cache_path.exists())

    def test_cache_isolated_by_endpoint_and_reuses_valid_decisions(self):
        self.settings.LLM_CACHE_ENABLED = True
        backend = self.backend()
        self.assertFalse(self.result(backend)["cache_hit"])
        self.assertTrue(self.result(backend, self.request("high-2"))["cache_hit"])
        self.assertEqual(len(self.calls), 1)
        key = backend._cache_key(self.request().state_summary)
        backend.base_url = "http://127.0.0.1:9000/v1"
        self.assertNotEqual(key, backend._cache_key(self.request().state_summary))

    def test_compact_wire_restores_controller_fields_and_usage(self):
        self.settings.VLLM_COMPACT_OUTPUT = True
        self.decision = {"m": 2, "t": 2, "h": 30}
        result = self.result(self.backend())
        self.assertNotIn("backend_error", result)
        self.assertEqual((result["mode"], result["target_sheep_id"], result["hold_frames"]),
                         ("HARD_TO_GUIDE", 2, 30))
        self.assertEqual(result["reason"], "")
        self.assertEqual(result["output_tokens"], 25)
        self.assertIn("-vllm-compact-v1", result["prompt_version"])
        body = self.calls[0][1]
        props = body["response_format"]["json_schema"]["schema"]["properties"]
        self.assertEqual(set(props), {"m", "t", "h"})
        self.assertEqual(props["m"]["enum"], [0, 1, 2])

    def test_compact_invalid_values_never_reach_cache(self):
        self.settings.VLLM_COMPACT_OUTPUT = True
        backend = self.backend()
        for index, wire in enumerate([
            {"m": True, "t": 2, "h": 30}, {"m": -1, "t": 2, "h": 30},
            {"m": 3, "t": 2, "h": 30}, {"m": "2", "t": 2, "h": 30},
            {"m": 2, "t": 2, "h": True}, {"m": 2, "t": 2, "h": 61},
            {"m": 2, "t": True, "h": 30}, {"m": 2, "t": 9, "h": 30},
            {"m": 0, "t": 2, "h": 30}, {"m": 2, "h": 30},
            {"m": 2, "t": 2, "h": 30, "reason": "extra"},
        ]):
            with self.subTest(wire=wire):
                self.decision = wire
                self.assertIn("backend_error", self.result(backend, self.request(str(index))))
        self.assertFalse(backend.cache_path.exists())

    def test_compact_respects_alarm_mode_and_target_only_permissions(self):
        self.settings.VLLM_COMPACT_OUTPUT = True
        self.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        backend = self.backend()
        self.decision = {"m": 1, "t": 3, "h": 30}
        self.assertIn("backend_error", self.result(backend))
        props = self.calls[0][1]["response_format"]["json_schema"]["schema"]["properties"]
        self.assertEqual(props["m"]["enum"], [2])
        self.decision = {"m": 2, "t": 2, "h": 30}
        self.assertNotIn("backend_error", self.result(backend))
        self.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = True
        backend = self.backend()
        self.assertIn("backend_error", self.result(backend, self.request(alarms=("flock_splitting",))))
        props = self.calls[-1][1]["response_format"]["json_schema"]["schema"]["properties"]
        self.assertEqual(props["m"]["enum"], [0, 1])

    def test_compact_mode_only_requires_null_target(self):
        self.settings.VLLM_COMPACT_OUTPUT = True
        self.settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        backend = self.backend()
        self.decision = {"m": 2, "t": 2, "h": 30}
        self.assertIn("backend_error", self.result(backend))
        self.decision["t"] = None
        self.assertNotIn("backend_error", self.result(backend))
        props = self.calls[-1][1]["response_format"]["json_schema"]["schema"]["properties"]
        self.assertEqual(props["t"], {"type": "null"})

    def test_compact_and_verbose_cache_entries_are_isolated(self):
        self.settings.LLM_CACHE_ENABLED = True
        verbose = self.backend()
        self.assertFalse(self.result(verbose)["cache_hit"])
        old_key = verbose._cache_key(self.request().state_summary)
        self.settings.VLLM_COMPACT_OUTPUT = True
        compact = self.backend()
        self.assertNotEqual(old_key, compact._cache_key(self.request().state_summary))
        self.decision = {"m": 2, "t": 2, "h": 30}
        self.assertFalse(self.result(compact)["cache_hit"])
        self.assertTrue(self.result(compact, self.request("cached"))["cache_hit"])
        self.assertEqual(len(self.calls), 2)

    def test_explicit_compact_toggle_overrides_config(self):
        from run_experiments import main as batch_main
        self.assertTrue(parse_main(["--controller", "vllm", "--vllm-compact-output"]).vllm_compact_output)
        self.assertFalse(parse_batch(["--controller", "vllm", "--no-vllm-compact-output"]).vllm_compact_output)
        original = config.VLLM_COMPACT_OUTPUT
        with patch("run_experiments.run_batch", side_effect=lambda args: config.VLLM_COMPACT_OUTPUT):
            self.assertFalse(batch_main(["--controller", "vllm", "--no-vllm-compact-output"]))
        self.assertEqual(config.VLLM_COMPACT_OUTPUT, original)

    def test_cli_and_factory_work_without_cloud_keys(self):
        self.assertEqual(parse_main(["--controller", "vllm"] ).controller, "vllm")
        for flag in ("--llm-mode-only", "--llm-target-only"):
            self.assertEqual(parse_batch(["--controller", "vllm", flag]).controller, "vllm")
        with patch.dict(os.environ, {}, clear=True):
            controller = build_controller("vllm")
            self.addCleanup(controller.backend.client.close)
            self.addCleanup(controller.close)
            self.assertEqual(controller.backend.source, "vllm")
            self.assertEqual(controller.backend.model, "nvidia/Qwen3.8-27B-NVFP4")

    def prefix_file(self, backend):
        ids = [151644, 8948, 198]
        plan = {"version": 1, "model": backend.model, "enable_thinking": False,
                "includes_dynamic_content": False, "prefixes": {
                    hashlib.sha256(backend.instructions.encode()).hexdigest(): {
                        "prompt_ids": ids, "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    },
                }}
        path = Path(self.directory.name) / "prefix.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        self.settings.VLLM_PREFIX_TOKEN_FILE = str(path)
        return plan, path

    def test_prefix_priming_does_not_change_decision_messages_or_permissions(self):
        baseline = self.backend()
        self.result(baseline)
        original = self.calls[-1][1]
        self.prefix_file(baseline)
        self.settings.VLLM_PREFIX_PRIMING = True
        primed = self.backend()
        self.calls.clear()
        first = self.result(primed)
        second = self.result(primed, self.request("high-2"))
        self.assertNotIn("backend_error", first)
        self.assertEqual(first["target_sheep_id"], second["target_sheep_id"])
        self.assertEqual([x[0] for x in self.calls],
                         ["/v1/completions", "/v1/chat/completions", "/v1/chat/completions"])
        primer = self.calls[0][1]
        self.assertEqual(primer["prompt"], [151644, 8948, 198])
        self.assertNotIn("messages", primer)
        self.assertEqual(primer["max_tokens"], 1)
        for _, call in self.calls[1:]:
            self.assertEqual(call["messages"], original["messages"])
            self.assertEqual(call["response_format"], original["response_format"])
            self.assertEqual(call["cache_salt"], primer["cache_salt"])
        self.assertNotEqual(baseline._cache_key(self.request().state_summary),
                            primed._cache_key(self.request().state_summary))

    def test_prefix_file_must_match_model_instructions_and_fixed_only_marker(self):
        baseline = self.backend()
        plan, path = self.prefix_file(baseline)
        self.settings.VLLM_PREFIX_PRIMING = True
        for field, value in (("model", "another-model"), ("includes_dynamic_content", True)):
            broken = copy.deepcopy(plan)
            broken[field] = value
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "matching file"):
                self.backend()
        path.write_text(json.dumps(plan), encoding="utf-8")
        self.settings.VLLM_COMPACT_OUTPUT = True
        with self.assertRaisesRegex(RuntimeError, "matching file"):
            self.backend()
        self.assertFalse(self.calls)

    def test_failed_priming_never_produces_a_decision_and_can_retry(self):
        self.prefix_file(self.backend())
        self.settings.VLLM_PREFIX_PRIMING = True
        backend = self.backend()
        self.error = "server secret must not escape"
        result = self.result(backend)
        self.assertIn("backend_error", result)
        self.assertNotIn("server secret", json.dumps(result))
        self.assertFalse(backend._cache)
        self.assertEqual([x[0] for x in self.calls], ["/v1/completions"])
        self.error = None
        self.assertNotIn("backend_error", self.result(backend, self.request("retry")))

    def test_prefix_cli_toggle_is_vllm_only_and_batch_restores_config(self):
        from run_experiments import main as batch_main
        original = config.VLLM_PREFIX_PRIMING
        for parse in (parse_main, parse_batch):
            self.assertTrue(parse(["--controller", "vllm", "--vllm-prefix-priming"]).vllm_prefix_priming)
            self.assertFalse(parse(["--controller", "vllm", "--no-vllm-prefix-priming"]).vllm_prefix_priming)
            with self.assertRaises(SystemExit):
                parse(["--controller", "rule", "--vllm-prefix-priming"])
        with patch("run_experiments.run_batch", side_effect=lambda args: config.VLLM_PREFIX_PRIMING):
            self.assertTrue(batch_main(["--controller", "vllm", "--vllm-prefix-priming"]))
        self.assertEqual(config.VLLM_PREFIX_PRIMING, original)

    def test_tables_use_actual_state_and_original_state_permissions(self):
        from tests.test_vllm_input import sample_state
        from vllm_input import unpack
        request = self.request()
        state = sample_state()
        state["repair_candidates"]["HARD_TO_GUIDE"][0]["target_id"] = 2
        request = HighLevelRequest(request.request_id, request.trigger_frame, request.alarms, request.primary_alarm, state)
        baseline = self.backend()
        baseline_key = baseline._cache_key(state)
        self.settings.VLLM_INPUT_FORMAT = "tables"
        backend = self.backend()
        result = self.result(backend, request)
        self.assertNotIn("backend_error", result)
        payload = json.loads(self.calls[-1][1]["messages"][1]["content"])
        self.assertEqual(unpack(payload), state)
        self.assertEqual(self.calls[-1][1]["response_format"]["json_schema"]["schema"]["properties"]["mode"]["enum"],
                         ["TARGET_TRACKING", "COHESION_PRIORITY", "HARD_TO_GUIDE"])
        self.assertNotEqual(backend._cache_key(state), baseline_key)
        self.decision["target_sheep_id"] = 999
        self.assertIn("backend_error", self.result(backend, request))
        self.settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        self.decision["target_sheep_id"] = 2
        self.assertIn("backend_error", self.result(self.backend(), request))

    def test_tables_cli_is_vllm_only_and_batch_restores_configuration(self):
        from run_experiments import main as batch_main
        original = config.VLLM_INPUT_FORMAT
        for parse in (parse_main, parse_batch):
            self.assertEqual(parse(["--controller", "vllm", "--vllm-input-format", "tables"]).vllm_input_format, "tables")
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse(["--controller", "deepseek", "--vllm-input-format", "tables"])
        with patch("run_experiments.run_batch", side_effect=lambda args: config.VLLM_INPUT_FORMAT):
            self.assertEqual(batch_main(["--controller", "vllm", "--vllm-input-format", "tables"]), "tables")
        self.assertEqual(config.VLLM_INPUT_FORMAT, original)

    def test_dedup_sdk_uses_named_state_and_validates_against_original_permissions(self):
        from tests.test_vllm_input import sample_state
        from vllm_named_input import unpack
        state = sample_state()
        request = HighLevelRequest("named", 10, tuple(state["alarms"]), "stagnation", state)
        baseline = self.backend()
        self.settings.VLLM_INPUT_FORMAT = "dedup"
        backend = self.backend()
        result = self.result(backend, request)
        self.assertNotIn("backend_error", result)
        body = self.calls[-1][1]
        wire = json.loads(body["messages"][1]["content"])
        self.assertEqual(wire["state"]["hard_to_guide_candidate"], {"target_id": 2, "geometry_from_repair_candidate": True})
        self.assertEqual(unpack(wire), state)
        self.assertNotEqual(backend._cache_key(state), baseline._cache_key(state))
        self.assertIn("PERMISSIONS AND FALLBACK", body["messages"][0]["content"])
        self.decision["target_sheep_id"] = 999
        self.assertIn("backend_error", self.result(backend, request))

    def test_dedup_cli_and_ablations_preserve_permission_boundaries(self):
        from run_experiments import main as batch_main
        for parse in (parse_main, parse_batch):
            self.assertEqual(parse(["--controller", "vllm", "--vllm-input-format", "dedup"]).vllm_input_format, "dedup")
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse(["--controller", "deepseek", "--vllm-input-format", "dedup"])
        original = config.VLLM_INPUT_FORMAT
        with patch("run_experiments.run_batch", side_effect=lambda args: config.VLLM_INPUT_FORMAT):
            self.assertEqual(batch_main(["--controller", "vllm", "--vllm-input-format", "dedup"]), "dedup")
        self.assertEqual(config.VLLM_INPUT_FORMAT, original)
        self.settings.VLLM_INPUT_FORMAT = "dedup"
        self.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = False
        self.assertIn("TARGET-ONLY", self.backend().instructions)
        self.decision["mode"] = "COHESION_PRIORITY"
        self.assertIn("backend_error", self.result(self.backend()))
        self.settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = True
        self.settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        self.assertIn("MODE-ONLY", self.backend().instructions)
        self.decision["mode"] = "HARD_TO_GUIDE"
        self.assertIn("backend_error", self.result(self.backend()))


if __name__ == "__main__":
    unittest.main()
