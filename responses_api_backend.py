"""Asynchronous Responses API backends for high-level mode selection."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import config


PROMPT_VERSION = "shepherd-mode-v9-aligned-candidates"
SYSTEM_INSTRUCTIONS = """You are the high-level mode selector for a 2D swarm
shepherding controller. The low-level controller remains active while you reason.
Calls normally contain concurrent splitting and stagnation alarms. Select exactly
one existing mode from the supplied state, giving first priority to successful
arrival, then to limiting dispersion and unsafe boundary approaches.

The rule_mode field is the deterministic choice that will run if you do
not provide a useful override. Prefer rule_mode unless another admissible mode
has a clear advantage in the current geometry and short-horizon estimates
without meaningfully worsening goal progress, dispersion, predicted goal
exits, or boundary safety.
Do not switch modes solely because a Type A or Type B sheep is present.
When both alarms are active, the rule fallback prioritizes HARD_TO_GUIDE;
choose COHESION_PRIORITY only when the state and forecast together justify it.

COHESION_PRIORITY targets a predicted outward flock outlier. Prefer it
when spread is high or worsening, the cohesion candidate is moving outward, or a
Type B sheep is the main dispersion target. HARD_TO_GUIDE targets the sheep
predicted to remain most distant from the goal. Prefer it when progress is poor,
the hard-to-guide candidate has weak goalward motion, or a Type A sheep is the
main lagging target.
TARGET_TRACKING is appropriate only when ordinary drive/collect behavior is
already recovering the flock or a repair mode is likely to prolong a poor
trajectory. COHESION_PRIORITY requires an active flock_splitting alarm;
HARD_TO_GUIDE requires an active stagnation alarm. If one alarm appears brief or
nearly resolved, avoid relying on its repair mode because API latency may leave
that mode invalid by execution time.

Use candidate dog travel, approach alignment, and drive-point boundary margin to
avoid selecting a geometrically slow or risky intervention. Alignment near 1
means the dog is already on the desired side; negative boundary margin means the
candidate drive point lies outside the workspace. Type A means reduced dog
repulsion with normal flocking and boundary response. Type B means weak cohesion
and stronger separation. Positive
goalward speed is toward the goal; positive outward speed is away from the flock.

The mode_forecasts field contains independent 30-frame shadow simulations using
the same kinematic model for all three existing modes. They are estimates, not
guarantees of long-horizon success. Prefer a mode with more
centroid_progress and inside_goal_change, and less final_max_spread,
predicted_goal_exits, predicted_splitting_frames, and peak_out_of_bounds. Treat
predicted goal exits and out-of-bounds motion as strong costs. Use static geometry
to interpret genuine forecast trade-offs. The execution interface currently
rejects a choice that is Pareto-dominated by another admissible mode; that
filter is not itself a proof of safety.

Use the supplied values exactly and do not invent missing state. Choose a 15-60
frame hold, normally close to the 30-frame forecast horizon; use a shorter hold
for fast-changing or risky states. Keep the reason under 20 words. Do not propose
coordinates, forces, new modes, or physical-model changes."""


class ResponsesAPIBackend:
    """Run one OpenAI-compatible Responses API call without blocking physics."""

    source = "responses_api"
    trigger_on_concurrent_only = True
    api_key_env = None
    model_env = None
    base_url_env = None
    default_model_setting = None
    default_base_url_setting = None
    default_cache_path_setting = None
    reasoning_effort_setting = None

    def __init__(
        self,
        settings=config,
        client=None,
        executor=None,
        model=None,
        base_url=None,
        cache_path=None,
    ):
        self.settings = settings
        configured_model = getattr(settings, self.default_model_setting)
        configured_base_url = (
            getattr(settings, self.default_base_url_setting)
            if self.default_base_url_setting
            else None
        )
        configured_cache_path = getattr(settings, self.default_cache_path_setting)
        self.reasoning_effort = (
            getattr(settings, self.reasoning_effort_setting)
            if self.reasoning_effort_setting
            else None
        )
        self.model = model or os.getenv(self.model_env, configured_model)
        self.base_url = base_url or os.getenv(
            self.base_url_env, configured_base_url
        )
        self.cache_enabled = bool(settings.LLM_CACHE_ENABLED)
        self.cache_path = Path(cache_path or configured_cache_path)
        self.client = client or self._make_client()
        self.executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"shepherd-{self.source}"
        )
        self._owns_executor = executor is None
        self._lock = threading.Lock()
        self._pending = {}
        self._cache = self._load_cache()

    def submit(self, request):
        cache_key = self._cache_key(request.state_summary)
        cached = self._cache.get(cache_key)
        if cached is not None:
            future = self.executor.submit(
                self._cached_result, request, cached
            )
        else:
            future = self.executor.submit(self._call_api, request, cache_key)
        with self._lock:
            self._pending[request.request_id] = future

    def poll(self, request_id, frame):
        del frame
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or not future.done():
            return None
        with self._lock:
            self._pending.pop(request_id, None)
        return future.result()

    def cancel(self, request_id):
        with self._lock:
            future = self._pending.pop(request_id, None)
        if future is not None:
            future.cancel()

    def close(self):
        if self._owns_executor:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _make_client(self):
        if not os.getenv(self.api_key_env):
            raise RuntimeError(
                f"{self.api_key_env} is required when --controller "
                f"{self.source} is selected"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required; install requirements.txt first"
            ) from exc

        options = {
            "api_key": os.environ[self.api_key_env],
            "timeout": self.settings.LLM_API_TIMEOUT_SECONDS,
        }
        if self.base_url:
            options["base_url"] = self.base_url
        return OpenAI(**options)

    def _call_api(self, request, cache_key):
        started = time.perf_counter()
        response = None
        try:
            request_options = {
                "model": self.model,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": json.dumps(
                    request.state_summary,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "text": {"format": self._response_format()},
                "max_output_tokens": self.settings.LLM_MAX_OUTPUT_TOKENS,
                "store": False,
            }
            if self.reasoning_effort:
                request_options["reasoning"] = {"effort": self.reasoning_effort}
            response = self.client.responses.create(**request_options)
            output_text = self._extract_output_text(response)
            if not output_text.strip():
                status = self._field(response, "status")
                incomplete_reason = self._incomplete_reason(response)
                detail = f"status={status or 'unknown'}"
                if incomplete_reason:
                    detail += f", incomplete_reason={incomplete_reason}"
                raise ValueError(f"empty response output ({detail})")
            decision = json.loads(output_text)
            latency = time.perf_counter() - started
            cache_value = {
                "mode": decision.get("mode"),
                "hold_frames": decision.get("hold_frames"),
                "reason": decision.get("reason"),
            }
            self._save_cache(cache_key, cache_value)
            return {
                "request_id": request.request_id,
                "trigger_frame": request.trigger_frame,
                **cache_value,
                "latency_seconds": latency,
                "cache_hit": False,
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                **self._response_provenance(response),
            }
        except Exception as exc:
            return {
                "request_id": request.request_id,
                "trigger_frame": request.trigger_frame,
                "backend_error": f"{type(exc).__name__}: {exc}"[:500],
                "latency_seconds": time.perf_counter() - started,
                "cache_hit": False,
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                **self._response_provenance(response),
            }

    def _extract_output_text(self, response):
        direct = self._field(response, "output_text", "") or ""
        if direct:
            return direct
        for item in self._field(response, "output", []) or []:
            if self._field(item, "type") != "message":
                continue
            for part in self._field(item, "content", []) or []:
                if self._field(part, "type") == "output_text":
                    text = self._field(part, "text", "") or ""
                    if text:
                        return text
        return ""

    def _response_provenance(self, response):
        if response is None:
            return {}
        usage = self._field(response, "usage")
        return {
            "response_id": self._field(response, "id"),
            "response_status": self._field(response, "status"),
            "incomplete_reason": self._incomplete_reason(response),
            "input_tokens": self._field(usage, "input_tokens"),
            "output_tokens": self._field(usage, "output_tokens"),
        }

    def _incomplete_reason(self, response):
        details = self._field(response, "incomplete_details")
        return self._field(details, "reason")

    @staticmethod
    def _field(value, name, default=None):
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _cached_result(self, request, cached):
        return {
            "request_id": request.request_id,
            "trigger_frame": request.trigger_frame,
            **cached,
            "latency_seconds": 0.0,
            "cache_hit": True,
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
        }

    def _response_format(self):
        return {
            "type": "json_schema",
            "name": "shepherd_mode_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": [
                            "TARGET_TRACKING",
                            "COHESION_PRIORITY",
                            "HARD_TO_GUIDE",
                        ],
                    },
                    "hold_frames": {
                        "type": "integer",
                        "minimum": self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES,
                        "maximum": self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES,
                    },
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": ["mode", "hold_frames", "reason"],
                "additionalProperties": False,
            },
        }

    def _cache_key(self, state_summary):
        payload = {
            "source": self.source,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "hold_range": [
                self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES,
                self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES,
            ],
            "state": state_summary,
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_cache(self):
        if not self.cache_enabled or not self.cache_path.exists():
            return {}
        cache = {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    cache[record["key"]] = record["decision"]
        except (OSError, KeyError, json.JSONDecodeError):
            return {}
        return cache

    def _save_cache(self, cache_key, decision):
        if not self.cache_enabled:
            return
        with self._lock:
            if cache_key in self._cache:
                return
            self._cache[cache_key] = decision
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"key": cache_key, "decision": decision},
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError:
                pass


class OpenAIBackend(ResponsesAPIBackend):
    """OpenAI Responses API provider."""

    source = "openai"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    base_url_env = "OPENAI_BASE_URL"
    default_model_setting = "OPENAI_MODEL"
    default_base_url_setting = "OPENAI_BASE_URL"
    default_cache_path_setting = "OPENAI_CACHE_PATH"


class DeepSeekBackend(ResponsesAPIBackend):
    """DeepSeek's OpenAI-compatible Responses API provider."""

    source = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    model_env = "DEEPSEEK_MODEL"
    base_url_env = "DEEPSEEK_BASE_URL"
    default_model_setting = "DEEPSEEK_MODEL"
    default_base_url_setting = "DEEPSEEK_BASE_URL"
    default_cache_path_setting = "DEEPSEEK_CACHE_PATH"
    reasoning_effort_setting = "DEEPSEEK_REASONING_EFFORT"
