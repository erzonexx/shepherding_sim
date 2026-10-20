"""Asynchronous Responses API backends for mode and repair-target selection."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import config
from experiment_audit import INPUT_SEMANTICS_VERSION, utc_now


PROMPT_VERSION = "shepherd-mode-v12-relative-outward"
SYSTEM_INSTRUCTIONS = """You are the high-level mode selector for a 2D swarm
shepherding controller. The low-level controller remains active while you reason.
Calls normally contain concurrent splitting and stagnation alarms. Select exactly
one existing mode and optionally one supplied repair target, giving first priority to successful
arrival, then to limiting dispersion and unsafe boundary approaches.
The response schema's mode enum lists only modes admissible for this request.
Choose from that enum; forecasts for other modes are context, not permission to use them.

The rule_mode field is the deterministic choice that will run if you do
not provide a useful override. Prefer rule_mode unless another admissible mode
has a clear advantage in the current geometry and short-horizon estimates
without meaningfully worsening goal progress, dispersion, predicted goal
exits, or boundary safety.
Do not switch modes or targets solely because a Type A, Type B, or Type C sheep is present.
When both alarms are active, the rule fallback prioritizes HARD_TO_GUIDE;
choose COHESION_PRIORITY only when the state and forecast together justify it.

COHESION_PRIORITY collects a selected sheep, by default a predicted outward flock outlier. Prefer it
when spread is high or worsening, the cohesion candidate is moving outward, or a
Type B sheep is the main dispersion target. HARD_TO_GUIDE drives a selected sheep
toward the goal, by default the predicted goal-distance laggard. Prefer it when progress is poor,
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
goalward speed is toward the goal; outward speed is the sheep's velocity relative
to the moving flock centroid, projected away from that centroid. Positive means
separating from the flock, not merely translating with it.

The repair_candidates field offers at most three targets per repair mode,
including that mode's rule-selected target when eligible. Choose target_sheep_id
only from the chosen mode's target_id values, or null to delegate target selection
to the low-level rule. TARGET_TRACKING always requires null. Compare candidates'
geometry and individual forecast fields, not just their sheep types. Type C has
normal flocking and stronger near-dog repulsion as described in type_c_response
when present. The fast layer computes all movement; you cannot change distances,
speeds, or force coefficients.

The mode_forecasts field contains independent 30-frame shadow simulations using
the same kinematic model for all three existing modes. They are estimates, not
guarantees of long-horizon success. Prefer a mode with more
centroid_progress and inside_goal_change, and less final_max_spread,
predicted_goal_exits, predicted_splitting_frames, and peak_out_of_bounds. Treat
predicted goal exits and out-of-bounds motion as strong costs. Use static geometry
to interpret genuine forecast trade-offs. The execution interface currently
rechecks the selected mode AND target at response arrival and rejects a choice
that is Pareto-dominated by another admissible mode or the same mode's rule target; that
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
        self.target_selection_enabled = getattr(
            settings, "HIGH_LEVEL_TARGET_SELECTION_ENABLED", True
        )
        self.mode_selection_enabled = getattr(
            settings, "HIGH_LEVEL_MODE_SELECTION_ENABLED", True
        )
        self.prompt_version = PROMPT_VERSION + (
            "-target-only-v1" if not self.mode_selection_enabled else
            "" if self.target_selection_enabled else "-mode-only-v2"
        )
        self.instructions = SYSTEM_INSTRUCTIONS + (
            "" if self.target_selection_enabled else
            "\n\nMODE-ONLY ABLATION: You may select only the mode and hold_frames. "
            "Always return target_sheep_id as JSON null, never a sheep ID. "
            "Repair candidates are context for mode selection only; "
            "the low-level rule selects every target."
        )
        if not self.mode_selection_enabled:
            self.instructions += (
                "\n\nTARGET-ONLY ABLATION: The rule selects the mode. "
                "Always copy rule_mode exactly into mode; you cannot switch modes. "
                "Choose target_sheep_id only from that rule mode's repair_candidates, "
                "or null to delegate to the rule. hold_frames holds only the target, "
                "never the mode; any rule mode change immediately releases it."
            )
        self.explicit_execution = bool(getattr(settings, 'HIGH_LEVEL_EXPLICIT_EXECUTION', False))
        from target_handoff import validate_settings, enabled, VERSION as HANDOFF_VERSION, INSTRUCTIONS as HANDOFF_INSTRUCTIONS
        validate_settings(settings)
        self.target_handoff_enabled = enabled(settings)
        if self.target_handoff_enabled:
            self.prompt_version += '-' + HANDOFF_VERSION
            self.instructions += HANDOFF_INSTRUCTIONS
        if self.explicit_execution:
            if not (self.mode_selection_enabled and self.target_selection_enabled):
                raise RuntimeError('Explicit execution currently requires both mode and target permissions.')
            from execution_contract import VERSION, INSTRUCTIONS
            self.prompt_version += '-' + VERSION
            self.instructions += INSTRUCTIONS
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
        self._audit_path = None
        self._audit_lock = threading.Lock()

    def set_audit_path(self, path):
        """Opt-in append-only journal; running canceled calls still record completion."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        self._audit_path = path

    def record_event(self, event, request_id, **fields):
        if self._audit_path is None:
            return
        record = {"event": event, "request_id": request_id,
                  "recorded_at_utc": utc_now(), **fields}
        with self._audit_lock, self._audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def submit(self, request):
        cache_key = self._cache_key(request.state_summary, alarms=request.alarms)
        cached = self._cache.get(cache_key)
        submitted, submitted_utc = time.perf_counter(), utc_now()
        self.record_event('submitted', request.request_id, trigger_frame=request.trigger_frame,
                          submitted_at_utc=submitted_utc, model=self.model,
                          prompt_version=self.prompt_version,
                          input_semantics_version=INPUT_SEMANTICS_VERSION,
                          state_summary=request.state_summary)
        future = self.executor.submit(
            self._execute_timed, request, cache_key, cached, submitted, submitted_utc
        )
        future._shepherd_submitted_clock = submitted
        with self._lock:
            self._pending[request.request_id] = future

    def poll(self, request_id, frame):
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or not future.done():
            return None
        with self._lock:
            self._pending.pop(request_id, None)
        result = dict(future.result())
        result.update(consumed_at_utc=utc_now(), consumed_frame=frame,
                      submit_to_consume_seconds=time.perf_counter() - future._shepherd_submitted_clock)
        self.record_event('consumed', request_id, consumed_frame=frame,
                          consumed_at_utc=result['consumed_at_utc'],
                          submit_to_consume_seconds=result['submit_to_consume_seconds'])
        return result

    def _execute_timed(self, request, cache_key, cached, submitted, submitted_utc):
        started, started_utc = time.perf_counter(), utc_now()
        self.record_event('worker_started', request.request_id, worker_started_at_utc=started_utc)
        result = (self._cached_result(request, cached) if cached is not None
                  else self._call_api(request, cache_key))
        ready = time.perf_counter()
        result.update(
            submitted_at_utc=submitted_utc, worker_started_at_utc=started_utc,
            response_ready_at_utc=utc_now(), queue_seconds=started - submitted,
            worker_seconds=ready - started, submit_to_ready_seconds=ready - submitted,
            observation_at_utc=request.observed_at_utc,
        )
        self.record_event('response_ready', request.request_id, **{
            key: value for key, value in result.items()
            if key not in ('request_id', 'backend_error')
        }, has_backend_error=bool(result.get('backend_error')))
        return result

    def cancel(self, request_id):
        with self._lock:
            future = self._pending.pop(request_id, None)
        if future is not None:
            canceled_before_start = future.cancel()
            self.record_event('canceled', request_id,
                              canceled_before_start=canceled_before_start,
                              future_done=future.done())

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
        api_timing = {}
        try:
            response_format = self._response_format(
                request.state_summary.get("rule_mode"), alarms=request.alarms
            )
            request_options = {
                "model": self.model,
                "instructions": self.instructions,
                "input": json.dumps(
                    request.state_summary,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "text": {"format": response_format},
                "max_output_tokens": self.settings.LLM_MAX_OUTPUT_TOKENS,
                "store": False,
            }
            if self.reasoning_effort:
                request_options["reasoning"] = {"effort": self.reasoning_effort}
            api_started = time.perf_counter()
            api_timing["api_started_at_utc"] = utc_now()
            try:
                response = self._create_response(request_options)
            finally:
                api_timing.update(api_returned_at_utc=utc_now(),
                                  api_seconds=time.perf_counter() - api_started)
            output_text = self._extract_output_text(response)
            if not output_text.strip():
                status = self._field(response, "status")
                incomplete_reason = self._incomplete_reason(response)
                detail = f"status={status or 'unknown'}"
                if incomplete_reason:
                    detail += f", incomplete_reason={incomplete_reason}"
                raise ValueError(f"empty response output ({detail})")
            decision = json.loads(output_text)
            if not isinstance(decision, Mapping) or decision.get("mode") not in response_format["schema"]["properties"]["mode"]["enum"]:
                raise ValueError("Response mode is not admissible for the request alarms.")
            latency = time.perf_counter() - started
            cache_value = {
                "mode": decision.get("mode"),
                "target_sheep_id": decision.get("target_sheep_id"),
                "hold_frames": decision.get("hold_frames"),
                "reason": decision.get("reason"),
            }
            if self.explicit_execution:
                policy = decision.get('execution_policy')
                from execution_contract import POLICIES
                if policy not in POLICIES:
                    raise ValueError('Invalid execution policy')
                if policy == 'FOLLOW_RULE' and (decision.get('mode') != request.state_summary.get('rule_mode') or decision.get('target_sheep_id') is not None):
                    raise ValueError('Invalid rule delegation')
                cache_value['execution_policy'] = policy
            self._save_cache(cache_key, cache_value)
            return {
                "request_id": request.request_id,
                "trigger_frame": request.trigger_frame,
                **cache_value,
                "latency_seconds": latency,
                "cache_hit": False,
                "model": self.model,
                "prompt_version": self.prompt_version,
                **api_timing,
                **self._response_provenance(response),
            }
        except Exception as exc:
            return {
                "request_id": request.request_id,
                "trigger_frame": request.trigger_frame,
                "backend_error": self._format_error(exc),
                "latency_seconds": time.perf_counter() - started,
                "cache_hit": False,
                "model": self.model,
                "prompt_version": self.prompt_version,
                **api_timing,
                **self._response_provenance(response),
            }

    def _create_response(self, request_options):
        return self.client.responses.create(**request_options)

    def _format_error(self, exc):
        return f"{type(exc).__name__}: {exc}"[:500]

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
            "served_model": self._field(response, "model"),
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
            "prompt_version": self.prompt_version,
        }

    def _response_format(self, rule_mode=None, alarms=None):
        allowed_modes = ["TARGET_TRACKING", "COHESION_PRIORITY", "HARD_TO_GUIDE"]
        if alarms is not None:
            allowed_modes = ["TARGET_TRACKING"]
            if "flock_splitting" in alarms:
                allowed_modes.append("COHESION_PRIORITY")
            if "stagnation" in alarms:
                allowed_modes.append("HARD_TO_GUIDE")
        if not self.mode_selection_enabled:
            if rule_mode not in allowed_modes:
                raise ValueError("Target-only requests require an admissible rule_mode.")
            allowed_modes = [rule_mode]
        result = {
            "type": "json_schema",
            "name": "shepherd_mode_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": allowed_modes,
                    },
                    "hold_frames": {
                        "type": "integer",
                        "minimum": self.settings.HIGH_LEVEL_MIN_HOLD_FRAMES,
                        "maximum": self.settings.HIGH_LEVEL_MAX_HOLD_FRAMES,
                    },
                    "target_sheep_id": (
                        {"type": ["integer", "null"], "minimum": 0}
                        if self.target_selection_enabled else {"type": "null"}
                    ),
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": ["mode", "target_sheep_id", "hold_frames", "reason"],
                "additionalProperties": False,
            },
        }
        if self.explicit_execution:
            from execution_contract import POLICIES
            result['schema']['properties']['execution_policy'] = {'type': 'string', 'enum': list(POLICIES)}
            result['schema']['required'].append('execution_policy')
        return result

    def _cache_key(self, state_summary, alarms=None):
        payload = {
            "source": self.source,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_version": self.prompt_version,
            "request_alarms": sorted(state_summary.get("alarms", ()) if alarms is None else alarms),
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
