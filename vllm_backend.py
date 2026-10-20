"""vLLM Chat Completions adapter for the existing asynchronous controller."""

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from responses_api_backend import ResponsesAPIBackend
from vllm_input import TABLE_INSTRUCTIONS, VERSION as TABLE_VERSION, encode_json
from vllm_named_input import VERSION as NAMED_VERSION, encode_json as encode_named_json, instructions as named_instructions

MODE_CODES = ("TARGET_TRACKING", "COHESION_PRIORITY", "HARD_TO_GUIDE")


class VLLMOutputError(ValueError):
    """Validation failure whose message contains no server or request content."""


class VLLMBackend(ResponsesAPIBackend):
    source = "vllm"
    model_env = "VLLM_MODEL"
    base_url_env = "VLLM_BASE_URL"
    default_model_setting = "VLLM_MODEL"
    default_base_url_setting = "VLLM_BASE_URL"
    default_cache_path_setting = "VLLM_CACHE_PATH"

    def __init__(self, *args, compact_output=None, prefix_priming=None, input_format=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_output = (
            bool(getattr(self.settings, "VLLM_COMPACT_OUTPUT", False))
            if compact_output is None else bool(compact_output)
        )
        if self.compact_output:
            self.prompt_version += "-vllm-compact-v1"
            self.instructions += (
                "\n\nCOMPACT RESPONSE: Return only the JSON keys m, t, h. "
                "m encodes mode: 0=TARGET_TRACKING, 1=COHESION_PRIORITY, "
                "2=HARD_TO_GUIDE. t is target_sheep_id or null; h is hold_frames. "
                "Apply all mode and target permissions above using this encoding. "
                "For target-only requests encode rule_mode exactly. "
                "Omit the textual reason; do not add other fields."
            )
        self.input_format = input_format or getattr(self.settings, "VLLM_INPUT_FORMAT", "json")
        if self.input_format not in ("json", "tables", "dedup"):
            raise RuntimeError("VLLM_INPUT_FORMAT must be json, tables or dedup")
        if (self.explicit_execution or self.target_handoff_enabled) and (self.compact_output or self.input_format != 'json'):
            raise RuntimeError('Explicit execution requires original JSON input and full output fields.')
        if self.input_format == "tables":
            self.instructions += TABLE_INSTRUCTIONS
            self.prompt_version += "-" + TABLE_VERSION
        elif self.input_format == "dedup":
            self.instructions = named_instructions(self.mode_selection_enabled, self.target_selection_enabled, self.compact_output)
            self.prompt_version += "-" + NAMED_VERSION
        self.prefix_priming = (
            bool(getattr(self.settings, "VLLM_PREFIX_PRIMING", False))
            if prefix_priming is None else bool(prefix_priming)
        )
        self._prefix_lock = threading.Lock()
        self._prefix_ready = False
        self._prefix_ids = None
        self._prefix_salt = None
        self._priming_seconds = 0.0
        if self.prefix_priming:
            self._load_prefix_plan()
            self.prompt_version += "-prefix-prime-v1"

    def _load_prefix_plan(self):
        """Accept only a plan prepared for these exact instructions and model."""
        path = os.getenv("VLLM_PREFIX_TOKEN_FILE") or getattr(
            self.settings, "VLLM_PREFIX_TOKEN_FILE", ""
        )
        try:
            plan = json.loads(Path(path).read_text(encoding="utf-8"))
            key = hashlib.sha256(self.instructions.encode()).hexdigest()
            ids = plan["prefixes"][key]["prompt_ids"]
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if (plan["version"] != 1 or plan["model"] != self.model
                    or plan["enable_thinking"] is not False
                    or plan["includes_dynamic_content"] is not False
                    or not isinstance(ids, list) or not ids
                    or any(type(token) is not int or token < 0 for token in ids)
                    or digest != plan["prefixes"][key]["sha256"]):
                raise ValueError()
        except (OSError, ValueError, TypeError, KeyError):
            raise RuntimeError(
                "Prefix priming needs a matching file from tools/prepare_vllm_prefix.py; "
                "set VLLM_PREFIX_TOKEN_FILE or disable --vllm-prefix-priming."
            ) from None
        self._prefix_ids = ids
        # Private to this backend lifetime: the first call really warms its prefix.
        # A new backend (or a failed request) re-primes rather than trusting old cache.
        self._prefix_salt = "shepherd-prefix-v1-" + uuid.uuid4().hex

    def prime_prefix(self):
        """Warm fixed tokens only. A primer is never interpreted as a decision."""
        if not self.prefix_priming:
            return 0.0
        with self._prefix_lock:
            if not self._prefix_ready:
                started = time.perf_counter()
                response = self.client.completions.create(
                    model=self.model, prompt=self._prefix_ids,
                    temperature=0, max_tokens=1,
                    extra_body={"cache_salt": self._prefix_salt},
                )
                usage = self._field(response, "usage")
                if self._field(usage, "prompt_tokens") != len(self._prefix_ids):
                    raise VLLMOutputError("prefix primer token count mismatch")
                self._priming_seconds = time.perf_counter() - started
                self._prefix_ready = True
            return self._priming_seconds

    def _make_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before using vllm") from exc
        # Optional vLLM credential only; never reuse the cloud provider's key.
        return OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("VLLM_API_KEY") or "EMPTY",
            timeout=self.settings.LLM_API_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def _create_response(self, options):
        fmt = options["text"]["format"]
        wire_format = self._wire_format(fmt)
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        user_input = encode_json(options["input"]) if self.input_format == "tables" else options["input"]
        if self.input_format == "dedup":
            user_input = encode_named_json(options["input"])
        if self.prefix_priming:
            self.prime_prefix()
            extra["cache_salt"] = self._prefix_salt
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": options["instructions"]},
                    {"role": "user", "content": user_input},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {key: value for key, value in wire_format.items() if key != "type"},
                },
                max_tokens=self.settings.VLLM_MAX_OUTPUT_TOKENS,
                temperature=0,
                extra_body=extra,
            )
        except Exception:
            self._prefix_ready = False
            raise
        content = self._extract_output_text(response)
        try:
            decision = json.loads(content)
        except (TypeError, ValueError):
            raise VLLMOutputError("content is not a JSON object") from None
        self._validate_decision(decision, fmt["schema"], json.loads(options["input"]))
        return response

    def _wire_format(self, standard_format):
        if not self.compact_output:
            return standard_format
        props = standard_format["schema"]["properties"]
        return {
            "type": "json_schema", "name": "shepherd_compact_decision", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "m": {"type": "integer", "enum": [MODE_CODES.index(mode) for mode in props["mode"]["enum"]]},
                    "t": props["target_sheep_id"],
                    "h": props["hold_frames"],
                },
                "required": ["m", "t", "h"], "additionalProperties": False,
            },
        }

    def _extract_output_text(self, response):
        text = self._extract_chat_text(response)
        if not self.compact_output:
            return text
        try:
            wire = json.loads(text)
        except (TypeError, ValueError):
            raise VLLMOutputError("compact content is not a JSON object") from None
        if not isinstance(wire, dict) or set(wire) != {"m", "t", "h"}:
            raise VLLMOutputError("missing or unexpected compact decision fields")
        code = wire["m"]
        if type(code) is not int or not 0 <= code < len(MODE_CODES):
            raise VLLMOutputError("invalid compact mode code")
        # No model explanation was requested: keep reason empty, never invent it.
        return json.dumps({
            "mode": MODE_CODES[code], "target_sheep_id": wire["t"],
            "hold_frames": wire["h"], "reason": "",
        })

    def _extract_chat_text(self, response):
        choices = self._field(response, "choices", []) or []
        if len(choices) != 1:
            raise VLLMOutputError("expected one chat completion choice")
        choice = choices[0]
        if self._field(choice, "finish_reason") != "stop":
            raise VLLMOutputError("completion is truncated or unfinished; check token budget")
        message = self._field(choice, "message")
        if self._field(message, "refusal"):
            raise VLLMOutputError("completion refused")
        content = self._field(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise VLLMOutputError("completion has no JSON content")
        return content

    def _validate_decision(self, decision, schema, state):
        if not isinstance(decision, dict) or set(decision) != set(schema["required"]):
            raise VLLMOutputError("missing or unexpected decision fields")
        props = schema["properties"]
        if self.explicit_execution:
            if decision.get('execution_policy') not in props['execution_policy']['enum']:
                raise VLLMOutputError('execution_policy violates schema')
            if decision['execution_policy'] == 'FOLLOW_RULE' and (decision.get('mode') != state.get('rule_mode') or decision.get('target_sheep_id') is not None):
                raise VLLMOutputError('FOLLOW_RULE requires rule_mode and a null target')
        if decision["mode"] not in props["mode"]["enum"]:
            raise VLLMOutputError("mode is not admissible for this request")
        hold = decision["hold_frames"]
        if type(hold) is not int or not props["hold_frames"]["minimum"] <= hold <= props["hold_frames"]["maximum"]:
            raise VLLMOutputError("hold_frames violates schema")
        if not isinstance(decision["reason"], str) or len(decision["reason"]) > props["reason"]["maxLength"]:
            raise VLLMOutputError("reason violates schema")
        target = decision["target_sheep_id"]
        if target is not None:
            if type(target) is not int or target < 0 or not self.target_selection_enabled or decision["mode"] == "TARGET_TRACKING":
                raise VLLMOutputError("target_sheep_id violates schema or mode permissions")
            candidates = state.get("repair_candidates", {}).get(decision["mode"], [])
            if target not in [item["target_id"] for item in candidates]:
                raise VLLMOutputError("target_sheep_id is not a supplied candidate")

    def _response_provenance(self, response):
        if response is None:
            return {}
        choices = self._field(response, "choices", []) or []
        finish = self._field(choices[0], "finish_reason") if choices else None
        usage = self._field(response, "usage")
        result = {
            "response_id": self._field(response, "id"),
            "response_status": "completed" if finish == "stop" else "incomplete",
            "incomplete_reason": None if finish == "stop" else finish,
            "input_tokens": self._field(usage, "prompt_tokens"),
            "output_tokens": self._field(usage, "completion_tokens"),
            "served_model": self._field(response, "model"),
        }
        if self.prefix_priming:
            # Primed does not imply a hit: eviction/server restart can remove KV.
            result.update(prefix_priming=True, prefix_priming_seconds=self._priming_seconds)
        return result

    def _format_error(self, exc):
        if isinstance(exc, VLLMOutputError):
            return f"VLLMOutputError: {exc}"
        # SDK exception strings can include server bodies, headers or URLs.
        status = getattr(exc, "status_code", None)
        detail = f"HTTP {status}" if isinstance(status, int) else "request failed"
        return f"{type(exc).__name__}: {detail}; check tunnel, model and schema support"

    def _cache_key(self, state_summary, alarms=None):
        base = super()._cache_key(state_summary, alarms=alarms)
        payload = [base, self.base_url, "chat-json-schema-v1-thinking-off", self.settings.VLLM_MAX_OUTPUT_TOKENS]
        if self.compact_output:
            payload.append("compact-v1")
        if self.input_format == "tables":
            payload.append(TABLE_VERSION)
        elif self.input_format == "dedup":
            payload.append(NAMED_VERSION)
        if self.prefix_priming:
            payload.extend(["prefix-prime-v1", hashlib.sha256(json.dumps(self._prefix_ids).encode()).hexdigest()])
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
