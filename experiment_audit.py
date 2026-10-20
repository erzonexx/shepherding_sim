"""Credential-free experiment provenance and shared timing field names."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

INPUT_SEMANTICS_VERSION = "centroid-relative-outward-v1"
TIMING_FIELDS = (
    "api_started_at_utc", "api_returned_at_utc", "api_seconds",
    "submitted_at_utc", "worker_started_at_utc", "response_ready_at_utc",
    "consumed_at_utc", "validated_at_utc", "first_applied_at_utc",
    "consumed_frame", "first_applied_frame", "queue_seconds",
    "worker_seconds", "submit_to_ready_seconds", "submit_to_consume_seconds",
    "observation_to_apply_seconds", "observation_at_utc",
)
SOURCE_NAMES = (
    "config.py", "detector.py", "environment.py", "dynamics.py",
    "control_types.py", "low_level_controller.py", "mode_forecast.py",
    "high_level_controller.py", "responses_api_backend.py", "vllm_backend.py",
    "vllm_input.py", "vllm_named_input.py", "run_experiments.py",
    "experiment_audit.py", "main.py",
    "execution_contract.py",
    "target_handoff.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config_snapshot(settings):
    """Record effective public settings, never environment credentials."""
    result = {}
    for key, value in vars(settings).items():
        if not key.isupper() or any(word in key for word in (
            "API_KEY", "SECRET", "PASSWORD", "AUTHORIZATION", "ACCESS_TOKEN"
        )):
            continue
        if isinstance(value, Path):
            value = str(value)
        if not isinstance(value, (str, int, float, bool, type(None))):
            continue
        if key.endswith("BASE_URL") and value:
            url = urlsplit(value)
            value = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
        result[key] = value
    return result


def source_hashes(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in SOURCE_NAMES}


def backend_snapshot(controller):
    backend = getattr(controller, "backend", None)
    if backend is None:
        return {"source": "rule"}
    result = {"source": backend.source, "input_semantics_version": INPUT_SEMANTICS_VERSION}
    for key in ("model", "prompt_version", "input_format", "compact_output",
                "prefix_priming", "cache_enabled", "reasoning_effort", "explicit_execution", "target_handoff_enabled"):
        if hasattr(backend, key):
            result[key] = getattr(backend, key)
    if hasattr(backend, "instructions"):
        result["instructions_sha256"] = hashlib.sha256(backend.instructions.encode()).hexdigest()
    return result


def write_json_exclusive(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=True)


def begin_run_record(path, controller, settings, **extra):
    """The manifest exists even if the run later fails or is interrupted."""
    path = Path(path)
    record = {
        "started_at_utc": utc_now(), "status": "started",
        "input_semantics_version": INPUT_SEMANTICS_VERSION,
        "backend": backend_snapshot(controller), "config": config_snapshot(settings),
        "source_sha256": source_hashes(settings.PROJECT_ROOT),
        "pacing": {"fixed_wall_time_step": False,
                   "pending_api_poll_seconds": settings.LLM_PENDING_POLL_SECONDS},
        **extra,
    }
    audit_path = path.with_suffix(".requests.jsonl")
    backend = getattr(controller, "backend", None)
    if hasattr(backend, "set_audit_path"):
        backend.set_audit_path(audit_path)
        record["requests_audit"] = str(audit_path)
    write_json_exclusive(path, record)
    return record


def finish_run_record(path, record, **outcome):
    record.update(status="completed", finished_at_utc=utc_now(), **outcome)
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")
