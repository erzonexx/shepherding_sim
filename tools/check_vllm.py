"""Check the actual local Chat Completions adapter without running physics."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from high_level_controller import HighLevelRequest
from vllm_backend import VLLMBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    permissions = parser.add_mutually_exclusive_group()
    permissions.add_argument("--llm-mode-only", action="store_true")
    permissions.add_argument("--llm-target-only", action="store_true")
    parser.add_argument("--vllm-compact-output", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    settings = SimpleNamespace(**{key: value for key, value in vars(config).items() if key.isupper()})
    settings.LLM_CACHE_ENABLED = False
    if args.vllm_compact_output is not None:
        settings.VLLM_COMPACT_OUTPUT = args.vllm_compact_output
    settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = not args.llm_mode_only
    settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = not args.llm_target_only
    try:
        backend = VLLMBackend(settings=settings)
    except Exception as exc:
        print(f"FAIL: setup {type(exc).__name__}; check dependencies and VLLM configuration")
        return 1
    # Synthetic state tests the protocol and permissions, not model quality.
    state = {
        "alarms": ["flock_splitting", "stagnation"],
        "rule_mode": "HARD_TO_GUIDE",
        "max_spread": 45.0,
        "max_spread_change": 2.0,
        "repair_candidates": {
            "HARD_TO_GUIDE": [{"target_id": 2, "goal_distance": 90.0, "goalward_speed": 0.01}],
            "COHESION_PRIORITY": [{"target_id": 3, "centroid_distance": 45.0, "outward_speed": 0.3}],
        },
    }
    request = HighLevelRequest("spark-smoke", 0, tuple(state["alarms"]), "stagnation", state)
    try:
        backend.submit(request)
        deadline = time.monotonic() + settings.LLM_API_TIMEOUT_SECONDS + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = backend.poll(request.request_id, 0)
            if result is None:
                time.sleep(0.05)
        if result is None:
            print("FAIL: polling deadline exceeded")
            return 1
        if "backend_error" in result:
            print("FAIL: " + result["backend_error"])
            print("HTTP 400: inspect vLLM structured-output/chat-template support. HTTP 401: check VLLM_API_KEY locally.")
            return 1
        fields = ("mode", "target_sheep_id", "hold_frames", "latency_seconds", "cache_hit", "input_tokens", "output_tokens")
        print("PASS: Chat Completions JSON schema and local validation")
        print(json.dumps({key: result.get(key) for key in fields}, indent=2))
        return 0
    finally:
        backend.close()
        backend.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
