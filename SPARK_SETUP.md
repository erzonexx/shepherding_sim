# DGX Spark / local vLLM

## Current entry point (2026-10-20)

Use NVIDIA Sync's **Qwen3.6 NVFP4 + MTP3** entry to start or stop the retained
fast service. The permanent local launcher source is
[`tools/spark/qwen36_fast_sync.py`](tools/spark/qwen36_fast_sync.py); the installed
remote command remains `~/.local/bin/shepherding-qwen36-fast-sync.py`.
The launcher runs on Spark, not on Windows. FP8 is not restored automatically.
The last verified cleanup stopped both services; it is not a live status probe.

Historical local `.codex_tmp` sessions are archived beside this project under
`../shepherding_sim_archive/cleanup_20261020/work_sessions/`. Their original paths
and byte hashes are in that archive's `organization_manifest.json`. This change
did not move remote directories, alter Sync settings or launch a model.
The completed experiment evidence remains under `logs/experiments/`.

## Remote workspace layout (2026-10-18)

Keep new remote experiment sessions under
`/home/csslab/shepherding/experiments/<experiment-name>`; do not create another
top-level `/home/csslab/shepherding-*` folder for each test. New permanent service
launchers belong under `/home/csslab/shepherding/services/`. The Windows project
remains the source of the simulation code and full experiment results.

32 finished, unmounted remote session directories (235 files) were moved into
`/home/csslab/shepherding/experiments/` with byte-for-byte SHA256 verification.
Historical scripts were preserved rather than rewritten or rerun. The original
and new paths plus per-file hashes are recorded in
`/home/csslab/shepherding/organization/move-20261018.json`; consult that mapping
when locating old remote evidence. Historical local audit files keep their
original paths and hashes.

The following paths were retained during the 2026-10-18 organization because
container mounts or, at that time, a service guardian depended on them:

- `/home/csslab/shepherding-qwen36-20261015` (FP8 service bind mounts)
- `/home/csslab/shepherding-qwen36-nvfp4-20261015` (retained NVFP4 bind mounts)
- `/home/csslab/shepherding-training-smoke-20261013` (retained training bind mounts)
- `/home/csslab/shepherding-realtime200-20261018` (live restored-service guardian)

The historical FP8 recovery policy is no longer the current policy. Do not run
old experiment managers to start today's service. Container-mounted paths must
still be checked before any future remote move; completed sessions are evidence,
not launch entry points. The 2026-10-18 organization did not restart a model,
modify container parameters, delete data or rewrite frozen experiment sources.

This adapter uses the current local controller, including Type C sheep, admissible
mode schemas, repair target selection, and mode-only / target-only ablations.
`rule` remains the default. DeepSeek and OpenAI retain their Responses API paths.

## Simulation model agreement (2026-10-16)

Future LLM simulations should preferentially use the retained optimized local
Qwen3.6 deployment: `nvidia/Qwen3.6-35B-A3B-NVFP4`, MTP3, main MoE Marlin and
draft MoE Triton (`codex-qwen36-35b-nvfp4-mtp3-20261015`). Verify `/v1/models`
and record the effective serving configuration before running comparisons.
Keep weights and optimization settings fixed across repeated experiments; do not
silently substitute FP8, Qwen3.8, a small model, or an online provider. Changes to
the model or serving recipe must be identified explicitly as a separate experiment.
The legacy adapter defaults shown below are not the current experimental model;
pass the verified Qwen3.6 model and endpoint explicitly for simulations.

## Compact output switch

`vllm` supports opt-in compact structured JSON: `{"m":2,"t":1,"h":30}`.
Mode codes are 0 = TARGET_TRACKING, 1 = COHESION_PRIORITY, 2 = HARD_TO_GUIDE.
The adapter restores the normal mode/target/hold fields and reuses the existing
validation; no explanation is requested, so `reason` is recorded as an empty
string. Mode-only and target-only permissions still apply. Malformed, truncated,
ineligible-target or disallowed-mode responses are rejected before caching.

```powershell
& ..\.venv\Scripts\python.exe main.py --controller vllm --scenario stress --seed 141 --no-video --vllm-compact-output
& ..\.venv\Scripts\python.exe main.py --controller vllm --scenario stress --seed 141 --no-video --no-vllm-compact-output
```

Both flags also work with `run_experiments.py` and `tools/check_vllm.py`.
The 10-seed pilot reduced mean API latency from 4.69s to 2.92s but success
fell from 9/10 to 7/10. This is a small regression signal, so verbose remains
the default; enable compact explicitly for further controlled evaluation.
The configured default is `VLLM_COMPACT_OUTPUT` in `config.py`; command-line
overrides affect only that invocation. Batch metadata records the effective
setting. Prompt/cache versions distinguish the two formats, so a verbose cached
decision cannot be reused as a compact decision. DeepSeek/OpenAI do not use this
switch. Existing 150-run frozen results remain unchanged.

The compact format reduces output-token count, not the GPU's token-generation
rate. Current single-request decode is about 13 tokens/s. It changes the model's
output instructions and can change choices; the paired pilot therefore checks
closed-loop behavior as well as latency, rather than claiming identical quality.

## What differs from Responses API

| Existing Responses backend | Spark Chat Completions adapter |
| --- | --- |
| `client.responses.create` | `client.chat.completions.create` |
| `instructions`, `input` | system/user `messages` |
| `text.format` | `response_format.json_schema` |
| `max_output_tokens` | `max_tokens` (independent 512-token budget) |
| `output_text` | `choices[0].message.content` |
| `status`, incomplete details | `finish_reason`; only `stop` is accepted |
| `usage.input_tokens/output_tokens` | `usage.prompt_tokens/completion_tokens` |

The schema is generated from each request's actual alarms and permission settings.
Returned JSON is checked locally before caching: exact fields, allowed mode,
integer hold range, reason length, nullable target, and supplied target candidates.
The existing controller still rechecks current eligibility and forecasts at arrival.
There is no automatic downgrade to free text or JSON-object mode when the server
rejects JSON Schema. See the [vLLM structured output documentation](https://docs.vllm.ai/en/stable/features/structured_outputs/).

The request sets `temperature=0` and
`chat_template_kwargs={"enable_thinking": false}`. This requires the deployed
Qwen template/server to support that option; the smoke check tests the actual
deployment. SDK retries are disabled to avoid hidden extra calls after timeouts.

## Step 1: use the local project and keep NVIDIA Sync connected

In PowerShell:

```powershell
Set-Location D:\programing_space\myproj\shepherding_sim
& ..\.venv\Scripts\python.exe tools/check_vllm.py
```

Defaults are `http://127.0.0.1:8000/v1` and `nvidia/Qwen3.8-27B-NVFP4`.
No DeepSeek/OpenAI key is needed. `VLLM_MODEL` and `VLLM_BASE_URL` environment
variables may override those defaults. If server authentication is enabled, set
`VLLM_API_KEY` privately; otherwise the SDK uses the dummy value `EMPTY`.
Never place credentials in source, launch arguments, screenshots or reports.
The command-line tools read process environment variables, not `.env` files;
VS Code's existing debug launches load the ignored `.env` file.

Expected: `PASS`, `cache_hit: false`, a valid mode, `target_sheep_id`, hold frames,
and latency. The state is synthetic: this checks transport and permissions, not
decision quality or simulation performance. It does not run or export physics.
Optionally check both existing ablations:

```powershell
& ..\.venv\Scripts\python.exe tools/check_vllm.py --llm-mode-only
& ..\.venv\Scripts\python.exe tools/check_vllm.py --llm-target-only
```

Mode-only must return a null target. Target-only must keep `HARD_TO_GUIDE`, the
synthetic state's rule mode. No smoke-test cache is read or written.

## Step 2: inspect one simulation

```powershell
& ..\.venv\Scripts\python.exe main.py --controller vllm --scenario mixed --seed 70 --no-video
```

Alternatively choose `Spark Qwen (local vLLM)` in the existing VS Code Run and
Debug controller menu. Type C remains available via `--scenario type-c`, or
explicit A/B/C counts as before.

The simulation continues using the fast controller while the API is pending.
Requests require concurrent splitting and stagnation alarms. A run without such
alarms may make zero API calls. A successful simulation alone does not prove the
LLM was used. Inspect the control log's `decision_source`, `llm_cache_hit`,
`llm_backend_error` and rejection information. A schema-valid response can still
be rejected because alarms/targets changed, it became stale, or forecasts reject
its choice. The fallback remains the rule controller.

## Step 3: matched comparison, after the single run works

Use the same local source, scenario, seed, repair settings, permissions and step
limit for every provider. DeepSeek credentials must already be set privately.

```powershell
& ..\.venv\Scripts\python.exe run_experiments.py --controller rule --scenario mixed --seeds 70 --max-steps 2000 --output logs/spark_trial/rule.csv
& ..\.venv\Scripts\python.exe run_experiments.py --controller vllm --scenario mixed --seeds 70 --max-steps 2000 --no-llm-cache --output logs/spark_trial/vllm.csv
& ..\.venv\Scripts\python.exe run_experiments.py --controller deepseek --scenario mixed --seeds 70 --max-steps 2000 --no-llm-cache --output logs/spark_trial/deepseek.csv
```

Each batch saves control logs and metadata. Verify actual successful/accepted
API decisions and errors as well as completion frames. Keep the 512-token vLLM
budget and 200-token existing provider budget in mind when interpreting latency;
this is an application comparison, not an equal-budget inference benchmark.
Environment model overrides appear in decision metadata; config metadata records
configured defaults. Do not infer that DeepSeek's Responses API path works merely
because the Spark smoke check passes; check that provider's own logs.

`--llm-mode-only` and `--llm-target-only` also accept `--controller vllm`.
The vLLM cache is separate at `logs/llm_cache/vllm_chat.jsonl`, and keys distinguish
endpoint, model, prompt, state, alarms, permissions and output budget.

## Troubleshooting and rollback

- Connection error: check NVIDIA Sync and the local forwarded port.
- HTTP 400: inspect the server's JSON Schema / chat template support. The tool
  suppresses raw server bodies; inspect server diagnostics privately.
- HTTP 401: check `VLLM_API_KEY` locally. Do not copy the key into chat.
- Truncation: adjust `VLLM_MAX_OUTPUT_TOKENS` in `config.py`, then repeat Step 1.
- Empty or invalid JSON / a thinking-only response: inspect the deployed template
  and reasoning parser; do not strip thinking text and silently accept it.

To switch back, select `--controller rule` or `--controller deepseek`.
The integration does not change the physics, default controller or cloud
credentials. Pre-integration copies of touched existing files are in
`D:/programing_space/myproj/shepherding_sim_archive/2026-10-16/.codex_tmp/spark-before/`; preserve any later edits before restoring them.
New adapter/test/check/document files can be removed when no longer needed.
No commit, push or GitHub update was performed by this integration.

## Verification on 2026-10-10

220 top-level unit tests passed, including 9 new adapter tests exercising the real
OpenAI SDK serialization against a mock HTTP transport. Full, mode-only and
target-only smoke checks use the actual forwarded Spark endpoint. These checks
do not establish long-horizon simulation performance; follow Steps 2 and 3.
