> 完整技术说明。以下命令均从项目根目录运行；旧实验描述保留作历史参考。

# Dual-Layer Autonomous Swarm Shepherding

A two-dimensional simulation of one shepherd agent guiding a heterogeneous flock
to a circular goal. The project combines a continuous kinematic controller with
an optional event-triggered LLM layer for anomaly recovery.

The implementation includes two baseline abnormal sheep types and an experimental third type:

- **Type A (weak response):** reduced dog repulsion with otherwise normal flocking and boundary response.
- **Type B (dispersing):** weak cohesion and stronger, longer-range separation.
- **Type C (near-dog sensitive, experimental):** normal flocking, but increasingly
  strong dog repulsion inside a sensitivity radius. Its speed cap stays normal.

The low-level controller remains active every frame. The high-level layer is
consulted only when stagnation and flock splitting are detected together, so API
latency never pauses the simulation.

## Architecture

方案名称、实验状态和后续计划见[方案记录](../EXPERIMENT_NOTES.md)。

```text
sheep/dog state
      |
      v
   Detector ------------------------------+
      | alarms                            |
      v                                   |
HighLevelController (optional, async)     |
      | bounded mode + repair target       |
      v                                   |
LowLevelController <--- forecast + guard -+
      |
      v
Kinematic dynamics -> next state
```

The controller has three admissible modes:

| Mode | Purpose | Typical action |
| --- | --- | --- |
| `TARGET_TRACKING` | Normal goal-directed shepherding | Drive or collect |
| `HARD_TO_GUIDE` | Recover from stagnation | Guide one difficult sheep |
| `COHESION_PRIORITY` | Recover a split flock | Collect the predicted outlier |

The rule controller maps splitting to `COHESION_PRIORITY`, stagnation to
`HARD_TO_GUIDE`, and no alarm to `TARGET_TRACKING`. When both alarms coexist,
the rule fallback now prioritizes `HARD_TO_GUIDE`; the LLM may choose a repair
mode from the same bounded set.

## Project Layout

```text
shepherding_sim/
|-- main.py                    interactive simulation
|-- run_experiments.py         reproducible C00-C08 experiment suite
|-- config.py                  simulation and controller parameters
|-- environment.py             state lifecycle and logging
|-- dynamics.py                sheep and shepherd kinematics
|-- detector.py                splitting and stagnation monitors
|-- control_types.py           controller input/output contracts
|-- low_level_controller.py    frame-by-frame vector controller
|-- high_level_controller.py   async LLM policy, forecast, and guard
|-- mode_forecast.py           short-horizon candidate evaluation
|-- responses_api_backend.py   OpenAI-compatible API adapters
|-- visualizer.py              live view, trajectory, and video output
|-- plot_analysis.py           metric plots
|-- tools/                     diagnostic and comparison utilities
|-- tests/                     unit and integration tests
|-- logs/                      generated output (ignored by Git)
`-- local_artifacts/           local archives (ignored by Git)
```

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run one visual simulation:

```powershell
python main.py --controller rule
```

Available controllers are `rule`, `mock-llm`, `openai`, `deepseek`, and `vllm`.

For the local DGX Spark Qwen Chat Completions backend, follow [SPARK_SETUP.md](../SPARK_SETUP.md).

### VSCode: Select Conditions Without Editing Code

Open this folder in VSCode and select your virtual environment using
`Python: Select Interpreter`. In **Run and Debug** (`Ctrl+Shift+D`), choose
**Shepherding - Select architecture and scenario**, then press the green play
button (or `F5`). The prompts let you choose independently:

1. Architecture: rule-based fast control, or fast control plus DeepSeek.
2. Population: Normal (0A+0B), Type A (4A+0B), Type B (0A+4B), Type C (4C),
   Mixed (2A+2B), or Stress (3A+3B). All presets except Type C use zero C sheep.
3. Seed, maximum frames, and whether to export MP4.

Use **Shepherding - Custom sheep counts** to enter arbitrary A/B/C counts.
Their sum must not exceed the flock size (20 by default).
Both launch options enable anomaly repair for either architecture. With the
dual-layer architecture, the fast controller continues every frame; actual LLM
requests still depend on concurrent alarms, so selecting DeepSeek does not
guarantee a request in every run. Keep population and seed identical for a
controller comparison.

The launch configurations read API credentials from the environment or a local
`.env` file in this project folder. For DeepSeek, that file can contain
`DEEPSEEK_API_KEY=your-key`. It is ignored by Git. The shared `launch.json`
contains no credentials. No API key is needed for rule-based control.

These options apply to **Run and Debug**, not the editor's **Run Python File**
button. Running `main.py` with no arguments continues to use `config.py`.
Selections affect only the current process and do not rewrite `config.py`.
The window title shows the controller, A/B/C counts, and seed. Effective settings
are also saved as `logs/data/run_<timestamp>.json` beside the simulation data.

The same conditions can be supplied from a terminal:

```powershell
python main.py --controller rule --scenario mixed --seed 70 --no-video
python main.py --controller deepseek --scenario mixed --seed 70 --no-video
python main.py --abnormal-a 1 --abnormal-b 6 --seed 70 --max-steps 2000
python main.py --controller rule --scenario type-c --seed 3101 --no-video
```

Use `--no-repair` for a no-repair baseline, or `--random-seed` for random initial
conditions. Scene presets change only sheep counts; they do not select an
architecture or toggle repair. The C00-C08 batch cases below retain their
original population and repair settings, always with zero C sheep. Running
`main.py` without arguments still uses the original 3A+3B+0C default.

### Experimental Type C Behavior

For dog distance `d`, Type C multiplies the normal dog-repulsion weight by
`1 + (k - 1) * clip(1 - d / R, 0, 1)`. Initially `R=30` and `k=3`.
At or beyond 30 units its response equals a normal sheep's; closer to the dog,
the force multiplier rises continuously toward 3. This does not multiply actual
speed: the normal 0.7-unit/frame cap still applies. At coincident positions,
the existing zero-direction handling remains unchanged.

C preserves normal cohesion, separation, boundary response, and goal retention.
Identities are assigned randomly once per run and remain fixed. Initial positions
use the unchanged standard deviation 15. These parameters are provisional,
not biologically fitted or selected to produce an LLM advantage. No new alarm,
repair mode, or LLM trigger has been added; C behavior is shared by live dynamics
and short-horizon forecasts. Its definition is included in high-level summaries
only when C sheep are present.

Focused tests and the first matched-seed pilot are documented in
`tests/anomaly_behavior/README.md`.

## Experiments

For a batch launched from VSCode, select **Batch - Select architecture and
scenario** in **Run and Debug** and press play. Choose architecture, population,
repair on/off, number of runs, seed mode, seed-group ID, frame limit, and outputs.
For example, Rule + Stress + repair on + 10 runs runs ten 3A+3B simulations.
**Batch - Custom sheep counts** supports arbitrary A/B/C counts.

In random mode, the seed-group ID initializes a reproducible draw of distinct
seeds. Reuse the same ID and run count when comparing architectures; change the
ID for a different sample. In sequence mode, it is the first seed (e.g. 141
with 10 runs gives 141-150). All actual seeds are printed and saved in metadata.
Both batch launch options load the project's ignored `.env` file for API keys.
These batches do not show live animations. MP4 export is optional.

The equivalent command-line options are:

```powershell
python run_experiments.py --controller rule --scenario stress --repair --runs 10 --seed-mode random --batch-seed 70
```

Scene presets specify population only; architecture and repair are independent.
Named scenes appear as `stress`, `mixed`, etc. in the summary. The original
C00-C08 cases below remain available with their fixed population/repair settings.

Run all nine predefined cases with seed 70:

```powershell
python run_experiments.py --seed 70
```

Run selected cases and several seeds:

```powershell
python run_experiments.py --cases C02 C08 --seeds 141 142 143 --controller rule
```

Add `--visuals`, `--video`, or `--analysis` only when those outputs are needed.
Each batch creates `logs/experiments/<timestamp>/summary.csv` by default.
The summary is updated after each completed run. Metadata alongside it records
the seed plan, actual seeds, case settings, base configuration, and source hashes.
Per-run data and control logs remain in `logs/data/` and `logs/control/`.
An explicit `--output` path overrides the default and is overwritten if reused.

### Experiment Matrix

| Case | Type A | Type B | Rule repair | Purpose |
| --- | ---: | ---: | --- | --- |
| C00 | 0 | 0 | Off | Normal baseline |
| C01 | 2 | 2 | Off | Mixed anomaly baseline |
| C02 | 2 | 2 | On | Mixed anomaly recovery |
| C03 | 4 | 0 | Off | Type A baseline |
| C04 | 4 | 0 | On | Type A recovery |
| C05 | 0 | 4 | Off | Type B baseline |
| C06 | 0 | 4 | On | Type B recovery |
| C07 | 3 | 3 | Off | Mixed stress baseline |
| C08 | 3 | 3 | On | Mixed stress recovery |

## LLM Layer

The LLM receives a structured state summary rather than raw trajectories. Its
response selects one admissible mode, an optional repair target, and a bounded hold time. The simulation
continues under rule control while the request is pending.

Before execution, a response is rejected when it is malformed, stale, outside
the allowed mode set, or worse than the rule candidate in the short-horizon
forecast. A minimum dwell period and bounded hold duration reduce rapid mode
switching. API responses may be cached for reproducibility.

### Repair Target Selection

For a mode-only API ablation, add `--llm-mode-only` to the batch command.
It preserves the base instructions, state summary, candidate forecasts, trigger,
and physics, adding an explicit mode-only instruction. The response schema
requires `target_sheep_id: null`. The execution
interface also rejects non-null targets. This restriction is recorded in batch
configuration and uses a separate prompt-version/cache namespace; ordinary runs
retain mode-and-target selection.

For the complementary target-only ablation, use `--llm-target-only` instead.
The API must copy `rule_mode` and can select only a repair target (or null).
The fast layer still determines the mode every frame. A rule-mode change releases
the held target immediately, even before the minimum dwell period; a response
for a no-longer-current rule mode is rejected. Input summaries and forecasts
remain unchanged. The two ablation flags are mutually exclusive.

The high-level layer can choose **how to repair and which sheep to prioritize**.
The low-level layer still computes drive points and all frame-by-frame motion.
There are no new modes, force parameters, or coordinates in the LLM response.

Each repair mode offers at most three sheep outside the goal, ranked by the
existing predictive rule and including its currently held target when eligible.
Each candidate includes its ID, type, geometry, and a separate short-horizon
forecast using that exact target. This is a bounded shortlist, not unrestricted
selection from every sheep. Response field `target_sheep_id` is a zero-based ID;
`null` delegates target selection to the existing rule. `TARGET_TRACKING`
requires `null` and retains its ordinary drive/collect logic.

At response arrival, the selected target must still be outside the goal and must
have been offered for that mode in the request. Validation uses the selected
target's current drive point and forecast, not the default target's prediction.
Invalid, no-longer-eligible, or forecast-dominated choices fall back to rule
control. During execution, the target stays selected within the existing mode
hold; goal entry releases the intervention. Forecasts also release an arrived
target and resume rule control. These remain short-horizon estimates, not a
guarantee of better final completion time.

Prompt/cache version `shepherd-mode-v10-repair-targets` separates these decisions
from the older mode-only responses. `mock-llm` remains a deterministic integration
test, not evidence of LLM performance. Run and Debug's existing DeepSeek option
uses this interface; rule runs and scene presets are unchanged.

Control logs record the selected ID, the same-frame rule target, and whether
there was a target-only override. Batch `HighLevelOverrideFrames` now counts
mode changes or target-only changes; `HighLevelTargetOverrideFrames` counts
same-mode target changes against the same-frame rule preview (with its current
target memory), not a separate never-intervened trajectory. Mode-selection
accuracy remains agreement with the existing mode reference labels; it does
not evaluate whether a sheep target was correct. No target-accuracy oracle is
claimed. Real API outcome comparisons are still needed for this extension.

### DeepSeek

Set the API key in the current PowerShell session, then start the experiment:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
python run_experiments.py --cases C08 --seeds 141 142 --controller deepseek
```

Optional environment variables are `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL`.
Use `--no-llm-cache` only when fresh provider calls are required. Never place an
API key in source files or commit it to Git.

OpenAI uses the corresponding `OPENAI_API_KEY`, `OPENAI_MODEL`, and
`OPENAI_BASE_URL` variables.

## Detection and Control

Flock splitting is reported when both conditions hold:

```text
maximum distance from flock centroid > 40
mean distance from flock centroid    > 15
```

Initial sheep positions use a Gaussian spread with standard deviation 15.
Because this is a broad initialization, a run may report splitting from frame
zero; that report describes the initial geometry, not necessarily a Type B effect.
The interactive run hides splitting-only terminal warnings during the first
30 frames; detection, control decisions, and saved alarms are unchanged.
This geometric alarm does not identify sheep types: Type B can increase spread
without crossing both danger thresholds. Use the matched detector screen in
`experiments/anomaly_behavior/run_detector_screen.py` to assess that distinction.

The default `TARGET_TRACKING` mode also collects the furthest sheep when maximum
spread exceeds `COHESION_THRESHOLD` (25). A missing splitting alarm therefore
does not mean the controller ignores every outlier.

Stagnation is reported when centroid progress toward the goal is less than
`1.0` over a 150-frame window.

Repair target selection uses a short velocity lookahead. A target is held for a
minimum period and changed only when another eligible sheep is meaningfully more
urgent. Sheep already inside the goal are deprioritized when an outside target
exists. A light goal-retention force reduces outward drift near the goal edge.

## Main Parameters

| Group | Important settings |
| --- | --- |
| Simulation | `MAX_STEPS`, `RANDOM_SEED`, `SPACE_SIZE`, `GOAL_RADIUS`, `SHEEP_START_RADIUS` |
| Population | `NUM_SHEEP`, `NUM_ABNORMAL_A`, `NUM_ABNORMAL_B`, `NUM_ABNORMAL_C` |
| Speed | `SHEEP_MAX_SPEED`, `DOG_MAX_SPEED` |
| Flocking | `WEIGHT_COHESION`, `WEIGHT_SEPARATION`, `WEIGHT_BOUNDARY` |
| Shepherd response | `WEIGHT_DOG_REPULSION`, `DOG_SENSING_RANGE` |
| Type A | `WEIGHT_DOG_REPULSION_A_FACTOR` |
| Type B | `WEIGHT_COHESION_B_FACTOR`, `WEIGHT_SEPARATION_B_FACTOR` |
| Type C (experimental) | `TYPE_C_SENSITIVITY_RADIUS`, `WEIGHT_DOG_REPULSION_C_FACTOR` |
| Detection | `DANGER_DISPERSION_THRESHOLD`, `DANGER_MEAN_SPREAD_THRESHOLD`, `DANGER_STAGNATION_*` |
| Repair geometry | `DOG_DRIVE_DISTANCE`, `HARD_TO_GUIDE_DISTANCE`, `COHESION_REPAIR_DISTANCE` |
| Target stability | `*_TARGET_MIN_HOLD_FRAMES`, `REPAIR_TARGET_SWITCH_MARGIN`, `REPAIR_TARGET_LOOKAHEAD_FRAMES` |
| High level | `HIGH_LEVEL_*`, `LLM_API_TIMEOUT_SECONDS`, `LLM_CACHE_ENABLED` |

`HIGH_LEVEL_CONCURRENT_MIN_FRAMES=1` and
`HIGH_LEVEL_ONE_REQUEST_PER_EPISODE=False` preserve immediate requests while
both alarms are active. Experimental values can debounce and limit repeat API
requests; they are not enabled by default.

For a reproducible trigger-cost ablation, add
`--concurrent-min-frames 3 --one-request-per-concurrent-episode` to a
`run_experiments.py --controller deepseek` command. These options change the
architecture trigger, not the chosen scenario or sheep behavior.

Change one parameter family at a time and compare identical seeds. A visually
desirable single trajectory is not enough to justify a parameter change.

## Diagnostics

Run a detailed rule-controller diagnostic:

```powershell
python tools\diagnose_failures.py --cases C08 --seeds 70
```

Compare legacy and predictive repair-target selection:

```powershell
python tools\diagnose_failures.py --cases C02 C04 C06 C08 --seeds 61 62 63 --compare-repair-targeting --summary-only
```

Compare two experiment summary files:

```powershell
python tools\analyze_experiment_pair.py baseline.csv candidate.csv --output-prefix logs\paired_comparison
```

## Tests

```powershell
python -m unittest discover -s tests -q
```

## Current Evidence

The rule controller is the stable baseline. Across two independent
40-seed C08 cohorts, goal-reaching within 2,000 frames was 25/80 without
alarm-driven repair, 75/80 with rule repair, and 76/80 with immediate
DeepSeek. Rule repair rescued 50 no-repair timeouts without a regression;
DeepSeek had two rescues and one regression relative to rule repair.
Only 19/80 LLM runs actually changed the applied rule mode. All five
rule runs that timed out at 2,000 frames completed when rerun with a
4,000-frame limit. These results support rule repair but do not
establish a reliable additional LLM advantage.

Detailed experiments, ablations, and local log paths are kept in
[Experiment Notes](../docs/early_experiments.md).

## Limitations

Formal batches (`run_experiments.py`) now keep raw data, controls, request journals,
and a source snapshot in `<summary name>_artifacts/` beside the summary CSV.
Batch data is exempt from rolling log cleanup; an existing output is refused.
Each run has a manifest with effective settings and the resolved backend model.
Request journals retain canceled calls, including later completion of calls that
were already running. A completed simulation does not imply every discarded API
call has finished; such completions can append to its journal afterward.

Prompt v12 defines every outward-speed input relative to the moving flock
centroid. This corrects inconsistent default-candidate and shortlist values;
the JSON field names and control/hold policies are unchanged. Older v11 cache
entries are not reused. Existing prefix-priming plans must be regenerated if that
optional feature is enabled.

`llm_latency_seconds` retains its historical worker-call meaning. New control
columns and request events distinguish queue time, API time, response readiness,
controller consumption/validation, and first physical application frame/time.
Simulation frames still do not have a fixed duration in wall-clock seconds.
These recording and input corrections do not establish improved task success.

The experimental `--explicit-execution` switch adds the required response field
`execution_policy`: `FOLLOW_RULE` declines high-level commitment and immediately
uses changing rule decisions; `COMMIT` intentionally holds the selected mode and
optional target under the existing release rules. A null target under `COMMIT`
delegates only target selection. `FOLLOW_RULE` retains the low-level controller's
original target hysteresis; it does not reset its memory. Its `hold_frames` is
ignored, and the normal retry interval controls the next request.

This experiment requires both mode and target permissions, original JSON input,
and full output fields. It has a separate prompt/cache version and remains off
by default. Its `rule_forecast` simulates dynamic rules; mode/target forecasts
simulate changing alarms, actual hold/release behavior, then rules. Arrival
forecasts use the returned hold duration and original trigger context. Future
model responses are excluded, and short forecasts are not safety guarantees.

`tools/evaluate_intervention_advantage.py --output <new directory>` runs an
offline numerical gate study on the existing trusted 100 checkpoints. It keeps
harmful, neutral and no-intervention examples, separates train/validation by
seed, and freezes train-only tuning before full 2000-frame continuation checks.
It compares rules, a learned gate, and a cheap short-forecast selector at fixed
20/50-frame ages and 30-frame holds. These are legacy-execution conditional
outcomes, not evidence for the explicit-execution profile or a live LLM policy.
The held-out states must not be reused for tuning, and the tool does not deploy
the learned gate or start LLM fine-tuning.

- The model is a 2D kinematic simulation, not a physical robot controller.
- Only two anomaly types and one shepherd are currently modeled.
- Sheep positions are not hard-clipped to the workspace. The guide-first
  rule can improve task success while increasing some Type B boundary
  excursions; an experimental outside-boundary fallback did not improve
  C08 stress success consistently and is not enabled by default.
- LLM results depend on provider latency and nondeterministic responses.
- The current dual-alarm trigger covers all Type B and C08 rule failures in
  one 80-seed audit, but also activates in successful normal runs. An alarm
  alone does not demonstrate that an LLM override is useful or cost-effective.
- Stagnation currently refers to flock-centroid progress. A naive per-sheep
  version also activates in many successful normal runs, so individual
  stagnation requires a better definition before it becomes a new alarm.
- A local short-horizon forecast mode selector regressed Type B and C08
  success on an independent 80-seed check. The forecast-based LLM rejection
  criterion should be validated separately before it is treated as a safety
  guarantee.
- The current v9 LLM prompt, candidate geometry, and stateful forecasts are
  interface corrections verified by local tests. The fresh paired API run
  above shows no clear improvement over the updated guide-first rule. The
  earlier positive LLM comparison used the former cohesion-first fallback
  and is not evidence for the current architecture.
- Repair-mode selection accuracy is undefined for the current concurrent-
  alarm API trigger: the reference labels cover single alarms only, while
  DeepSeek is called when both alarms occur. No ground-truth label or
  selection-accuracy claim should be inferred from these runs.

## Wall-clock continuous-control experiment

`tools/run_realtime_comparison.py` uses the new `realtime_controller.py` pipeline.
Request forecasts, local Qwen inference, and forecast revalidation run in separate
processes. The fast controller polls completed work and continues applying rules;
it never waits for a planner future inside a physical tick. Expired, canceled, and
post-completion results are discarded. The deterministic forecast selector is in
`forecast_selector.py` and does not call an LLM.

The registered experiment runs at 30 physical frames per wall-clock second with
measured pipeline latency, not a fixed 23-frame delay. Main-loop checks use current
alarms, mode, target eligibility, rule target and drive-point geometry; expensive
revalidation uses a snapshot at most 30 frames old. Per-tick time and scheduling
lateness are saved. This is a measured soft-realtime simulation, not a hard-realtime
or physical-robot guarantee.

The older unpaced batch and synchronous forecast paths remain available for
historical decision-quality studies. Their completion scores do not establish
real-time performance. See `EXPERIMENT_NOTES.md` for the frozen 200-scene protocol,
startup/recovery checks and results. Existing experiment folders are never
overwritten.

## License

See `LICENSE`.
