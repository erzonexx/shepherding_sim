# Dual-Layer Autonomous Swarm Shepherding

A two-dimensional simulation of one shepherd agent guiding a heterogeneous flock
to a circular goal. The project combines a continuous kinematic controller with
an optional event-triggered LLM layer for anomaly recovery.

The implementation focuses on two abnormal sheep types:

- **Type A (weak response):** weak dog repulsion and high inertia.
- **Type B (dispersing):** weak cohesion and stronger, longer-range separation.

The low-level controller remains active every frame. The high-level layer is
consulted only when stagnation and flock splitting are detected together, so API
latency never pauses the simulation.

## Architecture

```text
sheep/dog state
      |
      v
   Detector ------------------------------+
      | alarms                            |
      v                                   |
HighLevelController (optional, async)     |
      | bounded repair mode               |
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
the LLM may choose a repair mode from the same bounded set.

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

Available controllers are `rule`, `mock-llm`, `openai`, and `deepseek`.

### VSCode: Select Conditions Without Editing Code

Open this folder in VSCode and select your virtual environment using
`Python: Select Interpreter`. In **Run and Debug** (`Ctrl+Shift+D`), choose
**Shepherding - Select architecture and scenario**, then press the green play
button (or `F5`). The prompts let you choose independently:

1. Architecture: rule-based fast control, or fast control plus DeepSeek.
2. Population: Normal (0A+0B), Type A (4A+0B), Type B (0A+4B), Mixed (2A+2B),
   or Stress (3A+3B).
3. Seed, maximum frames, and whether to export MP4.

Use **Shepherding - Custom sheep counts** to enter arbitrary A/B counts.
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
The window title shows the controller, A/B counts, and seed. Effective settings
are also saved as `logs/data/run_<timestamp>.json` beside the simulation data.

The same conditions can be supplied from a terminal:

```powershell
python main.py --controller rule --scenario mixed --seed 70 --no-video
python main.py --controller deepseek --scenario mixed --seed 70 --no-video
python main.py --abnormal-a 1 --abnormal-b 6 --seed 70 --max-steps 2000
```

Use `--no-repair` for a no-repair baseline, or `--random-seed` for random initial
conditions. Scene presets change only sheep counts; they do not select an
architecture or toggle repair. The C00-C08 batch cases below retain their
original population and repair settings.

## Experiments

For a batch launched from VSCode, select **Batch - Select architecture and
scenario** in **Run and Debug** and press play. Choose architecture, population,
repair on/off, number of runs, seed mode, seed-group ID, frame limit, and outputs.
For example, Rule + Stress + repair on + 10 runs runs ten 3A+3B simulations.
**Batch - Custom sheep counts** supports arbitrary A/B counts.

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
response must select one admissible mode and a bounded hold time. The simulation
continues under rule control while the request is pending.

Before execution, a response is rejected when it is malformed, stale, outside
the allowed mode set, or worse than the rule candidate in the short-horizon
forecast. A minimum dwell period and bounded hold duration reduce rapid mode
switching. API responses may be cached for reproducibility.

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

Stagnation is reported when centroid progress toward the goal is less than
`1.0` over a 150-frame window.

Repair target selection uses a short velocity lookahead. A target is held for a
minimum period and changed only when another eligible sheep is meaningfully more
urgent. Sheep already inside the goal are deprioritized when an outside target
exists. A light goal-retention force reduces outward drift near the goal edge.

## Main Parameters

| Group | Important settings |
| --- | --- |
| Simulation | `MAX_STEPS`, `RANDOM_SEED`, `SPACE_SIZE`, `GOAL_RADIUS` |
| Population | `NUM_SHEEP`, `NUM_ABNORMAL_A`, `NUM_ABNORMAL_B` |
| Speed | `SHEEP_MAX_SPEED`, `DOG_MAX_SPEED` |
| Flocking | `WEIGHT_COHESION`, `WEIGHT_SEPARATION`, `WEIGHT_BOUNDARY` |
| Shepherd response | `WEIGHT_DOG_REPULSION`, `DOG_SENSING_RANGE` |
| Type A | `WEIGHT_DOG_REPULSION_A_FACTOR`, `INERTIA_FACTOR_A` |
| Type B | `WEIGHT_COHESION_B_FACTOR`, `WEIGHT_SEPARATION_B_FACTOR` |
| Detection | `DANGER_DISPERSION_THRESHOLD`, `DANGER_MEAN_SPREAD_THRESHOLD`, `DANGER_STAGNATION_*` |
| Repair geometry | `DOG_DRIVE_DISTANCE`, `HARD_TO_GUIDE_DISTANCE`, `COHESION_REPAIR_DISTANCE` |
| Target stability | `*_TARGET_MIN_HOLD_FRAMES`, `REPAIR_TARGET_SWITCH_MARGIN`, `REPAIR_TARGET_LOOKAHEAD_FRAMES` |
| High level | `HIGH_LEVEL_*`, `LLM_API_TIMEOUT_SECONDS`, `LLM_CACHE_ENABLED` |

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

These results describe completed local experiments; they are not universal
performance claims.

- In 80 paired rule-controller runs, predictive target selection improved task
  success from 57 to 61 runs with no aggregate success loss in any scenario.
- In a 10-seed C08 comparison (seeds 141-150), the rule controller succeeded in
  6 runs and the DeepSeek controller in 8. Mean capped completion frames changed
  from 1142.9 to 951.0; danger frames from 423.2 to 335.6; maximum dispersion
  from 76.19 to 73.16.
- The LLM comparison used 33 requests, 30 responses, 13 accepted decisions, and
  17 rejected decisions, with mean response latency of about 1.20 seconds.
- The sample is small and paired significance was not below 0.05. Repeating one
  seed also showed provider nondeterminism.

## Limitations

- The model is a 2D kinematic simulation, not a physical robot controller.
- Only two anomaly types and one shepherd are currently modeled.
- LLM results depend on provider latency and nondeterministic responses.
- More seeds, ablations, and statistical analysis are required before making
  strong comparative claims.

## License

See `LICENSE`.
