# Dual-Layer Autonomous Swarm Shepherding

A two-dimensional simulation of one shepherd dog guiding a heterogeneous flock
into a circular goal. A rule-based controller drives the dog every frame. An
optional high-level layer (an LLM or a fixed heuristic) is consulted only when
the flock gets into trouble, and its suggestions are checked by an
event-triggered model-predictive step before they are allowed to take control.

The project asks a simple question: **when some sheep behave abnormally, can a
high-level reasoning layer help the shepherd keep the task on track, beyond what
fixed rules already achieve?**

## Research goal

The long-term aim is a shepherding system that keeps its control objective
(getting the flock into the goal) when the flock or the environment does not
behave as designed. This follows a monitoring loop:

```text
monitor state -> detect anomaly -> diagnose cause -> plan repair -> execute -> verify
```

So far the project has built and evaluated this loop for **known** anomaly
types. The next stage moves to anomalies the rules were not designed for (see
[Next steps](#next-steps)).

## Simulation

- 2D square world (200 x 200), circular goal (radius 20), one dog, 20 sheep.
- Sheep follow boids-style cohesion, separation, boundary avoidance, and dog
  repulsion. The dog is about twice as fast as a sheep.
- An episode succeeds when all sheep are inside the goal; otherwise it is capped
  at 2000 frames.

Three abnormal sheep types can be mixed into the flock:

| Type | Behaviour | Main parameters |
| --- | --- | --- |
| A, weak response | Barely reacts to the dog | `WEIGHT_DOG_REPULSION_A_FACTOR` |
| B, dispersing | Weak cohesion, stronger and longer-range separation | `WEIGHT_COHESION_B_FACTOR`, `WEIGHT_SEPARATION_B_FACTOR`, `SEPARATION_RADIUS_B` |
| C, near-dog sensitive | Over-reacts when the dog is close; same speed limit | `WEIGHT_DOG_REPULSION_C_FACTOR`, `TYPE_C_SENSITIVITY_RADIUS` |

Most recent experiments use 5 A + 5 B + 5 C + 5 normal sheep.

## Control architecture

```text
            sheep / dog state
                   |
                   v
               Detector  ---- splitting + stagnation alarms
                   |
        +----------+------------------------------+
        |                                         |
        v                                         v
 Low-level rule controller            High-level layer (async, optional)
 runs every frame, never waits        LLM or heuristic proposes mode + target
        |                                         |
        |                                         v
        |                              Event-triggered MPC
        |                              rule branch vs. proposal branch
        |                                         |
        +<---------- adopt only if still valid ---+
        |
        v
   dog command -> kinematics -> next state
```

### Rule layer

The detector raises two alarms:

- **Splitting:** max distance to the flock centroid > 40 and mean distance > 15.
- **Stagnation:** centroid progress toward the goal < 1.0 over 150 frames.

The rule controller has three repair modes:

| Mode | Purpose | Rule trigger |
| --- | --- | --- |
| `TARGET_TRACKING` | Normal goal-directed herding | No alarm |
| `HARD_TO_GUIDE` | Push one hard-to-move sheep | Stagnation (also when both alarms fire) |
| `COHESION_PRIORITY` | Collect a separated sheep | Splitting |

Repair targets are chosen with a short velocity look-ahead and held for a
minimum time. A light goal-retention force keeps sheep near the goal edge from
drifting out.

### High-level layer

When both alarms are active, the high-level layer may suggest a different legal
mode and target sheep. It cannot produce motion forces or control the future
trajectory directly. The simulation never pauses while a request is pending;
the rule controller keeps driving the dog.

Two proposal sources are implemented:

- **LLM:** a local Qwen3.6-35B-A3B (NVFP4) model served by vLLM, or
  OpenAI/DeepSeek-compatible APIs. It receives a structured state summary,
  including the true sheep types.
- **Fixed heuristic:** keeps the rule-selected mode and picks the eligible sheep
  farthest from the goal (lowest index on ties). It reads no type labels.

### Event-triggered MPC

Each proposal is checked before use:

1. When the proposal arrives, copy the latest world state into two background
   simulations.
2. **Rule branch:** keep using the rules. **Proposal branch:** apply the
   suggestion for at most 30 frames (early release allowed), then hand back to
   the rules.
3. Run both branches until success or the 2000-frame budget
   ("completion-first" scoring).
4. Prefer the branch that succeeds, then the one that finishes earlier. Ties or
   double failures keep the rules.
5. Back in the live world, adopt the proposal only if the alarms, target
   eligibility, geometry, and snapshot age (at most 5 frames) are still valid.
   The live world is never rewound.

Background copies never call the LLM again, and the whole round is discarded
afterwards. The live simulation runs at a real 10 Hz clock while predictions are
computed in the background.

## Project layout

```text
shepherding_sim/
|-- main.py                       interactive single run
|-- run_experiments.py            batch runs (not real-time)
|-- config.py                     simulation and controller parameters
|-- environment.py, dynamics.py   world state and kinematics
|-- detector.py                   splitting and stagnation alarms
|-- control_types.py              controller input/output contracts
|-- execution_contract.py         admissible actions and execution rules
|-- low_level_controller.py       frame-by-frame rule controller
|-- high_level_controller.py      async high-level policy and validation
|-- realtime_*.py                 continuous clock and background workers
|-- mode_forecast.py              short-horizon forecasts
|-- forecast_selector.py          early fixed forecast-based selector
|-- target_handoff.py             historical target hand-off semantics
|-- mpc_controller.py             original MPC (candidates, cancellation, delivery)
|-- mpc_completion_controller.py  completion-first MPC
|-- mpc_component_ablation.py     proposal source x selection ablation
|-- vllm_*.py, responses_api_backend.py   LLM backends and input formats
|-- visualizer.py, plot_analysis.py       live view, videos, metric plots
|-- experiments/                  experiment entry points (see below)
|-- tools/                        diagnostics, comparisons, GPU-server helpers
|-- tests/                        unit and integration tests
|-- docs/                         technical reference and early notes (Chinese)
`-- EXPERIMENT_NOTES.md           full research log (Chinese)
```

Generated data (`logs/`) and local reports are not tracked by Git.

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick start

Run one simulation with the rule controller:

```powershell
python main.py --controller rule --scenario stress --seed 70 --no-video
```

Choose any population directly:

```powershell
python main.py --controller rule --abnormal-a 5 --abnormal-b 5 --abnormal-c 5 --seed 70 --max-steps 2000
```

Controllers: `rule`, `mock-llm`, `openai`, `deepseek`, `vllm`. Scenario presets
(`normal`, `type-a`, `type-b`, `type-c`, `mixed`, `stress`) only set sheep counts.
Use `--no-repair` for a no-repair baseline. VS Code users can also pick the
architecture and scenario from **Run and Debug**.

Batch runs (fast, not on a real-time clock):

```powershell
python run_experiments.py --help
python run_experiments.py --controller rule --scenario stress --repair --runs 10 --seed-mode random --batch-seed 70
```

### LLM backends

API keys are read from environment variables or a local `.env` file, which is
ignored by Git. No key is needed for the rule controller.

```powershell
$env:DEEPSEEK_API_KEY = "your-key"            # DeepSeek
$env:OPENAI_API_KEY = "your-key"              # OpenAI
$env:VLLM_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
$env:VLLM_BASE_URL = "http://127.0.0.1:8002/v1"   # any OpenAI-compatible vLLM server
```

The vLLM defaults in `config.py` are historical; the formal experiments set the
model and address explicitly.

### Formal experiments

The scripts in `experiments/` reproduce completed studies. Those that use the
LLM run on a GPU server with a local vLLM model. The scripts start only after
an explicit `--run` and refuse to overwrite a finished batch. Use `--help` to see their options.

| Script | Study |
| --- | --- |
| `mpc_pilot.py`, `mpc_candidate_comparison.py` | Early 6-scene MPC mechanism checks |
| `mpc_validation.py`, `mpc_horizon_review.py` | Original MPC, 100 new scenes, and review |
| `mpc_completion_validation.py` | Completion-first MPC, 200 exploratory scenes |
| `mpc_confirmation.py` | Independent 400-scene confirmation |
| `mpc_component_validation.py` | Five-arm 100-scene component split |
| `oracle_permissions.py` | Type-A-only difficulty calibration |
| `rule_timing.py`, `qwen_parallel_timing.py` | 10 Hz timing and concurrency checks |
| `anomaly_behavior/` | Early anomaly characterisation and repair screens |

Each finished batch keeps its frozen source, protocol, and evidence under
`logs/experiments/`; reproduce a batch with its frozen source, not the current
tree.

## Research timeline

| Stage | What was done | Takeaway |
| --- | --- | --- |
| Simulation and anomalies | World, Type A/B/C behaviour, alarms, three repair modes, goal retention | Rules with repair already handle many anomaly cases |
| LLM integration | DeepSeek/OpenAI APIs, then a local Qwen3.6 model with structured output and permission checks | Local model used for all later experiments |
| Hand-off and inputs | Target hand-off, early release, observation windows, short-term forecast ranking | Small positive examples did not generalise |
| Simple baselines | Fixed forecast-based selector compared with rules and the LLM | A simple program matched or beat the LLM in several batches |
| Real-time clock | Predictions and LLM calls run in the background; the world keeps moving | Later experiments run at a real 10 Hz |
| MPC | Rule vs. LLM candidates in background simulations, then completion-first scoring | The MPC combination improved success |
| Component split | Heuristic vs. LLM proposals, with and without MPC | No measurable added value from the LLM |

## Key results

Batches are compared only within themselves. Failures count as 2000 frames in
the mean.

**Completion-first MPC, 200 exploratory scenes (seeds 50001-50200)**

| Method | Success |
| --- | ---: |
| Rules only | 186/200 |
| Original MPC + LLM | 185/200 |
| Completion-first MPC + LLM | 193/200 |

11 scenes rescued and 4 lost vs. rules (exact McNemar p ~ 0.12, exploratory).

**Independent confirmation, 400 scenes (seeds 60001-60400)**

| Method | Success | Mean frames |
| --- | ---: | ---: |
| Rules only | 362/400 (90.5%) | 1084.71 |
| Completion-first MPC + LLM | 378/400 (94.5%) | 1006.44 |

24 rescued and 8 lost; exact two-sided McNemar p = 0.0070. The **combination**
beats the rules, but this does not show where the gain comes from.

**Component split, 100 scenes (seeds 70001-70100)**

| Arm | Success | Mean frames |
| --- | ---: | ---: |
| Rules only | 92/100 | 1138.40 |
| Heuristic, direct | 95/100 | 1128.71 |
| LLM, direct | 90/100 | 1107.31 |
| Heuristic + MPC | **99/100** | 1050.53 |
| LLM + MPC | 97/100 | 1025.27 |

All four pre-registered comparisons were non-significant after Holm correction
(e.g. LLM + MPC vs. heuristic + MPC: 1 rescued, 3 lost, p = 0.625). All four
proposal arms made the same LLM requests and full-tail predictions, so their
compute and waiting time are matched.

## Current problem

**In the current setting, the LLM does not add measurable value.** A fixed
heuristic plus MPC verification does at least as well.

The likely reason is structural, not a matter of prompt tuning:

- the anomaly types are known and given to the controller as labels;
- there are only three repair modes and a few dozen mode-target choices;
- the MPC uses the exact simulator, so it can test every cheap candidate.

When the option set is small and the model is perfect, "try each candidate and
keep the best" is close to optimal, and an LLM can at most match it.

A post-hoc review found two more details:

- The LLM did make a few useful choices the heuristic never proposes, mostly a
  switch to `COHESION_PRIORITY` (e.g. seed 70073 finished 268 frames earlier).
  These are local speed-ups, not an overall success advantage.
- In 19 of 95 actual take-overs, re-scoring from the real adoption state
  (1-4 frames later) reversed the ranking. The live check confirms age and
  validity, not that the predicted advantage still holds. This timing gap is
  not yet fixed.

## Next steps

Planned direction, subject to discussion with the supervisor:

1. **Remove the shortcuts.** Add anomalies the rules were not designed for
   (for example, a sheep attracted to the dog, a very slow sheep, a constant
   drift), hide the type labels, and let the MPC use only the nominal model.
2. **Check that there is room first, without any LLM.** Compare rules,
   heuristic + MPC with the nominal model, and heuristic + MPC with the true
   model. If the true-model upper bound is not clearly better, diagnosis cannot
   help and the setting must change.
3. **Then add LLM diagnosis.** The LLM reads the observed motion, infers the
   likely cause, and selects a repair (possibly from new modes such as leading
   from the front or offset positioning), with the MPC still verifying.
4. **Fair baselines.** Compare against heuristic + MPC with the same
   information, and against a programmatic system-identification baseline.
   Anomalies used to tune prompts are kept separate from test anomalies.

## Limitations

- 2D kinematic simulation, one dog, not a physical robot.
- Abnormal behaviour is hand-designed; current results cover known types only.
- LLM responses are nondeterministic and depend on server latency.
- Several earlier batches are small; only the 400-scene study is confirmatory.

## Tests

```powershell
python -B -m unittest discover -s tests
```

The tests do not call any real LLM service.

## Further documentation

- [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md): full research log (Chinese).
- [docs/technical_reference.md](docs/technical_reference.md): modes, interfaces,
  parameters (Chinese).
- [docs/early_experiments.md](docs/early_experiments.md) and
  [experiments/anomaly_behavior/README.md](experiments/anomaly_behavior/README.md):
  early experiments (Chinese).
- [SPARK_SETUP.md](SPARK_SETUP.md): GPU-server model setup (contains historical
  defaults).

## License

See [LICENSE](LICENSE).
