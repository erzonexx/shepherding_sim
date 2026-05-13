# Swarm Shepherding Simulation

A 2D multi-agent shepherding simulation for the first-month minimum demo loop:

1. A normal flock can be guided to a target region.
2. Abnormal sheep can create difficult or failed control cases.
3. A simple rule-based danger detector can trigger repair modes.
4. Repair can improve at least one abnormal scenario while still leaving clear limitation cases.

The current project is intentionally lightweight. It is not intended to be a final optimized shepherding algorithm.

## Project Structure

```text
shepherding_sim/
|-- main.py                 # Single-run simulation entry point
|-- run_experiments.py      # Fixed C00-C08 demo experiment runner
|-- config.py               # Tunable parameters and output settings
|-- environment.py          # Core environment, physics, agents, control modes
|-- detector.py             # Flock metrics and danger detection rules
|-- visualizer.py           # Live rendering, trajectory PNG, animation MP4
|-- plot_analysis.py        # Metric plotting from parquet logs
|-- docs/
|   |-- DEMO_REPORT.md
|   |-- experiment_summary.csv
|-- logs/                   # Auto-generated local outputs
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

MP4 export requires `ffmpeg` to be installed and available on `PATH`.

Run one simulation using the current `config.py`:

```bash
python main.py
```

Run the fixed C00-C08 demo suite:

```bash
python run_experiments.py
```

Save trajectory PNGs for the demo suite:

```bash
python run_experiments.py --visuals
```

Save both trajectory PNGs and MP4 animations:

```bash
python run_experiments.py --visuals --video
```

The experiment runner temporarily changes `config` values in memory for each case. It does not rewrite `config.py`.

## Model Overview

The simulation combines two standard ideas:

- Boids-style flocking: sheep move according to cohesion, separation, boundary avoidance, and repulsion from the shepherd dog.
- Shepherding control inspired by Strombom-style driving and collecting: the dog either drives the flock toward the goal or moves outside the most distant sheep to collect it back into the flock.

This model was chosen because it gives a visible 2D system, clear state variables, simple control inputs, and interpretable failure cases.

## World

| Item | Implementation |
| :--- | :--- |
| Space | Square 2D plane, size `config.SPACE_SIZE` |
| Goal | Random position sampled with the fixed seed |
| Goal region | Circle with radius `config.GOAL_RADIUS` |
| Success condition | All sheep are within `GOAL_RADIUS` of the goal |
| Time horizon | `config.MAX_STEPS` frames |
| Seed | `config.RANDOM_SEED` when `config.USE_FIXED_SEED = True` |

## Agents

### Sheep

Each sheep has:

- 2D position: `sheep_pos[i]`
- 2D velocity: `sheep_vel[i]`
- status label: `Normal`, `A`, or `B`

Normal sheep are affected by:

- cohesion toward the flock center of mass
- repulsion from the dog when the dog is within sensing range
- separation from nearby sheep
- boundary repulsion near the edge of the world
- max-speed clipping

### Shepherd Dog

The current model uses one dog. The dog has:

- 2D position: `dog_pos`
- 2D velocity: `dog_vel`
- selected control mode: `current_mode`
- target drive point: `drive_point`

The dog moves toward the current drive point and is clipped by `config.DOG_MAX_SPEED`.

## Abnormal Sheep

### Type A: Unresponsive

Type A represents sheep that respond weakly to guidance and change direction slowly.

Implementation:

- dog repulsion is reduced by `config.WEIGHT_DOG_REPULSION_A_FACTOR`
- force response is reduced using `config.INERTIA_FACTOR_A`
- max speed remains the same as normal sheep

### Type B: Dispersing

Type B represents sheep that are more likely to separate from the flock.

Implementation:

- cohesion is reduced by `config.WEIGHT_COHESION_B_FACTOR`
- separation strength is increased by `config.WEIGHT_SEPARATION_B_FACTOR`
- separation sensing radius is increased to `config.SEPARATION_RADIUS_B`
- dog response and max speed remain the same as normal sheep

## Danger Detector

The detector computes three main metrics:

| Metric | Meaning |
| :--- | :--- |
| `dist_to_goal` | Distance from flock center of mass to the goal |
| `dispersion` | Maximum distance from flock center of mass to any sheep |
| `mean_spread` | Mean sheep distance from flock center of mass |

It reports two danger types:

| Danger type | Rule |
| :--- | :--- |
| `flock_splitting` | `dispersion > DANGER_DISPERSION_THRESHOLD` and `mean_spread > DANGER_MEAN_SPREAD_THRESHOLD` |
| `stagnation` | over `DANGER_STAGNATION_FRAMES`, progress toward the goal is less than `DANGER_STAGNATION_THRESHOLD` |

Detector output includes:

- `is_danger`
- `danger_type`
- short description text
- alarm list for the controller

## Repair Control Modes

The controller receives an alarm list from the detector:

```text
[]                         -> no danger
["stagnation"]             -> flock is not making enough progress
["flock_splitting"]        -> flock is too dispersed
```

The environment maps alarms to dog control modes:

| Alarm condition | Selected mode |
| :--- | :--- |
| `REPAIR_ENABLED = False` | `TARGET_TRACKING` |
| `flock_splitting` | `COHESION_PRIORITY` |
| `stagnation` | `HARD_TO_GUIDE` |
| no alarm | `TARGET_TRACKING` |

If multiple alarms are introduced in the future, `flock_splitting` currently has higher priority than `stagnation`.

### TARGET_TRACKING

Default mode. If the flock is too dispersed, the dog collects the sheep farthest from the center of mass. Otherwise, the dog moves behind the flock relative to the goal and drives the group forward.

### COHESION_PRIORITY

Repair mode for flock splitting. The dog moves outside the sheep farthest from the flock center to push it back toward the group.

### HARD_TO_GUIDE

Repair mode for stagnation. The dog targets the sheep farthest from the goal and moves behind it relative to the goal direction.

## Fixed Demo Cases

The fixed demo suite is C00-C08 with seed `70`.

| Case | Scenario | A | B | Repair |
| :--- | :--- | ---: | ---: | :---: |
| C00 | Normal baseline | 0 | 0 | Off |
| C01 | Mixed anomaly | 2 | 2 | Off |
| C02 | Mixed anomaly | 2 | 2 | On |
| C03 | A-only | 4 | 0 | Off |
| C04 | A-only | 4 | 0 | On |
| C05 | B-only | 0 | 4 | Off |
| C06 | B-only | 0 | 4 | On |
| C07 | Stress mixed | 3 | 3 | Off |
| C08 | Stress mixed | 3 | 3 | On |

The existing presentation results are stored in:

- `docs/DEMO_REPORT.md`
- `docs/experiment_summary.csv`

Representative improvement case:

| Case | A | B | Repair | Success | Frames | Final `dist_to_goal` | Danger Frames |
| :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: |
| C01 | 2 | 2 | Off | No | 2000 | 180.94 | 1234 |
| C02 | 2 | 2 | On | Yes | 181 | 6.23 | 12 |

Full fixed-seed result summary:

| Case | Scenario | A | B | Repair | Success | Frames | Final Dist | Min Dist | Max Dispersion | Danger Frames |
| :--- | :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| C00 | Normal baseline | 0 | 0 | Off | Yes | 145 | 8.84 | 8.84 | 52.45 | 11 |
| C01 | Mixed anomaly | 2 | 2 | Off | No | 2000 | 180.94 | 1.91 | 55.45 | 1234 |
| C02 | Mixed anomaly | 2 | 2 | On | Yes | 181 | 6.23 | 1.87 | 52.45 | 12 |
| C03 | A-only | 4 | 0 | Off | Yes | 184 | 6.60 | 0.32 | 52.45 | 13 |
| C04 | A-only | 4 | 0 | On | Yes | 462 | 4.51 | 0.45 | 52.45 | 155 |
| C05 | B-only | 0 | 4 | Off | No | 2000 | 141.16 | 11.21 | 54.31 | 1010 |
| C06 | B-only | 0 | 4 | On | No | 2000 | 113.32 | 0.87 | 98.91 | 1148 |
| C07 | Stress mixed | 3 | 3 | Off | No | 2000 | 132.08 | 0.20 | 57.58 | 1120 |
| C08 | Stress mixed | 3 | 3 | On | No | 2000 | 35.03 | 0.37 | 104.29 | 1016 |

Relevant videos:

- C01 no repair: `logs/videos/animation_20260512_175825.mp4`
- C02 repair on: `logs/videos/animation_20260512_180028.mp4`

Videos and parquet logs are generated locally and are not committed to the public repository.

## Output Data

Simulation outputs are written under `logs/`.

| Directory | Content |
| :--- | :--- |
| `logs/data` | Agent states and detector metrics as parquet files |
| `logs/control` | Repair-controller diagnostic logs |
| `logs/visuals` | Trajectory PNG files |
| `logs/videos` | MP4 animation files |
| `logs/analysis` | Metric trend charts |

### `logs/data` Schema

| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Frame` | int | Simulation frame |
| `Agent_Type` | str | `Dog` or `Sheep` |
| `Agent_ID` | int | Agent identifier |
| `X`, `Y` | float | Agent position |
| `Status` | str | `Dog`, `Normal`, `A`, or `B` |
| `dist_to_goal` | float | Distance from flock center of mass to goal |
| `dispersion` | float | Maximum distance from center of mass to any sheep |
| `mean_spread` | float | Average distance from center of mass to sheep |
| `is_danger` | int | 1 if danger is detected, otherwise 0 |

### `logs/control` Schema

| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Frame` | int | Simulation frame |
| `repair_enabled` | int | 1 if repair control is enabled |
| `alarms` | str | Detected alarms, or `none` |
| `current_mode` | str | Selected dog control mode |
| `drive_point_x`, `drive_point_y` | float | Dog target point |
| `dog_x_before`, `dog_y_before` | float | Dog position before update |
| `dog_x_after`, `dog_y_after` | float | Dog position after update |

## Reading Logs

```python
import pandas as pd

df = pd.read_parquet("logs/data/data_20260512_180028.parquet")

metrics = (
    df[["Frame", "dist_to_goal", "dispersion", "mean_spread", "is_danger"]]
    .drop_duplicates()
    .sort_values("Frame")
)

print(metrics.tail())
```

## Current Limitations

- The controller switches modes immediately; there is no hysteresis or mode hold time.
- Repair distances are fixed constants in the environment logic.
- The model uses one dog even though `config.NUM_DOGS` exists as a parameter.
- The first demo is based on a fixed seed. Multi-seed evaluation should be added later for statistical validation.
- The close-range dog-sheep repulsion model is simplified.

## Attribution

This project was developed as a research demo related to the CSS Laboratory, Hiroshima University.

- CSS Lab: https://csslab.jp/

## License

This project is released under the MIT License. See `LICENSE` for details.

## Suggested Meeting Demo

For a short meeting, show:

1. `README.md`: model, abnormal sheep, detector, and control modes.
2. C00 video or trajectory: normal baseline succeeds.
3. C01 vs C02 videos: no-repair failure versus repair success.
4. C06 or C08: limitation case where repair helps partially or fails.
5. `run_experiments.py`: the C00-C08 suite can be reproduced with one command.
