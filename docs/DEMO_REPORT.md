# Minimum Demo Report

## Purpose

This demo validates a first-month minimum loop for a 2D shepherding simulation:

- Build a simple 2D shepherding environment.
- Generate normal and abnormal sheep scenarios.
- Detect danger states such as `stagnation` and `flock_splitting`.
- Switch between simple repair modes based on danger alarms.
- Compare runs with and without repair intervention.

The goal is not to produce a final optimized algorithm. The goal is to show a working research loop with visible success and limitation cases.

## Fixed Experiment Conditions

| Setting | Value |
| :--- | :--- |
| `USE_FIXED_SEED` | `True` |
| `RANDOM_SEED` | `70` |
| `NUM_SHEEP` | `20` |
| `MAX_STEPS` | `2000` |
| `SHEEP_MAX_SPEED` | `0.7` |
| `DOG_MAX_SPEED` | `1.5` |
| `GOAL_RADIUS` | `20.0` |

## Repair Modes

| Mode | Trigger | Control idea |
| :--- | :--- | :--- |
| `TARGET_TRACKING` | Default, or repair disabled | Collect the furthest sheep if needed; otherwise drive the flock toward the goal. |
| `HARD_TO_GUIDE` | `stagnation` | Move behind the sheep farthest from the goal and push it toward the target. |
| `COHESION_PRIORITY` | `flock_splitting` | Move outside the sheep farthest from the flock center to pull it back toward the group. |

## Result Summary

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

Machine-readable summary: `docs/experiment_summary.csv`.

## Main Improvement Case

The clearest improvement is C01 vs C02.

| Metric | C01: repair off | C02: repair on |
| :--- | ---: | ---: |
| Success | No | Yes |
| Frames | 2000 | 181 |
| Final `dist_to_goal` | 180.94 | 6.23 |
| Danger frames | 1234 | 12 |
| Max dispersion | 55.45 | 52.45 |

Interpretation:

- Without repair, the mixed abnormal scenario does not converge within 2000 frames.
- With alarm-driven repair, the flock reaches the goal in 181 frames.
- The controller mostly stays in `TARGET_TRACKING`, with a short `COHESION_PRIORITY` intervention.

## Limitation Cases

The current repair controller is not universally beneficial.

- C04 is slower than C03, which suggests the `HARD_TO_GUIDE` rule can over-intervene in A-only cases.
- C06 improves final distance compared with C05, but greatly increases maximum dispersion and still fails.
- C08 improves final distance compared with C07, but still does not reach the goal.

These cases are useful because they show where the next iteration should focus.

## Next Steps

- Add mode hysteresis or mode hold frames.
- Add a near-goal rule to avoid unnecessary repair.
- Smooth close-range dog-sheep repulsion.
- Run multi-seed evaluation after the minimum demo stabilizes.
- Compare rule-based repair with future LLM-assisted mode selection.
