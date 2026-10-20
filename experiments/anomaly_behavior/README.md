# Anomaly behavior characterization

This folder contains focused checks for the sheep anomaly model. The one-frame
and compact-flock trials call the production dynamics directly. The paired
full-simulation trials use the rule controller unless the explicit
`--population-api`, `--repair-feedback` or `--blind-diagnosis` diagnostic is selected.

Run the full characterization:

```powershell
python experiments\anomaly_behavior\run_characterization.py
```

Outputs are kept together under `logs/anomaly_behavior/<timestamp>/`:

- `report.json`: summary metrics, current parameters, and review flags.
- `type_a_dog_response.csv`: one-frame response at several dog distances.
- `type_b_dispersion.csv`: compact-flock dispersion over time.
- `normal_initial_split.csv`: frame-zero splitting result for each normal seed.

The review flags identify conditions that need discussion. They are diagnostic
thresholds, not claims that the current behavior is scientifically correct.

## Fixed-Environment Population Scale Check (2026-10-04)

### Return To The Actual LLM Control Interface

The ordinary-motion diagnosis trials are NOT the current production LLM input.
`HighLevelController` already supplies true type counts, type geometry and
candidate target statuses, and mode forecasts simulate the true sheep statuses.
Thus blind identity-diagnosis failure cannot explain the previous deployed LLM
repair regressions. Those trials only evaluate a possible observation-only
extension; it is not adopted. Stop treating type identification as a required
next step for the existing mode-selection controller.

Make a narrow interface correction: prompt v11 and the structured response mode
enum now follow the request's actual alarms. No alarm allows TARGET_TRACKING
only; splitting adds COHESION_PRIORITY; stagnation adds HARD_TO_GUIDE; both add
both repairs. Target-only requests must still copy the admissible rule mode.
Forecasts for an unavailable mode remain contextual information, not permission.
Locally reject a provider returning a mode outside that enum before caching it.
Cache identity includes the authoritative request alarms and new prompt version.
Existing arrival-time alarm/staleness/geometry/forecast checks are unchanged,
so a mode legal when requested can still be rejected when returned. No new mode,
force, trigger, controller layer or motion permission is added.

Offline audit of ALL20 feedback-LLM event files from `20261004_165251_420836`:
106 submitted requests,81 recorded returns,25 cancellations. Among returns:

| Historical choice | Count |
| --- | ---: |
| Legal at request time | 62 |
| Repair alarm already absent at request time | 17 |
| Missing/invalid mode | 2 |

Historical execution accepted50 and rejected31:17 absent-alarm choices,2 invalid
modes,6 out-of-bounds drive points,6 forecast-dominated choices. The17 absent-alarm
choices are NOT latency-induced staleness. New schema membership excludes them,
but this audit does not generate alternative answers, rerun physics, or establish
better success/latency. A compliant provider should stay within the enum; the
local check catches violations but cannot avoid a billed call already made.

No API calls in this step. Source event hashes remain unchanged. Report:
`logs/anomaly_behavior/20261004_180853_967348/request_mode_admissibility_audit.json`.
Tests cover all alarm combinations, target-only legality, a noncompliant fake
provider, cache separation and asynchronous integration. No commit or push.
Full suite:207 tests pass.

### V11 Real-API Paired Follow-Up (Seeds5101-5120)

```powershell
python experiments/anomaly_behavior/run_characterization.py --repair-feedback --first-seed 5101 --seed-count 20
```

Completed60 runs on20 new seeds with the existing feedback runner: original
rule, simple feedback rule, real uncached DeepSeek feedback. Each matched group
uses50 sheep (35Normal+5A+5B+5C), space200x200, start standard deviation15,
goal radius20, cap2000, unchanged dynamics/alarms/repair distances3/10 and
low-level target selection. Feedback review/hold remain60 frames, maximum6
requests per run, simulated rule delay20 frames and actual API delay. No tuning
during the batch. The LLM uses v11 request-admissible mode restrictions.

| Arm | Success | Mean capped frames | Mean inside /50 | Mean goal exits |
| --- | ---: | ---: | ---: | ---: |
| Original rule | 6/20 | 1761.75 | 37.90 | 118.05 |
| Simple feedback rule | 6/20 | 1800.50 | 37.05 | 130.35 |
| DeepSeek feedback | 5/20 | 1792.85 | 37.35 | 116.85 |

Simple feedback rescues5112 but regresses5104; net success is unchanged.
DeepSeek rescues no original-rule failures and regresses5114 (1378 -> capped2000,
50 ->18 sheep inside). Other19 capped completion counts match the original rule,
giving mean paired delta+31.10 frames. Its7 runs/255 frames of actual mode override
produce both partial-progress gains and losses; they do not increase full-task
success. Do not treat a few more sheep inside on selected failures as success.

DeepSeek submits94 requests:69 recorded returns,44 accepted,25 rejected,25
canceled.67 returns choose a mode legal at request time;2 return a mode outside
the requested enum and are locally rejected as backend errors. Therefore the
provider's adherence to the schema is NOT guaranteed. Other rejections:21
forecast-dominated choices and2 out-of-bounds drive points. None of the recorded
valid-mode returns had its required alarm absent at request time. Mean recorded
response time0.975s, mean accepted delay19.5 frames. Canceled submissions may
already have reached/billed the provider;69 returns do not mean only69 API calls.

Simple feedback submits98:65 returns,45 accepted,20 rejected,33 canceled;
8 runs/126 override frames, no backend errors. Recorded accepted delay is20 frames.
This remains a whole-policy comparison, not a latency-isolated LLM reasoning test.
Different seeds and prompts from the v10 cohort prevent treating aggregate
before/after rejection rates as an isolated causal effect of schema restriction.

Audit verifies all20 full-precision initial geometry/type fingerprints match,
source hashes remain unchanged, cache is disabled, request budgets are respected,
and each request has exactly one recorded return or cancellation. Descriptive
paired-seed bootstrap intervals are in the report, not confirmatory significance
claims or repeated API-output validation. All207 tests pass after execution.

Conclusion: legal-output filtering is a useful interface constraint, but this
trial does NOT support improved LLM repair performance or adopting feedback
control as the default. Keep the unchanged rule baseline and the negative
results; do not add further modes or tune a special-case fix for5114 here.
No commit or push.

Artifacts: `logs/anomaly_behavior/20261004_181524_900574/`, protocol,
`repair_feedback_runs.csv`, `repair_feedback_pairs.csv`, sampled states,
per-arm/per-seed event JSON and `repair_feedback_report.json` including audit.

### Controlled Behavior Observability Check

```powershell
python experiments/anomaly_behavior/run_characterization.py --controlled-diagnosis
```

After the weak passive blind-diagnosis results, test whether exposure conditions
can reveal the implemented traits at all. No API, classification policy, controller
change or parameter tuning. Protocol is saved before running the dynamics:

- Reset one isolated sheep at [100,100] with zero velocity, goal [180,180],
  stationary dog on the negative x axis at distances3/10/20/25/40/60.
  Four private statuses give24 one-update controlled configurations.
- Reset the existing nine-sheep grid with spacing4, eight Normal peers and
  focus ID5 as Normal/A/B/C. Dog is outside sensing range. Observe60 updates
  per trajectory, sampling every frame. Goal retention remains configured as
  in production but is inactive because the goal is far away.
- Recompute observed displacement and flock metrics from positions rounded
  to0.01, matching log precision. No true velocity is needed for these observed
  metrics. Full-precision measurements are retained separately.

Completed results at `logs/anomaly_behavior/20261004_174726_212729/`:

| Dog distance | Normal | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.70 | 0.15 | 0.70 | 0.70 |
| 25 | 0.40 | 0.06 | 0.40 | 0.53 |
| 40 | 0.25 | 0.04 | 0.25 | 0.25 |
| 60 | 0.00 | 0.00 | 0.00 | 0.00 |

Table entries are displacement per update reconstructed from rounded positions,
not measured repair performance. At distance10, Normal and C are both capped
near speed0.7. At25, neither is capped and their responses differ. At40, C's
extra proximity response is inactive. B's isolated dog response equals Normal.

Without the dog, the mean focus-to-centroid distance over frames1-60 is11.57
for B versus4.06 for Normal/A/C using rounded observations (full precision:
11.56675 versus4.05454). Final distances are11.54916 versus2.94608.
Normal/A/C trajectories are exactly equal in this no-dog configuration.

Conclusion: the simulated traits are distinguishable under complementary IDEAL
exposures; ordinary motion alone does not always expose them. This is not a
classification accuracy experiment or an online active-probing implementation.
An isolated sheep, known Normal peers and zero initial velocity are not guaranteed
in a real mixed flock. It neither explains all earlier LLM regressions nor proves
an LLM advantage. A next feasibility check would have to obtain useful response
evidence in ordinary mixed-flock control without resets or true type labels,
then compare simple numerical identification before spending more API calls.

Artifacts: `controlled_diagnosis_protocol.json`, `controlled_dog_response.csv`,
`controlled_no_dog_flock.csv`, `controlled_diagnosis_report.json`, all in one
existing diagnostic output directory. 199 full-suite tests pass. Production
configuration/controllers and default start standard deviation15 are unchanged.
No commit or push.

### Ordinary Mixed-Flock Observation Coverage

```powershell
python experiments/anomaly_behavior/run_characterization.py --mixed-observability
```

Reuse ALL20 rule-arm logs, seeds4701-4720, from
`20261004_015855_871537` (50 sheep, 35Normal+5A+5B+5C). No new simulations,
API calls, parameter changes or online controller changes. Inspect every sheep,
using fixed chronological nonoverlapping61-frame windows starting at150.
Incomplete final windows are omitted, so successful shorter runs provide fewer
windows. Reuse the previous numerical diagnosis thresholds unchanged; take the
FIRST definite answer per sheep, or abstain if none. Do not select later correct
answers. Later conflicting definite answers are counted separately.

Feature extraction explicitly whitelists Frame/Agent_Type/Agent_ID/X/Y. Type
labels are joined only after observation windows are computed, for grading.
Vectorized extraction is checked against the original single-clip extractor,
including varied close exposures. Goal geometry is checked against source logs,
and source file hashes are verified unchanged. Moderate-distance opportunity
means at least6 samples at distance20-30 below97% of the observed peer95th-percentile
speed in any window; it is descriptive, not a new classification threshold.

Completed report: `logs/anomaly_behavior/20261004_175913_992934/`.
27,750 sheep-windows from1,000 individuals in20 seed clusters:

| Private type | Individuals | Definite answers | First answer matches type |
| --- | ---: | ---: | ---: |
| Normal | 700 | 401 | 1 |
| A | 100 | 57 | 20 |
| B | 100 | 96 | 93 |
| C | 100 | 53 | 0 |
| Total | 1,000 | 607 | 114 |

393 individuals remain unclassified. Conditional identity agreement is114/607
(18.78%); unconditional identification is114/1000 (11.4%). These are not repair
metrics. An always-Normal identity guess would hit700/1000 solely because Normal
prevalence is70%; do not interpret the B-stratum93/100 as a useful standalone
detector. Of529 first answers labelled dispersing, only93 are actually B (17.58%
precision), including358 false B-identity assignments to Normal sheep. Weak-response
answers match A in20/69; enhanced-response answers match C in0/8. A descriptive
outward-motion state is not necessarily an incorrect description, but is NOT a
valid latent-type assignment.70 individuals later receive a conflicting answer.

527 individuals have a moderate unsaturated opportunity, but only197 ever supply
at least6 matched peer-response samples in one window. For C these counts are44
and12 respectively. Thus signal availability is limited AND the existing proxy
rules are not reliable identity classifiers. This does not prove identification
is impossible, nor establish that insufficient observation caused all earlier
LLM repair regressions. No classifier is adopted and no API expansion is justified
by these results. Sheep/windows are correlated within20 seeds, not1,000 independent
trials. All203 tests pass; no commit or push.

Artifacts stay together: `mixed_observability_protocol.json`, geometry-only
`mixed_observation_windows.csv`, separate `mixed_observability_private_scores.csv`,
and `mixed_observability_report.json` (coverage, confusion and prediction precision).

### Offline Blind Behavior Diagnosis

```powershell
python experiments/anomaly_behavior/run_characterization.py --blind-diagnosis
```

This small diagnostic reuses the existing rule-arm agent Parquet logs from
`20261004_015855_871537`, seeds 4701-4705 (50 sheep, 5A+5B+5C). It does not run
new physics or change a controller. With selection seed 9101, choose one random
individual per PRIVATE Normal/A/B/C stratum per seed (20 clips). For each, choose
the closest recorded dog approach between frame150 and the last frame minus30;
center a 61-frame window there. This selection uses geometry, not diagnosis
outputs. Cases are shuffled before anonymous clip IDs are assigned. Strata are
used only for private selection/grading, not sent to the provider.

Whitelist input: observed focus/dog/flock geometry, verified goal location,
estimated motion, nearest-neighbor distance and measured features. No true
identity, counts by type, force coefficients, controller modes/targets,
predictions, seeds, source paths or answer files are sent. Goal geometry is
reconstructed from the source initialization and verified against logged COM
goal distances. Source X/Y are rounded to0.01; velocity and acceleration use
successive finite differences, not unrecorded true velocities. Dog exposure
uses previous sheep positions and current dog positions, matching the logged
physics update order. Display motion every5 frames; summary features use all61.

Both numeric rules and LLM see the same payload. Comparable peers are unlabeled
and may themselves be heterogeneous: distance within5, COM spread within10,
at least3 peers. Close exposure is a diagnostic threshold25, NOT a supplied
true sheep sensitivity parameter. Empirical speed ceiling is the observed
peer95th-percentile speed, not a true force/model parameter. Numeric thresholds
are fixed before LLM outcomes and explicitly allow abstention. LLM is asked
for normal/weak-dog-response/dispersing/enhanced-near-dog-response/insufficient
evidence; supporting frames must be among displayed samples. Invalid output
is reported separately, never silently treated as correct abstention.

Private simulated identity is an exact grading reference, not proof the
identity is observable in a clip or that its behavior has expressed itself.
Report unconditional identity identification, coverage, selective accuracy
and abstentions separately. Five source seeds form only five clusters; 20
clips are NOT 20 independent simulation trials. Balanced selection does not
reflect deployment prevalence. This cannot demonstrate improved repair.

Calls use the existing DeepSeek model, uncached, 400 output tokens, custom
prompt `shepherd-blind-behavior-v1`, no control output or movement permission.
Protocol is saved before inference, private truth and exact outgoing payloads
are separate, and raw responses/results remain together in one timestamped
`logs/anomaly_behavior/` folder. Tests check label/prediction invariance,
missing observations, threshold branches, outgoing whitelist and abstention
scoring with a fake client (unit tests never call the API).

Completed 20 real DeepSeek requests (mean response time1.070s), with source and
diagnostic/backend hashes verified unchanged during execution. Outgoing payload
whitelist and balanced private strata are checked. Results:

| Method | Conforming definite answers | Latent identities hit | Abstentions | Invalid outputs |
| --- | ---: | ---: | ---: | ---: |
| Fixed numerical rules | 1/20 | 1/20 | 19/20 | 0 |
| DeepSeek, strict protocol | 17/20 | 4/20 | 2/20 | 1 |

Among conforming definite LLM answers,4/17 (23.53%) match the private identity.
The single invalid output is a243-character reason against the240-character
limit, not a wrong category: its Normal category matches private truth. A
clearly marked SECONDARY category-only sensitivity check therefore gives5/20
hits and5/18 (27.78%) among category answers; it does not replace strict primary
scoring. Category-only hits:4/5 Normal,0/5 A,1/5 B,0/5 C. Correctly describing
ordinary motion is not necessarily identifying a latent behavior trait.

Only1/20 clips supplies at least6 matched close-response samples, and14/20
spend at least80% of the window near the empirical peer speed ceiling. Saturation
can mask force amplitude differences and matched observational evidence is
scarce; this is not a proof of mathematical nonidentifiability of every clip.
The numerical rule's single hit is NOT evidence of broad reliable diagnosis;
the LLM's greater willingness to answer is NOT evidence of reliable identification.
This small clustered observation-only test does not support enabling an LLM
blind-diagnosis repair policy. No new physics, controller changes or API
follow-up experiment is made;196 tests pass. No commit or push.

Artifacts: `logs/anomaly_behavior/20261004_172734_177496/`, with
`blind_diagnosis_protocol.json`, separate `blind_diagnosis_truth_private.json`
and exact `blind_diagnosis_payloads.json`, `blind_diagnosis_runs.csv`, raw
`response_clip_*.json`, and `blind_diagnosis_report.json` (includes audit and
format-only sensitivity analysis). Truth/source identity files are never sent.

### Actual Repair-Outcome Feedback Trial

```powershell
python experiments/anomaly_behavior/run_characterization.py --repair-feedback
```

Predeclared exploratory comparison: seeds 5001-5020, 50 sheep, 5A+5B+5C,
one dog, space 200x200, start standard deviation 15, goal radius 20, 2,000
step calls, unchanged sheep behavior, alarms, target selection and 3/10 repair
distances. Three arms: original rule, simple repair-feedback rule, real uncached
DeepSeek repair-feedback selection. Initial geometry AND identities match.

Feedback records actual applied-mode windows, not forecast-only alternatives:
mode, elapsed frames, targeted IDs (possibly several), centroid progress,
spread reduction, inside-goal change and goal exits. Different modes never
share a window; short interrupted windows are retained but cannot trigger.
History is bounded to three records. A 60-frame repair window triggers review
only if no inside gain and no mode-specific progress: guide COM progress >=1
or cohesion max-spread reduction >=1 count as progress. This progress proxy
is a predeclared experimental rule, not a validated failure diagnosis.

Both feedback arms use the same data, 60-frame holds, existing alarm/schema/
staleness/forecast guards, low-level target selection and six requests maximum
per run. They can review single alarms, unlike the old dual-alarm-only trigger.
The simple rule tries the other repair mode only when its alarm is active;
otherwise it retains rule_mode. LLM can continue or choose an admissible existing
mode using the supplied outcomes/state/forecasts. No target, force, distance,
or new-mode permission is added. Original main/batch defaults are unchanged.

API model settings are unchanged; prompt suffix is `-repair-feedback-v1`, with
instructions describing actual repair outcomes and fixed holds. Fast control
continues during inference. Pending frame polling is 0.05s in both feedback
arms; simple-rule response delay is fixed at 20 frames while API delay is real.
This is a whole-policy comparison, NOT a latency-isolated reasoning comparison
or an isolated ablation of history (trigger and hold policy also differ from
the prior LLM trial). Mode-only API permission remains a trial setting.

Protocol/settings/source hashes are saved before execution. Raw per-seed request,
response and cancellation events, runs and sampled states remain together under
`logs/anomaly_behavior/<timestamp>/`. Requests without recorded responses may
have been canceled after submission; do not interpret them as zero billed calls.
All results, including errors and regressions, remain in the report. No cache
or credentials are saved. No new modules or production feedback policy are added.

Completed 60 runs (three arms x 20 seeds), with 20 full-precision matched initial
groups. Results are exploratory; neither this cohort nor older differently
configured/seeding cohorts should be pooled to claim a validated improvement.

| Arm | Complete task | Mean capped frames | Mean inside /50 | Mean goal exits |
| --- | ---: | ---: | ---: | ---: |
| Original rule | 3/20 | 1976.45 | 23.75 | 124.85 |
| Simple repair-feedback rule | 4/20 | 1908.80 | 25.20 | 102.60 |
| DeepSeek repair-feedback | 2/20 | 1979.95 | 21.60 | 119.35 |

Simple feedback rescues seed 5010 (2000 -> 1146), with no success regression;
it also shortens seed 5001 from 1930 to 1431. This is a small exploratory benefit,
not a validated policy or evidence of LLM value. DeepSeek rescues no failures and
regresses seed 5001 from successful 1930 to capped failure with 46/50 inside.
Its partial completion is also lower on average. This implementation does not
justify adopting LLM feedback in production; the result is not a universal
refutation of outcome-feedback methods.

Simple feedback submits 102 requests: 77 recorded returns, 53 accepted, 24
rejected, 25 canceled, 8 runs/256 frames with actual mode changes. DeepSeek
submits 106: 81 recorded returns, 50 accepted, 31 rejected, 25 canceled,
6 runs/310 frames with actual mode changes. Most accepted responses retain
the rule mode. Recorded API returns have mean latency 1.029s; accepted frame
latency 20.46 versus exactly20 for the simulated-delay feedback rule.

Recorded backend errors are zero, but outputs/choices are NOT all valid:
DeepSeek has 17 selections of a mode whose required alarm was already absent
AT REQUEST TIME (not latency-induced stale choices), 2 missing/invalid modes,
6 invalid drive-point geometries and 6 forecast-dominated choices. These are
rejected by the existing guard; rejected or canceled requests do not constitute
applied interventions. Event accounting verifies requests = returns + cancels.
All request histories refer only to actually executed, completed past windows.

Artifacts: `logs/anomaly_behavior/20261004_165251_420836/`, with protocol,
`repair_feedback_runs.csv`, sampled `repair_feedback_states.csv`, paired deltas
against both rule baselines in `repair_feedback_pairs.csv`, full per-seed
`events_<arm>_<seed>.json`, and `repair_feedback_report.json` (includes audit).
Production and runner source hashes stayed unchanged during execution.
192 tests pass; default main remains 20 sheep, rule control, original repair
distances and target selection. Experimental feedback lives only in this
diagnostic; the production high-level request-readiness hook defaults to true.
No commit or push is made.

### Single-Factor Repair-Distance Screen

```powershell
python experiments/anomaly_behavior/run_characterization.py --repair-distance
```

Predeclared screen: seeds 4801-4820, flock sizes 20 and 50, 5A+5B+5C in both
sizes, one dog, space 200x200, start standard deviation 15, goal radius 20,
2,000 step calls, and existing rule alarms/target selection/repair enabled.
The abnormal fractions differ (75% and 30%); this is not a constant-fraction
population scaling test. No API calls, changed sheep behavior, or changed modes.

Five settings (HARD_TO_GUIDE_DISTANCE / COHESION_REPAIR_DISTANCE): baseline
3/10, guide-near 1.5/10, guide-far 6/10, cohesion-near 3/5, cohesion-far 3/20.
Each candidate changes one parameter only. These are desired stand-off positions,
not guaranteed actual dog/sheep separations. Full-precision initial geometry and
individual identities must match across all five variants within each seed/size.

Primary outcomes: complete-flock success by the existing pre-physics check and
mean capped frames. Secondary outcomes: full-precision post-step inside counts,
goal exits, danger frames, spread and time in each repair mode. Post-step inside
counts can differ from older rounded pre-step batch-log metrics; do not mix them.
Protocol/config/source hashes are saved before runs; runs, sampled states (each
ten frames plus endpoints), paired deltas and report stay in one timestamped
`logs/anomaly_behavior/` directory. All temporary configuration is restored.

For exploration, rank candidates by 50-sheep successes, then mean capped frames.
Only a candidate better than baseline proceeds to frozen comparison on seeds
4901-4920, at both sizes. This is a distance sensitivity test, NOT evidence that
adaptive distance or an LLM is better. Do not adopt defaults on screening alone.

Completed 200 runs on seeds 4801-4820. All 40 seed/size groups match on
full-precision geometry and individual identities across all five variants.
Source hashes stayed unchanged during execution. Applied-mode frame counts
sum to the physics-update count (terminal pre-physics checks excluded).

| Guide / cohesion distance | N=20 success | N=20 mean capped frames | N=50 success | N=50 mean capped frames |
| --- | ---: | ---: | ---: | ---: |
| 3 / 10 (baseline) | 20/20 | 1158.70 | 6/20 | 1714.65 |
| 1.5 / 10 | 18/20 | 1298.25 | 3/20 | 1887.15 |
| 6 / 10 | 17/20 | 1274.10 | 5/20 | 1781.80 |
| 3 / 5 | 20/20 | 1214.10 | 1/20 | 1993.90 |
| 3 / 20 | 19/20 | 1161.15 | 2/20 | 1983.55 |

No candidate improves either primary endpoint over baseline on N=50. Therefore
the predeclared validation condition is not met: seeds 4901-4920 are NOT run,
defaults remain 3/10, and no LLM distance-selection permission is introduced.
For example, 6/10 rescues two N=50 runs but regresses three; 1.5/10 rescues one
but regresses four. A benefit on individual seeds is not general improvement.
This limited screen does not prove 3/10 globally optimal and does not test
adaptive distance or response-dependent tuning; it supplies no basis to add
them now. All screened configurations, including negative results, are retained.

Artifacts: `logs/anomaly_behavior/20261004_162127_613684/`, with
`repair_distance_protocol.json`, `repair_distance_runs.csv`,
`repair_distance_states.csv`, `repair_distance_pairs.csv`, and
`repair_distance_report.json` (includes post-run audit and screening decision).
187 tests pass; added coverage checks one-factor variants, actual production
drive-point geometry, paired identities, config restoration including errors,
and invalid-input rejection. No API calls, production controller changes,
commit or push.

The existing characterization runner now has a rule-only population diagnostic:

```powershell
python experiments/anomaly_behavior/run_characterization.py --population-scale
```

Seeds 4601-4620 were run with 20 and 50 sheep, in normal and mixed populations
(80 runs total). Repair is enabled in all groups. Mixed populations are
5A+5B+5C+5-normal for N=20 and 13A+13B+12C+12-normal for N=50, using largest-
remainder allocation of approximately equal shares. Sheep identities are fixed
within each run. This is not a test of time-varying anomaly identities.

Workspace 200x200, Gaussian start standard deviation 15, goal radius 20, one
dog, force weights, speeds, detector thresholds, and rule control are unchanged.
Within each size, normal/mixed initial positions, dog and goal match exactly.
Across sizes the goal, dog start and sampling center match; entire flocks and
individual type assignments are not identical. Increasing size in a fixed
starting region also increases density, so this is not an isolated constant-
density scaling test.

The primary budget is 2,000 step calls. Unfinished runs continue without reset
to 6,000; extended results are reported separately, never substituted into the
primary result. Existing pre-physics success-check semantics are retained.

| Group | Success by 2,000 | Mean capped frames (2,000) | Mean inside at primary endpoint | Success by 6,000 |
| --- | ---: | ---: | ---: | ---: |
| 20 normal | 20/20 | 529.20 | 20.0/20 | 20/20 |
| 20 mixed | 19/20 | 989.90 | 19.8/20 | 20/20 |
| 50 normal | 19/20 | 826.85 | 49.9/50 | 20/20 |
| 50 mixed | 0/20 | 2000.00 | 37.7/50 | 0/20 |

The unfinished 50-normal run completes at 2,013; the unfinished 20-mixed run
at 2,341. In contrast, 50-mixed mean inside counts at 2,000/4,000/6,000 are
37.7/39.45/35.5. Mean excess distance beyond the goal radius is respectively
3.036/2.678/3.548. Longer runtime does not resolve this mixed cohort.
At 6,000, mean remaining counts are 2.25 normal, 0.50 A, 9.20 B, and 2.55 C:
14.50 in total. B accounts for most remaining sheep but is not the only type.
These observations identify a larger mixed-flock limitation, not its cause.

Normal and mixed 50-sheep groups have mean primary goal exits 86.10 and 106.75
respectively, versus 5.75 and 13.75 at N=20. Counts are not normalized for sheep
number or run duration and do not establish crowding as a causal explanation.
Possible issues include goal-region crowding, interaction-force scaling, fixed
collect/alarm thresholds, and repair scheduling. More failures do not imply
an LLM can fix them. A subsequent single-factor goal-radius/density diagnostic
would assess model/task feasibility, not count as a controller improvement.

Artifacts are together in `logs/anomaly_behavior/20261004_012034_569242/`:
`population_scale_protocol.json` (saved before runs, configuration and source
hashes), `population_scale_runs.csv`, `population_scale_states.csv` (every ten
step calls plus primary/final endpoints), and `population_scale_report.json`.
Tail-distance reduction and inside-count changes are descriptive; no new
"stuck" detector is introduced. No API, production configuration changes,
new modules, commit, or push are involved. Tests cover paired initialization,
configuration restoration, cap semantics, and invalid protocol rejection.

### Fifty-Sheep Anomaly-Ratio Sweep with Real DeepSeek Calls

Following explicit authorization to call the API, the existing runner also
supports this paired exploratory sweep:

```powershell
python experiments/anomaly_behavior/run_characterization.py --population-api
```

Fifty sheep, one dog, workspace 200x200, start standard deviation 15, goal
radius 20, existing repair enabled, and a 2,000-frame cap remain fixed. Seeds
4601-4620 are shared across three populations: 5A+5B+5C+35-normal (30%),
10A+10B+10C+20-normal (60%), and 13A+13B+12C+12-normal (76%). Each population
has 20 rule and 20 dual-layer runs: 120 runs, 60 matched pairs total. All 60
initial logged agent positions and identities match within controller pairs.
Population/settings/source hashes were saved before execution. Seeds and the
highest-ratio rule condition were previously inspected; this is exploratory
scaling evidence, not an untouched confirmatory benchmark.

The dual-layer arm uses actual uncached `deepseek-v4-flash` calls with existing
prompt `shepherd-mode-v10-repair-targets`, mode+target permissions, the dual-alarm
trigger, forecast guard, and asynchronous fast-layer fallback unchanged.
This tests the CURRENT slow layer, not the proposed new behavior-diagnosis role.
Only simulated state/forecast summaries are sent; keys are read from the process
environment and never saved in experiment files. Default main.py remains N=20.

| Abnormal share | Rule success | DeepSeek success | Mean capped frames, rule/DeepSeek | Mean final inside, rule/DeepSeek |
| --- | ---: | ---: | ---: | ---: |
| 30% | 2/20 | 4/20 | 1861.75 / 1812.85 | 27.05 / 33.05 |
| 60% | 0/20 | 0/20 | 2000 / 2000 | 35.95 / 36.90 |
| 76% | 0/20 | 0/20 | 2000 / 2000 | 37.65 / 33.45 |

Final-inside values use the batch runner's established rounded, last-recorded
pre-physics-state convention. They need not exactly match the earlier scale
diagnostic's full-precision post-update endpoint. Both controller arms use the
same convention; capped frames also retain the existing pre-physics success check.
Frames measure simulation completion, not total wall-clock/API overhead.

At 30%, two failures are rescued, with no success-to-failure reversals; eight
pairs have more final-inside sheep, ten tie, and two have fewer. The mean frame
reduction is 48.90 (2.63%). At 60%, inside counts improve in nine pairs, tie in
seven, and worsen in four, without any complete task success. At 76%, inside
counts improve in six, tie in three, and worsen in eleven. Its mean danger
frames rise 824.10 -> 850.50 and maximum spread 86.40 -> 93.391. Goal exits
fall 106.80 -> 90.65, but fewer sheep are inside overall; reduced exits alone
must not be interpreted as more successful retention.

| Share | Requests | Recorded returns | Accepted / rejected | Actual override frames | Runs with overrides |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30% | 84 | 79 | 60 / 19 | 130 | 12/20 |
| 60% | 90 | 88 | 64 / 24 | 355 | 18/20 |
| 76% | 137 | 133 | 90 / 43 | 572 | 20/20 |

There are 311 submitted requests, 300 recorded returns, zero cache hits, and
one recorded backend error: a JSONDecodeError at ratio 76%, seed 4616. It is
retained in the results rather than excluded. Mean return latency is about
0.93 seconds in each population. Requests without recorded returns can be
pending/cancelled at the episode/run end; request totals are not provider billing
counts. Actual overrides count a mode differing from the rule fallback or a
different repair target; accepted responses may still execute the rule choice.

Conclusion: the current LLM can help some larger-flock trajectories, but more
anomalies do not reliably increase that benefit. Thirty percent supplies a
modest positive exploratory signal, not a statistically conclusive general
advantage; the larger ratios remain unsuccessful and the highest worsens partial
completion. Do not choose only favorable seeds/ratios as a claim of superiority.
If the 30% candidate is pursued, repeat on independently specified seeds before
changing the policy or claiming a robust effect. Task feasibility and goal-region
crowding remain separate questions; no force, threshold, goal, prompt or formal
controller changes were made for this sweep.

All artifacts are under `logs/anomaly_behavior/20261004_014212_595111/`:
`population_api_protocol.json`, `population_api_report.json`, and the supplementary
`population_api_audit.json`. Each ratio contains `paired.csv` and rule/deepseek
`summary.csv`, metadata, raw agent Parquet and raw control Parquet. Summaries are
updated after every run and systemic credential/billing/endpoint failures abort
the sweep rather than silently produce a fallback-only "LLM" result. The model,
latency, override and parse-error audit was derived from saved control logs after
completion. Re-running uses real provider calls and may produce different choices
and asynchronous intervention timing; it is not an exact deterministic API replay.

#### Independent-Seed Follow-Up of the 30% Condition

The 30% condition was selected from the preceding exploratory sweep, then
retested on seeds 4701-4720 (20 new rule/DeepSeek pairs, 40 runs). These seeds
were absent from existing CSV experiment records before execution. No threshold,
prompt, permission, trigger, force, workspace or goal change intervened. Production
source hashes and functional configuration match the preceding sweep; only the
diagnostic runner gained a ratio-selection argument, with its old default intact.

```powershell
python experiments/anomaly_behavior/run_characterization.py --population-api --ratios 30 --first-seed 4701 --seed-count 20
```

| Metric | Rule | DeepSeek | DeepSeek minus rule |
| --- | ---: | ---: | ---: |
| Task successes | 3/20 | 2/20 | -1 |
| Mean capped completion frames | 1863.05 | 1865.35 | +2.30 |
| Mean final-inside sheep | 26.10 | 36.70 | +10.60 |
| Mean danger frames | 771.15 | 754.75 | -16.40 |
| Mean maximum spread | 89.771 | 87.2045 | -2.5665 |
| Mean individual goal exits | 114.45 | 128.00 | +13.55 |

The LLM rescues seed 4716 (2,000 -> 933 frames) but regresses 4713
(1,514 -> 2,000, final 47/50 inside) and 4719 (1,373 -> 2,000, final 45/50).
There are eleven higher final-inside pairs, five ties, and four lower pairs.
Sixteen runs execute actual mode/target overrides for 295 frames total. The
batch records 82 requests, 80 returns, 66 accepted, 14 rejected, zero cache
hits and zero backend errors, using the same model/prompt as before. Mean
recorded response latency is 0.938 seconds. All 20 initial logged agent states
match within pairs, and the two cohorts' seed sets are disjoint.

The prior complete-task success improvement does NOT replicate. Final-inside
counts improve descriptively in both cohorts, but that is a separate secondary
endpoint, not completion of the original all-sheep task; goal exits also increase
in the new cohort. Keep exploratory and validation results separate, rather than
pooling them to claim a confirmed success-rate win or changing the success
definition after seeing results. The slow layer has evidence of helpful and
harmful interventions, not a demonstrated robust advantage on task completion.
This follow-up still evaluates the existing mode/target slow layer, not a new
behavior-diagnosis architecture. It does not establish why end-of-task failures
occur; goal-region/remaining-sheep limitations need a separate diagnosis.

Artifacts are under `logs/anomaly_behavior/20261004_015855_871537/`, with the
same protocol/report/raw-log layout plus `population_api_validation_audit.json`.
The audit records source/config agreement with the reference sweep, actual model
and prompt, API returns, overrides, latency, paired differences and failure
reversals. No formal controller edits or commit/push were made. The full test
suite passed 182 tests; focused tests also verify selected-ratio execution and
rejection of duplicate/unsupported ratios before output is created.

## Experimental Type C: Near-Dog Sensitivity

Type C preserves normal flocking and increases only dog repulsion within 30
units. The multiplier is `1 + 2 * clip(1 - distance / 30, 0, 1)`; actual speed
remains capped at 0.7 units/frame. These are provisional behavior parameters,
not biologically fitted values. No C-specific alarm or repair mode is added.

The focused tests live here rather than in a separate testing workspace:

```powershell
python -m unittest discover -s tests -p test_type_c_behavior.py -v
```

They check near/far response, continuity, speed limits, unchanged dog-free
flocking and goal retention, reproducible identities and geometry, population
validation, shared forecast dynamics, high-level summaries, and rendering.
The default A/B characterization outputs above are unchanged; the shared
`dog_response_curve` helper also accepts C for focused checks.

### First Matched-Seed Pilot

Seeds 3101-3120, 20 sheep, initial-position standard deviation 15, 2,000-frame
cap, rule controller only (no API requests), with identical initial geometry
in all three groups:

| Population / repair | Success | Mean capped frames | Mean danger frames | Mean maximum spread |
| --- | ---: | ---: | ---: | ---: |
| 20 normal, repair off | 18/20 | 613.65 | 204.6 | 41.151 |
| 16 normal + 4C, repair off | 17/20 | 805.40 | 309.1 | 42.518 |
| 16 normal + 4C, existing rule repair on | 20/20 | 544.40 | 128.4 | 45.947 |

Failed runs contribute the 2,000-frame cap to completion-time averages.
Danger frames count frames with at least one existing anomaly alarm active.
The results are descriptive, not a statistical claim. C changes behavior and
the existing repair rescues its three no-repair timeouts in this cohort, but
maximum spread increases with repair. C is not uniformly worse than normal
across seeds; this pilot does not establish a need for new repair modes or
an LLM benefit. Do not tune severity solely to make the LLM win.

Outputs are together under `logs/experiments/type_c_initial_validation/`:
`normal_no_repair/summary.csv`, `type_c_no_repair/summary.csv`, and
`type_c_rule_repair/summary.csv`. `ab_regression/summary.csv` contains the
unchanged 7A+7B+0C rule runs on seeds 3001-3020: 20/20 successes, mean
1,004.45 frames. All 20 agent and metric Parquet trajectories exactly match
the earlier `population_stress_3001_3020/rule/` runs, not just the summaries.

To repeat a C-only group using the existing batch runner:

```powershell
python run_experiments.py --scenario type-c --controller rule --repair --runs 20 --seed-mode sequence --batch-seed 3101 --max-steps 2000 --max-log-files 100000
```

Replace `--repair` with `--no-repair` for its baseline, or use `--scenario normal`
with `--no-repair` for the normal comparison. Reusing the same seeds preserves
initial geometry. Leave the high log-file limit in place when retaining all
raw trajectories for a cohort.

### A/B/C Mixed: Rule versus DeepSeek (2026-10-03)

The next pilot used 3A + 3B + 3C + 11 normal sheep on matched seeds 3201-3220,
with repair enabled, initial-position standard deviation 15, and a 2,000-frame
cap. Both arms used the same source hashes, case settings, and logged initial
positions and sheep identities (verified for all 20 pairs). No repair policy,
system prompt, or trigger was changed for this comparison. Type C's definition
was available through its structured state summary and shared forecast dynamics.

DeepSeek used real, uncached calls, model `deepseek-v4-flash`, and prompt
`shepherd-mode-v9-aligned-candidates`. The original trigger was retained:
one concurrent-alarm frame suffices, and requests are not limited to one per
episode. The fast controller stays active during inference; the existing batch
runner sleeps 0.05 seconds per pending-request frame. Completion frames below
measure simulation steps, not wall-clock time or total API overhead.

| Metric | Rule repair | DeepSeek dual-layer | DeepSeek minus rule |
| --- | ---: | ---: | ---: |
| Success | 19/20 | 19/20 | 0 |
| Mean capped completion frames | 728.1 | 724.0 | -4.1 (-0.56%) |
| Mean danger frames | 217.5 | 214.1 | -3.4 |
| Mean maximum spread | 62.8490 | 62.6925 | -0.1565 |
| Mean goal exits | 8.00 | 8.45 | +0.45 |

DeepSeek was faster in two pairs, equal in 17, and slower in one. Seed 3202
improved from 1,014 to 751 frames; seed 3204 was rescued from the 2,000-frame
cap to 1,474; seed 3213 regressed from 1,293 to the cap. The median paired frame
change was zero; the exploratory exact sign-randomization p-value was 1.0.
This small cohort shows both helpful and harmful interventions, not a reliable
overall LLM advantage. It does not prove an absence of benefit in other mixed
conditions or with a different high-level decision interface.

Eleven runs submitted API requests. Across the batch there were 32 requests,
30 recorded responses, 23 accepted decisions, seven rejected decisions, zero
cache hits, and zero recorded backend errors. Mean response latency was 0.8353
seconds. Non-rule mode execution occurred in three runs for 42 frames total;
584 high-level active frames include choices identical to the rule fallback.
The other 17 runs did not execute a different mode. All 32 logged request
summaries included the correct C count and response definition. The rule arm
had concurrent alarms in 11 runs, totaling 895 frames. Thus the trigger did
activate, but actual changes to rule control remained sparse.

Recorded responses used 57,172 input and 1,342 output tokens. These totals do
not include possible usage from the two requests without a logged response.
Rejections were: selected alarm no longer active (3), forecast dominated by
cohesion priority (2), candidate drive point out of bounds (1), and forecast
dominated by target tracking (1). An accepted decision is not automatically
an override or evidence of improved task performance.

Both summaries, metadata, and `comparison.csv` / `comparison.json` are together
under `logs/experiments/abc_mixed_3201_3220/`. The existing runner reproduces
the setup without adding a new experiment script:

```powershell
python run_experiments.py --scenario custom --abnormal-a 3 --abnormal-b 3 --abnormal-c 3 --repair --controller rule --runs 20 --seed-mode sequence --batch-seed 3201 --max-steps 2000 --max-log-files 100000
python run_experiments.py --scenario custom --abnormal-a 3 --abnormal-b 3 --abnormal-c 3 --repair --controller deepseek --runs 20 --seed-mode sequence --batch-seed 3201 --max-steps 2000 --no-llm-cache --max-log-files 100000
```

Fresh API outputs and response arrival frames can vary even with fixed simulation
seeds; these commands reproduce conditions, not guaranteed identical LLM results.

### Higher A/B/C Population: 5A + 5B + 5C (2026-10-03)

Following the 3A+3B+3C pilot, the abnormal population was increased to 5A +
5B + 5C + five normal sheep. Twenty rule and twenty uncached DeepSeek runs
reused seeds 3201-3220, the 2,000-frame cap, initial-position standard deviation
15, and unchanged behavior coefficients, controller, prompt, and trigger.
Source hashes and base configuration match the preceding pilot. Both architecture
arms have identical logged initial positions and identities in all 20 pairs.
Because seeds are reused and the larger population was chosen after inspecting
the earlier results, this is an exploratory sensitivity check, not an independent
confirmation cohort.

| Metric | Rule repair | DeepSeek dual-layer | DeepSeek minus rule |
| --- | ---: | ---: | ---: |
| Success | 19/20 | 19/20 | 0 |
| Mean capped completion frames | 1,018.30 | 1,113.35 | +95.05 (+9.33%) |
| Mean danger frames | 328.55 | 375.70 | +47.15 |
| Mean maximum spread | 71.3895 | 72.0800 | +0.6905 |
| Mean goal exits | 15.40 | 19.05 | +3.65 |

Both arms timed out on seed 3213; there were no rescues or additional timeouts.
DeepSeek was faster in one pair, equal in 13, and slower in six. Changed pairs
were: 3201 (1,409 -> 1,641), 3202 (1,026 -> 950), 3207 (544 -> 759), 3210
(1,159 -> 1,406), 3215 (695 -> 885), 3216 (359 -> 1,434), and 3220
(1,386 -> 1,404). The largest slowdown is seed 3216; the median paired frame
change is zero. These are simulation-frame comparisons, not wall-clock speedups.

Compared with the preceding 3A+3B+3C rule arm, rule mean capped frames rose
from 728.1 to 1,018.3 (+39.86%), with 16 slower and four faster seeds and the
same 19/20 aggregate success. Thus the larger population increased time pressure
in this cohort but did not create a larger success-rate deficit.

The rule arm had concurrent alarms in 18 runs, totaling 1,717 frames. DeepSeek
made 63 requests in 18 runs and recorded 61 responses: 41 accepted and 20
rejected, with zero cache hits and zero recorded backend errors. All 63 request
summaries contained the correct C count and behavior definition. Actual non-rule
mode execution occurred in seven runs, totaling 135 frames; 1,039 high-level
active frames also include decisions identical to rule fallback. Mean recorded
API response latency was 0.8687 seconds. Recorded responses used 116,328 input
and 2,841 output tokens, excluding possible usage from the two requests without
a logged response.

Rejections were: selected alarm no longer active (7), candidate drive point out
of bounds (4), forecast dominated by hard-to-guide (4), forecast dominated by
target tracking (3), and forecast dominated by cohesion priority (2).

Increasing the abnormal counts caused more requests and more overrides, but did
not improve overall results in this batch. Do not increase counts or severity
repeatedly solely to obtain a favorable LLM comparison. Inspect useful versus
harmful interventions before changing the existing decision interface; this
result does not rule out other high-level policies or heterogeneous scenarios.

Summaries, metadata, and `comparison.csv` / `comparison.json` are together under
`logs/experiments/abc_high_population_3201_3220/`. To reproduce the conditions,
use the preceding batch commands with `--abnormal-a 5 --abnormal-b 5
--abnormal-c 5`; actual fresh API outputs and arrival frames may differ.
No new experiment script, repair mode, or scenario preset was added.

### Offline Diagnosis of the Seed 3216 Slowdown

The paired 5A+5B+5C agent states are identical through frame 201, before the
first override is applied. Request `high-1` was triggered at frame 181 and
returned at frame 201 (0.9972 seconds, 20 simulation frames later). It chose
`TARGET_TRACKING` instead of the rule's `HARD_TO_GUIDE`, with a requested
30-frame hold. It actually executed on frames 201-215 inclusive, then released
when the alarm context changed after the minimum dwell period. The target at
this intervention was sheep 4, Type B. Later accepted choices were identical
to rule fallback; the later cohesion choice was rejected.

Two API-free, deterministic replays used the normal production environment,
detector, and low-level controller with seed 3216 and the same configuration.
Pure rule replay completed in 359 frames and exactly matched the entire logged
rule agent trajectory. Adding only the fixed `TARGET_TRACKING` override at
201-215, with rule control everywhere else, completed in 1,434 frames and
exactly matched the entire logged DeepSeek agent trajectory. No model or policy
was edited. Thus that single 15-frame intervention is sufficient to reproduce
the slowdown in this trajectory without API latency or later LLM choices.

The choice was not simply ignoring the supplied forecast. At request time,
30-frame shadow forecasts favored target tracking on centroid progress
(5.211 versus 3.655 for hard-to-guide) and final spread (31.917 versus 43.908).
Actual logged metrics also looked better briefly: at frame 231, spread was
31.32 versus rule's 56.41, with seven sheep inside in both arms; at frame 265,
the override trajectory had 15 inside versus rule's 13. Nevertheless, by frame
358, rule had all 20 inside while the override trajectory had 17, with Type B
sheep 0, 4, and 16 still outside. Short-term flock metrics did not predict
long-term completion. Goal-exit totals were 79 versus 12 over different-length
trajectories; these cumulative totals are descriptive, not duration-adjusted
evidence of an immediate exit-rate increase.

Switching to target tracking also clears the low-level repair-target memory.
The resulting motion and target-selection history can alter subsequent cleanup
even after rule control resumes. This replay identifies the harmful intervention,
but does not separately attribute the effect to memory reset versus changed dog
geometry. The forecast guard rejects invalid or dominated short-horizon choices;
it does not guarantee a shorter full-run completion time. Do not generalize this
single-seed explanation to every slowdown without inspecting the other cases.

### Repair-Target Interface Validation (2026-10-03)

The new interface adds only optional target selection to the three existing
modes. Each repair mode offers at most three currently outside-goal candidates,
including its held rule target; an explicit target is predicted and validated
as that target, while null preserves rule selection. Mode triggers, population
defaults, force parameters, and motion geometry are unchanged. Prompt/cache
version is `shepherd-mode-v10-repair-targets`.

New focused interface tests are here in `test_high_level_target_selection.py`.
They cover valid execution and holding, candidate bounds and preview purity,
null delegation, malformed/unoffered targets, arrival during inference and
execution, selected-target boundary/forecast checks, and target-only logging.
Some contract tests intentionally bypass forecast ranking to exercise target
execution; they are not evidence that the chosen target improves the task.
API serialization/cache tests extend the existing backend test module. The full
suite passes 154 tests without real API calls.

A fresh 20-run 5A+5B+5C rule regression reused seeds 3201-3220. Results remain
19/20 successes and 1,018.3 mean capped frames. The runner outputs are together
under `logs/experiments/target_selection_validation/rule_regression/`.
Target-only override counts compare against a same-frame rule preview with
current controller memory, not an independently simulated rule trajectory.
Mode-reference agreement does not measure target-selection accuracy. The first
fresh matched API comparison below evaluates outcomes separately from these
contract and regression checks.

### Target-Selection Interface: 50 Matched Runs per Architecture (2026-10-03)

Fifty rule and fifty uncached DeepSeek runs used new seeds 3301-3350, 5A + 5B +
5C + five normal sheep, repair enabled, a 2,000-frame cap, and initial-position
standard deviation 15. Both arms used identical source hashes, populations,
seed plans, and frame caps. Initial logged agent positions and identities match
exactly in all 50 pairs. No behavior, trigger, or controller parameter was tuned
during the batch. All 154 local tests passed before execution.

| Metric | Rule repair | DeepSeek mode + target | DeepSeek minus rule |
| --- | ---: | ---: | ---: |
| Success | 46/50 (92%) | 42/50 (84%) | -8 percentage points |
| Mean capped completion frames | 1,027.42 | 1,093.50 | +66.08 (+6.43%) |
| Mean danger frames | 332.52 | 361.10 | +28.58 |
| Mean maximum spread | 70.2536 | 74.5354 | +4.2818 |
| Mean goal exits | 16.12 | 14.76 | -1.36 |

DeepSeek was faster in 11 pairs, equal in 23, and slower in 16. It rescued
seeds 3308 and 3347 but regressed six rule successes (3303, 3318, 3324, 3338,
3342, 3345) to timeouts. The median paired frame difference was zero. The
exploratory paired-bootstrap 95% interval for the mean frame difference was
[-35.481, 169.1205] frames (20,000 resamples, RNG seed 20261003). The exact
paired success McNemar p-value was 0.2890625. Neither outcome difference is a
statistically conclusive population-level finding from this one cohort; the
observed point estimates do not show an improvement. Failure runs contribute
2,000 frames to the completion averages. These are simulation steps, not actual
elapsed time or a comparison of API/computation costs.

The largest improvement was seed 3347 (2,000 -> 915 frames); the largest
regression was seed 3345 (929 -> 2,000). Other substantial changes included
3335 (1,649 -> 912), 3330 (1,968 -> 1,264), and 3350 (540 -> 1,475). Do not
use one of these selected examples alone to choose a global policy change.

API requests occurred in 43 runs, with 138 requests, 128 recorded responses,
73 accepted and 55 rejected, zero cache hits, and zero recorded backend errors.
Recorded model was `deepseek-v4-flash`, prompt `shepherd-mode-v10-repair-targets`.
Mean response latency was 1.0219 seconds. Recorded responses used 451,104 input
and 7,020 output tokens; usage from the ten requests without logged responses
is not included. Among accepted responses, 66 supplied explicit target IDs.

Actual mode or target-only overrides occurred in 28 runs, totaling 593 frames:
405 mode-change frames in 14 runs and 188 same-mode target-change frames in
22 runs. These run counts overlap. The target-change comparison is against the
same-frame rule preview with its current memory, not a separate no-intervention
history. There were 1,753 high-level active frames in total; active does not
necessarily mean the command differs from rule fallback. All 138 logged
requests included the correct C count and bounded per-mode shortlists. On all
1,603 explicit-target execution frames, the selected ID matched the low-level
target, belonged to the request's shortlist for that mode, and remained outside
the goal. Thus target selection actually reached execution; the outcome cannot
be dismissed as an experiment that never used the new interface.

Rejected responses: selected alarm no longer active (20), drive point outside
the workspace (14), dominated by the same mode's rule target (9), dominated by
cohesion priority (5), dominated by target tracking (3), forecast preference
reversed (2), and dominated by hard-to-guide (2). The rule arm had concurrent
alarms in 43 runs, totaling 3,362 frames.

Summaries, metadata, and the existing pair-analysis tool's `comparison.csv` /
`comparison.json` are together under
`logs/experiments/target_selection_50_3301_3350/`. Reproduce the setup with the
existing runner:

```powershell
python run_experiments.py --scenario custom --abnormal-a 5 --abnormal-b 5 --abnormal-c 5 --repair --controller rule --runs 50 --seed-mode sequence --batch-seed 3301 --max-steps 2000 --max-log-files 100000
python run_experiments.py --scenario custom --abnormal-a 5 --abnormal-b 5 --abnormal-c 5 --repair --controller deepseek --runs 50 --seed-mode sequence --batch-seed 3301 --max-steps 2000 --no-llm-cache --max-log-files 100000
```

This evaluates the new mode-and-target interface as a whole against rules.
At the time of this batch there was no mode-only API arm on these 50 seeds, so the independent effect of
target selection cannot be separated from the prompt/mode decisions. Do not
pool with the earlier 20-seed v9 results to claim that target selection caused
a particular percentage change. Fixed simulation seeds do not make fresh API
outputs or response arrival frames deterministic. Keep rule control as the
default and treat the new interface as experimental; this cohort does not
justify calling it a performance improvement or expanding LLM permissions.

### Mode-Only Ablation: Schema-Only Pilot (2026-10-03)

After the original cohort, `--llm-mode-only` was added to the existing batch
runner. Default mode-and-target behavior is unchanged. It disables target
permission in the execution interface, records the restriction in configuration,
and isolates the response-cache namespace. Physics, triggers, candidate summaries,
and forecasts are unchanged. All 158 tests passed before execution.

The first supplemental 50 runs kept the exact original system instructions but
restricted the response schema to `target_sheep_id: null`. DeepSeek nonetheless
returned non-null IDs in 15 of 118 recorded responses; all were rejected as
`target_selection_disabled`, with zero explicit targets executed. This is a
provider schema-compliance issue, not an executed target override. These runs
remain under `mode_only/` with prompt `shepherd-mode-v10-repair-targets-mode-only`.

Pilot outcomes: 47/50 success, mean capped frames 1,042.36, mean danger frames
337.02, mean maximum spread 70.4086, mean goal exits 16.64. Against rules there
were five faster, forty equal, five slower runs; seed 3314 was rescued and no
rule success regressed to timeout. There were 122 requests, 118 recorded
responses, 63 accepted, 55 rejected, no cache hits or recorded backend errors,
and 218 actual mode-override frames across 12 runs. The 15 permission rejections
confound a clean mode-only comparison. Do not use this pilot as the main arm
or silently pool it with the corrected experiment.

The corrected mode-only prompt appends an explicit instruction that candidates
are context only and that the target must always be JSON null. Base instructions
and input fields remain identical to the mode-and-target arm. Prompt version is
`shepherd-mode-v10-repair-targets-mode-only-v2`; all fifty original seeds are
rerun under `mode_only_v2/`, rather than selecting just problematic or favorable
seeds. The prompt difference and fresh asynchronous API nondeterminism must be
acknowledged when interpreting this ablation.

### Mode-Only Ablation: Corrected 50-Run Comparison (2026-10-03)

The corrected arm completed all seeds 3301-3350 with 5A + 5B + 5C + five normal
sheep, repair enabled, 2,000-frame cap, initial standard deviation 15, no cache,
and unchanged dual-alarm trigger/polling settings. Its initial logged positions
and agent identities match both original arms in all 50 pairs. The only changed
configuration entry is the new target-permission flag (false here, previously
absent and implicitly enabled). Source hashes differ in configuration, runner,
API backend, and the high-level permission check; hashes for dynamics, environment,
detector, control contracts, low-level controller, and mode forecasts are unchanged.
The extra prompt instruction removes target permission, not state information.

| Metric | Rule repair | LLM mode only (v2) | LLM mode + target |
| --- | ---: | ---: | ---: |
| Success | 46/50 (92%) | 47/50 (94%) | 42/50 (84%) |
| Mean capped completion frames | 1,027.42 | 1,054.00 | 1,093.50 |
| Mean danger frames | 332.52 | 350.54 | 361.10 |
| Mean maximum spread | 70.2536 | 70.5422 | 74.5354 |
| Mean goal exits | 16.12 | 17.98 | 14.76 |

Against rules, mode-only was faster in four pairs, equal in 39, and slower in
seven. It rescued seed 3348 (2,000 -> 1,243) and did not turn any rule success
into a timeout. Mean frame difference was +26.58 (+2.59%), median zero. Other
improvements included 3310 (1,488 -> 991) and 3335 (1,649 -> 1,420); regressions
included 3312 (682 -> 1,298) and 3343 (770 -> 1,381). The paired-bootstrap 95%
interval for the mean difference was [-32.241, 84.1005] frames; exact paired
success McNemar p = 1. These figures use 20,000 percentile-bootstrap resamples
of the 50 seed pairs with RNG seed 20261003. Failures contribute the full
2,000-frame cap; frame averages are not API wall-clock costs.

Against corrected mode-only, mode-and-target was faster in 14 pairs, equal in
21, and slower in 15. It rescued 3308 and 3347 but lost seven successes (3303,
3318, 3324, 3338, 3342, 3345, 3348). Its mean frame difference was +39.50,
median zero; paired-bootstrap interval [-64.78, 145.2405], McNemar p = 0.1796875.
Thus the point estimates favor the narrower permission on success and capped
completion frames in this cohort, but not conclusively at population level.
Fresh LLM outputs, asynchronous arrival timing, and the necessary permission
instruction prevent treating this as an exact deterministic isolation of a
single target choice. One provider sample per seed is not a measure of repeated
API variability. Do not infer that target selection is universally harmful.

The corrected mode-only arm made 129 requests in 43 runs, with 119 recorded
responses, 65 accepted and 54 rejected. All 119 recorded targets were JSON null;
there were zero target overrides, cache hits, or recorded backend errors. Actual
mode overrides occurred in 11 runs for 238 frames; high-level active frames
totaled 1,688. All 129 request summaries retained C count, bounded repair
shortlists, and all three mode forecasts. Mean response latency was 1.0144 s.
Recorded input/output usage was 424,662 / 6,416 tokens; ten requests without
recorded responses are not included in this usage figure. Recorded model was
`deepseek-v4-flash`, prompt `shepherd-mode-v10-repair-targets-mode-only-v2`.

Rejections: selected alarm no longer active (19), drive point outside workspace
(15), dominated by target tracking (9), dominated by cohesion (6), dominated by
hard-to-guide (3), and forecast preference reversed (2). No response was rejected
for supplying a forbidden target in the corrected arm.

Under `logs/experiments/target_selection_50_3301_3350/`, the main ablation uses
`mode_only_v2/summary.csv`, `rule_vs_mode_only_v2.csv` / `.json`, and
`mode_only_v2_vs_mode_target.csv` / `.json`. The earlier `mode_only/` and the
comparison files without `v2` belong to the schema-only pilot, not the main arm.

```powershell
python run_experiments.py --scenario custom --abnormal-a 5 --abnormal-b 5 --abnormal-c 5 --repair --controller deepseek --llm-mode-only --runs 50 --seed-mode sequence --batch-seed 3301 --max-steps 2000 --no-llm-cache --max-log-files 100000
```

Conclusion: mode-only preserves success better than mode-and-target in this
sample, but is not an overall demonstrated improvement over rules. Do not add
anomalies, modes, or further LLM permissions on the strength of these results.
Keep rules as the stable default and both LLM permission variants experimental.
The next analysis, if undertaken, should use the existing mixed improved and
regressed pairs rather than selecting only favorable examples or changing the
physics to produce a desired result. No commit, push, or main-branch change was
made for this supplemental experiment.

### Target-Only Ablation: Completing the Four Arms (2026-10-03)

The existing runner now accepts `--llm-target-only`, mutually exclusive with
`--llm-mode-only`. It sets mode permission false and target permission true.
The base v10 instructions and state fields are retained, with an explicit
instruction to copy `rule_mode` and choose only from that mode's shortlist.
The output schema restricts mode to the request's rule mode. A different
returned mode is rejected; a response whose rule mode changed during inference
is also rejected. On execution, no mode override is passed to the fast layer.
If the rule mode changes during a target hold, the hold is released immediately,
even before the minimum dwell interval. Goal entry also releases the target.
The LLM still suggests the target hold duration within the existing bounds.
There are no new modes, triggers, movement parameters, or physical behaviors.
All 166 tests passed before execution.

Fifty uncached DeepSeek runs used the same seeds 3301-3350, 5A + 5B + 5C + five
normal sheep, repair enabled, initial standard deviation 15, and 2,000-frame cap.
Initial logged agent positions and identities match all three previous arms in
all 50 pairs. Dynamics, detector, low-level control, environment, control
contracts, and forecast source hashes are unchanged from the original cohort.
Source changes are limited to configuration, the runner, API permissions, and
high-level permission enforcement. Previously absent permission flags implicitly
enabled both choices; here mode permission is false and target permission true.

| Metric | Rule repair | LLM mode only (v2) | LLM target only | LLM mode + target |
| --- | ---: | ---: | ---: | ---: |
| Success | 46/50 (92%) | 47/50 (94%) | 45/50 (90%) | 42/50 (84%) |
| Mean capped completion frames | 1,027.42 | 1,054.00 | 1,114.72 | 1,093.50 |
| Mean danger frames | 332.52 | 350.54 | 375.28 | 361.10 |
| Mean maximum spread | 70.2536 | 70.5422 | 71.9252 | 74.5354 |
| Mean goal exits | 16.12 | 17.98 | 17.34 | 14.76 |

Target-only versus rules: nine faster, 25 equal, 16 slower runs. It rescued
3308 (2,000 -> 1,142) and 3348 (2,000 -> 1,723), but regressed three rule
successes to timeouts: 3335, 3338, 3344. Mean capped frame difference is +87.30
(+8.50%), median zero. The exploratory paired-bootstrap 95% interval is
[4.1585, 173.661] frames; paired success McNemar p = 1. Other substantial
regressions include 3313 (491 -> 1,521), 3334 (1,157 -> 1,956), and 3344
(1,087 -> 2,000). Improvement 3310 (1,488 -> 1,215) does not offset the overall
frame disadvantage. The bootstrap excludes zero for this comparison, but it is
an exploratory cohort estimate without multiple-comparison adjustment or
repeated API draws; do not generalize it to every target-selection policy.

Target-only versus corrected mode-only: 11 faster, 21 equal, 18 slower runs;
mean difference +60.72 frames, bootstrap interval [-36.6035, 159.901], success
rescues one (3308), regressions three (3335, 3338, 3344), McNemar p = 0.625.
Mode-and-target versus target-only: 14 faster, 20 equal, 16 slower runs;
mean difference -21.22 frames, bootstrap interval [-148.1025, 104.521], three
success rescues but six regressions, McNemar p = 0.5078125. All intervals use
20,000 percentile-bootstrap resamples of seed pairs, RNG seed 20261003. Failure
runs contribute 2,000 frames. These are simulation frames, not elapsed API time.

The target-only arm made 146 requests in 43 runs, with 132 recorded responses:
74 accepted, 58 rejected, no cache hits. All 74 accepted responses supplied an
explicit target. There were 1,763 active explicit-target execution frames and
328 actual same-mode target-change frames in 26 runs. The latter compares with
the same-frame rule preview using current controller memory, not an independent
never-intervened rule history. Actual mode-override frames were zero: all 55,691
logged control frames match the rule mode for their own contemporaneous alarms.
Chosen IDs belong to their request's offered shortlist, match the executed
target, and remain outside the goal on all explicit execution frames. Rounded
logged positions give a minimum selected-target goal distance of 22.8933.
All 24 runs without actual target overrides match the rule arm's entire logged
agent/metric trajectories exactly.

All 146 summaries retain C count, both mode shortlists, and all three forecasts.
Because requests still require concurrent alarms, their rule mode is always
`HARD_TO_GUIDE`. Thus this experiment tests target selection in that existing
trigger context, not LLM target selection for every repair mode. The mode-label
agreement is imposed by permission constraints and is not a meaningful measure
of LLM decision accuracy or task improvement.

Recorded model is `deepseek-v4-flash`, prompt
`shepherd-mode-v10-repair-targets-target-only-v1`. Of 132 replies, 128 supplied
the required rule mode. Two lacked a valid mode and were rejected as
`mode_not_allowed` (3306, 3330); two produced JSON decoding errors and were
rejected as `backend_error` (3315, 3346). None executed an unauthorized mode.
These failures are retained in the 50-run outcomes; individual seeds were not
rerun or replaced. Mean recorded response latency was 1.0424 s. Recorded
input/output usage was 472,450 / 7,373 tokens; fourteen requests without logged
responses are not included in those usage counts.

Other rejections: drive point outside workspace (15), rule mode changed during
inference (14), dominated by cohesion priority (9), dominated by the rule target
(8), and dominated by target tracking (8). Logged active releases: maximum hold
reached (51), rule mode changed (8), and all alarms cleared (15).

Data are under `logs/experiments/target_selection_50_3301_3350/target_only/`.
The existing analysis tool generated `rule_vs_target_only.csv` / `.json`,
`mode_only_v2_vs_target_only.csv` / `.json`, and
`target_only_vs_mode_target.csv` / `.json` alongside the earlier comparisons.

```powershell
python run_experiments.py --scenario custom --abnormal-a 5 --abnormal-b 5 --abnormal-c 5 --repair --controller deepseek --llm-target-only --runs 50 --seed-mode sequence --batch-seed 3301 --max-steps 2000 --no-llm-cache --max-log-files 100000
```

Conclusion: target-only does not improve the overall observed outcomes over
rules, despite actual target interventions and two rescued failures. Adding
target permission is therefore not supported as a performance improvement by
either this arm or the mode-and-target arm. The mode-only arm has the highest
observed success count, but rules have the best mean capped frames and danger
frames. Keep rules as the stable default; preserve all three LLM variants as
experimental. Fresh API decisions, hold choices, response timing, guard
rejections, and the mandatory scope instructions limit causal interpretation.
No commit, push, or main-branch update was performed.

### Offline Two-Stage Order Diagnostic (2026-10-04)

This diagnostic reuses `run_concurrent_mode_priority_trial.py`; it adds no
production controller, API call, anomaly, force parameter, or mode. It replays
all 50 original rule seeds 3301-3350 (5A + 5B + 5C + five normal sheep) and
checks the entire logged agent/metric trajectory against the original Parquet
data, as well as success, frames, danger frames, spread, and goal exits. All 50
baseline replays match exactly. All 172 tests passed before execution.

The checkpoint is the first concurrent splitting/stagnation state in each
rule trajectory, selected without filtering on final success or LLM outcomes.
Forty-three seeds have such a state; seven have none and retain rule outcomes.
Branches copy full-precision sheep/dog positions and velocities, sheep identities,
goal, detector history, and low-level held-target memory. Rounded log positions
are not used to reconstruct physics. The common checkpoint fingerprints are
checked before branching and after all branches to detect accidental mutation.

Five branches are compared: unchanged rules; collect then guide; guide then
collect; collect twice; guide twice. The repeated-skill controls avoid attributing
every difference to two different skills being sequenced. Each stage has the
same pre-existing maximum hold limit of 60 frames, so no plan can intervene for
more than 120 frames. Actual execution durations need not be equal: a stage is
skipped or ended when its required alarm is absent, collection ends when splitting
clears, and guiding can end when its last rule-selected target arrives. Target
selection and all motion remain the unmodified low-level rules. After the single
two-stage intervention, control returns to ordinary rules for the remainder of
the original 2,000-frame cap. This is conditional stage execution, not a fixed-time
open-loop permutation and not repeated replanning throughout the run.

| Branch | Success, all 50 | Mean capped frames | Faster / equal / slower than rule |
| --- | ---: | ---: | ---: |
| Rule | 46/50 | 1,027.42 | 0 / 50 / 0 |
| Collect then guide | 45/50 | 1,243.92 | 11 / 8 / 31 |
| Guide then collect | 44/50 | 1,114.84 | 2 / 39 / 9 |
| Collect twice | 41/50 | 1,285.96 | 9 / 8 / 33 |
| Guide twice | 46/50 | 1,027.42 | 0 / 50 / 0 |

For the 43 checkpoint-eligible seeds, successes are respectively 39, 38, 37,
34, 39; mean capped frames are 1,121.12, 1,372.86, 1,222.77, 1,421.74,
1,121.12. Mean danger frames across all 50 are 332.52, 431.48, 372.14,
457.58, 332.52; mean max spread is 70.2536, 79.1440, 71.3806, 78.7718,
70.2536. Mean outside-workspace frames are 48.44, 49.22, 46.66, 46.96,
48.44. Fixed sequences are not demonstrated improvements over rules.

Collect then guide rescues rule timeouts 3308, 3314, 3347, but loses rule
successes 3318, 3340, 3344, 3349. Guide then collect rescues none and loses
3311 and 3327. Comparing the opposite orders directly, collect-first is faster
in 14 seeds, guide-first in 28, and eight tie (including seven without a
checkpoint). There is no universally favorable fixed order in this cohort.

Both stages actually execute in 41 collect-then-guide and 17 guide-then-collect
runs; repeated collection executes both stages in nine runs. Guide-twice matches
rule outcomes in all 50 cases: it introduces no different mode priority. Collect
then guide differs from repeated collection in only eight seeds. Thus most of
the observed collection-first effect is not evidence that a rich multi-step
planner is necessary; it can result from the first collection intervention alone.

One genuine sequence comparison is seed 3302, checkpoint frame 193: rules finish
in 868 frames; collect then guide in 736; repeated collection in 1,390. Both
collection-first branches share their first 60 collection frames. Collect then
guide next executes 60 guiding frames, whereas repeated collection continues
nine further collection frames until splitting clears. Another example is 3347:
rules time out at 2,000, collect then guide finishes at 1,279, and repeated
collection at 1,916. Conversely, at 3336 repeated collection finishes at 884
versus collect then guide's 1,559 (rules finish at 763). These examples identify
trajectory-dependent alternatives, not a deployable rule or proof of LLM value.

Supplementary analysis of the saved branch CSV gives two hindsight bounds:
choosing the better of the two opposite orders after knowing both full outcomes
would yield 49/50 and mean 1,005.08 frames. Choosing among the original rule and
the two orders after knowing outcomes would yield 49/50 and mean 957.88 frames.
The latter is 69.54 frames below the rule mean. These are explicitly offline
opportunity bounds with future information, not actual online-controller results,
predictions available at the checkpoint, or LLM achievements. Branch choices are
saved in the report for audit; they must not be used as a claimed experiment win.

No guiding stage has the logged termination reason `target_arrived` in this
cohort; its stages terminate by alarm availability or the frame limit. This does
not mean no sheep ever arrive. It shows that the current alarm validity conditions
and short stage limits often end the stage before its intended completion signal.
The diagnostic is not yet a complete goal-based skill executor. It also bypasses
the API forecast guard to inspect alternative existing actions, so promising
branches are not automatically accepted by the deployed high-level interface.

Results and per-stage event records are under
`logs/anomaly_behavior/20261004_001655_059012/`: `repair_sequence_runs.csv`,
`repair_sequence_events.csv`, and `repair_sequence_report.json`. Reproduce raw
branch outcomes and the basic summary with:

```powershell
python experiments/anomaly_behavior/run_concurrent_mode_priority_trial.py --sequence-diagnostic
```

The report's supplementary order/stage and hindsight fields were calculated
from the saved CSV after the runner finished. They use per-seed comparisons and
rank choices by success first, then capped frames; they do not alter simulations.

Conclusion: existing actions offer some state-dependent room for better decisions
at the first dual-alarm state, without adding anomalies. Fixed collection-first
or guiding-first plans perform worse overall, and repeated-skill controls limit
claims about the necessity of multi-step planning. Before implementing an LLM
planner, determine whether current observable state/history can distinguish
beneficial collection interventions from harmful ones, with a held-out evaluation
and a rule selector using the same stage executor. No production settings,
controller, or prompt changed; no commit or push was performed.

### Observable-State Selector: Frozen Validation (2026-10-04)

The next diagnostic reuses the sequence executor above, but only chooses between
ordinary rules and one `collect_then_guide` intervention at the first concurrent
alarm. It does not add tasks, sheep types, repair modes, API calls, or production
parameters. The 50 previously inspected seeds 3301-3350 are exploration data, not
held-out evidence. All rule outcomes and checkpoint frames were reproduced.

Ten causal features were declared before the new runs: maximum/mean spread,
centroid distance, inside-goal count, dog-to-centroid distance, boundary margin,
and collection-versus-guiding forecast differences in progress, spread,
inside-goal count, and splitting frames. Forecasts use the existing 30-frame
model and retain the real controller's target-holding memory in copies.
They freeze alarms during prediction and are not complete two-stage forecasts.
Seed, future trajectories, and eventual outcomes are never selection inputs.

The screen considered the original rule, the existing forecast dominance test,
and single-feature thresholds with at least five eligible exploration examples
on either side. It considered 500 candidates in total, selecting by success
count first and capped completion frames second. This is exploratory fitting,
not 500 independent confirmations. The chosen condition was saved before any
validation trajectories were generated: at a concurrent alarm, collect first
only if mean spread is at most 15.066749555666721. Ordinary splitting already
requires mean spread above 15, so this is a very narrow near-threshold band.
Do not adopt that fitted numeric threshold as a justified physical parameter.

Fifty distinct seeds 4401-4450 were then run under the same 5A+5B+5C+5-normal,
standard deviation 15, 2,000-frame conditions. Their seed IDs were not found in
the existing CSV experiment records before this test. No refitting followed.
Outcomes for each policy are assembled from complete matched baseline and
collection-first trajectories using only its causal checkpoint decision.

| Cohort / policy | Success | Mean capped frames | Collection interventions |
| --- | ---: | ---: | ---: |
| Exploration: rule | 46/50 | 1027.42 | 0 |
| Exploration: forecast dominance | 45/50 | 1052.20 | 4 |
| Exploration: fitted single threshold | 48/50 | 1021.68 | 7 |
| Validation: rule | 46/50 | 1002.74 | 0 |
| Validation: forecast dominance | 46/50 | 993.16 | 8 |
| Validation: frozen single threshold | 47/50 | 979.68 | 11 |

The frozen selector reduced validation mean capped frames by 23.06 (2.30%),
with six faster pairs, four slower pairs, and 40 equal pairs. It rescued two
failures but introduced one new failure. Mean danger frames changed from 314.16
to 303.58, mean maximum spread from 68.3412 to 69.2592, and mean goal exits from
16.42 to 16.84. Median paired frame change is zero. Improvement is modest and
not uniform across metrics; this cohort does not establish a statistically
conclusive or robust scheduling improvement. The existing forecast dominance
test alone also remains unproven as a reliable long-horizon selector.

This identifies a weak observable scheduling candidate, not LLM value or proof
that a complex planner is needed. Keep it diagnostic. If pursued later, evaluate
the frozen condition on another untouched cohort, not a retuned condition on
these same validation seeds; compare any LLM against this same-information
rule selector rather than only the old fixed fallback.

Artifacts are together under `logs/anomaly_behavior/20261004_004453_893400/`:
`frozen_selector.json` (pre-validation protocol and source hashes),
`selector_states.json` (features, forecasts, and full-run paired metrics), and
`sequence_selector_report.json` (all screening scores and cohort summaries).
`branch_replay_audit.json` additionally confirms exact rule/collection outcomes
on seeds 3302, 3308, and 3336 against the original sequence CSV, covering helpful
and harmful interventions. Formal controllers, triggers, and prompts are unchanged.

```powershell
python experiments/anomaly_behavior/run_forecast_mode_trial.py --sequence-selector-diagnostic logs/anomaly_behavior/20261004_001655_059012/repair_sequence_runs.csv
```

Re-running this command repeats the specified validation seeds; it does not
create new independent evidence. Tests cover causal forecast copies, rule-first
ties, success-prioritized fitting, no-alarm fallback, and freezing before validation.

Type A currently uses normal flocking and boundary dynamics with a reduced
response to the dog. The focused tests check both its dog-only response and
its unchanged motion when the dog is out of range. The response factor is a
provisional model parameter, not an empirically fitted biological value.

For paired end-to-end rule-controller trials, run:

```powershell
python experiments\anomaly_behavior\run_type_a_trials.py --seed-count 20 --max-steps 1000
```

This uses the same seeds for normal with repair off/on (`C00`/`C00R`) and
Type A with repair off/on (`C03`/`C04`). It writes `type_a_trials.csv` and
`type_a_trials.json` under another timestamped `logs/anomaly_behavior/` folder.
Use `--dog-response-factor 0.15` to compare a candidate without editing the
production configuration.

With the guide-first rule and a 2,000-frame cap on independent seeds
1240-1319, no-repair normal succeeded on 77/80 and no-repair Type A on 72/80.
Eight seeds failed only for Type A and three failed only for normal. With
repair enabled, both scenarios succeeded on 80/80; mean capped frames were
441.3 normal versus 386.9 Type A. Type A is a modest no-repair challenge in
this cohort, but the existing repair controller already handles it. These
outcomes do not justify increasing its severity merely to produce more alarms.

To screen splitting detection in matched normal/Type B runs and count
concurrent alarms in mixed stress runs, use:

```powershell
python experiments\anomaly_behavior\run_detector_screen.py
```

Its `detector_screen.json` is saved under `logs/anomaly_behavior/<timestamp>/`.
The trajectory screen uses rule control without repair; the mixed stress case is also without
repair. Concurrent alarm frames are possible LLM trigger opportunities, not
actual API requests.

To compare alarm timing and rule-repair outcomes on the same seeds, run:

```powershell
python experiments\anomaly_behavior\run_alarm_outcome_trials.py --seed-count 20 --max-steps 1000
```

The CSV records each run; the JSON groups paired no-repair/repair outcomes by
the alarm pattern observed in the **no-repair** run: neither alarm, one alarm,
both at different times, or both in the same frame. This avoids grouping a
repair run by alarm changes caused by the repair itself. All results use the
rule controller and make no API requests. The scenario groups have different
numbers and types of abnormal sheep, so differences between those groups do
not by themselves prove that simultaneous alarms cause a performance drop.

With the current guide-first rule, a fresh 2,000-frame paired cohort on
seeds 1500-1579 measured the value of alarm-driven repair itself. Success
without versus with repair was 78/80 versus 80/80 normal, 70/80 versus
80/80 Type A, 19/80 versus 67/80 Type B, and 25/80 versus 76/80 C08 stress.
The paired rescue/regression counts were 2/0, 10/0, 49/1, and 53/2,
respectively. "No repair" still includes ordinary dog drive and collect
actions. The reported alarm categories are defined from the no-repair
trajectory, so they should not be interpreted as LLM trigger counts under
the repaired trajectory. All these outcomes use the rule controller only.

Current open questions:

1. Does Type B create sustained, repair-relevant dispersion beyond the spread
   generated by normal starting positions? Compare matched trajectories before
   changing its force parameters or the detector thresholds.
2. Is same-frame alarm overlap a useful slow-layer trigger? Compare rule-control
   outcomes for single, sequential, and overlapping alarms before changing the
   API request policy.

In a 20-seed, 1,000-step rule-only screen (seeds 0-19), the no-repair mixed
scenario had 9 stagnation-only runs, 8 sequential-alarm runs, 1 splitting-only
run, 1 concurrent-alarm run, and 1 run without alarms. The concurrent run failed,
but one observation does not establish an outcome cliff. The normal scenario
also produced 3 concurrent-alarm runs, so overlap is not specific to Type B or
mixed anomalies. These are diagnostic observations, not a validated causal
comparison or a reason to tune the model solely for more API requests.

To distinguish Type B dispersion from the wide initial placement, run:

```powershell
python experiments\anomaly_behavior\run_type_b_threshold_trials.py --seed-count 20 --max-steps 1000
```

This pairs normal and Type B runs by seed with repair disabled. It counts the
maximum-spread and mean-spread threshold crossings separately from frame 30
onward, and compares dispersion only over frames observed in both paired runs.
The 30-frame exclusion is an analysis window, not a change to the detector or
to the simulation's initial spread.

For seeds 0-19 with 1,000 maximum steps, Type B had higher maximum spread than
its normal counterpart in all 20 paired runs, but higher mean spread in only
15/20. From frame 30 onward, 18/20 Type B runs crossed only the maximum-spread
threshold at least once, while 1/20 crossed both thresholds and raised the
splitting alarm. Of 16 Type B failures without repair, 15 had no post-start
splitting alarm. Normal runs also crossed the individual thresholds and raised
post-start splitting alarms (6/20), so sheep type is not a ground-truth alarm
label. These observations suggest a mismatch between an outlier-shaped Type B
effect and the current two-threshold AND rule, but do not prove that every
Type B failure was caused by a missed splitting alarm. Keep the production
detector unchanged until alternative criteria are tested against normal runs.

To compare candidate splitting alarms on previously unused seeds 20-59:

```powershell
python experiments\anomaly_behavior\run_splitting_candidate_screen.py
```

This offline screen leaves the production detector and controller untouched.
All candidates exclude frames before 30 for a like-for-like comparison. The
alternatives require maximum spread above 40 for 1, 5, 10, or 20 consecutive
frames; the current rule additionally requires mean spread above 15. Reported
normal alerts and failed Type B alerts are descriptive, not false-positive or
true-positive rates: sheep type and task outcome are not ground-truth labels
for a splitting event.

To test whether the promising 10-frame candidate actually helps the rule
controller, run a separate paired intervention trial on seeds 60-99:

```powershell
python experiments\anomaly_behavior\run_splitting_repair_trial.py
```

This trial preserves the current alarm before frame 30, then uses maximum
spread above 40 for 10 consecutive frames. Both arms use the same repair
controller and seed. The candidate is injected only into the diagnostic runner;
the production detector remains unchanged. Compare success flips, capped
frames, and goal exits, not danger-frame totals across different alarm rules.

On candidate-screen seeds 20-59, the current rule raised post-start splitting
alarms in 15/40 normal, 10/40 Type B, and 8/40 mixed runs. The 10-frame
sustained-maximum candidate raised them in 16/40, 38/40, and 30/40 runs,
respectively. This improved detection coverage without strongly increasing
run-level normal alerts, but did not establish that the alerts were useful.

In the separate rule-repair intervention on seeds 60-99, the candidate changed
success from 33/40 to 37/40 for normal, 15/40 to 13/40 for Type B, and 17/40
to 19/40 for mixed. Mean capped frames increased for both abnormal scenarios:
Type B 835.1 to 860.1 and mixed 792.2 to 834.2. Type B had five rescued runs
but seven regressions. Therefore, do not adopt this candidate in the production
detector. In the seven regressed Type B seeds, candidate-triggered cohesion
occupied roughly 250-500 frames instead of the small amounts under the current
alarm, often targeting Type B sheep throughout. Several, but not all, runs also
spent many cohesion frames targeting sheep closer to the goal than the flock
centroid; overlong repair is a concern, not a proven single cause.

An independent startup-grace screen on seeds 2801-2840 kept the existing
thresholds and rule controller but suppressed the splitting alarm during
frames 0-29. At a 2,000-frame limit, normal success stayed 40/40, Type B
fell from 38/40 to 35/40, and mixed stress changed from 36/40 to 37/40
while mean capped frames increased from 983.6 to 1018.0. The three Type B
regressions (2807, 2829, 2838) first changed control mode at frame 0.
The same test on independent seeds 2901-2940 kept normal success at 40/40,
changed Type B from 34/40 to 35/40 (two rescues, one regression), and
mixed stress from 36/40 to 37/40. Type B mean capped frames rose from
1048.9 to 1135.3. Combined across both cohorts, Type B success was
72/80 versus 70/80 and mean capped frames approximately 1141.7 versus
1195.9; mixed stress was 72/80 versus 74/80 with nearly unchanged mean
capped frames. The results trade performance across scenarios, so do not
adopt a blanket 30-frame splitting-alarm grace period.

The current rule repair remains useful: on seeds 120-159 with a 1,000-step cap,
Type B improved from 9/40 without repair to 18/40 with repair, and mixed from
10/40 to 19/40. A one-parameter-at-a-time geometry screen on seeds 100-119
favored larger cohesion/guide standoff distances, but the Type B gains reversed
on independent seeds 120-159 (cohesion distance 20: 18/40 to 13/40; guide
distance 10: 18/40 to 14/40). Production distances remain unchanged.

To inspect Type B repair failures using the current rule controller:

```powershell
python experiments\anomaly_behavior\run_type_b_failure_trace.py --first-seed 120 --seed-count 40
```

On seeds 120-159 with a 1,000-step cap, 18/40 Type B runs succeeded and
22/40 failed. The failed runs reached a mean peak of only 5.5 sheep inside
the goal, so most failures are not solely last-sheep cleanup. Failed runs
spent more time in both repair modes, but this is partly because they ran to
the frame cap. Their initial centroid was also farther from the goal on
average (85.0 versus 54.5), so longer repair occupancy cannot be treated as
a causal explanation. In cohesion mode, the selected predictive target was
not the currently farthest sheep on 806/2,906 failed-run frames and 111/446
successful-run frames. The current drive-point formula scales its outward
direction by the maximum distance, giving a shorter standoff in these cases.
Long guide-target runs with increasing target-to-goal distance occurred in
both outcomes; these observations do not establish that guide mode caused
the failures.

The resulting unit-direction geometry candidate was tested without editing
the production controller:

```powershell
python experiments\anomaly_behavior\run_cohesion_direction_trial.py --first-seed 210 --seed-count 20
```

On seeds 210-229, Type B improved from 4/20 to 6/20, but normal fell from
17/20 to 16/20 and mixed from 9/20 to 8/20. On independent seeds 230-269,
it worsened each scenario by one success (normal 37/40 to 36/40, Type B
9/40 to 8/40, mixed 22/40 to 21/40). Do not adopt this candidate on the
current evidence; the shorter standoff is not by itself a proven defect.

## Type B behavior-intensity screen

Run the one-factor screen with paired seeds:

```powershell
python experiments\anomaly_behavior\run_type_b_intensity_screen.py --first-seed 270 --seed-count 20 --max-steps 1000
```

The baseline Type B has cohesion factor 0.15, separation-weight factor 1.5,
and separation radius 6. The screen changes one of these at a time. It records
both a 200-frame dog-free flock spread and closed-loop outcomes with the same
seed, detector, and rule controller. "No repair" still uses a moving shepherd;
it only disables alarm-driven repair. Comparisons to normal use shared
post-start frames, because successful runs end at different times. The
dog-free assay isolates flock dynamics but is not a shepherding success test.

In screening seeds 270-289, the current Type B succeeded 3/20 without repair
and 2/20 with repair. Several one-factor variants appeared to improve repair
success, including weaker cohesion (0.075: 8/20) and a shorter separation
radius (3: 8/20). This first batch was used only for candidate selection.

On independent seeds 290-329 at 1,000 frames, the current Type B succeeded
11/40 without repair and 18/40 with repair. Weaker cohesion fell to 9/40 and
16/40; stronger cohesion (0.30) fell to 9/40 and 14/40. The shorter radius
improved no-repair success to 16/40, but repair success only to 20/40, with
seven rescued and five regressed seeds. Dog-free Type B max spread remained
above normal for every tested variant (current +8.7, radius 3 +6.1 on average).

At the standard 2,000-frame cap with new seeds 330-369, the current Type B
scored 14/40 without repair and 23/40 with repair; radius 3 scored 19/40
without repair but 21/40 with repair (eight rescued, ten regressed). Normal
scored 37/40 without repair and 40/40 with repair. Thus Type B is a clear
challenge, but these tests do not establish that its chosen strength is
unrealistic or uniquely responsible for poor control. No Type B parameter
was changed in production. The opposite no-repair and repair responses to
radius 3 motivate studying controller/behavior interaction next.

## Dog approach to the guide target

The diagnostic runner can inject an experimental controller without altering
the production controller. A lateral-approach candidate makes the dog sidestep
and circle a `HARD_TO_GUIDE` target when the dog is on the goal-facing side:

```powershell
python experiments\anomaly_behavior\run_guide_approach_trial.py --first-seed 370 --seed-count 20 --max-steps 1000
```

On seeds 370-389, the 20-unit detour reduced the fraction of guide frames
with an outward dog-force component on the target from 0.65 to 0.55 for Type B,
but normal success fell from 19/20 to 16/20. Type B improved from 4/20 to
6/20 and mixed from 7/20 to 9/20; these small gains do not justify the normal
regressions. A 40-unit detour further reduced outward-force exposure but did
not improve Type B success (4/20) and reduced mixed success (6/20). Restricting
the 20-unit detour to when the dog was within its 50-unit sensing range also
failed: Type B remained 4/20, normal fell to 16/20, and mixed to 6/20.
The force sign is a diagnostic proxy, not a direct measurement of net sheep
movement. No approach candidate was adopted; the next experiment should
examine repair-target selection and switching independently of dog routing.

A later read-only audit of the current guide-first controller on Type B seeds
1920-1999 measured the selected sheep's actual one-frame goal progress during
the first 1,000 frames. When a Type B target was within the dog's 50-unit
sensing range, it moved toward the goal on 99% of 3,755 guide frames with the
dog behind it, versus 14% of 9,130 frames with the dog on the goal-facing
side. This confirms a strong geometry-dependent response, but successful runs
also contain many goal-facing dog frames. It does not establish that an
approach detour improves task success; the tested detours above remain rejected.

With the later guide-first dual-alarm rule, a near-goal-only lateral detour
also failed to show a stable gain. At an 18-of-20 sheep-in-goal gate on seeds
1090-1119, the 20-unit detour matched the baseline in all three scenarios
(normal 30/30, Type B 27/30, mixed 29/30); it rarely activated. Moving the
gate to 15-of-20 on independent seeds 1120-1159 changed Type B from 37/40 to
38/40 but mixed stress from 39/40 to 38/40, with normal 40/40 under both.
The tradeoff does not justify a production change.

## Repair-target selection and switching

The current rule controller predicts target urgency from velocity, holds a
`HARD_TO_GUIDE` target for at least 30 frames, and uses a switch margin of 3.
An initial paired comparison against legacy immediate farthest-target
selection on seeds 450-469 scored 8/20 versus 7/20 for Type B and 10/20
versus 10/20 for mixed. This is too small to justify removing the predictor.

One-factor target-selection ablations can be reproduced with:

```powershell
python experiments\anomaly_behavior\run_repair_target_selection_screen.py --first-seed 470 --seed-count 20 --max-steps 1000
```

Removing the guide target's minimum hold increased mean within-guide target
switches from 3.5 to 28.7 per Type B run on seeds 470-489, without improving
success (5/20 to 4/20). Doubling the minimum guide hold from 30 to 60 reduced
switches and initially looked promising. On independent seeds 490-529 at
1,000 frames it scored normal 38/40 versus 35/40, Type B 11/40 versus 12/40,
and mixed 19/40 versus 15/40. On new seeds 530-569 at 2,000 frames it scored
normal 40/40 versus 39/40, Type B 25/40 versus 25/40, and mixed 30/40 versus
25/40. A larger independent 2,000-frame check on seeds 570-649 reversed that
apparent benefit: normal 78/80 versus 79/80, Type B 43/80 versus 47/80, and
mixed 52/80 versus 53/80. The longer hold consistently reduced target
switching but did not reliably improve task outcome. Keep 30 frames in the
production controller; a low switch count alone is not a success metric.

A read-only audit of the current guide-first controller on Type B seeds
1920-1999 (2,000-frame outcome, first 1,000 action frames inspected) found
74/80 successes and 259 switches between `HARD_TO_GUIDE` targets. The newly
selected target was on the dog's goal-facing side in 75.7% of these switch
frames, but this happened in both successful and failed runs. Among runs with
initial centroid distance at least 100, successful runs (19) averaged 1.43
within-guide target switches per 100 guide frames; failed runs (5) averaged
1.48. The fraction of guide frames with the dog on the goal-facing side was
65.4% versus 71.0%, respectively. These are descriptive, small-sample
associations, not evidence that switching caused failure. In particular, they
do not justify increasing the target hold or changing the switch margin.

## Type B failures versus the no-repair baseline

On the same seeds 1920-1999, with four Type B sheep and a 2,000-frame cap,
the no-repair C05 case succeeded 31/80 times and the current rule-repair C06
case succeeded 74/80 times. Repair rescued 44 no-repair failures and caused
one regression (seed 1957). Thus disabling repair is not a general fix for
the remaining Type B failures. The six C06 failures were also run with normal
sheep at the same seeds: all six normal runs succeeded within 2,000 frames.
This last comparison is conditional on selecting C06 failures, so it is a
diagnostic check, not an independent estimate of a Type B effect. The failed
C06 runs had different residual states. The read-only diagnostic now records
goal membership, entries, and exits separately for normal, Type A, and Type B
sheep. In the last 500 frames, seeds 1925 and 1981 had no sheep enter the
goal; 1959 and 1979 made partial progress but still ended with 15 and 14
sheep outside, respectively. Seed 1957 ended with all 16 normal sheep inside
but only one of four Type B sheep; it had no goal exits in those 500 frames.
Seed 1993 ended with 15 normal and two Type B sheep inside, after 15 normal
and five Type B goal exits during those 500 frames. Thus the remaining
failures include no-arrival, partial-progress, and near-completion cases;
a single late-stage intervention would not address them all. Keep the current
Type B force parameters and repair policy until a specific failure mechanism
and a matched, held-out improvement are demonstrated.

The 2,000-frame failures are time-censored, not permanent failures. Continuing
those six C06 trajectories without changing the controller or physics reached
the goal at state frames 2650 (seed 1925), 2077 (1957), 2937 (1959), 2484
(1979), 3295 (1981), and 2088 (1993). The rule controller therefore reaches
80/80 on this cohort by 4,000 frames. `MAX_STEPS` is used to stop the run,
not in the rule controller or sheep dynamics, so the longer runs preserve
the first 2,000 frames. Report success together with the frame cap and
completion-time distribution; do not treat a 2,000-frame timeout as evidence
that the repair policy cannot eventually finish.

In a paired 4,000-frame check on all seeds 1920-1999, both the normal C00
baseline and the repaired Type B C06 case completed 80/80. Normal completion
time had median 402.5 frames (90th percentile 1613.6); Type B had median
948 frames (90th percentile 1968.1). Type B took longer on 61/80 matched
seeds, and the median paired increase was 313.5 frames. At the original
2,000-frame cap, C00 completed 75/80 and C06 74/80. For this cohort, binary
success at 2,000 frames barely distinguishes normal from Type B, whereas
completion time shows a substantial slowdown. This is a descriptive cohort
result, not a claim about all random starts or a demonstrated LLM benefit.
Because C00 disables repair, an additional normal-sheep control with repair
enabled was run on the same 80 seeds. It completed 80/80 within 2,000 frames
(median 388.5, 90th percentile 823.7, maximum 1410), so the Type B slowdown
is not explained merely by turning the repair system on. This comparison
still includes alarm-dependent changes in the selected control mode; it does
not isolate the Type B force coefficients from the controller's response.

On these 80 repaired Type B runs, all 16 normal sheep were inside while at
least one Type B sheep was outside in 78/80 runs. Across those states, the
controller spent 8,667 frames in `TARGET_TRACKING` and 1,084 in
`HARD_TO_GUIDE`; the last remaining outside sheep was Type B in 78/80 runs.
This identifies Type B cleanup as a time bottleneck, but does not imply that
always selecting `HARD_TO_GUIDE` would help. A one-rule in-memory screen on
new seeds 2000-2019 forced `HARD_TO_GUIDE` whenever at least 16 sheep were
inside and the ordinary rule would select `TARGET_TRACKING`. At 2,000 frames,
Type B success fell from 18/20 to 4/20 and mixed stress from 19/20 to 6/20;
mean goal exits rose from 12.6 to 76.0 and from 7.8 to 72.4, respectively.
Normal sheep with repair enabled remained 20/20, but mean capped completion
time rose from 316.1 to 362.8 frames. This candidate was rejected; the
large increase in goal exits shows why final-sheep progress cannot be
optimized independently of retaining sheep already inside the goal.

A second in-memory screen retained the rule mode and target but shortened the
`TARGET_TRACKING` collect standoff from 25 to 20 only when at least 16 sheep
were inside. On independent seeds 2020-2039 at the 2,000-frame cap, Type B
success fell from 20/20 to 18/20, with mean capped time 887.9 to 1089.9
frames and mean goal exits 12.2 to 24.8. Normal sheep with repair enabled
remained 20/20, but mean capped time rose from 401.8 to 497.0 and exits from
5.6 to 11.7. Mixed stress changed from 20/20 to 19/20. This closer standoff
was also rejected. Neither late forced guiding nor stronger late collecting
has demonstrated a safe completion-time improvement.

## Simultaneous-alarm mode priority

The former rule fallback prioritized `COHESION_PRIORITY` whenever splitting
was active, including frames with both splitting and stagnation alarms. The
counterfactual below changes only those simultaneous-alarm frames to
`HARD_TO_GUIDE`; single-alarm behavior, detection thresholds, sheep dynamics,
and repair geometry are unchanged. The runner now uses explicit controllers
for both policies, so it remains reproducible after updating the default:

```powershell
python experiments\anomaly_behavior\run_concurrent_mode_priority_trial.py --first-seed 710 --seed-count 80 --max-steps 2000
```

On seeds 670-709 with a 2,000-frame cap, cohesion-first versus guide-first
success was 28/40 versus 37/40 for Type B and 28/40 versus 40/40 for C08
stress. Normal and Type A were 40/40 under both policies. On independent
seeds 710-789, Type B was 48/80 versus 73/80, C08 stress 60/80 versus
79/80, normal 79/80 versus 80/80, and Type A 80/80 under both. The latter
cohort had 27 Type B rescues versus 2 regressions and 19 stress rescues with
no regressions. These paired results support guide-first as the new default
rule fallback for simultaneous alarms. They do not establish a benefit from
LLM decisions; the earlier DeepSeek comparison used cohesion-first and must
be repeated against this stronger rule baseline.

An additional independent cohort, seeds 930-1009 at 2,000 frames, confirmed
the main result: Type B success was 46/80 cohesion-first versus 71/80
guide-first, and C08 stress was 52/80 versus 74/80. Normal and Type A were
80/80 under both. However, guide-first increased Type B state samples with
any sheep outside the workspace from 46.09 to 51.95 per run and mean maximum
boundary excursion from 3.57 to 3.90 units. These are trajectory diagnostics,
not a claim that the simulator enforces a hard boundary.

An experimental `boundary_0` policy falls back to cohesion-first on a
simultaneous alarm when any sheep is already outside the workspace:

```powershell
python experiments\anomaly_behavior\run_concurrent_mode_priority_trial.py --first-seed 930 --seed-count 80 --max-steps 2000 --variants cohesion_first guide_first boundary_0
```

On seeds 930-1009, this reduced Type B outside-state samples from 51.95 to
35.92 per run and mean maximum excursion from 3.90 to 2.87, while both
policies succeeded on 71/80 Type B runs. The matched successes differed:
the boundary policy rescued five guide-first failures but caused five new
failures. In C08 stress, success fell from 74/80 to 72/80 and outside-state
samples rose slightly from 46.46 to 47.69. This policy is therefore kept
experimental; boundary safety remains an open controller-design issue.

An additional Type B trajectory audit on seeds 1410-1489 found that the
planned `HARD_TO_GUIDE` drive point was outside the workspace in a mean 9.4%
of guide frames for successful runs and 4.8% for failed runs. Because run
lengths and trajectories differ, these fractions do not imply that an
outside drive point helps. They also do not support blaming Type B failures
on that geometry alone. No automatic drive-point clipping was added.

A narrower boundary candidate switched to cohesion only when the sheep
selected by `HARD_TO_GUIDE` was within 5 units of a workspace edge. On
screening seeds 1600-1619, it improved Type B from 17/20 to 18/20 and reduced
mean outside-state samples from 104.4 to 50.9; C08 stress remained 20/20.
On independent seeds 1620-1699, however, Type B success fell from 74/80 to
67/80 (guide-first versus target-boundary guard), while outside-state samples
fell from 53.1 to 34.7 and mean maximum excursion from 3.67 to 2.78.
C08 stress remained 76/80, with outside-state samples 44.4 versus 36.7.
Because the Type B task regression is substantial, this guard was also
rejected. The current soft-wall model exposes a genuine success-versus-boundary
tradeoff; hard containment would be a change to the physical model and would
require re-running the full benchmark.

## Concurrent-alarm coverage for slow-layer triggering

The slow layer currently requests a decision when splitting and stagnation
coincide. The following read-only audit uses the production rule controller
with repair enabled in every scenario, including the normal scenario:

```powershell
python experiments\anomaly_behavior\run_alarm_coverage_audit.py --first-seed 1010 --seed-count 80 --max-steps 2000
```

On seeds 1010-1089, normal and Type A succeeded on 80/80 runs; Type B
succeeded on 72/80 and C08 stress on 74/80. All eight Type B failures and
all six stress failures had a concurrent alarm, and each had at least one
episode lasting ten consecutive frames. This establishes trigger coverage
for this cohort, not that an LLM decision would have fixed the failures.
Concurrent alarms also occurred in 25/80 successful normal runs and 8/80
successful Type A runs. Requiring three consecutive frames would have reduced
those counts only to 24/80 and 8/80; requiring ten would still trigger in
15/80 and 6/80. The dual-alarm predicate is therefore not a reliable proxy
for "LLM help is needed," and a simple debounce alone does not solve the
unnecessary-call problem. Do not change the default trigger based only on
this retrospective audit; evaluate any trigger policy against actual API
latency, cost, and matched control outcomes.

## Centroid versus individual stagnation

The production `stagnation` alarm measures flock-centroid progress, not an
individual sheep's progress. A read-only shadow audit applied the same
150-frame, less-than-1-unit progress test separately to sheep that remained
outside the goal:

```powershell
python experiments\anomaly_behavior\run_individual_stagnation_audit.py --first-seed 1160 --seed-count 80 --max-steps 2000
```

On seeds 1160-1239, an individual-only indication (at least one sheep stalled
while the centroid alarm was off) appeared in 59/80 normal, 52/80 Type A,
75/80 Type B, and 74/80 C08 stress runs. Normal and Type A still succeeded
on 80/80 runs; Type B succeeded on 72/80 and stress on 76/80. Per-run mean
individual-only frames were 54.9 normal, 40.2 Type A, 279.3 Type B, and
223.1 stress. The raw per-sheep test is therefore sensitive to Type B but
also common in successful normal trajectories; it is not a validated Type A
detector and should not be promoted to a production alarm or LLM trigger.

## Local forecast mode choice

The prototype already forecasts the short-term consequences of each control
mode. An experimental no-API selector chose `COHESION_PRIORITY` instead of
the guide-first fallback only when its 30-frame forecast strictly dominated
the guide forecast across the existing guard metrics, and reconsidered at
most once every 120 frames:

```powershell
python experiments\anomaly_behavior\run_forecast_mode_trial.py --first-seed 1340 --seed-count 80 --max-steps 2000
```

On screening seeds 1320-1339, Type B improved from 16/20 to 18/20, with
normal, Type A, and stress unchanged. The independent seeds 1340-1419
reversed this: Type B fell from 71/80 to 70/80 (four rescues, five
regressions) and stress from 70/80 to 68/80 (one rescue, three regressions),
while normal and Type A remained 80/80. No selector change was adopted.
Short-horizon Pareto dominance is not a demonstrated proxy for long-horizon
success. The LLM execution guard uses the same dominance test to reject
responses, so its effect on useful LLM choices needs a dedicated paired
evaluation; this local selector test alone does not quantify that effect.

## Paired 100-seed study (2026-10-08)

The frozen protocol and incremental records are under
`logs/experiments/paired_100_20261008/`. All arms use seeds 6101-6200,
the same initial conditions within each comparison, and a 2,000-frame cap.
The controller and dynamics were unchanged during the study. Live LLM
arms used uncached DeepSeek responses; fixed-sequence diagnostics did not
call the API. These are 100 distinct paired starts, not 100 API repetitions
of one initial state.

- Section 3.1: 20 sheep, 5A+5B+5C. Rule, mode-only, target-only, and
  mode+target completed 100 runs each, with 89, 88, 92, and 95 successes.
- Section 3.2: five fixed-sequence branches completed 100 results each.
  The rule baseline replay matched its recorded trajectory exactly.
  This is a sequence diagnostic, not an LLM planning experiment.
- Section 3.3: 50 sheep. At 30% heterogeneous sheep, rule and mode+target
  each completed 100 runs, with 15 and 21 successes. At 60%, 76 valid
  pairs completed; the LLM run for seed 6177 was interrupted by HTTP 402
  `Insufficient Balance`. The 76% condition has not started.

The interrupted seed is retained as raw evidence but excluded from funded
paired results and scheduled for a fresh LLM run from its initial state.
`remaining_runs.json` records the missing seeds; do not send further API
requests until the account is funded. Missing conditions are not failures
and must not be plotted as zero success. Existing records are preserved.

Charts report Wilson success intervals and 20,000-resample seed-paired
bootstrap intervals (random seed 20261008). The positive mode+target
signals are not statistically conclusive; no multiple-comparison correction
was applied. The completed data audit and 211 passing software tests are
recorded in `completed_audit.json`; software tests do not prove control
performance. Report sections 3.4-3.6 were removed at the user's request,
without deleting their historical raw records or changing production policy.

### Extension to 200 seeds (2026-10-09)

The target is now 200 cumulative runs per arm, with shared seeds 6101-6300,
unchanged dynamics, control parameters, prompts, and 2,000-frame cap. Existing
valid records are reused; missing starts are added, not rerun until favorable.
The next 100 seeds are an extension after inspecting the first 100 results,
not an independent confirmatory study or repeated API draws of one state.
Records are under `logs/experiments/paired_200_20261009/`.

Section 3.2 completed all five branches with 200 results each. Success counts
are 182 (rule), 178 (collect then guide), 183 (guide then collect), 177 (collect
twice), and 182 (guide twice). Guide then collect changes capped frames by
-4.54 on average (95% seed-paired interval [-35.34, 26.72]), with five rescues
and four regressions. No API was used, and all 200 rule replays matched the
recorded baseline trajectories exactly. These are not LLM planning results.

The account balance query still reported unavailable funds, so no inference
requests were sent in this extension. Rule baselines are expanded separately;
the report retains only genuinely completed matched API comparisons. Never
compare the new 200 rule runs directly with 100 LLM runs. Reaching 200 in all
live arms needs 724 further simulations: 100 for each of the three N20 LLM
permission arms, and 100/124/200 at N50 ratios 30%/60%/76%. The interrupted
seed 6177 must start again for LLM, not count as a valid funded failure.
`remaining_api_runs.json` records the exact pending starts. No new controller,
repair mode, or production default was introduced for this extension.

The user subsequently ended paid API expansion on 2026-10-09. The report now
uses completed200 rule baselines and fixed-sequence diagnostics as its main
results, with completed100 live comparisons separately labeled in an appendix.
Pending API runs are historical unexecuted plans, not scheduled work. Do not
resume inference automatically after funding; a new user request is required.
The final report's section 3.1 describes the N20 rule benchmark, section 3.2
the five fixed-sequence variants, and section 3.3 the three N50 rule settings.
`rule_200_statistics.json` contains their counts, means, and descriptive intervals.

### Report-only 50-seed subset

At the user's request, report section 3 now uses the first50 consecutive seeds
6101-6150 consistently across permissions, fixed sequences, and N50 ratios30/60.
This reuses existing records: zero new simulations and zero API requests.
Statistics and paired CSVs are saved under
`logs/experiments/report_subset_50_6101_6150_20261009/`. All250 initial-state
comparisons were checked, and the real-API arms have recorded returns and zero
cache hits. The fixed-sequence section remains explicitly no-API. Ratio76 lacks
real LLM data and is excluded from paired comparisons, not assigned zero success.
The50 subset is not independent validation or stronger evidence than the full
sample; full100/200 results and failed runs are preserved and summarized in the
report appendix. Do not resume inference or expand controller functionality for
this reporting operation.

### N50 missing permission arms (2026-10-13)

At the user's explicit request, `population_permission_completion.py` adds
mode-only and target-only DeepSeek arms at 30% (5A/5B/5C) and 60%
(10A/10B/10C) abnormal shares, with 50 sheep, seeds 6101-6150, and the
original 2,000-frame cap. The scope is 200 new simulations, not a restart
of the historical 724-run API expansion. Rule and mode+target records
are reused from the original N50 experiment.

The runner freezes the original configuration, checks unchanged physics
and repair sources, reconstructs original hashes after equivalent provider
interface changes, and verifies a free rule replay before paid calls.
New trajectories are checked against each original initial state. Cache is
disabled; failed or untriggered runs remain included. Fatal provider failures
stop the batch without counting the affected run as a valid funded result.
Outputs and paired statistics are under
`logs/experiments/population_permissions_50_6101_6150_20261013/`.
`--audit-only` checks compatibility without calling the API;
`--analyze-only` recomputes completed statistics without calling the API.
The supplemental arms and reused arms run on different dates, so this is
an exploratory comparison, not repeated same-state API-draw validation.
