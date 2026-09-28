# Experiment Notes

These are local simulation results, not universal performance claims.
Paths below are relative to the project root; generated logs are ignored by Git.

- In a fresh 80-seed, 2,000-frame paired rule experiment, enabling
  alarm-driven repair raised Type B success from 19/80 to 67/80 and C08
  stress from 25/80 to 76/80; normal went from 78/80 to 80/80 and Type A
  from 70/80 to 80/80. The shepherd still drives and collects when repair is
  disabled. This measures rule-repair value, not an LLM effect.
- In the current rule controller, simultaneous splitting and stagnation alarms
  prioritize `HARD_TO_GUIDE`. With the previous cohesion-first rule as the
  paired baseline, independent 2,000-frame cohorts improved Type B from
  28/40 to 37/40, 48/80 to 73/80, and 46/80 to 71/80; C08 stress improved
  from 28/40 to 40/40, 60/80 to 79/80, and 52/80 to 74/80. Normal and Type A
  did not regress in these cohorts. This is a stronger rule-only baseline, not
  evidence of an LLM benefit.
- In a separate 80-seed, 2,000-frame Type A comparison, no-repair success was
  72/80 versus 77/80 for normal sheep; enabling repair brought both to 80/80.
  Type A is a modest challenge, but not the current main source of failure.
- On matched seeds 1920-1999, repaired Type B and normal sheep with repair
  enabled both completed 80/80 by 4,000 frames. Their median completion times
  were 948 and 388.5 frames, respectively. At the 2,000-frame cap, the counts
  were 74/80 and 80/80. The Type B deficit is primarily completion speed in
  this cohort, so report capped success and completion time together.

The results below used the previous cohesion-first dual-alarm rule and are
historical baselines, not measurements of the current rule fallback.

- With the current Type A behavior and a 1,000-step cap, paired rule experiments
  on seeds 120-159 improved Type B success from 9/40 without repair to 18/40
  with repair, and mixed success from 10/40 to 19/40. The repair modes help,
  but many runs still fail.
- A sustained maximum-spread alarm detected more Type B outliers but reduced
  Type B repair success from 15/40 to 13/40 on seeds 60-99. It was not adopted.
  Distance-only repair changes that looked promising on seeds 100-119 also
  regressed Type B on independent seeds 120-159; defaults remain unchanged.
- With the current Type A behavior, prompt v7, and the former cohesion-first
  dual-alarm rule, C08 paired seeds 170-209 (2,000-step cap) succeeded in 33/40 rule
  runs and 35/40 DeepSeek runs. Mean capped frames were 1040.1 vs 979.1;
  danger frames were 387.6 vs 336.2. DeepSeek rescued two baseline failures
  without success-to-failure flips in this batch, but 18/40 DeepSeek runs made
  no API request. Of 98 requests, 95 returned, 56 were accepted, and 39 were
  rejected; mean observed response latency was about 0.96 seconds. This paired
  success difference is too small to establish a general advantage.
- An experimental three-frame debounce with one request per concurrent episode
  reduced requests from 58 to 41 on C08 seeds 160-169 under the earlier v6
  prompt, but both policies succeeded on 5/10 seeds. This is a cost-control
  lead, not evidence of improved task success.

With the current guide-first rule and v9 DeepSeek prompt, a fresh matched C08
comparison on seeds 1700-1719 (3 Type A + 3 Type B, 2,000-frame cap, at most
one request per concurrent-alarm episode) succeeded in 18/20 runs for both
controllers. Mean capped frames were 1069.1 for rule control and 1074.2 for
DeepSeek; mean danger frames were 336.1 and 333.6, respectively. Only two
seeds changed these outcome metrics; neither changed task success. DeepSeek
made 39 requests, of which 37 returned, 19 were accepted and 18 rejected.
Accepted decisions were 17 `HARD_TO_GUIDE` and two `COHESION_PRIORITY`;
the former matches the rule fallback at selection. A control-log comparison
found only 30 frames across two seeds where the applied DeepSeek mode differed
from the rule mode at the same frame. New batch summaries record this as
`HighLevelOverrideFrames`, separately from `HighLevelActiveFrames`. The 37
returned responses averaged
about 1.01 seconds of API latency and used 66,460 input and 1,699 output
tokens in total. This cohort does not show a task-success or completion-time
benefit from DeepSeek over the current rule controller. Run summaries are in
`logs/experiments/20260928_110026_755546/summary.csv` (rule) and
`logs/experiments/20260928_110047_829514/summary.csv` (DeepSeek).
The two non-rule overrides were 15 frames each on seeds 1702 and 1708;
completion took 7 and 94 more frames than the matched rule runs. Accepted
responses that merely repeat the rule mode should not be counted as control
improvements.

Another fresh matched C08 batch on seeds 2101-2120, before the forecast-reversal
guard below, succeeded in 19/20 rule runs and 18/20 DeepSeek runs at the
2,000-frame cap. Mean capped frames were 1095.2 and 1133.7. DeepSeek made 53
uncached requests: 50 returned, 36 were accepted, and 14 were rejected. It
actually overrode the rule mode for 98 frames across four seeds. There was
one success-to-timeout regression (seed 2112) and no rescue. The request-time
30-frame forecast on that seed favored `COHESION_PRIORITY` over the rule's
`HARD_TO_GUIDE` in centroid progress and maximum spread; by the time the
response arrived 22 frames later, a fresh forecast favored the rule on both.
The former Pareto guard still accepted the response because the selected mode
predicted one fewer goal exit. The controller now rejects a non-rule response
when that two-metric preference has reversed at execution time. Offline
replay of all 20 saved responses and arrival frames exactly reproduced the
original outcomes; with the new guard, only seed 2112 changed and completed
at frame 1308. This replay is a regression check, not an independent claim
that DeepSeek improves performance. Summaries are in
`logs/experiments/paired_2101_2120_rule/summary.csv` and
`logs/experiments/paired_2101_2120_deepseek/summary.csv`.

A prospective 40-seed C08 comparison with that guard enabled (seeds
2201-2240, 2,000-frame cap, uncached API calls, one request per concurrent
alarm episode) succeeded in 37/40 rule runs and 36/40 DeepSeek runs: no
rescues and one new timeout. Mean capped frames were 985.9 and 996.0; the
paired mean difference was +10.1 frames, with 30 unchanged, three faster,
and seven slower runs. A descriptive paired bootstrap 95% interval for that
mean difference was about -74 to +91 frames. DeepSeek made 80 requests;
76 returned, 51 were accepted, 25 rejected, and the applied mode differed
from the same-state rule for 259 frames. Two responses were rejected by the
new forecast-reversal check. Mean observed response latency was about 0.90
seconds; there was one response parsing error. This independent cohort does
not establish an LLM advantage. The paired files are
`logs/experiments/paired_2201_2240_rule/summary.csv`,
`logs/experiments/paired_2201_2240_deepseek/summary.csv`, and
`logs/experiments/paired_2201_2240_comparison.json`.

To test whether mode choice at the first concurrent alarm has room to matter,
`tools/evaluate_mode_opportunities.py` ran a separate API-free, paired C08
diagnostic on seeds 2301-2340. Each seed used the rule controller or one
fixed mode for 30 frames beginning at its first dual alarm, then returned to
the rule. Success was 39/40 for the rule, 38/40 for `TARGET_TRACKING`,
35/40 for `COHESION_PRIORITY`, and 39/40 for `HARD_TO_GUIDE`. Mean capped
frames were 1022.9, 1184.0, 1141.7, and 1004.4, respectively. The one
rule timeout was rescued by all three interventions, but the fixed-mode arms
also introduced 2, 5, and 1 new timeouts, respectively. The intervention
can continue after an alarm clears and has no API latency, so this is not a
deployable policy or an estimate of LLM performance. In particular, picking
the best arm separately for each seed would be a hindsight oracle. Run with
`python tools/evaluate_mode_opportunities.py --start-seed 2301 --runs 40`;
results are in `logs/experiments/first_dual_mode_opportunities_2301_2340.csv`.

An independent 80-seed timing diagnostic (C08 seeds 2401-2480) compared
the rule with 30 frames of immediate `COHESION_PRIORITY`, 30 frames of
`COHESION_PRIORITY` after a 30-frame delay if both alarms still persisted,
and the same delayed intervention held for only 15 frames. Success was
75/80, 67/80, 76/80, and 72/80, respectively; mean capped frames were
975.5, 1071.2, 1002.8, and 975.5. The delayed 30-frame arm rescued one
rule timeout without creating a new timeout, but slowed mean completion
by 27.4 frames, and only 25/80 runs actually changed their rule mode.
The 15-frame arm rescued one run but created four new timeouts. These
API-free interventions do not establish a superior trigger or hold time;
the delayed API trigger remains experimental. Run with
`python tools/evaluate_mode_opportunities.py --timing-study --start-seed 2401 --runs 80`;
results are in `logs/experiments/first_dual_timing_2401_2480.csv`.

A subsequent matched API comparison on new C08 seeds 2501-2540 used the
rule controller, uncached DeepSeek with immediate dual-alarm requests, and
uncached DeepSeek requiring 30 consecutive dual-alarm frames. All API arms
made at most one request per concurrent episode. Success was 38/40, 40/40,
and 38/40, and mean capped frames were 906.6, 887.7, and 906.6. Immediate
DeepSeek rescued seeds 2505 and 2518 with no new timeout in this cohort,
but the delayed arm had zero true override frames and exactly matched the
rule outcomes. Requests were 63 vs 26 and input tokens 109,512 vs 43,104
for immediate vs delayed. The 2505 rescue came from holding the rule's
`HARD_TO_GUIDE` mode for 14 frames after the stagnation alarm cleared; the
2518 rescue came from 15 frames of actual `COHESION_PRIORITY` override.
The rule and immediate-LLM recorded trajectories first diverged on frames
248 and 1418, respectively, just after those interventions began. These
two mechanisms must be ablated separately. In an API-free case-level replay,
forcing only `HARD_TO_GUIDE` on seed 2505 from frame 246 for 15 frames
reproduced its 1436-frame success, and forcing only `COHESION_PRIORITY` on
seed 2518 from frame 1417 for 15 frames reproduced its 1637-frame success.
This attributes the two observed rescues to the short control interventions,
not to other API interactions, but does not show that an online rule could
identify those intervention times. This cohort alone does not
establish a reliable LLM advantage, especially given the earlier negative
40-seed comparison. Summaries are in
`logs/experiments/trigger_2501_2540_rule/summary.csv`,
`logs/experiments/trigger_2501_2540_deepseek_immediate/summary.csv`, and
`logs/experiments/trigger_2501_2540_deepseek_delayed30/summary.csv`.

An audit of the two independent 40-seed immediate-DeepSeek cohorts above
(2201-2240 and 2501-2540) compared control logs with paired agent
trajectories. Across 80 seeds, the rule and LLM each succeeded 75 and 76
times, respectively: two rule failures were rescued and one rule success
regressed. In 61 runs the LLM never actually changed the applied rule mode
or the trajectory. Of the 19 runs with a real override, five changed mode
only after an initially rule-matching LLM choice was held across an alarm
transition, 13 switched mode immediately on activation, and one had both
types of intervention. In every divergent run, the recorded trajectory
first changed on the frame after the first actual mode override. One of
those 19 runs changed its path but not its completion time. These are
descriptive intervention categories, not evidence that LLM reasoning is
better than a deterministic timing or mode-selection policy.

Earlier development evidence used a different Type A implementation: predictive
repair targeting changed 57/80 rule successes to 61/80, and an older 10-seed
C08 DeepSeek comparison changed 6/10 rule successes to 8/10. Those numbers
should not be combined with the current-model results above.
