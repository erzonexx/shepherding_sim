import unittest
from unittest.mock import patch

import numpy as np

from control_types import ControlObservation
from environment import SwarmEnv
from low_level_controller import LowLevelController
from run_experiments import CASES, temporary_config
from experiments.anomaly_behavior.run_concurrent_mode_priority_trial import (
    CohesionFirstOnConcurrent,
    GuideFirstBoundaryGuard,
    GuideFirstOnConcurrent,
    GuideFirstTargetBoundaryGuard,
    run_trial,
    RepairSequenceTrial,
    clone_rule_state,
    state_fingerprint,
    continue_rule_branch,
    summarize_sequences,
)
from run_experiments import make_detector
from tools.diagnose_failures import diagnose


class ConcurrentModePriorityTrialTests(unittest.TestCase):
    def sequence_observation(self, frame=0, alarms=("flock_splitting", "stagnation")):
        return ControlObservation(
            sheep_positions=np.array([[40., 100.], [80., 100.], [120., 100.]]),
            dog_position=np.array([65., 110.]), goal_position=np.array([180., 100.]),
            sheep_velocities=np.zeros((3, 2)), frame=frame, alarms=alarms,
        )

    def test_sequence_orders_have_same_budget_and_then_return_to_rules(self):
        for sequence in (("COHESION_PRIORITY", "HARD_TO_GUIDE"),
                         ("HARD_TO_GUIDE", "COHESION_PRIORITY")):
            trial = RepairSequenceTrial(LowLevelController(), sequence, 2)
            for frame in range(5):
                decision = trial.decide(self.sequence_observation(frame))
                expected = sequence[frame // 2] if frame < 4 else "HARD_TO_GUIDE"
                self.assertEqual(decision.mode, expected)
            self.assertEqual([event["executed_frames"] for event in trial.events], [2, 2])
            self.assertEqual([event["reason"] for event in trial.events],
                             ["stage_timeout", "stage_timeout"])

    def test_sequence_finishes_collection_when_splitting_clears(self):
        trial = RepairSequenceTrial(LowLevelController(),
                                    ("COHESION_PRIORITY", "HARD_TO_GUIDE"), 60)
        trial.decide(self.sequence_observation())
        decision = trial.decide(self.sequence_observation(1, ("stagnation",)))
        self.assertEqual(decision.mode, "HARD_TO_GUIDE")
        self.assertEqual(trial.events[0]["reason"], "splitting_cleared")

    def test_sequence_releases_guide_stage_when_target_arrives(self):
        trial = RepairSequenceTrial(LowLevelController(),
                                    ("HARD_TO_GUIDE", "COHESION_PRIORITY"), 60)
        decision = trial.decide(self.sequence_observation())
        observation = self.sequence_observation(1)
        observation.sheep_positions[decision.target_sheep_id] = observation.goal_position
        decision = trial.decide(observation)
        self.assertEqual(decision.mode, "COHESION_PRIORITY")
        self.assertEqual(trial.events[0]["reason"], "target_arrived")

    def test_sequence_skips_inadmissible_stages_and_never_changes_targets_directly(self):
        trial = RepairSequenceTrial(LowLevelController(),
                                    ("COHESION_PRIORITY", "HARD_TO_GUIDE"), 60)
        decision = trial.decide(self.sequence_observation(0, ()))
        self.assertEqual(decision.mode, "TARGET_TRACKING")
        self.assertEqual(len(trial.events), 2)
        self.assertTrue(all(event["executed_frames"] == 0 for event in trial.events))
        with self.assertRaises(ValueError):
            RepairSequenceTrial(LowLevelController(), ("NEW_MODE",), 60)

    def test_full_precision_forks_preserve_detector_and_target_memory_without_aliases(self):
        with temporary_config(USE_FIXED_SEED=True, RANDOM_SEED=3301,
                              NUM_ABNORMAL_A=5, NUM_ABNORMAL_B=5, NUM_ABNORMAL_C=5):
            env = SwarmEnv()
            detector = make_detector()
            for _ in range(5):
                alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
                env.record_frame_metrics(metrics)
                env.step(alarms)
            original = state_fingerprint(env, detector)
            branch, branch_detector = clone_rule_state(env, detector)
            self.assertEqual(state_fingerprint(branch, branch_detector), original)
            self.assertIsNot(env.controller, branch.controller)
            self.assertIsNot(env.sheep_vel, branch.sheep_vel)
            self.assertIsNot(env.control_data, branch.control_data)
            with patch("high_level_controller.build_controller", side_effect=AssertionError("No API")):
                outcome = continue_rule_branch(branch, branch_detector, 12)
            self.assertEqual(outcome["capped_frames"], 12)
            self.assertEqual(state_fingerprint(env, detector), original)
            self.assertEqual(len(env.metrics_data), 5)
            second, second_detector = clone_rule_state(env, detector)
            other = continue_rule_branch(second, second_detector, 12)
            self.assertEqual(outcome, other)
            np.testing.assert_array_equal(branch.sheep_pos, second.sheep_pos)

    def test_sequence_summary_includes_untriggered_seeds_without_selection_bias(self):
        rows = []
        for seed, eligible in ((1, True), (2, False)):
            for name, frames in (("rule", 100), ("collect_then_guide", 80)):
                rows.append(dict(seed=seed, eligible=eligible, variant=name, success=True,
                                 capped_frames=frames if eligible else 100, danger_frames=5,
                                 max_spread=50, goal_exits=0, outside_frames=0))
        summary = summarize_sequences(rows)
        self.assertEqual(summary["all"]["collect_then_guide"]["mean_frame_delta"], -10)
        self.assertEqual(summary["first_dual_alarm"]["collect_then_guide"]["mean_frame_delta"], -20)
        self.assertEqual(summary["all"]["rule"]["runs"], 2)

    def test_target_boundary_guard_ignores_unrelated_outside_sheep(self):
        controller = GuideFirstTargetBoundaryGuard(0.0)
        observation = ControlObservation(
            sheep_positions=np.array([
                [-1., 100.], [100., 100.], [150., 100.]
            ]),
            dog_position=np.array([90., 100.]),
            goal_position=np.array([0., 100.]),
            alarms=("flock_splitting", "stagnation"),
            sheep_velocities=np.zeros((3, 2)),
        )
        self.assertEqual(controller.decide(observation).mode, "HARD_TO_GUIDE")

        observation = ControlObservation(
            sheep_positions=np.array([
                [50., 100.], [100., 100.], [201., 100.]
            ]),
            dog_position=np.array([90., 100.]),
            goal_position=np.array([0., 100.]),
            alarms=("flock_splitting", "stagnation"),
            sheep_velocities=np.zeros((3, 2)),
        )
        self.assertEqual(controller.decide(observation).mode, "COHESION_PRIORITY")

    def test_boundary_guard_changes_only_near_edge_dual_alarm(self):
        controller = GuideFirstBoundaryGuard(10.0)
        def observation(positions, alarms):
            return ControlObservation(
                sheep_positions=np.array(positions, dtype=float),
                dog_position=np.array([50., 50.]),
                goal_position=np.array([180., 100.]),
                alarms=alarms,
            )
        both = ("flock_splitting", "stagnation")
        self.assertEqual(
            controller.decide(observation([[5., 100.], [80., 100.]], both)).mode,
            "COHESION_PRIORITY",
        )
        self.assertEqual(
            controller.decide(observation([[60., 100.], [80., 100.]], both)).mode,
            "HARD_TO_GUIDE",
        )
        self.assertEqual(
            controller.decide(observation(
                [[5., 100.], [80., 100.]], ("stagnation",)
            )).mode,
            "HARD_TO_GUIDE",
        )

    def test_only_concurrent_alarm_changes_rule_mode(self):
        controller = GuideFirstOnConcurrent()
        legacy = CohesionFirstOnConcurrent()
        with temporary_config(REPAIR_ENABLED=True):
            self.assertEqual(
                controller._select_mode(("flock_splitting", "stagnation")),
                "HARD_TO_GUIDE",
            )
            self.assertEqual(
                legacy._select_mode(("flock_splitting", "stagnation")),
                "COHESION_PRIORITY",
            )
            self.assertEqual(
                controller._select_mode(("flock_splitting",)),
                "COHESION_PRIORITY",
            )
            self.assertEqual(
                controller._select_mode(("stagnation",)),
                "HARD_TO_GUIDE",
            )
        with temporary_config(REPAIR_ENABLED=False):
            self.assertEqual(
                controller._select_mode(("flock_splitting", "stagnation")),
                "TARGET_TRACKING",
            )

    def test_small_trial_has_paired_variants(self):
        rows, summary = run_trial(
            first_seed=650, seed_count=1, max_steps=40
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(summary["type_b"]["cohesion_first"]["runs"], 1)
        self.assertEqual(summary["stress"]["guide_first"]["runs"], 1)

    def test_production_rule_matches_explicit_guide_first_candidate(self):
        case = next(case for case in CASES if case["case"] == "C08")
        production, _, _ = diagnose(case, 670, 300)
        candidate, _, _ = diagnose(
            case, 670, 300, controller_factory=GuideFirstOnConcurrent
        )
        self.assertEqual(
            production["trajectory_sha256"],
            candidate["trajectory_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
