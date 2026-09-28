import unittest

import numpy as np

from control_types import ControlObservation
from run_experiments import CASES, temporary_config
from tests.anomaly_behavior.run_concurrent_mode_priority_trial import (
    CohesionFirstOnConcurrent,
    GuideFirstBoundaryGuard,
    GuideFirstOnConcurrent,
    GuideFirstTargetBoundaryGuard,
    run_trial,
)
from tools.diagnose_failures import diagnose


class ConcurrentModePriorityTrialTests(unittest.TestCase):
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
