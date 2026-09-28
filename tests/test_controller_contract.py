import unittest

import numpy as np

import config
from control_types import ControlDecision, ControlObservation
from environment import SwarmEnv
from high_level_controller import HighLevelController, MockLLMBackend
from low_level_controller import LowLevelController, rule_mode_for_alarms
from run_experiments import temporary_config


class FixedController:
    def __init__(self, navigation_point):
        self.navigation_point = np.array(navigation_point, dtype=float)
        self.observations = []

    def decide(self, observation):
        self.observations.append(observation)
        return ControlDecision(
            mode="TARGET_TRACKING",
            action="test_action",
            target_sheep_id=None,
            drive_point=self.navigation_point,
            navigation_point=self.navigation_point,
        )


class ControllerContractTests(unittest.TestCase):
    def test_fast_and_slow_layers_share_the_rule_fallback(self):
        for repair_enabled in (False, True):
            with temporary_config(REPAIR_ENABLED=repair_enabled):
                low = LowLevelController()
                high = HighLevelController(
                    low_level_controller=low,
                    backend=MockLLMBackend(),
                )
                for alarms in (
                    (),
                    ("flock_splitting",),
                    ("stagnation",),
                    ("flock_splitting", "stagnation"),
                ):
                    with self.subTest(repair_enabled=repair_enabled, alarms=alarms):
                        expected = rule_mode_for_alarms(
                            alarms, repair_enabled
                        )
                        self.assertEqual(low._select_mode(alarms), expected)
                        self.assertEqual(high._rule_mode(alarms), expected)

    def test_rule_controller_maps_existing_alarms(self):
        sheep = np.array([[20., 20.], [30., 30.]])
        dog = np.array([10., 10.])
        goal = np.array([180., 180.])
        controller = LowLevelController()
        with temporary_config(REPAIR_ENABLED=True):
            for alarms, mode in [([], "TARGET_TRACKING"),
                                 (["stagnation"], "HARD_TO_GUIDE"),
                                 (["flock_splitting"], "COHESION_PRIORITY"),
                                 (["stagnation", "flock_splitting"], "HARD_TO_GUIDE")]:
                with self.subTest(alarms=alarms):
                    decision = controller.decide(ControlObservation(sheep, dog, goal, alarms))
                    self.assertEqual(decision.mode, mode)

    def test_environment_accepts_an_injected_controller(self):
        controller = FixedController([150., 100.])
        env = SwarmEnv(controller=controller)
        env.goal_pos = np.array([180., 180.])
        env.sheep_pos = np.array([[20., 20.], [30., 30.]])
        env.sheep_vel = np.zeros((2, 2))
        env.sheep_status = np.array(["Normal", "Normal"], dtype=object)
        env.dog_pos = np.array([100., 100.])
        env.dog_vel = np.array([0.5, -0.25])
        before = env.dog_pos.copy()
        self.assertFalse(env.step(["stagnation"]))
        self.assertEqual(len(controller.observations), 1)
        observation = controller.observations[0]
        self.assertEqual(tuple(observation.alarms), ("stagnation",))
        self.assertEqual(observation.frame, 0)
        self.assertFalse(observation.sheep_positions.flags.writeable)
        self.assertFalse(observation.dog_position.flags.writeable)
        self.assertFalse(observation.goal_position.flags.writeable)
        self.assertFalse(observation.sheep_velocities.flags.writeable)
        self.assertFalse(observation.dog_velocity.flags.writeable)
        np.testing.assert_array_equal(observation.goal_position, env.goal_pos)
        np.testing.assert_array_equal(observation.sheep_velocities, [[0., 0.], [0., 0.]])
        np.testing.assert_array_equal(observation.dog_velocity, [0.5, -0.25])
        self.assertEqual(env.control_action, "test_action")
        self.assertEqual(env.control_data[-1]["decision_source"], "rule")
        self.assertGreater(env.dog_pos[0], before[0])
        self.assertAlmostEqual(np.linalg.norm(env.dog_pos - before), config.DOG_MAX_SPEED)

    def test_cohesion_target_uses_projected_outward_spread(self):
        controller = LowLevelController()
        observation = ControlObservation(
            sheep_positions=np.array([[30., 50.], [50., 50.], [69., 50.]]),
            dog_position=np.array([50., 80.]),
            goal_position=np.array([180., 180.]),
            alarms=("flock_splitting",),
            sheep_velocities=np.array([[0.7, 0.], [0., 0.], [0.7, 0.]]),
        )
        decision = controller.decide(observation)
        self.assertEqual(decision.target_sheep_id, 2)

    def test_hard_to_guide_target_uses_projected_goal_distance(self):
        controller = LowLevelController()
        observation = ControlObservation(
            sheep_positions=np.array([[20., 100.], [30., 100.]]),
            dog_position=np.array([10., 100.]),
            goal_position=np.array([180., 100.]),
            alarms=("stagnation",),
            sheep_velocities=np.array([[0.7, 0.], [-0.7, 0.]]),
        )
        decision = controller.decide(observation)
        self.assertEqual(decision.target_sheep_id, 1)

    def test_repair_target_is_held_before_switching(self):
        controller = LowLevelController()
        first = ControlObservation(
            sheep_positions=np.array([[20., 100.], [30., 100.]]),
            dog_position=np.array([10., 100.]),
            goal_position=np.array([180., 100.]),
            alarms=("stagnation",),
            sheep_velocities=np.zeros((2, 2)),
        )
        changed = ControlObservation(
            sheep_positions=np.array([[100., 100.], [10., 100.]]),
            dog_position=np.array([10., 100.]),
            goal_position=np.array([180., 100.]),
            alarms=("stagnation",),
            sheep_velocities=np.zeros((2, 2)),
        )
        with temporary_config(HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES=2):
            self.assertEqual(controller.decide(first).target_sheep_id, 0)
            self.assertEqual(controller.decide(changed).target_sheep_id, 0)
            self.assertEqual(controller.decide(changed).target_sheep_id, 1)

    def test_repair_selection_prefers_sheep_outside_goal(self):
        controller = LowLevelController()
        observation = ControlObservation(
            sheep_positions=np.array([[100., 100.], [125., 100.], [170., 100.]]),
            dog_position=np.array([50., 100.]),
            goal_position=np.array([100., 100.]),
            alarms=("flock_splitting",),
            sheep_velocities=np.zeros((3, 2)),
        )
        decision = controller.decide(observation)
        self.assertEqual(decision.target_sheep_id, 2)

    def test_disabled_repair_selection_preserves_original_rule(self):
        controller = LowLevelController()
        observation = ControlObservation(
            sheep_positions=np.array([[20., 100.], [30., 100.]]),
            dog_position=np.array([10., 100.]),
            goal_position=np.array([180., 100.]),
            alarms=("stagnation",),
            sheep_velocities=np.array([[0.7, 0.], [-0.7, 0.]]),
        )
        with temporary_config(REPAIR_TARGET_SELECTION_ENABLED=False):
            decision = controller.decide(observation)
        self.assertEqual(decision.target_sheep_id, 0)


if __name__ == "__main__":
    unittest.main()
