import unittest

import numpy as np

import config
from tools.diagnose_failures import diagnose, observe, summarize, segment_distances, motion_diagnostics
from environment import SwarmEnv
from run_experiments import CASES, temporary_config


class DiagnosticTests(unittest.TestCase):
    def test_finite_segment_distances(self):
        points = np.array([[-1., 0.], [1., 3.], [5., 0.]])
        np.testing.assert_allclose(segment_distances(points, np.array([0., 0.]), np.array([2., 0.])), [1., 3., 3.])
        np.testing.assert_allclose(segment_distances(points, np.array([0., 0.]), np.array([0., 0.])), [1., np.sqrt(10), 5.])

    def test_radial_dog_response_sign_and_type_a_scaling(self):
        env = self.make_env()
        env.goal_pos = np.array([100., 100.])
        env.sheep_pos = np.array([[110., 100.], [110., 100.]])
        env.sheep_status = np.array(["Normal", "A"], dtype=object)
        env.dog_pos = np.array([100., 100.])
        env.drive_point = env.dog_pos.copy()
        env.target_sheep_id = 0
        before = env.sheep_pos.copy()
        row = motion_diagnostics(env, before, env.sheep_vel, env.dog_pos.copy())
        self.assertGreater(row["target_outward_dog_response"], 0)
        env.target_sheep_id = 1
        type_a = motion_diagnostics(env, before, env.sheep_vel, env.dog_pos.copy())
        self.assertAlmostEqual(type_a["target_outward_dog_response"] / row["target_outward_dog_response"],
                               config.WEIGHT_DOG_REPULSION_A_FACTOR)
        env.dog_pos = np.array([120., 100.])
        inward = motion_diagnostics(env, before, env.sheep_vel, env.dog_pos.copy())
        self.assertLess(inward["target_outward_dog_response"], 0)
        np.testing.assert_array_equal(before, env.sheep_pos)

    def test_target_progress_uses_positions_before_and_after_step(self):
        env = self.make_env()
        env.target_sheep_id = 0
        before = env.sheep_pos.copy()
        env.sheep_pos[0] = [42.0, 100.0]

        row = motion_diagnostics(env, before, env.sheep_vel, env.dog_pos.copy())

        self.assertAlmostEqual(row["target_goal_distance_before"], 140.0)
        self.assertAlmostEqual(row["target_goal_distance_after"], 138.0)
        self.assertAlmostEqual(row["target_goal_progress"], 2.0)

    def make_env(self):
        env = SwarmEnv()
        env.goal_pos = np.array([180.0, 100.0])
        env.sheep_pos = np.array([[40.0, 100.0], [60.0, 100.0]])
        env.sheep_vel = np.zeros((2, 2))
        env.sheep_status = np.array(["Normal", "Normal"], dtype=object)
        return env

    def test_observation_detects_excursion_and_individual_goal_membership(self):
        env = self.make_env()
        env.goal_pos = np.array([100.0, 100.0])
        env.sheep_pos = np.array([[-0.25, 100.0], [100.0, 100.0]])
        env.sheep_status = np.array(["B", "Normal"], dtype=object)
        before = env.sheep_pos.copy()
        row = observe(env, 7)
        self.assertEqual(row["outside_count"], 1)
        self.assertEqual(row["outside_ids"], "0")
        self.assertEqual(row["max_excursion"], 0.25)
        self.assertEqual(row["inside_goal"], 1)
        self.assertEqual(row["inside_goal_normal"], 1)
        self.assertEqual(row["inside_goal_type_a"], 0)
        self.assertEqual(row["inside_goal_type_b"], 0)
        np.testing.assert_array_equal(before, env.sheep_pos)

    def test_status_goal_events_match_membership_changes(self):
        _, states, actions = diagnose(CASES[6], 70, 200)
        for before, action, after in zip(states, actions, states[1:]):
            for status in ("normal", "type_a", "type_b"):
                self.assertEqual(
                    after[f"inside_goal_{status}"] - before[f"inside_goal_{status}"],
                    action[f"goal_entries_{status}"] - action[f"goal_exits_{status}"],
                )
            self.assertEqual(
                sum(action[f"goal_entries_{status}"] for status in ("normal", "type_a", "type_b")),
                action["goal_entries"],
            )
            self.assertEqual(
                sum(action[f"goal_exits_{status}"] for status in ("normal", "type_a", "type_b")),
                action["goal_exits"],
            )

    def test_mode_target_and_action_logging(self):
        cases = [([], "TARGET_TRACKING", "drive", None),
                 (["stagnation"], "HARD_TO_GUIDE", "guide_individual", 0),
                 (["flock_splitting"], "COHESION_PRIORITY", "collect", 0),
                 (["stagnation", "flock_splitting"], "HARD_TO_GUIDE", "guide_individual", 0)]
        with temporary_config(REPAIR_ENABLED=True):
            for alarms, mode, action, target in cases:
                with self.subTest(alarms=alarms):
                    env = self.make_env()
                    self.assertFalse(env.step(alarms))
                    self.assertEqual(env.current_mode, mode)
                    self.assertEqual(env.control_action, action)
                    self.assertEqual(env.target_sheep_id, target)
                    self.assertEqual(env.control_data[-1]["target_sheep_id"], target)

    def test_default_collect_and_repair_disabled(self):
        with temporary_config(REPAIR_ENABLED=False):
            env = self.make_env()
            env.sheep_pos[0, 0] = 0.0
            env.step(["stagnation"])
            self.assertEqual(env.current_mode, "TARGET_TRACKING")
            self.assertEqual(env.control_action, "collect")
            self.assertEqual(env.target_sheep_id, 0)

    def test_terminal_call_is_not_logged_as_action(self):
        env = self.make_env()
        env.sheep_pos[:] = env.goal_pos
        before = env.sheep_pos.copy()
        self.assertTrue(env.step([]))
        self.assertEqual(env.control_data, [])
        np.testing.assert_array_equal(before, env.sheep_pos)

    def test_centroid_in_goal_is_not_success(self):
        env = self.make_env()
        env.goal_pos = env.sheep_pos.mean(axis=0)
        env.sheep_pos[0, 0] -= config.GOAL_RADIUS
        env.sheep_pos[1, 0] += config.GOAL_RADIUS
        self.assertFalse(env.step([]))

    def test_switch_count_separates_target_changes_and_action_transitions(self):
        env = self.make_env()
        states = [observe(env, 0)]
        actions = [{"mode": mode, "action": action, "target_id": target, "goal_exits": exits}
                   for mode, action, target, exits in [
                       ("T", "drive", None, 0), ("T", "collect", 0, 1),
                       ("T", "collect", 1, 0), ("C", "collect", 0, 0)]]
        result = summarize(states, actions, False, 4, "hash")
        self.assertEqual(result["mode_switches"], 1)
        self.assertEqual(result["target_switches"], 2)
        self.assertEqual(result["same_mode_action_target_switches"], 1)
        self.assertEqual(result["individual_goal_exits"], 1)
        self.assertEqual(result["decision_source_counts"], {"unknown": 4})
        self.assertEqual(result["high_level_request_count"], 0)

    def test_determinism_config_restore_and_frame_alignment(self):
        original = config.RANDOM_SEED
        first, states, actions = diagnose(CASES[0], 70, 200)
        second, _, _ = diagnose(CASES[0], 70, 200)
        self.assertEqual(first, second)
        self.assertEqual(config.RANDOM_SEED, original)
        self.assertTrue(first["success_reported_by_env"])
        self.assertEqual(len(states), len(actions) + 1)
        self.assertEqual(first["step_calls"], len(actions) + 1)
        self.assertEqual(states[-1]["inside_goal"], config.NUM_SHEEP)
        for index, action in enumerate(actions):
            self.assertEqual(action["from_state_frame"], index)
            self.assertEqual(action["to_state_frame"], index + 1)


if __name__ == "__main__":
    unittest.main()
