import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from control_types import ControlObservation
from low_level_controller import LowLevelController
from high_level_controller import MockLLMBackend
from environment import SwarmEnv
from dynamics import advance_sheep
from run_experiments import temporary_config
from experiments.anomaly_behavior.behavior_characterization import (
    build_report,
    dog_response_curve,
    normal_initial_split_screen,
    type_b_dispersion_trial,
    write_characterization,
    population_scale_run,
    write_population_scale_trial,
    write_population_api_trial,
    write_repair_distance_trial,
    REPAIR_DISTANCE_VARIANTS,
    RepairFeedbackTrialController,
    write_repair_feedback_trial,
    blind_motion_payload,
    blind_numeric_diagnosis,
    write_blind_diagnosis_trial,
    BLIND_DIAGNOSIS_CHOICES,
    write_controlled_diagnosis_trial,
    mixed_observation_windows,
    write_mixed_observability_trial,
    write_permission_trial,
)


class AnomalyBehaviorCharacterizationTests(unittest.TestCase):
    def test_permission_batch_pairs_arms_and_restores_defaults_without_api(self):
        original = (config.NUM_SHEEP, config.HIGH_LEVEL_MODE_SELECTION_ENABLED,
                    config.HIGH_LEVEL_TARGET_SELECTION_ENABLED, config.LLM_CACHE_ENABLED)
        calls = []
        agents = pd.DataFrame({"Frame": [0] * 20, "Agent_Type": ["Sheep"] * 20,
                               "Agent_ID": range(20), "X": [100.] * 20, "Y": [100.] * 20,
                               "Status": ["Normal"] * 20})

        def fake_run(case, seed, max_steps, retention, controller_kind, llm_cache_enabled):
            calls.append((seed, controller_kind, config.HIGH_LEVEL_MODE_SELECTION_ENABLED,
                          config.HIGH_LEVEL_TARGET_SELECTION_ENABLED))
            self.assertEqual(config.NUM_SHEEP, 20)
            self.assertFalse(llm_cache_enabled)
            return {"Case": "custom", "Seed": seed, "A": 5, "B": 5, "C": 5,
                    "Repair": True, "Controller": controller_kind, "Success": False, "Frames": 2000,
                    "Data": "fixture-data", "Control": "fixture-control",
                    "HighLevelRequests": 0, "HighLevelOverrideFrames": 0}

        with tempfile.TemporaryDirectory() as directory, patch(
                "experiments.anomaly_behavior.behavior_characterization.run_case", side_effect=fake_run), patch(
                "experiments.anomaly_behavior.behavior_characterization.pd.read_parquet",
                side_effect=lambda path, **kwargs: agents if path == "fixture-data" else pd.DataFrame({"llm_backend_error": [None]})), patch("builtins.print"):
            output, report = write_permission_trial(directory, first_seed=70, seed_count=2)
            self.assertEqual(report["matched_initial_groups"], 2)
            self.assertEqual(len(calls), 8)
            self.assertEqual({entry[1:] for entry in calls}, {("rule", True, True), ("deepseek", True, False),
                                                            ("deepseek", False, True), ("deepseek", True, True)})
            self.assertEqual(calls[0][1], "rule")
            self.assertEqual(calls[4][1:], ("deepseek", True, False))
            for name in report["summary"]:
                data = pd.read_csv(output / name / "summary.csv")
                self.assertEqual(sorted(data.Seed), [70, 71])
                metadata = json.loads((output / name / "summary.csv.metadata.json").read_text())
                self.assertEqual(metadata["config"]["NUM_SHEEP"], 20)
                self.assertFalse(metadata["llm_cache_enabled"])
        self.assertEqual(original, (config.NUM_SHEEP, config.HIGH_LEVEL_MODE_SELECTION_ENABLED,
                                    config.HIGH_LEVEL_TARGET_SELECTION_ENABLED, config.LLM_CACHE_ENABLED))

    def test_permission_batch_rejects_invalid_seed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in ({"seed_count": 0}, {"first_seed": -1}, {"first_seed": 2**32-1, "seed_count": 2}):
                with self.assertRaises(ValueError):
                    write_permission_trial(directory, **kwargs)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_temporary_anomaly_parameters_are_restored(self):
        initial_response = config.WEIGHT_DOG_REPULSION_A_FACTOR
        with temporary_config(
            WEIGHT_DOG_REPULSION_A_FACTOR=initial_response / 2,
        ):
            self.assertEqual(
                config.WEIGHT_DOG_REPULSION_A_FACTOR,
                initial_response / 2,
            )
        self.assertEqual(config.WEIGHT_DOG_REPULSION_A_FACTOR, initial_response)

    def test_dog_response_trial_is_deterministic_and_respects_sensing_range(self):
        first = dog_response_curve((20.0, 60.0))
        second = dog_response_curve((20.0, 60.0))
        self.assertEqual(first, second)
        rows = {(row["status"], row["dog_distance"]): row for row in first}
        self.assertGreater(rows[("Normal", 20.0)]["speed"], 0.0)
        self.assertGreater(rows[("A", 20.0)]["speed"], 0.0)
        self.assertLess(rows[("A", 20.0)]["speed"], rows[("Normal", 20.0)]["speed"])
        self.assertEqual(rows[("Normal", 60.0)]["speed"], 0.0)
        self.assertEqual(rows[("A", 60.0)]["speed"], 0.0)
        self.assertAlmostEqual(
            rows[("A", 20.0)]["speed"] / rows[("Normal", 20.0)]["speed"],
            config.WEIGHT_DOG_REPULSION_A_FACTOR,
        )

    def test_type_a_follows_the_same_flock_dynamics_without_a_dog(self):
        initial = np.array([[2.0, 100.0], [8.0, 101.0], [12.0, 99.0]])
        trajectories = []
        with temporary_config(GOAL_RETENTION_ENABLED=False):
            for focus_status in ("Normal", "A"):
                positions = initial.copy()
                velocities = np.zeros_like(positions)
                statuses = np.array([focus_status, "Normal", "Normal"])
                for _ in range(30):
                    positions, velocities = advance_sheep(
                        positions,
                        velocities,
                        statuses,
                        np.array([-1000.0, -1000.0]),
                        np.array([180.0, 180.0]),
                        config,
                    )
                trajectories.append((positions, velocities))
        for normal, type_a in zip(*trajectories):
            np.testing.assert_allclose(type_a, normal)

    def test_type_b_trial_starts_from_identical_geometry(self):
        rows = type_b_dispersion_trial(frames=10, sample_frames=(0, 10))
        initial = [row for row in rows if row["frame"] == 0]
        self.assertEqual(len(initial), 2)
        for metric in ("focus_distance_to_centroid", "mean_spread", "max_spread"):
            self.assertEqual(initial[0][metric], initial[1][metric])
        self.assertEqual(rows, type_b_dispersion_trial(frames=10, sample_frames=(0, 10)))

    def test_controlled_dog_exposure_shows_unsaturated_and_saturated_responses(self):
        rows = dog_response_curve((10.0, 25.0, 40.0, 60.0), ("Normal", "A", "B", "C"))
        speeds = {(row["status"], row["dog_distance"]): row["speed"] for row in rows}
        for distance in (10.0, 25.0, 40.0, 60.0):
            self.assertEqual(speeds[("Normal", distance)], speeds[("B", distance)])
        self.assertLess(speeds[("A", 25.0)], speeds[("Normal", 25.0)])
        self.assertGreater(speeds[("C", 25.0)], speeds[("Normal", 25.0)])
        self.assertLess(speeds[("C", 25.0)], config.SHEEP_MAX_SPEED)
        self.assertAlmostEqual(speeds[("C", 10.0)], speeds[("Normal", 10.0)], places=6)
        self.assertEqual(speeds[("C", 40.0)], speeds[("Normal", 40.0)])
        self.assertTrue(all(speeds[(status, 60.0)] == 0.0 for status in ("Normal", "A", "B", "C")))

    def test_controlled_no_dog_trial_separates_b_not_a_or_c(self):
        rows = type_b_dispersion_trial(
            frames=60, sample_frames=range(61), statuses=("Normal", "A", "B", "C"),
            observation_decimals=2)
        groups = {variant: [row for row in rows if row["variant"] == variant]
                  for variant in ("all_normal", "one_type_a", "one_type_b", "one_type_c")}
        for variant in ("one_type_a", "one_type_c"):
            for normal, other in zip(groups["all_normal"], groups[variant]):
                self.assertEqual({k: v for k, v in normal.items() if k != "variant"},
                                 {k: v for k, v in other.items() if k != "variant"})
        means = {variant: np.mean([row["observed_focus_distance_to_centroid"] for row in samples[1:]])
                 for variant, samples in groups.items()}
        self.assertGreater(means["one_type_b"], means["all_normal"])
        for samples in groups.values():
            self.assertEqual(samples[0]["focus_distance_to_centroid"], 4.0)
        self.assertEqual(rows, type_b_dispersion_trial(
            frames=60, sample_frames=range(61), statuses=("Normal", "A", "B", "C"),
            observation_decimals=2))

    def test_controlled_trial_keeps_settings_and_outputs_together_without_api(self):
        original = (config.SHEEP_MAX_SPEED, config.GOAL_RETENTION_ENABLED,
                    config.NUM_SHEEP, config.SHEEP_START_RADIUS)
        with tempfile.TemporaryDirectory() as directory, patch(
                "responses_api_backend.DeepSeekBackend", side_effect=AssertionError("No API")):
            output, report = write_controlled_diagnosis_trial(directory)
            self.assertEqual(output.parent, Path(directory))
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(report["dog_configurations"], 24)
            self.assertEqual(report["flock_trajectories"], 4)
            self.assertEqual({path.name for path in output.iterdir()}, {
                "controlled_diagnosis_protocol.json", "controlled_diagnosis_report.json",
                "controlled_dog_response.csv", "controlled_no_dog_flock.csv"})
            protocol = json.loads((output / "controlled_diagnosis_protocol.json").read_text())
            self.assertFalse(protocol["production_settings_changed"])
            self.assertEqual(protocol["settings"]["GOAL_RETENTION_ENABLED"], original[1])
            observed = report["summary"]["dog_response"]["25.0"]
            self.assertLess(observed["A"]["observed_displacement"], observed["Normal"]["observed_displacement"])
            self.assertGreater(observed["C"]["observed_displacement"], observed["Normal"]["observed_displacement"])
        self.assertEqual(original, (config.SHEEP_MAX_SPEED, config.GOAL_RETENTION_ENABLED,
                                    config.NUM_SHEEP, config.SHEEP_START_RADIUS))

    def test_normal_initialization_screen_is_repeatable(self):
        first = normal_initial_split_screen(10)
        second = normal_initial_split_screen(10)
        self.assertEqual(first, second)
        self.assertEqual([row["seed"] for row in first], list(range(10)))
        self.assertTrue(all(isinstance(row["splitting_alarm"], bool) for row in first))

    def test_report_flags_are_derived_from_measured_values(self):
        report, _, _, _ = build_report(initial_seed_count=20, dispersion_frames=10)
        ratio = report["type_a"]["type_a_to_normal_response_ratio"]
        split_rate = report["normal_initialization"]["frame_zero_splitting_rate"]
        self.assertEqual(
            report["review_flags"]["type_a_response_below_5_percent_of_normal"],
            ratio < 0.05,
        )
        self.assertEqual(
            report["review_flags"]["normal_initial_split_rate_above_5_percent"],
            split_rate > 0.05,
        )

    def test_outputs_stay_inside_the_requested_report_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = write_characterization(
                output_root=directory,
                initial_seed_count=5,
                dispersion_frames=3,
            )
            self.assertEqual(output.parent, Path(directory))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "report.json",
                    "type_a_dog_response.csv",
                    "type_b_dispersion.csv",
                    "normal_initial_split.csv",
                },
            )

    def test_population_trial_preserves_settings_and_paired_geometry_without_api(self):
        original = (config.NUM_SHEEP, config.NUM_ABNORMAL_A, config.NUM_ABNORMAL_B,
                    config.NUM_ABNORMAL_C, config.MAX_STEPS)
        with tempfile.TemporaryDirectory() as directory, patch(
                "high_level_controller.build_controller", side_effect=AssertionError("No API")):
            output, report = write_population_scale_trial(
                directory, first_seed=70, seed_count=1, sheep_counts=(4, 8),
                primary_cap=2, extended_cap=3)
            self.assertEqual(output.parent, Path(directory))
            self.assertEqual(set(report["summary"]), {"4_normal", "4_mixed", "8_normal", "8_mixed"})
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(len(list(output.iterdir())), 4)
        self.assertEqual(original, (config.NUM_SHEEP, config.NUM_ABNORMAL_A,
                                   config.NUM_ABNORMAL_B, config.NUM_ABNORMAL_C, config.MAX_STEPS))

    def test_population_primary_cap_is_not_relabelled_by_later_success(self):
        original_step = SwarmEnv.step

        def arrive_at_cap(env, alarms):
            success = original_step(env, alarms)
            if env.current_frame == 2:
                env.sheep_pos[:] = env.goal_pos
            return success

        with patch.object(SwarmEnv, "step", arrive_at_cap):
            row, _ = population_scale_run(70, 4, "normal", primary_cap=2, extended_cap=3)
        self.assertFalse(row["success_primary"])
        self.assertTrue(row["success_extended"])
        self.assertEqual(row["inside_primary"], 4)
        self.assertEqual(row["capped_frames_primary"], 2)
        self.assertEqual(row["capped_frames_extended"], 3)

    def test_population_trial_rejects_invalid_protocol_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ({"sheep_counts": (20, 20)}, {"sheep_counts": (3,)},
                            {"primary_cap": 10, "extended_cap": 9}, {"seed_count": 0}):
                with self.assertRaises(ValueError):
                    write_population_scale_trial(directory, **options)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_api_ratio_trial_pairs_all_levels_and_restores_population(self):
        initial_count = config.NUM_SHEEP
        calls = []
        agents = pd.DataFrame({"Frame": [1] * 50, "Agent_Type": ["Sheep"] * 50,
                               "Agent_ID": list(range(50)), "X": [100.] * 50,
                               "Y": [100.] * 50, "Status": ["Normal"] * 50})
        control = pd.DataFrame({"llm_request_id": [None], "llm_backend_error": [None]})

        def fake_case(case, seed, max_steps, max_log_files, controller_kind, llm_cache_enabled):
            self.assertEqual(config.NUM_SHEEP, 50)
            self.assertEqual(max_steps, 2000)
            self.assertFalse(llm_cache_enabled)
            self.assertTrue(case["repair_enabled"])
            population = tuple(case[f"num_abnormal_{kind}"] for kind in "abc")
            calls.append((population, controller_kind, seed))
            return {"Case": "custom", "Seed": seed, "Controller": controller_kind,
                    "Success": True, "Frames": 100, "DangerFrames": 20,
                    "MaxDisp": 45., "FinalInsideGoal": 50, "GoalExits": 0,
                    "Data": "unused-data.parquet", "Control": "unused-control.parquet",
                    **{key: 0 for key in ("HighLevelRequests", "HighLevelResponses", "HighLevelAccepted",
                                         "HighLevelRejected", "HighLevelOverrideFrames", "HighLevelCacheHits")}}

        def read_parquet(path, columns=None):
            return agents.copy() if columns else control.copy()

        with tempfile.TemporaryDirectory() as directory, patch(
                "experiments.anomaly_behavior.behavior_characterization.run_case", side_effect=fake_case), patch(
                    "experiments.anomaly_behavior.behavior_characterization.pd.read_parquet", side_effect=read_parquet), patch("builtins.print"):
            output, report = write_population_api_trial(directory, first_seed=70, seed_count=2)
            self.assertEqual(len(calls), 12)
            for population in ((5, 5, 5), (10, 10, 10), (13, 13, 12)):
                for seed in (70, 71):
                    self.assertIn((population, "rule", seed), calls)
                    self.assertIn((population, "deepseek", seed), calls)
            self.assertEqual(set(report["completed_levels"]), {"ratio_30", "ratio_60", "ratio_76"})
            self.assertTrue((output / "population_api_protocol.json").exists())
            self.assertTrue(all(value["matched_initial_states"] == 2
                                for value in report["completed_levels"].values()))
            calls.clear()
            _, validation = write_population_api_trial(directory, first_seed=80,
                                                        seed_count=1, ratios=(30,))
            self.assertEqual(calls, [((5, 5, 5), "rule", 80), ((5, 5, 5), "deepseek", 80)])
            self.assertEqual(set(validation["completed_levels"]), {"ratio_30"})
        self.assertEqual(config.NUM_SHEEP, initial_count)

    def test_api_ratio_trial_rejects_invalid_selection_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for ratios in ((), (30, 30), (50,)):
                with self.assertRaises(ValueError):
                    write_population_api_trial(directory, ratios=ratios)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_distance_variants_change_only_one_parameter(self):
        baseline = REPAIR_DISTANCE_VARIANTS["baseline"]
        self.assertEqual(baseline, (3.0, 10.0))
        for name, values in REPAIR_DISTANCE_VARIANTS.items():
            if name != "baseline":
                self.assertEqual(sum(value != ref for value, ref in zip(values, baseline)), 1)

    def test_distance_changes_reach_production_drive_points(self):
        observation = ControlObservation(np.array([[60., 100.], [100., 100.]]),
                                         np.array([30., 100.]), np.array([180., 100.]), ())
        for hard, cohesion in REPAIR_DISTANCE_VARIANTS.values():
            with temporary_config(HARD_TO_GUIDE_DISTANCE=hard, COHESION_REPAIR_DISTANCE=cohesion):
                for mode, expected in (("HARD_TO_GUIDE", hard), ("COHESION_PRIORITY", cohesion)):
                    decision = LowLevelController().decide(observation, mode_override=mode)
                    target = observation.sheep_positions[decision.target_sheep_id]
                    self.assertAlmostEqual(float(np.linalg.norm(decision.drive_point - target)), expected, places=5)

    def test_distance_trial_pairs_identities_and_restores_config_without_api(self):
        keys = ("NUM_SHEEP", "NUM_ABNORMAL_A", "NUM_ABNORMAL_B", "NUM_ABNORMAL_C",
                "MAX_STEPS", "CONTROLLER_KIND", "HARD_TO_GUIDE_DISTANCE",
                "COHESION_REPAIR_DISTANCE", "SPACE_SIZE", "SHEEP_START_RADIUS", "GOAL_RADIUS")
        original = {key: getattr(config, key) for key in keys}
        with tempfile.TemporaryDirectory() as directory, patch(
                "high_level_controller.build_controller", side_effect=AssertionError("No API")), patch("builtins.print"):
            output, report = write_repair_distance_trial(directory, first_seed=70, seed_count=1,
                                                         sheep_counts=(20, 50), max_steps=2)
            self.assertEqual(report["matched_initial_groups"], 2)
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(len(report["summary"]), 10)
            runs = pd.read_csv(output / "repair_distance_runs.csv")
            for _, group in runs.groupby("num_sheep"):
                self.assertEqual(group["initial_sha256"].nunique(), 1)
                self.assertEqual(group["initial_status_sha256"].nunique(), 1)
                self.assertTrue((group[["A", "B", "C"]] == 5).all().all())
                self.assertTrue((group["capped_frames_primary"] == 2).all())
            self.assertEqual(len(list(output.iterdir())), 5)
        self.assertEqual(original, {key: getattr(config, key) for key in keys})

    def test_distance_trial_restores_config_after_failure(self):
        keys = ("HARD_TO_GUIDE_DISTANCE", "COHESION_REPAIR_DISTANCE", "SHEEP_START_RADIUS")
        with temporary_config(HARD_TO_GUIDE_DISTANCE=99., COHESION_REPAIR_DISTANCE=88., SHEEP_START_RADIUS=17.):
            original = {key: getattr(config, key) for key in keys}
            with tempfile.TemporaryDirectory() as directory, patch(
                    "experiments.anomaly_behavior.behavior_characterization.population_scale_run", side_effect=RuntimeError("stop")):
                with self.assertRaises(RuntimeError):
                    write_repair_distance_trial(directory, seed_count=1)
            self.assertEqual(original, {key: getattr(config, key) for key in keys})

    def test_distance_trial_rejects_invalid_options_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ({"variants": ()}, {"variants": ("guide_near", "baseline")},
                            {"variants": ("baseline", "unknown")}, {"variants": ("baseline", "baseline")},
                            {"sheep_counts": (14,)}, {"sheep_counts": (20, 20)},
                            {"max_steps": 0}, {"seed_count": 0}):
                with self.assertRaises(ValueError):
                    write_repair_distance_trial(directory, **options)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def feedback_controller(self, request_budget=6):
        settings = SimpleNamespace(**{key: value for key, value in vars(config).items() if key.isupper()})
        settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = False
        settings.HIGH_LEVEL_FORECAST_FRAMES = 2
        settings.HIGH_LEVEL_MIN_HOLD_FRAMES = settings.HIGH_LEVEL_MAX_HOLD_FRAMES = 60
        backend = MockLLMBackend(latency_frames=10000, hold_frames=60)
        backend.source = "feedback_rule"
        backend.trigger_on_concurrent_only = False
        return RepairFeedbackTrialController(backend, settings, request_budget=request_budget)

    def feedback_observation(self, frame, shift=0., alarms=("flock_splitting", "stagnation")):
        return ControlObservation(np.array([[40.+shift, 100.], [80.+shift, 100.], [120.+shift, 100.]]),
                                  np.array([10., 100.]), np.array([180., 100.]), alarms, frame=frame,
                                  sheep_statuses=("A", "B", "C"))

    def test_feedback_waits_for_actual_full_window_and_records_targets(self):
        controller = self.feedback_controller()
        for frame in range(60):
            controller.decide(self.feedback_observation(frame))
        self.assertEqual(controller.request_counter, 0)
        controller.decide(self.feedback_observation(60))
        self.assertEqual(controller.request_counter, 1)
        history = controller.pending_request.state_summary["repair_history"]
        self.assertEqual(history[-1]["elapsed_frames"], 60)
        self.assertEqual(history[-1]["centroid_progress"], 0.)
        self.assertTrue(history[-1]["review_eligible"])
        self.assertEqual(history[-1]["target_ids"], [0])
        self.assertEqual(controller.backend.mode_mapping["flock_splitting"], "COHESION_PRIORITY")

    def test_feedback_does_not_review_a_progressing_window(self):
        controller = self.feedback_controller()
        for frame in range(61):
            controller.decide(self.feedback_observation(frame, shift=2. if frame == 60 else 0.))
        self.assertEqual(controller.request_counter, 0)
        self.assertFalse(controller.repair_history[-1]["review_eligible"])
        self.assertEqual(controller.repair_history[-1]["centroid_progress"], 2.)

    def test_feedback_windows_do_not_mix_modes_or_trigger_on_short_segments(self):
        controller = self.feedback_controller()
        for frame in range(61):
            alarms = ("flock_splitting", "stagnation") if frame < 30 else ("flock_splitting",)
            controller.decide(self.feedback_observation(frame, alarms=alarms))
        self.assertEqual(controller.request_counter, 0)
        self.assertEqual(controller.repair_history[0]["mode"], "HARD_TO_GUIDE")
        self.assertEqual(controller.repair_history[0]["elapsed_frames"], 30)
        self.assertFalse(controller.repair_history[0]["review_eligible"])
        for frame in range(61, 91):
            controller.decide(self.feedback_observation(frame, alarms=("flock_splitting",)))
        self.assertEqual(controller.request_counter, 1)
        self.assertEqual(controller.repair_history[-1]["mode"], "COHESION_PRIORITY")
        self.assertEqual(controller.repair_history[-1]["elapsed_frames"], 60)
        self.assertEqual(controller.backend.mode_mapping["flock_splitting"], "COHESION_PRIORITY")

    def test_feedback_budget_and_history_are_bounded(self):
        controller = self.feedback_controller(request_budget=1)
        for frame in range(241):
            controller.decide(self.feedback_observation(frame))
            if controller.pending_request is not None:
                controller._clear_pending()
        self.assertEqual(controller.request_counter, 1)
        self.assertEqual(len(controller.repair_history), 3)

    def test_feedback_trial_restores_defaults_and_uses_no_network_in_test(self):
        keys = ("NUM_SHEEP", "NUM_ABNORMAL_C", "HARD_TO_GUIDE_DISTANCE", "HIGH_LEVEL_TARGET_SELECTION_ENABLED")
        original = {key: getattr(config, key) for key in keys}
        calls = []

        def fake_backend(settings):
            self.assertFalse(settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED)
            self.assertFalse(settings.LLM_CACHE_ENABLED)
            self.assertEqual(settings.HIGH_LEVEL_MIN_HOLD_FRAMES, 60)
            calls.append(settings)
            backend = MockLLMBackend(latency_frames=10000, hold_frames=60)
            backend.instructions, backend.prompt_version = "test", "test"
            return backend

        with tempfile.TemporaryDirectory() as directory, patch(
                "responses_api_backend.DeepSeekBackend", side_effect=fake_backend), patch("builtins.print"):
            output, report = write_repair_feedback_trial(directory, first_seed=70, seed_count=1, max_steps=2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["matched_initial_groups"], 1)
            self.assertEqual(set(report["summary"]), {"rule", "feedback_rule", "feedback_llm"})
            self.assertTrue(all(value["requests"] == 0 for value in report["summary"].values()))
            self.assertTrue((output / "repair_feedback_protocol.json").exists())
        self.assertEqual(original, {key: getattr(config, key) for key in keys})

    def blind_observation_data(self):
        rows = []
        for frame in range(66):
            for agent_id in range(5):
                rows.append({"Frame": frame, "Agent_Type": "Sheep", "Agent_ID": agent_id,
                             "X": 50.+agent_id*4.+.1*frame, "Y": 100.+agent_id,
                             "Status": "PRIVATE_CANARY", "forecast": "PRIVATE_CANARY"})
            rows.append({"Frame": frame, "Agent_Type": "Dog", "Agent_ID": 0,
                         "X": 10.+.08*frame, "Y": 100.,
                         "Status": "PRIVATE_CANARY", "forecast": "PRIVATE_CANARY"})
        return pd.DataFrame(rows)

    def test_blind_payload_is_invariant_to_private_labels_and_predictions(self):
        data = self.blind_observation_data()
        first = blind_motion_payload(data, 0, 2, [180., 180.])
        data["Status"], data["forecast"] = "A", "future-secret"
        data["true_coefficient"] = 99.
        second = blind_motion_payload(data, 0, 2, [180., 180.])
        self.assertEqual(first, second)
        encoded = json.dumps(first, allow_nan=False)
        for private in ("PRIVATE_CANARY", "future-secret", "Status", "forecast", "true_coefficient", "Agent_ID"):
            self.assertNotIn(private, encoded)
        self.assertEqual([sample["relative_frame"] for sample in first["samples"]], list(range(0, 61, 5)))
        self.assertAlmostEqual(first["samples"][0]["focus_velocity_estimate"][0], .1)

    def test_blind_payload_rejects_missing_observations(self):
        data = self.blind_observation_data()
        data = data.loc[~(data.Frame.eq(20) & data.Agent_Type.eq("Dog"))]
        with self.assertRaises(ValueError):
            blind_motion_payload(data, 0, 2, [180., 180.])

    def test_mixed_windows_match_original_features_without_private_labels(self):
        data = self.blind_observation_data()
        extra = data.loc[data.Frame >= 5].copy()
        extra["Frame"] += 61
        data = pd.concat([data, extra], ignore_index=True)
        first = mixed_observation_windows(data, first_frame=2)
        self.assertEqual(len(first), 10)
        self.assertEqual(sorted({row["start_frame"] for row in first}), [2, 63])
        for row in first:
            payload = blind_motion_payload(data, row["focus_id"], row["start_frame"], [180., 180.])
            self.assertEqual({key: row[key] for key in payload["features"]}, payload["features"])
            self.assertEqual(row["diagnosis"], blind_numeric_diagnosis(payload))
        data["Status"], data["forecast"], data["true_coefficient"] = "A", "private", 999
        self.assertEqual(first, mixed_observation_windows(data, first_frame=2))
        self.assertNotIn("PRIVATE_CANARY", json.dumps(first))

    def test_mixed_windows_reject_missing_geometry(self):
        data = self.blind_observation_data()
        with self.assertRaises(ValueError):
            mixed_observation_windows(data.loc[~(data.Frame.eq(20) & data.Agent_Type.eq("Dog"))], first_frame=2)
        with self.assertRaises(ValueError):
            mixed_observation_windows(data, first_frame=1)

    def test_mixed_vectorized_features_match_on_varied_close_exposures(self):
        rng = np.random.default_rng(70)
        positions = np.array([[100.+i*2., 100.] for i in range(8)])[None, :, :] + np.cumsum(
            rng.uniform(-.6, .6, size=(127, 8, 2)), axis=0)
        rows = []
        for frame in range(127):
            for agent_id in range(8):
                rows.append({"Frame": frame, "Agent_Type": "Sheep", "Agent_ID": agent_id,
                             "X": round(positions[frame, agent_id, 0], 2),
                             "Y": round(positions[frame, agent_id, 1], 2)})
            rows.append({"Frame": frame, "Agent_Type": "Dog", "Agent_ID": 0,
                         "X": 80.+10.*np.sin(frame/10.), "Y": 100.})
        data = pd.DataFrame(rows)
        windows = mixed_observation_windows(data, first_frame=2)
        self.assertTrue(any(row["comparable_response_samples"] > 0 for row in windows))
        for row in windows:
            expected = blind_motion_payload(data, row["focus_id"], row["start_frame"], [180., 180.])["features"]
            self.assertEqual({key: row[key] for key in expected}, expected)

    def test_mixed_report_uses_first_answer_not_hindsight_and_never_calls_api(self):
        data = self.blind_observation_data()
        types = {0: "Normal", 1: "A", 2: "B", 3: "C", 4: "Normal"}
        sheep = data.Agent_Type.eq("Sheep")
        data.loc[sheep, "Status"] = data.loc[sheep, "Agent_ID"].map(types)
        goal = np.random.RandomState(70).uniform(30., 170., 2)
        centers = data.loc[sheep].groupby("Frame")[["X", "Y"]].mean()
        distances = dict(zip(centers.index, np.linalg.norm(centers.to_numpy() - goal, axis=1)))
        data["dist_to_goal"] = data.Frame.map(distances)
        windows = [{"focus_id": 1, "start_frame": 150, "end_frame": 210,
                    "diagnosis": "normal_response", "moderate_unsaturated_frames": 6,
                    "comparable_response_samples": 6},
                   {"focus_id": 1, "start_frame": 211, "end_frame": 271,
                    "diagnosis": "weak_dog_response", "moderate_unsaturated_frames": 6,
                    "comparable_response_samples": 6}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ratio_30/rule").mkdir(parents=True)
            (root / "population_api_protocol.json").write_text(json.dumps({
                "base_config": {"GOAL_MARGIN": 30., "SPACE_SIZE": 200.}}))
            source = root / "source.parquet"
            source.write_bytes(b"private fixture")
            pd.DataFrame([{"Seed": 70, "Data": str(source)}]).to_csv(root / "ratio_30/rule/summary.csv", index=False)
            with patch("experiments.anomaly_behavior.behavior_characterization.pd.read_parquet", return_value=data), patch(
                    "experiments.anomaly_behavior.behavior_characterization.mixed_observation_windows", return_value=windows), patch(
                    "responses_api_backend.DeepSeekBackend", side_effect=AssertionError("No API")), patch("builtins.print"):
                output, report = write_mixed_observability_trial(root / "output", root)
            scores = pd.read_csv(output / "mixed_observability_private_scores.csv")
            first = scores.loc[scores.focus_id.eq(1)].iloc[0]
            self.assertEqual(first.diagnosis, "normal_response")
            self.assertFalse(first.correct)
            self.assertTrue(first.later_diagnosis_disagrees)
            self.assertEqual(report["summary"]["all"]["identified"], 1)
            self.assertEqual(report["summary"]["all"]["correct"], 0)
            self.assertTrue(report["source_hashes_unchanged"])
            self.assertEqual(report["api_calls"], 0)

    def test_blind_numeric_rule_can_abstain_and_identify_each_pattern(self):
        features = {"minimum_dog_exposure_distance": 20., "spread_change": 1., "median_outward_speed": 0.,
                    "median_response_ratio_to_unlabeled_peers": 1., "comparable_response_samples": 10,
                    "dog_exposure_distance_span": 10., "fraction_near_observed_peer_speed_ceiling": .2}
        self.assertEqual(blind_numeric_diagnosis({"features": features}), "normal_response")
        for ratio, expected in ((.3, "weak_dog_response"), (2., "enhanced_near_dog_response")):
            self.assertEqual(blind_numeric_diagnosis({"features": {**features, "median_response_ratio_to_unlabeled_peers": ratio}}), expected)
        self.assertEqual(blind_numeric_diagnosis({"features": {**features, "minimum_dog_exposure_distance": 50.,
                                                              "spread_change": 6., "median_outward_speed": .1}}), "dispersing")
        for overrides in ({"comparable_response_samples": 0}, {"fraction_near_observed_peer_speed_ceiling": .9},
                          {"median_response_ratio_to_unlabeled_peers": None}):
            self.assertEqual(blind_numeric_diagnosis({"features": {**features, **overrides}}), "insufficient_evidence")

    def test_blind_trial_sends_only_whitelisted_payload_and_scores_abstention(self):
        payload = blind_motion_payload(self.blind_observation_data(), 0, 2, [180., 180.])
        cases = [{"clip_id": f"clip_{i+1:02d}", "seed": 4701+i//4, "focus_id": i,
                  "true_status": "PRIVATE_CANARY", "truth": BLIND_DIAGNOSIS_CHOICES[i % 4],
                  "source": "PRIVATE_CANARY", "source_sha256": "PRIVATE_CANARY", "payload": payload}
                 for i in range(20)]
        calls = []

        def create(**options):
            calls.append(options)
            self.assertNotIn("PRIVATE_CANARY", options["input"])
            self.assertEqual(set(json.loads(options["input"])), {"clip_id", "observation", "goal_position", "samples", "features"})
            self.assertFalse(options["store"])
            return SimpleNamespace(output_text=json.dumps({"diagnosis": "insufficient_evidence",
                                                          "evidence_frames": [], "reason": "Motion is ambiguous"}))

        def fake_backend(settings):
            self.assertFalse(settings.LLM_CACHE_ENABLED)
            return SimpleNamespace(model="mock", base_url="mock", reasoning_effort=None,
                                   client=SimpleNamespace(responses=SimpleNamespace(create=create)),
                                   _extract_output_text=lambda response: response.output_text, close=lambda: None)

        with tempfile.TemporaryDirectory() as directory, patch(
                "experiments.anomaly_behavior.behavior_characterization.prepare_blind_diagnosis_cases", return_value=cases), patch(
                "responses_api_backend.DeepSeekBackend", side_effect=fake_backend), patch("builtins.print"):
            output, report = write_blind_diagnosis_trial(directory, source_root=directory)
            self.assertEqual(len(calls), 20)
            self.assertEqual(report["summary"]["llm"]["abstained"], 20)
            self.assertEqual(report["summary"]["llm"]["coverage"], 0.)
            self.assertIsNone(report["summary"]["llm"]["selective_accuracy"])
            self.assertEqual(report["summary"]["llm"]["invalid"], 0)
            self.assertTrue((output / "blind_diagnosis_truth_private.json").exists())


if __name__ == "__main__":
    unittest.main()
