import copy
import json
import unittest

from vllm_input import encode_json, pack, unpack, VERSION, FORECAST_COLUMNS


def sample_state():
    geometry = dict(target_id=17, target_status="C", target_goal_distance=3.574,
                    target_spread=12.0, target_goalward_speed=-0.0, target_outward_speed=-0.7,
                    dog_to_target=8.125, dog_to_drive_point=11.25, estimated_travel_frames=7.5,
                    dog_approach_alignment=-0.125, drive_point_boundary_margin=-1.75)
    forecast = dict(horizon_frames=45, centroid_progress=-0.25, final_centroid_distance=12.875,
                    max_spread_change=0.2, final_max_spread=20.0, mean_spread_change=-0.15,
                    final_mean_spread=7.0, inside_goal_change=-1, final_inside_goal=11,
                    predicted_goal_exits=1, predicted_splitting_frames=8, peak_out_of_bounds=2,
                    minimum_boundary_margin=-4.75, dog_travel=2.5, predicted_success=False)
    second = {**geometry, "target_id": 2, "target_goal_distance": 9.0}
    return dict(alarms=["flock_splitting", "stagnation"], rule_mode="HARD_TO_GUIDE",
                num_sheep=13, type_a_count=2, type_b_count=4, type_c_count=1,
                max_spread_threshold=33.0, mean_spread_threshold=11.0, dog_speed=1.25,
                type_c_response=dict(sensitivity_radius=19.0, max_dog_repulsion_factor=2.0),
                type_geometry={"C": dict(count=1, mean_goal_distance=6.0, max_goal_distance=7.25,
                                         mean_spread=4.25, mean_goalward_speed=-0.5, mean_outward_speed=0.125)},
                cohesion_candidate=geometry, hard_to_guide_candidate=second,
                mode_forecasts={"TARGET_TRACKING": forecast, "COHESION_PRIORITY": {**forecast, "predicted_success": True}},
                repair_candidates={"HARD_TO_GUIDE": [{**geometry, "forecast": forecast},
                                                     {**second, "forecast": {**forecast, "horizon_frames": 60}}],
                                   "COHESION_PRIORITY": []})


class VLLMInputTests(unittest.TestCase):
    def test_roundtrip_preserves_signs_types_dynamic_parameters_and_candidate_order(self):
        state = sample_state()
        original = copy.deepcopy(state)
        encoded = json.loads(encode_json(json.dumps(state)))
        restored = unpack(encoded)
        self.assertEqual(encoded["input_format"], VERSION)
        forecast = encoded["state"]["mode_forecasts"]["TARGET_TRACKING"]
        self.assertEqual(forecast["horizon_frames"], 45)
        self.assertEqual(forecast["v"][0], state["mode_forecasts"]["TARGET_TRACKING"]["centroid_progress"])
        self.assertEqual(json.dumps(restored, sort_keys=True), json.dumps(state, sort_keys=True))
        self.assertEqual(state, original)
        candidates = restored["repair_candidates"]["HARD_TO_GUIDE"]
        self.assertEqual([r["target_id"] for r in candidates], [17, 2])
        self.assertEqual(candidates[0]["forecast"]["horizon_frames"], 45)
        self.assertEqual(candidates[1]["forecast"]["horizon_frames"], 60)
        self.assertIs(candidates[0]["forecast"]["predicted_success"], False)
        self.assertIn('-0.0', json.dumps(restored))
        self.assertEqual(restored["num_sheep"], 13)

    def test_schema_change_falls_back_to_complete_original_json(self):
        for section in ("cohesion_candidate", "type_geometry", "mode_forecasts", "repair_candidates"):
            state = sample_state()
            if section == "cohesion_candidate":
                state[section]["future_risk"] = 9.125
            elif section == "type_geometry":
                state[section]["C"]["future_risk"] = 9.125
            elif section == "mode_forecasts":
                del state[section]["TARGET_TRACKING"]["predicted_success"]
            else:
                state[section]["HARD_TO_GUIDE"][0]["future_risk"] = 9.125
            text = json.dumps(state, indent=2)
            self.assertEqual(encode_json(text), text)

    def test_unknown_global_fields_are_preserved_and_empty_groups_supported(self):
        state = sample_state()
        state["new_context"] = {"values": [1, False, None, -0.25]}
        state["type_geometry"] = {}
        state["mode_forecasts"] = {}
        state["repair_candidates"] = {}
        self.assertEqual(unpack(pack(state)), state)

    def test_truncated_forecast_is_not_silently_decoded(self):
        payload = pack(sample_state())
        payload["state"]["mode_forecasts"]["TARGET_TRACKING"]["v"] = [0] * (len(FORECAST_COLUMNS) - 1)
        with self.assertRaises(ValueError):
            unpack(payload)


if __name__ == "__main__":
    unittest.main()
