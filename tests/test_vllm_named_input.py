import copy
import json
import unittest

from tests.test_vllm_input import sample_state
from vllm_named_input import canonical, encode_json, pack, unpack


class NamedInputTests(unittest.TestCase):
    def test_only_exact_duplicate_is_removed_and_all_values_roundtrip(self):
        state = sample_state()
        original = copy.deepcopy(state)
        payload = json.loads(encode_json(canonical(state)))
        self.assertEqual(payload["state"]["hard_to_guide_candidate"], {"target_id": 2, "geometry_from_repair_candidate": True})
        self.assertEqual(payload["state"]["cohesion_candidate"], state["cohesion_candidate"])
        self.assertEqual(payload["state"]["repair_candidates"], state["repair_candidates"])
        self.assertEqual(payload["state"]["mode_forecasts"], state["mode_forecasts"])
        self.assertEqual(canonical(unpack(payload)), canonical(original))
        self.assertEqual(state, original)

    def test_differing_geometry_signed_zero_and_unknown_fields_are_retained(self):
        for key, value in (("target_goal_distance", 1.23), ("target_goalward_speed", 0.0), ("future_risk", False)):
            state = sample_state()
            state["hard_to_guide_candidate"][key] = value
            payload = pack(state)
            self.assertIn("hard_to_guide_candidate", payload["state"])
            self.assertIn(key, payload["state"]["hard_to_guide_candidate"])
            self.assertEqual(canonical(unpack(payload)), canonical(state))

    def test_ambiguous_or_absent_candidate_is_never_referenced(self):
        for rows in ([], [sample_state()["repair_candidates"]["HARD_TO_GUIDE"][1]] * 2):
            state = sample_state()
            state["repair_candidates"]["HARD_TO_GUIDE"] = rows
            payload = pack(state)
            self.assertIn("hard_to_guide_candidate", payload["state"])
            self.assertNotIn("default_target_ids", payload["state"])
            self.assertNotIn("geometry_from_repair_candidate", payload["state"]["hard_to_guide_candidate"])
            self.assertEqual(canonical(unpack(payload)), canonical(state))

    def test_reserved_future_fields_fall_back_without_dropping_any_data(self):
        state = sample_state()
        state["default_target_ids"] = {"new_mode": 42}
        original = json.dumps(state, indent=2)
        self.assertEqual(encode_json(original), original)


if __name__ == "__main__":
    unittest.main()
