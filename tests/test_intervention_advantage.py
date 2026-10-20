import copy
import unittest

import numpy as np

from tools.evaluate_intervention_advantage import (
    FEATURE_NAMES, features, training_rows, fit, predictions, choose,
    select_model, short_forecast_choice,
)


class InterventionAdvantageTests(unittest.TestCase):
    def record(self, seed=1, split='train'):
        forecast = dict(centroid_progress=1., final_centroid_distance=20., final_max_spread=35.,
            final_mean_spread=13., inside_goal_change=0, final_inside_goal=10,
            predicted_goal_exits=0, predicted_splitting_frames=10, peak_out_of_bounds=0,
            minimum_boundary_margin=10., dog_travel=5., predicted_success=False)
        candidate = dict(target_id=3, target_status='B', target_goal_distance=40., target_spread=35.,
            target_goalward_speed=.1, dog_to_target=10., dog_to_drive_point=15.,
            estimated_travel_frames=10., dog_approach_alignment=.2, drive_point_boundary_margin=8.,
            forecast={**forecast, 'centroid_progress': 2.})
        summary = dict(rule_mode='HARD_TO_GUIDE', centroid_distance=30., num_sheep=20,
            mode_forecasts={mode:forecast.copy() for mode in ('TARGET_TRACKING','HARD_TO_GUIDE','COHESION_PRIORITY')},
            repair_candidates={'HARD_TO_GUIDE':[candidate], 'COHESION_PRIORITY':[candidate]},
            hard_to_guide_candidate=candidate, cohesion_candidate=candidate)
        def score(steps):
            return dict(success=steps < 600, elapsed_frames=steps, peak_out_of_bounds=0)
        actions = [dict(mode='HARD_TO_GUIDE', target_sheep_id=3, hold_frames=30),
                   dict(mode='COHESION_PRIORITY', target_sheep_id=3, hold_frames=30)]
        return dict(id=f'{seed}-one', seed=seed, split=split, summary=summary,
            baseline={'600':score(400)}, outcomes=[dict(action=a, scores={'600':score(steps)}) for a,steps in zip(actions,(300,500))])

    def test_negative_and_no_intervention_rows_are_present(self):
        x,y,w = training_rows([self.record()])
        self.assertEqual(x.shape, (3, len(FEATURE_NAMES)))
        self.assertEqual(y[0], 0)
        self.assertGreater(y[1], 0)
        self.assertLess(y[2], 0)
        np.testing.assert_array_equal(x[0], 0)
        self.assertAlmostEqual(w.sum(), 1.)

    def test_seed_weight_is_not_doubled_by_two_snapshots(self):
        first, second = self.record(), self.record()
        second['id'] = 'same-seed-second-state'
        _,_,weights = training_rows([first,second,self.record(2)])
        self.assertAlmostEqual(weights[:6].sum(), weights[6:].sum())

    def test_feature_extraction_never_reads_future_labels_or_arbitrary_target_numbers(self):
        record = self.record()
        action = record['outcomes'][0]['action']
        before = features(record['summary'], action)
        changed = copy.deepcopy(record)
        changed['outcomes'][0]['scores']['600']['elapsed_frames'] = -99999
        candidate = changed['summary']['repair_candidates']['HARD_TO_GUIDE'][0]
        candidate['target_id'] = 999
        changed['summary']['hard_to_guide_candidate']['target_id'] = 999
        changed_action = {**action, 'target_sheep_id':999}
        np.testing.assert_array_equal(before, features(changed['summary'],changed_action))

    def test_tuning_refuses_validation_and_seed_folds_keep_snapshots_together(self):
        with self.assertRaisesRegex(AssertionError, 'Validation'):
            select_model([self.record(split='validation')])
        records = [self.record(seed) for seed in range(1,11)]
        duplicate = self.record(1)
        duplicate['id'] = '1-two'
        _,selection = select_model(records + [duplicate])
        self.assertEqual(len(selection['seed_folds']), 10)
        self.assertEqual(len(selection['trials']), 9)

    def test_unpromising_model_can_choose_no_intervention(self):
        record = self.record()
        self.assertIsNone(choose(record, np.array([10., -20.]), 20.))
        model = fit([record], 1.)
        values = predictions(model, record)
        self.assertEqual(len(values), 2)
        self.assertTrue(np.isfinite(values).all())
        self.assertIsNotNone(short_forecast_choice(record))


if __name__ == '__main__':
    unittest.main()
