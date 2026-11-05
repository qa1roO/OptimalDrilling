import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import drilling_advisory as core
from evaluate_independent import (TIME_COLUMNS, apply_selected_actions, combine_directions,
                                  evaluate_selected_actions, partition_wells, summarize_wells)


class ConstantRotation:
    def predict(self, frame):
        return np.full(len(frame), 90.0)


class AdverseEvaluator:
    def predict(self, frame):
        return .02 - (frame.pressure_axis.to_numpy()-10000)*1e-6


def context():
    values = {c: 0.0 for c in core.NUMERIC_FEATURES}
    values.update(pressure_axis=10000.0, pressure_rotation=9000.0,
                  pressure_axis_lag1=9500.0, pressure_rotation_lag1=8800.0,
                  rotation=95.0, speed=.02, dt=5.0, hardness_score_smooth=.5,
                  rock_energy_type_final="soft_low_energy")
    return core.add_control_features(pd.DataFrame([values]))


class IndependentEvaluationTests(unittest.TestCase):
    def test_training_roles_are_disjoint_and_cover_only_training_wells(self):
        train, test = [str(i) for i in range(21)], ["100", "101"]
        roles = partition_wells(train, test)
        self.assertTrue(set(roles["A"]).isdisjoint(roles["B"]))
        self.assertEqual(set(roles["A"]) | set(roles["B"]), set(train))
        self.assertTrue((set(roles["A"]) | set(roles["B"])).isdisjoint(test))
        self.assertEqual(roles, partition_wells(train, test))
        with self.assertRaises(ValueError):
            partition_wells(train, ["1"])

    def test_evaluator_can_reject_selected_action_without_reselecting_it(self):
        observed = context()
        selected = pd.DataFrame({"pressure_axis": [10800.0], "pressure_rotation": [9000.0]})
        before = selected.copy(deep=True)
        baseline, recommended = evaluate_selected_actions(observed, selected, ConstantRotation(), AdverseEvaluator())
        self.assertLess(recommended[0], baseline[0])
        assert_frame_equal(selected, before)
        transformed = apply_selected_actions(observed, selected)
        self.assertEqual(transformed.pressure_axis_diff1.iloc[0], 10800-9500)
        self.assertEqual(transformed.hardness_score_smooth.iloc[0], observed.hardness_score_smooth.iloc[0])

    def test_no_change_is_zero_for_independent_evaluator(self):
        observed = context()
        baseline, recommended = evaluate_selected_actions(observed, observed, ConstantRotation(), AdverseEvaluator())
        np.testing.assert_array_equal(baseline, recommended)

    def test_bootstrap_allows_negative_effect_and_negative_confidence_interval(self):
        wells = pd.DataFrame({"evaluator_current_time_sec": [100.0, 200.0, 300.0],
                              "evaluator_recommended_time_sec": [110.0, 220.0, 330.0],
                              "selector_current_time_sec": [100.0, 200.0, 300.0],
                              "selector_recommended_time_sec": [90.0, 180.0, 270.0]})
        summary = summarize_wells(wells, "test", bootstrap_reps=200)
        self.assertAlmostEqual(summary["evaluator_time_saved_pct"], -10)
        self.assertAlmostEqual(summary["selector_self_time_saved_pct"], 10)
        self.assertFalse(summary["positive_conditional_ci"])
        self.assertLess(summary["time_saved_ci975"], 0)
        self.assertEqual(summary["negative_wells_below_zero"], 3)

    def test_two_directions_are_one_cluster_per_well(self):
        rows = []
        for direction, offset in [("A_to_B", 0), ("B_to_A", 20)]:
            for well in ["one", "two"]:
                rows.append({"direction": direction, "well_id": well,
                             **{c: 100.0+offset for c in TIME_COLUMNS}})
        result = combine_directions(pd.DataFrame(rows))
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[TIME_COLUMNS].to_numpy(), np.full((2, 4), 110.0))
        with self.assertRaises(ValueError):
            combine_directions(pd.DataFrame(rows[:-1]))


if __name__ == "__main__":
    unittest.main()
