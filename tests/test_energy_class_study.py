import unittest

import numpy as np
import pandas as pd

from study_energy_classes import (Scheme, SIGNAL, VARIANTS, specification, choose_anchors,
                                 summarize_prediction)


class EnergyClassStudyTests(unittest.TestCase):
    def test_quantile_categories_are_ordered_and_balanced(self):
        frame = pd.DataFrame({SIGNAL: np.arange(240, dtype=float)})
        for k in range(2, 9):
            scheme = Scheme.fit(frame, k)
            result = scheme.transform(frame)
            self.assertEqual(result.rock_energy_type_final.nunique(), k)
            self.assertLessEqual(result.rock_energy_type_final.value_counts().max()-result.rock_energy_type_final.value_counts().min(), 1)
            self.assertTrue(result.hardness_score_smooth.is_monotonic_increasing)

    def test_validation_distribution_does_not_change_fitted_scheme(self):
        train = pd.DataFrame({SIGNAL: np.arange(100, dtype=float)})
        scheme = Scheme.fit(train, 4)
        before = scheme.json()
        transformed = scheme.transform(pd.DataFrame({SIGNAL: [-10000.0, 10000.0]}))
        self.assertEqual(before, scheme.json())
        self.assertEqual(transformed.rock_energy_type_final.tolist(), ["E1", "E4"])

    def test_ablations_remove_only_declared_inputs(self):
        _, numeric, categorical = specification("no_E")
        self.assertNotIn("hardness_score_smooth", numeric)
        self.assertEqual(categorical, [])
        _, numeric, categorical = specification("continuous_E")
        self.assertIn("hardness_score_smooth", numeric)
        self.assertEqual(categorical, [])
        k, numeric, categorical = specification("q4")
        self.assertEqual(k, 4)
        self.assertEqual(categorical, ["rock_energy_type_final"])

    def test_anchor_sampling_respects_warmup_and_nonoverlap(self):
        frame = pd.DataFrame({"well_id": ["a"]*100+["b"]*60,
                              "_row_in_well": list(range(100))+list(range(60))})
        selected = choose_anchors(frame, 3)
        self.assertEqual(len(selected), 6)
        self.assertTrue((selected._row_in_well >= 29).all())
        self.assertTrue((selected._row_in_well % 5 == 4).all())
        for _, part in selected.groupby("well_id"):
            self.assertTrue((part._row_in_well.diff().dropna() >= 5).all())

    def test_paired_bootstrap_detects_deliberately_worse_variant(self):
        rows = []
        for i in range(12):
            for variant in VARIANTS:
                rows.append({"well_id": str(i), "variant": variant, "fold": i%3,
                             "n": 10, "sse": 40.0 if variant=="q8" else 10.0,
                             "sae": 5.0, "y_sum": 50.0, "y2_sum": 300.0})
        summary, _ = summarize_prediction(pd.DataFrame(rows), reps=200)
        q8 = summary.set_index("variant").loc["q8"]
        self.assertAlmostEqual(q8.RMSE_delta_vs_q4_pct, 100)
        self.assertGreater(q8.delta_vs_q4_family_ci_low, 0)


if __name__ == "__main__":
    unittest.main()
