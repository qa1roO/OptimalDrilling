import unittest

import numpy as np
import pandas as pd

from study_energy_classes import SIGNAL
from study_learned_energy_groups import KMeansScheme, GMMScheme


class LearnedEnergyGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.training = pd.DataFrame({SIGNAL: np.repeat([17., 18., 19., 20.], 240)+rng.normal(0, .04, 960),
                                     "_row_in_well": np.arange(960)})

    def test_learned_labels_follow_well_separated_component_order(self):
        points = pd.DataFrame({SIGNAL: [17., 18., 19., 20.]})
        for typ in [KMeansScheme, GMMScheme]:
            model = typ.fit(self.training, 4)
            self.assertEqual(model.transform(points).rock_energy_type_final.tolist(), ['E1','E2','E3','E4'])

    def test_transform_does_not_refit_on_new_distribution(self):
        for typ in [KMeansScheme, GMMScheme]:
            model = typ.fit(self.training, 4)
            before = model.json()
            model.transform(pd.DataFrame({SIGNAL: [-1000., 1000.]}))
            self.assertEqual(before, model.json())

    def test_missing_energy_keeps_unknown_label(self):
        for typ in [KMeansScheme, GMMScheme]:
            model = typ.fit(self.training, 4)
            self.assertEqual(model.transform(pd.DataFrame({SIGNAL: [np.nan]})).rock_energy_type_final.iloc[0], 'unknown')


if __name__=='__main__':
    unittest.main()
