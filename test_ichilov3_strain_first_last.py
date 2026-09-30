import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from ichilov3_strain_first_last import cohort_from_tables, CurveFeatures


class FirstLastTests(unittest.TestCase):
    def tables(self):
        visits, curves = [], []
        for patient, magnitudes in [('decline', [20, 18, 16]), ('recover', [20, 15, 21])]:
            for i, magnitude in enumerate(magnitudes):
                key = f'{patient}_{i}'
                visits.append(dict(patient_id=patient, visit_id=key,
                                   study_datetime=f'2020-0{i+1}-01', gls_mid_peak_avg=-magnitude))
                for segment in range(1, 19):
                    for layer in ['endo', 'mid']:
                        curves.append(dict(visit_id=key, curve_family='longitudinal_strain',
                                           layer=layer, segment_number=segment,
                                           resampled_values=-magnitude * np.sin(np.linspace(0, np.pi, 96))))
        return pd.DataFrame(visits), pd.DataFrame(curves)

    def test_labels_and_recovery_exclusion(self):
        visits, curves = self.tables()
        cohort, X, audit = cohort_from_tables(visits, curves, {'decline', 'recover'}, 'last')
        self.assertEqual(set(cohort.patient_id), {'decline'})
        self.assertEqual(X.shape, (72, 96))
        self.assertEqual(cohort.groupby('label').size().to_dict(), {0: 36, 1: 36})
        self.assertEqual(set(cohort.visit_id), {'decline_0', 'decline_2'})
        self.assertEqual(audit.set_index('patient_id').loc['recover', 'status'], 'no_qualifying_deterioration')

    def test_missing_endpoint_never_replaced_by_middle(self):
        visits, curves = self.tables()
        curves = curves[~(curves.visit_id.eq('decline_2') & curves.segment_number.eq(1))]
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            cohort_from_tables(visits, curves, {'decline'}, 'last')

    def test_grouped_folds_and_features(self):
        groups = np.repeat(np.arange(10), 72)
        X = np.zeros((len(groups), 96))
        for train, test in GroupKFold(5).split(X, groups=groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(CurveFeatures('summary').fit_transform(X).shape, (720, 7))
        self.assertEqual(CurveFeatures('waveform').fit_transform(X).shape, (720, 24))
        with self.assertRaises(ValueError):
            CurveFeatures().transform(np.zeros((2, 18, 2, 96)))


if __name__ == '__main__':
    unittest.main()
