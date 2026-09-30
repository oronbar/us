import unittest
import numpy as np
import pandas as pd
from ichilov3_align_modalities import align_visits,history_ids,historical_video,strain_tensors


class AlignmentTests(unittest.TestCase):
    def visit(self):
        return pd.DataFrame([dict(visit_id='p__V01',patient_id='p',study_uid='report-uid',study_datetime='2020-01-01',visit_order=1)])

    def clips(self,uid='report-uid'):
        return [dict(patient='p',visit_date='2020-01-01',view=v,expected_study=uid,
                file_id=v,source_path=v,expected_sop=v,selection_source='manual',crop_output=v,qc_flags=[])
                for v in ['A2C','A3C','A4C']]

    def test_date_match_does_not_override_study_uid(self):
        _,audit,_=align_visits(self.visit(),self.clips('different-uid'))
        self.assertEqual(audit.match_status.iloc[0],'study_uid_mismatch')
        _,audit,_=align_visits(self.visit(),self.clips())
        self.assertEqual(audit.match_status.iloc[0],'exact_patient_date_study_uid')

    def test_ambiguous_same_day_strain_visits_are_rejected(self):
        visits=pd.concat([self.visit(),self.visit()],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            align_visits(visits,self.clips())

    def test_history_uses_true_previous_visit_and_missing_video_has_mask(self):
        visits=pd.DataFrame([dict(visit_id='v'+str(i),patient_id='p',visit_order=i) for i in range(1,5)]).set_index('visit_id')
        self.assertEqual(history_ids(visits,'v3'),('v1','v2'))
        self.assertEqual(history_ids(visits,'v1'),(None,None))
        delta,mask=historical_video(np.ones((3,4)),None)
        self.assertFalse(mask.any());self.assertTrue((delta==0).all())

    def test_technical_replicates_are_averaged_without_merging_visits(self):
        rows=[dict(visit_id='v',curve_family='longitudinal_strain',layer=l,segment_number=s,
                   resampled_values=np.full(96,value)) for l in ['endo','mid'] for s in range(1,19) for value in [2.,4.]]
        tensors=strain_tensors(pd.DataFrame(rows));self.assertTrue((tensors['v']==3.).all())


if __name__=='__main__':unittest.main()
