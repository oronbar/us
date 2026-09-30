import unittest
import numpy as np
from ichilov3_train_fusion import preprocess,matrix


class FusionTests(unittest.TestCase):
    def test_preprocessing_is_fitted_only_on_training_rows(self):
        x=np.array([[1.,np.nan],[3.,4.],[10000.,8000.]])
        transformed,pipes=preprocess({'clinical':x},np.array([0,1]),np.array([2]),7)
        self.assertTrue(np.allclose(pipes['clinical']['impute'].statistics_,[2.,4.]))
        self.assertTrue(np.allclose(pipes['clinical']['scale'].mean_,[2.,4.]))
        self.assertTrue(np.isfinite(transformed['clinical'][1]).all())

    def test_video_representation_does_not_mix_encoders(self):
        blocks={name:(np.full((2,1),value),np.full((1,1),value)) for name,value in
                [('clinical',1),('strain_scalars',2),('strain_curves',3),
                 ('echoprime_0',4),('echoprime_1',5),('echoprime_2',6),
                 ('panecho_0',7),('panecho_1',8),('panecho_2',9)]}
        a,b=matrix(blocks,'echoprime');self.assertEqual(a.shape,(2,6));self.assertEqual(b.tolist(),[[1,2,3,4,5,6]])
        a,b=matrix(blocks,'clinical');self.assertEqual(a.shape,(2,1))


if __name__=='__main__':unittest.main()
