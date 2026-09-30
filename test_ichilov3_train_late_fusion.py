import unittest
import numpy as np
from ichilov3_train_late_fusion import fixed_blend,video_preprocess


class LateFusionTests(unittest.TestCase):
    def test_fixed_blend_endpoints_and_weights(self):
        self.assertTrue(np.allclose(fixed_blend([.8],[.2],.25),[.65]))
        self.assertTrue(np.allclose(fixed_blend([.8],[.2],0),[.8]))
        with self.assertRaises(ValueError):fixed_blend([.8],[.2],2)

    def test_video_scaling_excludes_heldout_patients(self):
        x=np.array([[[1.],[1.],[1.]],[[3.],[3.],[3.]],[[999.],[999.],[999.]]])
        train,test,pipes=video_preprocess(x,np.array([0,1]),np.array([2]),'full_embedding',1)
        self.assertEqual(train.shape,(2,3));self.assertEqual(test.shape,(1,3))
        for pipe in pipes:self.assertTrue(np.allclose(pipe['scale'].mean_,[2.]))


if __name__=='__main__':unittest.main()
