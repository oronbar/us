import unittest
import numpy as np
from ichilov3_encode_selected import windows,timestamps


class SamplingTests(unittest.TestCase):
    def test_four_windows_cover_both_ends(self):
        for stride in (1,2):
            ix=windows(100,stride)
            self.assertEqual(ix.shape,(4,16))
            self.assertEqual(ix[0,0],0)
            self.assertEqual(ix[-1,-1],99)
            np.testing.assert_array_equal(np.diff(ix,axis=1),np.full((4,15),stride))

    def test_short_clip_repeats_only_existing_frames(self):
        ix=windows(5,2)
        self.assertEqual(ix.shape,(1,16))
        self.assertEqual(ix.max(),4)
        self.assertEqual(ix.min(),0)
        self.assertEqual(ix[0,-1],4)

    def test_variable_frame_timing(self):
        record={'frames':4,'frame_time_vector_ms':[0,10,20,30]}
        values,source=timestamps(record,np.array([[0,1,2,3]]))
        np.testing.assert_allclose(values,[[0,.01,.03,.06]])
        self.assertEqual(source,'FrameTimeVector')

    def test_unknown_rate_is_not_invented(self):
        values,source=timestamps({'frames':3},np.array([[0,1,2]]))
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(source,'unavailable')


if __name__=='__main__':unittest.main()
