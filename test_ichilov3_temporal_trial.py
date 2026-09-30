import unittest
import numpy as np
from ichilov3_temporal_trial import heartbeat_windows

class TemporalTests(unittest.TestCase):
    def test_variable_timing_nearest_frames_and_period(self):
        vector=np.resize([10.,20.,15.],150).tolist()
        r=dict(frames=150,frame_time_vector_ms=vector,heart_rate=75)
        idx,info=heartbeat_windows(r)
        self.assertEqual(idx.shape,(4,16));self.assertTrue((np.diff(idx)>0).all())
        self.assertLess(abs(info['estimated_beats_per_window']-1),.03)
        self.assertTrue((idx>=0).all());self.assertTrue((idx<150).all())
    def test_short_clip_does_not_fabricate_cycle(self):
        idx,info=heartbeat_windows(dict(frames=30,frame_time_ms=10,heart_rate=60))
        self.assertIsNone(idx);self.assertEqual(info['reason'],'cine_shorter_than_estimated_period')
    def test_invalid_timing_or_hr_falls_back(self):
        for r in [dict(frames=100,heart_rate=60),dict(frames=100,frame_time_ms=20,heart_rate=0),dict(frames=100,heart_rate=60,frame_time_vector_ms=[0.]*100)]:
            self.assertIsNone(heartbeat_windows(r)[0])
    def test_cine_rate_and_low_temporal_resolution(self):
        idx,_=heartbeat_windows(dict(frames=100,cine_rate=50,heart_rate=60))
        self.assertEqual(idx.shape,(4,16))
        idx,info=heartbeat_windows(dict(frames=100,cine_rate=5,heart_rate=60))
        self.assertIsNone(idx);self.assertEqual(info['reason'],'insufficient_unique_frames_for_16_samples')

if __name__=='__main__':unittest.main()
