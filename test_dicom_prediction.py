import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, UltrasoundMultiFrameImageStorage, generate_uid

from dicom_prediction_inventory import inspect_file, report_inventory
from dicom_prediction_video import spatial_geometry, decode_frames, normalize, normalize_appearance
from dicom_prediction_encode import window_indices
from dicom_prediction_evaluate import features_for_transitions
import pandas as pd
import torch


class DicomPredictionTests(unittest.TestCase):
    def test_extensionless_missing_sop_and_frame_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'I1000000'
            fm=FileMetaDataset();fm.TransferSyntaxUID=ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID=UltrasoundMultiFrameImageStorage;fm.MediaStorageSOPInstanceUID=generate_uid()
            d=FileDataset(str(path),{},file_meta=fm,preamble=b'\0'*128)
            d.StudyInstanceUID=generate_uid();d.Modality='US';d.Rows=20;d.Columns=40
            d.NumberOfFrames=3;d.SamplesPerPixel=3;d.PhotometricInterpretation='RGB';d.PlanarConfiguration=0
            d.BitsAllocated=8;d.BitsStored=8;d.HighBit=7;d.PixelRepresentation=0
            arr=np.stack([np.full((20,40,3),v,np.uint8) for v in [30,80,140]])
            d.PixelData=arr.tobytes();d.save_as(path,enforce_file_format=True)
            row=inspect_file((str(path),path.stat().st_size,path.stat().st_mtime_ns,path.name))
            self.assertEqual(row['status'],'dicom');self.assertEqual(row['sop_uid'],'')
            frames=decode_frames(path,[2,0,2],[0,0,40,20])
            self.assertTrue(np.array_equal(frames[0],frames[2]));self.assertEqual(frames[0,112,112,0],140)
            self.assertEqual(frames[1,112,112,0],30);self.assertEqual(frames[0,0,0,0],0)

    def test_no_dicom_false_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plain';path.write_text('ordinary text, not a medical image')
            row=inspect_file((str(path),path.stat().st_size,path.stat().st_mtime_ns,path.name))
            self.assertNotEqual(row['status'],'dicom')

    def test_clamped_region_and_no_region_rejection(self):
        d=Dataset();d.Rows=100;d.Columns=200
        self.assertFalse(spatial_geometry(d)['eligible_region'])
        r=Dataset();r.RegionSpatialFormat=1;r.RegionDataType=1
        r.RegionLocationMinX0=10;r.RegionLocationMinY0=20;r.RegionLocationMaxX1=250;r.RegionLocationMaxY1=150
        d.SequenceOfUltrasoundRegions=[r]
        self.assertEqual(spatial_geometry(d)['crop'],[10,20,200,100])

    def test_reanalysis_exports_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['a.csv','b.csv']:(Path(directory)/name).write_text('Study UID,1.2.3\nStudy Date and Time,2020-01-01,12:00:00\n')
            reports=report_inventory(Path(directory))
            self.assertEqual(len(reports),2);self.assertEqual(reports.study_uid.nunique(),1)

    def test_normalization_channel_axis_for_video(self):
        x=torch.zeros(2,3,16,2,2)
        out=normalize(x,'echoprime')
        self.assertAlmostEqual(out[0,0,0,0,0].item(),-29.110628/47.989223,places=5)
        self.assertEqual(tuple(out.shape),tuple(x.shape))

    def test_window_indices_keep_timing_and_bounds(self):
        windows=window_indices(100,2)
        self.assertEqual(len(windows),3)
        self.assertEqual(windows[0].tolist(),list(range(0,32,2)))
        self.assertEqual(windows[-1][-1],99)
        self.assertTrue(all((x>=0).all() and (x<10).all() for x in window_indices(10,2)))

    def test_tint_normalization_preserves_multicolor_flow(self):
        amber=np.zeros((2,20,20,3),np.uint8);amber[:]=[180,100,20]
        gray,flag=normalize_appearance(amber)
        self.assertEqual(flag,'monochrome-tint-v1');self.assertTrue((gray==180).all())
        flow=amber.copy();flow[:,:,:10]=[200,0,0];flow[:,:,10:]=[0,0,200]
        unchanged,flag=normalize_appearance(flow)
        self.assertEqual(flag,'native_rgb');self.assertTrue(np.array_equal(flow,unchanged))

    def test_history_never_uses_future_or_substitutes_baseline(self):
        visits=pd.DataFrame(dict(visit_id=['v1','v2','v3'],patient_id=['p']*3,visit_order=[1,2,3],study_uid=['s1','s2','s3']))
        transitions=pd.DataFrame(dict(patient_id=['p'],current_visit_id=['v2']))
        vectors={'s2':(np.array([[2.]],np.float32),np.ones(1,np.float32)),
                 's3':(np.array([[999.]],np.float32),np.ones(1,np.float32))}
        f=features_for_transitions(transitions,visits,vectors,history=True)
        self.assertEqual(f.tolist(),[[2.,1.,0.,0.,0.,0.]])
        vectors['s1']=(np.array([[1.]],np.float32),np.ones(1,np.float32))
        self.assertEqual(features_for_transitions(transitions,visits,vectors,True).tolist(),[[2.,1.,1.,1.,1.,1.]])


if __name__=='__main__':unittest.main()
