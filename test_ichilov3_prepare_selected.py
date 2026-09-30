"""Regression checks for spatial geometry and preservation of cine data."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

UltrasoundMultiframeImageStorage = '1.2.840.10008.5.1.4.1.1.3.1'

from ichilov3_prepare_selected import padded_crop, header, process, digest
from ichilov_crop_dicoms import _crop_square


class CropTests(unittest.TestCase):
    def test_wide_sector_preserved_without_distortion(self):
        frame=np.ones((20,40,1),dtype=np.uint8)*123
        padded=padded_crop(frame,(0,20,0,40))
        self.assertEqual(padded.shape,(40,40,1))
        self.assertTrue(np.array_equal(padded[10:30],frame))
        self.assertEqual(np.count_nonzero(padded),np.count_nonzero(frame))
        self.assertEqual(np.count_nonzero(_crop_square(frame,(0,20,0,40),20)),400)

    def test_cine_roundtrip_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);source=folder/'source_without_extension'
            fm=FileMetaDataset();fm.TransferSyntaxUID=ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID=UltrasoundMultiframeImageStorage
            fm.MediaStorageSOPInstanceUID=generate_uid()
            ds=FileDataset(str(source),{},file_meta=fm,preamble=b'\0'*128)
            ds.SOPClassUID=fm.MediaStorageSOPClassUID;ds.SOPInstanceUID=fm.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID=generate_uid();ds.Rows=32;ds.Columns=64
            ds.NumberOfFrames=3;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
            ds.BitsAllocated=8;ds.BitsStored=8;ds.HighBit=7;ds.PixelRepresentation=0;ds.FrameTime=20
            arr=np.zeros((3,32,64),dtype=np.uint8)
            for i in range(3):arr[i,4:28,8:56]=50+i*40
            ds.PixelData=arr.tobytes();ds.save_as(source,enforce_file_format=True)
            original=digest(source)
            task=header(dict(file_id='example',source_path=str(source),patient='p',visit_date='2020-01-01',view='A2C',expected_sop=ds.SOPInstanceUID,expected_study=ds.StudyInstanceUID))
            args=argparse.Namespace(output=folder/'derived',method='padded',size=48)
            record=process(task,args)
            self.assertNotEqual(record['status'],'processing_error',record.get('error'))
            self.assertEqual(record['frames'],3)
            self.assertEqual(record['frame_time_ms'],20)
            self.assertEqual(record['source_sha256'],original)
            self.assertEqual(digest(source),original)
            with np.load(record['crop_output']) as cache:
                self.assertEqual(cache['frames'].shape,(3,48,48))
                np.testing.assert_array_equal(cache['source_frame_indices'],[0,1,2])
                np.testing.assert_array_equal(cache['frames'][:,24,24],[50,90,130])
            self.assertEqual(process(task,args)['source_sha256'],original)


if __name__=='__main__':unittest.main()
