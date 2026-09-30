import json, tempfile, unittest
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset
from server import Store, tissue_box, selected_box


class ReviewTests(unittest.TestCase):
    def test_region_bounds_and_manual_rectangle(self):
        ds=Dataset();ds.Rows=100;ds.Columns=200
        r=Dataset();r.RegionSpatialFormat=1;r.RegionDataType=1
        r.RegionLocationMinX0=20;r.RegionLocationMinY0=10;r.RegionLocationMaxX1=179;r.RegionLocationMaxY1=89
        ds.SequenceOfUltrasoundRegions=[r]
        self.assertEqual(tissue_box(ds),[20,10,180,90])
        self.assertEqual(selected_box(tissue_box(ds),[.25,.25,.75,.75]),[60,30,140,70])
        for rect in [[-1,0,1,1],[0,0,.01,1],[0,0,float('nan'),1]]:
            with self.assertRaises(ValueError):selected_box(tissue_box(ds),rect)

    def test_reviews_persist_and_rejected_clips_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);crops=p/'crops';crops.mkdir();output=p/'output'
            record=dict(file_id='test',patient='p1',visit_date='2020-01-01',view='A2C',frames=16,
                manufacturer='test',model='test',removed_foreground_max=.2,status='needs_review',crop_output='original.npz')
            (crops/'full_manifest.json').write_text(json.dumps([record]))
            store=Store(output,crops,p/'embeddings')
            self.assertEqual(store.state()['clips'][0]['review_index'],1)
            original=(crops/'full_manifest.json').read_bytes()
            result=store.decide('test','good');self.assertIsNone(result['before'])
            self.assertEqual(Store(output,crops,p/'embeddings').state()['good'],1)
            store.decide('test','bad');self.assertFalse(store.export_manifest()[0]['training_eligible_crop'])
            store.decide('test',None);self.assertEqual(store.state()['bad'],0)
            self.assertEqual(store.export_manifest()[0]['review_quality'],'pending')
            self.assertEqual((crops/'full_manifest.json').read_bytes(),original)
            with self.assertRaises(ValueError):store.decide('test','repaired','../unsafe')
            with self.assertRaises(ValueError):store.decide('missing','good')
            with self.assertRaises(ValueError):store.candidate('test','not-classified')


if __name__=='__main__':unittest.main()
