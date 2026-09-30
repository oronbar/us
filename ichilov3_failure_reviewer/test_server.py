import csv
import json
import tempfile
import unittest
from pathlib import Path
from pydicom.dataset import Dataset, FileDataset

from server import Store


class ReviewStoreTests(unittest.TestCase):
    def test_error_order_and_durable_review_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=[]
            for patient in ('p1','p2'):
                for view in ('A2C','A3C','A4C'):
                    manifest.append(dict(patient=patient,visit_date='2020-01-01',view=view,
                        file_id=f'{patient}-{view}',frames=20,model='test',manufacturer='Philips',
                        selection_source='TOMTEC bookmark',source_path=f'{patient}-{view}.dcm',
                        crop_output=f'{patient}-{view}.npz',frame_time_ms=40,status='processed'))
            manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest))
            prediction_path=root/'predictions.csv'
            with prediction_path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['target','visit_id','patient_id','visit_date',
                    'manufacturer','vendor_short','value','prediction','error','absolute_error','all_bookmark'])
                writer.writeheader()
                for patient,mid_error in [('p1',1.0),('p2',3.0)]:
                    for target,error in [('mid_gls',mid_error),('endo_gls',2.0)]:
                        writer.writerow(dict(target=target,visit_id=f'{patient}__V01',patient_id=patient,
                            visit_date='2020-01-01',manufacturer='Philips',vendor_short='Philips',
                            value=15,prediction=15+error,error=error,absolute_error=abs(error),all_bookmark=True))
            before=manifest_path.read_bytes(),prediction_path.read_bytes()
            store=Store(root/'output',manifest_path,prediction_path)
            self.assertEqual([v['visit_id'] for v in store.visits],['p2__V01','p1__V01'])
            with self.assertRaises(ValueError):store.save_review(dict(visit_id='p2__V01',status='no_issue',reasons=['wrong_view']))
            store.save_review(dict(visit_id='p2__V01',status='suspected_issue',reasons=['wrong_view'],note='A3C'))
            fresh=Store(root/'output',manifest_path,prediction_path)
            self.assertEqual(fresh.reviews()['p2__V01']['reasons'],['wrong_view'])
            self.assertIn(b'suspected_issue',fresh.export_csv())
            self.assertEqual(before,(manifest_path.read_bytes(),prediction_path.read_bytes()))

    def test_approved_replacement_is_same_study_and_reversible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=root/'original.dcm'
            original.write_bytes(b'original-selected-cine')
            alternative=root/'alternative.dcm'
            ds=FileDataset(str(alternative),{},file_meta=Dataset(),preamble=b'\0'*128)
            ds.SOPInstanceUID='1.2.826.0.1.3680043.10.999.12'
            ds.StudyInstanceUID='1.2.826.0.1.3680043.10.999.1'
            ds.save_as(alternative)
            selected=[]
            for view in ('A2C','A3C','A4C'):
                selected.append(dict(patient='p1',visit_date='2020-01-01',view=view,file_id='p1-'+view,
                    frames=20,model='test',manufacturer='Philips',selection_source='TOMTEC bookmark',
                    source_path=str(original),expected_study=str(ds.StudyInstanceUID),
                    expected_sop=f'1.2.826.0.1.3680043.10.999.{len(selected)+2}',
                    crop_output=str(root/'crop.npz'),frame_time_ms=40,status='processed'))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(selected))
            predictions=root/'predictions.csv'
            with predictions.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['target','visit_id','patient_id','visit_date',
                    'manufacturer','vendor_short','value','prediction','error','absolute_error','all_bookmark'])
                writer.writeheader()
                for target in ('mid_gls','endo_gls'):
                    writer.writerow(dict(target=target,visit_id='p1__V01',patient_id='p1',visit_date='2020-01-01',
                        manufacturer='Philips',vendor_short='Philips',value=15,prediction=18,error=3,absolute_error=3,all_bookmark=True))
            store=Store(root/'out',manifest,predictions)
            alternative_record=dict(selected[0],source_path=str(alternative),expected_sop=str(ds.SOPInstanceUID))
            cached=[dict(candidate_id='candidate-1',name=alternative.name,view='A2C',probability=.9,
                         score=.85,quality={'duration_seconds':1.0},warnings=[],record=alternative_record)]
            store.candidate_cache(selected[0]).write_text(json.dumps(cached))
            with self.assertRaisesRegex(ValueError,'Suspected issue'):
                store.set_replacement('p1__V01','A2C','candidate-1')
            store.save_review(dict(visit_id='p1__V01',status='suspected_issue',reasons=['wrong_view'],note='Review'))
            saved=store.set_replacement('p1__V01','A2C','candidate-1')
            self.assertEqual(saved['source_path'],str(alternative))
            fresh=Store(root/'out',manifest,predictions)
            self.assertEqual(fresh.state()['visits'][0]['replacements']['A2C']['candidate_id'],'candidate-1')
            self.assertIn(str(alternative).encode(),fresh.export_csv())
            with self.assertRaisesRegex(ValueError,'Revert replacement'):
                fresh.save_review(dict(visit_id='p1__V01',status='no_issue',reasons=[],note=''))
            self.assertIsNone(fresh.set_replacement('p1__V01','A2C',None))
            self.assertFalse(fresh.replacements())
            self.assertEqual(original.read_bytes(),b'original-selected-cine')


if __name__=='__main__': unittest.main()
