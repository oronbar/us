"""Read-only metadata audit for identifiable TOMTEC analysis objects.

A negative metadata search is not proof of absence after anonymization.
Images are not OCRed and proprietary payloads are not validated by this scan.
"""
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pydicom

OUT=Path('D:/us/output/dicom_prediction/retrieval_20260921')
RECON=Path('D:/us/output/dicom_prediction/reconciliation/workbook_data.json')
BULK={0x7FE00010,0x7FE00008,0x7FE00009,0x54001010}
MARKERS=('tomtec','tom-tec','autostrain','imagearena','image arena','tomtec-arena')


def scan_file(path):
    result=dict(file=str(path),private_tags=0,content_sequence=False,encapsulated_document=False,markers=[])
    try:
        ds=pydicom.dcmread(path,defer_size=1024*1024)
        result.update(study_uid=str(ds.get('StudyInstanceUID','')),instance=str(ds.get('InstanceNumber','')),
                      model=str(ds.get('ManufacturerModelName','')),software=str(ds.get('SoftwareVersions','')))
        def walk(dataset):
            for tag in dataset.keys():
                if int(tag) in BULK:continue
                elem=dataset[tag]
                if tag.is_private:result['private_tags']+=1
                if elem.keyword=='ContentSequence':result['content_sequence']=True
                if elem.keyword=='EncapsulatedDocument':result['encapsulated_document']=True
                if elem.VR=='SQ':
                    for item in elem.value:walk(item)
                elif elem.VR in {'LO','SH','ST','LT','UT','CS','UC'}:
                    value=str(elem.value)
                    if any(s in value.lower() for s in MARKERS):
                        result['markers'].append(dict(tag=str(tag),keyword=elem.keyword,value=value[:500]))
        walk(ds)
        result['status']='readable'
    except Exception as exc:result.update(status='read_error',error=str(exc)[:250])
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    studies=json.loads(RECON.read_text())['dicom_studies']
    def scan_folder(row):
        records=[scan_file(p) for p in sorted(Path(row['patient_folder']).rglob('*')) if p.is_file()]
        return dict(study_uid=row['study_uid'],folder=row['patient_folder'],expected_files=row['files'],
            files_scanned=len(records),errors=sum(r['status']!='readable' for r in records),
            marker_files=sum(bool(r['markers']) for r in records),
            private_files=sum(r['private_tags']>0 for r in records),
            structured_files=sum(r['content_sequence'] or r['encapsulated_document'] for r in records),records=records)
    results=[];start=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,result in enumerate(pool.map(scan_folder,studies),1):
            results.append(result)
            if n%40==0:print(f'Audited {n}/{len(studies)} folders in {time.time()-start:.0f}s',flush=True)
    (OUT/'tomtec_metadata_audit.json').write_text(json.dumps(results,indent=2))
    summary=dict(folders=len(results),files=sum(r['files_scanned'] for r in results),
        errors=sum(r['errors'] for r in results),folders_with_markers=sum(r['marker_files']>0 for r in results),
        folders_with_private_fields=sum(r['private_files']>0 for r in results),
        folders_with_structured_objects=sum(r['structured_files']>0 for r in results),
        scope='Full DICOM dataset metadata; pixel data deferred and not OCRed; proprietary payloads not decoded',
        limitation='No marker does not establish absence after anonymization; confirmed project availability requires source-link validation')
    assert all(r['files_scanned']==r['expected_files'] for r in results)
    (OUT/'tomtec_audit_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
