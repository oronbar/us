"""Read-only reconciliation of every mapping row, strain export and DICOM folder."""
import collections
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import pandas as pd
from dicom_prediction_inventory import report_inventory, inspect_file

ROOT=Path('D:/us')
OUT=ROOT/'output/dicom_prediction/reconciliation'
DRIVE=Path('E:/')


def join(values):
    return '; '.join(sorted(set(str(x) for x in values if pd.notna(x) and str(x))))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mapping_path=DRIVE/'anonymization_mapping_hashed.xlsx'
    mapping=pd.read_excel(mapping_path,dtype=str).fillna('')
    mapping['excel_row']=range(2,len(mapping)+2)
    mapping['report_name']=mapping.anonymized_file.map(lambda x:Path(x).name)
    reports=report_inventory(Path('D:/DS/anonymized_reports'))
    reports['report_name']=reports.report_path.map(lambda x:Path(x).name)
    assert mapping.report_name.is_unique and reports.report_name.is_unique
    audit=mapping.merge(reports,on='report_name',how='outer',validate='one_to_one',indicator=True)
    assert audit._merge.eq('both').all(), 'Mapping/report filename mismatch: inspect outer join'
    assert audit.study_date_and_time.eq(audit.study_datetime).all(), 'Mapping/report date disagreement'
    audit['report_id']=audit.report_name.map(lambda x:re.sub(r'^AutoStrainCap_|_\d{8}_\d{6}\.txt$','',x))
    audit['report_date']=audit.study_datetime.str[:10]
    old=pd.read_parquet(ROOT/'output/dicom_prediction/dicom_inventory.parquet')
    cache=old.set_index('path').to_dict('index')
    files=[];new=[];folders=[];errors=[];roots=sorted(DRIVE.glob('echo_*'))
    for root in roots:
        if not root.is_dir():continue
        for directory,dirs,names in os.walk(root,onerror=lambda e:errors.append(str(e))):
            if 'US' in dirs:
                folders.append(dict(patient_folder=str(Path(directory)),PatID=Path(directory).name,phase=root.name))
            for name in names:
                p=Path(directory)/name;s=p.stat();key=str(p)
                saved=cache.get(key)
                if saved and int(saved['size'])==s.st_size and int(saved['mtime'])==s.st_mtime_ns:
                    files.append(dict(path=key,**saved))
                else:new.append((key,s.st_size,s.st_mtime_ns,str(p.relative_to(DRIVE))))
    print(f'Fresh filesystem scan: {len(files)+len(new)} files, {len(folders)} patient-visit folders, {len(new)} new/changed headers',flush=True)
    if errors:raise RuntimeError(errors)
    files.extend(inspect_file(item) for item in new)
    inv=pd.DataFrame(files)
    assert inv.status.eq('dicom').all(), 'Unreadable/non-DICOM files need separate review'
    def owner(path):
        p=Path(path);parts=p.parts;ix=parts.index('US')
        return str(Path(*parts[:ix])),parts[ix-1]
    inv[['patient_folder','PatID']]=pd.DataFrame(inv.path.map(owner).tolist(),index=inv.index)
    inv['phase']=inv.relative_path.map(lambda x:Path(x).parts[0])
    inv['frames']=pd.to_numeric(inv.frames,errors='coerce').fillna(1)
    dicoms=inv.groupby(['patient_folder','PatID','phase','study_uid','study_date'],dropna=False).agg(
        files=('path','size'),videos=('frames',lambda x:int(x.gt(1).sum())),sample_file=('path','first')).reset_index()
    dicoms['dicom_date']=pd.to_datetime(dicoms.study_date,format='%Y%m%d',errors='raise').dt.strftime('%Y-%m-%d')
    dicoms['report_exports']=dicoms.study_uid.map(audit.study_uid.value_counts()).fillna(0).astype(int)
    dicoms['in_mapping']=dicoms.PatID.isin(mapping.PatID)
    known={k:g for k,g in dicoms.groupby('PatID')}
    uid_index={k:g for k,g in dicoms.groupby('study_uid')}
    visits=[]
    for (patient,uid),g in audit.groupby(['PatID','study_uid']):
        date=g.report_date.iloc[0];d=known.get(patient,dicoms.iloc[:0]);exact=uid_index.get(uid,dicoms.iloc[:0])
        assert exact.empty or exact.PatID.eq(patient).all(), 'UID found under different patient'
        same=d[d.dicom_date.eq(date)]
        status='Exact Study UID' if len(exact) else ('Same date, different UID: review' if len(same) else 'No DICOM on report date')
        target=exact if len(exact) else same
        visits.append(dict(PatID=patient,report_id=join(g.report_id),report_date=date,study_datetime=g.study_datetime.iloc[0],
            report_uid=uid,report_exports=len(g),excel_rows=join(g.excel_row),report_files=join(g.report_name),
            status=status,exact_files=int(exact.files.sum()),exact_videos=int(exact.videos.sum()),
            matched_phase=join(exact.phase),matched_folder=join(exact.patient_folder),
            candidate_uid=join(same.study_uid) if exact.empty else '',candidate_folder=join(same.patient_folder) if exact.empty else '',
            available_dicom_dates=join(d.dicom_date),
            action='None' if len(exact) else ('Verify same-date study UID against original export' if len(same) else 'Retrieve DICOM study for this exact report date and UID')))
    visits=pd.DataFrame(visits).sort_values(['report_id','report_date','report_uid']).reset_index(drop=True)
    visits['report_visit_order']=visits.groupby('PatID').cumcount()+1
    audit=audit.merge(visits[['PatID','report_uid','status','matched_phase','matched_folder','candidate_uid','candidate_folder','report_exports']],
                      left_on=['PatID','study_uid'],right_on=['PatID','report_uid'],validate='many_to_one')
    patients=[]
    for patient in sorted(set(mapping.PatID)|set(dicoms.PatID)):
        v=visits[visits.PatID.eq(patient)];d=known.get(patient,dicoms.iloc[:0]);m= mapping[mapping.PatID.eq(patient)]
        missing=v[~v.status.eq('Exact Study UID')];extra=d[d.report_exports.eq(0)]
        patients.append(dict(PatID=patient,report_id=join(v.report_id) or '(no strain mapping)',
            report_exports=len(m),report_visits=len(v),extra_report_exports=len(m)-len(v),
            dicom_folders=d.patient_folder.nunique(),dicom_studies=d.study_uid.nunique(),
            exact_matched_visits=int(v.status.eq('Exact Study UID').sum()),
            no_dicom_on_date=int(v.status.eq('No DICOM on report date').sum()),
            uid_review=int(v.status.eq('Same date, different UID: review').sum()),
            dicom_without_report=len(extra),missing_report_dates=join(missing.report_date),
            available_dicom_dates=join(d.dicom_date),extra_dicom_dates=join(extra.dicom_date),
            available_by_folder=join(f'{r.phase.replace("echo_","")}: {r.dicom_date}' for r in d.itertuples()),
            mapping_rows=join(m.excel_row)))
    patients=pd.DataFrame(patients).sort_values(['no_dicom_on_date','uid_review','report_id'],ascending=[False,False,True]).reset_index(drop=True)
    phases=dicoms.assign(matched=dicoms.report_exports.gt(0).astype(int),unmatched=dicoms.report_exports.eq(0).astype(int)).groupby('phase').agg(
        folders=('patient_folder','nunique'),studies=('study_uid','nunique'),with_report=('matched','sum'),without_report=('unmatched','sum')).reset_index()
    empty=set(x['patient_folder'] for x in folders)-set(dicoms.patient_folder)
    summary=dict(audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),excel_entries=len(mapping),strain_files=len(reports),
        unique_report_visits=len(visits),duplicate_export_visits=int(visits.report_exports.gt(1).sum()),extra_exports=len(audit)-len(visits),
        dicom_files=len(inv),dicom_folders=len(folders),dicom_studies=dicoms.study_uid.nunique(),empty_folders=sorted(empty),
        fresh_new_or_changed_files=len(new),removed_since_inventory=len(set(cache)-set(inv.path)),
        status_counts=visits.status.value_counts().to_dict(),report_row_status_counts=audit.status.value_counts().to_dict(),
        mapping_patients=mapping.PatID.nunique(),dicom_patients=dicoms.PatID.nunique(),
        patients_missing_exact_match=int((patients.no_dicom_on_date+patients.uid_review).gt(0).sum()),
        patients_no_dicom_on_date=int(patients.no_dicom_on_date.gt(0).sum()),
        patients_all_report_visits_exact=int(((patients.report_visits>0)&patients.no_dicom_on_date.eq(0)&patients.uid_review.eq(0)).sum()),
        dicom_only_patients=sorted(set(dicoms.PatID)-set(mapping.PatID)),
        dicom_studies_without_report=int(dicoms.report_exports.eq(0).sum()),
        missing_by_report_order=visits[~visits.status.eq('Exact Study UID')].report_visit_order.value_counts().sort_index().to_dict(),
        mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest())
    # These are dated audit snapshots. No source files or prior prediction links are changed.
    frames={'patients':patients,'visits':visits,'report_entries':audit.drop(columns=['_merge','patient_date_of_birth','patient_sex']),
            'dicom_studies':dicoms,'folder_counts':phases}
    for name,frame in frames.items():
        frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    visits[~visits.status.eq('Exact Study UID')].to_csv(OUT/'missing_or_mismatched_visits.csv',index=False,encoding='utf-8-sig')
    dicoms[dicoms.report_exports.eq(0)].to_csv(OUT/'dicom_studies_without_reports.csv',index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    payload={name:json.loads(frame.to_json(orient='records')) for name,frame in frames.items()}
    payload['summary']=summary
    (OUT/'workbook_data.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(phases.to_string(index=False),flush=True)


if __name__=='__main__':main()
