"""Read-only extension-independent DICOM inventory and exact study linkage.

Source media is never renamed or modified. SQLite caches successful and failed
header reads by absolute path, byte size and mtime. No pixel decoding is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

FIELDS = {
    'StudyInstanceUID': 'study_uid', 'SOPInstanceUID': 'sop_uid',
    'SeriesInstanceUID': 'series_uid', 'SOPClassUID': 'sop_class',
    'Modality': 'modality', 'StudyDate': 'study_date',
    'NumberOfFrames': 'frames', 'Rows': 'rows', 'Columns': 'columns',
    'PhotometricInterpretation': 'photometric', 'FrameTime': 'frame_time_ms',
    'CineRate': 'cine_rate', 'RecommendedDisplayFrameRate': 'display_fps',
    'Manufacturer': 'manufacturer', 'ManufacturerModelName': 'machine',
    'ViewPosition': 'view_position', 'SamplesPerPixel': 'samples_per_pixel',
    'BitsAllocated': 'bits_allocated',
}
REFERENCE_TAGS = ['SourceImageSequence', 'ReferencedImageSequence',
                  'ReferencedSeriesSequence', 'CurrentRequestedProcedureEvidenceSequence',
                  'PertinentOtherEvidenceSequence', 'ContentSequence']


def inspect_file(item):
    path, size, mtime, relative = item
    row = dict(path=path, size=size, mtime=mtime, relative_path=relative,
               file_id=hashlib.sha256(relative.encode()).hexdigest()[:20])
    try:
        forced = False
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True,
                specific_tags=list(FIELDS) + REFERENCE_TAGS)
        except InvalidDicomError:
            forced = True
            ds = pydicom.dcmread(path, force=True, stop_before_pixels=True,
                specific_tags=list(FIELDS) + REFERENCE_TAGS)
        valid = bool(ds.get('StudyInstanceUID')) and (
            (int(ds.get('Rows', 0)) > 0 and int(ds.get('Columns', 0)) > 0)
            or bool(ds.get('SOPClassUID')))
        if not valid:
            return dict(row, status='not_validated_dicom')
        row.update({v: str(ds.get(k, '')) for k, v in FIELDS.items()})
        row['transfer_syntax'] = str(ds.file_meta.get('TransferSyntaxUID', ''))
        row['forced_read'] = forced
        refs = []
        for elem in ds.iterall():
            if elem.keyword == 'ReferencedSOPInstanceUID':
                refs.append(str(elem.value))
        row['referenced_sop_uids'] = json.dumps(sorted(set(refs)))
        row['status'] = 'dicom'
    except Exception as exc:
        row.update(status='read_error', error=type(exc).__name__ + ': ' + str(exc)[:200])
    return row


def report_inventory(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.csv', '.txt'}:
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('cp1252')
        fields = {}
        for cells in csv.reader(text.splitlines()):
            if cells and cells[0].strip() in {'Study UID', 'Study Date and Time'}:
                fields[cells[0].strip()] = ','.join(cells[1:]).strip()
        rows.append(dict(report_path=str(path), report_sha256=hashlib.sha256(raw).hexdigest(),
            study_uid=fields.get('Study UID', ''), study_datetime=fields.get('Study Date and Time', '')))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dicom-root', type=Path, default=Path('E:/'))
    parser.add_argument('--reports', type=Path, default=Path('D:/DS/anonymized_reports'))
    parser.add_argument('--output', type=Path, default=Path('D:/us/output/dicom_prediction'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mapping', type=Path, default=Path('E:/anonymization_mapping_hashed.xlsx'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = report_inventory(args.reports)
    reports.to_parquet(args.output / 'strain_reports.parquet', index=False)
    db = sqlite3.connect(args.output / 'inventory.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,size INTEGER,mtime INTEGER,payload TEXT)')
    cached = {r[0]: (r[1], r[2]) for r in db.execute('SELECT path,size,mtime FROM files')}
    items, current_paths = [], set()
    roots = sorted(p for p in args.dicom_root.glob('echo_*') if p.is_dir())
    if not roots:
        raise RuntimeError('No echo_* data directories found; refusing an ambiguous drive-wide scan')
    started = time.time()
    for root in roots:
        for directory, _, names in os.walk(root):
            for name in names:
                p = Path(directory) / name
                s = p.stat()
                current_paths.add(str(p))
                if cached.get(str(p)) != (s.st_size, s.st_mtime_ns):
                    items.append((str(p), s.st_size, s.st_mtime_ns, str(p.relative_to(args.dicom_root))))
    print(json.dumps(dict(files=len(current_paths), headers_to_read=len(items), reports=len(reports))), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(inspect_file, items), 1):
            db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?)',
                (row['path'], row['size'], row['mtime'], json.dumps(row)))
            if n % 500 == 0:
                db.commit()
                print(f'Headers {n}/{len(items)}; elapsed {time.time()-started:.0f}s', flush=True)
    db.commit()
    rows = [json.loads(r[1]) for r in db.execute('SELECT path,payload FROM files') if r[0] in current_paths]
    db.close()
    inventory = pd.DataFrame(rows)
    for key in ['frames', 'rows', 'columns', 'frame_time_ms', 'cine_rate', 'display_fps']:
        inventory[key] = pd.to_numeric(inventory.get(key), errors='coerce')
    inventory['has_strain_report'] = inventory.get('study_uid', pd.Series(dtype=str)).isin(set(reports.study_uid) - {''})
    if args.mapping.exists():
        mapping = pd.read_excel(args.mapping, dtype=str)
        ids = set(mapping.PatID)
        inventory['hashed_patient'] = inventory.relative_path.map(
            lambda x: next((part for part in Path(x).parts if part in ids), ''))
        mapping['report_name'] = mapping.anonymized_file.map(lambda x: Path(x).name)
        report_map = reports.assign(report_name=reports.report_path.map(lambda x: Path(x).name)).merge(
            mapping[['report_name', 'PatID']], on='report_name', validate='one_to_one')
        report_map['date'] = report_map.study_datetime.str[:10].str.replace('-', '')
        image_map = inventory.loc[inventory.status.eq('dicom'), ['study_uid', 'study_date', 'hashed_patient']].drop_duplicates()
        exact = report_map.merge(image_map, on='study_uid')
        if not exact.PatID.eq(exact.hashed_patient).all():
            raise RuntimeError('Study UID linkage conflicts with mapped patient identity')
        date_audit = report_map.merge(image_map, left_on=['PatID','date'], right_on=['hashed_patient','study_date'],
            how='left', suffixes=('_report','_dicom'))
        date_audit['link_status'] = date_audit.apply(lambda r: 'exact_uid' if r.study_uid_report==r.study_uid_dicom
            else ('patient_date_candidate_requires_review' if pd.notna(r.study_uid_dicom) else 'unmatched'), axis=1)
        date_audit.to_parquet(args.output/'patient_date_linkage_audit.parquet', index=False)
    inventory.to_parquet(args.output / 'dicom_inventory.parquet', index=False)
    good = inventory[inventory.status.eq('dicom')].copy()
    studies = good.groupby('study_uid').agg(files=('file_id','size'), videos=('frames',lambda x: int((x>1).sum())),
        bytes=('size','sum'), machine=('machine',lambda x: '|'.join(sorted(set(x)))),
        referenced_objects=('referenced_sop_uids',lambda x: int((x!='[]').sum()))).reset_index()
    links = reports.groupby('study_uid').agg(strain_exports=('report_path','size')).reset_index().merge(studies, on='study_uid', how='left', validate='one_to_one')
    links.to_csv(args.output / 'study_linkage.csv', index=False)
    summary = dict(source_directories=[p.name for p in roots], files=len(inventory),
        status_counts=inventory.status.value_counts().to_dict(), dicom_studies=good.study_uid.nunique(),
        strain_exports=len(reports), strain_studies=reports.study_uid.nunique(),
        matched_strain_studies=int(links.files.notna().sum()), unmatched_strain_studies=int(links.files.isna().sum()),
        linked_files=int(good.has_strain_report.sum()), linked_videos=int((good.has_strain_report & good.frames.gt(1)).sum()),
        extensionless_dicoms=int(good.path.map(lambda x:not Path(x).suffix).sum()),
        missing_sop_uid=int(good.sop_uid.eq('').sum()),
        dicoms_with_source_references=int(good.referenced_sop_uids.ne('[]').sum()),
        elapsed_seconds=round(time.time()-started,1))
    (args.output/'inventory_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
