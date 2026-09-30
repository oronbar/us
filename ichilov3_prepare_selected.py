"""Stage 2: read-only source audit and spatial processing of final selected clips.

Reuses the crop functions called by ichilov_pipeline4. No temporal sampling,
enhancement, DICOM rewriting or changes to source data. Outputs lossless numpy
arrays, geometry, source hashes, timing, QC and visual comparison sheets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openpyxl
import pydicom
from PIL import Image, ImageDraw

from ichilov_crop_dicoms import (
    _normalize_frames, _largest_blob_mask, _mask_bbox,
    _mask_longest_y_center, _crop_square, _resize_frame,
)

VERSION = "selected-crop-v1"
VIEWS = ("A2C", "A3C", "A4C")


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def registry(path):
    w = openpyxl.load_workbook(path, read_only=True, data_only=True)
    it = iter(w['Combined visits'].values)
    headers = next(it)
    visits = [dict(zip(headers, row)) for row in it if row[0]]
    w.close()
    tasks = []
    for row in visits:
        date = row['Visit date'].strftime('%Y-%m-%d')
        if row['Final status'] != 'complete':
            continue
        for view in VIEWS:
            path = Path(row[f'Final {view} DICOM path'])
            file_id = hashlib.sha256(f"{row['Patient ID']}|{date}|{view}|{path}".encode()).hexdigest()[:20]
            tasks.append(dict(patient=str(row['Patient ID']), visit_date=date, view=view,
                file_id=file_id, source_path=str(path),
                expected_sop=str(row[f'Final {view} SOP Instance UID'] or ''),
                expected_study=str(row[f'Final {view} Study Instance UID'] or ''),
                selection_source=row[f'Final {view} selection source'],
                strain_report=row['Strain TXT report']))
    assert len({t['file_id'] for t in tasks}) == len(tasks)
    return visits, tasks


def header(task):
    r = dict(task)
    try:
        p = Path(task['source_path'])
        stat = p.stat()
        ds = pydicom.dcmread(p, force=True, stop_before_pixels=True)
        n = int(ds.get('NumberOfFrames', 1))
        r.update(status='header_ok', source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
            frames=n, rows=int(ds.Rows), columns=int(ds.Columns),
            channels=int(ds.get('SamplesPerPixel', 1)), bits=int(ds.get('BitsAllocated',8)),
            manufacturer=str(ds.get('Manufacturer','')), model=str(ds.get('ManufacturerModelName','')),
            transfer_syntax=str(ds.file_meta.get('TransferSyntaxUID','')),
            frame_time_ms=float(ds.FrameTime) if ds.get('FrameTime') is not None else None,
            has_frame_time_vector=bool(ds.get('FrameTimeVector')),
            multiple_tissue_regions=sum(int(x.get('RegionSpatialFormat',0))==1 and int(x.get('RegionDataType',0))==1 for x in ds.get('SequenceOfUltrasoundRegions',[])))
        if task['expected_sop'] and task['expected_sop'] != str(ds.SOPInstanceUID):
            raise ValueError('Selected SOP UID does not match file')
        if task['expected_study'] and task['expected_study'] != str(ds.StudyInstanceUID):
            raise ValueError('Selected Study UID does not match file')
        if n < 2:
            raise ValueError('Selected file is not a multi-frame cine')
    except Exception as e:
        r.update(status='header_error', error=f'{type(e).__name__}: {e}')
    return r


def padded_crop(frame, box):
    y0,y1,x0,x1 = box
    patch=frame[y0:y1,x0:x1]
    side=max(patch.shape[:2])
    result=np.zeros((side,side,frame.shape[-1]),dtype=frame.dtype)
    yy,xx=(side-patch.shape[0])//2,(side-patch.shape[1])//2
    result[yy:yy+patch.shape[0],xx:xx+patch.shape[1]]=patch
    return result


def display(frame):
    if frame.shape[-1]==1:frame=np.repeat(frame,3,axis=-1)
    if frame.dtype!=np.uint8:
        # Preview only; saved arrays keep source dtype and values.
        maximum=np.iinfo(frame.dtype).max
        frame=np.clip(frame.astype(float)*255/maximum,0,255).astype(np.uint8)
    return Image.fromarray(frame)


def process(task, args, comparison=False):
    root=args.output/'clips'/task['file_id']
    metadata=root.with_suffix('.json')
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        p=Path(task['source_path']);stat=p.stat()
        if saved.get('version')==VERSION and saved.get('method')==args.method and saved.get('out_size')==args.size and saved.get('source_mtime_ns')==stat.st_mtime_ns and saved.get('source_size')==stat.st_size and root.with_suffix('.npz').exists():
            return saved
        raise ValueError(f'Existing cache differs: use a new output folder ({root})')
    r=dict(task,version=VERSION,method=args.method,out_size=args.size)
    try:
        path=Path(task['source_path']);before=path.stat()
        content=path.read_bytes()
        r['source_sha256']=hashlib.sha256(content).hexdigest()
        ds=pydicom.dcmread(io.BytesIO(content),force=True)
        frames,n,channels=_normalize_frames(ds.pixel_array,int(ds.get('SamplesPerPixel',1)))
        if n!=task['frames']:raise ValueError('Decoded frame count differs from header')
        if channels not in (1,3):raise ValueError(f'Unsupported channels: {channels}')
        if not np.issubdtype(frames.dtype,np.integer):raise ValueError('Unvalidated pixel dtype')
        # Exactly the mask and averaging strategy in pipeline4's crop script.
        accumulator=np.zeros(frames.shape[1:3],dtype=np.float32)
        masks=[]
        for frame in frames:
            raw=np.any(frame[...,:3]>5,axis=-1) if channels>1 else frame[...,0]>5
            keep=_largest_blob_mask(raw)
            accumulator+=keep
            masks.append(keep)
        average=accumulator/n
        binary=average>.5
        if not binary.any():binary=average>0
        if not binary.any():raise ValueError('No foreground found; crop needs manual review')
        box=_mask_bbox(binary);center=_mask_longest_y_center(binary)
        y0,y1,x0,x1=box;height=y1-y0
        legacy_left=round(center-height/2)
        retained=binary[y0:y1,max(0,legacy_left):min(frames.shape[2],legacy_left+height)].sum()/binary.sum()
        ix=np.unique(np.linspace(0,n-1,5).round().astype(int))
        output=[];preview=[];removed=[]
        for index,(frame,keep) in enumerate(zip(frames,masks)):
            raw=np.any(frame[...,:3]>5,axis=-1) if channels>1 else frame[...,0]>5
            drop=raw & ~keep
            removed.append(float(drop.sum()/max(1,raw.sum())))
            clean=frame.copy();clean[drop]=0
            crop=_crop_square(clean,box,center) if args.method=='legacy' else padded_crop(clean,box)
            output.append(_resize_frame(crop,args.size))
            if index in ix:
                if comparison:
                    # Show baseline versus width-preserving alternative, same masks.
                    left=display(_resize_frame(_crop_square(clean,box,center),256))
                    right=display(_resize_frame(padded_crop(clean,box),256))
                    panel=Image.new('RGB',(512,278));panel.paste(left,(0,22));panel.paste(right,(256,22))
                    draw=ImageDraw.Draw(panel);draw.text((5,5),'Pipeline4 original',fill='white');draw.text((261,5),'Full bbox + square padding',fill='white')
                    preview.append(panel)
                else:preview.append(display(_resize_frame(crop,256)))
        stack=np.stack(output)
        if channels==1:stack=stack[...,0]
        flags=[]
        if retained<.99 and args.method=='legacy':flags.append('legacy_crop_truncates_foreground')
        if max(removed)>.10:flags.append('disconnected_foreground_removed_gt10pct')
        if binary.mean()<.05:flags.append('small_foreground')
        if task['multiple_tissue_regions']>1:flags.append('multiple_tissue_regions')
        if task['frame_time_ms'] is None and not task['has_frame_time_vector']:flags.append('missing_frame_timing')
        if ds.get('PhotometricInterpretation')=='MONOCHROME1':flags.append('monochrome1_requires_encoder_inversion')
        r.update(status='needs_review' if flags else 'processed',qc_flags=flags,
            bbox_y0_y1_x0_x1=list(box),legacy_x_center=center,legacy_foreground_retained=float(retained),
            foreground_fraction=float(binary.mean()),removed_foreground_max=max(removed),
            original_shape=list(frames.shape),output_shape=list(stack.shape),dtype=str(stack.dtype),
            preview_frame_indices=ix.tolist(),source_sop=str(ds.SOPInstanceUID),source_study=str(ds.StudyInstanceUID),
            photometric_interpretation=str(ds.PhotometricInterpretation),
            frame_time_vector_ms=[float(v) for v in ds.get('FrameTimeVector',[])],
            cine_rate=float(ds.CineRate) if ds.get('CineRate') is not None else None,
            heart_rate=float(ds.HeartRate) if ds.get('HeartRate') is not None else None,
            temporal_processing='none; all original frames in source order',
            crop_output=str(root.with_suffix('.npz')))
        root.parent.mkdir(parents=True,exist_ok=True)
        temporary=root.with_suffix('.partial.npz')
        np.savez_compressed(temporary,frames=stack,source_frame_indices=np.arange(n,dtype=np.int32))
        temporary.replace(root.with_suffix('.npz'))
        # Reload lossless cache and check every frame index and array value.
        with np.load(root.with_suffix('.npz')) as saved:
            assert np.array_equal(saved['frames'],stack)
            assert np.array_equal(saved['source_frame_indices'],np.arange(n))
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during processing')
        r['source_stat_verified_unchanged']=True
        sheet=Image.new('RGB',(preview[0].width*len(preview),preview[0].height))
        for i,image in enumerate(preview):sheet.paste(image,(i*image.width,0))
        sheet.save(root.with_suffix('.jpg'),quality=90)
    except Exception as e:
        r.update(status='processing_error',error=f'{type(e).__name__}: {e}')
    metadata.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(metadata,r)
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(r'C:\Users\Oron\Downloads\ichilov3_combined_full_table.xlsx'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['inventory','pilot','full'],default='pilot')
    parser.add_argument('--method',choices=['legacy','padded'],default='legacy')
    parser.add_argument('--size',type=int,default=518)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot-clips',type=int,default=12)
    args=parser.parse_args()
    args.output=args.output.resolve()
    source_root=Path(r'E:\ichilov3').resolve()
    if args.output==source_root or source_root in args.output.parents:raise ValueError('Output must be separate from source dataset')
    args.output.mkdir(parents=True,exist_ok=True)
    visits,tasks=registry(args.input)
    for task in tasks:
        if source_root not in Path(task['source_path']).resolve().parents:raise ValueError('Unexpected source root')
    fingerprint=digest(args.input)
    inventory_path=args.output/'inventory.json'
    if inventory_path.exists():
        inv=json.loads(inventory_path.read_text())
        if inv['registry_sha256']!=fingerprint:raise ValueError('Registry changed: use a new output directory')
        tasks=inv['clips']
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks=list(pool.map(header,tasks))
        inv=dict(registry=str(args.input),registry_sha256=fingerprint,
            visits=len(visits),complete_visits=sum(r['Final status']=='complete' for r in visits),
            excluded_visits=[{'patient':str(r['Patient ID']),'date':r['Visit date'].strftime('%Y-%m-%d'),'status':r['Final status']} for r in visits if r['Final status']!='complete'],
            clips=tasks)
        atomic_json(inventory_path,inv)
    print('Inventory:',dict(Counter(t['status'] for t in tasks)),flush=True)
    if args.stage=='inventory':return
    eligible=[t for t in tasks if t['status']=='header_ok']
    if args.stage=='pilot':
        # Balanced deterministic sample across machines and views; no outcomes.
        groups={}
        for t in eligible:groups.setdefault((t['manufacturer'],t['model'],t['view']),[]).append(t)
        eligible=[]
        for group in groups.values():
            eligible.append(sorted(group,key=lambda t:t['file_id'])[0])
        eligible=eligible[:args.pilot_clips]
        atomic_json(args.output/'pilot_selection.json',eligible)
    started=time.time();records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(lambda t:process(t,args,args.stage=='pilot'),eligible),1):
            records.append(r)
            if i%10==0 or i==len(eligible):
                print(f'{i}/{len(eligible)} ({time.time()-started:.0f}s): {dict(Counter(x["status"] for x in records))}',flush=True)
                atomic_json(args.output/f'{args.stage}_manifest.json',records)
    atomic_json(args.output/f'{args.stage}_manifest.json',records)
    with (args.output/f'{args.stage}_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['patient','visit_date','view','source_path','crop_output','status','qc_flags','source_sha256','frames','frame_time_ms','selection_source','strain_report','error']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for r in records:writer.writerow({**r,'qc_flags':'; '.join(r.get('qc_flags',[]))})
    links=[]
    for r in records:
        name=r['file_id']+'.jpg'
        if (args.output/'clips'/name).exists():
            links.append(f'<p>{r["patient"]} / {r["visit_date"]} / {r["view"]} — {r["status"]}: {", ".join(r.get("qc_flags",[]))}</p><img width="100%" src="clips/{name}">')
    (args.output/f'{args.stage}_review.html').write_text('<html><meta charset="utf-8"><title>Stage 2 crop review</title><body style="background:#142033;color:white;font:16px Arial"><h1>Selected clip crop review</h1>'+''.join(links)+'</body></html>',encoding='utf-8')
    assert digest(args.input)==fingerprint
    print('Registry hash verified unchanged; sources were opened read-only.',flush=True)


if __name__=='__main__':main()
