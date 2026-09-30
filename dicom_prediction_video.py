"""Spatial pilot, pretrained view classification and frozen video encoding.

Original DICOMs remain the canonical full-length videos. Spatial cache records
only geometry and five preview frames; encoding resamples the original using
explicit saved frame indices. No patient outcomes are used in selection.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
from PIL import Image

VIEWS = ['A2C','A3C','A4C','A5C','Apical_Doppler','Doppler_Parasternal_Long',
         'Doppler_Parasternal_Short','Parasternal_Long','Parasternal_Short','SSN','Subcostal']
VERSION = 'spatial-region-v1'
APPEARANCE_VERSION = 'monochrome-tint-v1'


def spatial_geometry(ds):
    h,w = int(ds.Rows),int(ds.Columns)
    regions = list(ds.get('SequenceOfUltrasoundRegions', []))
    tissue = [r for r in regions if int(r.get('RegionSpatialFormat',0))==1 and int(r.get('RegionDataType',0))==1]
    if not tissue:
        return dict(crop=[0,0,w,h], crop_source='unverified_full_frame', eligible_region=False,
            region_types=[int(r.get('RegionDataType',0)) for r in regions])
    def bounds(r):
        return [max(0,int(r.RegionLocationMinX0)),max(0,int(r.RegionLocationMinY0)),
                min(w,int(r.RegionLocationMaxX1)+1),min(h,int(r.RegionLocationMaxY1)+1)]
    boxes = [bounds(r) for r in tissue]
    boxes = [b for b in boxes if b[2]>b[0] and b[3]>b[1]]
    if not boxes:
        raise ValueError('Invalid ultrasound tissue region bounds')
    b = max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    return dict(crop=b,crop_source='dicom_tissue_region',eligible_region=True,
        region_types=[int(r.get('RegionDataType',0)) for r in regions],
        multiple_tissue_regions=len(boxes)>1)


def fit_square(rgb, size=224):
    h,w = rgb.shape[:2]
    scale = size/max(h,w)
    hh,ww = max(1,round(h*scale)),max(1,round(w*scale))
    resized=cv2.resize(rgb,(ww,hh),interpolation=cv2.INTER_AREA)
    out=np.zeros((size,size,3),dtype=np.uint8)
    y,x=(size-hh)//2,(size-ww)//2
    out[y:y+hh,x:x+ww]=resized
    return out


def doppler_pixels(frames):
    """Separate multihue flow overlays from monochrome amber tissue palettes.

    This is a screening heuristic, not a validated modality classifier.
    """
    hsv=np.stack([cv2.cvtColor(f,cv2.COLOR_RGB2HSV) for f in frames])
    visible=hsv[...,2]>20
    saturated=(hsv[...,1]>80)&visible
    hue=hsv[...,0]
    blue=((hue>80)&(hue<140)&saturated).sum()/max(visible.sum(),1)
    red=(((hue<15)|(hue>165))&saturated).sum()/max(visible.sum(),1)
    return bool(blue>.01 and red>.01)


def normalize_appearance(frames):
    """Remove a nearly uniform display tint, preserving the value channel.

    Requires a dominant hue across most visible tissue. Multihue red/blue flow
    is not normalized. This is deterministic appearance normalization, not a
    learned image enhancement or a substitute for modality review.
    """
    hsv=np.stack([cv2.cvtColor(f,cv2.COLOR_RGB2HSV) for f in frames])
    visible=hsv[...,2]>20
    colored=(hsv[...,1]>60)&visible
    fraction=float(colored.sum()/max(visible.sum(),1))
    if fraction<.6 or doppler_pixels(frames):return frames,'native_rgb'
    angles=hsv[...,0][colored].astype(float)*(2*np.pi/180)
    concentration=float(abs(np.mean(np.exp(1j*angles)))) if angles.size else 0.
    if concentration<.95:return frames,'native_rgb'
    return np.repeat(frames.max(-1)[...,None],3,axis=-1),APPEARANCE_VERSION


def decode_frames(path, indices, crop, buffered=False):
    unique=sorted(set(int(x) for x in indices))
    frames={}
    source=io.BytesIO(Path(path).read_bytes()) if buffered and Path(path).stat().st_size<512*1024**2 else path
    for index,frame in zip(unique,pydicom.pixels.iter_pixels(source,indices=unique)):
        if frame.ndim==2:frame=np.repeat(frame[...,None],3,axis=-1)
        if frame.ndim!=3 or frame.shape[-1]!=3:raise ValueError(f'Unsupported pixel shape {frame.shape}')
        if frame.dtype != np.uint8:raise ValueError(f'Unvalidated pixel depth {frame.dtype}')
        x0,y0,x1,y1=crop
        frames[index]=fit_square(frame[y0:y1,x0:x1])
    return np.stack([frames[int(i)] for i in indices])


def normalize(x, model):
    if model=='echoprime':
        mean,std=[29.110628,28.076836,29.096405],[47.989223,46.456997,47.20083]
    else:
        x=x/255.
        mean,std=[.485,.456,.406],[.229,.224,.225]
    shape=[1,3]+[1]*(x.ndim-2)
    return (x-torch.tensor(mean,device=x.device).reshape(shape))/torch.tensor(std,device=x.device).reshape(shape)


def select_pilot(inv, out, count):
    p=out/'pilot_studies.json'
    if p.exists():return set(json.loads(p.read_text()))
    # Stratify on acquisition period and machine, with stable hashes, no outcomes.
    s=inv[['study_uid','machine','relative_path']].drop_duplicates('study_uid').copy()
    s['period']=s.relative_path.str.split(r'[\\/]').str[0]
    s['hash']=s.study_uid.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    groups=[g.sort_values('hash').study_uid.tolist() for _,g in s.groupby(['period','machine'])]
    selected=[]
    while groups and len(selected)<count:
        for g in groups:
            if g and len(selected)<count:selected.append(g.pop(0))
        groups=[g for g in groups if g]
    p.write_text(json.dumps(selected,indent=2))
    return set(selected)


def crop_stage(args):
    out=args.output;inv=pd.read_parquet(out/'dicom_inventory.parquet')
    inv=inv[inv.status.eq('dicom') & inv.has_strain_report & inv.frames.gt(1)].copy()
    pilot=select_pilot(inv,out,args.pilot_studies)
    if not args.all:inv=inv[inv.study_uid.isin(pilot)]
    cache=out/'spatial_cache';cache.mkdir(exist_ok=True)
    started=time.time()
    def process(row):
        meta=cache/f'{row.file_id}.json'
        if meta.exists():
            try:saved=json.loads(meta.read_text())
            except (OSError,json.JSONDecodeError):saved={}
            if saved.get('source_size')==row.size and saved.get('source_mtime')==row.mtime and saved.get('version')==VERSION:
                return
        record=dict(file_id=row.file_id,study_uid=row.study_uid,path=row.path,
                    source_size=row.size,source_mtime=row.mtime,version=VERSION,
                    pilot=row.study_uid in pilot,frames=int(row.frames))
        try:
            ds=pydicom.dcmread(row.path,stop_before_pixels=True)
            geometry=spatial_geometry(ds)
            # Do not publish full-frame previews when the crop is unverified.
            if not geometry['eligible_region']:
                record.update(geometry,status='excluded_declared_color_doppler' if 2 in geometry['region_types'] else 'needs_crop_review')
            else:
                ix=np.unique(np.linspace(0,int(row.frames)-1,5).round().astype(int))
                frames=decode_frames(row.path,ix,geometry['crop'])
                gray=frames.mean(-1)
                chroma=(frames.max(-1).astype(float)-frames.min(-1))
                foreground=gray>12
                color_fraction=float(((chroma>30)&foreground).sum()/max(foreground.sum(),1))
                motion=float(np.abs(np.diff(gray,axis=0)).mean()) if len(frames)>1 else 0.
                record.update(geometry,status='ok',preview_indices=ix.tolist(),color_fraction=color_fraction,
                    foreground_fraction=float(foreground.mean()),motion_proxy=motion,
                    doppler_pixel_flag=doppler_pixels(frames),
                    sampled_pixel_hash=hashlib.sha256(frames.tobytes()).hexdigest())
                np.savez_compressed(cache/f'{row.file_id}.npz',frames=frames)
                Image.fromarray(np.concatenate(list(frames),axis=1)).save(cache/f'{row.file_id}.jpg',quality=85)
        except Exception as exc:record.update(status='decode_error',error=type(exc).__name__+': '+str(exc)[:200])
        temporary=meta.with_suffix('.partial');temporary.write_text(json.dumps(record,indent=2));temporary.replace(meta)
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n,_ in enumerate(pool.map(process,inv.itertuples()),1):
            if n%100==0:print(f'Spatial {n}/{len(inv)} elapsed {time.time()-started:.0f}s',flush=True)
    records=[json.loads(p.read_text()) for p in cache.glob('*.json')]
    pd.DataFrame(records).to_parquet(out/'spatial_manifest.parquet',index=False)
    print(pd.DataFrame(records).status.value_counts().to_dict(),flush=True)


def load_view_model(weights,device):
    model=torchvision.models.convnext_base(weights=None)
    model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,11)
    model.load_state_dict(torch.load(weights/'view_classifier.pt',map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False).to(device)


def classify_stage(args):
    out=args.output;device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_view_model(out/'weights',device)
    rows=pd.read_parquet(out/'spatial_manifest.parquet');rows=rows[rows.status.eq('ok')]
    cache=out/'view_cache';cache.mkdir(exist_ok=True)
    started=time.time()
    for n,row in enumerate(rows.itertuples(),1):
        path=cache/f'{row.file_id}.json'
        if path.exists():
            try:
                cached=json.loads(path.read_text())
                if cached.get('file_id')==row.file_id:
                    if cached.get('appearance_version')==APPEARANCE_VERSION:continue
                    if row.color_fraction<.6:
                        cached.update(appearance='native_rgb',appearance_version=APPEARANCE_VERSION)
                        temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(cached,indent=2));temporary.replace(path)
                        continue
            except (OSError,json.JSONDecodeError):pass
        frames=np.load(out/'spatial_cache'/f'{row.file_id}.npz')['frames']
        canonical,appearance=normalize_appearance(frames)
        x=torch.from_numpy(canonical).permute(0,3,1,2).float().to(device)
        with torch.inference_mode():
            probs=model(normalize(x,'echoprime')).softmax(-1).cpu().numpy()
        avg=probs.mean(0);label=int(avg.argmax());confidence=float(avg[label]);agreement=float((probs.argmax(1)==label).mean())
        accepted=confidence>=.75 and agreement>=.6
        r=dict(file_id=row.file_id,study_uid=row.study_uid,predicted_view=VIEWS[label],
            view=VIEWS[label] if accepted else 'uncertain',confidence=confidence,agreement=agreement,
            probabilities=avg.tolist(),frame_predictions=[VIEWS[x] for x in probs.argmax(1)],
            appearance=appearance,appearance_version=APPEARANCE_VERSION,
            is_bmode_candidate=(not doppler_pixels(frames)) and 2 not in list(row.region_types) and not row.multiple_tissue_regions,
            eligible_motion=row.motion_proxy>1 and row.foreground_fraction>.05)
        temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(r,indent=2));temporary.replace(path)
        if n%50==0:print(f'Views {n}/{len(rows)} elapsed {time.time()-started:.0f}s',flush=True)
    views=pd.DataFrame([json.loads(p.read_text()) for p in cache.glob('*.json')])
    views.to_parquet(out/'view_manifest.parquet',index=False)
    print(views.view.value_counts().to_dict(),flush=True)
    review_stage(args)


def review_stage(args):
    out=args.output
    views=pd.read_parquet(out/'view_manifest.parquet')
    rows=pd.read_parquet(out/'spatial_manifest.parquet').merge(views,on=['file_id','study_uid'])
    pilot=rows[rows.pilot].sort_values(['study_uid','predicted_view','confidence'],ascending=[True,True,False])
    cards=[]
    for study,group in pilot.groupby('study_uid',sort=False):
        sid=hashlib.sha256(study.encode()).hexdigest()[:10]
        cards.append(f'<h2>Study {sid} · {len(group)} videos</h2>')
        for row in group.itertuples():
            fid=row.file_id
            cards.append(f'<article><b>{html.escape(fid)} · {row.view} · {row.confidence:.2f}</b> '
                f'({row.frames} frames; color fraction {row.color_fraction:.3f}; agreement {row.agreement:.1f})'
                f'<br>Model appearance: {html.escape(str(getattr(row,"appearance","native_rgb")))}'
                f'<br><img loading="lazy" src="spatial_cache/{fid}.jpg" width="100%">'
                f'<p>Reviewer: <select data-id="{fid}" data-field="view"><option></option>'+''.join(f'<option>{v}</option>' for v in VIEWS+['Other','Uncertain'])+'</select> '
                f'Quality: <select data-id="{fid}" data-field="quality"><option></option><option>Usable</option><option>Borderline</option><option>Reject</option></select> '
                f'<input data-id="{fid}" data-field="notes" placeholder="Crop / foreshortening / modality notes"></article>')
    page='''<!doctype html><meta charset="utf-8"><title>DICOM view-selection pilot</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;background:#f2f4f8;color:#15213a}article{background:white;padding:16px;margin:12px 0;border-radius:12px}select,input,button{padding:8px}header{position:sticky;top:0;background:#f2f4f8;padding:12px;border-bottom:1px solid #aaa}h2{margin-top:44px}</style>
<header><h1>DICOM view-selection pilot</h1><p>Automated predictions, not clinician-validated. Five spatial previews per video. View confidence does not measure quality. Previews retain native color; flagged monochrome tints are normalized for model inference.</p><button onclick="download()">Download review JSON</button> <span id="count"></span></header>
'''+''.join(cards)+'''
<script>const key='dicom-view-review-v1';let data=JSON.parse(localStorage.getItem(key)||'{}');
function count(){document.querySelector('#count').textContent=Object.keys(data).length+' clips edited'}
document.querySelectorAll('[data-id]').forEach(e=>{e.value=data[e.dataset.id]?.[e.dataset.field]||'';e.onchange=()=>{data[e.dataset.id]??={};data[e.dataset.id][e.dataset.field]=e.value;localStorage.setItem(key,JSON.stringify(data));count()}});count();
function download(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='dicom_pilot_review.json';a.click()}
</script>'''
    (out/'pilot_review.html').write_text(page,encoding='utf-8')
    pilot[['file_id','study_uid','view','confidence','color_fraction','crop_source']].assign(reviewer_view='',reviewer_quality='',reviewer_notes='').to_csv(out/'pilot_review_template.csv',index=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['crop','classify','review'])
    p.add_argument('--output',type=Path,default=Path('D:/us/output/dicom_prediction'))
    p.add_argument('--pilot-studies',type=int,default=24)
    p.add_argument('--all',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(4)
    {'crop':crop_stage,'classify':classify_stage,'review':review_stage}[args.stage](args)


if __name__=='__main__':main()
