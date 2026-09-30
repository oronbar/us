"""Encode the completed stage-2 cache using frozen pretrained video backbones.

Four deterministic native-rate windows per selected view; EchoPrime stride=2,
PanEcho stride=1. Full cropped videos are never rewritten or duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize
from ichilov3_prepare_selected import atomic_json, digest

VERSION='selected-reference-windows-v1'


def windows(n, stride, count=4):
    if n<1:raise ValueError('Empty cine')
    span=1+15*stride
    starts=np.unique(np.linspace(0,max(0,n-span),count).round().astype(int))
    return np.stack([np.minimum(start+np.arange(16)*stride,n-1) for start in starts])


def timestamps(record, indices):
    n=record['frames'];vector=record.get('frame_time_vector_ms',[])
    if len(vector)==n:
        timeline=np.cumsum(np.asarray(vector,dtype=float));timeline-=timeline[0]
        return timeline[indices]/1000,'FrameTimeVector'
    ft=record.get('frame_time_ms')
    if ft and ft>0:return indices*float(ft)/1000,'FrameTime'
    rate=record.get('cine_rate')
    if rate and rate>0:return indices/float(rate),'CineRate fallback'
    return np.full(indices.shape,np.nan),'unavailable'


def ready(args):
    inv=json.loads((args.crops/'inventory.json').read_text())
    expected={r['file_id'] for r in inv['clips'] if r['status']=='header_ok'}
    while True:
        if args.limit:
            records=[json.loads(p.read_text()) for p in sorted((args.crops/'clips').glob('*.json'))]
            return records[:args.limit]
        p=args.crops/'full_manifest.json'
        if p.exists():
            try:records=json.loads(p.read_text())
            except (OSError,json.JSONDecodeError):records=[]
            if {r['file_id'] for r in records}==expected:
                assert digest(inv['registry'])==inv['registry_sha256'],'Selection registry changed'
                return records
        if not args.wait_for_crops:raise RuntimeError('Full crop stage is not complete')
        # Detect a stopped crop worker instead of waiting indefinitely.
        pid_file=args.crops/'run.pid'
        if pid_file.exists():
            import psutil
            pid=int(pid_file.read_text().strip())
            if not psutil.pid_exists(pid):raise RuntimeError('Crop worker stopped before full manifest completed')
        time.sleep(30)


def verify_artifacts(root):
    provenance=json.loads((root/'source_provenance.json').read_text())
    path=root/'vendor_sources/PanEcho_models.py'
    assert hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()==provenance['PanEcho_models.py']['text_sha256'],'Architecture source mismatch'
    expected=json.loads((root/'weights_manifest.json').read_text())['sha256']['panecho.pt']
    assert digest(root/'weights/panecho.pt')==expected,'PanEcho checkpoint mismatch'
    with zipfile.ZipFile(root/'weights/model_data.zip') as z:
        member=next(n for n in z.namelist() if n.endswith('/weights/echo_prime_encoder.pt'))
        with z.open(member) as f:expected=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest(root/'weights/echo_prime_encoder.pt')==expected,'EchoPrime checkpoint differs from downloaded archive'


def encode(record,name,model,args,sha,device):
    folder=args.output/name;folder.mkdir(parents=True,exist_ok=True)
    target=folder/(record['file_id']+'.npz')
    meta=target.with_suffix('.json')
    if target.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old['version']==VERSION and old['checkpoint_sha256']==sha and old['source_sha256']==record['source_sha256'] and old['crop_metadata_sha256']==digest(Path(record['crop_output']).with_suffix('.json')):
            return old
        raise ValueError('Existing embedding differs; use a new output directory')
    with np.load(record['crop_output']) as cache:
        frames=cache['frames'];source_indices=cache['source_frame_indices']
    assert len(frames)==record['frames']
    assert np.array_equal(source_indices,np.arange(len(frames))), 'Unexpected temporal sampling in crop cache'
    if frames.dtype!=np.uint8:raise ValueError('Pixel depth needs validated model scaling')
    if record.get('photometric_interpretation')=='MONOCHROME1':raise ValueError('MONOCHROME1 needs validated inversion before encoding')
    indices=windows(len(frames),2 if name=='echoprime' else 1)
    embeddings=[]
    for sampled in indices:
        window=frames[sampled]
        if window.ndim==3:window=np.repeat(window[...,None],3,axis=-1)
        resized=np.stack([cv2.resize(f,(224,224),interpolation=cv2.INTER_AREA) for f in window])
        tensor=torch.from_numpy(resized.copy()).permute(3,0,1,2).unsqueeze(0).float().to(device)
        with torch.inference_mode():emb=model(normalize(tensor,name)).cpu().numpy()[0]
        if not np.isfinite(emb).all():raise ValueError('Nonfinite embedding')
        embeddings.append(emb)
    embeddings=np.asarray(embeddings,dtype=np.float32)
    assert embeddings.shape==(len(indices),512 if name=='echoprime' else 768)
    pooled=embeddings.mean(axis=0)
    sampled_times,timing_source=timestamps(record,indices)
    temporary=target.with_suffix('.partial.npz')
    np.savez_compressed(temporary,window_embeddings=embeddings,view_embedding=pooled,
        source_frame_indices=indices,sampled_times_seconds=sampled_times)
    temporary.replace(target)
    with np.load(target) as check:
        assert np.array_equal(check['window_embeddings'],embeddings)
        assert np.array_equal(check['source_frame_indices'],indices)
    result={k:record.get(k) for k in ['patient','visit_date','view','file_id','source_path','source_sha256','qc_flags','strain_report','selection_source']}
    result.update(version=VERSION,model=name,checkpoint_sha256=sha,embedding_path=str(target),
        crop_output=record['crop_output'],crop_method=record['method'],crop_status=record['status'],
        crop_metadata_sha256=digest(Path(record['crop_output']).with_suffix('.json')),
        status='encoded',training_crop_review_required=record['status']=='needs_review',
        windows=len(indices),short_clip_edge_repeat=record['frames']<(31 if name=='echoprime' else 16),
        temporal_stride=2 if name=='echoprime' else 1,timing_source=timing_source,
        input_shape=[1,3,16,224,224],embedding_dimensions=embeddings.shape[1])
    atomic_json(meta,result)
    return result


def export_visits(name,records,args):
    keys=sorted({(r['patient'],r['visit_date']) for r in records})
    dimension=512 if name=='echoprime' else 768
    array=np.full((len(keys),3,dimension),np.nan,dtype=np.float32)
    available=np.zeros((len(keys),3),dtype=bool);review=np.zeros_like(available)
    index={k:i for i,k in enumerate(keys)}
    for r in records:
        if r['status']!='encoded':continue
        i=index[(r['patient'],r['visit_date'])];v=['A2C','A3C','A4C'].index(r['view'])
        with np.load(r['embedding_path']) as f:array[i,v]=f['view_embedding']
        available[i,v]=True;review[i,v]=r['training_crop_review_required']
    temp=args.output/(name+'_visits.partial.npz')
    np.savez_compressed(temp,embeddings=array,patient_ids=np.asarray([k[0] for k in keys]),
        visit_dates=np.asarray([k[1] for k in keys]),views=np.asarray(['A2C','A3C','A4C']),
        available=available,crop_review_required=review)
    temp.replace(args.output/(name+'_visits.npz'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops',type=Path,default=Path(r'D:\DS\ichilov3_stage2_padded_20260926'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,default=Path(r'D:\us\output\dicom_prediction'))
    p.add_argument('--wait-for-crops',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.output.resolve()==args.crops.resolve():raise ValueError('Embedding output must be separate')
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    verify_artifacts(args.artifacts)
    print(f'Checkpoint hashes verified. Device: {device}. Waiting for completed crop manifest.',flush=True)
    records=ready(args)
    print(f'Crop stage ready: {len(records)} clips; {dict(Counter(r["status"] for r in records))}',flush=True)
    started=time.time()
    for name in ['echoprime','panecho']:
        checkpoint=args.artifacts/'weights'/('echo_prime_encoder.pt' if name=='echoprime' else 'panecho.pt')
        sha=digest(checkpoint);model=load_encoder(name,args.artifacts,device)
        results=[]
        for n,r in enumerate(records,1):
            try:
                if r['status'] not in ['processed','needs_review']:raise ValueError('Crop did not process successfully')
                result=encode(r,name,model,args,sha,device)
            except Exception as e:
                result={k:r.get(k) for k in ['patient','visit_date','view','file_id','qc_flags']}
                result.update(status='encoding_error',error=f'{type(e).__name__}: {e}')
            results.append(result)
            if n%25==0 or n==len(records):
                atomic_json(args.output/(name+'_manifest.json'),results)
                print(f'{name} {n}/{len(records)} ({time.time()-started:.0f}s): {dict(Counter(x["status"] for x in results))}',flush=True)
        export_visits(name,results,args)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    atomic_json(args.output/'run_complete.json',dict(version=VERSION,clips=len(records),
        finished_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        torch=torch.__version__,device=device,frozen=True,dtype='float32',
        preprocessing='518px padded pipeline4 crop resized to 224px, native RGB, encoder-specific normalization',
        windows='up to four evenly distributed unique start positions; 16 frames per window',
        cardiac_cycle_sampling='not applied; reliable consecutive cycle boundaries are not yet established'))
    print('Both encoders finished. Crop review flags retained in visit arrays.',flush=True)


if __name__=='__main__':main()
