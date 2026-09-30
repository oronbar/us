"""Frozen EchoPrime/PanEcho encoders with checkpoint-strict loading and caching."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
from dicom_prediction_video import decode_frames, normalize, doppler_pixels


def window_indices(frames, stride):
    span=1+15*stride
    starts=sorted(set([0,max(0,frames-span)//2,max(0,frames-span)]))
    return [np.minimum(np.arange(16)*stride+start,frames-1).astype(int) for start in starts]


def load_encoder(name, out, device):
    if name=='echoprime':
        model=torchvision.models.video.mvit_v2_s(weights=None)
        model.head[-1]=torch.nn.Linear(model.head[-1].in_features,512)
        weights=torch.load(out/'weights/echo_prime_encoder.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(weights,strict=True)
    else:
        # Execute only the three architecture class definitions from the saved
        # official source. Preserve its forward semantics exactly. Avoid an
        # unnecessary ImageNet download because all weights are replaced.
        source=out/'vendor_sources/PanEcho_models.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'ImageEncoder','FrameTransformer','PositionalEncoding'}]
        namespace=dict(torch=torch,torchvision=torchvision,math=math)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        original=torchvision.models.convnext_tiny
        with patch.object(torchvision.models,'convnext_tiny',side_effect=lambda **kw:original(weights=None)):
            model=namespace['FrameTransformer']('convnext_tiny',8,4,0.,'mean',16)
        weights=torch.load(out/'weights/panecho.pt',map_location='cpu',weights_only=True)['weights']
        model.load_state_dict({k[len('encoder.'):]:v for k,v in weights.items() if k.startswith('encoder.')},strict=True)
    return model.eval().requires_grad_(False).to(device)


def selection(out,pilot_only=False):
    spatial=pd.read_parquet(out/'spatial_manifest.parquet')
    views=pd.read_parquet(out/'view_manifest.parquet')
    rows=spatial.merge(views,on=['file_id','study_uid'],validate='one_to_one')
    if pilot_only:rows=rows[rows.pilot]
    # Recompute screening from cached pixels so pilot caches from the initial
    # color-fraction heuristic cannot exclude amber-tinted B-mode images.
    rows=rows[rows.view.isin(['A2C','A3C','A4C','Parasternal_Long','Parasternal_Short']) & rows.eligible_motion].copy()
    def bmode(row):
        frames=np.load(out/'spatial_cache'/f'{row.file_id}.npz')['frames']
        return not doppler_pixels(frames) and 2 not in list(row.region_types) and not row.multiple_tissue_regions
    rows=rows[np.array([bmode(r) for r in rows.itertuples()],dtype=bool)]
    rows=rows[rows.frames.ge(32)].copy()
    rows=rows.sort_values(['study_uid','view','confidence','file_id'],ascending=[True,True,False,True])
    rows=rows.drop_duplicates(['study_uid','sampled_pixel_hash'])
    # Representative by view confidence, not claimed best diagnostic quality.
    rows=rows.groupby(['study_uid','view'],sort=False).head(2)
    rows.to_parquet(out/('pilot_selected_clips.parquet' if pilot_only else 'selected_clips.parquet'),index=False)
    return rows


def prepared_clips(row,out):
    cache=out/'encoding_clip_cache';cache.mkdir(exist_ok=True)
    target=cache/f'{row.file_id}.npz'
    if target.exists():
        old=np.load(target)
        old_appearance=str(old['appearance']) if 'appearance' in old else 'native_rgb'
        if int(old['source_size'])==row.source_size and int(old['source_mtime'])==row.source_mtime and np.array_equal(old['crop'],row.crop) and old_appearance==row.appearance:
            return {k:old[k] for k in ['echoprime','panecho','echoprime_indices','panecho_indices']}
    ep=window_indices(int(row.frames),2);pan=window_indices(int(row.frames),1)
    indices=np.concatenate(ep+pan)
    decoded=decode_frames(row.path,indices,row.crop,buffered=True)
    if row.appearance=='monochrome-tint-v1':decoded=np.repeat(decoded.max(-1)[...,None],3,axis=-1)
    split=len(ep)*16
    result=dict(echoprime=decoded[:split].reshape(len(ep),16,224,224,3),
                panecho=decoded[split:].reshape(len(pan),16,224,224,3),
                echoprime_indices=np.stack(ep),panecho_indices=np.stack(pan))
    temporary=target.with_suffix('.partial.npz')
    np.savez_compressed(temporary,**result,source_size=row.source_size,source_mtime=row.source_mtime,crop=row.crop,appearance=row.appearance)
    temporary.replace(target)
    return result


def prefetch_clips(pending,out):
    """Bounded two-clip prefetch; never accumulate the full cohort in RAM."""
    iterator=iter(pending)
    with ThreadPoolExecutor(max_workers=2) as pool:
        queue=deque()
        for _ in range(2):
            item=next(iterator,None)
            if item is not None:queue.append((item,pool.submit(prepared_clips,item[1],out)))
        while queue:
            item,future=queue.popleft()
            prepared=future.result()
            nxt=next(iterator,None)
            if nxt is not None:queue.append((nxt,pool.submit(prepared_clips,nxt[1],out)))
            yield item[0],item[1],prepared


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=Path('D:/us/output/dicom_prediction'))
    p.add_argument('--pilot-only',action='store_true');p.add_argument('--model',choices=['echoprime','panecho','both'],default='both')
    args=p.parse_args();out=args.output
    cv2.setNumThreads(1)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=selection(out,args.pilot_only)
    print(f'Selected {len(rows)} clips in {rows.study_uid.nunique()} studies',flush=True)
    for name in (['echoprime','panecho'] if args.model=='both' else [args.model]):
        model=load_encoder(name,out,device);cache=out/'embeddings'/name;cache.mkdir(parents=True,exist_ok=True)
        wp=out/'weights'/('echo_prime_encoder.pt' if name=='echoprime' else 'panecho.pt')
        with wp.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        started=time.time()
        pending=[]
        for n,row in enumerate(rows.itertuples(),1):
            target=cache/f'{row.file_id}.npz'
            if target.exists():
                old=np.load(target)
                old_appearance=str(old['appearance']) if 'appearance' in old else 'native_rgb'
                if str(old['checkpoint_sha256'])==sha and int(old['source_size'])==row.source_size and int(old['source_mtime'])==row.source_mtime and old_appearance==row.appearance and np.array_equal(old['crop'],row.crop):
                    continue
            pending.append((n,row))
        print(f'{name}: {len(pending)} remaining, {len(rows)-len(pending)} cached',flush=True)
        for n,row,prepared in prefetch_clips(pending,out):
            target=cache/f'{row.file_id}.npz'
            indices=prepared[name+'_indices'];clips=prepared[name]
            x=torch.from_numpy(clips.copy()).permute(0,4,1,2,3).float().to(device)
            with torch.inference_mode():emb=model(normalize(x,name)).cpu().numpy()
            if not np.isfinite(emb).all():raise ValueError('Non-finite encoder output')
            temporary=target.with_suffix('.partial.npz')
            np.savez_compressed(temporary,embeddings=emb,frame_indices=np.stack(indices),checkpoint_sha256=sha,
                source_size=row.source_size,source_mtime=row.source_mtime,crop=row.crop,appearance=row.appearance)
            temporary.replace(target)
            if n%25==0:print(f'{name} {n}/{len(rows)} elapsed {time.time()-started:.0f}s',flush=True)
        (cache/'provenance.json').write_text(json.dumps(dict(model=name,checkpoint_sha256=sha,frozen=True,
            temporal_windows='start,middle,end',stride=2 if name=='echoprime' else 1,
            short_clip_policy='exclude clips below 32 frames; edge-repeat within model windows',
            preprocessing='DICOM tissue rectangle, aspect-preserving square padding; detected monochrome tint normalized to value-channel grayscale; model-specific normalization',
            fp32=True,torch=torch.__version__,device=device),indent=2))
        print(f'{name} complete',flush=True);del model
        if device=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
