"""Read-only encoding/crop audit; all metadata/windows and a label-blind source-pixel sample."""
from __future__ import annotations
import hashlib,json,os,time,traceback
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image,ImageDraw
from dicom_prediction_video import normalize,normalize_appearance,doppler_pixels
from ichilov3_encode_selected import windows,timestamps,verify_artifacts
from ichilov3_prepare_selected import digest,padded_crop,_resize_frame
from ichilov_crop_dicoms import _largest_blob_mask
from ichilov3_crop_reviewer.server import tissue_box
from ichilov3_train_fusion import atomic,hashfile

CROPS=Path(r'D:\DS\ichilov3_stage2_padded_20260926')
EMBEDDINGS=Path(r'D:\DS\ichilov3_embeddings_reference_20260926')
OUTPUT=Path(r'D:\DS\ichilov3_preprocessing_audit_20260927')
ARTIFACTS=Path(r'D:\us\output\dicom_prediction')


def pixel_color(frames):
    if frames.ndim==3:frames=np.repeat(frames[...,None],3,axis=-1)
    _,appearance=normalize_appearance(frames)
    hsv=np.stack([cv2.cvtColor(f,cv2.COLOR_RGB2HSV) for f in frames])
    visible=hsv[...,2]>20;colored=(hsv[...,1]>60)&visible
    fraction=float(colored.sum()/max(1,visible.sum()))
    angles=hsv[...,0][colored].astype(float)*(2*np.pi/180)
    concentration=float(abs(np.mean(np.exp(1j*angles)))) if angles.size else 0.
    return dict(tint_candidate=appearance!='native_rgb',colored_fraction=fraction,hue_concentration=concentration,doppler_pixel_heuristic=doppler_pixels(frames))


def sample_records(records,colors,count=24):
    # Scanner/view/crop flag strata and color outliers, without physiology/outcome labels.
    ordered=sorted(records,key=lambda r:hashlib.sha256(r['file_id'].encode()).hexdigest())
    chosen=[];seen=set();strata=set()
    for r in ordered:
        key=(r['manufacturer'],r['view'],r['status']=='needs_review')
        if key not in strata:
            chosen.append(r);seen.add(r['file_id']);strata.add(key)
        if len(chosen)>=count:return chosen
    extra=sorted(ordered,key=lambda r:(not colors[r['file_id']]['tint_candidate'],r['frames'],r['file_id']))
    for r in extra:
        if r['file_id'] not in seen:chosen.append(r);seen.add(r['file_id'])
        if len(chosen)>=count:break
    return chosen


def numeric_normalization_check():
    x=torch.tensor([0.,128.,255.]).reshape(1,1,1,1,3).repeat(1,3,16,1,1)
    result={}
    for name,mean,std,scale in [('echoprime',[29.110628,28.076836,29.096405],[47.989223,46.456997,47.20083],1.),('panecho',[.485,.456,.406],[.229,.224,.225],255.)]:
        expected=(x/scale-torch.tensor(mean).reshape(1,3,1,1,1))/torch.tensor(std).reshape(1,3,1,1,1)
        actual=normalize(x,name);assert torch.allclose(actual,expected)
        result[name]=dict(matches_reference_constants=True,input_scale=scale,minimum=float(actual.min()),maximum=float(actual.max()))
    return result


def source_sample(r,out):
    ds=pydicom.dcmread(r['source_path'],stop_before_pixels=True)
    region=tissue_box(ds);indices=r['preview_frame_indices']
    assert str(ds.SOPInstanceUID)==r['expected_sop'] and str(ds.StudyInstanceUID)==r['expected_study']
    assert digest(Path(r['source_path']))==r['source_sha256']
    with np.load(r['crop_output']) as f:
        frames=f['frames'];assert np.array_equal(f['source_frame_indices'],np.arange(r['frames']))
        cached=frames[indices].copy();del frames
    raw=list(pydicom.pixels.iter_pixels(r['source_path'],indices=indices))
    tx0,ty0,tx1,ty1=region;y0,y1,x0,x1=r['bbox_y0_y1_x0_x1']
    dropped=[];outside=[];exact=[];original_previews=[]
    for source,stored in zip(raw,cached):
        if source.ndim==2:source=source[...,None]
        mask=np.any(source[...,:3]>5,axis=-1);keep=_largest_blob_mask(mask)
        tissue=np.zeros_like(mask);tissue[ty0:ty1,tx0:tx1]=True
        bbox=np.zeros_like(mask);bbox[y0:y1,x0:x1]=True
        denominator=max(1,int((mask&tissue).sum()))
        dropped.append(float((mask & ~keep & tissue).sum()/denominator))
        outside.append(float((mask & ~bbox & tissue).sum()/denominator))
        clean=source.copy();clean[mask & ~keep]=0
        recreated=_resize_frame(padded_crop(clean,(y0,y1,x0,x1)),518)
        if recreated.shape[-1]==1:recreated=recreated[...,0]
        exact.append(bool(np.array_equal(recreated,stored)))
        tissue_frame=source[ty0:ty1,tx0:tx1]
        if tissue_frame.shape[-1]==1:tissue_frame=np.repeat(tissue_frame,3,axis=-1)
        original_previews.append(tissue_frame)
    # Review-only image, no measurements or identities from the raw full-frame border.
    sheet=Image.new('RGB',(256*len(indices),560),'#101924');draw=ImageDraw.Draw(sheet)
    draw.text((8,5),'Original declared tissue region (no blob suppression)',fill='white')
    draw.text((8,285),'Current encoded crop (per-frame blob suppression)',fill='white')
    for i,(source,stored) in enumerate(zip(original_previews,cached)):
        h,w=source.shape[:2];size=max(h,w);pad=np.zeros((size,size,3),np.uint8);pad[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w]=source
        if stored.ndim==2:stored=np.repeat(stored[...,None],3,axis=-1)
        sheet.paste(Image.fromarray(cv2.resize(pad,(256,256))),(i*256,24))
        sheet.paste(Image.fromarray(cv2.resize(stored,(256,256))),(i*256,304))
    image=out/'comparisons'/(r['file_id']+'.jpg');sheet.save(image,quality=92)
    small=np.stack([cv2.resize(f,(224,224)) for f in original_previews])
    return dict(file_id=r['file_id'],patient=r['patient'],visit_date=r['visit_date'],view=r['view'],model=r['model'],
                preview_frames=indices,removed_inside_tissue_max=max(dropped),foreground_outside_bbox_inside_tissue_max=max(outside),
                cached_pixels_reproduced=all(exact),source_sha256_verified=True,comparison_image=str(image),**pixel_color(small))


def main():
    out=OUTPUT;out.mkdir(exist_ok=True,parents=True);(out/'comparisons').mkdir(exist_ok=True)
    started=time.time();cv2.setNumThreads(1);torch.set_num_threads(4)
    records=json.loads((CROPS/'full_manifest.json').read_text());manifests={m:json.loads((EMBEDDINGS/(m+'_manifest.json')).read_text()) for m in ['echoprime','panecho']}
    verify_artifacts(ARTIFACTS);normalization=numeric_normalization_check()
    atomic(out/'protocol.json',dict(clips=len(records),source_pixel_sample=24,sample_selection='fixed hash ordering within scanner/view/crop-flag strata; fill with color/short-clip outliers; no outcome labels',
        color_screen='all stored JPEG comparison previews, confirmed on source pixels in sample; heuristic only',
        crop_manifest_sha256=hashfile(CROPS/'full_manifest.json'),code_sha256=hashfile(Path(__file__))))
    colors={};headers=[];window_rows=[];errors=[]
    lookup={m:{r['file_id']:r for r in data} for m,data in manifests.items()}
    for n,r in enumerate(records,1):
        atomic(out/'status.json',dict(status='metadata_and_window_audit',completed=n-1,total=len(records),pid=os.getpid()))
        try:
            source=Path(r['source_path']);stat=source.stat()
            ds=pydicom.dcmread(source,stop_before_pixels=True)
            header_ok=stat.st_size==r['source_size'] and stat.st_mtime_ns==r['source_mtime_ns'] and str(ds.SOPInstanceUID)==r['expected_sop'] and str(ds.StudyInstanceUID)==r['expected_study']
            if not header_ok:raise ValueError('Source stat or UID changed')
            preview=np.asarray(Image.open(Path(r['crop_output']).with_suffix('.jpg')).convert('RGB'))
            color=pixel_color(preview[None]);colors[r['file_id']]=color
            headers.append(dict(file_id=r['file_id'],patient=r['patient'],visit_date=r['visit_date'],view=r['view'],manufacturer=r['manufacturer'],scanner=r['model'],frames=r['frames'],heart_rate=r.get('heart_rate'),
                frame_time_ms=r.get('frame_time_ms'),cine_rate=r.get('cine_rate'),removed_foreground_max=r['removed_foreground_max'],crop_flagged=r['status']=='needs_review',source_header_verified=True,**color))
            for model in manifests:
                meta=lookup[model][r['file_id']]
                assert meta['crop_metadata_sha256']==digest(Path(r['crop_output']).with_suffix('.json'))
                assert meta['source_sha256']==r['source_sha256']
                with np.load(meta['embedding_path']) as f:
                    emb=f['window_embeddings'];pooled=f['view_embedding'];idx=f['source_frame_indices'];times=f['sampled_times_seconds']
                assert np.isfinite(emb).all() and np.isfinite(pooled).all()
                assert np.array_equal(idx,windows(r['frames'],2 if model=='echoprime' else 1))
                expected_times,timing_source=timestamps(r,idx);assert np.allclose(times,expected_times,equal_nan=True)
                assert np.allclose(pooled,emb.mean(0))
                norms=np.linalg.norm(emb,axis=1);unit=emb/np.maximum(norms[:,None],1e-12);cos=unit@unit.T
                pair=cos[np.triu_indices(len(emb),1)]
                spans=times[:,-1]-times[:,0];hr=r.get('heart_rate')
                valid_hr=hr is not None and np.isfinite(hr) and hr>0
                window_rows.append(dict(file_id=r['file_id'],model=model,windows=len(emb),stride=meta['temporal_stride'],
                    edge_repeat=meta['short_clip_edge_repeat'],timing_source=timing_source,window_span_seconds_mean=float(np.nanmean(spans)) if np.isfinite(spans).any() else None,
                    estimated_beats_per_window_mean=float(np.mean(spans)*hr/60) if valid_hr and np.isfinite(spans).all() else None,
                    window_pair_cosine_mean=float(pair.mean()) if len(pair) else None,window_relative_dispersion=float(np.linalg.norm(emb-pooled,axis=1).mean()/max(np.linalg.norm(pooled),1e-12)),
                    pooled_embedding_norm=float(np.linalg.norm(pooled)),indices_verified=True,pooled_embedding_verified=True))
        except Exception as e:errors.append(dict(file_id=r['file_id'],error=f'{type(e).__name__}: {e}'))
        if n%100==0:print(f'Metadata/window audit {n}/{len(records)}',flush=True)
    header_frame=pd.DataFrame(headers);window_frame=pd.DataFrame(window_rows)
    header_frame.to_parquet(out/'clip_audit.parquet',index=False);window_frame.to_parquet(out/'window_audit.parquet',index=False)
    sample=sample_records([r for r in records if r['file_id'] in colors],colors)
    atomic(out/'sample_selection.json',[r['file_id'] for r in sample]);pixel_rows=[]
    for n,r in enumerate(sample,1):
        atomic(out/'status.json',dict(status='source_pixel_audit',completed=n-1,total=len(sample),file_id=r['file_id'],pid=os.getpid()))
        try:pixel_rows.append(source_sample(r,out))
        except Exception as e:errors.append(dict(file_id=r['file_id'],stage='source_pixel',error=f'{type(e).__name__}: {e}'))
        print(f'Source pixel audit {n}/{len(sample)}',flush=True)
    pixel_frame=pd.DataFrame(pixel_rows);pixel_frame.to_parquet(out/'source_pixel_audit.parquet',index=False)
    variance={}
    for model in manifests:
        with np.load(EMBEDDINGS/(model+'_visits.npz')) as f:
            x=f['embeddings'];variance[model]=dict(nonfinite_values=int((~np.isfinite(x)).sum()),constant_dimensions_per_view=[int((x[:,v].std(0)<1e-8).sum()) for v in range(3)],
                pooled_norm_median=float(np.median(np.linalg.norm(x,axis=-1))))
    summary=dict(clips=len(records),verified_metadata_clips=len(headers),errors=errors,normalization=normalization,
        crop_flagged=int(header_frame.crop_flagged.sum()),tint_candidates_in_preview=int(header_frame.tint_candidate.sum()),
        tint_candidates_note='JPEG-derived screening; not a whole-dataset exact-pixel finding',sample_clips=len(pixel_rows),
        sample_cached_pixels_reproduced=int(pixel_frame.cached_pixels_reproduced.sum()) if len(pixel_frame) else 0,
        sample_tint_candidates=int(pixel_frame.tint_candidate.sum()) if len(pixel_frame) else 0,
        sample_tissue_removal_gt_1pct=int((pixel_frame.removed_inside_tissue_max>.01).sum()) if len(pixel_frame) else 0,
        sample_tissue_foreground_outside_bbox_gt_1pct=int((pixel_frame.foreground_outside_bbox_inside_tissue_max>.01).sum()) if len(pixel_frame) else 0,
        embedding_variance=variance,windows={m:dict(edge_repeat_clips=int(g.edge_repeat.sum()),median_span_seconds=float(g.window_span_seconds_mean.median()),
            clips_with_heart_rate=int(g.estimated_beats_per_window_mean.notna().sum()),median_estimated_beats_per_window=float(g.estimated_beats_per_window_mean.median()),
            median_window_cosine=float(g.window_pair_cosine_mean.median())) for m,g in window_frame.groupby('model')},seconds=time.time()-started)
    atomic(out/'summary.json',summary)
    lines=['# Read-only preprocessing audit','',f"{len(headers)}/{len(records)} source metadata checks; {len(errors)} errors.",'',
        'Normalization matches the reference constants: EchoPrime raw 0-255 scale and PanEcho ImageNet 0-1 scale. Official checkpoint/architecture hashes were verified.',
        'Saved frame indices, sampled timestamps and mean pooling are checked for every encoded clip. This verifies implementation consistency, not optimal temporal sampling.',
        f"Monochrome-tint candidates in JPEG previews: {summary['tint_candidates_in_preview']}. The current encoder branch retains native RGB; the earlier dataset pipeline applied deterministic tint normalization.",
        f"Source-pixel sample: {len(pixel_rows)} clips; cached pixels reproduced for {summary['sample_cached_pixels_reproduced']}. Tissue-region foreground removal >1%: {summary['sample_tissue_removal_gt_1pct']}; foreground outside crop bbox >1%: {summary['sample_tissue_foreground_outside_bbox_gt_1pct']}.",
        'Foreground removal is not equivalent to loss of ventricular anatomy; annotations and speckle may contribute. Comparison images require visual inspection.',
        'Short native-rate windows need not cover a complete beat. Reference encoder settings use short windows, so partial beat coverage alone is not an implementation error.',
        'No source DICOMs, crops, embeddings or review decisions were changed. Color-normalized/unmasked-crop encoding trials are not included in this audit.']
    (out/'results.md').write_text('\n\n'.join(lines),encoding='utf-8')
    html=['<!doctype html><meta name="viewport" content="width=device-width"><title>Source crop comparison</title><style>body{background:#111;color:#eee;font:16px system-ui;margin:20px}img{width:100%;max-width:1280px}article{margin:32px 0}</style><h1>Original tissue versus encoded crop</h1><p>Fixed audit sample. Upper row: unmasked declared tissue region. Lower row: current crop. Foreground statistics do not identify anatomy.</p>']
    for r in pixel_rows:html.append(f"<article><h2>{r['patient']} · {r['visit_date']} · {r['view']}</h2><p>Removed within tissue: {r['removed_inside_tissue_max']:.1%}; outside bbox: {r['foreground_outside_bbox_inside_tissue_max']:.1%}</p><img src='comparisons/{r['file_id']}.jpg'></article>")
    (out/'comparison_review.html').write_text('\n'.join(html),encoding='utf-8')
    atomic(out/'status.json',dict(status='complete' if not errors else 'complete_with_errors',seconds=time.time()-started,summary=summary))
    atomic(out/'run_complete.json',dict(clips=len(headers),source_pixel_clips=len(pixel_rows),errors=len(errors),seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except Exception:
        OUTPUT.mkdir(exist_ok=True,parents=True);atomic(OUTPUT/'status.json',dict(status='error',error=traceback.format_exc()));traceback.print_exc();raise
