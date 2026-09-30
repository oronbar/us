"""Paired preview-gated uniform-tint ablation. Originals are read-only."""
from pathlib import Path
import json, os, subprocess, sys, time, traceback, shutil
import cv2
import numpy as np
import pandas as pd
import torch
from dicom_prediction_video import normalize_appearance, normalize
from dicom_prediction_encode import load_encoder
from ichilov3_encode_selected import verify_artifacts
from ichilov3_train_fusion import atomic, hashfile

ROOT=Path(r'D:\DS\ichilov3_tint_trial_20260927')
BASE=Path(r'D:\DS\ichilov3_embeddings_reference_20260926')
CROPS=Path(r'D:\DS\ichilov3_stage2_padded_20260926')
ART=Path(r'D:\us\output\dicom_prediction')
ALIGN=Path(r'D:\DS\ichilov3_aligned_20260927')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);torch.set_num_threads(4)
    records=json.loads((CROPS/'full_manifest.json').read_text())
    screen=pd.read_parquet(r'D:\DS\ichilov3_preprocessing_audit_20260927\clip_audit.parquet')
    candidates=set(screen.loc[screen.tint_candidate,'file_id'])
    protocol=dict(policy='JPEG preview gate then exact sampled-window uniform-hue test; max RGB channel replicated to grayscale before resize; preserve multihue Doppler',
        candidates=sorted(candidates),clips=len(records),crop_manifest_sha256=hashfile(CROPS/'full_manifest.json'),
        preview_screen_sha256=hashfile(Path(r'D:\DS\ichilov3_preprocessing_audit_20260927\clip_audit.parquet')),
        code_sha256=hashfile(Path(__file__)),sampling='reuse exact baseline source frame indices and timestamps',
        unchanged='reuse baseline embeddings exactly',evaluation='same nested patient folds and hyperparameter grids; exploratory full-cohort paired ablation')
    if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==protocol
    atomic(ROOT/'protocol.json',protocol);verify_artifacts(ART)
    started=time.time();rows=[]
    for name in ['echoprime','panecho']:
        model=load_encoder(name,ART,'cuda');folder=ROOT/name;folder.mkdir(exist_ok=True)
        manifests={r['file_id']:r for r in json.loads((BASE/(name+'_manifest.json')).read_text())}
        result=[]
        for n,r in enumerate(records,1):
            old=manifests[r['file_id']];new=dict(old);changed=0
            dest=folder/(r['file_id']+'.npz');meta=dest.with_suffix('.json')
            if meta.exists():new=json.loads(meta.read_text());changed=new['normalized_windows']
            else:
                with np.load(old['embedding_path']) as f:values={k:f[k].copy() for k in f.files}
                if r['file_id'] in candidates:
                    assert hashfile(Path(r['crop_output']).with_suffix('.json'))==old['crop_metadata_sha256']
                    with np.load(r['crop_output']) as f:frames=f['frames']
                    for i,idx in enumerate(values['source_frame_indices']):
                        window=frames[idx]
                        if window.ndim==3:window=np.repeat(window[...,None],3,-1)
                        clean,appearance=normalize_appearance(window)
                        if appearance=='native_rgb':continue
                        assert np.array_equal(clean.max(-1),window.max(-1))
                        resized=np.stack([cv2.resize(f,(224,224),interpolation=cv2.INTER_AREA) for f in clean])
                        tensor=torch.from_numpy(resized.copy()).permute(3,0,1,2).unsqueeze(0).float().cuda()
                        with torch.inference_mode():emb=model(normalize(tensor,name)).cpu().numpy()[0]
                        assert np.isfinite(emb).all()
                        values['window_embeddings'][i]=emb;changed+=1
                    del frames
                if changed:
                    values['view_embedding']=values['window_embeddings'].mean(0)
                    np.savez_compressed(dest,**values);new['embedding_path']=str(dest)
                new.update(normalized_windows=changed,tint_policy=protocol['policy'],baseline_embedding_path=old['embedding_path'])
                atomic(meta,new)
            result.append(new)
            rows.append(dict(model=name,file_id=r['file_id'],patient=r['patient'],visit_date=r['visit_date'],view=r['view'],manufacturer=r['manufacturer'],preview_candidate=r['file_id'] in candidates,normalized_windows=changed))
            if n%25==0 or n==len(records):
                atomic(ROOT/'status.json',dict(stage='encoding',model=name,completed=n,total=len(records),pid=os.getpid(),seconds=time.time()-started))
                print(f'{name} {n}/{len(records)} ({time.time()-started:.0f}s)',flush=True)
        atomic(ROOT/(name+'_manifest.json'),result)
        with np.load(Path(r'D:\DS\ichilov3_model_inputs_20260927')/(name+'_visits.npz')) as f:visits={k:f[k].copy() for k in f.files}
        lookup={(str(p),str(d)):i for i,(p,d) in enumerate(zip(visits['patient_ids'],visits['visit_dates']))}
        for r in result:
            if r['normalized_windows']:
                with np.load(r['embedding_path']) as f:visits['embeddings'][lookup[(r['patient'],r['visit_date'])],['A2C','A3C','A4C'].index(r['view'])]=f['view_embedding']
        assert np.isfinite(visits['embeddings']).all()
        np.savez_compressed(ROOT/(name+'_visits.npz'),**visits)
        del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_parquet(ROOT/'color_changes.parquet',index=False)
    paired=ROOT/'aligned';paired.mkdir(exist_ok=True)
    for filename in ['cohort.parquet','patient_folds.parquet','retained_baseline_oof.parquet']:shutil.copy2(ALIGN/filename,paired/filename)
    cohort=pd.read_parquet(ALIGN/'cohort.parquet')
    with np.load(ALIGN/'features.npz') as f:features={k:f[k].copy() for k in f.files}
    # Resolve each current visit through its exact alignment identity.
    visits=pd.read_parquet(ALIGN/'visit_alignment.parquet').set_index('visit_id')
    print('Cohort columns: '+str(list(cohort.columns)),flush=True)
    for name in ['echoprime','panecho']:
        with np.load(ROOT/(name+'_visits.npz')) as f:lookup={(str(p),str(d)):v.copy() for p,d,v in zip(f['patient_ids'],f['visit_dates'],f['embeddings'])}
        current_column='current_visit_id'
        if current_column not in cohort:current_column='visit_id'
        for i,r in cohort.iterrows():
            visit=visits.loc[r[current_column]]
            features[name+'_current'][i]=lookup[(str(visit.patient_id),str(visit.visit_date))]
    np.savez_compressed(paired/'features.npz',**features)
    atomic(ROOT/'status.json',dict(stage='physiology_evaluation',pid=os.getpid()))
    subprocess.run([sys.executable,r'D:\us\ichilov3_physiology_probe.py','--model-inputs',str(ROOT),'--output',str(ROOT/'physiology')],check=True)
    atomic(ROOT/'status.json',dict(stage='future_endpoint_evaluation',pid=os.getpid()))
    subprocess.run([sys.executable,r'D:\us\ichilov3_train_late_fusion.py','--input',str(paired),'--output',str(ROOT/'late_fusion')],check=True)
    atomic(ROOT/'run_complete.json',dict(seconds=time.time()-started,normalized_clips_by_model=pd.DataFrame(rows).groupby('model').normalized_windows.apply(lambda x:int((x>0).sum())).to_dict()))
    atomic(ROOT/'status.json',dict(stage='complete',seconds=time.time()-started,pid=os.getpid()))

if __name__=='__main__':
    try:main()
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True);atomic(ROOT/'status.json',dict(stage='error',error=traceback.format_exc()));raise
