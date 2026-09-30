"""Read-only native-color, estimated-heartbeat sampling ablation."""
from pathlib import Path
import json, os, sys, time, traceback, shutil, subprocess
import cv2
import numpy as np
import pandas as pd
import torch
from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize
from ichilov3_encode_selected import timestamps, verify_artifacts
from ichilov3_train_fusion import atomic, hashfile

ROOT=Path(r'D:\DS\ichilov3_temporal_trial_20260927')
BASE=Path(r'D:\DS\ichilov3_embeddings_reference_20260926')
CROPS=Path(r'D:\DS\ichilov3_stage2_padded_20260926')
ART=Path(r'D:\us\output\dicom_prediction')
ALIGN=Path(r'D:\DS\ichilov3_aligned_20260927')
MODELS=('echoprime','panecho')

def heartbeat_windows(record):
    """One HR-estimated period, with 16 nearest-time samples and <=4 starts.

    No ECG alignment, interpolation, wrapping or fabricated short-cine cycles.
    Invalid timing/HR and cines shorter than one period retain the baseline.
    """
    n=int(record['frames'])
    timeline,source=timestamps(record,np.arange(n))
    hr=record.get('heart_rate')
    if hr is None or not np.isfinite(hr) or not 30<=hr<=220:
        return None,dict(reason='unreliable_heart_rate',timing_source=source)
    if n<2 or not np.isfinite(timeline).all() or np.any(np.diff(timeline)<=0):
        return None,dict(reason='missing_or_nonmonotonic_timing',timing_source=source)
    period=60/float(hr);duration=float(timeline[-1])
    if duration<period:
        return None,dict(reason='cine_shorter_than_estimated_period',timing_source=source,estimated_period_seconds=period,duration_seconds=duration)
    latest=duration-period
    starts=np.unique(np.linspace(0,latest,4))
    targets=starts[:,None]+np.linspace(0,period,16)[None,:]
    right=np.clip(np.searchsorted(timeline,targets),0,n-1)
    left=np.maximum(0,right-1)
    indices=np.where(np.abs(timeline[left]-targets)<=np.abs(timeline[right]-targets),left,right)
    indices=np.unique(indices,axis=0)
    if np.any(np.diff(indices,axis=1)<=0):
        return None,dict(reason='insufficient_unique_frames_for_16_samples',timing_source=source,estimated_period_seconds=period)
    actual=timeline[indices];span=actual[:,-1]-actual[:,0]
    assert np.all(indices>=0) and np.all(indices<n)
    return indices,dict(reason='estimated_heartbeat_resampled',timing_source=source,estimated_period_seconds=period,
        duration_seconds=duration,windows=len(indices),mean_span_seconds=float(span.mean()),
        estimated_beats_per_window=float(span.mean()/period),maximum_frame_quantization_error_seconds=float(np.max(np.abs(actual-(timeline[indices[:,0],None]+np.linspace(0,period,16)[None,:])))))

def export_inputs(manifests):
    for name,records in manifests.items():
        with np.load(Path(r'D:\DS\ichilov3_model_inputs_20260927')/(name+'_visits.npz')) as f:visits={k:f[k].copy() for k in f.files}
        lookup={(str(p),str(d)):i for i,(p,d) in enumerate(zip(visits['patient_ids'],visits['visit_dates']))}
        for r in records:
            with np.load(r['embedding_path']) as f:
                visits['embeddings'][lookup[(r['patient'],r['visit_date'])],['A2C','A3C','A4C'].index(r['view'])]=f['view_embedding']
        assert np.isfinite(visits['embeddings']).all()
        np.savez_compressed(ROOT/(name+'_visits.npz'),**visits)
    paired=ROOT/'aligned';paired.mkdir(exist_ok=True)
    for filename in ['cohort.parquet','patient_folds.parquet','retained_baseline_oof.parquet']:shutil.copy2(ALIGN/filename,paired/filename)
    cohort=pd.read_parquet(ALIGN/'cohort.parquet')
    with np.load(ALIGN/'features.npz') as f:features={k:f[k].copy() for k in f.files}
    visits=pd.read_parquet(ALIGN/'visit_alignment.parquet').set_index('visit_id')
    for name in MODELS:
        with np.load(ROOT/(name+'_visits.npz')) as f:lookup={(str(p),str(d)):v.copy() for p,d,v in zip(f['patient_ids'],f['visit_dates'],f['embeddings'])}
        for i,r in cohort.iterrows():
            visit=visits.loc[r.current_visit_id]
            features[name+'_current'][i]=lookup[(str(visit.patient_id),str(visit.visit_date))]
    np.savez_compressed(paired/'features.npz',**features)
    return paired

def main():
    ROOT.mkdir(parents=True,exist_ok=True);started=time.time()
    records=json.loads((CROPS/'full_manifest.json').read_text())
    plans={r['file_id']:heartbeat_windows(r) for r in records}
    summary=pd.DataFrame([dict(file_id=r['file_id'],patient=r['patient'],visit_date=r['visit_date'],view=r['view'],manufacturer=r['manufacturer'],resampled=plans[r['file_id']][0] is not None,**plans[r['file_id']][1]) for r in records])
    protocol=dict(policy='16 nearest native frames spanning 60/HR seconds; up to four evenly distributed valid starts; average window embeddings',
        phase_alignment='none: estimated period from recorded HR, not ECG-confirmed cardiac cycles',
        fallback='retain exact original embeddings for cines without a reliable, sufficiently long timeline; no padding to fabricate a cycle',
        preprocessing='original padded crop, native color and reference encoder normalization',
        code_sha256=hashfile(Path(__file__)),crop_manifest_sha256=hashfile(CROPS/'full_manifest.json'),
        baseline_manifest_hashes={m:hashfile(BASE/(m+'_manifest.json')) for m in MODELS},
        models=list(MODELS),clips=len(records),sampling_counts=summary.reason.value_counts().to_dict(),
        primary='future endpoint AP of fixed 75% retained baseline + 25% EchoPrime; GLS probes diagnostic; all other comparisons exploratory',
        validation='unchanged 3x5 patient folds and nested inner hyperparameter selection; paired patient bootstrap; no outer-test policy tuning')
    if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==protocol
    atomic(ROOT/'protocol.json',protocol);summary.to_parquet(ROOT/'sampling_audit.parquet',index=False)
    atomic(ROOT/'status.json',dict(stage='loading_encoders',pid=os.getpid(),sampling_counts=protocol['sampling_counts']))
    cv2.setNumThreads(1);torch.set_num_threads(4);verify_artifacts(ART)
    models={m:load_encoder(m,ART,'cuda') for m in MODELS}
    base={m:{r['file_id']:r for r in json.loads((BASE/(m+'_manifest.json')).read_text())} for m in MODELS}
    manifests={m:[] for m in MODELS}
    print(f'Sampling preflight: {protocol["sampling_counts"]}',flush=True)
    for n,r in enumerate(records,1):
        indices,info=plans[r['file_id']]
        pending=[]
        for m in MODELS:
            folder=ROOT/m;folder.mkdir(exist_ok=True);dest=folder/(r['file_id']+'.npz');meta=dest.with_suffix('.json')
            if meta.exists():
                saved=json.loads(meta.read_text());assert saved['temporal_trial']==protocol['policy'];assert Path(saved['embedding_path']).exists()
                manifests[m].append(saved)
            elif indices is None:
                saved=dict(base[m][r['file_id']],temporal_trial=protocol['policy'],sampling_info=info,resampled=False)
                atomic(meta,saved);manifests[m].append(saved)
            else:pending.append((m,dest,meta))
        if pending:
            assert hashfile(Path(r['crop_output']).with_suffix('.json'))==base['echoprime'][r['file_id']]['crop_metadata_sha256']
            with np.load(r['crop_output']) as f:
                frames=f['frames'];assert np.array_equal(f['source_frame_indices'],np.arange(len(frames)))
            assert frames.dtype==np.uint8
            # Load each compressed cine once, then encode the same resized windows in both models.
            resized=[]
            for idx in indices:
                window=frames[idx]
                if window.ndim==3:window=np.repeat(window[...,None],3,-1)
                resized.append(np.stack([cv2.resize(f,(224,224),interpolation=cv2.INTER_AREA) for f in window]))
            del frames
            times,source=timestamps(r,indices)
            for m,dest,meta in pending:
                embeddings=[]
                for window in resized:
                    tensor=torch.from_numpy(window.copy()).permute(3,0,1,2).unsqueeze(0).float().cuda()
                    with torch.inference_mode():emb=models[m](normalize(tensor,m)).cpu().numpy()[0]
                    assert np.isfinite(emb).all();embeddings.append(emb)
                embeddings=np.asarray(embeddings,dtype=np.float32)
                temporary=dest.with_suffix('.partial.npz')
                np.savez_compressed(temporary,window_embeddings=embeddings,view_embedding=embeddings.mean(0),source_frame_indices=indices,sampled_times_seconds=times)
                temporary.replace(dest)
                saved=dict(base[m][r['file_id']],embedding_path=str(dest),temporal_trial=protocol['policy'],sampling_info=info,
                    resampled=True,windows=len(indices),temporal_stride=None,short_clip_edge_repeat=False,timing_source=source,
                    version='estimated-heartbeat-v1',baseline_embedding_path=base[m][r['file_id']]['embedding_path'])
                atomic(meta,saved);manifests[m].append(saved)
        if n%25==0 or n==len(records):
            for m in MODELS:atomic(ROOT/(m+'_manifest.json'),manifests[m])
            atomic(ROOT/'status.json',dict(stage='encoding_both_models',completed=n,total=len(records),pid=os.getpid(),seconds=time.time()-started))
            print(f'Both encoders {n}/{len(records)} ({time.time()-started:.0f}s)',flush=True)
    del models;torch.cuda.empty_cache()
    paired=export_inputs(manifests)
    atomic(ROOT/'status.json',dict(stage='physiology_evaluation',pid=os.getpid()))
    subprocess.run([sys.executable,r'D:\us\ichilov3_physiology_probe.py','--model-inputs',str(ROOT),'--output',str(ROOT/'physiology')],check=True)
    atomic(ROOT/'status.json',dict(stage='future_endpoint_evaluation',pid=os.getpid()))
    subprocess.run([sys.executable,r'D:\us\ichilov3_train_late_fusion.py','--input',str(paired),'--output',str(ROOT/'late_fusion')],check=True)
    atomic(ROOT/'status.json',dict(stage='paired_comparison',pid=os.getpid()))
    subprocess.run([sys.executable,r'D:\us\ichilov3_compare_temporal_trial.py'],check=True)
    atomic(ROOT/'run_complete.json',dict(seconds=time.time()-started,clips=len(records),sampling_counts=protocol['sampling_counts']))
    atomic(ROOT/'status.json',dict(stage='complete',seconds=time.time()-started,pid=os.getpid()))

if __name__=='__main__':
    try:main()
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True);atomic(ROOT/'status.json',dict(stage='error',error=traceback.format_exc()));raise
