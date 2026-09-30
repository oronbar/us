"""Read-only alignment of original Ichilov3 video embeddings and existing strain endpoint.

No imputation, scaling, PCA, feature selection, or training is fitted here.
Visit matching requires patient + date + exact StudyInstanceUID agreement.
Future-visit measurements remain in the audit/label table, never feature arrays.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEFAULT_OUTPUT=Path(r'D:\DS\ichilov3_aligned_20260927')
MODELS=('echoprime','panecho')
VIEWS=('A2C','A3C','A4C')
TASK='mid_first_rel15'


def align_visits(visits, records):
    visits=visits.copy()
    visits['visit_date']=pd.to_datetime(visits.study_datetime).dt.strftime('%Y-%m-%d')
    if visits.duplicated(['patient_id','visit_date']).any():
        raise ValueError('Ambiguous multiple strain visits on the same patient/date')
    clips=pd.DataFrame(records)
    if clips.duplicated(['patient','visit_date','view']).any():
        raise ValueError('Duplicate source cine for a patient/date/view')
    if (clips.groupby('expected_study').patient.nunique()>1).any():
        raise ValueError('DICOM Study UID appears under multiple patients')
    groups={(p,d):g for (p,d),g in clips.groupby(['patient','visit_date'])}
    rows=[]
    for r in visits.itertuples():
        g=groups.get((r.patient_id,r.visit_date))
        status='missing_video_triplet'
        if g is not None:
            if set(g.view)!=set(VIEWS):raise ValueError('Incomplete encoded view triplet')
            status='exact_patient_date_study_uid' if set(g.expected_study)=={r.study_uid} else 'study_uid_mismatch'
        row=dict(visit_id=r.visit_id,patient_id=r.patient_id,visit_date=r.visit_date,
                 study_uid=r.study_uid,visit_order=int(r.visit_order),match_status=status)
        for view in VIEWS:
            chosen=g[g.view.eq(view)].iloc[0] if g is not None else None
            for field in ['file_id','source_path','expected_sop','expected_study','selection_source','crop_output','qc_flags']:
                row[f'{view}_{field}']=chosen[field] if chosen is not None else None
        rows.append(row)
    audit=pd.DataFrame(rows)
    strain_keys=set(zip(visits.patient_id,visits.visit_date))
    extra=clips[~pd.MultiIndex.from_frame(clips[['patient','visit_date']]).isin(strain_keys)].copy()
    return visits,audit,extra


def strain_tensors(curves):
    selected=curves[curves.curve_family.eq('longitudinal_strain') & curves.layer.isin(['endo','mid']) & curves.segment_number.notna()]
    result={}
    for visit,g in selected.groupby('visit_id'):
        tensor=np.full((18,2,96),np.nan,np.float32)
        for (layer,segment),series in g.groupby(['layer','segment_number']):
            values=[np.asarray(x,np.float32) for x in series.resampled_values]
            if any(x.shape!=(96,) or not np.isfinite(x).all() for x in values):
                raise ValueError(f'Invalid strain samples: {visit}/{layer}/{segment}')
            if not 1<=int(segment)<=18:raise ValueError('Invalid segment number')
            tensor[int(segment)-1,0 if layer=='endo' else 1]=np.mean(values,axis=0)
        if np.isfinite(tensor).all():result[str(visit)]=tensor
    return result


def history_ids(visits,current_id):
    current=visits.loc[current_id]
    same=visits[visits.patient_id.eq(current.patient_id)].sort_values('visit_order')
    prior=same[same.visit_order.lt(current.visit_order)]
    if prior.empty:return None,None
    # True immediately preceding strain visit; never nearest available video.
    return str(same.index[0]),str(prior.index[-1])


def historical_video(current,prior):
    if prior is None:return np.zeros_like(current),np.zeros(len(current),bool)
    return current-prior,np.ones(len(current),bool)


def checksum(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    paths={
        'visits':ROOT/'amber_full_105_preprocessed/Ichilov_july_visits.parquet',
        'curves':ROOT/'amber_full_105_preprocessed/Ichilov_july_dataset.parquet',
        'transitions':ROOT/'cardiotoxicity_next_visit_gpu_results/next_visit_transitions.parquet',
        'feature_manifest':ROOT/'cardiotoxicity_next_visit_gpu_results/feature_manifest.csv',
        'patient_folds':ROOT/'cardiotoxicity_cnn_length_ablation_results/cnn_length_ablation_patient_folds.csv',
        'retained_oof':ROOT/'cardiotoxicity_timeseries_round4_results/round4_oof_predictions.parquet',
        'active_crop_manifest':Path(r'D:\DS\ichilov3_crop_review\reviewed_crop_manifest.json'),
        **{m:Path(r'D:\DS\ichilov3_model_inputs_20260927')/(m+'_visits.npz') for m in MODELS},
    }
    records=json.loads(paths['active_crop_manifest'].read_text())
    if not all(r['training_eligible_crop'] and not r.get('alternative_source') and not (r.get('review_decision') or {}).get('revision') for r in records):
        raise ValueError('Expected user-accepted original crops; rebuild inputs if selections changed')
    visits,audit,extra=align_visits(pd.read_parquet(paths['visits']),records)
    tensors=strain_tensors(pd.read_parquet(paths['curves'],columns=['visit_id','curve_family','layer','segment_number','resampled_values']))
    audit['complete_strain_curves']=audit.visit_id.isin(tensors)
    audit.to_parquet(out/'visit_alignment.parquet',index=False)
    extra.to_parquet(out/'video_only_clips.parquet',index=False)
    audit[~audit.match_status.eq('exact_patient_date_study_uid')].to_parquet(out/'unmatched_strain_visits.parquet',index=False)
    by_id=visits.set_index('visit_id');matched=audit[audit.match_status.eq('exact_patient_date_study_uid')]
    video_index={r.visit_id:(r.patient_id,r.visit_date) for r in matched.itertuples()}
    videos={}
    for model in MODELS:
        with np.load(paths[model]) as src:
            if tuple(src['views'])!=VIEWS:raise ValueError('Unexpected video view order')
            if len(set(zip(src['patient_ids'],src['visit_dates'])))!=len(src['patient_ids']):raise ValueError('Duplicate video visits')
            if not src['available'].all() or not src['training_eligible_crop'].all():raise ValueError('Incomplete/unaccepted video triplet')
            raw={tuple(k):x for k,x in zip(zip(src['patient_ids'],src['visit_dates']),src['embeddings'])}
            videos[model]={visit:raw[key] for visit,key in video_index.items()}
            if not all(np.isfinite(x).all() for x in videos[model].values()):raise ValueError('Nonfinite video embeddings')
    t=pd.read_parquet(paths['transitions'])
    if not t.transition_id.is_unique:raise ValueError('Duplicate prediction transition')
    t['endpoint_eligible']=t[f'mask__{TASK}'].astype(bool)
    t['has_exact_current_video']=t.current_visit_id.isin(video_index)
    t['has_complete_current_strain']=t.current_visit_id.isin(tensors)
    t['has_complete_previous_strain']=[(history_ids(by_id,x)[1] or x) in tensors for x in t.current_visit_id]
    t['included']=t.endpoint_eligible & t.has_exact_current_video & t.has_complete_current_strain & t.has_complete_previous_strain
    def reason(r):
        reasons=[]
        if not r.endpoint_eligible:reasons.append('endpoint_ineligible')
        if not r.has_exact_current_video:reasons.append('no_exact_current_video_triplet')
        if not r.has_complete_current_strain or not r.has_complete_previous_strain:reasons.append('incomplete_strain_curves')
        return ';'.join(reasons) or 'included'
    t['alignment_status']=t.apply(reason,axis=1)
    t.to_parquet(out/'transition_alignment.parquet',index=False)
    cohort=t[t.included].copy().reset_index(drop=True)
    metadata=cohort[['transition_id','patient_id','current_visit_id','target_visit_id','current_visit_order','target_visit_order']].copy()
    for prefix,column in [('current','current_visit_id'),('target','target_visit_id')]:
        metadata[prefix+'_visit_date']=cohort[column].map(by_id.visit_date)
        metadata[prefix+'_study_uid']=cohort[column].map(by_id.study_uid)
    assert (pd.to_datetime(metadata.target_visit_date)>pd.to_datetime(metadata.current_visit_date)).all()
    metadata['label']=cohort[f'label__{TASK}'].astype(int)
    metadata.to_parquet(out/'cohort.parquet',index=False)
    features=pd.read_csv(paths['feature_manifest'])
    clinical=features.loc[features.feature_set.eq('clinical'),'feature'].tolist()
    scalar=features.loc[features.feature_set.eq('gpu_scalars'),'feature'].tolist()
    strain_scalar=[c for c in scalar if c not in clinical]
    assert len(clinical)==len(set(clinical))
    assert not any(c.startswith(('target','label','mask','decline__','latent_')) for c in clinical+strain_scalar)
    arrays={'transition_ids':cohort.transition_id.to_numpy(str),'patient_ids':cohort.patient_id.to_numpy(str),
            'clinical':cohort[clinical].to_numpy(np.float32),'clinical_names':np.array(clinical),
            'strain_scalars':cohort[strain_scalar].to_numpy(np.float32),'strain_scalar_names':np.array(strain_scalar),
            'views':np.array(VIEWS),'strain_layers':np.array(['endo','mid']),
            'strain_channel_names':np.array(['endo','mid','endo_minus_mid','delta_endo','delta_mid','delta_endo_minus_mid'])}
    strain=[];first_ids=[];previous_ids=[]
    for r in cohort.itertuples():
        current=by_id.loc[r.current_visit_id];target=by_id.loc[r.target_visit_id]
        assert current.patient_id==target.patient_id==r.patient_id and target.visit_order==current.visit_order+1
        first,previous=history_ids(by_id,r.current_visit_id);first_ids.append(first);previous_ids.append(previous)
        if previous:assert by_id.loc[previous].study_datetime<current.study_datetime
        tensor=tensors[r.current_visit_id];endo=tensor[:,0];mid=tensor[:,1]
        prior=tensors[previous] if previous else tensor
        d_endo=endo-prior[:,0];d_mid=mid-prior[:,1]
        strain.append(np.clip(np.stack([endo,mid,endo-mid,d_endo,d_mid,d_endo-d_mid],axis=1)/30.,-2.,2.))
    arrays['strain_curves']=np.stack(strain).astype(np.float32)
    arrays['strain_curves_current_native_units']=np.stack([tensors[x] for x in cohort.current_visit_id])
    for model in MODELS:
        cur=np.stack([videos[model][x] for x in cohort.current_visit_id])
        arrays[model+'_current']=cur
        for which,ids in [('first',first_ids),('previous',previous_ids)]:
            values=[historical_video(x,videos[model].get(visit)) for x,visit in zip(cur,ids)]
            arrays[model+'_delta_'+which]=np.stack([x[0] for x in values])
            arrays[model+'_'+which+'_available']=np.stack([x[1] for x in values])
    np.savez_compressed(out/'features.npz',**arrays)
    np.savez_compressed(out/'labels.npz',transition_ids=arrays['transition_ids'],patient_ids=arrays['patient_ids'],label=metadata.label.to_numpy(np.int64),task=np.array(TASK))
    folds=pd.read_csv(paths['patient_folds']);folds=folds[folds.patient_id.isin(cohort.patient_id)].copy()
    assert folds.role.eq('test').all() and folds.groupby(['patient_id','repeat']).size().eq(1).all()
    assert set(folds.patient_id)==set(cohort.patient_id)
    folds.to_parquet(out/'patient_folds.parquet',index=False)
    oof=pd.read_parquet(paths['retained_oof'])
    baseline=oof[oof.task.eq(TASK)&oof.model.eq('ensemble_cnn_moment')&oof.transition_id.isin(cohort.transition_id)].copy()
    assert baseline.transition_id.is_unique and len(baseline)==len(cohort)
    paired=metadata[['transition_id','patient_id','label']].merge(baseline,on=['transition_id','patient_id','label'],validate='one_to_one')
    assert len(paired)==len(cohort)
    paired.to_parquet(out/'retained_baseline_oof.parquet',index=False)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'endpoint':'Immediately following strain visit is the first >=15% relative Mid-GLS deterioration from first visit',
        'strain_visits':len(visits),'strain_patients':int(visits.patient_id.nunique()),'video_visits':len(records)//3,
        'visit_match_counts':audit.match_status.value_counts().to_dict(),'video_only_visits':len(extra[['patient','visit_date']].drop_duplicates()),
        'all_transitions':len(t),'original_primary_transitions':int(t.endpoint_eligible.sum()),'aligned_primary_transitions':len(cohort),
        'aligned_patients':int(cohort.patient_id.nunique()),'events':int(metadata.label.sum()),
        'clinical_features':len(clinical),'strain_scalar_features':len(strain_scalar),'feature_shapes':{k:list(x.shape) for k,x in arrays.items()},
        'missing_strain_visit_video':audit.loc[~audit.match_status.eq('exact_patient_date_study_uid'),['visit_id','patient_id','visit_date','match_status']].to_dict('records'),
        'primary_exclusions':t.loc[t.endpoint_eligible & ~t.included,['transition_id','alignment_status']].to_dict('records'),
        'input_files':{k:{'path':str(p),'sha256':checksum(p)} for k,p in paths.items()},'code_sha256':checksum(Path(__file__)),
        'method':['Exact patient/date/Study UID match; no nearest-date substitution','Technical report reanalyses averaged within true visit','Strain curves phase-normalized to 96 samples; no frame-by-frame video/strain phase alignment asserted','Features restricted to current and earlier visits; target data used only for labels and audit','Original QC flags retained; accepted by user override','No fitted preprocessing, PCA, imputation, scaling or training','27 legacy clinical features are echo/history features; no demographics or oncology treatment variables added','Raw curve-table latent PCA columns excluded','Retained baseline OOF for paired comparison only; do not train a stacker on pooled existing OOF scores']}
    (out/'alignment_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Ichilov3 multimodal alignment','',f"{len(cohort)} eligible transitions, {cohort.patient_id.nunique()} patients, {metadata.label.sum()} events.",'',
        f"{len(matched)} of {len(visits)} strain visits match all three video cines by patient, date and exact Study UID. {summary['video_only_visits']} video visits have no strain report.",'',
        '## Files','', 'features.npz: clinical (27), strain scalars (72), segment/layer strain curves and both video encoders; no future-visit predictors.',
        'labels.npz: endpoint labels in exactly the same transition order.',
        'cohort.parquet: identity and chronology; target visit fields are audit metadata, not predictors.',
        'visit_alignment.parquet / transition_alignment.parquet: all input rows, match status and exclusion reasons.',
        'patient_folds.parquet: retained 3×5 patient-held-out assignments.',
        'retained_baseline_oof.parquet: CNN+MOMENT baseline scores for paired evaluation.',
        'alignment_summary.json: input hashes, feature shapes and matching protocol.','',
        '## Exclusions','']
    for r in summary['missing_strain_visit_video']:lines.append(f"- {r['visit_id']} ({r['visit_date']}): {r['match_status']}")
    lines+=['','## Training rules','']+[f'- {x}' for x in summary['method']]
    (out/'alignment_report.md').write_text('\n'.join(lines),encoding='utf-8')
    with np.load(out/'features.npz') as checked,np.load(out/'labels.npz') as labels:
        assert np.array_equal(checked['transition_ids'],labels['transition_ids'])
        for name in ['strain_curves',*[m+'_current' for m in MODELS]]:assert np.isfinite(checked[name]).all()
    print(json.dumps({k:summary[k] for k in ['visit_match_counts','video_only_visits','aligned_primary_transitions','aligned_patients','events','primary_exclusions']},indent=2))


if __name__=='__main__':main()
