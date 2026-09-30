"""Prespecified frozen-video probes and paired incremental-value analysis.

All learned preprocessing and ridge penalties are nested within patient folds.
Historical differences use only the true baseline/previous visit, never a future
or nearest available study. Fixed blends use the existing averaged OOF baseline;
no learned stacking or selection on these held-out scores is performed.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

APICAL=['A2C','A3C','A4C']
ALL_VIEWS=APICAL+['Parasternal_Long','Parasternal_Short']
ROOT=Path('D:/us')


def visit_vectors(rows,out,model,views):
    dim=512 if model=='echoprime' else 768
    vectors={}
    for study,g in rows[rows.view.isin(views)].groupby('study_uid'):
        blocks=np.zeros((len(views),dim),dtype=np.float32);present=np.zeros(len(views),np.float32)
        for i,view in enumerate(views):
            feats=[]
            for r in g[g.view.eq(view)].itertuples():
                f=out/'embeddings'/model/f'{r.file_id}.npz'
                if not f.exists():raise RuntimeError(f'Missing {model} embedding: {r.file_id}')
                with np.load(f) as a:
                    emb=a['embeddings'];indices=a['frame_indices']
                    if emb.ndim!=2 or emb.shape[1]!=dim or not np.isfinite(emb).all():
                        raise ValueError(f'Invalid {model} embedding: {r.file_id}')
                    if indices.shape!=(len(emb),16) or np.any(indices<0) or np.any(indices>=r.frames):
                        raise ValueError(f'Invalid frame indices: {r.file_id}')
                    appearance=str(a['appearance']) if 'appearance' in a else 'native_rgb'
                    if appearance!=r.appearance or int(a['source_size'])!=r.source_size or int(a['source_mtime'])!=r.source_mtime or not np.array_equal(a['crop'],r.crop):
                        raise ValueError(f'Stale embedding metadata: {r.file_id}')
                    feats.append(emb.mean(0))
            if feats:blocks[i]=np.mean(feats,axis=0);present[i]=1
        vectors[study]=(blocks,present)
    return vectors


def features_for_transitions(transitions,visits,vectors,history=False):
    by_id=visits.set_index('visit_id')
    ordered=visits.sort_values(['patient_id','visit_order'])
    histories={p:g.visit_id.tolist() for p,g in ordered.groupby('patient_id')}
    result=[]
    for t in transitions.itertuples():
        current=by_id.loc[t.current_visit_id]
        cur,mask=vectors[current.study_uid]
        blocks=[cur.ravel(),mask]
        if history:
            visits_for_patient=histories[t.patient_id];ix=visits_for_patient.index(t.current_visit_id)
            for prior_id in [visits_for_patient[0] if ix>0 else None,visits_for_patient[ix-1] if ix>0 else None]:
                if prior_id is not None:
                    previous=by_id.loc[prior_id]
                    assert previous.visit_order<current.visit_order
                    prior=vectors.get(previous.study_uid)
                else:prior=None
                if prior is None:delta=np.zeros_like(cur);available=np.zeros_like(mask)
                else:
                    available=mask*prior[1];delta=(cur-prior[0])*available[:,None]
                blocks.extend([delta.ravel(),available])
        result.append(np.concatenate(blocks))
    return np.stack(result)


def fit_probe(x,y,groups,train,test,seed,video):
    inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]
    if video:steps.append(('pca',PCA(n_components=8,svd_solver='randomized',random_state=seed)))
    steps.append(('ridge',LogisticRegression(penalty='l2',solver='lbfgs',max_iter=2000)))
    search=GridSearchCV(Pipeline(steps),{'ridge__C':[.001,.01,.1,1.]},scoring='average_precision',
        cv=inner,n_jobs=1,error_score='raise')
    for a,b in inner.split(x[train],y[train],groups[train]):
        assert not set(groups[train][a])&set(groups[train][b])
        if len(np.unique(y[train][a]))<2 or len(np.unique(y[train][b]))<2:
            raise RuntimeError('Insufficient events for prespecified inner folds; revise protocol before running')
    search.fit(x[train],y[train],groups=groups[train])
    return search.predict_proba(x[test])[:,1],search.best_params_['ridge__C']


def metric(y,p):
    return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),brier=brier_score_loss(y,p))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/dicom_prediction')
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args();out=args.output
    rows=pd.read_parquet(out/'selected_clips.parquet')
    visits=pd.read_parquet(ROOT/'amber_full_105_preprocessed/Ichilov_july_visits.parquet')
    t=pd.read_parquet(ROOT/'cardiotoxicity_next_visit_gpu_results/next_visit_transitions.parquet')
    # A current apical video is required for every model's paired evaluation.
    eligible_studies=set(rows[rows.view.isin(APICAL)].study_uid)
    eligible_visits=set(visits[visits.study_uid.isin(eligible_studies)].visit_id)
    t=t[t.mask__mid_first_rel15.astype(bool)&t.current_visit_id.isin(eligible_visits)].reset_index(drop=True)
    if len(t)<60 or t.label__mid_first_rel15.sum()<15:
        raise RuntimeError('Too few matched transitions/events for planned evaluation')
    y=t.label__mid_first_rel15.to_numpy(int);groups=t.patient_id.to_numpy()
    features={}
    manifest=pd.read_csv(ROOT/'cardiotoxicity_next_visit_gpu_results/feature_manifest.csv')
    clinical_cols=manifest.loc[manifest.feature_set.eq('clinical'),'feature'].tolist()
    features['clinical_ridge_refit']=t[clinical_cols].to_numpy(float)
    # Negative control: acquisition availability and clip length, without pixels.
    study_by_visit=visits.set_index('visit_id').study_uid.to_dict()
    acquisition={}
    for study,g in rows.groupby('study_uid'):
        acquisition[study]=np.array([value for view in APICAL for value in
            [int(g.view.eq(view).sum()),float(g.loc[g.view.eq(view),'frames'].mean()) if g.view.eq(view).any() else 0.]])
    features['acquisition_only_control']=np.stack([acquisition[study_by_visit[x]] for x in t.current_visit_id])
    for model in ['echoprime','panecho']:
        for broad in [False,True]:
            vectors=visit_vectors(rows,out,model,ALL_VIEWS if broad else APICAL)
            for history in ([False] if broad else [False,True]):
                key=model+('_allviews' if broad else '_apical')+('_history' if history else '')
                features[key]=features_for_transitions(t,visits,vectors,history)
    folds=pd.read_csv(ROOT/'cardiotoxicity_cnn_length_ablation_results/cnn_length_ablation_patient_folds.csv')
    folds=folds[folds.role.eq('test')]
    counts=folds.groupby(['patient_id','repeat']).size()
    assert counts.eq(1).all() and set(groups)<=set(folds.patient_id)
    cache=out/'evaluation_folds';cache.mkdir(exist_ok=True)
    fold_predictions=[];log=[]
    # The protocol file is written before looking at any held-out performance.
    protocol=dict(endpoint='next visit first >=15% relative Mid-GLS decline',transitions=len(t),events=int(y.sum()),
        patients=int(t.patient_id.nunique()),outer_folds='existing CNN 3x5 patient folds',inner_folds=3,
        C_grid=[.001,.01,.1,1.],video_pca_components=8,selection_metric='inner AP',
        video_pooling='equal means within window,clip,view; 2 clips/view maximum',
        models=list(features),fixed_blend_weight_video=.25,clinical_features=clinical_cols,
        baseline='retained CNN+MOMENT OOF; baseline was trained on its original outer training cohort',
        limitations=['Automated view/quality screening, pending clinician review',
            'Existing cohort repeatedly explored; no new external confirmation',
            'Current-video availability defines subset; report coverage and missingness',
            'Secondary history/all-view comparisons exploratory; no winner selected for deployment'])
    provenance_paths={
        'evaluation_code':Path(__file__),
        'selected_clips':out/'selected_clips.parquet',
        'patient_folds':ROOT/'cardiotoxicity_cnn_length_ablation_results/cnn_length_ablation_patient_folds.csv',
        'retained_oof':ROOT/'cardiotoxicity_timeseries_round4_results/round4_oof_predictions.parquet',
    }
    protocol['file_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in provenance_paths.items()}
    protocol['feature_sha256']={name:hashlib.sha256(x.tobytes()).hexdigest() for name,x in features.items()}
    code_hash=protocol['file_sha256']['evaluation_code'].encode()
    (out/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
    t[['transition_id','patient_id','current_visit_id','target_visit_id','label__mid_first_rel15']].to_parquet(out/'evaluation_cohort.parquet',index=False)
    with threadpool_limits(limits=4):
        for (repeat,fold),assign in folds.groupby(['repeat','fold']):
            test=np.flatnonzero(np.isin(groups,assign.patient_id));train=np.flatnonzero(~np.isin(groups,assign.patient_id))
            assert not set(groups[train])&set(groups[test])
            for name,x in features.items():
                fingerprint=hashlib.sha256(code_hash+x.tobytes()+y.tobytes()+json.dumps(groups.tolist()).encode()+train.tobytes()+test.tobytes()).hexdigest()[:12]
                f=cache/f'{name}_{fingerprint}_r{repeat}_f{fold}.parquet'
                if f.exists():pred=pd.read_parquet(f)
                else:
                    score,c=fit_probe(x,y,groups,train,test,20260914+int(repeat)*5+int(fold),name not in {'clinical_ridge_refit','acquisition_only_control'})
                    pred=t.iloc[test][['transition_id','patient_id']].assign(label=y[test],score=score,model=name,repeat=repeat,fold=fold,C=c)
                    pred.to_parquet(f,index=False)
                fold_predictions.append(pred)
                log.append(dict(model=name,repeat=int(repeat),fold=int(fold),train_patients=len(set(groups[train])),test_patients=len(set(groups[test])),C=float(pred.C.iloc[0])))
            print(f'Completed nested repeat {repeat} fold {fold}',flush=True)
    raw=pd.concat(fold_predictions,ignore_index=True);raw.to_parquet(out/'video_fold_predictions.parquet',index=False)
    pred=raw.groupby(['model','transition_id','patient_id','label'],as_index=False).score.mean()
    old=pd.read_parquet(ROOT/'cardiotoxicity_timeseries_round4_results/round4_oof_predictions.parquet')
    old=old[old.task.eq('mid_first_rel15')&old.transition_id.isin(t.transition_id)]
    base=old[old.model.eq('ensemble_cnn_moment')][['model','transition_id','patient_id','label','score']].copy()
    assert len(base)==len(t) and base.transition_id.is_unique
    base['model']='retained_strain_clinical'
    frames=[pred,base]
    for model in ['echoprime','panecho']:
        video=pred[pred.model.eq(model+'_apical')]
        blend=base.merge(video,on=['transition_id','patient_id','label'],suffixes=('_base','_video'),validate='one_to_one')
        blend['score']=.75*blend.score_base+.25*blend.score_video
        blend['model']='strain_clinical_plus_'+model
        frames.append(blend[['model','transition_id','patient_id','label','score']])
    predictions=pd.concat(frames,ignore_index=True)
    predictions.to_parquet(out/'paired_oof_predictions.parquet',index=False)
    wide=predictions.pivot(index=['transition_id','patient_id','label'],columns='model',values='score')
    assert not wide.isna().any().any()
    yy=wide.index.get_level_values('label').to_numpy();patients=wide.index.get_level_values('patient_id').to_numpy()
    metrics={name:metric(yy,wide[name].to_numpy()) for name in wide.columns}
    unique=np.unique(patients);idx={p:np.flatnonzero(patients==p) for p in unique};rng=np.random.default_rng(20260914)
    boot={name:{m:[] for m in ['auc','ap','brier']} for name in wide.columns}
    for _ in range(args.bootstraps):
        take=np.concatenate([idx[p] for p in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(yy[take]))<2:continue
        for name in wide.columns:
            for m,value in metric(yy[take],wide[name].to_numpy()[take]).items():boot[name][m].append(value)
    results=[];deltas=[]
    for name in wide.columns:
        result=dict(model=name,**metrics[name])
        delta=dict(model=name,reference='retained_strain_clinical')
        for m in ['auc','ap','brier']:
            result[m+'_lo'],result[m+'_hi']=np.quantile(boot[name][m],[.025,.975])
            delta['delta_'+m]=metrics[name][m]-metrics['retained_strain_clinical'][m]
            delta['delta_'+m+'_lo'],delta['delta_'+m+'_hi']=np.quantile(np.array(boot[name][m])-np.array(boot['retained_strain_clinical'][m]),[.025,.975])
        results.append(result);deltas.append(delta)
    pd.DataFrame(results).to_csv(out/'paired_metrics.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv',index=False)
    pd.DataFrame(log).to_csv(out/'nested_training_log.csv',index=False)
    lines=['# Exploratory frozen-video prediction experiment','',f'{len(t)} transitions, {int(y.sum())} events, {len(unique)} patients. Same cohort for all rows.','',
        'The retained strain–clinical baseline is CNN + MOMENT from Round 4. New video probes and clinical ridge use nested patient folds. Fixed blends allocate 25% to video; no blend weight was selected on evaluation outcomes.','',
        '| Model | AUROC (95% patient-bootstrap CI) | AP | Brier |','|---|---|---|---|']
    for r in results:lines.append(f"| {r['model']} | {r['auc']:.3f} ({r['auc_lo']:.3f}–{r['auc_hi']:.3f}) | {r['ap']:.3f} | {r['brier']:.3f} |")
    lines+=['','These are internally validated exploratory results, conditional on automated clip screening and video availability. Clinician view/quality review is pending. Bootstrap intervals do not capture the full uncertainty of repeated model development. No clinical deployment or confirmed incremental benefit follows from point estimates.','',
        'See paired_deltas.csv for paired confidence intervals; negative delta Brier is improvement. Baseline models were retained rather than retrained on fewer patients; the refitted clinical ridge is the matched-training-cohort control.']
    (out/'prediction_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(pd.DataFrame(results)[['model','auc','ap','brier']].to_string(index=False),flush=True)


if __name__=='__main__':main()
