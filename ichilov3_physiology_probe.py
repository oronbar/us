"""Video-only prediction of current GLS/EF, using all exactly matched visits."""
from __future__ import annotations
import argparse,json,os,time,traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from ichilov3_train_late_fusion import video_preprocess
from ichilov3_train_fusion import atomic,hashfile

ROOT=Path(__file__).resolve().parent
OUTPUT=Path(r'D:\DS\ichilov3_physiology_audit_20260927')
TARGETS={'mid_gls':'gls_mid_magnitude','endo_gls':'gls_endo_magnitude','ef':'ef_biplane'}
ALPHAS=(.1,1.,10.,100.,1000.,10000.)
REPRESENTATIONS=('full_embedding','pca8_per_view')


def regression_metric(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--model-inputs',type=Path,default=Path(r'D:\DS\ichilov3_model_inputs_20260927'))
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
    for name in ['folds','models']:(out/name).mkdir(exist_ok=True)
    paths={'alignment':Path(r'D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet'),
           'visits':ROOT/'amber_full_105_preprocessed/Ichilov_july_visits.parquet',
           'folds':Path(r'D:\DS\ichilov3_aligned_20260927\patient_folds.parquet')}
    matched=pd.read_parquet(paths['alignment']);matched=matched[matched.match_status.eq('exact_patient_date_study_uid')]
    visits=pd.read_parquet(paths['visits'])
    cohort=matched[['visit_id','patient_id','visit_date','study_uid']].merge(visits[['visit_id',*TARGETS.values()]],on='visit_id',validate='one_to_one').sort_values(['patient_id','visit_date']).reset_index(drop=True)
    x={}
    for model in ['echoprime','panecho']:
        paths[model]=args.model_inputs/(model+'_visits.npz')
        with np.load(paths[model]) as f:
            assert list(f['views'])==['A2C','A3C','A4C']
            lookup={(str(p),str(d)):v for p,d,v in zip(f['patient_ids'],f['visit_dates'],f['embeddings'])}
            x[model]=np.stack([lookup[(r.patient_id,r.visit_date)] for r in cohort.itertuples()])
            assert np.isfinite(x[model]).all()
    cohort.to_parquet(out/'cohort.parquet',index=False);folds=pd.read_parquet(paths['folds']);groups=cohort.patient_id.to_numpy(str)
    protocol={'targets':TARGETS,'inputs':'video only; current visit, three views, frozen encoder','cohort':len(cohort),
              'target_counts':{k:int(np.isfinite(cohort[c]).sum()) for k,c in TARGETS.items()},
              'outer_folds':'existing 3x5 patient-held-out','inner_folds':3,'alpha_grid':list(ALPHAS),'representations':list(REPRESENTATIONS),
              'controls':['outer-training target mean','outer-training target median'],'bootstraps':2000,
              'code_sha256':hashfile(Path(__file__)),'input_hashes':{k:hashfile(v) for k,v in paths.items()}}
    pp=out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Protocol changed; use fresh output')
    atomic(pp,protocol);started=time.time();completed=len(list((out/'folds').glob('*.parquet')))
    with threadpool_limits(limits=4):
        for (repeat,fold),a in folds.groupby(['repeat','fold']):
            heldout=np.isin(groups,a.patient_id)
            for target,column in TARGETS.items():
                y=cohort[column].to_numpy(float);valid=np.isfinite(y);train=np.flatnonzero(valid & ~heldout);test=np.flatnonzero(valid & heldout)
                assert not set(groups[train]) & set(groups[test])
                seed=20260929+int(repeat)*5+int(fold)
                for model in x:
                    dest=out/'folds'/f'{model}_{target}_r{repeat}_f{fold}.parquet'
                    if dest.exists():continue
                    atomic(out/'status.json',dict(status='training',completed=completed,total=90,target=target,model=model,pid=os.getpid()))
                    candidates=[]
                    splits=list(GroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(len(train)),y[train],groups[train]))
                    for representation in REPRESENTATIONS:
                        errors={alpha:[] for alpha in ALPHAS}
                        for ia,ib in splits:
                            fit,val=train[ia],train[ib];assert not set(groups[fit]) & set(groups[val])
                            xx,vx,_=video_preprocess(x[model],fit,val,representation,seed)
                            for alpha in ALPHAS:
                                ridge=Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(xx,y[fit])
                                errors[alpha].append(mean_absolute_error(y[val],ridge.predict(vx)))
                        candidates += [dict(representation=representation,alpha=alpha,inner_mae=float(np.mean(errors[alpha]))) for alpha in ALPHAS]
                    best=min(candidates,key=lambda r:r['inner_mae'])
                    xx,vx,pipelines=video_preprocess(x[model],train,test,best['representation'],seed)
                    ridge=Ridge(alpha=best['alpha'],solver='lsqr',tol=1e-5).fit(xx,y[train]);pred=ridge.predict(vx)
                    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
                    result=cohort.iloc[test][['visit_id','patient_id','visit_date']].copy()
                    result=result.assign(target=target,model=model,repeat=int(repeat),fold=int(fold),value=y[test],prediction=pred,
                                         train_mean=float(np.mean(y[train])),train_median=float(np.median(y[train])))
                    temp=dest.with_suffix('.tmp');result.to_parquet(temp,index=False);temp.replace(dest)
                    joblib.dump(dict(pipelines=pipelines,predictor=ridge),out/'models'/f'{model}_{target}_r{repeat}_f{fold}.joblib',compress=3)
                    atomic(out/'models'/f'{model}_{target}_r{repeat}_f{fold}.json',dict(best=best,candidates=candidates,train_patients=sorted(set(groups[train])),test_patients=sorted(set(groups[test]))))
                    completed+=1;print(f'{completed}/90 {model} {target} r{repeat} f{fold}',flush=True)
        atomic(out/'status.json',dict(status='evaluating',completed=completed,total=90,pid=os.getpid()))
        raw=pd.concat([pd.read_parquet(f) for f in sorted((out/'folds').glob('*.parquet'))],ignore_index=True)
        assert raw.groupby(['target','model','visit_id']).size().eq(3).all()
        oof=raw.groupby(['target','model','visit_id','patient_id','value'],as_index=False)[['prediction','train_mean','train_median']].mean()
        oof.to_parquet(out/'oof_predictions.parquet',index=False)
        metrics=[];rng=np.random.default_rng(20260929)
        for (target,model),g in oof.groupby(['target','model']):
            y=g.value.to_numpy();pred=g.prediction.to_numpy();control=g.train_median.to_numpy();mean=g.train_mean.to_numpy()
            row=dict(target=target,model=model,visits=len(g),patients=g.patient_id.nunique(),**regression_metric(y,pred),
                     median_control_mae=float(mean_absolute_error(y,control)),mean_control_mae=float(mean_absolute_error(y,mean)))
            pts=g.patient_id.unique();ix={p:np.flatnonzero(g.patient_id.to_numpy()==p) for p in pts};samples=[]
            for draw in range(2000):
                take=np.concatenate([ix[p] for p in rng.choice(pts,len(pts),replace=True)])
                samples.append(mean_absolute_error(y[take],pred[take])-mean_absolute_error(y[take],control[take]))
            row['delta_mae_vs_median']=row['mae']-row['median_control_mae']
            row['delta_mae_lo'],row['delta_mae_hi']=np.quantile(samples,[.025,.975]).tolist();metrics.append(row)
        frame=pd.DataFrame(metrics);frame.to_parquet(out/'metrics.parquet',index=False)
        lines=['# Same-visit physiology probe','',f'{len(cohort)} exactly matched visits; patients held out across all visits. No GLS, EF or clinical features are supplied as predictors.','',
               '| Target | Encoder | Visits | MAE | Median-control MAE | R² | MAE difference 95% CI |','|---|---|---:|---:|---:|---:|---|']
        for r in metrics:lines.append(f"| {r['target']} | {r['model']} | {r['visits']} | {r['mae']:.3f} | {r['median_control_mae']:.3f} | {r['r2']:.3f} | {r['delta_mae_lo']:.3f} to {r['delta_mae_hi']:.3f} |")
        lines+=['','MAE is in percentage points. Negative differences favor video. Intervals use paired patient-cluster bootstrap draws.',
                'This is a same-visit diagnostic, not future-deterioration prediction. Failure can reflect restricted target variation or probe capacity; it does not prove a preprocessing bug.']
        (out/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic(out/'status.json',dict(status='complete',completed=completed,total=90,seconds=time.time()-started,metrics=metrics))
    atomic(out/'run_complete.json',dict(runs=completed,seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except Exception:
        OUTPUT.mkdir(exist_ok=True,parents=True);atomic(OUTPUT/'status.json',dict(status='error',error=traceback.format_exc()));traceback.print_exc();raise
