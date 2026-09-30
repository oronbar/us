"""Six prespecified models on nested patient-held-out folds; resumable local training."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

INPUT=Path(r'D:\DS\ichilov3_aligned_20260927')
OUTPUT=Path(r'D:\DS\ichilov3_fusion_training_20260927')
MODELS={'clinical_ridge':('clinical','ridge'),
 'strain_clinical_ridge':('strain','ridge'),
 'strain_clinical_echoprime_ridge':('echoprime','ridge'),
 'strain_clinical_panecho_ridge':('panecho','ridge'),
 'strain_clinical_echoprime_tabpfn':('echoprime','tabpfn'),
 'strain_clinical_panecho_tabpfn':('panecho','tabpfn')}
CS=[.001,.01,.1,1.]


def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


def hashfile(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def raw_blocks(features):
    blocks={'clinical':features['clinical'],'strain_scalars':features['strain_scalars'],
            'strain_curves':features['strain_curves'].reshape(len(features['clinical']),-1)}
    for model in ['echoprime','panecho']:
        for view in range(3):blocks[f'{model}_{view}']=features[model+'_current'][:,view]
    return blocks


def preprocess(blocks,train,test,seed):
    transformed={};pipelines={}
    for name,x in blocks.items():
        components=None if name=='clinical' else (8 if name.startswith('strain') else 4)
        steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]
        if components:steps.append(('pca',PCA(n_components=min(components,len(train)-1,x.shape[1]),svd_solver='randomized',random_state=seed)))
        # Unit scaling after PCA gives each compact block a comparable ridge penalty.
        if components:steps.append(('component_scale',StandardScaler()))
        pipe=Pipeline(steps);a=pipe.fit_transform(x[train]);b=pipe.transform(x[test])
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite fitted inputs')
        transformed[name]=(a,b);pipelines[name]=pipe
    return transformed,pipelines


def matrix(transformed,representation):
    names=['clinical']
    if representation!='clinical':names+=['strain_scalars','strain_curves']
    if representation in ['echoprime','panecho']:names += [f'{representation}_{i}' for i in range(3)]
    return tuple(np.concatenate([transformed[n][i] for n in names],axis=1) for i in [0,1])


def ridge(x,y,c):
    return LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=3000).fit(x,y)


def metric(y,p):
    return {'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p))}


def summarize(out,cohort,bootstraps=2000):
    raw=pd.concat([pd.read_parquet(p) for p in sorted((out/'folds').glob('*.parquet'))],ignore_index=True)
    raw.to_parquet(out/'fold_predictions.parquet',index=False)
    if raw.groupby(['model','transition_id']).size().ne(3).any():raise ValueError('Incomplete repeated held-out predictions')
    pred=raw.groupby(['model','transition_id','patient_id','label'],as_index=False).score.mean()
    base=pd.read_parquet(INPUT/'retained_baseline_oof.parquet')[['transition_id','patient_id','label','score']].copy()
    base['model']='retained_cnn_moment_reference';pred=pd.concat([pred,base],ignore_index=True)
    pred.to_parquet(out/'oof_predictions.parquet',index=False)
    wide=pred.pivot(index=['transition_id','patient_id','label'],columns='model',values='score')
    if wide.isna().any().any():raise ValueError('Unequal paired evaluation cohorts')
    y=wide.index.get_level_values('label').to_numpy();groups=wide.index.get_level_values('patient_id').to_numpy()
    unique=np.unique(groups);patient_rows={p:np.flatnonzero(groups==p) for p in unique}
    estimates={m:metric(y,wide[m].to_numpy()) for m in wide.columns}
    samples={m:{k:[] for k in ['auc','ap','brier']} for m in wide.columns};rng=np.random.default_rng(20260927)
    for i in range(bootstraps):
        take=np.concatenate([patient_rows[p] for p in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[take]))<2:continue
        for m in wide.columns:
            for k,v in metric(y[take],wide[m].to_numpy()[take]).items():samples[m][k].append(v)
    metrics=[];deltas=[]
    for m in wide.columns:
        r={'model':m,**estimates[m]}
        for k in ['auc','ap','brier']:r[k+'_lo'],r[k+'_hi']=np.quantile(samples[m][k],[.025,.975]).tolist()
        metrics.append(r)
        for ref in ['strain_clinical_ridge','retained_cnn_moment_reference']:
            d={'model':m,'reference':ref}
            for k in ['auc','ap','brier']:
                d['delta_'+k]=estimates[m][k]-estimates[ref][k]
                d['delta_'+k+'_lo'],d['delta_'+k+'_hi']=np.quantile(np.array(samples[m][k])-np.array(samples[ref][k]),[.025,.975]).tolist()
            deltas.append(d)
    pd.DataFrame(metrics).to_parquet(out/'metrics.parquet',index=False)
    pd.DataFrame(deltas).to_parquet(out/'paired_deltas.parquet',index=False)
    lines=['# Ichilov3 six-model fusion comparison','',f'{len(cohort)} transitions, {cohort.patient_id.nunique()} patients, {cohort.label.sum()} events. Identical examples across models.','',
        '| Model | AUROC | AP | Brier |','|---|---:|---:|---:|']
    for r in metrics:lines.append(f"| {r['model']} | {r['auc']:.3f} | {r['ap']:.3f} | {r['brier']:.3f} |")
    lines+=['','Confidence intervals and paired differences are in metrics.parquet and paired_deltas.parquet.',
        'The new strain-clinical baseline is a regularized compact feature model. CNN+MOMENT is a retained reference, not a retrained branch or stacking input.',
        'Strain curves are compressed with training-only PCA. Historical-video fusion, learned late fusion, and neural fine-tuning were not included in these six configurations.',
        'All models use the retained 3x5 patient-held-out folds. Ridge penalties are chosen using grouped inner folds; TabPFN settings and PCA sizes are fixed in advance.',
        'Patient-bootstrap intervals condition on these OOF predictions. Repeated development on this cohort makes these exploratory comparisons.']
    (out/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    return metrics


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--smoke',action='store_true');p.add_argument('--ridge-only',action='store_true');args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'folds').mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    sys.path.insert(0,str(OUTPUT/'runtime'))
    cohort=pd.read_parquet(INPUT/'cohort.parquet');assign=pd.read_parquet(INPUT/'patient_folds.parquet')
    with np.load(INPUT/'features.npz') as f:features={k:f[k].copy() for k in f.files}
    with np.load(INPUT/'labels.npz') as f:
        y=f['label'].copy();assert np.array_equal(f['transition_ids'],cohort.transition_id.to_numpy(str))
    groups=cohort.patient_id.to_numpy(str);blocks=raw_blocks(features)
    models={k:v for k,v in MODELS.items() if not args.ridge_only or v[1]=='ridge'}
    splits=list(assign.groupby(['repeat','fold']))
    if args.smoke:splits=splits[:1]
    protocol={'models':{k:list(v) for k,v in models.items()},'outer_folds':'retained 3x5 patient-held-out','inner_folds':3,'ridge_C_grid':CS,
        'PCA':{'strain_curves':8,'strain_scalars':8,'video_each_view':4},'video_history':False,
        'strain_clinical_baseline':'new compact PCA + ridge; retained CNN+MOMENT separately evaluated',
        'tabpfn':{'package':'6.3.2','version':'2.5','estimators':4,'device':'cuda','telemetry':False,'local_only':True},
        'samples':len(y),'patients':len(set(groups)),'events':int(y.sum()),'smoke':args.smoke,
        'hashes':{n:hashfile(INPUT/n) for n in ['features.npz','labels.npz','patient_folds.parquet','cohort.parquet']},
        'code_sha256':hashfile(Path(__file__)),'bootstrap_draws':2000}
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Protocol/input changed: use a fresh output directory')
    atomic(protocol_path,protocol);expected=len(splits)*len(models);completed=len(list((out/'folds').glob('*.parquet')))
    started=time.time();errors=[]
    atomic(out/'status.json',{'status':'running','completed':completed,'total':expected,'pid':os.getpid()})
    with threadpool_limits(limits=4):
        for (repeat,fold),assignment in splits:
            test=np.flatnonzero(np.isin(groups,assignment.patient_id));train=np.flatnonzero(~np.isin(groups,assignment.patient_id))
            assert not set(groups[train])&set(groups[test]) and len(np.unique(y[test]))==2
            seed=20260927+int(repeat)*5+int(fold)
            remaining=[m for m in models if not (out/'folds'/f'{m}_r{repeat}_f{fold}.parquet').exists()]
            if not remaining:continue
            print(f'Preparing repeat {repeat} fold {fold}',flush=True)
            outer,pipelines=preprocess(blocks,train,test,seed)
            inner=[]
            if any(models[m][1]=='ridge' for m in remaining):
                for a,b in StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(len(train)),y[train],groups[train]):
                    fit,val=train[a],train[b];assert not set(groups[fit])&set(groups[val])
                    if len(np.unique(y[val]))!=2 or len(np.unique(y[fit]))!=2:raise ValueError('Insufficient inner-fold events')
                    transformed,_=preprocess(blocks,fit,val,seed);inner.append((transformed,fit,val))
            joblib.dump(pipelines,out/'models'/f'preprocessing_r{repeat}_f{fold}.joblib',compress=3)
            for name in remaining:
                representation,kind=models[name];tick=time.time()
                atomic(out/'status.json',{'status':'running','completed':completed,'total':expected,'model':name,'repeat':int(repeat),'fold':int(fold),'pid':os.getpid(),'elapsed_seconds':time.time()-started})
                try:
                    x_train,x_test=matrix(outer,representation);best_c=None;scores=[]
                    if kind=='ridge':
                        for c in CS:
                            values=[]
                            for transformed,fit,val in inner:
                                a,b=matrix(transformed,representation);predictor=ridge(a,y[fit],c)
                                values.append(average_precision_score(y[val],predictor.predict_proba(b)[:,1]))
                            scores.append(float(np.mean(values)))
                        best_c=CS[int(np.argmax(scores))];predictor=ridge(x_train,y[train],best_c)
                        pred=predictor.predict_proba(x_test)[:,1]
                        joblib.dump(predictor,out/'models'/f'{name}_r{repeat}_f{fold}.joblib',compress=3)
                    else:
                        from tabpfn import TabPFNClassifier
                        from tabpfn.constants import ModelVersion
                        predictor=TabPFNClassifier.create_default_for_version(ModelVersion.V2_5,device='cuda',n_estimators=4,n_preprocessing_jobs=1,random_state=seed)
                        predictor.fit(x_train,y[train]);pred=predictor.predict_proba(x_test)[:,1]
                        # Save local training context rather than duplicating the foundation checkpoint.
                        np.savez_compressed(out/'models'/f'{name}_r{repeat}_f{fold}_context.npz',x_train=x_train,y_train=y[train],train_index=train,test_index=test,seed=seed)
                    if pred.shape!=(len(test),) or not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):raise ValueError('Invalid held-out probabilities')
                    result=cohort.iloc[test][['transition_id','patient_id','label']].copy()
                    result=result.assign(model=name,repeat=int(repeat),fold=int(fold),score=pred)
                    dest=out/'folds'/f'{name}_r{repeat}_f{fold}.parquet';temp=dest.with_suffix('.partial.parquet');result.to_parquet(temp,index=False);temp.replace(dest)
                    atomic(out/'models'/f'{name}_r{repeat}_f{fold}.json',{'C':best_c,'inner_AP':scores,'seconds':time.time()-tick,'features':x_train.shape[1],'train_patients':sorted(set(groups[train])),'test_patients':sorted(set(groups[test]))})
                    completed+=1;print(f'{completed}/{expected}: {name} r{repeat} f{fold} ({time.time()-tick:.1f}s)',flush=True)
                    del predictor;gc.collect()
                    if kind=='tabpfn':
                        import torch
                        torch.cuda.empty_cache()
                except Exception as error:
                    problem={'model':name,'repeat':int(repeat),'fold':int(fold),'error':str(error),'traceback':traceback.format_exc()};errors.append(problem)
                    atomic(out/'errors.json',errors);print(f'ERROR {name}: {error}',flush=True)
                    # Do not repeatedly attempt a gated/unavailable checkpoint on every fold.
                    if kind=='tabpfn':raise
    if completed!=expected:raise RuntimeError(f'Only {completed}/{expected} runs completed')
    if not args.smoke and not args.ridge_only:
        atomic(out/'status.json',{'status':'evaluating','completed':completed,'total':expected,'pid':os.getpid()})
        summary=summarize(out,cohort)
    else:summary=[]
    atomic(out/'status.json',{'status':'complete','completed':completed,'total':expected,'seconds':time.time()-started,'metrics':summary})
    atomic(out/'run_complete.json',{'runs':completed,'seconds':time.time()-started,'smoke':args.smoke})


if __name__=='__main__':
    try:main()
    except Exception:
        failed_output=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else OUTPUT
        status_path=failed_output/'status.json'
        if status_path.exists():
            previous=json.loads(status_path.read_text());previous.update(status='error',error=traceback.format_exc())
            atomic(status_path,previous)
        traceback.print_exc();sys.exit(1)
