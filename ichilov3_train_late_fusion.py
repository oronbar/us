"""Fixed late fusion with nested video-only probes and paired patient-bootstrap evaluation."""
from __future__ import annotations
import argparse,json,os,time,traceback
from pathlib import Path
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
from ichilov3_train_fusion import atomic,hashfile

INPUT=Path(r'D:\DS\ichilov3_aligned_20260927')
OUTPUT=Path(r'D:\DS\ichilov3_late_fusion_20260927')
MODELS=('echoprime','panecho')
CS=(.00001,.0001,.001,.01,.1,1.)
REPRESENTATIONS=('full_embedding','pca8_per_view')
WEIGHTS=(.25,.5)


def video_preprocess(x,train,test,representation,seed):
    pipelines=[];fit=[];heldout=[]
    if x.ndim!=3 or x.shape[1]!=3:raise ValueError('Expected three ordered video views')
    for view in range(3):
        steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]
        if representation=='pca8_per_view':
            steps += [('pca',PCA(n_components=min(8,len(train)-1,x.shape[2]),svd_solver='randomized',random_state=seed)),('component_scale',StandardScaler())]
        elif representation!='full_embedding':raise ValueError('Unknown representation')
        pipe=Pipeline(steps);fit.append(pipe.fit_transform(x[train,view]));heldout.append(pipe.transform(x[test,view]));pipelines.append(pipe)
    return np.concatenate(fit,axis=1),np.concatenate(heldout,axis=1),pipelines


def fixed_blend(baseline,video,weight):
    if not 0<=weight<=1:raise ValueError('Invalid blend weight')
    return (1-weight)*np.asarray(baseline)+weight*np.asarray(video)


def metric(y,p):
    return {'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p))}


def select_probe(x,y,groups,train,seed):
    splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(len(train)),y[train],groups[train]))
    candidates=[]
    for representation in REPRESENTATIONS:
        score={c:[] for c in CS}
        for a,b in splits:
            fit,val=train[a],train[b]
            assert not set(groups[fit])&set(groups[val])
            if len(np.unique(y[val]))!=2 or len(np.unique(y[fit]))!=2:raise ValueError('Insufficient inner-fold events')
            fit_x,val_x,_=video_preprocess(x,fit,val,representation,seed)
            for c in CS:
                classifier=LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=3000).fit(fit_x,y[fit])
                score[c].append(average_precision_score(y[val],classifier.predict_proba(val_x)[:,1]))
        candidates += [dict(representation=representation,C=c,inner_AP=float(np.mean(score[c]))) for c in CS]
    best=max(candidates,key=lambda r:r['inner_AP'])
    return best,candidates


def evaluate(out,cohort,draws):
    files=sorted((out/'folds').glob('video_*_r*_f*.parquet'))
    raw=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    if len(files)!=30 or raw.groupby(['model','transition_id']).size().ne(3).any():raise ValueError('Incomplete held-out video predictions')
    raw.to_parquet(out/'video_fold_predictions.parquet',index=False)
    video=raw.groupby(['model','transition_id','patient_id','label'],as_index=False).agg(score=('score','mean'),prevalence=('prevalence','mean'))
    baseline=pd.read_parquet(INPUT/'retained_baseline_oof.parquet')[['transition_id','patient_id','label','score']].copy()
    baseline['model']='cnn_moment_baseline';frames=[baseline]
    for model in MODELS:
        v=video[video.model.eq('video_'+model)].copy();frames.append(v.drop(columns='prevalence'))
        pair=baseline.merge(v,on=['transition_id','patient_id','label'],suffixes=('_baseline','_video'),validate='one_to_one')
        assert len(pair)==len(cohort)
        for weight in WEIGHTS:
            r=pair[['transition_id','patient_id','label']].copy()
            r['model']=f'late_{model}_video{int(weight*100)}';r['score']=fixed_blend(pair.score_baseline,pair.score_video,weight);frames.append(r)
            if model==MODELS[0]:
                control=r.copy();control['model']=f'prevalence_control_video{int(weight*100)}'
                control['score']=fixed_blend(pair.score_baseline,pair.prevalence,weight);frames.append(control)
    predictions=pd.concat(frames,ignore_index=True);predictions.to_parquet(out/'paired_oof_predictions.parquet',index=False)
    wide=predictions.pivot(index=['transition_id','patient_id','label'],columns='model',values='score')
    if wide.isna().any().any():raise ValueError('Unequal prediction cohorts')
    y=wide.index.get_level_values('label').to_numpy();groups=wide.index.get_level_values('patient_id').to_numpy()
    p=wide.to_numpy();names=list(wide.columns);point=[metric(y,p[:,j]) for j in range(len(names))]
    samples={m:{k:[] for k in ['auc','ap','brier']} for m in names}
    patients=np.unique(groups);indices={g:np.flatnonzero(groups==g) for g in patients};rng=np.random.default_rng(20260928)
    for draw in range(draws):
        rows=np.concatenate([indices[g] for g in rng.choice(patients,len(patients),replace=True)])
        if len(np.unique(y[rows]))!=2:continue
        for j,m in enumerate(names):
            values=metric(y[rows],p[rows,j])
            for k,v in values.items():samples[m][k].append(v)
    metrics=[];deltas=[]
    for j,m in enumerate(names):
        r={'model':m,**point[j]}
        for k in ['auc','ap','brier']:r[k+'_lo'],r[k+'_hi']=np.quantile(samples[m][k],[.025,.975]).tolist()
        metrics.append(r)
        references=['cnn_moment_baseline']
        if m.startswith('late_'):references.append('prevalence_control_video'+m.split('video')[-1])
        for ref in references:
            d={'model':m,'reference':ref}
            for k in ['auc','ap','brier']:
                d['delta_'+k]=r[k]-point[names.index(ref)][k]
                d['delta_'+k+'_lo'],d['delta_'+k+'_hi']=np.quantile(np.array(samples[m][k])-np.array(samples[ref][k]),[.025,.975]).tolist()
            deltas.append(d)
    pd.DataFrame(metrics).to_parquet(out/'metrics.parquet',index=False)
    pd.DataFrame(deltas).to_parquet(out/'paired_deltas.parquet',index=False)
    lines=['# Ichilov3 late fusion','',f'{len(cohort)} transitions, {cohort.patient_id.nunique()} patients, {int(cohort.label.sum())} events.','',
        'Primary comparisons allocate 25% to video and 75% to the retained CNN+MOMENT strain-clinical prediction. Equal weighting is secondary sensitivity analysis; weights were not selected using outcomes.','',
        '| Model | AUROC | AP | Brier |','|---|---:|---:|---:|']
    for r in metrics:lines.append(f"| {r['model']} | {r['auc']:.3f} | {r['ap']:.3f} | {r['brier']:.3f} |")
    lines+=['','Video-only ridge models select raw embeddings versus 8 PCA components per view, and regularization, within grouped inner folds.',
        'Prevalence controls blend the same baseline toward the outer-training event rate. They check whether calibration gains reflect generic shrinkage rather than useful image signal.',
        'No retraining of the retained baseline, learned blend weights, learned stacker, encoder fine-tuning or historical-video features.',
        'All predictions are held out by patient. Patient-bootstrap confidence intervals and paired differences are in metrics.parquet and paired_deltas.parquet.',
        'This cohort has been used repeatedly for development. These results are exploratory and do not establish clinical performance.']
    (out/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    return metrics


def main():
    global INPUT
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--bootstraps',type=int,default=2000)
    parser.add_argument('--input',type=Path,default=INPUT)
    args=parser.parse_args();INPUT=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    for name in ['folds','models']:(out/name).mkdir(exist_ok=True)
    cohort=pd.read_parquet(INPUT/'cohort.parquet');folds=pd.read_parquet(INPUT/'patient_folds.parquet')
    with np.load(INPUT/'features.npz') as f:
        x={m:f[m+'_current'].copy() for m in MODELS};assert np.array_equal(f['transition_ids'],cohort.transition_id.to_numpy(str))
    y=cohort.label.to_numpy(int);groups=cohort.patient_id.to_numpy(str)
    protocol={'endpoint':'mid_first_rel15','cohort':len(y),'patients':len(set(groups)),'events':int(y.sum()),'models':list(MODELS),
        'representations':list(REPRESENTATIONS),'C_grid':list(CS),'inner_folds':3,'outer_folds':'retained 3x5 patient-held-out',
        'primary_video_weight':.25,'secondary_video_weight':.5,'weights_learned':False,
        'control':'blend toward outer-training event prevalence, averaged over repeats',
        'bootstraps':args.bootstraps,'input_hashes':{n:hashfile(INPUT/n) for n in ['features.npz','cohort.parquet','patient_folds.parquet','retained_baseline_oof.parquet']},'code_sha256':hashfile(Path(__file__))}
    protocol_file=out/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=protocol:raise ValueError('Protocol changed; use a fresh output directory')
    atomic(protocol_file,protocol);tick=time.time();completed=len(list((out/'folds').glob('video_*_r*_f*.parquet')))
    with threadpool_limits(limits=4):
        for (repeat,fold),assignment in folds.groupby(['repeat','fold']):
            test=np.flatnonzero(np.isin(groups,assignment.patient_id));train=np.flatnonzero(~np.isin(groups,assignment.patient_id))
            assert not set(groups[train])&set(groups[test])
            for model in MODELS:
                dest=out/'folds'/f'video_{model}_r{repeat}_f{fold}.parquet'
                if dest.exists():continue
                atomic(out/'status.json',{'status':'training','completed':completed,'total':30,'encoder':model,'repeat':int(repeat),'fold':int(fold),'pid':os.getpid()})
                start=time.time();seed=20260928+int(repeat)*5+int(fold)
                best,candidates=select_probe(x[model],y,groups,train,seed)
                fit,heldout,pipelines=video_preprocess(x[model],train,test,best['representation'],seed)
                classifier=LogisticRegression(C=best['C'],penalty='l2',solver='lbfgs',max_iter=3000).fit(fit,y[train])
                score=classifier.predict_proba(heldout)[:,1]
                if not np.isfinite(score).all():raise ValueError('Nonfinite probabilities')
                result=cohort.iloc[test][['transition_id','patient_id','label']].copy()
                result=result.assign(model='video_'+model,repeat=int(repeat),fold=int(fold),score=score,prevalence=float(y[train].mean()))
                temp=dest.with_suffix('.tmp');result.to_parquet(temp,index=False);temp.replace(dest)
                joblib.dump({'pipelines':pipelines,'classifier':classifier},out/'models'/f'{model}_r{repeat}_f{fold}.joblib',compress=3)
                atomic(out/'models'/f'{model}_r{repeat}_f{fold}.json',{'best':best,'candidates':candidates,'train_patients':sorted(set(groups[train])),'test_patients':sorted(set(groups[test])),'seconds':time.time()-start})
                completed+=1;print(f"{completed}/30 {model} r{repeat} f{fold}: {best['representation']} C={best['C']} ({time.time()-start:.1f}s)",flush=True)
        atomic(out/'status.json',{'status':'evaluating','completed':completed,'total':30,'pid':os.getpid()})
        metrics=evaluate(out,cohort,args.bootstraps)
    atomic(out/'status.json',{'status':'complete','completed':completed,'total':30,'seconds':time.time()-tick,'metrics':metrics})
    atomic(out/'run_complete.json',{'runs':completed,'seconds':time.time()-tick})


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        out=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else OUTPUT
        out.mkdir(parents=True,exist_ok=True);atomic(out/'status.json',{'status':'error','error':traceback.format_exc()});traceback.print_exc();sys.exit(1)
