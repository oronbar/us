"""Plot held-out GLS predictions by scanner vendor and audit Philips errors.

Reads existing derived artifacts only. Writes new figures and tables under the
temporal-trial directory; does not modify source DICOMs or prior predictions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from PIL import Image, ImageDraw

BASE = Path(r'D:\DS\ichilov3_temporal_trial_20260927')
OUT = BASE / 'vendor_failure_analysis'
OUT.mkdir(exist_ok=True)
PRED = BASE / 'physiology' / 'oof_predictions.parquet'
CLIPS = Path(r'D:\DS\ichilov3_preprocessing_audit_20260927\clip_audit.parquet')
SAMPLE = BASE / 'sampling_audit.parquet'
MANIFEST = Path(r'D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json')

pred = pd.read_parquet(PRED)
pred = pred[(pred.model == 'echoprime') & pred.target.isin(['mid_gls','endo_gls'])].copy()
assert pred.groupby('target').size().to_dict() == {'endo_gls':398,'mid_gls':398}
manifest = pd.DataFrame(json.loads(MANIFEST.read_text(encoding='utf-8')))
meta = manifest.groupby(['patient','visit_date']).agg(
    manufacturer=('manufacturer','first'),
    all_bookmark=('selection_source',lambda x: bool(x.eq('TOMTEC bookmark').all())),
    review_flags=('status',lambda x: int(x.eq('needs_review').sum())),
).reset_index().rename(columns={'patient':'patient_id'})
aligned = pd.read_parquet(Path(r'D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet'))
visits = aligned[['visit_id','patient_id','visit_date']].merge(meta,on=['patient_id','visit_date'],validate='many_to_one')
df = pred.merge(visits,on=['visit_id','patient_id'],validate='many_to_one')
assert len(df) == 796 and df.groupby('target').manufacturer.value_counts().to_dict()[('mid_gls','Philips Medical Systems')] == 100
df['error'] = df.prediction - df.value
df['absolute_error'] = df.error.abs()
df['vendor_short'] = df.manufacturer.map({'GE Healthcare Ultrasound':'GE Healthcare','GE Vingmed Ultrasound':'GE Vingmed','Philips Medical Systems':'Philips'})
df.to_csv(OUT/'prediction_points.csv',index=False)

def patient_bootstrap_ci(g, metric, reps=2000, seed=27):
    rng = np.random.default_rng(seed)
    groups = [x[['value','prediction','absolute_error']].to_numpy() for _,x in g.groupby('patient_id')]
    values=[]
    for _ in range(reps):
        sampled = np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if metric == 'r':
            if sampled[:,0].std() == 0 or sampled[:,1].std() == 0: continue
            values.append(float(np.corrcoef(sampled[:,0],sampled[:,1])[0,1]))
        else: values.append(float(sampled[:,2].mean()))
    return np.quantile(values,[.025,.975]).tolist()

metrics=[]
for (target,vendor),g in df.groupby(['target','vendor_short']):
    r,p=pearsonr(g.value,g.prediction)
    metrics.append(dict(target=target,vendor=vendor,n_visits=len(g),n_patients=g.patient_id.nunique(),
                        pearson_r=r,pearson_p_unadjusted=p,r_ci95_patient_cluster=patient_bootstrap_ci(g,'r'),
                        mae=g.absolute_error.mean(),mae_ci95_patient_cluster=patient_bootstrap_ci(g,'mae'),
                        mean_signed_error=g.error.mean(),gt_sd=g.value.std(),gt_min=g.value.min(),gt_max=g.value.max()))
metrics_df=pd.DataFrame(metrics)
metrics_df.to_csv(OUT/'vendor_metrics.csv',index=False)

vendors=['GE Healthcare','GE Vingmed','Philips']
colors={'GE Healthcare':'#1764a5','GE Vingmed':'#358f8e','Philips':'#d36b28'}
fig,axs=plt.subplots(2,3,figsize=(15,9),sharex='row',sharey='row',layout='constrained')
for row,target in enumerate(['mid_gls','endo_gls']):
    all_t=df[df.target==target]
    lim=(min(all_t.value.min(),all_t.prediction.min())-0.7,max(all_t.value.max(),all_t.prediction.max())+0.7)
    for col,vendor in enumerate(vendors):
        ax=axs[row,col]; g=all_t[all_t.vendor_short==vendor]
        ax.scatter(g.value,g.prediction,s=23,c=colors[vendor],alpha=.68,edgecolors='none')
        ax.plot(lim,lim,'k--',lw=1,alpha=.7,label='Perfect prediction')
        ax.set(xlim=lim,ylim=lim,aspect='equal')
        m=metrics_df[(metrics_df.target==target)&(metrics_df.vendor==vendor)].iloc[0]
        ax.set_title(f'{vendor}  |  {target.replace("_","-")}\nn={len(g)} visits / {m.n_patients} patients; r={m.pearson_r:.2f}; MAE={m.mae:.2f}',fontsize=10)
        if row==1: ax.set_xlabel('Report GLS, absolute %')
        if col==0: ax.set_ylabel('Predicted GLS, absolute %')
        ax.grid(alpha=.16)
fig.suptitle('EchoPrime heartbeat-sampled video → same-visit GLS\nPatient-held-out out-of-fold predictions; dashed line = perfect agreement',fontsize=14)
fig.savefig(OUT/'gls_pred_vs_gt_by_vendor.png',dpi=210,bbox_inches='tight')
fig.savefig(OUT/'gls_pred_vs_gt_by_vendor.pdf',bbox_inches='tight')
plt.close(fig)

# Visit-level clip metadata. These are exploratory covariates, never model inputs.
clip=pd.read_parquet(CLIPS).rename(columns={'patient':'patient_id'})
sample=pd.read_parquet(SAMPLE).rename(columns={'patient':'patient_id'})
clip=clip.merge(sample[['file_id','duration_seconds','resampled','timing_source','estimated_beats_per_window',
                         'maximum_frame_quantization_error_seconds']],on='file_id',validate='one_to_one')
assert clip.groupby(['patient_id','visit_date']).size().eq(3).all()
clip['fps_from_frame_time']=1000/clip.frame_time_ms
numeric=['frames','heart_rate','frame_time_ms','fps_from_frame_time','duration_seconds','removed_foreground_max',
         'colored_fraction','hue_concentration','estimated_beats_per_window','maximum_frame_quantization_error_seconds']
wide=clip.pivot(index=['patient_id','visit_date'],columns='view',values=numeric)
wide.columns=[f'{field}_{view}' for field,view in wide.columns]
wide=wide.reset_index()
agg=clip.groupby(['patient_id','visit_date']).agg(scanner=('scanner',lambda x:x.mode().iloc[0]),
    n_short_cines=('resampled',lambda x:int((~x).sum())),
    n_crop_flagged=('crop_flagged','sum'),
    n_tint_candidates=('tint_candidate','sum'),
    n_doppler_flags=('doppler_pixel_heuristic','sum'),
    min_frames=('frames','min'),min_duration=('duration_seconds','min'),
    max_crop_removed=('removed_foreground_max','max')).reset_index()
mid=df[df.target=='mid_gls'].merge(agg,on=['patient_id','visit_date'],validate='one_to_one').merge(wide,on=['patient_id','visit_date'],validate='one_to_one')
phil=mid[mid.vendor_short=='Philips'].copy()
ge=mid[mid.vendor_short=='GE Healthcare'].copy()
phil.to_csv(OUT/'philips_visit_errors.csv',index=False)

# Static tri-view sheet to guide a focused source-cine review. A middle frame
# cannot establish diagnostic quality, so no automated clip exclusions follow.
review_cases=pd.concat([phil.nlargest(8,'absolute_error'),phil.nsmallest(4,'absolute_error')])
canvas=Image.new('RGB',(3*260,12*210),'white')
draw=ImageDraw.Draw(canvas)
for row,(_,visit) in enumerate(review_cases.iterrows()):
    subset=manifest[(manifest.patient==visit.patient_id)&(manifest.visit_date==visit.visit_date)]
    for col,view in enumerate(['A2C','A3C','A4C']):
        item=subset[subset.view==view].iloc[0]
        with np.load(item['crop_output']) as cine:
            frames=cine['frames']
            frame=frames[len(frames)//2].copy()
        thumb=Image.fromarray(frame).resize((190,175),Image.Resampling.BILINEAR)
        x,y=col*260,row*210
        canvas.paste(thumb,(x,y+30))
        draw.text((x+5,y+4),f'{visit.visit_id} {view} GT={visit.value:.1f} pred={visit.prediction:.1f}',fill='black')
canvas.save(OUT/'philips_failure_contact_sheet.jpg',quality=88)

# One-priority view of failures; no threshold is used to alter training data.
phil['large_error_2p5']=phil.absolute_error>=2.5
rows=[]
for key in ['scanner','all_bookmark','n_short_cines','n_crop_flagged','n_tint_candidates','n_doppler_flags']:
    for val,g in phil.groupby(key,dropna=False):
        rows.append(dict(feature=key,group=str(val),n_visits=len(g),n_patients=g.patient_id.nunique(),
                         mae=g.absolute_error.mean(),mean_signed_error=g.error.mean(),large_error_fraction=g.large_error_2p5.mean()))
pd.DataFrame(rows).to_csv(OUT/'philips_subgroups.csv',index=False)

associations=[]
for key in numeric:
    for view in ['A2C','A3C','A4C']:
        col=f'{key}_{view}'
        if phil[col].notna().sum()<30 or phil[col].nunique()<4: continue
        rho,p=spearmanr(phil[col],phil.absolute_error,nan_policy='omit')
        associations.append(dict(feature=col,n=int(phil[col].notna().sum()),rho_abs_error=rho,p_unadjusted=p,
                                 median=phil[col].median(),q25=phil[col].quantile(.25),q75=phil[col].quantile(.75)))
associations=pd.DataFrame(associations).sort_values('rho_abs_error',key=lambda s:s.abs(),ascending=False)
associations.to_csv(OUT/'philips_feature_associations.csv',index=False)

# Severity-adjusted descriptive comparison: compare vendors in broad common GT bins.
bins=[-np.inf,12,16,20,np.inf]
mid['gt_bin']=pd.cut(mid.value,bins,labels=['<12','12–16','16–20','≥20'])
bin_table=mid.groupby(['gt_bin','vendor_short'],observed=True).agg(n=('absolute_error','size'),mae=('absolute_error','mean'),
                                mean_error=('error','mean')).reset_index()
bin_table.to_csv(OUT/'mid_gls_target_bins.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for vendor in vendors:
    g=mid[mid.vendor_short==vendor]
    axs[0].scatter(g.value,g.error,s=20,alpha=.52,label=f'{vendor} (n={len(g)})',color=colors[vendor])
    axs[1].hist(g.absolute_error,bins=np.arange(0,7.6,.5),histtype='step',density=True,lw=2,label=vendor,color=colors[vendor])
axs[0].axhline(0,color='black',ls='--',lw=1)
axs[0].set(xlabel='Report Mid-GLS, absolute %',ylabel='Predicted − report Mid-GLS (points)',title='Bias versus target severity')
axs[1].set(xlabel='Absolute Mid-GLS error (points)',ylabel='Density',title='Error distribution')
for ax in axs: ax.legend(fontsize=8);ax.grid(alpha=.16)
fig.savefig(OUT/'mid_gls_error_diagnostics.png',dpi=210,bbox_inches='tight')
plt.close(fig)

report=['# Vendor prediction and Philips failure audit','',
    'Frozen EchoPrime, estimated-heartbeat temporal sampling, same-visit GLS. Predictions are from patient-held-out folds. Philips results are descriptive and do not identify a causal vendor effect.',
    '', '## Correlation and error by vendor','',
    '| Target | Vendor | Visits / patients | Pearson r (patient-bootstrap 95% CI) | MAE (95% CI) | GT SD |',
    '|---|---|---:|---:|---:|---:|']
for _,m in metrics_df.iterrows():
    report.append(f'| {m.target} | {m.vendor} | {m.n_visits} / {m.n_patients} | {m.pearson_r:.3f} ({m.r_ci95_patient_cluster[0]:.3f}–{m.r_ci95_patient_cluster[1]:.3f}) | {m.mae:.3f} ({m.mae_ci95_patient_cluster[0]:.3f}–{m.mae_ci95_patient_cluster[1]:.3f}) | {m.gt_sd:.2f} |')
report+=['','## Philips Mid-GLS','',
         f'{len(phil)} visits from {phil.patient_id.nunique()} patients; {phil.large_error_2p5.sum()} visits have ≥2.5-point error.',
         f'Mean signed error {phil.error.mean():+.2f} points; target SD {phil.value.std():.2f} versus {ge.value.std():.2f} in GE Healthcare.',
         'Scanner counts: '+', '.join(f'{k}: {v}' for k,v in phil.scanner.value_counts().items())+'.',
         'The scanner model comparison is underpowered because nearly all Philips visits come from iE33.',
         '', 'Largest absolute-error feature associations (exploratory; unadjusted and correlated across views):','']
for _,x in associations.head(8).iterrows(): report.append(f'- {x.feature}: Spearman rho {x.rho_abs_error:+.2f} (n={x.n}).')
report+=['','Target-bin comparison (MAE; number of visits in parentheses):','']
for gtbin,g in bin_table.groupby('gt_bin',observed=True):
    report.append('- '+str(gtbin)+': '+', '.join(f'{x.vendor_short} {x.mae:.2f} ({x.n})' for _,x in g.iterrows()))
report+=['','Interpretation: The plots and correlations describe held-out predictions, but vendor groups may differ in target spread, patient mix, label workflow and scanner generation. Feature associations cannot distinguish preprocessing failure from disease severity or acquisition practice. Review high-error Philips source clips and report provenance before deciding on filtering or vendor-specific modeling. No input data or prior outputs were changed.']
(OUT/'results.md').write_text('\n'.join(report),encoding='utf-8')
print(f'Wrote {OUT} ({len(df)} prediction rows, {len(phil)} Philips visits).')
