"""Plot saved patient-held-out predictions; does not refit models."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

OUT = Path(r'D:\DS\ichilov3_physiology_audit_20260927')
data = pd.read_parquet(OUT / 'oof_predictions.parquet')
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
rows = []
for i, target in enumerate(['mid_gls', 'endo_gls']):
    for j, model in enumerate(['echoprime', 'panecho']):
        g = data[(data.target == target) & (data.model == model)]
        y, p = g.value.to_numpy(), g.prediction.to_numpy()
        e = p-y
        row = dict(target=target, model=model, visits=len(g), patients=g.patient_id.nunique(),
                   mae=float(np.abs(e).mean()), median_absolute_error=float(np.median(np.abs(e))),
                   rmse=float(np.sqrt(np.mean(e**2))), r2=float(r2_score(y,p)),
                   pearson_r=float(np.corrcoef(y,p)[0,1]), bias=float(e.mean()),
                   difference_sd=float(e.std(ddof=1)), prediction_vs_gt_slope=float(np.polyfit(y,p,1)[0]))
        rows.append(row)
        ax = axes[i,j]
        ax.scatter(y,p,s=23,alpha=.48,color=['#007f9e','#c56620'][j],edgecolors='none',rasterized=True)
        lim = (3,31) if i else (3,28)
        ax.plot(lim,lim,'--',color='#555555',lw=1.3,label='Perfect agreement')
        ax.set(xlim=lim,ylim=lim,aspect='equal',xlabel='Measured GLS magnitude (%)',ylabel='Predicted GLS magnitude (%)',
               title=f'{["EchoPrime","PanEcho"][j]} — {["Mid-wall","Endocardial"][i]} GLS')
        ax.text(.04,.96,f'n = {len(g)} visits / {g.patient_id.nunique()} patients\nMAE = {row["mae"]:.2f} percentage points\nMedian absolute error = {row["median_absolute_error"]:.2f}\nR² = {row["r2"]:.2f}; r = {row["pearson_r"]:.2f}',
                transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ax.grid(alpha=.15)
fig.suptitle('GLS ground truth versus video-only prediction\nPatient-held-out predictions averaged over 3 CV repeats',fontsize=15)
fig.savefig(OUT/'gls_gt_vs_prediction.png',dpi=190)
fig.savefig(OUT/'gls_gt_vs_prediction.pdf')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
for ax,target,title in zip(axes,['mid_gls','endo_gls'],['Mid-wall GLS','Endocardial GLS']):
    g=data[(data.target==target)&(data.model=='echoprime')]
    y,p=g.value.to_numpy(),g.prediction.to_numpy();e=p-y;bias=e.mean();sd=e.std(ddof=1)
    ax.scatter((y+p)/2,e,s=23,alpha=.45,color='#007f9e',edgecolors='none')
    ax.axhline(bias,color='#333333',label=f'Bias {bias:.2f}')
    for v in [bias-1.96*sd,bias+1.96*sd]: ax.axhline(v,ls='--',color='#b9513b',label=f'{v:.2f}')
    ax.set(title=title,xlabel='Mean of measured and predicted magnitude (%)',ylabel='Prediction − measurement (percentage points)')
    ax.legend(fontsize=9);ax.grid(alpha=.15)
fig.suptitle('EchoPrime agreement: descriptive Bland–Altman plots\nRepeated visits per patient; lines are descriptive, not independent-sample inference',fontsize=12)
fig.savefig(OUT/'gls_agreement.png',dpi=190);fig.savefig(OUT/'gls_agreement.pdf');plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'gls_plot_metrics.csv',index=False)
(OUT/'gls_plot_metrics.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
