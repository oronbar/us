"""Consolidate cohort coverage, validation limitations and paired prediction results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dicom_prediction_coverage import main as summarize_coverage
from dicom_prediction_calibration_diagnostic import main as calibration_diagnostic

OUT=Path('D:/us/output/dicom_prediction')


def main():
    summarize_coverage()
    calibration_diagnostic()
    calibration=json.loads((OUT/'calibration_diagnostic.json').read_text())['results']
    inventory=json.loads((OUT/'inventory_summary.json').read_text())
    spatial=pd.read_parquet(OUT/'spatial_manifest.parquet')
    views=pd.read_parquet(OUT/'view_manifest.parquet')
    selected=pd.read_parquet(OUT/'selected_clips.parquet')
    apical=['A2C','A3C','A4C']
    native_path=OUT/'view_manifest_native_rgb.parquet'
    native=pd.read_parquet(native_path) if native_path.exists() else None
    native_apical=native[native.view.isin(apical)].study_uid.nunique() if native is not None else None
    corrected_apical=views[views.view.isin(apical)].study_uid.nunique()
    tint_count=int(views.appearance.eq('monochrome-tint-v1').sum())
    coverage_note=(f'Confident apical-view coverage increased from **{native_apical} to {corrected_apical} studies** before final clip screening. '
                   if native_apical is not None else f'Confident apical-view coverage is **{corrected_apical} studies** before final clip screening. ')
    cohort=pd.read_parquet(OUT/'evaluation_cohort.parquet')
    metrics=pd.read_csv(OUT/'paired_metrics.csv').set_index('model')
    deltas=pd.read_csv(OUT/'paired_deltas.csv').set_index('model')
    names={
        'retained_strain_clinical':'Existing strain + clinical',
        'strain_clinical_plus_echoprime':'+ EchoPrime (25% fixed blend)',
        'strain_clinical_plus_panecho':'+ PanEcho (25% fixed blend)',
        'clinical_ridge_refit':'Clinical ridge, refitted',
        'acquisition_only_control':'Acquisition-only control',
        'echoprime_apical':'EchoPrime: apical',
        'panecho_apical':'PanEcho: apical',
        'echoprime_apical_history':'EchoPrime: apical + history',
        'panecho_apical_history':'PanEcho: apical + history',
        'echoprime_allviews':'EchoPrime: apical + parasternal',
        'panecho_allviews':'PanEcho: apical + parasternal',
    }
    order=[k for k in names if k in metrics.index];y=np.arange(len(order))
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for ax,metric,title in zip(axes,['auc','ap'],['AUROC','Average precision']):
        r=metrics.loc[order];value=r[metric].to_numpy();lo=r[metric+'_lo'].to_numpy();hi=r[metric+'_hi'].to_numpy()
        # Percentile intervals need not contain the original point estimate.
        for i,key in enumerate(order):
            color='#174e87' if key=='retained_strain_clinical' else ('#b86519' if key.startswith('strain_clinical_plus') else '#65758a')
            ax.plot([lo[i],hi[i]],[i,i],color=color,lw=2)
            ax.scatter(value[i],i,color=color,s=35,zorder=3)
        ax.axvline(.5 if metric=='auc' else cohort.label__mid_first_rel15.mean(),color='#aaa',ls=':',lw=1)
        ax.set_title(title);ax.set_xlim(0,1);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Estimate and 95% patient-bootstrap interval')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,[names[x] for x in order]);axes[0].invert_yaxis()
    fig.suptitle(f'Frozen video adds to strain?  {len(cohort)} transitions · {cohort.patient_id.nunique()} patients',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/'paired_performance.png',dpi=180);plt.close(fig)
    summary=dict(inventory=inventory,spatial_status=spatial.status.value_counts().to_dict(),
        view_counts=views.view.value_counts().to_dict(),selected_clips=len(selected),
        tint_normalized_clips=tint_count,native_confident_apical_studies=native_apical,
        corrected_confident_apical_studies=corrected_apical,
        selected_studies=selected.study_uid.nunique(),selected_views=selected.view.value_counts().to_dict(),
        evaluation_transitions=len(cohort),evaluation_patients=cohort.patient_id.nunique(),events=int(cohort.label__mid_first_rel15.sum()))
    (OUT/'experiment_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# DICOM + strain experiment: first results','',
        f"Inventoried **{inventory['files']:,} extensionless DICOMs**. Exact Study UID matches cover **{inventory['matched_strain_studies']} of {inventory['strain_studies']} strain visits**. The {inventory['unmatched_strain_studies']} unmatched visits were not joined by guesswork. Two same-patient/same-date candidates are separately listed for review.",'',
        f"Spatial processing produced **{len(spatial[spatial.status.eq('ok')]):,}** tissue-cropped video previews. **{len(spatial[spatial.status.eq('excluded_declared_color_doppler')]):,}** declared color-Doppler videos were excluded from this initial B-mode branch. **{len(selected):,}** clips from **{selected.study_uid.nunique()}** studies were selected for both frozen encoders.",'',
        f"A visual input audit found amber B-mode palettes being mistaken for Doppler. A deterministic single-hue detector normalized **{tint_count:,}** previews to value-channel grayscale for classification and applied the same transformation to encoder windows. "+coverage_note+"Raw previews and DICOMs remain unchanged. Multihue red/blue flow is not normalized by this rule. This correction was chosen before prediction results were evaluated; its clinical validation remains pending.",'',
        f"The paired prediction cohort contains **{len(cohort)} transitions, {cohort.patient_id.nunique()} patients and {int(cohort.label__mid_first_rel15.sum())} events**. The task remains next-visit first >=15% relative Mid-GLS deterioration. Clinical disease adjudication is not available.",'',
        '| Comparison | AUROC | AP | Delta AUROC vs existing (95% CI) | Delta AP (95% CI) |','|---|---|---|---|---|']
    for key in ['retained_strain_clinical','strain_clinical_plus_echoprime','strain_clinical_plus_panecho']:
        r=metrics.loc[key];d=deltas.loc[key]
        lines.append(f"| {names[key]} | {r.auc:.3f} | {r.ap:.3f} | {d.delta_auc:+.3f} ({d.delta_auc_lo:+.3f}, {d.delta_auc_hi:+.3f}) | {d.delta_ap:+.3f} ({d.delta_ap_lo:+.3f}, {d.delta_ap_hi:+.3f}) |")
    primary=['strain_clinical_plus_echoprime','strain_clinical_plus_panecho']
    if all(deltas.loc[k,'delta_auc_lo']<=0<=deltas.loc[k,'delta_auc_hi'] for k in primary):
        lines+=['','Neither primary video addition demonstrated an AUROC gain: both paired 95% intervals include zero. This result applies to these frozen encoders, preprocessing and low-capacity probes; it does not establish that the raw videos contain no useful information.']
    prior=calibration['training_prior_blend']
    lines+=['',f"**Post-hoc calibration diagnostic (added after the primary results):** the retained model's Brier error is {metrics.loc['retained_strain_clinical','brier']:.4f}; the EchoPrime and PanEcho blends score {metrics.loc[primary[0],'brier']:.4f} and {metrics.loc[primary[1],'brier']:.4f}. A fixed blend toward outer-training prevalence, using no image information, scores **{prior['brier']:.4f}** with AUROC **{prior['auc']:.3f}**. This control tests whether probability shrinkage can explain the Brier change. It is not a tuned replacement model or a new primary endpoint. Full paired differences are in [calibration_diagnostic.json](calibration_diagnostic.json)."]
    lines+=['','![Paired performance](paired_performance.png)','',
        'The two primary additions use a fixed 25% video / 75% existing-model probability blend. No weight was selected on held-out results. New model hyperparameters and preprocessing were selected within patient-grouped inner folds. The retained baseline was trained on its original outer-training cohort; new probes use the available-video subset. All displayed models are scored on identical examples.','',
        'View selection and simple quality screening remain automated. The pilot gallery is ready for clinician review; confidence is not a validated quality score. Original strain clips have not been identified. Crops can retain ECG/measurement annotations, and native fixed-stride windows do not guarantee full-cycle coverage. Additional-view/history models are exploratory controls, not a search for a reportable winner.','',
        'This cohort has supported repeated previous experiments. Patient-bootstrap intervals do not account for all model-development uncertainty or confirm external validity. Inspect paired intervals before interpreting a point improvement.','',
        'The next review should check apical view labels, crop quality and temporal coverage on the pilot before adding model complexity. Keep the existing strain baseline for now. Any subsequent preprocessing or modeling experiment should be recorded as a new development iteration, rather than presented as confirmation on untouched data.','',
        'Useful outputs: [pilot review](pilot_review.html), [all model results](prediction_report.md), [paired differences](paired_deltas.csv), [missing visits](unmatched_strain_visits.csv), [protocol](evaluation_protocol.json), and [source validation](strain_source_validation.json).']
    (OUT/'experiment_report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
