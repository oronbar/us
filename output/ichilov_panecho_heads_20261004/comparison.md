# PanEcho measurement-head test

## Data and extraction

The released PanEcho regression heads were applied to the **saved per-window PanEcho embeddings**, using the identical pretrained checkpoint (`896a279d...132e2da35`). Since these heads are linear, their regression outputs can be computed from the cached encoder vectors without a new GPU pass. This is not a new encoder or fine-tuned model. All 1,233 selected A2C/A3C/A4C cines yielded outputs, covering 411 visits and 103 patients. Original DICOMs and crops were not changed.

The prespecified prognosis features were estimated EF, LVEDV, LVESV, LA volume, LA diameter, TAPSE, and RV systolic velocity. LV estimates were averaged over three views, LA over A2C/A4C, and RV from A4C. These are **model estimates**, not observed clinical measurements. Predicted GLS was extracted for quality checking but excluded from the seven-feature prognosis group.

As a rough current-visit plausibility check on the 298 exactly matched GE visits, predicted GLS magnitude correlated with reported Mid-GLS at **r=0.692**; predicted EF correlated with reported biplane EF at **r=0.714** among 283 visits with both values. These comparisons were not used to select features or visits. The selected dataset has one cine per view/visit, so true *repeat-cine* consistency was not assessed. Median within-cine window SD for estimated EF was 0.35 percentage points; this is not a measure of acquisition repeatability.

## Next-visit deterioration

Same 182 non-Philips transitions, 86 patients, 38 events, retained 3×5 patient-held-out folds. Penalty selection occurred only within training patients. Earlier selected video was available for 102 transitions (15 events); the other 80 used an explicit no-history indicator. AP is area under the precision-recall curve; PPV is precision among the 20% with highest predicted risk.

| Model | AUROC | AP | PPV top 20% |
|---|---:|---:|---:|
| Retained CNN+MOMENT strain/clinical reference | **0.726** | 0.379 | 0.351 |
| PanEcho heads only, current visit | 0.405 | 0.180 | 0.162 |
| PanEcho heads only, current plus prior change | 0.566 | 0.224 | 0.135 |
| Retained reference 75% + current heads 25% | 0.719 | 0.382 | 0.351 |
| Retained reference 75% + heads history 25% | 0.720 | 0.400 | 0.324 |
| Refit compact strain/clinical reference | 0.659 | 0.352 | 0.351 |
| Refit compact reference + current heads | 0.683 | 0.360 | 0.351 |
| Refit compact reference + heads history | 0.653 | 0.350 | 0.270 |

For the refit compact comparison, adding current heads changed AUROC by **+0.023** (paired patient-bootstrap 95% interval **−0.025 to +0.076**) and AP by **+0.008** (**−0.065 to +0.085**). For fixed 25% late fusion against the retained reference, current heads changed AUROC by **−0.006** (**−0.015 to +0.003**) and AP by **+0.002** (**−0.022 to +0.029**). Heads history changed AUROC by **−0.005** (**−0.021 to +0.007**) and AP by **+0.021** (**−0.020 to +0.048**). None establishes incremental benefit. The AP increase in the history fusion is unstable and accompanied by lower top-20% precision.

## Interpretation

PanEcho's pretrained measurements carry recognizable **same-visit physiology**, but the selected seven summaries did not reliably improve this next-visit target beyond strain and clinical data. The trial is limited to three LV-strain-selected apical cines; LA/RV measurements may need other views or duplicate acquisitions. The existing selected data cannot estimate repeat-cine reliability. Repeated exploration on this small cohort is exploratory; independent patient validation remains necessary.

The next distinct test is ECG-anchored, phase-resolved motion rather than another pooled feature/head or fusion sweep. The GE source DICOMs often contain R-wave timestamps, making a short ED/ES localization pilot feasible.

Machine-readable predictions, metrics, patient-bootstrap paired deltas, and fold selections are saved in this directory. Visit- and clip-level PanEcho head outputs are in `D:\DS\ichilov3_panecho_heads_20261004`.
