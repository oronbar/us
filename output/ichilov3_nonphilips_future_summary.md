# Next-visit deterioration: non-Philips current cines

Exploratory rerun on 182 transitions from 86 patients (38 deterioration events). We excluded 56 transitions whose **current-visit** three-view cine set was made by Philips. The retained current cines are 147 GE Healthcare and 35 GE Vingmed transitions. Labels remain the next-visit ≥15% relative Mid-GLS deterioration endpoint. The target visit supplies no input video.

| Experiment | Model | AUROC | AP |
|---|---|---:|---:|
| Retained reference, evaluated on same subset | CNN+MOMENT strain/clinical | 0.726 | 0.379 |
| Refit on non-Philips only | strain/clinical Ridge | 0.695 | 0.322 |
| Refit on non-Philips only | strain/clinical + EchoPrime Ridge | 0.675 | 0.328 |
| Refit on non-Philips only | strain/clinical + PanEcho Ridge | 0.633 | 0.376 |
| Refit video on non-Philips only; fixed 25% blend with retained reference | EchoPrime late fusion | 0.701 | 0.347 |
| Refit video on non-Philips only; fixed 25% blend with retained reference | PanEcho late fusion | 0.722 | 0.378 |
| Refit video on non-Philips only | EchoPrime video only | 0.387 | 0.168 |
| Refit video on non-Philips only | PanEcho video only | 0.486 | 0.211 |

The video models were re-trained within the retained patient-held-out 3×5 folds; their regularization and representation were selected using grouped inner folds. For the early-fusion comparison, both the strain/clinical Ridge and the fused Ridge models were re-trained only on non-Philips examples. The retained CNN+MOMENT reference was **not** re-trained; its original patient-held-out predictions were evaluated on the same 182 examples. This distinction matters when interpreting late fusion.

No tested video model improved both AUROC and AP over the strongest reference. The 25% PanEcho blend was nearly identical to it (AUROC difference −0.004; AP difference −0.001). The 25% EchoPrime blend lowered AUROC by 0.024 (paired patient-bootstrap 95% CI −0.048 to −0.003). In the fully re-trained Ridge comparison, video additions did not improve AUROC; the apparent PanEcho AP increase of 0.054 has a paired 95% CI of −0.049 to +0.150.

The reference AUROC is 0.726 on this subset, versus 0.698 on the full cohort. This is a change in evaluation cohort, not evidence that removing Philips improved the model. Vendor is an acquisition proxy, not a verified cine-quality label. The subgroup and model choices were explored on the same cohort, so these results need independent confirmation.

Source DICOMs and prior results were not changed. Detailed out-of-fold predictions, fold audit, and bootstrap intervals are in the `late_fusion` and `ridge_fusion` subdirectories.
