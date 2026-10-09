# Frozen EchoJEPA comparison

## Encoding

- Official EchoJEPA V-JEPA 2 ViT-L checkpoint `vitl-vmix22m-pt220-c55.pt`; repository commit `77208e1a8521c0e82f79579eebe4b22f46c5ccbd`.
- Same 1,233 selected, previously cropped cines as the earlier encoder trial: A2C/A3C/A4C across 411 visits and 103 patients. No original DICOM was modified.
- Native-color 224-pixel input, 16 frames over an HR-estimated heartbeat, up to four windows per cine. Mean final-layer token embedding per window; visit representation has three 1024-dimensional view vectors. All window embeddings and frame indices are retained.
- Encoding completed in 1,681 seconds with 0 failed clips. Exact source manifest and checkpoint hashes are recorded in `D:\DS\ichilov3_echojepa_20261004\protocol.json`.

## Held-out next-visit deterioration

Same 182 transitions, 86 patients, 38 events, retained 3×5 patient-held-out folds. Inner selection uses training patients only. Average precision (AP) is the area under the precision-recall curve; PPV is precision within the top 20% predicted risk.

| Input | AUROC | AP | PPV top 20% |
|---|---:|---:|---:|
| Strain + clinical reference | 0.726 | 0.379 | 0.351 |
| EchoJEPA current video only | 0.516 | 0.230 | 0.216 |
| EchoJEPA current + previous-visit video delta | 0.524 | 0.218 | 0.216 |
| 75% reference + 25% EchoJEPA current | 0.720 | 0.368 | 0.351 |
| 75% reference + 25% EchoJEPA history | 0.715 | 0.319 | 0.378 |

The paired patient-bootstrap AUROC difference for the best AUROC EchoJEPA fusion versus the reference is −0.006 (95% interval −0.035 to 0.024); AP difference is −0.011 (−0.090 to 0.074). Thus this trial does not establish an improvement. History raises video-only AUROC slightly but lowers AP and does not help the fused model. The higher history-fusion PPV at one operating point is not supported by AUROC or AP.

## Same-visit Mid-GLS

Same 298 GE visits, 89 patients, retained patient folds: EchoJEPA MAE 1.840 GLS percentage points, Pearson r 0.576. Previous EchoPrime on this cohort: MAE 1.340, r 0.809. Previous EchoFM: MAE 1.546, r 0.734.

## Interpretation

This is a **frozen, mean-pooled representation** test. It does not rule out EchoJEPA with a trainable attentive or temporal probe, and it does not evaluate EchoJEPA fine-tuning. Since both next-visit discrimination and GLS probing are weaker than the existing alternatives, this result alone does not justify launching a costly EchoJEPA fine-tuning run. If we revisit this encoder, first test a leakage-safe, low-capacity probe that uses the retained window and token structure on the same folds.

Machine-readable per-transition and per-visit predictions, fold selections, metrics, and paired uncertainty are in this directory.
