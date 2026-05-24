# Ichilov Pipeline5

Pipeline5 predicts DICOM/view-level global longitudinal strain curves, peak GLS, and normalized time-to-peak from phase-aligned echocardiography cine DICOMs.

## Run

```powershell
python ichilov_pipeline5.py
```

Use a custom config:

```powershell
set ICHILOV_PIPELINE5_CONFIG=path\to\ichilov_pipeline5.yaml && python ichilov_pipeline5.py
```

## Main Artifacts

- `validate_reports/validation_report.md`: detected Excel columns, duplicate keys, unmatched rows, missing labels.
- `curve_dataset/prepared_strain_curve_dataset.parquet`: merged DICOM/view table with ED/ES, parsed curves, resampled curves, GLS labels, and quality heuristics.
- `phase_embeddings/phase_frame_embeddings.parquet`: one row per DICOM/view with selected ED/ES phase frame indices and frozen DINOv2 `[T,D]` embeddings.
- `strain_curve_model/strain_curve_model_best.pt`: best temporal model checkpoint from a patient-grouped train/validation split.
- `evaluation/predictions.parquet`: true/predicted curves, peak GLS, time-to-peak, and quality fields.
- `evaluation/plots/`: true-vs-predicted curves, peak GLS scatter, Bland-Altman plot, and time-to-peak scatter.
- `quality_report/quality_report.md`: VVI curve quality and residual-based label-noise diagnostics.

All paths and column mappings are YAML-configurable. Leave `reports.column_map` entries as `null` to let the scripts infer the current report schema and save `resolved_column_map.json`.
