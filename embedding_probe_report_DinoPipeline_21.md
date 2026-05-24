# Embedding Probe Report

## Inputs
- Embeddings parquet: `C:\Users\oron\OneDrive - Technion\Experiments\DinoPipeline_21\frame_embeddings\Ichilov_frame_embeddings_DinoPipeline_21.parquet`
- GLS report xlsx: `C:\Users\oron\OneDrive - Technion\DS\Report_Ichilov_GLS_and_Strain_oron.xlsx`
- Seed: `42`
- Test size: `0.2`

## Dataset Summary
- Rows: `54255`
- Patients: `67`
- Visits (patient+datetime): `260`
- Views: `3`
- GLS-labeled rows: `54255` (100.00%)
- GLS-labeled visits: `260`

## Probe 1: Linear Probe for View
- Classes: `3`
- Train size: `43404`
- Test size: `10851`
- Train accuracy: `0.9649`
- Test accuracy: `0.9576`
- Test balanced accuracy: `0.9576`
- Test weighted F1: `0.9576`

## Probe 2: Linear Probe for Patient ID
- Classes: `67`
- Train size: `43404`
- Test size: `10851`
- Train accuracy: `0.9012`
- Test accuracy: `0.8882`
- Test balanced accuracy: `0.8844`
- Test weighted F1: `0.8880`

## Probe 3: Visit-Change Signal vs dGLS
### All-frames visit embedding
- Visits with GLS: `260`
- Consecutive visit pairs: `193`
- Corr[L2 emb delta, |dGLS|] Pearson r: `0.2068` (p=`0.0039`)
- Corr[L2 emb delta, |dGLS|] Spearman rho: `0.2185` (p=`0.0023`)
- Corr[cosine emb delta, |dGLS|] Pearson r: `0.1879` (p=`0.0089`)
- Corr[L2 emb delta, signed dGLS] Pearson r: `-0.0399` (p=`0.5813`)
- Scatter: `D:\us\embedding_probe_report_artifacts\visit_change_all_frames_l2_vs_abs_delta_gls.png`

### ED/ES-only visit embedding
- Visits with GLS: `260`
- Consecutive visit pairs: `193`
- Corr[L2 emb delta, |dGLS|] Pearson r: `0.1905` (p=`0.0080`)
- Corr[L2 emb delta, |dGLS|] Spearman rho: `0.2092` (p=`0.0035`)
- Corr[cosine emb delta, |dGLS|] Pearson r: `0.1647` (p=`0.0221`)
- Corr[L2 emb delta, signed dGLS] Pearson r: `-0.0365` (p=`0.6146`)
- Scatter: `D:\us\embedding_probe_report_artifacts\visit_change_edes_frames_l2_vs_abs_delta_gls.png`

## Interpretation Guide
- Probe 1 high score means view information is linearly recoverable from embeddings.
- Probe 2 near-perfect score means embeddings strongly encode patient identity (high leakage risk for patient-overlapping splits).
- Probe 3 near-zero correlations mean embedding change is weakly aligned with visit-level GLS change.