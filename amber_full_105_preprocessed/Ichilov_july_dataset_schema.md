# Ichilov July Parquet dataset schema

`Ichilov_july_dataset.parquet` is one row per exported curve series. It retains all 139 analysis instances and explicitly identifies the 128 true visits.

Original variable-length arrays and 96-point phase-resampled arrays are both stored. The main longitudinal analysis averages technical reanalyses within Study UID; the raw rows are never discarded.

| Column | Meaning |
| --- | --- |
| `analysis_id` | One source export/analysis instance; filename stem. |
| `patient_id` | Patient identifier parsed from the filename and kept as text. |
| `study_uid` | DICOM Study UID; the true-visit identity. |
| `study_datetime` | Clinical visit time from the file's Study Date and Time. |
| `visit_id` | Patient-local ordered visit ID; technical reanalyses share it. |
| `visit_order` | Chronological true-visit order within patient. |
| `technical_replicate_index` | Order of reanalysis exports for the same Study UID. |
| `technical_replicate_count` | Number of exports for that Study UID (1 or 2). |
| `curve_family` | Strain, strain-rate, global strain/rate, or EF volume. |
| `layer` | Endo or Mid where applicable. |
| `view` | A2C/A3C/A4C, Biplane, or all_apical. |
| `segment_number` | Vendor segment 1–18; null for Total/global curves. |
| `segment_ring` | Basal (1–6), mid (7–12), or apical (13–18). |
| `segment_view` | A2C/A3C/A4C membership inferred from segment topology. |
| `time_ms` | Original variable-length time axis as a Parquet list<float>. |
| `values` | Original curve samples as a Parquet list<float>. |
| `resampled_values` | Curve resampled to normalized phase for comparison. |
| `peak_abs` | Absolute magnitude of the minimum strain value. |
| `time_to_peak_ms` | Time of minimum strain from cycle start. |
| `time_to_peak_norm` | Time-to-peak divided by cycle duration. |
| `post_systolic_index` | Additional shortening after ES, zero if peak is not after ES. |

The Parquet file also repeats visit-level EF/GLS scalars on each curve row and contains engineered curve features plus PCA coordinates (`latent_01` …).

GLS is stored signed in the vendor scalar columns. The companion visit table adds positive magnitudes, where a decrease means deterioration.