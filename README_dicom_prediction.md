# Raw DICOM + strain prediction experiment

This pipeline inventories extensionless DICOM files, links them to strain studies,
screens views, caches spatial previews, extracts frozen EchoPrime/PanEcho features,
and evaluates their incremental value for the existing next-visit endpoint.
Source DICOMs and reports are read-only. Local artifacts, identifiers and model
weights live in the git-ignored `D:/us/output/dicom_prediction/` directory.

## Run or resume

From `D:/us`, using the existing environment:

```powershell
.venv\Scripts\python.exe -m unittest test_dicom_prediction -v
.venv\Scripts\python.exe dicom_prediction_setup.py
.venv\Scripts\python.exe dicom_prediction_inventory.py
.venv\Scripts\python.exe dicom_prediction_video.py crop --all
.venv\Scripts\python.exe dicom_prediction_video.py classify
.venv\Scripts\python.exe dicom_prediction_encode.py
.venv\Scripts\python.exe dicom_prediction_evaluate.py
.venv\Scripts\python.exe dicom_prediction_report.py
```

After setup, `dicom_prediction_run.py --from-stage inventory` runs the stages
sequentially. Use `--from-stage encode` to resume after completed classification.
Progress is recorded in `run_status.json` and `logs/` under the output directory.

The original pilot uses `crop` without `--all` (24 studies) and encoding with
`--pilot-only`. Do not run classification or encoding before the preceding stage
has completed. Caches permit interrupted runs to resume. For changed model or
preprocessing implementations use a fresh output directory; cache reuse is intended
for the same protocol and source data, not arbitrary code changes.

## Inputs and matching

- Reports: `D:/DS/anonymized_reports` (416 exports, 400 unique studies).
- DICOMs: data folders named `echo_*` under `E:/`. Suffixes are ignored.
- Patient mapping: `E:/anonymization_mapping_hashed.xlsx` (read locally).
- Existing visits, outcomes and baseline: the repo's `amber_full_105_preprocessed`,
  `cardiotoxicity_next_visit_gpu_results` and `cardiotoxicity_timeseries_round4_results`.
- Grouped folds: `cardiotoxicity_cnn_length_ablation_results/cnn_length_ablation_patient_folds.csv`.

Use exact Study UID matches and verify mapped patient identity. Same-patient,
same-date matches with different Study UIDs are review candidates, not automatic
matches. Do not replace the true baseline or preceding visit with a later study.
Technical reanalyses are retained in the report inventory and already consolidated
into the existing visit/outcome tables. Mid/Endo GLS values were independently
re-read from all supplied reports and matched the existing 400 visits exactly.

## Spatial preprocessing and view selection

The canonical full-length video remains the source DICOM. The spatial cache stores
the DICOM tissue-region crop geometry, five preview frame indices, image summaries
and previews. It does not replace the original with a shortened video. RGB decoding
uses pydicom, including its photometric conversion. Only verified uint8 RGB or
grayscale-derived RGB pixels are accepted. Crops are clamped to image bounds and
resized with padding to preserve aspect ratio. Tissue-region cropping can retain
ECG traces or small measurement annotations; this is visible in the review gallery.

The official EchoPrime 11-class view classifier averages five frame probabilities.
Acceptance requires mean probability >=0.75 and frame agreement >=0.6. These are
prespecified screening thresholds, not calibrated confidence or quality scores.
Uncertain clips remain recorded. Declared color Doppler is excluded from this
initial branch. A multihue pixel heuristic additionally screens Doppler overlays
while allowing amber-tinted B-mode tissue; it is not a validated modality classifier.

The native-RGB classifier initially mislabeled many amber-tinted tissue videos as
apical Doppler. A deterministic correction detects strong, nearly uniform hue
across visible tissue (colored fraction >=0.6, hue concentration >=0.95, no
red/blue flow flag) and converts the value channel to grayscale before view
classification and video encoding. Raw previews remain unchanged for review.
Appearance metadata invalidates incompatible cached embeddings. The original
view manifest is retained as `view_manifest_native_rgb.parquet`. This correction
was selected by image inspection before any prediction results were evaluated.

Selection requires at least 32 frames, a nontrivial motion/foreground proxy, and
a recognized apical or parasternal view. Within each study, identical sampled-pixel
fingerprints are deduplicated. This is sampled-content deduplication, not a claim
of bytewise identity. At most two clips/view are ranked by view confidence with a
stable file-ID tie break. They are representative high-confidence views, not
clinically certified best-quality clips or identified original strain acquisitions.

`pilot_review.html` displays the pilot with editable view, quality and notes fields.
Review is saved in the browser's local storage and can be exported as JSON.
Clinician review is still needed; automated predictions are not ground truth.
Exported review edits are not silently ingested into training in this first run.

## Frozen model provenance

Official sources and licenses are saved under `vendor_sources/`; pinned source
URLs and hashes are in `source_provenance.json`. Download locations and archive
checksums are in `weights_manifest.json`. Each embedding also contains the exact
checkpoint hash, source size/mtime, crop rectangle and frame indices.

- EchoPrime: official MViTv2-S encoder, 512 features; 16 frames at native stride 2.
- PanEcho: official ConvNeXt-T + four-layer FrameTransformer, 768 features;
  16 consecutive frames. The saved official architecture's forward semantics are
  preserved, with all encoder weights loaded strictly. No ImageNet initialization
  download or random output head is used.
- Both: start/middle/end windows; FP32 inference; evaluation mode; frozen parameters.
- Model-specific normalization is used. Spatial preprocessing is the explicit
  local DICOM-region/padding pipeline, not claimed identical to either paper's
  source preprocessing. Windows are not guaranteed to span a full cardiac cycle
  at every frame rate. Frame timing remains available for later experiments.

`encoding_clip_cache/` shares decoded model-specific windows across the two
encoders. It is disposable; embeddings are the reusable downstream features.
Original 8-bit RGB frames are retained in spatial previews. Flagged monochrome
display tints are normalized as described above; no denoising, synthetic
augmentation or fitted preprocessing is applied during encoding.

## Statistical protocol

Primary target: the immediately following visit is the first with >=15% relative
Mid-GLS deterioration from the first visit. This is the existing imaging endpoint,
not adjudicated clinical cardiotoxicity. Inputs stop at the current visit.

Every model is evaluated on the same subset with an eligible current apical video.
Three repeated five-fold patient-held-out assignments are reused from the retained
CNN experiments. New probes use three patient-grouped inner folds to select ridge
regularization by AP. Imputation, scaling and eight-component video PCA are fitted
inside those folds. Video models test current apical features, historical apical
differences, and additional parasternal views. A clinical-only refit and acquisition
availability/clip-length control are included.

The retained strain–clinical baseline is the Round 4 CNN+MOMENT averaged OOF
prediction. It was trained on its original training patients, not retrained on the
smaller video subset. New video probes train only on the eligible subset. The
primary fusion is a fixed 75% baseline / 25% video probability average. No learned
stacker or blend-weight selection uses held-out predictions. Baseline and fusion
metrics use identical held-out examples. Historical and additional-view probes
are secondary exploratory comparisons.

AUROC, AP and Brier are reported with 2,000 paired patient-cluster bootstrap draws.
These intervals condition on the existing OOF predictions and do not include all
uncertainty from repeated development on this cohort. Patient folds prevent
within-patient overlap, but cannot substitute for external confirmation. Model
selection and clip screening must be validated before claiming clinical utility.

## Environment

The existing `.venv` provides torch, torchvision, pydicom, pandas, scipy,
scikit-learn, pyarrow and Pillow. This task added
`opencv-python-headless==4.11.0.86`. The observed GPU is an RTX 4060 Ti (8 GB).
All source reading, image processing and inference occur locally.

## First completed run: 2026-09-15

Both frozen encoders completed all 2,771 selected clips. The paired cohort has
181 transitions from 90 patients, including 41 events. Existing strain + clinical
AUROC was 0.693; fixed EchoPrime and PanEcho blends scored 0.687 and 0.691.
Both paired AUROC intervals included zero. These initial probes did not show
incremental discrimination. Clinician view/quality review remains pending.

The report includes a separately labeled post-hoc diagnostic showing that a
fixed blend toward outer-training prevalence reproduces the Brier improvement
without image information. It does not modify the primary comparison.
`dicom_prediction_calibration_diagnostic.py` reproduces that check;
`dicom_prediction_coverage.py` refreshes the selection and availability audit.

See [experiment report](output/dicom_prediction/experiment_report.md),
[review gallery](output/dicom_prediction/pilot_review.html), and
[final validation](output/dicom_prediction/final_validation.json).
