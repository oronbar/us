# Evidence audit and continuation notes

Prepared 6 September 2026 for the English thesis draft. This is an internal research note, not a substitute for institutional study records.

## Review coverage

The draft uses the complete extracted text of the five-page MSc proposal and the 90-slide Israel Innovation Authority background PDF, with visual review of page contact sheets. It synthesizes 29 local Markdown research reports, structured results, key prediction and preprocessing code, three root-level PowerPoint presentations and the existing Hebrew early-detection Word report. The supporting source manifest hashes 211 relevant files. A hash entry records provenance; it does not claim that every line or every underlying training run was independently audited.

The review did not retrain historical models, review every raw cine or manually adjudicate patient outcomes. Third-party model repositories and every binary checkpoint were not exhaustively audited. The thesis is a substantive first draft based on the available research record, not a completed clinical validation.

## Structured checks completed

- The current visit parquet contains 400 rows; its SHA-256 is f5e97dc2687c9eddcf1732b8cef4bfa61c8eb6560e3dae3279875faf49aac6df, matching the main run metadata.
- The transition parquet contains 297 rows from 103 patients. The primary eligibility mask retains 238 transitions with 49 events.
- The patient fold table contains 309 entries and exactly one test assignment per patient per repeat. The inspected split code groups all visits by patient.
- AUC and AP were recalculated from round3_oof_predictions.parquet: clinical ridge 0.6309253860 / 0.2894655416; retained CNN 0.6826476622 / 0.3394808363; fixed CNN plus random Mantis plus TimeMIL 0.7203325775 / 0.3659634242. These match the reported point estimates.
- Reported confidence intervals and paired deltas were read from saved tables. The historical bootstrap and training processes were not rerun.
- The new manuscript and figures use aggregate metrics. Patient identifiers and mapping files are not included in the delivered manuscript.

## Endpoint distinctions

| Analysis | Eligible sample | Operational target | Consequence |
| --- | --- | --- | --- |
| Main next-visit study | 103 patients, 238 transitions, 49 events | First next-visit Mid magnitude decline at least 15% from first visit | Primary results chapter |
| Amber selected endpoint | 71 patients, 15 events | Strictly greater than 15%, sufficient history, final interval at most 180 days, one selected endpoint per patient | Separate secondary analysis |
| Early landmark incident subset | 76 patients, 17 events | First two visits used to predict a later incident outcome | Separate secondary analysis |
| Prior AS study | Separate published aortic-stenosis population | Detection/classification in AS | Literature and grant background only |

Neither the principal script name nor a slide title containing cardiotoxicity establishes a clinical CTRCD diagnosis. The primary outcome is an imaging surrogate. Clinical history, treatment exposure, biomarkers and adjudication must be linked before claiming clinical cardiotoxicity prediction.

## Documentation and method issues

1. The nominal transfer and folder name refer to 105 patients; current parser and prediction tables contain 103. Reconcile using the clinical handover manifest rather than infer two exclusions.
2. The dataset schema Markdown retains smaller historical counts. The current research report also contains stale limitations from an earlier cohort. Current structured tables and metadata take precedence in the draft.
3. The earlier attention run reports AP around 0.333; the retained later reference reports 0.339. Identify the run when quoting either.
4. The best-performing Mantis component is randomly initialized and frozen. It is not evidence of benefit from pretrained Mantis weights.
5. The main CPU feature screen is fitted on training rows. The neural scalar availability screen is applied before splitting. Although it does not use labels, move it inside the training fold for strict evaluation.
6. The learned-weight ensemble function uses a global OOF score table. Its second-level patient split does not prove nesting of the base-model training. Inner OOF predictions must be regenerated within each outer development fold. The draft uses fixed-weight ensembles as the principal comparison and still acknowledges selection across many tested combinations.
7. Cluster bootstrap on saved predictions does not account for all training and model-selection uncertainty. Reported small improvements remain exploratory.
8. Several Brier scores exceed the evaluation-cohort prevalence-only reference of approximately 0.164. AUC improvement is not evidence of calibration.
9. Pre-event feature screening found no paired within-patient change surviving FDR correction, minimum q 0.467. Candidate rankings should be frozen for independent replication.
10. Technical reanalyses are not independent scan reacquisitions. Label-noise simulations should not be called a physiological ceiling.

## Proposal reference audit

| Supplied reference or item | Resolution in draft |
| --- | --- |
| Lyon et al 2022 ESC guideline | Verified; reference 2, DOI 10.1093/eurheartj/ehac244 |
| Yahav and Adam 2024 | Verified; reference 9, DOI 10.1111/echo.70007; aortic stenosis, not oncology validation |
| Thomas M., Suter MSE | Corrected to Suter TM and Ewer MS; reference 1, DOI 10.1093/eurheartj/ehs181 |
| Pineiro-Lamas et al 2023 dataset | Verified; reference 12, DOI 10.1038/s41597-023-02419-1; tissue Doppler modality differs from local strain curves |
| Khamis et al 2015 | Online December 2015, journal issue 2016; reference 11, DOI 10.5430/jbei.v2n2p57 |
| Lang et al 2015 | Verified chamber-quantification guideline; reference 3 |
| Opdahl et al 2015 illustration | Incomplete citation in slides; not guessed or used as a standalone literature claim |
| Klaeboe et al 2017 mechanical dispersion | Incomplete citation; exact source remains to be resolved |
| Khamis, Yahav et al 2017 and Yahav et al 2020 curve-quality work | Incomplete source details; prior laboratory background only pending exact papers |
| Yahav, Adam, Carasso manuscript in preparation | Not counted as a published or independently verified oncology result |
| Stoylen tutorial and teaching/blog links | Useful conceptual background; primary papers used for thesis evidence |
| Segmental strain presentation citation to Narayan et al | Verified publication is first-authored by Demissei BG, with Narayan HK a coauthor; reference 8 |

The review adds clinical layer-strain studies, SUCCOUR, inter-vendor reproducibility, segmental prediction, EchoNet-related work, a 2026 EchoRisk preprint, original time-series papers and model-reporting/evaluation methods. There are 24 references in the draft. Search coverage is focused and narrative, with no claim of systematic completeness. EchoRisk is explicitly a preprint; Mantis uses the revised 2026 paper metadata. Published model performance is not directly ranked against local scores because endpoints and validation designs differ.

## Relevant mailbox context

Scope of the earlier mailbox review: correspondence involving Dan Adam, Shemy Carasso or Ichilov, from 29 April 2025 through 6 September 2026. The following notes preserve project context; they are not patient-level data or independently verified institutional milestones.

- May 2025: Dan emphasized novelty in differences across layers and across visits, with comparison of at least two methods. This is reflected in the thesis question and comparisons.
- June 2026: SZMC correspondence described a broader clinical list of 183 patients and 696 visits, with substantially fewer fully labeled longitudinal examinations. These counts are not added to the Ichilov modeling cohort.
- July 2026: Ichilov handover described 416 strain examinations and 105 HER2-treated patients. The processed analytical tables contain 103 patient identifiers. Dan requested matching clinical spreadsheets.
- August 2026: the ethics correspondence discussed protocol revisions, AI methods and data handling. A final approval record was not verified. The draft does not invent one.
- September 2026: SZMC labeling correspondence described 102 analyses for 16 patients across 2CH and 4CH views, with examination-date linkage still being resolved. This is a distinct operational batch, not a completed external prediction study.
- 6 September 2026: Ichilov correspondence said raw-video deidentification was complete and files were ready; this did not itself verify delivery into the analysis workspace.
- Dan's question about changes before deterioration is addressed by the local pre-event feature report. Its negative FDR result is retained rather than treating the question as entirely unexamined.

No messages were sent to collaborators as part of preparing this draft.

## Highest priority next steps

Confirm the thesis title and approved ethics wording; obtain a verified clinical manifest; reconcile the cohort; freeze the endpoint and comparison; complete nested preprocessing and learned-weight validation; then evaluate on an untouched hospital cohort. These items can be completed incrementally while editing the current manuscript. The PDF and Markdown contain the same substantive draft.

## Delivered files

- output/pdf/thesis_draft.pdf is the readable manuscript.
- output/thesis/thesis_draft.md is the editable source with resolved tables and citations.
- output/thesis/references.bib is the bibliography for later LaTeX or reference-manager work.
- output/thesis/source_manifest.csv records the local evidence snapshot.
- output/thesis/figures/model_comparison.png is the aggregate scientific figure.

The temporary harvesting, auditing and rendering scripts are under tmp/thesis_review. Existing research code and results were not modified.
