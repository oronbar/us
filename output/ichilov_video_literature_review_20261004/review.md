# Video information for next-visit deterioration: targeted literature review

Reviewed 4 October 2026. This is a targeted review of original papers and official repositories, not a systematic review or meta-analysis. Proposals below are research hypotheses. No new training was launched and no source datasets were changed.

## Recommendation

Test complementary physiology and a different representation-learning objective in small, controlled experiments. Prioritize (1) pretrained PanEcho measurement outputs, (2) frozen EchoJEPA, and (3) phase-aligned chamber/motion profiles with within-visit repeatability controls. Expand beyond the three LV-strain-selected clips when the new features require other anatomy. Reserve another large adaptation campaign for a representation that first demonstrates useful held-out signal.

## What our results establish

The current non-Philips prognosis comparison contains 182 transitions, 86 patients and 38 events, with three repetitions of five patient-held-out folds. The retained strain/clinical reference has AUROC 0.726 and AP 0.379. Adapted EchoPrime with A3C+A4C and fixed 25% video weight reaches 0.736 and 0.355, respectively. This is an exploratory subset result: selecting views inside training folds did not reproduce an improvement. Nested weight tuning used a separately refitted compact baseline, not the retained CNN+MOMENT model, and did not improve its performance.

We have tested frozen EchoPrime, PanEcho and Sekeun Kim's EchoFM, raw embedding differences across visits, temporal-window variability, and GE-specific EchoPrime adaptation. These tests do not establish that video lacks incremental information. They establish that these particular representations and small-data learning procedures have not reliably extracted it.

Local verification: [fusion results](D:/DS/ichilov3_view_fusion_20261004/results.md), [execution protocol](D:/us/output/ichilov_video_experiments_20260930/PLAN.md), and `D:/DS/ichilov3_nonphilips_future_20260930/aligned/features.npz`.

Two practical constraints were rechecked:

- Only 102 of the 182 transitions have an earlier selected video. Longitudinal experiments must compare methods on matched availability and report coverage.
- The next examination occurs 21–448 days later (median 96; IQR 84–112). This mixes prediction horizons. Keep the present endpoint for the main comparison; evaluate a prespecified time-window sensitivity analysis separately. Do not supply the actual future examination interval as an input that would be unavailable at prediction time.

The compact fusion file contains 27 timing and GLS/EF-history variables plus rich strain features. That particular clinical block contains no treatment dose, biomarkers or age. This does not establish what every earlier baseline used. Audit the retained model before claiming these variables were tested.

## Literature and its relevance

### Direct prognostic evidence

**EchoRisk, 2026 preprint and challenge.** This multicentre cardio-oncology resource has 422 patients overall and a 280-patient baseline prognostic cohort. Its best reported reference video model achieves test AUROC 0.541 (95% CI 0.390–0.692) for future cardiotoxicity. Current LV dysfunction classification is much easier, with test AUROC 0.849. Its baseline-only input, follow-up window, outcome and event prevalence differ from ours; these numbers are not a ranking against Ichilov3. [Original benchmark](https://arxiv.org/html/2607.01039v1), [official evaluation repository](https://github.com/EchoRisk-MICCAI/echorisk-benchmark).

**One Encoder to Address All, MICCAI EchoRisk workshop 2026.** On 39 validation patients, EchoPrime head-only prognosis reaches AUROC 0.618; unfreezing four blocks gives 0.497 and two blocks at low learning rate 0.621. These are validation results, not the benchmark's held-out test results. The paper supports caution about assuming better current-function encoders automatically provide better prognosis. [Original paper](https://papers.miccai.org/miccai-2026-sat/paper/EchoRisk2026_010.pdf).

**Ulloa Cerna et al., Nature Biomedical Engineering 2021.** Raw video models trained using 812,278 videos from 34,362 individuals improved one-year mortality prediction relative to clinical/echo comparators. This is evidence that echo pixels can contain prognostic information, but mortality in a much larger population differs substantially from incremental prediction of GLS deterioration. [Original paper](https://www.nature.com/articles/s41551-020-00667-9).

**DROID, JACC 2023.** Automated LV/LA structural measurements were associated with incident cardiovascular outcomes. The incident-outcome analysis involved 5,970 individuals; the surrounding health-system population must not be mistaken for the model's training sample. This motivates interpretable measurements without proving incremental benefit beyond our strain history. [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10696641/).

### New encoders and study-level learning

**EchoJEPA, February 2026 preprint.** Predicts masked spatiotemporal representations using a target encoder, instead of reconstructing every pixel. Crucial distinction: the flagship 1.1-billion-parameter G model used 18 million videos; the paper's released L model used 525,000 public videos. In the paper's LVEF table, L has MAE 5.97 versus EchoPrime 5.33 on Toronto, while G improves to 4.26. Thus the headline improvement cannot be assigned to the public L model. Reported tasks assess current physiology and robustness, not our future endpoint. [Paper, including Tables 2–3](https://arxiv.org/html/2602.02603v3).

The current repository links downloadable L checkpoints and newer V-JEPA 2.1 L/B variants. It recommends V-JEPA-2-initialized L for functional tasks and 2.1 for dense tasks. Record exact checkpoint identity and pretraining provenance; later releases need not reproduce the original paper's table. A batch-one frozen-inference memory/throughput test is necessary on our 8 GB GPU before estimating runtime. [Official code and checkpoint descriptions](https://github.com/bowang-lab/EchoJEPA).

**PanEcho, JAMA 2025.** Its image encoder, temporal Transformer and pretrained task heads provide 39 reporting outputs. Public inference and weights are available. We used its backbone embeddings; the local checkpoint also supports testing clinically supervised output directions. This is a different downstream representation, but does not introduce information absent from the embedding. It may make relevant information easier to learn with few patients. [Paper](https://jamanetwork.com/journals/jama/fullarticle/2835630), [official model usage](https://github.com/CarDS-Yale/PanEcho).

**Multi-View Echocardiographic Embedding, 2025 preprint.** A compact masked Transformer combines EchoPrime clip embeddings across whole studies, trained using 7,169 MIMIC studies. Reported improvements concern current clinical tasks; external validation was limited. This supports study-level representation learning as a possibility, not transplanting a large attention model into 86 outcome patients. [Original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12393585.1/).

**LAMAE / Video-LAMAE, 2026 preprint/workshop.** Exchanges latent information across frames and views during masked pretraining. It is distinct from the preceding frozen-EchoPrime embedding model. In ICD-code prediction, full-finetuning AUROC is 0.75 versus VideoMAE 0.73; frozen results are about 0.62 versus 0.61. This does not demonstrate future-event benefit or a strong frozen-transfer gain. A ready pretrained checkpoint was not verified. [Original paper](https://arxiv.org/html/2604.15096v1).

**EchoPrime and EchoFM.** EchoPrime's comprehensive interpretation uses view-informed anatomical aggregation; our three-clip backbone experiment is not a test of every capability of its complete study pipeline. Sekeun Kim's EchoFM uses masked/periodic representation learning and is already benchmarked locally. Echo-Vision-FM is a separate model; names must not be conflated. [EchoPrime](https://github.com/echonet/EchoPrime), [our EchoFM family](https://github.com/SekeunKim/EchoFM), [different Echo-Vision-FM paper](https://www.nature.com/articles/s41467-025-66340-4).

### Motion and complementary physiology

**SplineAE, MICCAI 2026; OrbitalEcho, September 2026 preprint.** These learn cardiac phase from low-dimensional motion trajectories. They motivate measuring comparable contraction/relaxation phases rather than only averaging complete clip embeddings. They validate phase localization, not future deterioration. SplineAE's repository explicitly excludes trained checkpoints; its semantic ED/ES assignment needs calibration. OrbitalEcho provides code, but a pretrained checkpoint was not verified. [SplineAE paper](https://papers.miccai.org/miccai-2026/0941-Paper1703.html), [code limitations](https://github.com/LCJKwan/SplineAE-Cardiac), [OrbitalEcho paper](https://arxiv.org/html/2609.11650v1), [code](https://github.com/BonniciJ/OrbitalEcho).

**AIC prospective cohort, Circulation Journal 2024.** Among 383 patients, 42 developed cardiotoxicity. Early decline in LA reservoir strain at three months was independently associated with subsequent cardiotoxicity, alongside LV GLS and biomarkers. Its endpoint was LVEF decline, not our GLS threshold. [Original study](https://www.jstage.jst.go.jp/article/circj/88/5/88_CJ-24-0065/_article).

**SUCCOUR subanalysis, JACC Cardiovascular Imaging 2024.** In 229 patients, baseline LA reservoir strain did not predict CTRCD. This counterexample matters: serial atrial change is a hypothesis worth testing; a single atrial measurement is not an established solution. [Original study](https://www.jacc.org/doi/10.1016/j.jcmg.2024.01.008).

**EchoNet-Segmentation.** Released code links area-segmentation weights for LA and RA on apical B-mode images. These masks can support an exploratory area-over-time profile after temporal QC. An area curve is not myocardial strain, and frame segmentation accuracy does not guarantee valid motion tracking. [Official implementation and supported inputs](https://raw.githubusercontent.com/echonet/measurements/main/Segmentation/README.md).

## Proposed experiments

### 1. Cheap physiology pilot using existing PanEcho features

Apply the pretrained heads with their correct regression/classification transforms. Start with a small prespecified group of chamber size/volume and RV-function outputs supported by the available views. Measure agreement across repeat cines and current-versus-prior change. Keep estimated values explicitly labeled as model estimates. Do not represent an inferred Doppler-related score from B-mode as an actual Doppler measurement.

Compare baseline alone against baseline plus these features on exactly the same patients. Training only a small regularized prediction layer reduces the burden on our 38 events. Refit the baseline inside the required nesting; reused OOF scores are not automatically suitable for weight selection.

### 2. One genuinely different frozen encoder: EchoJEPA

Use a documented public checkpoint on the same three selected views first, so the experiment isolates representation choice. Preserve temporal/spatial tokens where supported, but begin with a simple prespecified pooling/readout to limit model search. Cache embeddings once, fit small patient-held-out probes, and compare both prognosis and current GLS. Improved GLS is supportive evidence of signal, not the success criterion.

Only if frozen inference is feasible and the representation looks useful should we test lightweight latent-prediction adaptation. No full foundation-model pretraining on Ichilov is justified by the number of independent patients.

### 3. Research direction: phase-aligned motion with repeatability controls

Build a short physiology trajectory rather than a single summary per clip:

1. Retain the full native-rate cine. Identify ED/ES using a validated pretrained segmentation/phase method and inspect an outcome-blind pilot. Heart-rate metadata estimates duration but does not identify phase.
2. Preserve contraction and relaxation separately. Start with chamber area/shape trajectories and a small number of phase-specific learned features. Retain actual timing as well as normalized phase so alignment does not erase slowing of relaxation.
3. Within the same visit, compare multiple acceptable cines of the same view. Estimate feature repeatability and nuisance variation. Repeated acquisitions can differ physiologically too, so variability is an estimate rather than pure noise ground truth.
4. Across visits, compare the same structure and phase. Use changes exceeding typical within-visit variation as candidate predictors. Learn any scaling, feature filtering or invariance transform using training patients only.
5. For patients without prior video, use current-state features and an explicit missing-history indicator; report the paired-history subgroup separately.

This differs from our prior experiment: that subtracted globally pooled embeddings. Our mean-plus-window-SD test measured variation between windows, not where within the cardiac cycle a change occurs. The proposed repeatability weighting is our hypothesis, not a published guarantee of improved prognosis.

Use atrial/RV-focused or additional apical clips when the LV-strain-selected clip does not show the relevant anatomy. Later expand to PLAX/PSAX or Doppler only through suitable view-specific preprocessing/models. Balance contributions by view so visits with many duplicate clips do not dominate. Exclude derived strain overlays and reports from the video-input arm.

For physical size changes, retain source spatial calibration; otherwise use explicitly dimensionless shape/area-change descriptors. Resized pixel area is not comparable anatomy across zoom settings. Automated LA area change must not be named LA strain.

## What I would postpone

- Another broad hyperparameter or fusion-weight sweep: current tests do not show a dependable video contribution to optimize.
- Large end-to-end outcome fine-tuning: 38 events provide weak supervision for millions of weights.
- Training pixel-generative models or synthetic patient expansion: reconstruction realism does not establish prognostic information or create independent outcomes.
- Forcing all visits of the same patient into identical embeddings: this can suppress the physiological change we need.
- Training an unconstrained next-visit latent predictor: visit timing and treatment affect the future, and only 102 transitions have video history. Start with measured change and small supervised models.
- Treating LA/RV measures as proven solutions: the clinical literature is mixed and the endpoint differs.

## Evaluation and decision rules

Freeze the existing target, cohort and patient folds for comparisons. Prespecify feature groups and a small search before running results. Every training-derived transform, adaptation and selection must exclude held-out patients across all three Ichilov datasets. Use continuous next-visit GLS change as a secondary task with its own clearly specified reference value; do not silently substitute it for the binary endpoint.

Primary outputs: AUROC, AP, precision among the top-risk 20%, calibration, and paired patient-bootstrap intervals. Correlation belongs to a defined continuous target and is secondary. Show video-only and baseline-plus-video results; a representation useful on its own may still be redundant with strain. If baseline residuals are used for training, obtain them by cross-fitting inside each training fold.

Run a 30–50-cine feasibility/QC pilot selected without reference to errors or outcomes. Verify masks, phases, repeatability, coverage and GPU throughput. Only then extract the entire cohort. A small nominal AUROC improvement with worse AP is not sufficient to replace the baseline. Repeated development on these patients remains exploratory even with nested folds; an independent future cohort is needed for a strong confirmation claim.

If treatment regimen, cumulative dose, biomarkers or relevant comorbidities exist, audit whether they are available at the prediction visit and present in the actual baseline. Video cannot reveal unobserved future treatment exposure. This audit can proceed alongside video research without changing the main comparison.
