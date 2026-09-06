import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "D:/us/cardiotoxicity_models_features_explained.pptx";
const RENDER_DIR = "D:/us/presentation_sources/models_explained/rendered";
const W = 1280;
const H = 720;

const C = {
  ink: "#050505",
  muted: "#5B616B",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  cyan: "#6DCBF4",
  blue: "#3D8DFF",
  navy: "#17365D",
  teal: "#1F8A8A",
  green: "#2E8B57",
  orange: "#E8893A",
  red: "#C94949",
  white: "#FFFFFF",
  paleBlue: "#EAF5FF",
  paleCyan: "#E9FAFF",
  paleGreen: "#EBF6EF",
  paleOrange: "#FFF2E7",
  paleRed: "#FBECEC",
};

const FONT = "Arial";

function addShape(slide, geometry, left, top, width, height, fill = "none", lineFill = "none", lineWidth = 0, radius) {
  const s = slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    ...(radius ? { borderRadius: radius } : {}),
  });
  return s;
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const t = addShape(slide, "textbox", left, top, width, height, opts.fill ?? "none", opts.lineFill ?? "none", opts.lineWidth ?? 0);
  t.text = String(text);
  t.text.style = {
    fontSize: opts.fontSize ?? 18,
    typeface: opts.typeface ?? FONT,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: "square",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return t;
}

function addLine(slide, x1, y1, x2, y2, color = C.rule, width = 2) {
  return slide.shapes.add({
    geometry: "line",
    // The PPTX exporter rejects negative extents. Using the bounding box keeps
    // every line editable and export-safe; our diagrams do not rely on arrow
    // direction, only on visible connections.
    position: { left: Math.min(x1, x2), top: Math.min(y1, y2), width: Math.abs(x2 - x1), height: Math.abs(y2 - y1) },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addPill(slide, text, left, top, width, fill = C.ink, color = C.white) {
  addShape(slide, "roundRect", left, top, width, 28, fill, fill, 0, "rounded-full");
  addText(slide, text, left + 9, top + 5, width - 18, 18, { fontSize: 12, bold: true, color, align: "center" });
}

function addHeader(slide, title, kicker, page) {
  slide.background.fill = C.white;
  addText(slide, kicker.toUpperCase(), 42, 28, 330, 20, { fontSize: 12, bold: true, color: C.blue });
  addText(slide, title, 42, 54, 1170, 68, { fontSize: 40, bold: true, color: C.ink });
  addLine(slide, 42, 126, 1238, 126, C.rule, 1);
  addText(slide, String(page).padStart(2, "0"), 1190, 675, 48, 20, { fontSize: 12, color: C.muted, align: "right" });
}

function addSourceTag(slide, text) {
  addText(slide, `SOURCE  ${text}`, 42, 676, 1040, 16, { fontSize: 10.5, color: C.muted });
}

function notes(slide, body, sources = []) {
  const sourceBlock = sources.length
    ? `\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`
    : "";
  slide.speakerNotes.textFrame.setText(`${body}${sourceBlock}`);
  slide.speakerNotes.setVisible(true);
}

function addCard(slide, x, y, w, h, title, body, opts = {}) {
  addShape(slide, "roundRect", x, y, w, h, opts.fill ?? C.panel, opts.line ?? C.panel, 1, "rounded-xl");
  if (opts.number) addPill(slide, opts.number, x + 18, y + 16, 44, opts.accent ?? C.ink);
  addText(slide, title, x + 18, y + (opts.number ? 56 : 18), w - 36, 34, { fontSize: opts.titleSize ?? 21, bold: true, color: opts.titleColor ?? C.ink });
  addText(slide, body, x + 18, y + (opts.number ? 96 : 62), w - 36, h - (opts.number ? 112 : 78), { fontSize: opts.bodySize ?? 16.5, color: opts.bodyColor ?? C.ink });
}

function addFormula(slide, formula, x, y, w, h, caption) {
  addShape(slide, "roundRect", x, y, w, h, C.ink, C.ink, 0, "rounded-lg");
  addText(slide, formula, x + 18, y + 16, w - 36, h - (caption ? 48 : 26), { fontSize: 20, bold: true, color: C.white, valign: "middle", align: "center" });
  if (caption) addText(slide, caption, x + 14, y + h - 28, w - 28, 18, { fontSize: 11.5, color: C.cyan, align: "center" });
}

function addBars(slide, rows, x, y, w, rowH = 54, max = null, color = C.blue, suffix = "") {
  const mx = max ?? Math.max(...rows.map((r) => r.value));
  rows.forEach((r, i) => {
    const yy = y + i * rowH;
    addText(slide, r.label, x, yy, w * 0.42, 34, { fontSize: 15.5, bold: i === 0 });
    addShape(slide, "roundRect", x + w * 0.44, yy + 4, w * 0.42, 18, C.panel, C.panel, 0, "rounded-full");
    addShape(slide, "roundRect", x + w * 0.44, yy + 4, Math.max(3, w * 0.42 * r.value / mx), 18, r.color ?? color, r.color ?? color, 0, "rounded-full");
    addText(slide, `${r.value.toFixed(r.digits ?? 3)}${suffix}`, x + w * 0.88, yy, w * 0.12, 26, { fontSize: 15, bold: true, align: "right" });
    if (r.ci) addText(slide, r.ci, x + w * 0.44, yy + 26, w * 0.54, 18, { fontSize: 11.5, color: C.muted });
  });
}

function addFlow(slide, items, y, opts = {}) {
  const x0 = opts.x ?? 58;
  const totalW = opts.width ?? 1164;
  const gap = opts.gap ?? 22;
  const h = opts.height ?? 118;
  const w = (totalW - gap * (items.length - 1)) / items.length;
  // Connectors first, then nodes.
  for (let i = 0; i < items.length - 1; i++) {
    const x = x0 + i * (w + gap) + w;
    addLine(slide, x + 3, y + h / 2, x + gap - 3, y + h / 2, C.blue, 3);
  }
  items.forEach((it, i) => {
    const x = x0 + i * (w + gap);
    addShape(slide, "roundRect", x, y, w, h, it.fill ?? C.panel, it.line ?? C.panel, 1, "rounded-xl");
    addText(slide, it.title, x + 14, y + 16, w - 28, 32, { fontSize: it.titleSize ?? 18, bold: true, align: "center" });
    addText(slide, it.body ?? "", x + 14, y + 54, w - 28, h - 64, { fontSize: it.bodySize ?? 14.5, color: C.muted, align: "center", valign: "middle" });
  });
}

function addTable(slide, headers, rows, x, y, widths, opts = {}) {
  const rowH = opts.rowH ?? 38;
  const headerH = opts.headerH ?? 40;
  let xx = x;
  headers.forEach((h, i) => {
    addShape(slide, "rect", xx, y, widths[i], headerH, C.ink, C.white, 0);
    addText(slide, h, xx + 8, y + 9, widths[i] - 16, headerH - 14, { fontSize: opts.headerSize ?? 14, bold: true, color: C.white, align: opts.aligns?.[i] ?? "left" });
    xx += widths[i];
  });
  rows.forEach((row, r) => {
    xx = x;
    row.forEach((cell, c) => {
      const fill = r % 2 === 0 ? C.panel : C.white;
      addShape(slide, "rect", xx, y + headerH + r * rowH, widths[c], rowH, fill, C.rule, 0.6);
      addText(slide, cell, xx + 8, y + headerH + r * rowH + 8, widths[c] - 16, rowH - 12, { fontSize: opts.fontSize ?? 14.5, color: c === 0 ? C.ink : C.muted, bold: c === 0 && opts.boldFirst !== false, align: opts.aligns?.[c] ?? "left" });
      xx += widths[c];
    });
  });
}

function sectionSlide(presentation, number, title, subtitle, accent = C.cyan) {
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  addText(slide, number, 58, 56, 240, 170, { fontSize: 128, bold: true, color: accent });
  addText(slide, title, 58, 286, 1120, 120, { fontSize: 56, bold: true, color: C.white });
  addText(slide, subtitle, 58, 430, 990, 92, { fontSize: 24, color: C.rule });
  addLine(slide, 58, 584, 1220, 584, accent, 4);
  addText(slide, `MODEL ${number}`, 58, 620, 300, 28, { fontSize: 13, bold: true, color: accent });
  return slide;
}

function normalSlide(presentation, title, kicker, page) {
  const slide = presentation.slides.add();
  addHeader(slide, title, kicker, page);
  return slide;
}

const pres = Presentation.create({ slideSize: { width: W, height: H } });
let page = 1;

// 1 — Cover
{
  const s = pres.slides.add();
  s.background.fill = C.ink;
  addText(s, "EARLY CARDIOTOXICITY PREDICTION", 58, 54, 560, 28, { fontSize: 14, bold: true, color: C.cyan });
  addText(s, "Models, features\nand physiological meaning", 58, 134, 820, 220, { fontSize: 60, bold: true, color: C.white });
  addText(s, "A sourceable guide to the CNN, MOMENT, RDST and Catch22 pipelines—and the two best ensembles.", 58, 392, 760, 108, { fontSize: 24, color: C.rule });
  addShape(s, "rect", 940, 0, 340, 720, C.cyan, C.cyan, 0);
  addText(s, "15%", 978, 146, 260, 108, { fontSize: 82, bold: true, color: C.ink, align: "center" });
  addText(s, "relative Mid-GLS\ndeterioration", 978, 272, 260, 74, { fontSize: 23, bold: true, color: C.ink, align: "center" });
  addLine(s, 990, 386, 1226, 386, C.ink, 2);
  addText(s, "103 patients\n238 visit transitions\n49 events", 978, 418, 260, 126, { fontSize: 22, color: C.ink, align: "center" });
  addText(s, "Prepared from the complete local experimental pipeline · 11 Aug 2026", 58, 652, 790, 22, { fontSize: 13, color: C.rule });
  notes(s, "This deck is designed as a teaching and audit document. It separates the exact implementation from physiological interpretation and from held-out evidence.", [
    "Local results: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md",
    "Local implementation: D:/us/cardiotoxicity_top_ensemble_feature_importance.py",
  ]);
  page++;
}

// 2 — Reading guide
{
  const s = normalSlide(pres, "How to read this deck", "Orientation", page++);
  addCard(s, 42, 160, 280, 414, "1 · What enters", "Raw strain curves\nClinical trajectory values\nInter-segment variability\nCurrent-versus-previous changes", { number: "01", accent: C.blue, fill: C.paleBlue });
  addCard(s, 338, 160, 280, 414, "2 · What the model does", "Architecture and dimensions\nTransform or representation\nClassifier and regularization\nHow model outputs are ensembled", { number: "02", accent: C.teal, fill: C.paleGreen });
  addCard(s, 634, 160, 280, 414, "3 · What was important", "Held-out permutation importance\nTreeSHAP when available\nModel-level Shapley values\nConfidence intervals and caveats", { number: "03", accent: C.orange, fill: C.paleOrange });
  addCard(s, 930, 160, 280, 414, "4 · Why it may matter", "Contractile amplitude\nMechanical timing\nSpatial heterogeneity\nTransmural Endo–Mid discordance", { number: "04", accent: C.red, fill: C.paleRed });
  addSourceTag(s, "deck structure; local code + primary literature in speaker notes");
  notes(s, "The presentation moves from the common prediction task to each constituent model, then compares the two ensembles. The appendix decodes every scalar feature family and the Catch22 catalog.", ["Local implementation and results listed throughout the deck."]);
}

// 3 — Task
{
  const s = normalSlide(pres, "The prediction target is one visit ahead", "Prediction task", page++);
  addFlow(s, [
    { title: "First visit", body: "Defines the patient-specific Mid-GLS baseline", fill: C.paleBlue },
    { title: "Current visit t", body: "Curves + scalar trajectory available now", fill: C.paleCyan },
    { title: "Model alert", body: "Probability of first deterioration at t+1", fill: C.paleOrange },
    { title: "Next visit t+1", body: "Label is revealed here—not given to the model", fill: C.paleRed },
  ], 176, { height: 126 });
  addFormula(s, "relative decline = 1 − |Mid-GLS(t+1)| / |Mid-GLS(first)|", 178, 362, 924, 104, "Event when decline ≥ 0.15");
  addText(s, "Important", 92, 516, 146, 30, { fontSize: 20, bold: true, color: C.red });
  addText(s, "The model is not tied to visit number. Every eligible transition becomes a landmark: visit t predicts the immediately following visit.", 238, 512, 920, 58, { fontSize: 20 });
  addSourceTag(s, "cardiotoxicity_next_visit_gpu.py · ESC 2022 definition context");
  notes(s, "Use a patient with four visits as the intuitive example: if visit 4 first crosses the 15% relative Mid-GLS threshold, the prediction is made from visit 3. Earlier transitions from the same patient can also be training examples, but patient-held-out folds prevent leakage.", [
    "Local target construction: D:/us/cardiotoxicity_next_visit_gpu.py",
    "https://academic.oup.com/eurheartj/article/43/41/4229/6673995",
  ]);
}

// 4 — cohort
{
  const s = normalSlide(pres, "Small, longitudinal and imbalanced", "Cohort & evaluation", page++);
  const metrics = [
    ["103", "patients", C.blue],
    ["238", "visit-to-next-visit transitions", C.teal],
    ["49", "events", C.orange],
    ["20.6%", "event prevalence", C.red],
  ];
  metrics.forEach((m, i) => {
    const x = 42 + i * 292;
    addShape(s, "roundRect", x, 160, 270, 142, i === 3 ? C.paleRed : C.panel, i === 3 ? C.red : C.panel, 1, "rounded-xl");
    addText(s, m[0], x + 18, 176, 234, 58, { fontSize: 42, bold: true, color: m[2], align: "center" });
    addText(s, m[1], x + 18, 244, 234, 36, { fontSize: 15, color: C.muted, align: "center" });
  });
  addCard(s, 42, 344, 370, 248, "Validation", "Repeated 5-fold cross-validation\n3 repetitions\nPatient-held-out folds\nSame folds used across models", { fill: C.paleBlue });
  addCard(s, 430, 344, 370, 248, "Metrics", "ROC AUC: discrimination across thresholds\nAP: precision–recall summary\nAP random baseline ≈ event prevalence = 0.206", { fill: C.paleGreen });
  addCard(s, 818, 344, 392, 248, "Uncertainty", "Patient-level bootstrap confidence intervals\nImportance is measured on held-out predictions\nSmall samples make ranking intervals wide", { fill: C.paleOrange });
  addSourceTag(s, "top ensemble feature-importance report; patient-level resampling");
  notes(s, "Average precision must be interpreted against prevalence. Here random ranking has expected AP around 0.206, so AP values around 0.36 are meaningfully better than random even though they look numerically lower than AUC.", [
    "Local results: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md",
  ]);
}

// 5 — channels
{
  const s = normalSlide(pres, "Six channels preserve anatomy and longitudinal change", "Raw curve tensor", page++);
  addText(s, "Per sample tensor: 18 segments × 6 channels × 96 normalized time samples", 42, 146, 1196, 34, { fontSize: 21, bold: true });
  const rows = [
    ["1", "Current Endo", "Eₜ(s,τ)", "Endocardial longitudinal shortening"],
    ["2", "Current Mid", "Mₜ(s,τ)", "Mid-wall longitudinal shortening"],
    ["3", "Current gap", "Eₜ − Mₜ", "Transmural Endo–Mid discordance"],
    ["4", "Change Endo", "Eₜ − Eₜ₋₁", "Evolution of endocardial mechanics"],
    ["5", "Change Mid", "Mₜ − Mₜ₋₁", "Evolution of mid-wall mechanics"],
    ["6", "Change gap", "(E−M)ₜ − (E−M)ₜ₋₁", "Evolution of the transmural gradient"],
  ];
  addTable(s, ["Ch", "Content", "Exact calculation", "Physiological question"], rows, 42, 198, [70, 230, 360, 536], { rowH: 58, fontSize: 15.5, headerSize: 15 });
  addText(s, "Preprocessing: curve / 30 → clip to [−2, 2]. Missing previous-visit channels are encoded consistently with the pipeline.", 42, 602, 1190, 38, { fontSize: 16, color: C.muted });
  addSourceTag(s, "cardiotoxicity_cnn_channel_ablation.py; 96-sample ablation retained");
  notes(s, "Here s indexes segment and tau indexes normalized cardiac-cycle time. Endo and Mid are signed strain curves; summary GLS magnitudes use absolute values. Channel 6 directly operationalizes the hypothesis that early deterioration is hidden in a changing Endo–Mid relationship.", [
    "Local tensor construction: D:/us/cardiotoxicity_cnn_channel_ablation.py",
    "Local sample-length experiments in D:/us/cardio*sample* and CNN ablation outputs",
    "https://pubmed.ncbi.nlm.nih.gov/26661049/",
  ]);
}

// 6 scalar map
{
  const s = normalSlide(pres, "The 96 scalar features provide clinical context", "Scalar branch", page++);
  addCard(s, 42, 164, 276, 404, "24 trajectory features", "Visit history and timing\nCurrent Mid/Endo GLS and EF\nFirst-visit Mid/Endo GLS and EF\nRelative change from first visit\nRelative change from previous visit\n100-day slopes\nEndo–Mid gap\nTwo-visit rolling comparison", { fill: C.paleBlue, titleSize: 22 });
  addCard(s, 334, 164, 276, 404, "36 Endo variability", "18 current inter-segment descriptors\n+ the same 18 changes from previous visit\n\nAmplitude dispersion\nTiming dispersion\nCurve/shape incoherence\nSpatial roughness\nRegional gradients", { fill: C.paleCyan, titleSize: 22 });
  addCard(s, 626, 164, 276, 404, "36 Mid variability", "The identical 18-descriptor vocabulary is calculated for Mid-wall curves, then duplicated as current value and current-minus-previous change.", { fill: C.paleGreen, titleSize: 22 });
  addCard(s, 918, 164, 292, 404, "Fold-safe scaling", "Median imputation fitted on training fold\nRobustScaler using 10th–90th quantiles\nClip to [−5, 5]\n\nNo information from held-out patients is used to fit preprocessing.", { fill: C.paleOrange, titleSize: 22 });
  addSourceTag(s, "cardiotoxicity_next_visit_gpu.py; cardiotoxicity_early_detection.py");
  notes(s, "The appendix provides every scalar family with a calculation and interpretation. Scalars are intentionally shared across models, so raw-curve models can be compared while retaining the same clinical context.", [
    "Local scalar construction: D:/us/cardiotoxicity_next_visit_gpu.py",
    "Local variability vocabulary: D:/us/cardiotoxicity_early_detection.py:44-62",
  ]);
}

// 7 ensembles
{
  const s = normalSlide(pres, "Two equal-weight ensembles reached the top", "Model map", page++);
  addText(s, "ENSEMBLE 1", 42, 150, 250, 22, { fontSize: 13, bold: true, color: C.blue });
  addFlow(s, [
    { title: "CNN", body: "local curve motifs + attention + scalars", fill: C.paleBlue },
    { title: "MOMENT", body: "frozen foundation-model embeddings", fill: C.paleCyan },
    { title: "RDST", body: "random dilated multivariate shapelets", fill: C.paleGreen },
    { title: "Mean probability", body: "AUC 0.706 · AP 0.362", fill: C.paleOrange },
  ], 184, { height: 112 });
  addText(s, "ENSEMBLE 2", 42, 354, 250, 22, { fontSize: 13, bold: true, color: C.teal });
  addFlow(s, [
    { title: "CNN", body: "local curve motifs + attention + scalars", fill: C.paleBlue },
    { title: "MOMENT", body: "frozen foundation-model embeddings", fill: C.paleCyan },
    { title: "Catch22 + XGB", body: "interpretable curve descriptors", fill: C.paleGreen },
    { title: "Mean probability", body: "AUC 0.698 · AP 0.364", fill: C.paleOrange },
  ], 388, { height: 112 });
  addText(s, "Why ensemble? Different representations can make different errors. Equal averaging avoids fitting another high-variance meta-model on only 103 patients.", 92, 556, 1096, 58, { fontSize: 19, align: "center" });
  addSourceTag(s, "round 4 OOF predictions; equal probability averaging");
  notes(s, "Ensemble 1 has the higher AUC; Ensemble 2 has slightly higher AP. The difference between them is not large enough to claim a definitive winner. The important scientific point is that three different representations contribute complementary information.", [
    "Local round-4 implementation: D:/us/cardiotoxicity_timeseries_round4.py",
    "Local importance reproduction: D:/us/cardiotoxicity_top_ensemble_feature_importance.py",
  ]);
}

// 8 performance
{
  const s = normalSlide(pres, "Performance is above random—but still exploratory", "Held-out performance", page++);
  addText(s, "ROC AUC", 42, 156, 170, 30, { fontSize: 22, bold: true });
  addBars(s, [
    { label: "CNN + MOMENT + RDST", value: 0.706, ci: "95% CI 0.630–0.779", color: C.blue },
    { label: "CNN + MOMENT + Catch22", value: 0.698, ci: "95% CI 0.622–0.769", color: C.teal },
    { label: "Random ranking", value: 0.500, ci: "reference", color: C.rule },
  ], 42, 204, 560, 76, 0.8);
  addText(s, "Average precision", 662, 156, 220, 30, { fontSize: 22, bold: true });
  addBars(s, [
    { label: "CNN + MOMENT + RDST", value: 0.362, ci: "95% CI 0.262–0.498", color: C.blue },
    { label: "CNN + MOMENT + Catch22", value: 0.364, ci: "95% CI 0.264–0.493", color: C.teal },
    { label: "Random ranking", value: 0.206, ci: "equals event prevalence", color: C.rule },
  ], 662, 204, 548, 76, 0.45);
  addShape(s, "roundRect", 42, 484, 1168, 122, C.ink, C.ink, 0, "rounded-xl");
  addText(s, "Interpretation", 64, 506, 174, 34, { fontSize: 22, bold: true, color: C.cyan });
  addText(s, "AUC ≈ 0.70 means useful rank separation, not clinical readiness. AP ≈ 0.36 is ~1.76× the random baseline, so it has not ‘fallen to random’; class imbalance makes AP numerically lower than AUC.", 236, 502, 946, 78, { fontSize: 19, color: C.white });
  addSourceTag(s, "round4_results; patient-bootstrap confidence intervals");
  notes(s, "AUC and AP answer different questions. AUC asks how often an event transition ranks above a non-event. AP emphasizes the positive class and precision under imbalance; its chance baseline is prevalence, not 0.5.", ["Local metrics: D:/us/cardiotoxicity_timeseries_round4_results", "Local importance report: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
}

// 9 importance methods
{
  const s = normalSlide(pres, "‘Important’ has three different meanings here", "Interpretability protocol", page++);
  addCard(s, 42, 164, 364, 388, "Held-out permutation", "Shuffle one feature or feature group only in held-out patients. Recompute AUC/AP. The performance drop estimates predictive information that cannot be recovered from the other unshuffled inputs.", { fill: C.paleBlue });
  addCard(s, 424, 164, 364, 388, "Native TreeSHAP", "For Catch22-XGBoost only: decompose each prediction into additive contributions of individual tree inputs. Useful for ranking, but correlated features can split or exchange attribution.", { fill: C.paleGreen });
  addCard(s, 806, 164, 404, 388, "Model-level Shapley", "Evaluate all component subsets and average each model’s marginal contribution. This allocates shared value more fairly than simple leave-one-model-out analysis.", { fill: C.paleOrange });
  addText(s, "Do not read attention weights, shapelet coefficient energy or SHAP as proof of causal myocardial mechanisms.", 116, 588, 1048, 34, { fontSize: 18, bold: true, color: C.red, align: "center" });
  addSourceTag(s, "importance code reproduces OOF scores exactly before perturbation");
  notes(s, "The report uses confidence intervals because the ranking is unstable in a small cohort. Group-level permutation is often more credible than one-feature-at-a-time permutation when predictors are correlated.", ["Local importance implementation: D:/us/cardiotoxicity_top_ensemble_feature_importance.py", "Local report: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
}

// 10 component shapley
{
  const s = normalSlide(pres, "No single component owns the ensemble signal", "Model-level Shapley allocation", page++);
  addText(s, "Ensemble 1 · CNN + MOMENT + RDST", 42, 150, 520, 30, { fontSize: 20, bold: true });
  addBars(s, [
    { label: "CNN", value: 0.0741, color: C.blue },
    { label: "MOMENT", value: 0.0709, color: C.teal },
    { label: "RDST", value: 0.0613, color: C.green },
  ], 42, 200, 540, 66, 0.085);
  addText(s, "Ensemble 2 · CNN + MOMENT + Catch22", 654, 150, 520, 30, { fontSize: 20, bold: true });
  addBars(s, [
    { label: "MOMENT", value: 0.0745, color: C.teal },
    { label: "CNN", value: 0.0744, color: C.blue },
    { label: "Catch22", value: 0.0490, color: C.orange },
  ], 654, 200, 556, 66, 0.085);
  addShape(s, "roundRect", 112, 452, 1056, 128, C.panel, C.panel, 0, "rounded-xl");
  addText(s, "AUC Shapley contribution", 134, 474, 280, 28, { fontSize: 18, bold: true, color: C.blue });
  addText(s, "The values are similar across components because the models share useful signal and add complementary residual information. Catch22 adds almost no unique AUC by leave-one-out, but adds ~0.011 AP.", 412, 470, 730, 76, { fontSize: 18 });
  addSourceTag(s, "exact subset Shapley values from held-out OOF predictions");
  notes(s, "Shapley values distribute shared credit. Leave-one-out asks a different question: what is lost if this model is removed from the full ensemble? Correlated components can have meaningful Shapley value but small leave-one-out loss.", ["Local component table: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/ensemble_component_shapley.csv"]);
}

// 11 section CNN
{
  const s = sectionSlide(pres, "01", "Attention CNN", "A compact supervised model that learns local curve motifs per segment, then combines them with clinical and variability scalars.", C.cyan);
  notes(s, "Section 1 explains the retained CNN architecture and the features that mattered most under held-out permutation.", ["Local architecture: D:/us/cardiotoxicity_cnn_channel_ablation.py"]);
  page++;
}

// 12 CNN architecture
{
  const s = normalSlide(pres, "CNN architecture and exact dimensions", "Attention CNN", page++);
  addFlow(s, [
    { title: "Per segment", body: "[6, 96]", fill: C.paleBlue },
    { title: "Conv1D", body: "6→16 · k=7\nGELU", fill: C.paleCyan },
    { title: "Conv1D", body: "16→24 · k=5\nGELU", fill: C.paleCyan },
    { title: "Avg pool", body: "time→1\n[24]", fill: C.paleGreen },
    { title: "18 segments", body: "[18, 24]", fill: C.paleOrange },
  ], 170, { height: 118 });
  addFlow(s, [
    { title: "Attention center", body: "weighted sum\n[24]", fill: C.paleBlue },
    { title: "Segment SD", body: "dispersion\n[24]", fill: C.paleGreen },
    { title: "Segment max", body: "extreme response\n[24]", fill: C.paleOrange },
    { title: "Curve summary", body: "concatenate\n[72]", fill: C.paleRed },
  ], 358, { x: 170, width: 940, height: 112 });
  addFormula(s, "curve 72 + scalar embedding 32 → shared 104 → 64 → binary risk head", 148, 528, 984, 82, "Multi-task training; 15% Mid-GLS task retained for reporting");
  addSourceTag(s, "ChannelAttentionNet in cardiotoxicity_cnn_channel_ablation.py");
  notes(s, "A one-dimensional kernel moves along time, but every convolutional filter sees all six channels simultaneously. Therefore channel information is already shared inside each temporal kernel. The model applies the same encoder to every segment.", ["Local class ChannelAttentionNet: D:/us/cardiotoxicity_cnn_channel_ablation.py:83"]);
}

// 13 CNN scalar
{
  const s = normalSlide(pres, "The scalar branch is a learned clinical context vector", "Attention CNN", page++);
  addFlow(s, [
    { title: "96 scalars", body: "24 trajectory + 72 variability", fill: C.paleBlue },
    { title: "Dense 96→48", body: "GELU + regularization", fill: C.paleCyan },
    { title: "Dense 48→32", body: "compact scalar embedding", fill: C.paleGreen },
    { title: "Join curves", body: "32 + 72 = 104", fill: C.paleOrange },
  ], 174, { height: 126 });
  addCard(s, 42, 352, 366, 232, "Why include scalars?", "The same curve shape can have different meaning depending on baseline GLS, elapsed time, prior decline and existing spatial heterogeneity.", { fill: C.panel });
  addCard(s, 426, 352, 366, 232, "Why compress to 32?", "A bottleneck forces correlated measurements into a lower-dimensional context and reduces the number of downstream parameters.", { fill: C.panel });
  addCard(s, 810, 352, 400, 232, "What importance showed", "Permuting the entire scalar block reduced AUC by 0.128—far more than permuting all curves (0.009). This is a group effect, not proof that curves are useless.", { fill: C.paleOrange });
  addSourceTag(s, "CNN group permutation importance; fold-specific preprocessing");
  notes(s, "Curve features can be redundant with scalar summaries derived from those curves. When the scalar block is permuted, many correlated measurements are disrupted simultaneously; single curve-channel permutations leave the remaining channels and scalar summaries intact.", ["Local CNN importance output: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/cnn_permutation_importance.csv"]);
}

// 14 CNN importance
{
  const s = normalSlide(pres, "CNN: scalar context dominated held-out importance", "Attention CNN · top inputs", page++);
  addBars(s, [
    { label: "All 96 scalar features", value: 0.1283, ci: "95% CI 0.0118–0.2413", color: C.blue },
    { label: "All raw curves", value: 0.0091, ci: "95% CI −0.0014–0.0209", color: C.teal },
    { label: "Endo within-ring peak robust SD", value: 0.0069, ci: "95% CI −0.0003–0.0163", color: C.orange },
    { label: "Current Mid curve channel", value: 0.0067, ci: "95% CI 0.0008–0.0140", color: C.green },
    { label: "Mid decline slope / 100 d", value: 0.0066, ci: "small, positive rank contribution", color: C.red },
  ], 62, 168, 760, 80, 0.14);
  addCard(s, 856, 168, 354, 320, "Read this carefully", "Block importance is not comparable to a single feature on equal footing. The 96-feature scalar block contains correlated baseline, trajectory and variability information.\n\nThe most defensible result: the CNN used scalar context strongly; individual curve channels added smaller, partly redundant information.", { fill: C.paleOrange });
  addCard(s, 856, 506, 354, 116, "AP result", "Scalar block ΔAP 0.116; all curves ΔAP 0.013.", { fill: C.paleBlue, titleSize: 18 });
  addSourceTag(s, "held-out permutation ΔAUC; patient-bootstrap intervals");
  notes(s, "Do not conclude that the network ignores raw curves. Scalar features are themselves curve-derived and strongly correlated. Also, the small sample favors stable hand-engineered summaries over high-dimensional raw inputs.", ["Local CNN permutation table: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/cnn_permutation_importance.csv"]);
}

// 15 deep ring
{
  const s = normalSlide(pres, "Deep dive: within-ring peak robust dispersion", "Attention CNN · named feature", page++);
  addFormula(s, "robust SD(x₁…xₖ) = 1.4826 × median |xᵢ − median(x)|", 42, 162, 560, 104, "Calculated separately within anatomical rings, then averaged");
  addCard(s, 42, 296, 560, 286, "Exact meaning", "For each ring, collect the absolute peak longitudinal strain across the segments belonging to that ring. Compute a median-absolute-deviation scale estimate. Average the ring-level dispersions.\n\nHigher value = segments at the same ventricular level have less uniform peak contraction.", { fill: C.paleBlue });
  addCard(s, 630, 162, 580, 188, "Why robust?", "One badly tracked segment can inflate ordinary SD. Median/MAD reduces—but does not eliminate—the influence of outliers and vendor measurement noise.", { fill: C.paleGreen });
  addCard(s, 630, 370, 580, 212, "Physiological hypothesis", "A myocardium losing coordinated function may become regionally heterogeneous before the global average crosses the deterioration threshold. Within-ring comparison controls partly for the normal apex-to-base strain gradient.", { fill: C.paleOrange });
  addSourceTag(s, "robust_sd implementation; segmental cardiotoxicity evidence");
  notes(s, "This is the highest-ranked named CNN scalar, but its AUC confidence interval includes zero. Treat it as a mechanistically plausible candidate, not a validated biomarker.", [
    "Local robust SD: D:/us/cardiotoxicity_early_detection.py:97-103",
    "Local feature name: D:/us/cardiotoxicity_early_detection.py:58",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7984733/",
  ]);
}

// 16 raw channels
{
  const s = normalSlide(pres, "Current Mid and Endo curves add distinct raw evidence", "Attention CNN · curve channels", page++);
  addCard(s, 42, 164, 558, 392, "Current Mid curve · channel 2", "Input: Mₜ(s,τ) for every segment and normalized cycle time.\n\nThe CNN can detect local motifs such as shallower peaks, delayed recovery, altered systolic contour or post-systolic shortening without predefining them.\n\nHeld-out ΔAUC: 0.0067; CI excluded zero in this analysis.", { fill: C.paleBlue });
  addCard(s, 628, 164, 582, 392, "Current Endo curve · channel 1", "Input: Eₜ(s,τ). Healthy hearts normally show a transmural longitudinal-strain gradient, with larger endocardial shortening magnitude than outer layers. Endo may therefore reveal early subendocardial or layer-specific change.\n\nHeld-out ΔAUC: 0.0058; small and partly redundant.", { fill: C.paleCyan });
  addText(s, "The kernels are temporal (1D), but their input depth is six channels—so each learned filter can combine Endo, Mid and change information at the same time point.", 112, 588, 1056, 48, { fontSize: 18, bold: true, align: "center" });
  addSourceTag(s, "CNN channel permutation; layer-specific strain physiology");
  notes(s, "Raw curve importance is smaller because the scalar branch already contains GLS, slopes and variability summaries derived from the same underlying curves. Channel permutation measures unique residual value, not total biological relevance.", ["Local architecture: D:/us/cardiotoxicity_cnn_channel_ablation.py", "https://pubmed.ncbi.nlm.nih.gov/26661049/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5491258/"]);
}

// 17 attention
{
  const s = normalSlide(pres, "Segment attention is a learned pooling rule—not a causal map", "Attention CNN", page++);
  addFormula(s, "aₛ = softmax(MLP(zₛ + eₛ))   ;   center = Σₛ aₛ zₛ", 42, 158, 720, 100, "zₛ: 24-D segment embedding · eₛ: learned segment identity");
  addBars(s, [
    { label: "Segment 9", value: 0.0601, color: C.blue, digits: 4 },
    { label: "Segment 18", value: 0.0600, color: C.teal, digits: 4 },
    { label: "Segment 16", value: 0.0595, color: C.green, digits: 4 },
    { label: "Segment 8", value: 0.0590, color: C.orange, digits: 4 },
    { label: "Segment 13", value: 0.0589, color: C.red, digits: 4 },
  ], 42, 304, 650, 58, 0.065);
  addCard(s, 744, 158, 466, 396, "What the weights say", "Uniform weight over 18 segments is 1/18 = 0.0556. The learned averages are close to uniform, so the model did not collapse onto one anatomical region.\n\nWeights depend on the segment representation and training fold. They indicate pooling emphasis, not that a segment causes toxicity.", { fill: C.paleOrange });
  addText(s, "The SD and max pooling branches still preserve heterogeneity even when attention stays nearly uniform.", 744, 572, 450, 48, { fontSize: 17, bold: true });
  addSourceTag(s, "average learned attention weights across held-out predictions");
  notes(s, "Attention uses an MLP 24→12→1 followed by softmax across 18 segments. The narrow range of mean weights supports a diffuse, whole-heart signal rather than a single dominant segment.", ["Local attention class: D:/us/cardiotoxicity_cnn_channel_ablation.py:83", "Local attention output: D:/us/cardiotoxicity_top_ensemble_feature_importance_results"]);
}

// 18 section moment
{
  const s = sectionSlide(pres, "02", "MOMENT-small", "A frozen time-series foundation model converts each curve into a 512-dimensional representation learned from a large, heterogeneous pretraining corpus.", C.teal);
  notes(s, "Section 2 explains how MOMENT is used as a frozen feature extractor and why segment-max aggregation was the strongest feature family.", ["https://arxiv.org/abs/2402.03885", "Local implementation: D:/us/cardiotoxicity_timeseries_round1.py"]);
  page++;
}

// 19 moment architecture
{
  const s = normalSlide(pres, "MOMENT is frozen; only the small classifier is fitted", "MOMENT-small", page++);
  addFlow(s, [
    { title: "96 samples", body: "each segment × channel curve", fill: C.paleBlue },
    { title: "Resize to 512", body: "only for MOMENT input format", fill: C.paleCyan },
    { title: "MOMENT-1-small", body: "frozen encoder\nno fine-tuning", fill: C.paleGreen },
    { title: "Patch mean", body: "64 patches → 512-D", fill: C.paleOrange },
    { title: "[18,6,512]", body: "segment × channel embedding", fill: C.paleRed },
  ], 172, { height: 126 });
  addFlow(s, [
    { title: "Mean across segments", body: "6×512", fill: C.paleBlue },
    { title: "SD across segments", body: "6×512", fill: C.paleGreen },
    { title: "Max across segments", body: "6×512", fill: C.paleOrange },
    { title: "Concatenate", body: "9,216 features", fill: C.paleRed },
  ], 370, { x: 150, width: 980, height: 108 });
  addFormula(s, "PCA → 32 whitened components + 96 scalars → balanced L2 logistic regression", 146, 532, 988, 84, "Fold-specific PCA; C = 0.3");
  addSourceTag(s, "MOMENTPipeline embedding task; model weights frozen");
  notes(s, "The frozen approach limits overfitting in a small cohort. It also means the representation was not learned specifically for cardiac strain, so it may capture general morphology better than domain-specific physiology.", ["https://arxiv.org/abs/2402.03885", "https://github.com/moment-timeseries-foundation-model/moment", "Local implementation: D:/us/cardiotoxicity_timeseries_round1.py:117-121"]);
}

// 20 moment math
{
  const s = normalSlide(pres, "MOMENT aggregation keeps central, variable and extreme patterns", "MOMENT-small · representation", page++);
  addCard(s, 42, 164, 362, 384, "Segment mean", "For each channel and embedding coordinate j:\n\nμ(c,j) = (1/18) Σₛ h(s,c,j)\n\nCaptures the typical whole-heart representation.", { fill: C.paleBlue });
  addCard(s, 424, 164, 362, 384, "Segment SD", "σ(c,j) = √meanₛ[h(s,c,j)−μ(c,j)]²\n\nCaptures how heterogeneous the learned representation is across segments.", { fill: C.paleGreen });
  addCard(s, 806, 164, 404, 384, "Segment max", "m(c,j) = maxₛ h(s,c,j)\n\nPreserves the strongest activation anywhere in the ventricle for each learned embedding coordinate. Different coordinates may select different segments.", { fill: C.paleOrange });
  addText(s, "The 9,216 raw embedding values are compressed inside each training fold to 32 whitened principal components before classification.", 132, 584, 1016, 40, { fontSize: 18, bold: true, align: "center" });
  addSourceTag(s, "local embedding pooling and fold-specific PCA");
  notes(s, "Segment max is not the maximum raw strain and is not tied to one anatomical segment. It is the element-wise maximum in a learned representation. That distinction is essential for correct interpretation.", ["Local MOMENT aggregation: D:/us/cardiotoxicity_timeseries_round1.py", "Local importance reproduction: D:/us/cardiotoxicity_top_ensemble_feature_importance.py"]);
}

// 21 moment importance
{
  const s = normalSlide(pres, "MOMENT: segment max was the strongest feature family", "MOMENT-small · top inputs", page++);
  addBars(s, [
    { label: "Embedding segment max", value: 0.0901, ci: "95% CI 0.0108–0.1618", color: C.orange },
    { label: "All scalar features", value: 0.0551, ci: "95% CI −0.0063–0.1119", color: C.blue },
    { label: "All curve embeddings", value: 0.0542, ci: "95% CI −0.0066–0.1174", color: C.teal },
    { label: "Change in Endo−Mid channel", value: 0.0532, ci: "95% CI −0.0219–0.1317", color: C.red },
    { label: "Change Endo channel", value: 0.0206, ci: "smaller unique contribution", color: C.green },
  ], 62, 168, 770, 80, 0.10);
  addCard(s, 864, 168, 346, 338, "Interpretation", "The model benefited most from retaining the strongest learned motif across segments. This aligns with a focal or heterogeneous early-abnormality hypothesis.\n\nThe changing Endo–Mid gap ranks high but its CI crosses zero, so the evidence is supportive—not conclusive.", { fill: C.paleOrange });
  addCard(s, 864, 524, 346, 98, "AP", "Segment max ΔAP 0.095; CI 0.034–0.188.", { fill: C.paleBlue, titleSize: 17 });
  addSourceTag(s, "held-out permutation before fold-specific PCA");
  notes(s, "Permutation occurs on embedding groups before PCA, then held-out predictions are recomputed. This keeps the interpretation attached to the meaningful aggregation family rather than anonymous principal components.", ["Local MOMENT importance: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/moment_permutation_importance.csv"]);
}

// 22 segment max
{
  const s = normalSlide(pres, "Deep dive: MOMENT segment-max embedding", "MOMENT-small · strongest feature", page++);
  addFormula(s, "m(c,j) = max over segments s of h(s,c,j)", 42, 162, 560, 100, "6 channels × 512 embedding coordinates = 3,072 values");
  addCard(s, 42, 294, 560, 286, "How it is calculated", "1. Encode every segment/channel curve with frozen MOMENT.\n2. Average its temporal patch embeddings.\n3. For each channel c and embedding coordinate j, keep the largest value among 18 segments.\n4. Concatenate with mean and SD pools; PCA compresses the result.", { fill: C.paleBlue });
  addCard(s, 630, 162, 580, 174, "What it is not", "Not peak strain. Not one ‘most important segment.’ Coordinate 17 may be maximized by segment 4 while coordinate 301 is maximized by segment 12.", { fill: C.paleRed });
  addCard(s, 630, 356, 580, 224, "Physiological interpretation", "If injury is initially regional, averaging can dilute it. Max pooling can preserve a strong learned contour abnormality from a small number of segments—consistent with reported non-uniform segmental strain worsening after doxorubicin.", { fill: C.paleOrange });
  addSourceTag(s, "MOMENT pooling code; segmental strain cardiotoxicity cohort");
  notes(s, "We cannot translate an anonymous MOMENT embedding coordinate into a single physiological variable without further probing. The defensible physiological statement concerns the pooling operation: it preserves extreme segment-level learned responses.", ["Local implementation: D:/us/cardiotoxicity_timeseries_round1.py", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7984733/", "https://arxiv.org/abs/2402.03885"]);
}

// 23 gap
{
  const s = normalSlide(pres, "Deep dive: change in the Endo–Mid difference", "MOMENT-small · hypothesis feature", page++);
  addFormula(s, "Δgap(s,τ) = [Eₜ(s,τ) − Mₜ(s,τ)] − [Eₜ₋₁(s,τ) − Mₜ₋₁(s,τ)]", 42, 160, 1168, 106, "Channel 6 · calculated point-by-point for all segments and normalized times");
  addCard(s, 42, 302, 362, 260, "Meaning", "Measures whether the relationship between endocardial and mid-wall deformation has changed since the prior visit—even if both layers changed in the same direction.", { fill: C.paleBlue });
  addCard(s, 424, 302, 362, 260, "Why it may matter", "Healthy myocardium has a layer-dependent strain gradient. Selective vulnerability, altered transmural mechanics or tracking changes could change this gradient before global Mid-GLS deteriorates.", { fill: C.paleGreen });
  addCard(s, 806, 302, 404, 260, "Evidence strength", "MOMENT-only ΔAUC 0.053; ensemble ΔAUC 0.020–0.022. Confidence intervals cross zero. This is promising mechanistic evidence, not confirmation of layer-selective cardiotoxicity.", { fill: C.paleOrange });
  addText(s, "A shared decrease in Endo and Mid can cancel in this channel; therefore it complements—not replaces—the original layer curves and their changes.", 112, 596, 1056, 36, { fontSize: 17.5, bold: true, align: "center" });
  addSourceTag(s, "channel 6 definition; layer-specific strain normal ranges");
  notes(s, "This feature directly tests the first biological hypothesis. It is sensitive to longitudinal change in the transmural relationship, but it may also amplify layer-specific measurement noise because it differences four curves.", ["Local channel construction: D:/us/cardiotoxicity_cnn_channel_ablation.py", "https://pubmed.ncbi.nlm.nih.gov/26661049/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5491258/"]);
}

// 24 section RDST
{
  const s = sectionSlide(pres, "03", "RDST shapelets", "A randomized dictionary of short, dilated multivariate curve patterns turns the entire 108-signal heart representation into distance, timing and occurrence features.", C.green);
  notes(s, "Section 3 explains Random Dilated Shapelet Transform and why best-match location was the strongest RDST feature family.", ["https://arxiv.org/abs/2109.13514", "Local implementation: D:/us/cardiotoxicity_timeseries_round4.py"]);
  page++;
}

// 25 RDST architecture
{
  const s = normalSlide(pres, "RDST converts curve motifs into linear-model inputs", "RDST shapelets", page++);
  addFlow(s, [
    { title: "[18,6,96]", body: "flatten segment×channel → [108,96]", fill: C.paleBlue },
    { title: "RDST", body: "1,200 random dilated multivariate shapelets", fill: C.paleCyan },
    { title: "3 outputs each", body: "distance · location · count", fill: C.paleGreen },
    { title: "+ 96 scalars", body: "direct concatenation", fill: C.paleOrange },
    { title: "Logistic regression", body: "balanced · L2/liblinear · C=0.1", fill: C.paleRed },
  ], 178, { height: 132 });
  addCard(s, 42, 368, 552, 210, "Why dilation?", "A dilated shapelet samples a motif with gaps. It can represent patterns at different temporal scales without requiring a long contiguous template.", { fill: C.paleBlue });
  addCard(s, 618, 368, 592, 210, "Implementation detail", "Curves were multiplied by 100 and received the same tiny label-independent broadband perturbation (SD 0.001) to satisfy Aeon numeric constraints. This is not task augmentation and does not use labels.", { fill: C.paleOrange });
  addSourceTag(s, "RandomDilatedShapeletTransform; 1,200 shapelets");
  notes(s, "Each multivariate shapelet spans all 108 signals. It is not one segment or one channel. That increases expressive power but makes anatomical interpretation more diffuse.", ["https://arxiv.org/abs/2109.13514", "Local RDST fitting: D:/us/cardiotoxicity_timeseries_round4.py:525", "Local reproduction: D:/us/cardiotoxicity_top_ensemble_feature_importance.py"]);
}

// 26 shapelet concept
{
  const s = normalSlide(pres, "A shapelet asks: ‘where does this motif occur?’", "RDST shapelets · intuition", page++);
  // simple waveform drawings
  addText(s, "Learned shapelet q", 42, 154, 320, 28, { fontSize: 20, bold: true });
  addShape(s, "roundRect", 42, 194, 386, 174, C.panel, C.panel, 0, "rounded-xl");
  const qPts = [[68,288],[104,276],[140,238],[176,222],[212,246],[248,304],[284,324],[320,286],[356,252],[396,264]];
  for (let i=0;i<qPts.length-1;i++) addLine(s,qPts[i][0],qPts[i][1],qPts[i+1][0],qPts[i+1][1],C.green,4);
  addText(s, "dilation skips intermediate time points", 72, 326, 326, 20, { fontSize: 13, color: C.muted, align: "center" });
  addText(s, "Patient curve x", 488, 154, 320, 28, { fontSize: 20, bold: true });
  addShape(s, "roundRect", 488, 194, 722, 174, C.paleBlue, C.paleBlue, 0, "rounded-xl");
  const xPts = [[514,278],[554,270],[594,262],[634,246],[674,228],[714,244],[754,302],[794,326],[834,286],[874,250],[914,264],[954,284],[994,276],[1034,260],[1074,248],[1114,264],[1180,280]];
  for (let i=0;i<xPts.length-1;i++) addLine(s,xPts[i][0],xPts[i][1],xPts[i+1][0],xPts[i+1][1],C.blue,3);
  addShape(s, "roundRect", 652, 210, 234, 138, "none", C.orange, 4, "rounded-lg");
  addText(s, "best match", 700, 320, 140, 20, { fontSize: 13, bold: true, color: C.orange, align: "center" });
  addFlow(s, [
    { title: "1 · Slide", body: "compare q at candidate positions", fill: C.paleBlue },
    { title: "2 · Score", body: "L1 distance at each position", fill: C.paleGreen },
    { title: "3 · Summarize", body: "minimum, argmin, occurrence count", fill: C.paleOrange },
  ], 430, { x: 150, width: 980, height: 118 });
  addText(s, "Diagram is conceptual; actual RDST shapelets are multivariate across 108 signals.", 286, 590, 708, 28, { fontSize: 16, color: C.muted, align: "center" });
  addSourceTag(s, "Guillaume et al. Random Dilated Shapelet Transform");
  notes(s, "The simplified line drawing shows one signal only. In this implementation each shapelet is multivariate, so its match jointly reflects all segment/channel inputs.", ["https://arxiv.org/abs/2109.13514", "Local RDST input reshape: D:/us/cardiotoxicity_timeseries_round4.py"]);
}

// 27 outputs
{
  const s = normalSlide(pres, "Every RDST shapelet yields three interpretable statistics", "RDST shapelets · features", page++);
  addCard(s, 42, 164, 362, 388, "Minimum L1 distance", "d(q,x) = minₚ Σ|q − xₚ|\n\nHow closely the learned motif appears anywhere in the cycle. Smaller distance means a better match.", { fill: C.paleBlue, number: "01", accent: C.blue });
  addCard(s, 424, 164, 362, 388, "Best-match location", "p* = argminₚ distance(q,xₚ)\n\nWhere in normalized cardiac-cycle time the motif matches best. This is the strongest RDST family in our data.", { fill: C.paleGreen, number: "02", accent: C.green });
  addCard(s, 806, 164, 404, 388, "Occurrence count", "countₚ[distance(q,xₚ) < threshold]\n\nHow often a sufficiently similar motif appears. The threshold is learned for each shapelet.", { fill: C.paleOrange, number: "03", accent: C.orange });
  addText(s, "The linear classifier learns a positive or negative coefficient for each statistic after standardization.", 184, 588, 912, 36, { fontSize: 18, bold: true, align: "center" });
  addSourceTag(s, "RDST paper and Aeon transform semantics");
  notes(s, "Location can encode timing shifts even when the shape remains similar. Distance can encode morphology, and occurrence count can encode repeated or prolonged patterns.", ["https://arxiv.org/abs/2109.13514", "Local transform: D:/us/cardiotoxicity_timeseries_round4.py:525"]);
}

// 28 RDST importance
{
  const s = normalSlide(pres, "RDST: motif timing was more valuable than motif identity", "RDST shapelets · top inputs", page++);
  addBars(s, [
    { label: "Best-match location family", value: 0.0700, ci: "95% CI 0.0078–0.1258", color: C.green },
    { label: "All shapelet features", value: 0.0628, ci: "95% CI −0.0211–0.1516", color: C.teal },
    { label: "All 96 scalar features", value: 0.0489, ci: "95% CI 0.0195–0.0824", color: C.blue },
    { label: "Endo vendor peak robust SD", value: 0.0065, ci: "95% CI 0.0006–0.0136", color: C.orange },
    { label: "Mid peak robust SD", value: 0.0046, ci: "95% CI 0.0002–0.0088", color: C.red },
  ], 62, 168, 780, 80, 0.08);
  addCard(s, 870, 168, 340, 342, "Interpretation", "The predictive motif was often defined by when it occurred rather than only whether it occurred. This supports the inter-segment/mechanical-timing hypothesis.\n\nBoth named scalar features quantify between-segment amplitude dispersion, reinforcing a heterogeneity signal.", { fill: C.paleOrange });
  addCard(s, 870, 528, 340, 94, "AP", "Best-match location ΔAP 0.069.", { fill: C.paleGreen, titleSize: 17 });
  addSourceTag(s, "held-out RDST feature-family permutation");
  notes(s, "The location family is the only broad RDST family with an AUC confidence interval clearly above zero. It does not directly say which segment is late, because the shapelets are multivariate.", ["Local RDST importance: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/rdst_permutation_importance.csv"]);
}

// 29 location
{
  const s = normalSlide(pres, "Deep dive: RDST best-match location", "RDST shapelets · strongest feature", page++);
  addFormula(s, "location(q,x) = argmin over p of Σ |q − window(x,p)|", 42, 162, 662, 100, "Position p is normalized within the 96-sample cardiac cycle");
  addCard(s, 42, 294, 662, 282, "How to interpret it", "A learned multivariate contour may match near peak systole in one visit but later in another. The model can use that displacement as a feature even if the motif’s amplitude remains similar.\n\nThis captures timing at a richer level than one hand-engineered time-to-peak per segment.", { fill: C.paleBlue });
  addCard(s, 732, 162, 478, 184, "Physiological link", "Delayed or dispersed contraction can reflect mechanical dyssynchrony, post-systolic shortening or regional contractile impairment.", { fill: C.paleGreen });
  addCard(s, 732, 364, 478, 212, "Critical caveat", "Because the shapelet spans 108 signals, location is a whole-pattern timing feature—not a direct segment time-to-peak. Inspecting individual top shapelets would be needed for anatomical localization.", { fill: C.paleOrange });
  addSourceTag(s, "RDST feature definition; timing interpretation is hypothesis-level");
  notes(s, "This is a model-derived timing feature. Its physiological interpretation is plausible but indirect. Future work should visualize the top stable shapelets and their best-match windows in true-positive and false-positive patients.", ["https://arxiv.org/abs/2109.13514", "Local RDST report: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
}

// 30 RDST scalar support
{
  const s = normalSlide(pres, "RDST’s named scalar features quantify amplitude heterogeneity", "RDST shapelets · scalar support", page++);
  addCard(s, 42, 164, 558, 402, "Endo vendor peak-systolic robust SD", "Collect the vendor-reported absolute peak-systolic Endo strain across segments.\n\nrobust SD = 1.4826 × MAD\n\nHigh values mean segmental endocardial peak-systolic contraction is uneven. Vendor peak-systolic timing may differ from the absolute minimum of the resampled curve.", { fill: C.paleBlue });
  addCard(s, 628, 164, 582, 402, "Mid peak absolute robust SD", "Collect |peak Mid strain| from each segment and calculate robust SD.\n\nHigh values mean some segments shorten much more than others. In early regional injury, dispersion may rise before the global Mid-GLS mean drops by 15%.\n\nIt can also rise from tracking inconsistency, so it is not purely biological.", { fill: C.paleGreen });
  addText(s, "Both features had small but positive held-out ΔAUC confidence intervals; magnitudes were ~0.005–0.007.", 170, 596, 940, 32, { fontSize: 17.5, bold: true, align: "center" });
  addSourceTag(s, "robust amplitude-dispersion features; current visit only");
  notes(s, "The prefix cur_var__ means current-visit variability. d_var__ would mean current minus previous variability. The suffix identifies the within-visit calculation.", ["Local robust SD: D:/us/cardiotoxicity_early_detection.py:97-103", "Local feature definitions: D:/us/cardiotoxicity_early_detection.py:44-62", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7984733/"]);
}

// 31 section catch22
{
  const s = sectionSlide(pres, "04", "Catch22 + XGBoost", "Twenty-two canonical descriptors summarize each curve; segment aggregation preserves central tendency, dispersion and extremes before gradient-boosted trees learn nonlinear interactions.", C.orange);
  notes(s, "Section 4 explains the 660 Catch22 curve features, the XGBoost classifier, and the strongest descriptor: positive-outlier timing in current Endo curves.", ["https://link.springer.com/article/10.1007/s10618-019-00647-x", "https://doi.org/10.1145/2939672.2939785"]);
  page++;
}

// 32 catch architecture
{
  const s = normalSlide(pres, "Catch22 turns every curve into an interpretable fingerprint", "Catch22 + XGBoost", page++);
  addFlow(s, [
    { title: "18×6 curves", body: "each length 96", fill: C.paleBlue },
    { title: "22 descriptors", body: "per segment × channel", fill: C.paleCyan },
    { title: "5 segment pools", body: "mean · SD · min · max · median", fill: C.paleGreen },
    { title: "660 curve features", body: "6 × 22 × 5", fill: C.paleOrange },
    { title: "+96 scalars", body: "XGBoost classifier", fill: C.paleRed },
  ], 174, { height: 130 });
  addCard(s, 42, 360, 556, 224, "Descriptor families", "Distribution and histogram modes · outlier timing · linear/nonlinear autocorrelation · low-frequency power · forecasting error · fluctuation scaling · symbolic run lengths.", { fill: C.paleBlue });
  addCard(s, 622, 360, 588, 224, "XGBoost configuration", "300 trees · maximum depth 2 · learning rate 0.03 · subsample 0.8 · column sample 0.7 · L1 = 1 · L2 = 5 · balanced class weighting · 3 seeds.", { fill: C.paleOrange });
  addSourceTag(s, "Catch22 paper; XGBoost KDD 2016; local round-4 config");
  notes(s, "Depth-2 trees limit interaction complexity in a small dataset. Catch22 provides named features, making this constituent easier to audit than learned neural embeddings.", ["https://link.springer.com/article/10.1007/s10618-019-00647-x", "https://doi.org/10.1145/2939672.2939785", "Local Catch22 extraction: D:/us/cardiotoxicity_timeseries_round4.py:393-405"]);
}

// 33 catch construction
{
  const s = normalSlide(pres, "Feature names encode channel, segment pool and descriptor", "Catch22 + XGBoost · naming", page++);
  addFormula(s, "c1_min_DN_OutlierInclude_p_001_mdrmd", 42, 160, 1168, 92, "channel 1 · minimum across segments · positive-outlier timing descriptor");
  const chunks = [
    ["c1", "Current Endo", C.blue],
    ["min", "Minimum of the descriptor across 18 segments", C.green],
    ["DN_Outlier…", "Catch22 positive-outlier timing statistic", C.orange],
  ];
  chunks.forEach((d,i)=>addCard(s,42+i*390,288,370,236,d[0],d[1],{fill:i===0?C.paleBlue:i===1?C.paleGreen:C.paleOrange,titleColor:d[2],titleSize:25}));
  addText(s, "Channel map: c1 current Endo · c2 current Mid · c3 current gap · c4 ΔEndo · c5 ΔMid · c6 Δgap", 92, 562, 1096, 42, { fontSize: 18, bold: true, align: "center" });
  addSourceTag(s, "catch22_structured_features naming convention");
  notes(s, "The aggregation occurs after calculating a descriptor within each segment curve. Therefore c1_min is an extreme segment summary: it retains the segment with the smallest descriptor value.", ["Local feature-name output: D:/us/cardiotoxicity_timeseries_round4_results/catch22_feature_names.csv", "Local construction: D:/us/cardiotoxicity_timeseries_round4.py:393-405"]);
}

// 34 catch importance
{
  const s = normalSlide(pres, "Catch22: an extreme Endo timing feature led the ranking", "Catch22 + XGBoost · top inputs", page++);
  addBars(s, [
    { label: "c1 min · positive-outlier timing", value: 0.0357, ci: "95% CI 0.0029–0.0653", color: C.orange },
    { label: "All curve descriptors", value: 0.0362, ci: "group permutation", color: C.teal },
    { label: "All scalar features", value: 0.0267, ci: "group permutation", color: C.blue },
    { label: "c5 median · high fluctuation", value: 0.0123, ci: "95% CI 0.0038–0.0205", color: C.green },
  ], 62, 176, 760, 90, 0.04);
  addCard(s, 856, 176, 354, 270, "Native TreeSHAP agrees", "The same c1-min outlier-timing feature ranked first. Baseline Endo GLS, current Mid peak dispersion, gap histogram mode and low-frequency power also ranked highly.", { fill: C.paleOrange });
  addCard(s, 856, 466, 354, 142, "Ensemble value", "Catch22 unique leave-one-out AUC ≈ 0.000, but AP +0.011. It may refine positive ranking more than global discrimination.", { fill: C.paleBlue, titleSize: 18 });
  addSourceTag(s, "held-out permutation + OOF TreeSHAP");
  notes(s, "The most important single Catch22 feature nearly accounts for the curve-group AUC drop, but correlated tree features mean this should not be interpreted as a univariate causal effect.", ["Local permutation: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/catch22_permutation_importance.csv", "Local TreeSHAP: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/catch22_native_tree_shap.csv"]);
}

// 35 outlier timing
{
  const s = normalSlide(pres, "Deep dive: positive-outlier timing in current Endo curves", "Catch22 + XGBoost · strongest feature", page++);
  addFormula(s, "feature = min across segments of positive-outlier timing(curve)", 42, 160, 698, 98, "Catch22 DN_OutlierInclude_p_001_mdrmd · c1 = current Endo");
  addCard(s, 42, 290, 698, 286, "How it is calculated conceptually", "Within each Endo segment curve, Catch22 progressively identifies unusually high values and summarizes the spacing/timing of those positive extremes. The model then keeps the minimum descriptor value across 18 segments.\n\nBecause longitudinal strain is mostly negative, ‘positive outlier’ means upward relative to that curve’s distribution—not necessarily strain > 0%.", { fill: C.paleBlue });
  addCard(s, 768, 160, 442, 186, "Possible physiological signal", "Atypical timing of upward excursions may reflect altered relaxation, post-systolic behavior, rebound or a distorted contour in one segment.", { fill: C.paleGreen });
  addCard(s, 768, 366, 442, 210, "Measurement caveat", "An upward excursion can also arise from tracking failure, valve-plane contamination or interpolation. The feature is best described as curve-contour timing—not a direct biomarker of one mechanism.", { fill: C.paleOrange });
  addSourceTag(s, "Catch22 feature documentation; local c1_min aggregation");
  notes(s, "The exact Catch22 algorithm uses a range of thresholds and summarizes median inter-event intervals; the readable physiological description should remain at the level of positive-extreme timing. The minimum aggregation means one unusually patterned segment can drive the feature.", ["https://time-series-features.gitbook.io/catch22/information-about-catch22/feature-descriptions/feature-overview-table", "https://link.springer.com/article/10.1007/s10618-019-00647-x", "Local construction: D:/us/cardiotoxicity_timeseries_round4.py:393-405"]);
}

// 36 catch other
{
  const s = normalSlide(pres, "Other high-value Catch22 features describe fluctuation and timing", "Catch22 + XGBoost · feature meanings", page++);
  const rows = [
    ["c5_median_MD_hrv_classic_pnn40", "ΔMid · segment median", "Fraction of successive changes > 0.04×curve SD", "Point-to-point contour instability or sharp local change"],
    ["c3_min_DN_HistogramMode_5", "Current gap · segment min", "Mode of a 5-bin value histogram", "Typical level/distribution of Endo−Mid discordance"],
    ["c1_min_SP_…area_5_1", "Current Endo · segment min", "Power in lowest 20% of frequencies", "Broad, smooth cycle-scale variation vs rapid fluctuation"],
    ["c3_max_CO_f1ecac", "Current gap · segment max", "First 1/e crossing of autocorrelation", "Persistence/time scale of the gap waveform"],
    ["c3_min_SB_Binary…longstretch1", "Current gap · segment min", "Longest run above the curve mean", "Duration of sustained above-mean gap behavior"],
  ];
  addTable(s, ["Technical name", "Scope", "Calculation", "Possible meaning"], rows, 42, 164, [310, 210, 326, 350], { rowH: 78, fontSize: 14.5, headerSize: 14 });
  addText(s, "These are generic time-series descriptors. Physiological meaning depends on the channel to which the descriptor is applied.", 126, 596, 1028, 34, { fontSize: 17.5, bold: true, align: "center" });
  addSourceTag(s, "Catch22 feature overview; native TreeSHAP ranking");
  notes(s, "Technical names are preserved so every slide can be traced to the exported feature table. The short readable names are from the pycatch22 implementation and the Catch22 documentation.", ["https://time-series-features.gitbook.io/catch22/information-about-catch22/feature-descriptions/feature-overview-table", "https://time-series-features.gitbook.io/catch22/information-about-catch22/feature-descriptions/incremental-differences", "Local TreeSHAP: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/catch22_native_tree_shap.csv"]);
}

// 37 ensemble top5
{
  const s = normalSlide(pres, "Top five feature groups inside each ensemble", "Integrated importance", page++);
  addText(s, "CNN + MOMENT + RDST", 42, 150, 510, 28, { fontSize: 20, bold: true, color: C.blue });
  addBars(s, [
    { label: "MOMENT segment max", value: 0.039, ci: "CI −0.000–0.074", color: C.orange },
    { label: "RDST best-match location", value: 0.030, ci: "CI 0.001–0.058", color: C.green },
    { label: "MOMENT ΔEndo−Mid", value: 0.020, ci: "CI −0.021–0.058", color: C.red },
    { label: "MOMENT all curve embeds", value: 0.018, ci: "CI −0.012–0.048", color: C.teal },
    { label: "RDST all shapelets", value: 0.018, ci: "CI −0.017–0.051", color: C.blue },
  ], 42, 194, 560, 72, 0.045);
  addText(s, "CNN + MOMENT + Catch22", 650, 150, 530, 28, { fontSize: 20, bold: true, color: C.teal });
  addBars(s, [
    { label: "MOMENT segment max", value: 0.042, ci: "CI 0.002–0.082", color: C.orange },
    { label: "MOMENT ΔEndo−Mid", value: 0.022, ci: "CI −0.022–0.069", color: C.red },
    { label: "MOMENT all curve embeds", value: 0.021, ci: "CI −0.010–0.052", color: C.teal },
    { label: "CNN all scalars", value: 0.015, ci: "CI 0.001–0.029", color: C.blue },
    { label: "Catch22 Endo outlier timing", value: 0.015, ci: "CI 0.002–0.027", color: C.green },
  ], 650, 194, 560, 72, 0.045);
  addSourceTag(s, "joint ensemble perturbation ΔAUC; correlated groups can share importance");
  notes(s, "The same three ideas recur: extreme segment representation, timing/location and changing Endo–Mid relationship. Only some intervals exclude zero, so the ranking should guide hypotheses and ablations rather than establish biomarkers.", ["Local ensemble importance: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/ensemble_feature_importance.csv", "Local report: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
}

// 38 synthesis
{
  const s = normalSlide(pres, "The strongest evidence favors spatial–temporal heterogeneity", "Physiological synthesis", page++);
  addCard(s, 42, 164, 372, 394, "Hypothesis 1\nEndo–Mid discordance", "Evidence: MOMENT change-gap channel ranks 2nd–3rd inside both ensembles. Current gap Catch22 features also appear in TreeSHAP.\n\nStrength: supportive; intervals usually cross zero.\n\nInterpretation: changing transmural mechanics may precede global Mid-GLS decline.", { fill: C.paleBlue, titleColor: C.blue });
  addCard(s, 434, 164, 372, 394, "Hypothesis 2\nInter-segment variability", "Evidence: MOMENT segment max, RDST match location, robust peak dispersion, Catch22 segment extremes and CNN SD/max pooling.\n\nStrength: more consistent across model families.\n\nInterpretation: focal or dyssynchronous dysfunction can be diluted by global averaging.", { fill: C.paleGreen, titleColor: C.green });
  addCard(s, 826, 164, 384, 394, "Clinical context\nBaseline and trajectory", "Evidence: scalar block is strong in CNN and remains useful in RDST/MOMENT/Catch22. Baseline GLS and current variability rank highly in TreeSHAP.\n\nInterpretation: curve morphology is most informative when interpreted relative to the patient’s starting function and recent course.", { fill: C.paleOrange, titleColor: C.orange });
  addText(s, "Best current scientific story: early deterioration may be a diffuse risk state with locally expressed timing and heterogeneity abnormalities.", 106, 594, 1068, 36, { fontSize: 18, bold: true, align: "center" });
  addSourceTag(s, "cross-model convergence; interpretation remains associative");
  notes(s, "This synthesis is an inference from converging model feature families, not a causal conclusion. A future validation cohort should prespecify a small set of heterogeneity and Endo–Mid features.", ["https://pmc.ncbi.nlm.nih.gov/articles/PMC7984733/", "https://pubmed.ncbi.nlm.nih.gov/26661049/", "Local ensemble report: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
}

// 39 limitations
{
  const s = normalSlide(pres, "What limits the physiological claims", "Limitations & performance ceiling", page++);
  const rows = [
    ["Small sample", "103 patients / 49 events", "Wide intervals; unstable rankings; overfitting risk"],
    ["Label derived from strain", "Outcome is a 15% Mid-GLS change", "Measurement noise affects both predictors and labels"],
    ["Correlated inputs", "Raw curves, GLS and variability overlap", "Importance is redistributed across substitutes"],
    ["Single pipeline/site", "Same vendor/report ecosystem", "Unknown transportability and calibration"],
    ["No causal intervention", "Retrospective observational prediction", "Physiological interpretations are hypotheses"],
    ["Event imbalance", "20.6% positives", "AP is sensitive to prevalence and false positives"],
  ];
  addTable(s, ["Constraint", "Where it enters", "Consequence"], rows, 42, 160, [250, 390, 556], { rowH: 66, fontSize: 15.5, headerSize: 15 });
  addText(s, "Publication positioning: a hypothesis-generating ML study with internal validation—not a clinically deployable alert system.", 104, 596, 1072, 36, { fontSize: 18, bold: true, color: C.red, align: "center" });
  addSourceTag(s, "error/noise analysis and bootstrap uncertainty");
  notes(s, "Label noise may impose a performance ceiling because the endpoint is itself based on strain measurement. Repeatability analysis, threshold sensitivity and probabilistic/noise-aware labels should be reported alongside model performance.", ["Local noise analysis: D:/us/cardiotoxicity_error_noise.py", "Local plateau results: D:/us/cardiotoxicity_plateau_results", "https://academic.oup.com/eurheartj/article/43/41/4229/6673995"]);
}

// 40 grammar
{
  const s = normalSlide(pres, "Appendix: how to decode a scalar feature name", "Feature dictionary", page++);
  addFormula(s, "d_var__mid_spatial_timing_graph_roughness", 42, 160, 1168, 94, "change feature · Mid layer · neighboring-segment timing disagreement");
  addCard(s, 42, 288, 362, 246, "Prefix", "cur_var__ = value at current visit\nd_var__ = current − previous visit\nfirst_ = first-visit baseline\ncurrent_ = current landmark value\nlast_ = current vs previous", { fill: C.paleBlue });
  addCard(s, 424, 288, 362, 246, "Layer", "endo = endocardial strain\nmid = mid-wall strain\ntm / gap = Endo–Mid relationship\n\nPeak features generally use absolute strain magnitude.", { fill: C.paleGreen });
  addCard(s, 806, 288, 404, 246, "Suffix", "Names the within-visit operation: amplitude dispersion, timing dispersion, curve dispersion, shape incoherence, spatial roughness, impairment fraction or regional gradient.", { fill: C.paleOrange });
  addText(s, "All outcome decline calculations use first visit as the baseline; rolling features are predictor context only.", 164, 580, 952, 32, { fontSize: 17.5, bold: true, align: "center" });
  addSourceTag(s, "cardiotoxicity_next_visit_gpu.py naming and landmark construction");
  notes(s, "This naming grammar lets a reader trace any exported scalar column back to its calculation. The baseline for the label is always first visit, as requested.", ["Local scalar construction: D:/us/cardiotoxicity_next_visit_gpu.py", "Local variability suffix list: D:/us/cardiotoxicity_early_detection.py:44-62"]);
}

// 41 clinical catalog 1
{
  const s = normalSlide(pres, "Appendix: visit timing and current-state features", "Clinical scalar catalog · 1/2", page++);
  const rows = [
    ["history_visits", "Number of visits available through t", "Exposure/history depth; opportunity for trend estimation"],
    ["days_since_first", "date(t) − date(first)", "Elapsed treatment/follow-up time"],
    ["days_since_previous", "date(t) − date(t−1)", "Controls for irregular follow-up intervals"],
    ["has_previous_visit", "1 if t has a prior visit, else 0", "Distinguishes true change channels from first landmark"],
    ["current_mid_gls / current_endo_gls", "|GLS| at t", "Current global longitudinal contractile magnitude"],
    ["current_ef", "Biplane EF at t", "Conventional global systolic function"],
    ["first_mid_gls / first_endo_gls / first_ef", "Value at first visit", "Patient-specific starting function and reserve"],
  ];
  addTable(s, ["Feature", "Exact calculation", "Why it may matter"], rows, 42, 154, [330, 410, 456], { rowH: 63, fontSize: 15, headerSize: 15 });
  addSourceTag(s, "cardiotoxicity_next_visit_gpu.py · magnitudes use abs(GLS)");
  notes(s, "Current and first GLS are stored as positive magnitudes for trajectory features, even though raw longitudinal strain curves are signed and usually negative during systole.", ["Local feature construction: D:/us/cardiotoxicity_next_visit_gpu.py"]);
}

// 42 clinical catalog 2
{
  const s = normalSlide(pres, "Appendix: relative trajectory and gap features", "Clinical scalar catalog · 2/2", page++);
  const rows = [
    ["current_*_decline_from_first", "1 − current / first", "Relative loss from patient-specific baseline"],
    ["last_*_relative_change", "1 − current / previous", "Most recent relative step; interval length ignored"],
    ["*_decline_slope_per_100d", "(1 − current/previous) × 100 / Δdays", "Rate-normalized worsening between visits"],
    ["current_endo_mid_gap", "current Endo GLS − current Mid GLS", "Global transmural strain-gradient magnitude"],
    ["last_endo_mid_gap_change", "gap(t) − gap(t−1)", "Recent evolution of global Endo–Mid relationship"],
    ["current_*_decline_from_roll2", "1 − current / mean(last 2 values)", "Smooths one prior noisy measurement; predictor only"],
  ];
  addTable(s, ["Feature family", "Exact calculation", "Why it may matter"], rows, 42, 160, [350, 410, 436], { rowH: 70, fontSize: 15.5, headerSize: 15 });
  addText(s, "* is Mid-GLS, Endo-GLS or EF. Absolute decline in percentage points is not used for the outcome.", 168, 604, 944, 30, { fontSize: 17, bold: true, align: "center" });
  addSourceTag(s, "relative decline only; first visit is outcome baseline");
  notes(s, "The rolling two-visit comparison is an input feature, not an alternative outcome baseline. The endpoint remains a relative Mid-GLS decline from the first visit.", ["Local feature construction: D:/us/cardiotoxicity_next_visit_gpu.py"]);
}

// 43 variability catalog 1
{
  const s = normalSlide(pres, "Appendix: amplitude and timing variability", "Variability suffixes · 1/3", page++);
  const rows = [
    ["peak_abs_robust_sd", "1.4826 × MAD of segment |peak strain|", "Robust regional amplitude heterogeneity"],
    ["peak_abs_cv", "SD(|peak|) / mean(|peak|)", "Amplitude dispersion relative to overall contraction"],
    ["time_to_peak_norm_circular_std", "Circular SD of peak time / cycle", "Segmental timing dispersion with wrap-around handling"],
    ["vendor_peak_systolic_abs_robust_sd", "Robust SD of vendor peak-systolic magnitudes", "Vendor-defined systolic amplitude heterogeneity"],
    ["vendor_time_to_peak_norm_circular_std", "Circular SD of vendor time-to-peak", "Vendor timing heterogeneity"],
    ["post_systolic_fraction", "Fraction of segments whose peak occurs after systole", "Post-systolic shortening / delayed contraction burden"],
  ];
  addTable(s, ["Suffix", "Calculation", "Physiological interpretation"], rows, 42, 154, [350, 410, 436], { rowH: 72, fontSize: 15, headerSize: 15 });
  addSourceTag(s, "same suffixes calculated for Endo and Mid; current and Δ-from-previous");
  notes(s, "Circular SD treats normalized time as a position on a cycle, avoiding an artificial large difference between values near 0 and 1. Post-systolic fraction depends on the systolic boundary available in the report pipeline.", ["Local variability suffixes: D:/us/cardiotoxicity_early_detection.py:44-62", "Local circular SD reference implementation: D:/us/cardiotoxicity_nonapical_qc.py:98-105"]);
}

// 44 variability catalog 2
{
  const s = normalSlide(pres, "Appendix: full-curve and normalized-shape variability", "Variability suffixes · 2/3", page++);
  const rows = [
    ["curve_dispersion_rms", "RMS of segment curves around timewise mean curve", "Whole-cycle between-segment disagreement"],
    ["curve_pairwise_rmse", "Mean RMSE over every segment pair", "Average pairwise curve dissimilarity"],
    ["curve_integrated_robust_mad", "Robust segment MAD integrated over time", "Noise-resistant whole-cycle amplitude dispersion"],
    ["shape_dispersion_rms", "RMS after dividing each curve by its max |strain|", "Morphology/timing disagreement independent of amplitude"],
    ["shape_pairwise_rmse", "Mean pairwise RMSE of normalized shapes", "Pairwise contour dissimilarity"],
    ["shape_incoherence", "1 − mean pairwise shape correlation", "Loss of coordinated curve morphology"],
  ];
  addTable(s, ["Suffix", "Calculation", "Physiological interpretation"], rows, 42, 154, [350, 410, 436], { rowH: 72, fontSize: 15, headerSize: 15 });
  addSourceTag(s, "raw-curve features mix amplitude + shape; normalized-shape features emphasize contour");
  notes(s, "Normalization is only applied when curve magnitude is sufficiently large; otherwise shape values are treated as invalid to avoid amplifying noise. Shape features are designed to distinguish ‘contracts less’ from ‘contracts differently.’", ["Local normalized shape: D:/us/cardiotoxicity_early_detection.py:130-137", "Local pairwise/RMS definitions: D:/us/cardiotoxicity_nonapical_qc.py:76-95 and 205-219"]);
}

// 45 variability catalog 3
{
  const s = normalSlide(pres, "Appendix: regional and spatial variability", "Variability suffixes · 3/3", page++);
  const rows = [
    ["within_view_peak_robust_sd_mean", "Robust peak SD within each apical view; average", "Heterogeneity within imaging view"],
    ["within_ring_peak_robust_sd_mean", "Robust peak SD within each LV ring; average", "Regional heterogeneity controlling for level"],
    ["spatial_peak_graph_roughness", "Mean |peakᵢ−peakⱼ| over anatomical neighbor edges", "Abrupt spatial amplitude discontinuity"],
    ["spatial_timing_graph_roughness", "Mean |TTPᵢ−TTPⱼ| over neighbor edges", "Local mechanical timing discontinuity"],
    ["impaired_segment_fraction_lt15", "Fraction with |peak strain| <15%", "Burden of weakly shortening segments"],
    ["apical_basal_peak_gradient", "mean apical peak − mean basal peak", "Change in normal base-to-apex strain gradient"],
  ];
  addTable(s, ["Suffix", "Calculation", "Physiological interpretation"], rows, 42, 154, [350, 410, 436], { rowH: 72, fontSize: 15, headerSize: 15 });
  addSourceTag(s, "spatial features depend on the segment adjacency/ring/view map");
  notes(s, "Normal layer-specific strain also varies from base to apex. Ring- and gradient-aware features reduce the chance that normal anatomy is mistaken for pathology, although vendor segment mapping and apex tracking remain possible sources of noise.", ["Local suffix list: D:/us/cardiotoxicity_early_detection.py:44-62", "Local spatial reference implementation: D:/us/cardiotoxicity_nonapical_qc.py:223-244", "https://pubmed.ncbi.nlm.nih.gov/26661049/"]);
}

// 46 catch22 full catalog
{
  const s = normalSlide(pres, "Appendix: the complete Catch22 descriptor vocabulary", "Catch22 feature dictionary", page++);
  const left = [
    ["HistogramMode_5 / _10", "Mode of 5- or 10-bin value histogram"],
    ["OutlierInclude_p / n", "Timing of positive / negative extremes"],
    ["CO_f1ecac", "First 1/e crossing of autocorrelation"],
    ["CO_FirstMin_ac", "First minimum of autocorrelation"],
    ["Welch area_5_1", "Power in lowest frequency fifth"],
    ["Welch centroid", "Frequency-spectrum centroid"],
    ["LocalSimple_mean3_stderr", "3-point mean forecast error"],
    ["LocalSimple_mean1_tauresrat", "ACF-timescale change after differencing"],
    ["MD_hrv_classic_pnn40", "Fraction of increments >0.04×SD"],
    ["BinaryStats_diff_longstretch0", "Longest run of decreasing increments"],
    ["BinaryStats_mean_longstretch1", "Longest run above series mean"],
  ];
  const right = [
    ["MotifThree_quantile_hh", "Entropy of successive symbols"],
    ["HistogramAMI_even_2_5", "Lag-2 histogram mutual information"],
    ["CO_trev_1_num", "Time-reversibility statistic"],
    ["AutoMutualInfoStats_fmmi", "First minimum of automutual information"],
    ["TransitionMatrix_3ac", "Variance structure of symbolic transitions"],
    ["PeriodicityWang", "Periodicity metric"],
    ["Embed2_Dist_tau", "Exponential fit of embedding distances"],
    ["FluctAnal_rsrangefit", "Rescaled-range scaling transition"],
    ["FluctAnal_dfa", "Detrended-fluctuation scaling transition"],
    ["SC_FluctAnal…", "Multiscale fluctuation behavior"],
    ["All descriptors", "Computed per curve, then pooled across segments"],
  ];
  addTable(s, ["Descriptor", "Meaning"], left, 42, 150, [310, 282], { rowH: 38, headerH: 38, fontSize: 13.5, headerSize: 14 });
  addTable(s, ["Descriptor", "Meaning"], right, 646, 150, [316, 248], { rowH: 38, headerH: 38, fontSize: 13.5, headerSize: 14 });
  addText(s, "Names may be abbreviated on slides; exported CSV retains the exact pycatch22 technical name.", 158, 616, 964, 28, { fontSize: 16.5, bold: true, align: "center" });
  addSourceTag(s, "Lubba et al. 2019; Catch22 feature overview");
  notes(s, "Catch22 was selected as a compact, minimally redundant feature set from a much larger time-series feature library. These descriptors are generic; the physiological meaning comes from applying them to a specific strain channel and segment aggregation.", ["https://link.springer.com/article/10.1007/s10618-019-00647-x", "https://time-series-features.gitbook.io/catch22/information-about-catch22/feature-descriptions/feature-overview-table", "Local names: D:/us/cardiotoxicity_timeseries_round4_results/catch22_feature_names.csv"]);
}

// 47 reproducibility
{
  const s = normalSlide(pres, "Reproducibility and source map", "Audit trail", page++);
  const rows = [
    ["Landmarks, label and scalars", "cardiotoxicity_next_visit_gpu.py"],
    ["CNN architecture and six channels", "cardiotoxicity_cnn_channel_ablation.py"],
    ["MOMENT extraction", "cardiotoxicity_timeseries_round1.py"],
    ["RDST, Catch22 and ensembles", "cardiotoxicity_timeseries_round4.py"],
    ["Held-out importance / Shapley", "cardiotoxicity_top_ensemble_feature_importance.py"],
    ["Detailed tables and interpretation", "cardiotoxicity_top_ensemble_feature_importance_results/"],
    ["Error and label-noise analysis", "cardiotoxicity_error_noise.py; cardiotoxicity_plateau_results/"],
  ];
  addTable(s, ["Purpose", "Local source under D:/us/"], rows, 42, 158, [430, 766], { rowH: 53, fontSize: 16, headerSize: 15 });
  addShape(s, "roundRect", 42, 584, 1168, 54, C.ink, C.ink, 0, "rounded-lg");
  addText(s, "Reproduction check: all four constituent OOF score vectors matched the original outputs exactly (correlation = 1.000; identical AUC/AP).", 64, 600, 1124, 24, { fontSize: 16.5, bold: true, color: C.white, align: "center" });
  addSourceTag(s, "all analyses patient-held-out; same folds preserved across comparisons");
  notes(s, "This slide is the fastest way for a reviewer or supervisor to trace a feature from the presentation to the implementation and exported result tables.", ["D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
}

// 48 references
{
  const s = normalSlide(pres, "Primary references", "Methods & physiology", page++);
  const refs = [
    ["Clinical definition", "Lyon et al. 2022 ESC Guidelines on cardio-oncology", "Eur Heart J. 2022;43:4229–4361."],
    ["Segmental cardiotoxicity", "Narayan et al. Left ventricular segmental strain…", "Eur Heart J Cardiovasc Imaging. 2021."],
    ["Layer-specific strain", "Shi et al. LV layer-specific strains in healthy subjects", "Echocardiography. 2016."],
    ["MOMENT", "Goswami et al. A family of open time-series foundation models", "ICML 2024."],
    ["RDST", "Guillaume et al. Random Dilated Shapelet Transform", "2021/2022."],
    ["Catch22", "Lubba et al. CAnonical Time-series CHaracteristics", "Data Min Knowl Disc. 2019."],
    ["XGBoost", "Chen & Guestrin. A scalable tree boosting system", "KDD 2016."],
  ];
  refs.forEach((r,i)=>{
    const y=154+i*66;
    addPill(s, r[0], 42, y+5, 180, i<3?C.blue:i===3?C.teal:i===4?C.green:i===5?C.orange:C.ink);
    addText(s, r[1], 240, y, 670, 28, { fontSize: 16.5, bold: true });
    addText(s, r[2], 240, y+31, 670, 22, { fontSize: 14.5, color: C.muted });
  });
  addCard(s, 938, 154, 272, 426, "Source policy", "Primary papers and official guideline sources support methodological and physiological claims.\n\nExact feature calculations and model dimensions are sourced to the local implementation.\n\nSpeaker notes on every slide contain URLs or local paths.", { fill: C.paleOrange, titleSize: 21 });
  addSourceTag(s, "full URLs are embedded in speaker notes");
  notes(s, "Primary sources used throughout the deck are listed here; exact URLs are preserved in this slide’s notes.", [
    "https://academic.oup.com/eurheartj/article/43/41/4229/6673995",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7984733/",
    "https://pubmed.ncbi.nlm.nih.gov/26661049/",
    "https://arxiv.org/abs/2402.03885",
    "https://arxiv.org/abs/2109.13514",
    "https://link.springer.com/article/10.1007/s10618-019-00647-x",
    "https://doi.org/10.1145/2939672.2939785",
  ]);
}

// 49 conclusion
{
  const s = pres.slides.add();
  s.background.fill = C.ink;
  addText(s, "TAKE-HOME MESSAGE", 58, 58, 360, 24, { fontSize: 13, bold: true, color: C.cyan });
  addText(s, "The models converge on a coherent—yet still exploratory—signal.", 58, 126, 1080, 112, { fontSize: 50, bold: true, color: C.white });
  addCard(s, 58, 294, 350, 206, "01 · Context", "First-visit baseline and recent trajectory remain essential.", { fill: C.paleBlue, titleColor: C.blue });
  addCard(s, 430, 294, 350, 206, "02 · Heterogeneity", "Extreme segment representations and timing features are most consistent.", { fill: C.paleGreen, titleColor: C.green });
  addCard(s, 802, 294, 420, 206, "03 · Endo–Mid", "Changing transmural discordance is promising but not yet statistically conclusive.", { fill: C.paleOrange, titleColor: C.orange });
  addText(s, "Best model: CNN + MOMENT + RDST · AUC 0.706 · AP 0.362", 58, 558, 900, 34, { fontSize: 22, bold: true, color: C.cyan });
  addText(s, "Next scientific step: external validation with prespecified heterogeneity and Endo–Mid features, plus repeatability-aware labels.", 58, 610, 1120, 54, { fontSize: 19, color: C.rule });
  notes(s, "Close by emphasizing the distinction between a coherent mechanistic hypothesis and a clinically validated biomarker. The deck supports a thesis methods/results narrative, but the performance and uncertainty require cautious positioning.", ["Local full report: D:/us/cardiotoxicity_top_ensemble_feature_importance_results/top_ensemble_feature_importance_report.md"]);
  page++;
}

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(RENDER_DIR, { recursive: true });
for (let i = 0; i < pres.slides.items.length; i++) {
  const slide = pres.slides.items[i];
  const stem = `slide-${String(i + 1).padStart(2, "0")}`;
  const png = await pres.export({ slide, format: "png", scale: 1 });
  await writeBlob(path.join(RENDER_DIR, `${stem}.png`), png);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await layout.text());
}
const montage = await pres.export({ format: "webp", montage: true, scale: 0.5 });
await writeBlob(path.join(RENDER_DIR, "deck-montage.webp"), montage);
const pptx = await PresentationFile.exportPptx(pres);
await pptx.save(OUT);
console.log(JSON.stringify({ slides: pres.slides.items.length, output: OUT, renderDir: RENDER_DIR }, null, 2));
