import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
import { buildSlide02 } from "./slide-02.mjs";
import { buildSlide04 } from "./slide-04.mjs";
import { buildSlide14 } from "./slide-14.mjs";
import { buildSlide17 } from "./slide-17.mjs";
import { buildSlide19 } from "./slide-19.mjs";
import { buildSlide20 } from "./slide-20.mjs";

const OUT_DIR = "D:/us/presentation_sources/feature_importance/rendered";
const OUT_PPTX = "D:/us/cardiotoxicity_feature_importance_presentation.pptx";

const C = {
  navy: "#17324D",
  ink: "#17202A",
  muted: "#5B6773",
  clinical: "#277DA1",
  clinicalPale: "#E8F3F8",
  endo: "#F8961E",
  endoPale: "#FFF1DF",
  variability: "#43AA8B",
  variabilityPale: "#E6F6F1",
  grayPale: "#F4F6F8",
  red: "#D95D5D",
  white: "#FFFFFF",
};

const report = "D:/us/cardiotoxicity_feature_importance_results/noncnn_feature_importance_report.md";
const topCsv = "D:/us/cardiotoxicity_feature_importance_results/noncnn_feature_importance_top.csv";
const familyCsv = "D:/us/cardiotoxicity_feature_importance_results/noncnn_feature_family_importance.csv";
const metricsCsv = "D:/us/cardiotoxicity_feature_importance_results/noncnn_base_metric_reproduction.csv";

function para(run, size = 21, color = C.ink, bold = false, spaceAfter = 440) {
  return {
    runs: [{ run, textStyle: { fontSize: `${size}px`, typeface: "Aptos", color, bold } }],
    spaceAfter,
    paragraphStyle: { lineSpacingPercent: 112000 },
  };
}

function title(run) {
  return para(run, 39, C.navy, true, 0);
}

function label(run, color = C.muted) {
  return para(run, 17, color, true, 0);
}

function addTopRule(slide, color = C.navy) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: 1280, height: 10 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addCardAccent(slide, left, top, width, color) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height: 10 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addFamilyLegend(slide, y = 666) {
  const items = [
    [C.clinical, "Clinical"],
    [C.endo, "Endo–Mid"],
    [C.variability, "Variability"],
  ];
  let x = 42;
  for (const [color, text] of items) {
    slide.shapes.add({ geometry: "ellipse", position: { left: x, top: y + 2, width: 12, height: 12 }, fill: color, line: { style: "solid", fill: color, width: 0 } });
    const box = slide.shapes.add({ geometry: "textbox", position: { left: x + 18, top: y - 2, width: 118, height: 22 }, fill: "none", line: { style: "solid", fill: "none", width: 0 } });
    box.text = text;
    box.text.style = { fontSize: 14, typeface: "Aptos", color: C.muted };
    x += 142;
  }
}

function notes(slide, summary, sources) {
  const sourceBlock = sources.map((source) => `- ${source}`).join("\n");
  slide.speakerNotes.textFrame.setText(`${summary}\n\n[Sources]\n${sourceBlock}\n[/Sources]`);
  slide.speakerNotes.setVisible(true);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1 — title
  {
    const slide = buildSlide02(presentation, {
      title: label("NON-CNN FEATURE ANALYSIS", C.endo),
      title2: label("15% MID-GLS DECLINE", C.clinical),
      title3: para("WHAT PREDICTS NEXT-VISIT\nCARDIOTOXICITY?", 68, C.navy, true, 0),
    });
    addTopRule(slide, C.endo);
    slide.shapes.add({ geometry: "rect", position: { left: 1055, top: 270, width: 55, height: 230 }, fill: C.clinical, line: { style: "solid", fill: C.clinical, width: 0 } });
    slide.shapes.add({ geometry: "rect", position: { left: 1120, top: 335, width: 55, height: 165 }, fill: C.endo, line: { style: "solid", fill: C.endo, width: 0 } });
    slide.shapes.add({ geometry: "rect", position: { left: 1185, top: 400, width: 55, height: 100 }, fill: C.variability, line: { style: "solid", fill: C.variability, width: 0 } });
    notes(slide, "A concise presentation of the non-CNN feature-importance analysis.", [report]);
  }

  // 2 — prediction task and cohort
  {
    const slide = buildSlide19(presentation, {
      title: title("One-visit-ahead prediction, defined consistently"),
      body1: {
        topic: para("TASK", 16, C.endo, true, 160),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("At each eligible visit, use only information already available to predict whether the immediately following visit shows ≥15% relative Mid-GLS decline from the first-visit baseline.", 23, C.ink, false, 0),
      },
      stat1: para("103", 62, C.clinical, true, 0),
      stat2: para("238", 62, C.endo, true, 0),
      stat3: para("20.6%", 62, C.variability, true, 0),
      body2: para("patients", 22, C.muted, true, 0),
      body3: para("current → next\nvisit predictions", 22, C.muted, true, 0),
      body4: para("events\n49 vs 189 non-events", 22, C.muted, true, 0),
      footer1: "2",
    });
    addTopRule(slide, C.navy);
    addCardAccent(slide, 41.33, 317.33, 374.67, C.clinical);
    addCardAccent(slide, 452.67, 317.33, 374.67, C.endo);
    addCardAccent(slide, 864.28, 317.33, 374.67, C.variability);
    notes(slide, "The unit is a current-to-next-visit transition, and all transitions from a patient stay in the same held-out fold.", [report]);
  }

  // 3 — model performance
  {
    const slide = buildSlide20(presentation, {
      title: title("Non-CNN model performance"),
      body1: {
        titleGoesHere: para("BEST AUC", 16, C.endo, true, 120),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("0.672\nCombined Extra Trees", 30, C.navy, true, 0),
      },
      body2: {
        titleGoesHere: para("BEST AP", 16, C.endo, true, 120),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("0.313\nCombined Extra Trees", 30, C.navy, true, 0),
      },
      body3: {
        titleGoesHere: para("REFERENCE", 16, C.clinical, true, 120),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("Random: AUC 0.500\nExpected AP 0.206", 27, C.navy, true, 0),
      },
      chart: {
        categories: ["Clinical", "+ Endo–Mid", "+ Variability", "Extra Trees"],
        series: [
          { name: "AUC ×100", values: [63.1, 64.4, 65.0, 67.2], fill: C.clinical },
          { name: "AP ×100", values: [28.9, 30.9, 28.8, 31.3], fill: C.endo },
        ],
        min: 0,
        max: 75,
        majorUnit: 15,
      },
      footer1: "3",
    });
    addTopRule(slide, C.navy);
    addCardAccent(slide, 657.68, 41.33, 580.99, C.endo);
    addCardAccent(slide, 657.68, 248.8, 580.99, C.endo);
    addCardAccent(slide, 657.68, 460, 580.99, C.clinical);
    notes(slide, "AUC improved from 0.631 for the clinical ridge to 0.672 for Combined Extra Trees. AP improved from 0.289 to 0.313; the event-rate reference is 0.206.", [report, metricsCsv]);
  }

  // 4 — method
  {
    const slide = buildSlide17(presentation, {
      title: title("Feature contribution was tested on held-out patients"),
      label1: label("1  FIT", C.clinical),
      label2: label("2  SHUFFLE", C.endo),
      label3: label("3  MEASURE", C.variability),
      body1: {
        titleHere: para("Train by patient", 25, C.navy, true, 220),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("Three repeated five-fold splits; every visit from one patient stays together.", 20, C.muted, false, 0),
      },
      body2: {
        titleHere: para("Break one feature", 25, C.navy, true, 220),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("Randomly permute that feature in held-out patients while leaving all others unchanged.", 20, C.muted, false, 0),
      },
      body3: {
        titleHere: para("Observe the loss", 25, C.navy, true, 220),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("The decrease in AUC or AP is its predictive contribution. Patient bootstrap gives 95% CIs.", 20, C.muted, false, 0),
      },
      footer1: "4",
    });
    addTopRule(slide, C.navy);
    slide.shapes.add({ geometry: "ellipse", position: { left: 35.46, top: 348.58, width: 11.24, height: 11.24 }, fill: C.clinical, line: { style: "solid", fill: C.clinical, width: 0 } });
    slide.shapes.add({ geometry: "ellipse", position: { left: 446.38, top: 348.58, width: 11.24, height: 11.24 }, fill: C.endo, line: { style: "solid", fill: C.endo, width: 0 } });
    slide.shapes.add({ geometry: "ellipse", position: { left: 858.38, top: 348.58, width: 11.24, height: 11.24 }, fill: C.variability, line: { style: "solid", fill: C.variability, width: 0 } });
    notes(slide, "Permutation importance asks whether held-out prediction gets worse when one feature is made uninformative. Positive drops indicate genuine model dependence.", [report, topCsv]);
  }

  // 5 — leading Endo–Mid feature
  {
    const slide = buildSlide04(presentation, {
      title: title("Top sparse-model feature: changing Endo–Mid heterogeneity"),
      body1: {
        titleHere: para("d_tm_sd_gap_dct04", 19, C.endo, true, 260),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("1. Pair Endocardial and Mid-wall curves from each matched segment.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing2: para("2. At every cardiac-cycle point, compute Endo − Mid for each segment.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing3: para("3. Across segments, calculate the SD of that gap, producing an SD-gap curve.", 21, C.ink, false, 0),
      },
      body2: {
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("4. Apply the Discrete Cosine Transform and keep DCT04, a phase-dependent pattern component.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing2: para("5. Subtract the previous visit: current DCT04 − previous DCT04.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing3: para("Meaning: the timing pattern of layer-to-layer disagreement across segments changed—not simply that variability became larger.\n\nΔAUC 0.037 (95% CI 0.001–0.066); ΔAP 0.043.", 21, C.endo, true, 0),
      },
      footer1: "5",
    });
    addTopRule(slide, C.endo);
    addFamilyLegend(slide);
    notes(slide, "This is the strongest engineered feature in the sparse Endo–Mid model. DCT04 describes a mid-frequency temporal pattern rather than a single time point.", [report, topCsv]);
  }

  // 6 — leading Extra Trees feature
  {
    const slide = buildSlide04(presentation, {
      title: title("Top Extra Trees feature: normalized layer separation"),
      body1: {
        titleHere: para("cur_tm_mean_shape_gap_dct01\n", 18, C.endo, true, 260),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("1. Normalize each Endocardial curve by its own maximum absolute amplitude.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing2: para("2. Normalize the matched Mid-wall curve the same way.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing3: para("3. Compute normalized Endo − Mid at each time point and average across segments.", 21, C.ink, false, 0),
      },
      body2: {
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("4. Apply the DCT and keep DCT01—the broad average level of the shape-gap curve.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing2: para("Because amplitude is normalized first, the feature emphasizes shape differences more than strain magnitude.", 21, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing3: para("Meaning: systematic separation between Endocardial and Mid-wall contraction patterns at the current visit. Extra Trees uses nonlinear thresholds, so there is no single global risk direction.\n\nΔAUC 0.017 (95% CI 0.008–0.026); ΔAP 0.020.", 20, C.endo, true, 0),
      },
      footer1: "6",
    });
    addTopRule(slide, C.endo);
    addFamilyLegend(slide);
    notes(slide, "DCT01 summarizes broad curve separation after per-curve normalization. This was the most important feature within Combined Extra Trees.", [report, topCsv]);
  }

  // 7 — clinical and variability context
  {
    const slide = buildSlide04(presentation, {
      title: title("Clinical GLS was useful; segment variability was weaker"),
      body1: {
        titleHere: para("CLINICAL  •  first_endo_gls", 18, C.clinical, true, 260),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("Absolute magnitude of first-visit Endocardial GLS: |baseline Endo-GLS|.", 22, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing2: para("Example: −21.4% becomes 21.4% and is carried forward for that patient.", 22, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing3: para("ΔAUC 0.018 (95% CI 0.003–0.035). Its direction may reflect reserve, correlation, or regression to the mean—not causality.", 21, C.clinical, true, 0),
      },
      body2: {
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("VARIABILITY  •  Endo time-to-peak dispersion", 18, C.variability, true, 260),
        loremIpsumDolorSitAmetConsecteturAdipiscing2: para("For each Endocardial segment, obtain normalized time-to-peak. Because cardiac time is cyclic, summarize their spread with circular SD.", 22, C.ink),
        loremIpsumDolorSitAmetConsecteturAdipiscing3: para("Meaning: how synchronized—or dispersed—the segments are in reaching peak strain.\n\nΔAUC 0.008 (95% CI 0.003–0.014); ΔAP 0.007. Positive but smaller than the leading Endo–Mid signals.", 21, C.variability, true, 0),
      },
      footer1: "7",
    });
    addTopRule(slide, C.navy);
    addFamilyLegend(slide);
    notes(slide, "Baseline Endocardial GLS was the strongest stable clinical feature. The most stable variability feature measured circular dispersion of segmental time-to-peak.", [report, topCsv]);
  }

  // 8 — stable feature ranking
  {
    const tableValues = [
      ["Plain-language feature", "Family", "Model", "ΔAUC", "95% CI"],
      ["Change in phase pattern of segmental layer-gap variability", "Endo–Mid", "+ Endo–Mid", "0.037", "0.001–0.066"],
      ["Baseline Endocardial GLS magnitude", "Clinical", "Clinical ridge", "0.018", "0.003–0.035"],
      ["Overall normalized Endo–Mid shape separation", "Endo–Mid", "Extra Trees", "0.017", "0.008–0.026"],
      ["Finer temporal pattern of normalized shape separation", "Endo–Mid", "Extra Trees", "0.009", "0.002–0.015"],
      ["Endocardial segment time-to-peak dispersion", "Variability", "Extra Trees", "0.008", "0.003–0.014"],
      ["Change in mean vendor Endo–Mid peak gap", "Endo–Mid", "Extra Trees", "0.005", "0.001–0.010"],
    ];
    const slide = buildSlide14(presentation, {
      title: title("Six features had stable positive held-out AUC importance"),
      body1: {
        topic: para("RANKING", 16, C.endo, true, 120),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("ΔAUC is the performance lost after shuffling the feature. Only patient-bootstrap intervals entirely above zero are shown.", 21, C.muted, false, 0),
      },
      tableValues,
      columnWidths: [440, 150, 225, 120, 262.33],
      footer1: "8",
    });
    addTopRule(slide, C.navy);
    const rowTop = 236.33;
    const rowH = 412.87 / tableValues.length;
    const familyColors = [C.navy, C.endo, C.clinical, C.endo, C.endo, C.variability, C.endo];
    familyColors.forEach((color, i) => {
      slide.shapes.add({ geometry: "rect", position: { left: 41.33, top: rowTop + i * rowH, width: 7, height: rowH }, fill: color, line: { style: "solid", fill: color, width: 0 } });
    });
    notes(slide, "The stable list is dominated by Endo–Mid features. Baseline Endocardial GLS and one segment-timing dispersion feature also remained stable.", [report, topCsv]);
  }

  // 9 — family-level result
  {
    const slide = buildSlide19(presentation, {
      title: title("At family level, Endo–Mid carried the clearest signal"),
      body1: {
        topic: para("COMBINED EXTRA TREES • FAMILY PERMUTATION", 16, C.endo, true, 160),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("All features from one family were shuffled together. This captures contribution even when correlated features can substitute for one another.", 23, C.ink, false, 0),
      },
      stat1: para("−0.008", 54, C.clinical, true, 0),
      stat2: para("+0.054", 54, C.endo, true, 0),
      stat3: para("+0.016", 54, C.variability, true, 0),
      body2: para("Clinical\nΔAUC", 22, C.muted, true, 0),
      body3: para("Endo–Mid\nΔAUC", 22, C.muted, true, 0),
      body4: para("Variability\nΔAUC", 22, C.muted, true, 0),
      footer1: "9",
    });
    addTopRule(slide, C.navy);
    addCardAccent(slide, 41.33, 317.33, 374.67, C.clinical);
    addCardAccent(slide, 452.67, 317.33, 374.67, C.endo);
    addCardAccent(slide, 864.28, 317.33, 374.67, C.variability);
    notes(slide, "In Combined Extra Trees, grouped shuffling caused the largest AUC loss for Endo–Mid features. Clinical features were redundant in this nonlinear combined model.", [report, familyCsv]);
  }

  // 10 — take-home
  {
    const slide = buildSlide19(presentation, {
      title: title("What to tell the supervisor"),
      body1: {
        topic: para("TAKE-HOME", 16, C.endo, true, 160),
        loremIpsumDolorSitAmetConsecteturAdipiscing: para("The results support the Endo–Mid hypothesis more strongly than the inter-segment variability hypothesis, but the gain over clinical prediction remains modest.", 25, C.ink, true, 0),
      },
      stat1: para("1", 58, C.clinical, true, 0),
      stat2: para("2", 58, C.endo, true, 0),
      stat3: para("3", 58, C.variability, true, 0),
      body2: para("First-visit baseline\nOne-visit-ahead target", 20, C.ink, false, 0),
      body3: para("Prioritize Endo–Mid\nshape + change", 20, C.ink, false, 0),
      body4: para("Interpret DCT cautiously\nValidate externally", 20, C.ink, false, 0),
      footer1: "10",
    });
    addTopRule(slide, C.navy);
    addCardAccent(slide, 41.33, 317.33, 374.67, C.clinical);
    addCardAccent(slide, 452.67, 317.33, 374.67, C.endo);
    addCardAccent(slide, 864.28, 317.33, 374.67, C.variability);
    notes(slide, "The deck is deliberately concise. The detailed Markdown report contains the full feature descriptions, model-specific rankings, and interpretation cautions.", [report, topCsv, familyCsv]);
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(OUT_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    await fs.writeFile(path.join(OUT_DIR, `${stem}.layout.json`), await (await slide.export({ format: "layout" })).text());
  }
  await writeBlob(path.join(OUT_DIR, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  await (await PresentationFile.exportPptx(presentation)).save(OUT_PPTX);
  await fs.writeFile(path.join(OUT_DIR, "source-notes.txt"), [report, topCsv, familyCsv, metricsCsv].join("\n") + "\n", "utf8");
  console.log(OUT_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
