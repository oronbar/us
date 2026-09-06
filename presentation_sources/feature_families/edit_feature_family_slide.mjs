import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const starter = "D:/us/presentation_sources/feature_families/template-starter.pptx";
const output = "D:/us/cardiotoxicity_feature_importance_presentation_with_families.pptx";
const previewDir = "D:/us/presentation_sources/feature_families/final-preview";
const layoutDir = "D:/us/presentation_sources/feature_families/final-layout";

const C = {
  navy: "#17324D",
  ink: "#17202A",
  clinical: "#277DA1",
  endo: "#F8961E",
  variability: "#43AA8B",
};

function paragraph(run, fontSize, color, bold = false, spaceAfter = 0) {
  return {
    runs: [{ run, textStyle: { fontSize: `${fontSize}px`, typeface: "Aptos", color, bold } }],
    spaceAfter,
    paragraphStyle: { lineSpacingPercent: 112000 },
  };
}

async function saveBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function shapeByName(slide, name) {
  const found = slide.shapes.items.find((shape) => shape.name === name);
  if (!found) throw new Error(`Missing inherited shape: ${name}`);
  return found;
}

function footerShape(slide) {
  const found = slide.shapes.items.find((shape) => {
    const p = shape.position || shape.frame;
    return p && p.left > 1180 && p.top > 650 && shape.text;
  });
  if (!found) throw new Error("Missing inherited footer shape");
  return found;
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(starter));

  const newSlide = presentation.slides.getItem(2);
  const title = shapeByName(newSlide, "Title-2-5");
  title.text.set([paragraph("The three feature families capture different information", 39, C.navy, true)]);

  const overview = shapeByName(newSlide, "Content-Placeholder-15-12");
  overview.text.set([
    paragraph("FEATURE FAMILIES", 16, C.endo, true, 160),
    paragraph(
      "Clinical features summarize global function and trajectory. Endo–Mid features compare matched layer curves. Variability features measure disagreement among myocardial segments.",
      23,
      C.ink,
    ),
  ]);

  shapeByName(newSlide, "Content-Placeholder-9-9").text.set([
    paragraph("CLINICAL", 44, C.clinical, true),
  ]);
  shapeByName(newSlide, "Content-Placeholder-9-10").text.set([
    paragraph("ENDO–MID", 44, C.endo, true),
  ]);
  shapeByName(newSlide, "Content-Placeholder-9-11").text.set([
    paragraph("VARIABILITY", 40, C.variability, true),
  ]);

  shapeByName(newSlide, "Content-Placeholder-9-6").text.set([
    paragraph("Current and first-visit GLS/EF\nplus relative changes and slopes", 18, C.ink, false),
  ]);
  shapeByName(newSlide, "Content-Placeholder-11-8").text.set([
    paragraph("Matched-layer amplitude, timing,\ncorrelation, and curve-shape gaps", 18, C.ink, false),
  ]);
  shapeByName(newSlide, "Content-Placeholder-10-7").text.set([
    paragraph("Dispersion across segments in\npeaks, timing, and curve behavior", 18, C.ink, false),
  ]);
  shapeByName(newSlide, "Slide-Number-Placeholder-1-4").text.set("3");

  newSlide.speakerNotes.textFrame.setText(
    "Clinical example: first_endo_gls is the absolute first-visit Endocardial GLS magnitude. Endo–Mid example: d_tm_sd_gap_dct04 compares paired layer curves, summarizes segmental gap variability over time, applies DCT04, and takes current minus previous visit. Variability example: circular dispersion of Endocardial segment time-to-peak.\n\n[Sources]\n- D:/us/cardiotoxicity_feature_importance_results/noncnn_feature_importance_report.md\n- D:/us/cardiotoxicity_next_visit_gpu_results/feature_manifest.csv\n[/Sources]",
  );
  newSlide.speakerNotes.setVisible(true);

  for (let index = 3; index < presentation.slides.items.length; index += 1) {
    footerShape(presentation.slides.getItem(index)).text.set(String(index + 1));
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `final-slide-${String(index + 1).padStart(2, "0")}`;
    await saveBlob(path.join(previewDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    await saveBlob(path.join(layoutDir, `${stem}.layout.json`), await presentation.export({ slide, format: "layout" }));
  }
  await saveBlob(
    "D:/us/presentation_sources/feature_families/final-montage.webp",
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );
  await (await PresentationFile.exportPptx(presentation)).save(output);
  console.log(output);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
