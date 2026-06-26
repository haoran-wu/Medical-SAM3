import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = process.cwd();
const RUNTIME = path.join(
  process.env.HOME,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs",
);
const { Presentation, PresentationFile } = await import(pathToFileURL(RUNTIME).href);

const OUT = path.join(HERE, "output", "medical_sam3_group_meeting_editorial_v2.pptx");
const PREVIEW = path.join(HERE, "preview");
const ASSET = path.join(REPO, "results/visium_hd_exp1/figures/group_meeting");
const W = 1280;
const H = 720;

const C = {
  ink: "#14213D",
  navy: "#1E3A5F",
  blue: "#277DA1",
  teal: "#43AA8B",
  green: "#79A857",
  orange: "#F18701",
  red: "#B23A48",
  plum: "#6A4C93",
  slate: "#5D6D7E",
  paper: "#F7F4ED",
  cream: "#FFFDF6",
  line: "#D7D2C4",
  paleBlue: "#E8F1F6",
  paleGreen: "#EAF4EA",
  paleOrange: "#FFF0DA",
  paleRed: "#FBE7EA",
  white: "#FFFFFF",
};

function slide(prs, eyebrow, title, subtitle = "") {
  const s = prs.slides.add();
  s.background.fill = { type: "solid", color: C.paper };
  rect(s, 0, 0, W, 58, C.ink, "none");
  txt(s, 46, 18, 260, 22, eyebrow, 13, C.white, true, "left");
  txt(s, 46, 86, 820, 62, title, 34, C.ink, true);
  if (subtitle) txt(s, 48, 150, 930, 38, subtitle, 15, C.slate);
  return s;
}

function rect(s, x, y, w, h, fill, line = C.line, r = "roundRect") {
  const sh = s.shapes.add({
    geometry: r,
    position: { left: x, top: y, width: w, height: h },
    fill: fill === "none" ? { type: "none" } : { type: "solid", color: fill },
    line: line === "none" ? { fill: { type: "none" } } : { style: "solid", fill: line, width: 1.2 },
  });
  return sh;
}

function txt(s, x, y, w, h, value, size = 18, color = C.ink, bold = false, align = "left") {
  const sh = rect(s, x, y, w, h, "none", "none", "rect");
  sh.text = value;
  sh.text.fontSize = size;
  sh.text.color = color;
  sh.text.bold = bold;
  sh.text.typeface = "Aptos";
  sh.text.alignment = align;
  sh.text.verticalAlignment = "middle";
  return sh;
}

function pill(s, x, y, w, label, fill, color = C.white) {
  rect(s, x, y, w, 32, fill, "none", "roundRect");
  txt(s, x + 10, y + 5, w - 20, 20, label, 12, color, true, "center");
}

function metricCard(s, x, y, label, value, note, fill = C.white, accent = C.teal) {
  rect(s, x, y, 235, 126, fill, C.line);
  rect(s, x, y, 8, 126, accent, "none", "rect");
  txt(s, x + 22, y + 16, 170, 20, label, 12, C.slate, true);
  txt(s, x + 22, y + 40, 170, 44, value, 30, C.ink, true);
  txt(s, x + 22, y + 86, 180, 24, note, 11, C.slate);
}

function arrow(s, x1, y1, x2, y2, color = C.slate) {
  if (Math.abs(y2 - y1) < Math.abs(x2 - x1)) {
    const left = Math.min(x1, x2);
    const top = y1 - 2;
    rect(s, left, top, Math.abs(x2 - x1) - 10, 4, color, "none", "rect");
    const tri = s.shapes.add({
      geometry: x2 >= x1 ? "triangle" : "triangle",
      position: { left: x2 - 12, top: y2 - 8, width: 16, height: 16, rotation: x2 >= x1 ? 90 : 270 },
      fill: { type: "solid", color },
      line: { fill: { type: "none" } },
    });
    return tri;
  }
  const top = Math.min(y1, y2);
  rect(s, x1 - 2, top, 4, Math.abs(y2 - y1) - 10, color, "none", "rect");
  const tri = s.shapes.add({
    geometry: "triangle",
    position: { left: x2 - 8, top: y2 - 12, width: 16, height: 16, rotation: y2 >= y1 ? 180 : 0 },
    fill: { type: "solid", color },
    line: { fill: { type: "none" } },
  });
  return tri;
}

function image(s, filename, x, y, w, h, fit = "contain") {
  const source = path.join(ASSET, filename);
  const dataUrl = `data:image/png;base64,${fsSync.readFileSync(source).toString("base64")}`;
  s.images.add({
    dataUrl,
    alt: filename,
    position: { left: x, top: y, width: w, height: h },
    fit,
  });
}

function tableLike(s, x, y, cols, rows, widths, rowH = 42) {
  const totalW = widths.reduce((a, b) => a + b, 0);
  rect(s, x, y, totalW, rowH, C.ink, "none", "rect");
  let cx = x;
  cols.forEach((c, i) => {
    txt(s, cx + 8, y + 7, widths[i] - 16, rowH - 12, c, 11, C.white, true);
    cx += widths[i];
  });
  rows.forEach((row, r) => {
    const yy = y + rowH * (r + 1);
    rect(s, x, yy, totalW, rowH, r % 2 ? C.cream : C.white, C.line, "rect");
    let xx = x;
    row.forEach((c, i) => {
      txt(s, xx + 8, yy + 6, widths[i] - 16, rowH - 10, c, 11, i === 0 ? C.ink : C.slate, i === 0);
      xx += widths[i];
    });
  });
}

async function build() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(PREVIEW, { recursive: true });
  const prs = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const s = slide(prs, "Medical-SAM3 / VisiumHD Exp1", "From SAM3 future directions to validated H&E–expression training", "Group meeting deck v2: fewer slides, stronger story, method first, evidence second.");
    pill(s, 48, 223, 180, "Current question", C.orange);
    txt(s, 48, 270, 470, 100, "Can paired H&E morphology and VisiumHD expression be trained into a useful shared representation before we plug it into SAM3?", 27, C.ink, true);
    metricCard(s, 660, 230, "Best expr acc", "0.765", "GigaPath last-1/2", C.white, C.teal);
    metricCard(s, 915, 230, "Best macro-F1", "0.694", "GigaPath last-2", C.white, C.blue);
    metricCard(s, 660, 390, "Best raw R@1", "0.745", "GigaPath CE-only", C.white, C.orange);
    metricCard(s, 915, 390, "Best fused R@1", "0.880", "GigaPath last-2", C.white, C.plum);
    txt(s, 48, 610, 980, 36, "Main message: the model is already useful; the next decision is how to connect this representation to SAM3 without overclaiming the biology.", 17, C.ink, true);
  }

  {
    const s = slide(prs, "1 / Pipeline map", "Where this work intervenes in SAM3", "The current work is not replacing SAM3 yet. It builds the molecularly informed representation that can later guide prompts, retrieval, or mask ranking.");
    const y = 285;
    rect(s, 55, y, 185, 82, "#BFC4C9", C.line); txt(s, 72, y + 20, 150, 34, "H&E image\npatch/tile", 16, C.ink, true, "center");
    rect(s, 305, y, 210, 82, C.navy, "none"); txt(s, 322, y + 20, 175, 34, "Image encoder\nGigaPath / ResNet", 15, C.white, true, "center");
    rect(s, 580, y, 205, 82, C.ink, "none"); txt(s, 602, y + 20, 160, 34, "Prompt / mask\nSAM3 modules", 15, C.white, true, "center");
    rect(s, 855, y, 165, 82, C.blue, "none"); txt(s, 877, y + 22, 120, 30, "Mask\noutput", 16, C.white, true, "center");
    arrow(s, 240, y + 42, 305, y + 42); arrow(s, 515, y + 42, 580, y + 42); arrow(s, 785, y + 42, 855, y + 42);
    rect(s, 340, 170, 205, 70, C.paleGreen, C.teal); txt(s, 360, 186, 165, 34, "Expression MLP\n18K genes", 15, C.ink, true, "center");
    arrow(s, 442, 240, 442, y);
    rect(s, 295, 394, 265, 80, C.paleOrange, C.orange); txt(s, 318, 412, 220, 34, "This week:\ntrain image-expression space", 15, C.ink, true, "center");
    rect(s, 570, 394, 265, 80, C.paleBlue, C.blue); txt(s, 594, 412, 217, 34, "Next:\nuse space for SAM3 ranking", 15, C.ink, true, "center");
    txt(s, 55, 585, 1080, 48, "Why this matters: before asking SAM3 to segment molecular concepts, we need evidence that morphology and expression can agree in a compact embedding space.", 18, C.ink, true);
  }

  {
    const s = slide(prs, "2 / What is trained", "Our training sample is paired morphology + expression + label", "Each example is one tissue location represented three ways. The label is a supervision anchor; retrieval tests whether the two towers actually align.");
    rect(s, 70, 225, 250, 150, C.white, C.line); txt(s, 95, 256, 200, 60, "H&E crop\n64 or 128 px\nresized to 224", 21, C.ink, true, "center");
    rect(s, 405, 225, 250, 150, C.white, C.line); txt(s, 430, 256, 200, 60, "VisiumHD\nexpression vector\n~18K genes", 21, C.ink, true, "center");
    rect(s, 740, 225, 250, 150, C.white, C.line); txt(s, 765, 256, 200, 60, "Tissue label\n3-class or\n8-class", 21, C.ink, true, "center");
    arrow(s, 320, 300, 405, 300, C.teal); arrow(s, 655, 300, 740, 300, C.teal);
    const rows = [
      ["3-class", "coarse target", "tumor / stroma / immune-style grouping", "stable, easier to explain"],
      ["8-class", "fine target", "bronchiola, erythrocytes, vessels, pigment, etc.", "harder, more biological noise"],
    ];
    tableLike(s, 92, 465, ["Target", "Granularity", "Meaning", "Readout"], rows, [120, 155, 455, 270], 42);
  }

  {
    const s = slide(prs, "3 / Model architecture", "Two towers are projected into one shared space", "Classification asks whether each tower learned tissue semantics. Retrieval asks whether image and expression embeddings are geometrically aligned.");
    rect(s, 75, 215, 210, 80, C.paleBlue, C.blue); txt(s, 95, 232, 170, 36, "H&E crop\nimage tower", 18, C.ink, true, "center");
    rect(s, 75, 395, 210, 80, C.paleGreen, C.teal); txt(s, 95, 412, 170, 36, "Expression vector\nMLP tower", 18, C.ink, true, "center");
    rect(s, 390, 215, 220, 80, C.navy, "none"); txt(s, 415, 234, 170, 30, "Image projection\nMLP", 17, C.white, true, "center");
    rect(s, 390, 395, 220, 80, C.green, "none"); txt(s, 415, 414, 170, 30, "Expression projection\nMLP", 17, C.white, true, "center");
    rect(s, 720, 300, 230, 90, C.ink, "none"); txt(s, 742, 320, 185, 35, "Shared embedding\nspace", 19, C.white, true, "center");
    rect(s, 1030, 300, 170, 90, C.orange, "none"); txt(s, 1050, 321, 130, 34, "Shared\nclassifier", 18, C.white, true, "center");
    arrow(s, 285, 255, 390, 255, C.blue); arrow(s, 285, 435, 390, 435, C.teal); arrow(s, 610, 255, 720, 330, C.blue); arrow(s, 610, 435, 720, 360, C.teal); arrow(s, 950, 345, 1030, 345, C.orange);
    txt(s, 75, 570, 1040, 42, "Training knobs we tested: backbone (ResNet/GigaPath), labels (3 vs 8), loss (CE/MSE/InfoNCE), and image encoder fine-tuning (frozen/last-1/last-2).", 17, C.ink, true);
  }

  {
    const s = slide(prs, "4 / Loss design", "The ablation asks what each supervision signal contributes", "This is the part that makes the experiment defensible: we are not assuming three losses are necessary; we test each role.");
    const rows = [
      ["CE_image", "image embedding -> tissue label", "does H&E morphology predict tissue semantics?"],
      ["CE_expression", "expression embedding -> tissue label", "does expression become a supervised molecular anchor?"],
      ["MSE alignment", "paired embeddings close in L2/cosine space", "simple pairwise closeness baseline"],
      ["InfoNCE", "positive pair vs in-batch negatives", "CLIP-style retrieval geometry"],
    ];
    tableLike(s, 70, 205, ["Loss part", "What it optimizes", "Why we need to know"], rows, [190, 390, 540], 52);
    pill(s, 95, 520, 180, "CE-only", C.slate); pill(s, 315, 520, 230, "CE + InfoNCE", C.teal); pill(s, 585, 520, 185, "InfoNCE-only", C.plum); pill(s, 810, 520, 190, "CE + MSE", C.orange);
    txt(s, 92, 592, 1050, 32, "Interpretation rule: CE should help label prediction; InfoNCE should help retrieval/fusion; if InfoNCE-only fails classification, that is a diagnostic, not a failed project.", 16, C.ink, true);
  }

  {
    const s = slide(prs, "5 / Experiment grid", "We deliberately separated four axes instead of chasing one model", "This makes the results easier to defend in group meeting: each run answers a different question.");
    const axes = [
      ["Backbone", "ResNet50\nGigaPath\nMUSK pending", C.paleBlue, C.blue],
      ["Label target", "3-class coarse\n8-class fine", C.paleGreen, C.teal],
      ["Loss", "CE-only\nCE+InfoNCE\nInfoNCE-only\nCE+MSE baseline", C.paleOrange, C.orange],
      ["Training", "Frozen\nwarmup/cosine\nlast-1 block\nlast-2 blocks", C.paleRed, C.red],
    ];
    axes.forEach(([h, b, fill, accent], i) => {
      const x = 78 + i * 295;
      rect(s, x, 230, 245, 255, fill, accent);
      txt(s, x + 22, 252, 195, 28, h, 19, C.ink, true, "center");
      txt(s, x + 26, 310, 190, 110, b, 20, C.ink, true, "center");
    });
    txt(s, 88, 560, 1080, 44, "Result slides are split the same way: first supervised metrics, then retrieval metrics, then training behavior. That prevents one metric from hiding the tradeoff.", 18, C.ink, true);
  }

  {
    const s = slide(prs, "6 / Supervised results", "Partial GigaPath fine-tuning improves tissue recognition", "The best supervised 3-class model fine-tunes the last two image blocks with warmup/cosine, close to the Haiku training strategy.");
    const rows = [
      ["Frozen constant", "0.7568", "0.6799", "0.6912"],
      ["Frozen warmup/cosine", "0.7532", "0.6796", "0.6908"],
      ["Last-1 warmup/cosine", "0.7652", "0.6929", "0.7920"],
      ["Last-2 warmup/cosine", "0.7652", "0.6945", "0.8312"],
      ["8-class last-2", "0.6316", "0.4512", "0.7084"],
    ];
    tableLike(s, 70, 205, ["Run", "Expr acc", "Macro-F1", "Image acc"], rows, [340, 180, 180, 180], 44);
    metricCard(s, 1010, 230, "Decision", "Last-2", "best supervised candidate", C.white, C.teal);
    metricCard(s, 1010, 390, "Caution", "8-class", "still much harder", C.white, C.red);
  }

  {
    const s = slide(prs, "7 / Training behavior", "3-class converges; 8-class exposes label noise and imbalance", "The plots are useful for explaining why the fine-label task should not be judged by loss alone.");
    image(s, "presentation_3class_vs_8class_metrics.png", 70, 190, 560, 315, "contain");
    image(s, "presentation_expression_ce_generalization_gap.png", 670, 190, 520, 315, "contain");
    txt(s, 83, 555, 1030, 50, "Speaker note: 3-class has cleaner morphology-label mapping; 8-class includes rare and ambiguous labels, so CE supervision can become noisy even when the model is learning useful structure.", 17, C.ink, true);
  }

  {
    const s = slide(prs, "8 / Retrieval evidence", "The best model depends on what we optimize for", "Raw nearest-neighbor R@1 favors CE-only, but last-2 fine-tuning gives the strongest fused readout.");
    image(s, "retrieval_overall_recall.png", 70, 170, 610, 360, "contain");
    image(s, "gigapath_3class_per_label_recall1.png", 720, 190, 455, 290, "contain");
    metricCard(s, 730, 500, "Raw R@1", "0.745", "CE-only", C.white, C.orange);
    metricCard(s, 980, 500, "Fused R@1", "0.880", "last-2", C.white, C.plum);
  }

  {
    const s = slide(prs, "9 / What Haiku changes", "Haiku is the right design reference, not a direct copy", "It shows a mature version of shared-space multimodal training: molecular anchor, projection heads, contrastive alignment, partial fine-tuning, and fusion at inference.");
    const rows = [
      ["Haiku", "H&E + mIF + text", "tri-modal InfoNCE", "mIF frozen; H&E/text last blocks"],
      ["Ours now", "H&E + expression", "CE + MSE/InfoNCE ablation", "GigaPath frozen or last-1/2"],
      ["Borrow next", "add text/gene-set anchor", "contrastive + supervised", "freeze molecular anchor; fine-tune image carefully"],
    ];
    tableLike(s, 70, 205, ["System", "Modalities", "Alignment", "Training strategy"], rows, [160, 280, 300, 380], 58);
    txt(s, 96, 565, 1010, 42, "Key distinction for the talk: we are using Haiku to justify the strategy, while our current evidence is from H&E-expression experiments on VisiumHD.", 18, C.ink, true);
  }

  {
    const s = slide(prs, "10 / SAM3 integration plan", "Use the validated embedding space before changing the whole model", "The safest next step is proposal/mask ranking: keep SAM3 intact, use expression-aware retrieval to select or score candidate masks.");
    rect(s, 70, 230, 205, 82, C.paleBlue, C.blue); txt(s, 90, 250, 165, 30, "SAM3 candidate\nmasks", 18, C.ink, true, "center");
    rect(s, 370, 210, 250, 122, C.paleGreen, C.teal); txt(s, 395, 235, 200, 55, "Image-expression\nshared embedding", 20, C.ink, true, "center");
    rect(s, 715, 230, 210, 82, C.paleOrange, C.orange); txt(s, 738, 250, 165, 30, "Mask ranking /\nregion retrieval", 18, C.ink, true, "center");
    rect(s, 1010, 230, 170, 82, C.ink, "none"); txt(s, 1030, 250, 130, 30, "Selected\noutput", 18, C.white, true, "center");
    arrow(s, 275, 272, 370, 272, C.teal); arrow(s, 620, 272, 715, 272, C.orange); arrow(s, 925, 272, 1010, 272, C.ink);
    const rows = [
      ["Immediate", "use last-2/CE-only checkpoints to rank SAM3 proposals", "low risk"],
      ["Next", "test gene-set/text prompts as expression summaries", "Haiku-inspired"],
      ["Later", "replace/fine-tune SAM3 image encoder or decoder", "higher risk"],
    ];
    tableLike(s, 95, 430, ["Stage", "Action", "Risk"], rows, [170, 690, 170], 48);
  }

  {
    const s = slide(prs, "11 / Current status", "Validation set is complete and archived", "Everything needed for this group meeting is saved locally and on the project/HPC side.");
    metricCard(s, 75, 205, "HPC jobs", "Done", "squeue empty", C.white, C.teal);
    metricCard(s, 330, 205, "Training rows", "33", "summary snapshot", C.white, C.blue);
    metricCard(s, 585, 205, "Retrieval rows", "28", "summary snapshot", C.white, C.orange);
    metricCard(s, 840, 205, "PPT figures", "8", "group_meeting/", C.white, C.plum);
    txt(s, 90, 405, 980, 92, "Saved artifacts:\n- results/visium_hd_exp1/summaries/final_group_meeting_results_20260507.md/json\n- results/visium_hd_exp1/figures/group_meeting/*.png\n- this new v2 editorial PPT in Presentation/Medical_SAM3_GroupMeeting_20260507_v2_editorial/", 18, C.ink, false);
    txt(s, 90, 590, 940, 34, "MUSK remains pending because Hugging Face gated access has not been approved yet.", 17, C.red, true);
  }

  {
    const s = slide(prs, "12 / Talk track", "One clean way to present this tomorrow", "This keeps the audience oriented and prevents the details from turning into a wall of experiment names.");
    const rows = [
      ["1", "Problem", "SAM3 needs molecular context; first prove H&E-expression alignment."],
      ["2", "Method", "Two-tower model, projection heads, CE and InfoNCE/MSE losses."],
      ["3", "Experiment", "Backbone x label granularity x loss x fine-tuning."],
      ["4", "Result", "3-class stable; last-2 best supervised/fused; 8-class harder."],
      ["5", "Literature link", "Haiku supports contrastive shared-space + partial fine-tuning."],
      ["6", "Next step", "Use embedding for SAM3 proposal/mask ranking before deeper model surgery."],
    ];
    tableLike(s, 90, 185, ["#", "Section", "One sentence to say"], rows, [70, 210, 760], 55);
  }

  const pptx = await PresentationFile.exportPptx(prs);
  await pptx.save(OUT);

  const previews = [];
  for (let i = 0; i < prs.slides.count; i += 1) {
    const p = path.join(PREVIEW, `slide-${String(i + 1).padStart(2, "0")}.png`);
    const blob = await prs.export({ slide: prs.slides.getItem(i), format: "png", scale: 1 });
    await fs.writeFile(p, Buffer.from(await blob.arrayBuffer()));
    previews.push(p);
  }
  const sheetScript = path.join(
    process.env.HOME,
    ".codex/plugins/cache/openai-primary-runtime/presentations/26.505.10851/skills/presentations/scripts/make_contact_sheet.py",
  );
  const contact = path.join(HERE, "qa", "contact_sheet.png");
  const result = spawnSync("/Users/haoranwu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3", [sheetScript, "--output", contact, ...previews], { encoding: "utf8" });
  if (result.status !== 0) {
    console.error(result.stdout);
    console.error(result.stderr);
  }
  console.log(JSON.stringify({ output: OUT, slideCount: prs.slides.count, contact }, null, 2));
}

await build();
