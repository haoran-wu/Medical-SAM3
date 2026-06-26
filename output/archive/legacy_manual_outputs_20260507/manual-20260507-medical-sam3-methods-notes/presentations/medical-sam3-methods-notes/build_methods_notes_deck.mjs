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

const OUT = path.join(HERE, "output", "Medical_SAM3_Training_Ablation_GroupMeeting_20260507_structured_v6.pptx");
const PREVIEW = path.join(HERE, "preview");
const ASSET = path.join(REPO, "results/visium_hd_exp1/figures/group_meeting");
const W = 1280;
const H = 720;

const C = {
  ink: "#13213A",
  navy: "#203B63",
  blue: "#2F6F9F",
  cyan: "#DDEFF5",
  teal: "#3F8F75",
  mint: "#E3F3EA",
  olive: "#6B8F43",
  orange: "#D97706",
  amber: "#FFF2D8",
  red: "#AA3A45",
  rose: "#F8E1E4",
  plum: "#665191",
  lavender: "#E9E3F6",
  slate: "#5F6F7E",
  line: "#D8D2C3",
  paper: "#F7F3EA",
  cream: "#FFFDF6",
  white: "#FFFFFF",
};

function rect(s, x, y, w, h, fill, line = C.line, r = "roundRect") {
  return s.shapes.add({
    geometry: r,
    position: { left: x, top: y, width: w, height: h },
    fill: fill === "none" ? { type: "none" } : { type: "solid", color: fill },
    line: line === "none" ? { fill: { type: "none" } } : { style: "solid", fill: line, width: 1.1 },
  });
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

function titleSlide(prs, eyebrow, title, subtitle = "") {
  const s = prs.slides.add();
  s.background.fill = { type: "solid", color: C.paper };
  rect(s, 0, 0, W, 60, C.ink, "none", "rect");
  txt(s, 48, 18, 390, 22, eyebrow, 13, C.white, true);
  txt(s, 52, 95, 1085, 78, title, 36, C.ink, true);
  if (subtitle) txt(s, 54, 171, 980, 38, subtitle, 16, C.slate);
  return s;
}

function contentSlide(prs, section, title, subtitle = "", note = "") {
  const s = prs.slides.add();
  s.background.fill = { type: "solid", color: C.paper };
  rect(s, 0, 0, W, 56, C.ink, "none", "rect");
  txt(s, 46, 17, 260, 22, section, 12, C.white, true);
  txt(s, 46, 78, 920, 55, title, 30, C.ink, true);
  if (subtitle) txt(s, 48, 135, 1080, 35, subtitle, 14, C.slate);
  if (note) s.speakerNotes.setText(note);
  return s;
}

function pill(s, x, y, w, label, fill, color = C.white) {
  rect(s, x, y, w, 30, fill, "none");
  txt(s, x + 10, y + 5, w - 20, 18, label, 11, color, true, "center");
}

function metric(s, x, y, label, value, note, fill = C.white, accent = C.teal) {
  rect(s, x, y, 220, 118, fill, C.line);
  rect(s, x, y, 8, 118, accent, "none", "rect");
  txt(s, x + 23, y + 14, 170, 19, label, 11, C.slate, true);
  txt(s, x + 23, y + 38, 170, 42, value, 29, C.ink, true);
  txt(s, x + 23, y + 84, 170, 20, note, 10.5, C.slate);
}

function arrow(s, x1, y1, x2, y2, color = C.slate) {
  if (Math.abs(x2 - x1) >= Math.abs(y2 - y1)) {
    const left = Math.min(x1, x2);
    rect(s, left, y1 - 2, Math.max(1, Math.abs(x2 - x1) - 13), 4, color, "none", "rect");
    s.shapes.add({
      geometry: "triangle",
      position: { left: x2 - 13, top: y2 - 8, width: 16, height: 16, rotation: x2 >= x1 ? 90 : 270 },
      fill: { type: "solid", color },
      line: { fill: { type: "none" } },
    });
    return;
  }
  const top = Math.min(y1, y2);
  rect(s, x1 - 2, top, 4, Math.max(1, Math.abs(y2 - y1) - 13), color, "none", "rect");
  s.shapes.add({
    geometry: "triangle",
    position: { left: x2 - 8, top: y2 - 13, width: 16, height: 16, rotation: y2 >= y1 ? 180 : 0 },
    fill: { type: "solid", color },
    line: { fill: { type: "none" } },
  });
}

function embedImage(s, filename, x, y, w, h, fit = "contain") {
  const source = path.join(ASSET, filename);
  const dataUrl = `data:image/png;base64,${fsSync.readFileSync(source).toString("base64")}`;
  s.images.add({ dataUrl, alt: filename, position: { left: x, top: y, width: w, height: h }, fit });
}

function table(s, x, y, headers, rows, widths, rowH = 42, headerFill = C.ink) {
  const total = widths.reduce((a, b) => a + b, 0);
  rect(s, x, y, total, rowH, headerFill, "none", "rect");
  let xx = x;
  headers.forEach((h, i) => {
    txt(s, xx + 8, y + 5, widths[i] - 16, rowH - 10, h, 10.5, C.white, true);
    xx += widths[i];
  });
  rows.forEach((row, r) => {
    const yy = y + rowH * (r + 1);
    rect(s, x, yy, total, rowH, r % 2 ? C.cream : C.white, C.line, "rect");
    xx = x;
    row.forEach((cell, i) => {
      txt(s, xx + 8, yy + 5, widths[i] - 16, rowH - 10, cell, 10.3, i === 0 ? C.ink : C.slate, i === 0);
      xx += widths[i];
    });
  });
}

function formulaBox(s, x, y, label, formula, note, fill, accent) {
  rect(s, x, y, 355, 118, fill, accent);
  txt(s, x + 18, y + 13, 315, 20, label, 13, C.ink, true);
  txt(s, x + 18, y + 42, 315, 28, formula, 18, C.ink, true, "center");
  txt(s, x + 18, y + 80, 315, 20, note, 11, C.slate);
}

function sectionCard(s, x, y, title, body, fill, accent) {
  rect(s, x, y, 340, 142, fill, accent);
  txt(s, x + 18, y + 16, 300, 24, title, 17, C.ink, true);
  txt(s, x + 18, y + 52, 295, 62, body, 13, C.ink, true);
}

const notes = {
  title:
    "Start by framing the update. The point of this section is not to claim a finished SAM3 segmentation model. The point is to explain the representation-learning step: paired H&E morphology and VisiumHD gene expression are trained into a shared space. Once that is clear, the loss ablations and retrieval results become much easier to interpret.",
  map:
    "Use this slide to connect the work back to the earlier SAM3 future-directions slide. The current experiment sits before full segmentation integration. It trains an image-expression representation that can later be used for prompt retrieval, exemplar retrieval, or mask ranking. So the boundary is important: segmentation quality is not the main readout yet; representation quality is.",
  testOverview:
    "This slide is the roadmap for the entire results section. I would say: before looking at any plots, here is the complete set of experiments. There are five categories: backbone, label granularity, loss ablation, training strategy, and retrieval or fusion evaluation. For each category, the slide shows the exact data, model choices, loss, metric, and the main conclusion. The rest of the talk follows this order.",
  backboneTest:
    "This is the first result category: backbone testing. The dataset is the same paired H&E and VisiumHD Exp1 dataset. The question is whether a general ResNet50 baseline is enough, or whether a pathology foundation encoder like GigaPath is better. The main conclusion is that GigaPath crop128 is the stronger and more stable backbone, so it becomes the main line for later experiments.",
  labelTest:
    "This is the label-granularity test. The data and model are held mostly constant, but the label target changes from coarse 3-class to fine 8-class. The purpose is to ask whether the model can handle fine-grained histology. The result is that 3-class is stable and interpretable, while 8-class is harder and has weaker macro-F1, probably because of class imbalance and ambiguous morphology.",
  lossTest:
    "This is the loss-ablation section. The key point is that loss design was tested explicitly. The CE-only model tests whether shared tissue labels are enough. CE plus MSE tests simple pairwise alignment. CE plus InfoNCE tests CLIP-style contrastive alignment. InfoNCE-only tests whether alignment alone can work without label supervision. This lets the presentation separate classification learning from cross-modal retrieval learning.",
  strategyTest:
    "This is the training-strategy test. The question is whether the pretrained image encoder should stay frozen or be partially adapted to this dataset. The result is that last-one-block and last-two-block fine-tuning improve the image branch a lot, and last-two-block gives the best supervised and fused-readout candidate.",
  retrievalTest:
    "This is the retrieval and fusion evaluation. Classification alone does not prove that the two modalities are aligned. Retrieval asks whether an expression embedding can find the matching H&E region. The key result is that CE-only is strong for raw recall at one, while last-two-block fine-tuning gives the best fused recall at one. That is why the next SAM3 integration should use the embedding for retrieval or mask ranking.",
  data:
    "This slide defines one training example. Each example corresponds to one spatial tissue location. It has an H&E crop, a VisiumHD expression vector from the same location, and a tissue label. The 3-class task is coarse and more stable. The 8-class task is more fine-grained, and therefore more sensitive to class imbalance and ambiguous morphology.",
  input:
    "Explain the data construction step. The H&E image is cropped around the VisiumHD coordinate and resized for the image encoder. The expression vector is log transformed and scaled. The tissue label is used as a supervision anchor. It does not mean the final goal is only classification; classification is used to check whether the learned embedding contains tissue-level semantic information.",
  architecture:
    "This is the core method diagram. There are two towers: an image tower for the H&E crop and an expression tower for the gene-expression vector. Each tower feeds a projection head, and both projections land in the same embedding space. The classifier checks whether each embedding predicts the tissue label. Retrieval checks whether expression embeddings can retrieve matching image embeddings.",
  projection:
    "Emphasize that the encoder output is not compared directly. It first goes through a projection MLP and is normalized into the shared embedding space. In the base ablations, the image backbone is mostly frozen, while the projection heads, expression encoder, and classifier are trained. The training-strategy experiments then test whether unfreezing the last one or two image blocks helps.",
  loss:
    "This slide explains the loss terms. Cross entropy is the supervised tissue-label loss. It is applied to the image branch and the expression branch. MSE alignment is the simple baseline that pulls paired image and expression embeddings closer together. InfoNCE is the CLIP-style contrastive loss: the matched image-expression pair is the positive pair, and other examples in the batch act as negatives. This directly optimizes retrieval geometry.",
  training:
    "This is the practical training setup. AdamW is the optimizer, but the learning rate only changes when a scheduler is used. The main learning rate is 1e-4 for the new heads and expression MLP. When image blocks are unfrozen, the backbone uses a smaller learning rate of 1e-5. Dropout is 0.3. Early stopping uses patience 2 and monitors validation expression macro-F1, because macro-F1 is safer than accuracy under class imbalance.",
  recipe:
    "Use this as the main methods slide if time is short. It concentrates the entire training recipe onto one page: what data is used, what is trainable, what the optimizer is, what the loss function is, and exactly which loss ablations were tested. The key message is that the loss design was treated as an experiment, not as an assumption.",
  matrix:
    "This slide explains why the experiment grid is structured. The runs are not random. Each axis answers one question: whether the image backbone matters, whether coarse versus fine labels change stability, whether InfoNCE improves retrieval, and whether partial image fine-tuning helps compared with a frozen backbone.",
  coverage:
    "This slide fixes the concern that only one training curve was shown. The project actually contains thirty-eight history files. They cover GigaPath and ResNet, 3-class and 8-class, loss ablations, and training-strategy runs. The plots in the next few slides are generated from those histories, not from a single hand-picked example.",
  allLoss:
    "This slide shows validation total loss for the major loss-ablation families. Be careful with interpretation: CE-only, CE plus InfoNCE, and InfoNCE-only do not have identical loss scales. So this plot should mainly be used for stability and convergence behavior, not for comparing the absolute value of loss across different objective families.",
  components:
    "This slide decomposes the loss into image cross entropy, expression cross entropy, and alignment. This answers what is actually being trained. In the joint objective, the image branch, expression branch, and alignment term can all receive learning signal. In CE-only, the alignment term is zero. In InfoNCE-only, the classification terms are recorded for diagnostics but are not the main optimization objective.",
  strategyCurves:
    "This slide shows why the fine-tuning result is not just a single final number. Last-one-block and last-two-block fine-tuning clearly improve the image branch, while expression macro-F1 does not collapse. That supports using last-two-block fine-tuning as the strongest candidate for the next SAM3 integration experiment.",
  supervised:
    "This slide summarizes supervised classification results. The last-two-block fine-tuned GigaPath model is the strongest supervised 3-class setting, especially for image accuracy and expression macro-F1. The 8-class model is still much harder, which suggests that fine labels are noisier, more imbalanced, or less visually separable.",
  behavior:
    "This slide explains the difference between the 3-class and 8-class behavior. The 3-class target is coarse, so the morphology-expression relationship is cleaner. The 8-class target contains rarer and more ambiguous categories, so the model can improve training loss while validation expression CE remains high. That is why macro-F1 and retrieval are important alongside loss.",
  retrieval:
    "This slide connects training to retrieval. Raw recall at one is strongest for CE-only, meaning supervised tissue labels already create a useful neighborhood structure. But the best fused recall at one comes from last-two-block fine-tuning. InfoNCE is especially useful for broader top-k retrieval, which matches the contrastive-learning intuition.",
  haiku:
    "This slide explains what is borrowed from Haiku at the design level. The key idea is not early fusion. It is shared-space alignment through projection heads and contrastive learning, with controlled fine-tuning of pretrained encoders. The current project adapts that idea from H&E, mIF, and text to H&E and gene expression, with SAM3 integration as the next downstream use case.",
  conclusion:
    "End with three points. First, the H&E-expression shared representation is already useful. Second, the 3-class setting is currently the most stable main line, while 8-class remains a difficult stress test. Third, the next step should not be replacing SAM3. It should be using the learned embedding to retrieve examples or rank SAM3 masks.",
};

async function build() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(PREVIEW, { recursive: true });
  const prs = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const s = titleSlide(
      prs,
      "Medical-SAM3 / VisiumHD Exp1 / May 2026",
      "H&E–Expression Shared-Space Training",
      "Methods-first group meeting update: what was trained, how it was trained, and what the current results mean.",
    );
    s.speakerNotes.setText(notes.title);
    pill(s, 56, 255, 172, "Current task", C.orange);
    txt(s, 56, 300, 520, 92, "Train paired H&E morphology and VisiumHD expression into one shared representation space.", 28, C.ink, true);
    metric(s, 682, 230, "Best 3-class expr acc", "0.765", "GigaPath last-1/2", C.white, C.teal);
    metric(s, 930, 230, "Best 3-class macro-F1", "0.694", "GigaPath last-2", C.white, C.blue);
    metric(s, 682, 390, "Best raw retrieval R@1", "0.745", "GigaPath CE-only", C.white, C.orange);
    metric(s, 930, 390, "Best fused retrieval R@1", "0.880", "GigaPath last-2", C.white, C.plum);
    txt(s, 58, 610, 1030, 36, "Take-home: the representation is promising; the next step is controlled SAM3 integration, not overclaiming a final segmentation model.", 17, C.ink, true);
  }

  {
    const s = contentSlide(prs, "1 / What was tested", "Five experiment categories, each answering a different question", "The rest of the deck follows this order: setup, result, and conclusion for each category.", notes.testOverview);
    table(
      s,
      42,
      178,
      ["Test category", "Dataset / target", "Backbone", "Loss / setting", "Main readout"],
      [
        ["Backbone baseline", "VisiumHD Exp1, paired H&E-expression", "ResNet50 vs GigaPath", "same shared-space recipe", "which image encoder is strongest"],
        ["Label granularity", "3-class vs 8-class tissue labels", "mainly GigaPath crop128", "same CE/alignment family", "whether fine histology is stable"],
        ["Loss ablation", "3-class and 8-class", "GigaPath + ResNet50", "CE+MSE, CE-only, CE+InfoNCE, InfoNCE-only", "which loss helps classification vs retrieval"],
        ["Training strategy", "GigaPath crop128 3-class plus 8-class check", "GigaPath", "frozen, warmup/cosine, last-1, last-2", "whether partial fine-tuning helps"],
        ["Retrieval / fusion", "held-out expression-to-image retrieval", "trained checkpoints", "embedding similarity + fused score", "R@1/R@5/R@10 and fused R@1"],
      ],
      [190, 250, 190, 300, 270],
      45,
    );
    txt(s, 70, 620, 1060, 30, "Talk structure: first define the training task, then walk through each experiment category separately so the result plots have context.", 15, C.ink, true);
  }

  {
    const s = contentSlide(prs, "2 / SAM3 context", "Where this training intervenes in the SAM3 pipeline", "The current work trains a molecularly informed representation that can later guide prompt retrieval or mask ranking.", notes.map);
    const y = 295;
    rect(s, 58, y, 175, 76, "#C8CDD1", C.line);
    txt(s, 75, y + 18, 138, 34, "H&E image\npatch/tile", 15.5, C.ink, true, "center");
    rect(s, 300, y, 205, 76, C.navy, "none");
    txt(s, 316, y + 18, 172, 34, "Image encoder\nViT / ResNet", 15.5, C.white, true, "center");
    rect(s, 578, y, 196, 76, C.ink, "none");
    txt(s, 595, y + 18, 160, 34, "Prompt + mask\nSAM3 modules", 15.5, C.white, true, "center");
    rect(s, 847, y, 155, 76, C.blue, "none");
    txt(s, 865, y + 18, 118, 34, "Mask\noutput", 16, C.white, true, "center");
    arrow(s, 233, y + 38, 300, y + 38);
    arrow(s, 505, y + 38, 578, y + 38);
    arrow(s, 774, y + 38, 847, y + 38);
    rect(s, 336, 173, 214, 70, C.mint, C.teal);
    txt(s, 357, 188, 172, 34, "Expression MLP\n~18K genes", 15, C.ink, true, "center");
    arrow(s, 443, 243, 443, y);
    rect(s, 294, 405, 280, 84, C.amber, C.orange);
    txt(s, 315, 421, 238, 40, "This training:\nimage-expression shared space", 15.5, C.ink, true, "center");
    rect(s, 604, 405, 280, 84, C.cyan, C.blue);
    txt(s, 627, 421, 234, 40, "Later SAM3 use:\nretrieve prompts / rank masks", 15.5, C.ink, true, "center");
    rect(s, 934, 190, 235, 92, C.lavender, C.plum);
    txt(s, 954, 207, 195, 46, "Reference DB\nannotated patches / embeddings", 14, C.ink, true, "center");
    arrow(s, 934, 236, 774, y + 12, C.plum);
    txt(s, 62, 585, 1070, 48, "Current boundary: segmentation quality is not the main result yet. The main result is whether morphology and expression form a usable joint space.", 18, C.ink, true);
  }

  {
    const s = contentSlide(prs, "3 / Training sample", "Each sample has paired morphology, expression, and a supervision label", "The pairing is spatial: H&E crop and expression vector correspond to the same tissue location.", notes.data);
    rect(s, 70, 225, 250, 150, C.white, C.line);
    txt(s, 96, 252, 198, 76, "H&E crop\n64 or 128 px crop\nresized for encoder", 20, C.ink, true, "center");
    rect(s, 405, 225, 250, 150, C.white, C.line);
    txt(s, 430, 252, 200, 76, "VisiumHD expression\nlog1p + scaled\n~18K genes", 20, C.ink, true, "center");
    rect(s, 740, 225, 250, 150, C.white, C.line);
    txt(s, 765, 252, 200, 76, "Tissue label\n3-class coarse\nor 8-class fine", 20, C.ink, true, "center");
    arrow(s, 320, 300, 405, 300, C.teal);
    arrow(s, 655, 300, 740, 300, C.teal);
    table(
      s,
      86,
      470,
      ["Target", "Granularity", "What it tests", "Current interpretation"],
      [
        ["3-class", "coarse", "robust morphology-expression signal", "stable main baseline"],
        ["8-class", "fine", "whether finer histology is separable", "harder; more imbalance/noise"],
      ],
      [120, 150, 390, 410],
      44,
    );
  }

  {
    const s = contentSlide(prs, "4 / Input construction", "How raw VisiumHD data becomes a training batch", "The model never sees a whole slide at once in these experiments; it sees paired local crops and expression vectors.", notes.input);
    const boxes = [
      ["Spatial coordinate", "spot / bin location\nfrom patch dataset", C.white, C.slate],
      ["H&E crop", "crop around coordinate\n64 or 128 px", C.cyan, C.blue],
      ["Image tensor", "resize + normalize\n224 input for GigaPath", C.cyan, C.blue],
      ["Expression vector", "expr_log1p.npz\nscaler.npz", C.mint, C.teal],
      ["Label", "3-class or 8-class\nbalanced sampling", C.amber, C.orange],
    ];
    boxes.forEach(([h, b, fill, accent], i) => {
      const x = 62 + i * 235;
      rect(s, x, 240, 190, 150, fill, accent);
      txt(s, x + 16, 260, 158, 22, h, 15, C.ink, true, "center");
      txt(s, x + 17, 305, 156, 52, b, 14, C.slate, true, "center");
      if (i < boxes.length - 1) arrow(s, x + 190, 315, x + 235, 315, C.slate);
    });
    rect(s, 90, 475, 1015, 92, C.white, C.line);
    txt(s, 114, 492, 965, 48, "Stored source objects: tissue_hires_image.png, items.csv, expr_log1p.npz, scaler.npz, and label mappings under the VisiumHD Exp1 project directory.", 18, C.ink, true);
    txt(s, 116, 545, 960, 18, "Important wording for the talk: the label supervises representation learning; it is not the final biological endpoint.", 12.5, C.slate, true);
  }

  {
    const s = contentSlide(prs, "5 / Architecture", "Two towers produce aligned embeddings in one shared space", "Classification evaluates semantic content; retrieval evaluates cross-modal alignment.", notes.architecture);
    rect(s, 76, 220, 205, 76, C.cyan, C.blue);
    txt(s, 96, 238, 165, 34, "H&E crop\nimage tower", 16.5, C.ink, true, "center");
    rect(s, 76, 402, 205, 76, C.mint, C.teal);
    txt(s, 96, 420, 165, 34, "Expression vector\nMLP tower", 16.5, C.ink, true, "center");
    rect(s, 372, 220, 220, 76, C.navy, "none");
    txt(s, 392, 238, 180, 34, "Image projection\nMLP head", 16, C.white, true, "center");
    rect(s, 372, 402, 220, 76, C.teal, "none");
    txt(s, 392, 420, 180, 34, "Expression projection\nMLP head", 16, C.white, true, "center");
    rect(s, 703, 303, 228, 92, C.ink, "none");
    txt(s, 725, 323, 184, 38, "Shared embedding\nspace", 19, C.white, true, "center");
    rect(s, 1032, 303, 166, 92, C.orange, "none");
    txt(s, 1050, 323, 130, 38, "Shared\nclassifier", 18, C.white, true, "center");
    arrow(s, 281, 258, 372, 258, C.blue);
    arrow(s, 281, 440, 372, 440, C.teal);
    arrow(s, 592, 258, 703, 330, C.blue);
    arrow(s, 592, 440, 703, 368, C.teal);
    arrow(s, 931, 349, 1032, 349, C.orange);
    txt(s, 80, 565, 1045, 45, "Backbones tested in this report: ResNet50 baseline and GigaPath pathology tile encoder. Fine-tuning tests unfreeze only the last image blocks, not the entire encoder.", 17, C.ink, true);
  }

  {
    const s = contentSlide(prs, "6 / Representation details", "Projection heads define the space where losses are applied", "Encoder features are first mapped into a compact shared embedding, then normalized before alignment/retrieval.", notes.projection);
    formulaBox(s, 78, 220, "Image branch", "H&E -> encoder -> MLP -> z_img", "z_img enters CE and alignment losses", C.cyan, C.blue);
    formulaBox(s, 462, 220, "Expression branch", "genes -> MLP -> MLP -> z_expr", "z_expr is the molecular anchor", C.mint, C.teal);
    formulaBox(s, 846, 220, "Shared classifier", "z -> tissue logits", "same classifier reads both branches", C.amber, C.orange);
    table(
      s,
      105,
      430,
      ["Component", "Base ablation", "Training-strategy ablation"],
      [
        ["Image backbone", "frozen", "frozen / last-1 block / last-2 blocks"],
        ["Projection heads", "trainable", "trainable"],
        ["Expression encoder", "trainable", "trainable"],
        ["Classifier", "trainable", "trainable"],
      ],
      [240, 365, 435],
      42,
    );
  }

  {
    const s = contentSlide(prs, "7 / Losses", "Each loss answers a different scientific question", "The ablation tests whether label supervision, pairwise alignment, or contrastive alignment is responsible for useful retrieval.", notes.loss);
    table(
      s,
      68,
      190,
      ["Loss term", "Definition", "Question it answers"],
      [
        ["CE_image", "image embedding -> tissue label", "does H&E morphology contain the label signal?"],
        ["CE_expression", "expression embedding -> tissue label", "does expression form a supervised molecular anchor?"],
        ["MSE_align", "same-location z_img and z_expr are pulled close", "is simple pairwise closeness sufficient?"],
        ["InfoNCE", "positive pair vs in-batch negatives, symmetric", "does CLIP-style contrast improve retrieval?"],
      ],
      [185, 445, 500],
      50,
    );
    pill(s, 95, 530, 180, "Baseline: CE + 0.5 MSE", C.orange);
    pill(s, 305, 530, 140, "CE-only", C.slate);
    pill(s, 475, 530, 255, "CE + lambda InfoNCE", C.teal);
    pill(s, 760, 530, 165, "InfoNCE-only", C.plum);
    txt(s, 95, 592, 990, 28, "Lambda sweep for InfoNCE: 0.05, 0.10, 0.20, 0.50. Temperature follows CLIP-style contrastive training: tau = 0.07.", 15, C.ink, true);
  }

  {
    const s = contentSlide(prs, "8 / Training strategy", "Hyperparameters and regularization were kept explicit", "This is the slide to use if someone asks exactly how the models were trained.", notes.training);
    table(
      s,
      72,
      188,
      ["Setting", "Value", "Reason"],
      [
        ["Optimizer", "AdamW, weight decay 1e-4", "stable optimizer with decoupled L2 regularization"],
        ["Main learning rate", "1e-4 for projection / expression / classifier", "train new heads and expression MLP"],
        ["Backbone learning rate", "1e-5 when image blocks are unfrozen", "avoid destroying pretrained features"],
        ["Scheduler", "constant or warmup + cosine", "test whether gradual LR improves fine-tuning"],
        ["Dropout", "0.3", "reduce overfitting in projection/MLP layers"],
        ["Epochs", "8 for ablation runs", "fast screen across many conditions"],
        ["Early stopping", "patience = 2", "stop when validation stops improving"],
        ["Monitor", "validation expression macro-F1", "robust to class imbalance"],
        ["Imbalance handling", "balanced sampler + sqrt inverse class weights", "avoid majority classes dominating CE"],
      ],
      [205, 395, 480],
      39,
    );
  }

  {
    const s = contentSlide(prs, "9 / One-page recipe", "Training parameters and loss functions in one place", "This is the compact methods slide: data, trainable parts, optimizer, loss, and ablations.", notes.recipe);
    table(
      s,
      60,
      185,
      ["Block", "Exact setting"],
      [
        ["Data", "paired H&E crop + VisiumHD expression vector + tissue label; crop128 main; 3-class and 8-class targets"],
        ["Encoders", "image: GigaPath or ResNet50; expression: MLP over ~18K genes; projection heads map both branches into shared embedding space"],
        ["Trainable parts", "base ablation freezes image backbone; trains image projection, expression MLP, expression projection, shared classifier"],
        ["Fine-tuning test", "GigaPath frozen constant, frozen warmup/cosine, last-1-block warmup/cosine, last-2-block warmup/cosine"],
        ["Optimizer", "AdamW; weight decay 1e-4; projection/expression/classifier LR 1e-4; unfrozen image-backbone LR 1e-5"],
        ["Regularization", "dropout 0.3; balanced sampler; sqrt inverse class weights for class imbalance"],
        ["Stopping", "8 epochs for ablation screen; early stopping patience 2; monitor validation expression macro-F1"],
      ],
      [190, 930],
      37,
    );
    rect(s, 75, 525, 530, 92, C.amber, C.orange);
    txt(s, 96, 541, 480, 24, "Total loss families tested", 17, C.ink, true);
    txt(s, 96, 575, 470, 28, "CE_img + CE_expr + 0.5*MSE  |  CE_img + CE_expr  |  CE_img + CE_expr + lambda*InfoNCE  |  InfoNCE-only", 12.5, C.ink, true);
    rect(s, 640, 525, 500, 92, C.mint, C.teal);
    txt(s, 662, 541, 450, 24, "InfoNCE details", 17, C.ink, true);
    txt(s, 662, 575, 440, 28, "positive = same-location H&E-expression pair; negatives = other batch examples; lambda = 0.05, 0.10, 0.20, 0.50; tau = 0.07", 12.5, C.ink, true);
  }

  {
    const s = contentSlide(prs, "10 / Experiment matrix", "The grid separates four choices instead of mixing them together", "Each axis has a purpose, so the presentation can explain why a run exists.", notes.matrix);
    const axes = [
      ["Backbone", "ResNet50\nGigaPath", C.cyan, C.blue],
      ["Label granularity", "3-class coarse\n8-class fine", C.mint, C.teal],
      ["Loss family", "CE + MSE\nCE-only\nCE + InfoNCE\nInfoNCE-only", C.amber, C.orange],
      ["Fine-tuning", "frozen\nlast-1 block\nlast-2 blocks\nwarmup/cosine", C.rose, C.red],
    ];
    axes.forEach(([h, b, fill, accent], i) => {
      const x = 70 + i * 295;
      rect(s, x, 230, 245, 255, fill, accent);
      txt(s, x + 22, 254, 200, 26, h, 18, C.ink, true, "center");
      txt(s, x + 28, 315, 188, 94, b, 20, C.ink, true, "center");
    });
    txt(s, 92, 565, 1060, 42, "Reading rule: classification metrics identify semantic supervision; retrieval metrics identify cross-modal alignment; both are needed before connecting the representation to SAM3.", 17, C.ink, true);
  }

  {
    const s = contentSlide(prs, "11 / Test 1: Backbone", "Backbone test: ResNet50 baseline versus GigaPath", "Question: does pathology-pretrained image encoding improve the shared H&E-expression space?", notes.backboneTest);
    sectionCard(s, 72, 205, "Setup", "Dataset: VisiumHD Exp1 paired H&E-expression samples\nTargets: mainly 3-class / 8-class\nBackbones: ResNet50 vs GigaPath", C.cyan, C.blue);
    sectionCard(s, 472, 205, "Loss", "Same shared-space recipe:\nCE_img + CE_expr, with alignment variants tested later", C.amber, C.orange);
    sectionCard(s, 872, 205, "Result", "GigaPath crop128 3-class became the strongest and most stable main line.", C.mint, C.teal);
    table(
      s,
      110,
      430,
      ["Backbone test", "Best use in this project", "Conclusion"],
      [
        ["ResNet50", "baseline / sanity check", "useful comparison, weaker than GigaPath for main line"],
        ["GigaPath", "main pathology image encoder", "better stability and stronger downstream retrieval/fusion candidate"],
      ],
      [230, 390, 520],
      48,
    );
  }

  {
    const s = contentSlide(prs, "12 / Test 2: Label granularity", "3-class is stable; 8-class is a harder fine-label target", "Same paired data, but the supervision target changes from coarse to fine tissue labels.", notes.labelTest);
    sectionCard(s, 72, 190, "Setup", "Dataset: VisiumHD Exp1\nBackbone: mainly GigaPath crop128\nTargets: 3-class vs 8-class", C.cyan, C.blue);
    sectionCard(s, 472, 190, "Metric", "Validation expression accuracy, expression macro-F1, image accuracy, and training curves.", C.amber, C.orange);
    sectionCard(s, 872, 190, "Conclusion", "3-class is the current main baseline. 8-class remains a stress test because fine labels are noisier.", C.rose, C.red);
    embedImage(s, "presentation_3class_vs_8class_metrics.png", 90, 370, 510, 245, "contain");
    embedImage(s, "presentation_expression_ce_generalization_gap.png", 650, 370, 500, 245, "contain");
  }

  {
    const s = contentSlide(prs, "13 / Test 3: Loss ablation", "Loss test: separate classification supervision from cross-modal alignment", "This is where CE-only, CE+MSE, CE+InfoNCE, and InfoNCE-only are compared.", notes.lossTest);
    table(
      s,
      58,
      185,
      ["Loss variant", "Formula", "Question answered", "Current readout"],
      [
        ["CE + MSE baseline", "CE_img + CE_expr + 0.5*MSE_align", "does simple pairwise closeness help?", "older baseline, useful but less retrieval-targeted"],
        ["CE-only", "CE_img + CE_expr", "are shared tissue labels enough?", "strong raw R@1 baseline"],
        ["CE + InfoNCE", "CE_img + CE_expr + lambda*InfoNCE", "does contrastive alignment improve retrieval?", "helps broader top-k / fusion behavior"],
        ["InfoNCE-only", "InfoNCE", "can alignment alone work?", "diagnostic; weaker classification signal"],
      ],
      [190, 270, 390, 330],
      48,
    );
    txt(s, 82, 615, 1060, 28, "InfoNCE sweep: lambda = 0.05, 0.10, 0.20, 0.50; positive pair = same spatial H&E-expression sample; negatives = other batch samples.", 14, C.ink, true);
  }

  {
    const s = contentSlide(prs, "14 / Test 3 result: loss curves", "Validation loss is available for all main loss-ablation families", "Use this plot for stability and convergence, not absolute cross-family loss comparison.", notes.allLoss);
    embedImage(s, "all_loss_ablation_val_loss_grid.png", 45, 165, 1185, 480, "contain");
    txt(s, 70, 642, 1080, 26, "Result: 3-class is more stable than 8-class; InfoNCE weight changes training behavior, so lambda should stay small rather than dominate CE.", 13.5, C.slate, true);
  }

  {
    const s = contentSlide(prs, "15 / Test 3 result: loss components", "The objective separates image CE, expression CE, and alignment", "This makes clear which part of the model each loss term trains.", notes.components);
    embedImage(s, "gigapath_loss_components_by_variant.png", 62, 165, 1155, 470, "contain");
    txt(s, 78, 642, 1080, 26, "Conclusion: CE teaches tissue semantics; InfoNCE/MSE teaches image-expression geometry; the ablation is needed because these are not the same objective.", 13.5, C.slate, true);
  }

  {
    const s = contentSlide(prs, "16 / Test 4: Training strategy", "Fine-tuning test: frozen GigaPath versus partial adaptation", "Question: should the image encoder stay frozen, or should the last blocks adapt to VisiumHD tissue?", notes.strategyTest);
    table(
      s,
      68,
      190,
      ["Strategy", "Backbone", "Loss", "Result"],
      [
        ["Frozen constant", "GigaPath crop128", "CE + InfoNCE lambda 0.05", "good baseline, weaker image branch"],
        ["Frozen warmup/cosine", "GigaPath crop128", "CE + InfoNCE lambda 0.05", "scheduler alone did not help much"],
        ["Last-1 warmup/cosine", "GigaPath crop128", "CE + InfoNCE lambda 0.05", "image branch improves strongly"],
        ["Last-2 warmup/cosine", "GigaPath crop128", "CE + InfoNCE lambda 0.05", "best supervised/fused candidate"],
      ],
      [260, 230, 300, 350],
      48,
    );
  }

  {
    const s = contentSlide(prs, "17 / Test 4 result: fine-tuning curves", "Partial GigaPath fine-tuning changes the image branch most clearly", "The curves explain why last-2 is selected as the next checkpoint candidate.", notes.strategyCurves);
    embedImage(s, "training_strategy_all_curves.png", 58, 165, 1160, 470, "contain");
    txt(s, 78, 642, 1060, 26, "Result: last-block fine-tuning improves image accuracy while preserving expression macro-F1, supporting last-2 as the practical next model.", 13.5, C.slate, true);
  }

  {
    const s = contentSlide(prs, "18 / Supervised result table", "GigaPath last-2-block fine-tuning is the strongest supervised 3-class setting", "This table summarizes the classification side before retrieval evaluation.", notes.supervised);
    table(
      s,
      68,
      190,
      ["Run", "Expr acc", "Expr macro-F1", "Image acc", "Interpretation"],
      [
        ["Frozen constant", "0.7568", "0.6799", "0.6912", "strong expression anchor, weaker image branch"],
        ["Frozen warmup/cosine", "0.7532", "0.6796", "0.6908", "scheduler alone does not change much"],
        ["Last-1 warmup/cosine", "0.7652", "0.6929", "0.7920", "image adaptation helps"],
        ["Last-2 warmup/cosine", "0.7652", "0.6945", "0.8312", "best supervised candidate"],
        ["8-class last-2", "0.6316", "0.4512", "0.7084", "fine labels remain difficult"],
      ],
      [275, 105, 130, 105, 475],
      45,
    );
  }

  {
    const s = contentSlide(prs, "19 / Test 5: Retrieval and fusion", "Retrieval asks whether expression can find matching H&E regions", "Classification is not enough; cross-modal retrieval tests the shared-space geometry.", notes.retrievalTest);
    table(
      s,
      62,
      185,
      ["Evaluation", "Dataset / checkpoint", "Metric", "Main result"],
      [
        ["Expression-to-image retrieval", "held-out VisiumHD Exp1 embeddings", "R@1, R@5, R@10", "CE-only gives best raw R@1 = 0.745"],
        ["Fused retrieval", "image-expression fused score", "best fused R@1", "last-2 fine-tune gives best fused R@1 = 0.880"],
        ["Top-k retrieval", "loss-ablation checkpoints", "R@5 and R@10", "InfoNCE improves broader top-k behavior"],
      ],
      [250, 330, 230, 360],
      54,
    );
  }

  {
    const s = contentSlide(prs, "20 / Test 5 result: retrieval plot", "Retrieval and fusion show a different tradeoff from classification", "Raw nearest-neighbor retrieval and fused score retrieval identify slightly different best settings.", notes.retrieval);
    embedImage(s, "retrieval_overall_recall.png", 58, 170, 605, 365, "contain");
    embedImage(s, "gigapath_3class_per_label_recall1.png", 706, 190, 458, 275, "contain");
    metric(s, 710, 492, "Best raw R@1", "0.745", "GigaPath CE-only", C.white, C.orange);
    metric(s, 960, 492, "Best fused R@1", "0.880", "GigaPath last-2", C.white, C.plum);
    txt(s, 80, 615, 1040, 28, "Conclusion: use classification metrics to choose semantic stability, and retrieval metrics to choose SAM3 integration candidates.", 14.5, C.ink, true);
  }

  {
    const s = contentSlide(prs, "21 / Training-history coverage", "Training curves cover the full experiment batch", "This is the QA guardrail: the plots are generated from all available history.json files.", notes.coverage);
    table(
      s,
      90,
      200,
      ["Group", "Backbone", "Target", "Histories"],
      [
        ["initial", "GigaPath", "3-class / metric", "4"],
        ["initial", "ResNet50", "metric", "1"],
        ["loss ablation", "GigaPath", "3-class", "10"],
        ["loss ablation", "GigaPath", "8-class", "6"],
        ["loss ablation", "ResNet50", "3-class", "6"],
        ["loss ablation", "ResNet50", "8-class", "6"],
        ["training strategy", "GigaPath", "3-class / 8-class", "5"],
      ],
      [240, 220, 300, 160],
      42,
    );
    rect(s, 850, 260, 250, 155, C.mint, C.teal);
    txt(s, 878, 290, 195, 34, "38 histories\nloaded", 28, C.ink, true, "center");
    txt(s, 872, 350, 205, 28, "coverage saved as CSV", 13, C.slate, true, "center");
    txt(s, 94, 565, 980, 40, "Guardrail for future slides: before presenting results, first check which runs have per-epoch curves and which runs only have final metrics.", 17, C.ink, true);
  }

  {
    const s = contentSlide(prs, "22 / Literature link", "Haiku suggests the next training design, not a one-to-one copy", "Useful pieces: shared embedding, contrastive alignment, partial fine-tuning, and score-level fusion.", notes.haiku);
    table(
      s,
      70,
      190,
      ["Design element", "Haiku-style lesson", "Current adaptation"],
      [
        ["Modality setup", "paired local regions across modalities", "H&E crop paired with expression vector"],
        ["Projection", "encoder outputs are mapped into normalized shared vectors", "projection heads before CE / alignment / retrieval"],
        ["Alignment loss", "contrastive objective uses batch negatives", "symmetric InfoNCE added to CE baselines"],
        ["Fine-tuning", "freeze most pretrained encoders; tune last blocks", "frozen / last-1 / last-2 image tests"],
        ["Inference fusion", "combine modality scores rather than early fusion", "image-expression fused retrieval score"],
      ],
      [210, 430, 460],
      50,
    );
    txt(s, 92, 595, 1010, 32, "Important boundary: this slide motivates method choices only; it is not claiming tri-modal clinical modeling yet.", 15.5, C.ink, true);
  }

  {
    const s = contentSlide(prs, "23 / Conclusion", "Current result is a usable H&E-expression representation", "The next SAM3 experiment should use this space as a guide, not replace the segmentation model.", notes.conclusion);
    const cards = [
      ["What is solid", "3-class GigaPath models reach ~0.765 expression accuracy and ~0.694 macro-F1.", C.mint, C.teal],
      ["What is nuanced", "InfoNCE helps top-k/fusion behavior, but CE-only is still strong at raw R@1.", C.amber, C.orange],
      ["What is hard", "8-class fine labels remain unstable because class boundaries and morphology are noisier.", C.rose, C.red],
      ["Next action", "Use the best embedding to retrieve exemplars or rank SAM3 masks on matched tissue regions.", C.cyan, C.blue],
    ];
    cards.forEach(([h, b, fill, accent], i) => {
      const x = i % 2 === 0 ? 94 : 668;
      const y = i < 2 ? 212 : 405;
      rect(s, x, y, 455, 135, fill, accent);
      txt(s, x + 24, y + 18, 400, 24, h, 18, C.ink, true);
      txt(s, x + 24, y + 58, 390, 42, b, 17, C.ink, true);
    });
    txt(s, 102, 612, 960, 30, "Suggested final sentence: this validates the representation-learning step and narrows the SAM3 integration point.", 15.5, C.slate, true);
  }

  const pptx = await PresentationFile.exportPptx(prs);
  await pptx.save(OUT);

  for (let i = 0; i < prs.slides.count; i += 1) {
    const blob = await prs.export({ slide: prs.slides.getItem(i), format: "png", scale: 1 });
    const buffer = Buffer.from(await blob.arrayBuffer());
    await fs.writeFile(path.join(PREVIEW, `slide-${String(i + 1).padStart(2, "0")}.png`), buffer);
  }

  const python = path.join(
    process.env.HOME,
    ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
  );
  const contact = path.join(
    process.env.HOME,
    ".codex/plugins/cache/openai-primary-runtime/presentations/26.505.10851/skills/presentations/scripts/make_contact_sheet.py",
  );
  await fs.mkdir(path.join(HERE, "qa"), { recursive: true });
  const rendered = (await fs.readdir(PREVIEW))
    .filter((name) => name.endsWith(".png"))
    .sort()
    .map((name) => path.join(PREVIEW, name));
  const res = spawnSync(python, [contact, ...rendered, "--output", path.join(HERE, "qa", "contact_sheet.png"), "--cols", "3"], {
    stdio: "inherit",
  });
  if (res.status !== 0) throw new Error(`contact sheet failed with ${res.status}`);
  console.log(OUT);
}

await build();
