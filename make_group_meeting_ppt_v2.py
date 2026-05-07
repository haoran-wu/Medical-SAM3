"""
Group meeting PPT v2: SAM3 for Pathology Region Segmentation

12 slides with English speaker notes, based on the updated narrative:
- Exp1 first as the main dataset
- TMA24 as supporting pathology prompt-behavior evidence
- cross-modal alignment moved to future directions
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


BASE = Path(__file__).parent
OUT_DIR = BASE / "output" / "group_meeting_ppt"
ASSETS = OUT_DIR / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

OUTPUT_PPTX = OUT_DIR / f"group_meeting_updated_{date.today().strftime('%Y%m%d')}.pptx"

W = Inches(13.333)
H = Inches(7.5)

C = {
    "navy": RGBColor(0x16, 0x32, 0x4B),
    "blue": RGBColor(0x2E, 0x75, 0xB6),
    "light": RGBColor(0xF7, 0xF9, 0xFC),
    "line": RGBColor(0xD7, 0xE0, 0xE8),
    "gray": RGBColor(0x5E, 0x69, 0x75),
    "ink": RGBColor(0x1F, 0x25, 0x2C),
    "white": RGBColor(0xFF, 0xFF, 0xFF),
    "green": RGBColor(0x4B, 0x9B, 0x68),
    "red": RGBColor(0xD6, 0x45, 0x45),
    "gold": RGBColor(0xD6, 0xA4, 0x3A),
}

FIG = {
    "exp1_panels": BASE / "output/visium_hd_exp1/final_sam3_medicalsam3_overlays/overview/exp1_sam3_vs_medicalsam3_all_8_panels.png",
    "exp1_overlays": BASE / "output/visium_hd_exp1/final_sam3_medicalsam3_overlays/overview/exp1_sam3_vs_medicalsam3_all_8_prediction_overlays.png",
    "exp1_table": ASSETS / "exp1_results_table.png",
    "tile_table": ASSETS / "tile_ablation_table.png",
    "tile_schematic": ASSETS / "tile_schematic.png",
    "tma24_mapping": ASSETS / "tma24_cell_level_mapping_final_with_legend.png",
    "tma24_box_panel": ASSETS / "tma24_box_panel.png",
    "tma24_results_table": ASSETS / "tma24_results_table.png",
    "cross_summary": ASSETS / "cross_dataset_summary.png",
    "dual_encoder_arch": ASSETS / "dual_encoder_arch.png",
    "sam3_intervention": ASSETS / "sam3_intervention_diagram.png",
    "training_curves": ASSETS / "training_curves_summary.png",
    "title_img": BASE / "output/visium_hd_exp1/sam3_runs/labeled_overlay_collection/overview/base_sam3_all_8_labeled_prediction_overlays.png",
    "exp1_geojson_attr": Path("/Users/haoranwu/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_qgp2rh10glzm22_2458/temp/RWTemp/2026-04/af8c70125100720605dfb50603ed41d6/434f55130bc3f541cbd214e20e40b59b.png"),
    "exp1_geojson_on_he": Path("/Users/haoranwu/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_qgp2rh10glzm22_2458/temp/RWTemp/2026-04/af8c70125100720605dfb50603ed41d6/edfa917b618c7c171384a98ee601ac0c.png"),
}


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


def blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(1, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    return shape


def add_text(slide, text, x, y, w, h, size=18, bold=False, color=None, align=PP_ALIGN.LEFT):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color or C["ink"]
    return txb


def add_bullets(slide, items, x, y, w, h, size=15):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        level = 0
        text = item
        if item.startswith("  "):
            level = 1
            text = item.strip()
        p.level = level
        p.text = ("• " if level == 0 else "– ") + text
        p.font.size = Pt(size if level == 0 else size - 1)
        p.font.color.rgb = C["ink"] if level == 0 else C["gray"]
    return txb


def add_image(slide, path: Path, x, y, w, h):
    if not path.exists():
        add_rect(slide, x, y, w, h, C["light"], C["line"])
        add_text(slide, f"Missing:\n{path.name}", x + Inches(0.1), y + h / 2 - Inches(0.3), w - Inches(0.2), Inches(0.6), size=11, color=C["gray"], align=PP_ALIGN.CENTER)
        return
    with Image.open(path) as im:
        iw, ih = im.size
    aspect = iw / ih
    box_aspect = float(w) / float(h)
    nw, nh = w, h
    if aspect > box_aspect:
        nh = w / aspect
    else:
        nw = h * aspect
    slide.shapes.add_picture(str(path), x + (w - nw) / 2, y + (h - nh) / 2, nw, nh)


def header(slide, title, tag=""):
    add_rect(slide, 0, 0, W, Inches(0.58), C["navy"])
    add_text(slide, title, Inches(0.35), Inches(0.1), Inches(9.5), Inches(0.3), size=23, bold=True, color=C["white"])
    if tag:
        add_text(slide, tag, Inches(9.2), Inches(0.14), Inches(3.6), Inches(0.2), size=9, color=RGBColor(0xDD, 0xE7, 0xF2), align=PP_ALIGN.RIGHT)
    add_text(slide, "Haoran Wu | Group meeting | May 2026", Inches(0.35), Inches(7.16), Inches(4.4), Inches(0.12), size=9, color=C["gray"])


def panel(slide, x, y, w, h, title=""):
    add_rect(slide, x, y, w, h, C["white"], C["line"])
    if title:
        add_text(slide, title, x + Inches(0.15), y + Inches(0.08), w - Inches(0.3), Inches(0.22), size=16, bold=True, color=C["navy"])


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def load_json(path: Path):
    return json.loads(path.read_text())


def render_training_curves() -> None:
    contrastive = load_json(BASE / "output/visium_hd_exp1/hpc_snapshot_2026-05-06/checkpoint_meta/contrastive_history.json")
    cls_long = load_json(BASE / "output/visium_hd_exp1/hpc_snapshot_2026-05-06/checkpoint_meta/cls_history.json")
    g64 = load_json(BASE / "output/visium_hd_exp1/hpc_snapshot_2026-05-06/checkpoint_meta/cls_gigapath_crop64_quick2h_r2_history.json")
    g128 = load_json(BASE / "output/visium_hd_exp1/hpc_snapshot_2026-05-06/checkpoint_meta/cls_gigapath_crop128_quick2h_r2_history.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot([x["epoch"] for x in contrastive], [x["train_loss"] for x in contrastive], label="train", color="#2E75B6", lw=2)
    ax.plot([x["epoch"] for x in contrastive], [x["val_loss"] for x in contrastive], label="val", color="#D64545", lw=2)
    ax.set_title("Contrastive InfoNCE Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot([x["epoch"] for x in cls_long], [x["train"]["loss"] for x in cls_long], label="train total", color="#2E75B6", lw=2)
    ax.plot([x["epoch"] for x in cls_long], [x["val"]["loss"] for x in cls_long], label="val total", color="#D64545", lw=2)
    ax.plot([x["epoch"] for x in cls_long], [x["val"]["ce_expr"] for x in cls_long], label="val ce_expr", color="#4B9B68", lw=1.8, ls="--")
    ax.plot([x["epoch"] for x in cls_long], [x["val"]["ce_img"] for x in cls_long], label="val ce_img", color="#D6A43A", lw=1.8, ls="--")
    ax.set_title("Classification-Alignment Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 0]
    ax.plot([x["epoch"] for x in g64], [x["train"]["loss"] for x in g64], label="crop64 train", color="#2E75B6", lw=2)
    ax.plot([x["epoch"] for x in g64], [x["val"]["loss"] for x in g64], label="crop64 val", color="#2E75B6", lw=2, ls="--")
    ax.plot([x["epoch"] for x in g128], [x["train"]["loss"] for x in g128], label="crop128 train", color="#4B9B68", lw=2)
    ax.plot([x["epoch"] for x in g128], [x["val"]["loss"] for x in g128], label="crop128 val", color="#4B9B68", lw=2, ls="--")
    ax.set_title("GigaPath Quick Screen Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks([1, 2])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 1]
    labels = ["64-img", "64-expr", "128-img", "128-expr"]
    vals = [
        max(x["val"]["acc_img"] for x in g64),
        max(x["val"]["acc_expr"] for x in g64),
        max(x["val"]["acc_img"] for x in g128),
        max(x["val"]["acc_expr"] for x in g128),
    ]
    colors = ["#2E75B6", "#7BAFDE", "#4B9B68", "#94C9A3"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 0.85)
    ax.set_title("Best Validation Accuracy")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG["training_curves"], dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


prs = new_prs()
render_training_curves()

# Slide 1
s = blank_slide(prs)
add_rect(s, 0, 0, W, H, C["light"])
add_rect(s, Inches(0.55), Inches(0.6), Inches(5.35), Inches(6.1), C["white"], C["line"])
add_image(s, FIG["title_img"], Inches(0.8), Inches(0.95), Inches(4.85), Inches(5.35))
add_text(s, "SAM3 for Pathology Region Segmentation", Inches(6.35), Inches(1.2), Inches(6.2), Inches(0.55), size=26, bold=True, color=C["navy"])
add_text(s, "Evaluating SAM3 and MedicalSAM3 on semantic tissue-region segmentation in pathology and spatial transcriptomics", Inches(6.35), Inches(1.9), Inches(5.9), Inches(0.95), size=16, color=C["gray"])
add_bullets(s, [
    "Main datasets: VisiumHD Exp1 and TMA24",
    "Main question: can SAM-style prompting recover pathology-defined semantic regions?",
    "Take-home: prompt-following exists, but semantic region recovery remains weak",
    "Follow-up: cross-modal patch-level training is the more promising direction",
], Inches(6.35), Inches(3.2), Inches(5.9), Inches(1.7), size=15)
add_text(s, "Haoran Wu | Yan Lab | May 2026", Inches(6.35), Inches(6.2), Inches(4.2), Inches(0.2), size=13, color=C["gray"])
notes(s, """Today I’ll talk about our SAM3 experiments on pathology region segmentation. The main question is simple: can SAM-style prompting actually recover semantic tissue regions that pathologists care about? I’ll focus on two datasets: VisiumHD Exp1 as the main dataset, and TMA24 as an earlier prompt-behavior study. The short answer is that the models can follow strong prompts, but they still do not reliably segment semantic pathology regions.""")

# Slide 2
s = blank_slide(prs)
header(s, "Background and Related Work", "IAMSAM as the main point of comparison")
panel(s, Inches(0.4), Inches(0.95), Inches(6.1), Inches(5.95), "Problem")
add_bullets(s, [
    "Goal: automatic segmentation of semantic pathology regions from H&E",
    "Examples: tumor, stroma, immune infiltration, granuloma border",
    "Challenge: these are semantic tissue states, not compact object instances",
    "Therefore, pathology segmentation is different from ordinary object segmentation",
], Inches(0.7), Inches(1.35), Inches(5.5), Inches(2.3), size=15)
add_text(s, "Related work: IAMSAM (Genome Biology 2024)", Inches(0.7), Inches(4.1), Inches(3.6), Inches(0.22), size=16, bold=True, color=C["blue"])
add_bullets(s, [
    "Combines SAM with spatial transcriptomics for ROI-based analysis",
    "Supports everything-mode and user-interactive prompt-mode",
    "SAM segments H&E from morphology first",
    "ST / gene expression is used only after segmentation for downstream analysis",
], Inches(0.7), Inches(4.45), Inches(5.3), Inches(1.4), size=14)
panel(s, Inches(6.75), Inches(0.95), Inches(6.15), Inches(5.95), "Simple comparison")
add_bullets(s, [
    "IAMSAM: first cut the H&E image, then analyze gene expression inside the ROI",
    "  a human still gives the model strong spatial hints",
    "Our direction: use gene expression together with H&E to help decide where to cut",
    "  in other words, gene expression helps drive segmentation itself",
], Inches(7.1), Inches(1.45), Inches(5.45), Inches(2.1), size=15)
add_rect(s, Inches(7.2), Inches(3.85), Inches(4.8), Inches(1.35), RGBColor(0xF7, 0xF9, 0xFC), C["line"])
add_text(s, "IAMSAM", Inches(7.45), Inches(4.02), Inches(0.9), Inches(0.18), size=13, bold=True, color=C["navy"])
add_text(s, "H&E", Inches(8.1), Inches(4.02), Inches(0.6), Inches(0.18), size=11, color=C["ink"])
add_text(s, "->", Inches(9.2), Inches(4.0), Inches(0.3), Inches(0.18), size=13, bold=True, color=C["gray"])
add_text(s, "SAM", Inches(9.55), Inches(4.02), Inches(0.5), Inches(0.18), size=11, color=C["ink"])
add_text(s, "->", Inches(10.1), Inches(4.0), Inches(0.3), Inches(0.18), size=13, bold=True, color=C["gray"])
add_text(s, "gene analysis", Inches(10.42), Inches(4.02), Inches(1.1), Inches(0.18), size=11, color=C["ink"])
add_text(s, "Our direction", Inches(7.45), Inches(4.45), Inches(1.2), Inches(0.18), size=13, bold=True, color=C["navy"])
add_text(s, "gene + H&E", Inches(8.05), Inches(4.45), Inches(0.95), Inches(0.18), size=11, color=C["ink"])
add_text(s, "->", Inches(9.0), Inches(4.43), Inches(0.3), Inches(0.18), size=13, bold=True, color=C["gray"])
add_text(s, "guide SAM", Inches(9.28), Inches(4.45), Inches(0.9), Inches(0.18), size=11, color=C["ink"])
add_text(s, "->", Inches(10.25), Inches(4.43), Inches(0.3), Inches(0.18), size=13, bold=True, color=C["gray"])
add_text(s, "region mask", Inches(10.54), Inches(4.45), Inches(0.95), Inches(0.18), size=11, color=C["ink"])
add_rect(s, Inches(7.1), Inches(4.0), Inches(5.2), Inches(1.6), RGBColor(0xF4, 0xF8, 0xFC), C["line"])
add_text(s, "Interpretation frame for this talk", Inches(7.3), Inches(4.2), Inches(3.4), Inches(0.22), size=16, bold=True, color=C["navy"])
add_text(s, "The simple difference is: IAMSAM uses gene expression after cutting. We want gene expression to help with the cutting step itself.", Inches(7.3), Inches(4.55), Inches(4.8), Inches(0.85), size=14, color=C["ink"])
notes(s, """This is the framing for the talk. IAMSAM is the most relevant related paper. The simple version is: in IAMSAM, SAM first cuts the H and E image based on morphology, and only after that do they look at gene expression inside the selected ROI. So gene expression is downstream.

Our direction is different. We want to use gene expression together with H and E. In plain language, gene expression should help tell the model where it should look or what kind of tissue state it is dealing with, and then the segmentation model does the cutting. So the information flow is reversed. That is why IAMSAM is related, but not the same problem.""")

# Slide 3
s = blank_slide(prs)
header(s, "VisiumHD Exp1 Dataset", "Main dataset for this talk")
panel(s, Inches(0.4), Inches(0.95), Inches(5.0), Inches(5.95), "Dataset facts")
add_bullets(s, [
    "Human lung VisiumHD dataset",
    "H&E image: 3524 × 6000 px",
    "8 expert region labels from GeoJSON",
    "448,109 in-tissue bins at 8 μm resolution",
    "18,085 genes",
    "Priority labels: tumor, stroma, immune infiltration",
], Inches(0.75), Inches(1.35), Inches(4.3), Inches(2.7), size=15)
add_rect(s, Inches(0.75), Inches(4.45), Inches(4.0), Inches(1.45), RGBColor(0xF4, 0xF8, 0xFC), C["line"])
add_text(s, "Why this dataset is important", Inches(0.95), Inches(4.65), Inches(2.6), Inches(0.2), size=15, bold=True, color=C["navy"])
add_text(s, "It directly connects histology, expert semantic annotation, and spatial transcriptomics.", Inches(0.95), Inches(5.0), Inches(3.55), Inches(0.55), size=14, color=C["ink"])
panel(s, Inches(5.65), Inches(0.95), Inches(7.25), Inches(5.95), "GeoJSON annotations")
add_image(s, FIG["exp1_geojson_attr"], Inches(5.9), Inches(1.25), Inches(3.15), Inches(4.95))
add_image(s, FIG["exp1_geojson_on_he"], Inches(9.2), Inches(1.25), Inches(3.25), Inches(4.95))
add_text(s, "GeoJSON by label", Inches(6.35), Inches(6.18), Inches(2.1), Inches(0.18), size=11, color=C["gray"], align=PP_ALIGN.CENTER)
add_text(s, "GeoJSON mapped onto H&E", Inches(9.45), Inches(6.18), Inches(2.5), Inches(0.18), size=11, color=C["gray"], align=PP_ALIGN.CENTER)
notes(s, """This is VisiumHD Exp1, the main dataset for the talk. On the left is the GeoJSON annotation colored by label, and on the right is the same annotation mapped back onto the H and E image. So this dataset gives us two things at the same time: expert semantic region labels and paired spatial transcriptomics. There are 8 labels in total, and the main ones we care about are tumor, stroma, and immune infiltration.""")

# Slide 4
s = blank_slide(prs)
header(s, "VisiumHD Experimental Design", "Oracle-style prompt baseline")
panel(s, Inches(0.4), Inches(0.95), Inches(6.2), Inches(5.95), "Method")
add_bullets(s, [
    "Map expert GeoJSON annotations onto the H&E image and rasterize one binary mask per label",
    "These masks are used in two ways:",
    "  as the prompt-sampling source",
    "  as the evaluation target",
    "Tile-based inference: the 3524 × 6000 H&E image is split into 28 tiles",
    "  tile size = 1024 px, overlap = 128 px",
    "Oracle multipoint prompt for each label and each relevant tile:",
    "  10 positive points sampled inside the current label mask",
    "  5 negative points sampled near the label bounding box but outside the mask",
    "Prediction masks are stitched back to whole-image space and compared to the corresponding GeoJSON-derived binary mask",
    "Metrics: Dice, IoU, Precision, Recall",
], Inches(0.75), Inches(1.35), Inches(5.6), Inches(3.75), size=13)
add_rect(s, Inches(0.75), Inches(5.15), Inches(5.25), Inches(0.95), RGBColor(0xFE, 0xF5, 0xF5), C["line"])
add_text(s, "Important interpretation", Inches(0.95), Inches(5.02), Inches(2.8), Inches(0.2), size=15, bold=True, color=C["red"])
add_text(s, "Oracle here just means the prompt points are sampled from the ground-truth mask, so this is not automatic discovery.", Inches(0.95), Inches(5.42), Inches(4.7), Inches(0.42), size=13, color=C["ink"])
panel(s, Inches(6.85), Inches(0.95), Inches(6.05), Inches(5.95), "Pipeline schematic")
add_image(s, FIG["tile_schematic"], Inches(7.05), Inches(1.35), Inches(5.65), Inches(5.0))
notes(s, """This slide is the experimental method. We start with the H and E image and the expert GeoJSON file. We map the GeoJSON onto the H and E image and rasterize one binary mask for each label. Those masks are used in two ways: first, to sample prompt points; second, as the ground-truth target for evaluation.

We split the whole image into 28 tiles because the H and E image is large, and running SAM on the full image in one pass would force us to downsample too much and lose local histology detail. So each tile is run separately, and then we stitch the predictions back together.

For the prompt, we use multiple positive and negative points. Positive points come from inside the current label mask. Negative points come from near the bounding box, but outside the mask. We do this because these regions are large and irregular. A single box would include too much mixed tissue, and text prompts were already weak. The multipoint setup gives the model a clearer signal about what is in the region and what is not.

Also, I want to define oracle clearly. Oracle here just means the prompt points come from the ground-truth mask. So this is not asking whether the model can discover the region by itself. It is asking a simpler question: if I give the model very favorable points, can it recover the right semantic region?""")

# Slide 5
s = blank_slide(prs)
header(s, "VisiumHD Results: SAM3 vs MedicalSAM3", "Semantic region recovery remains weak")
panel(s, Inches(0.4), Inches(0.95), Inches(6.1), Inches(5.95), "Quantitative results")
add_image(s, FIG["exp1_table"], Inches(0.65), Inches(1.35), Inches(5.6), Inches(4.9))
panel(s, Inches(6.75), Inches(0.95), Inches(6.15), Inches(5.95), "Visual pattern")
add_image(s, FIG["exp1_overlays"], Inches(7.0), Inches(1.25), Inches(5.65), Inches(3.95))
add_rect(s, Inches(7.0), Inches(5.18), Inches(5.55), Inches(0.72), RGBColor(0xF4, 0xF8, 0xFC), C["line"])
add_text(s, "How to read the metrics", Inches(7.18), Inches(5.3), Inches(1.95), Inches(0.18), size=13, bold=True, color=C["navy"])
add_text(s, "Dice = overlap between prediction and GT; Precision = of predicted region pixels, how many are correct; Recall = of GT region pixels, how many are recovered.", Inches(7.18), Inches(5.48), Inches(5.08), Inches(0.34), size=11.2, color=C["ink"])
add_bullets(s, [
    "SAM3: higher recall, lower precision, tends to oversegment",
    "MedicalSAM3: higher precision, lower recall, tends to undersegment",
    "Best Dice examples:",
    "  stroma: 0.382 with SAM3",
    "  Lung Bronchiola: 0.470 with MedicalSAM3",
    "So the issue is not just weak prompting",
    "It suggests a mismatch between SAM-style masks and pathology semantic regions",
], Inches(7.0), Inches(5.95), Inches(5.5), Inches(1.0), size=12.5)
notes(s, """This is the main Exp1 result slide. Before talking about the numbers, I want to define the metrics very simply. Dice is the overlap score between the predicted mask and the ground-truth mask. Precision asks: among the pixels predicted as this region, how many are actually correct? Recall asks: among the pixels that truly belong to this region, how many did the model recover?

With that in mind, the pattern is pretty clear. SAM3 usually has higher recall but lower precision, which means it tends to spread too much and oversegment. MedicalSAM3 is more conservative. It can get better precision, but recall often drops a lot, which means it only captures a smaller part of the region.

So the takeaway is not just that the Dice values are modest. The failure modes are systematic: SAM3 oversegments, MedicalSAM3 undersegments, and neither one reliably recovers semantic tissue regions like tumor, stroma, or immune infiltration. That already suggests the problem is not just that our prompts are weak. Something about the task itself is mismatched to what SAM is good at.""")

# Slide 6
s = blank_slide(prs)
header(s, "Tile Size Ablation on Exp1", "Context alone does not fix the mismatch")
panel(s, Inches(0.4), Inches(0.95), Inches(7.0), Inches(5.95), "Ablation result")
add_image(s, FIG["tile_table"], Inches(0.7), Inches(1.45), Inches(6.4), Inches(3.7))
add_bullets(s, [
    "Larger tiles increased recall in several labels",
    "But precision and Dice generally decreased",
    "Interpretation: larger context encourages broader region expansion",
    "Conclusion: tile size is not the main bottleneck",
], Inches(0.85), Inches(5.45), Inches(5.8), Inches(0.95), size=14)
panel(s, Inches(7.7), Inches(0.95), Inches(5.2), Inches(5.95), "Takeaway")
add_rect(s, Inches(8.0), Inches(1.4), Inches(4.55), Inches(3.8), RGBColor(0xF4, 0xF8, 0xFC), C["line"])
add_text(s, "More context did not solve semantic segmentation.", Inches(8.25), Inches(2.0), Inches(4.0), Inches(0.5), size=20, bold=True, color=C["navy"], align=PP_ALIGN.CENTER)
add_text(s, "The main limitation is more consistent with model-task mismatch than with insufficient tile size.", Inches(8.2), Inches(3.0), Inches(4.1), Inches(1.0), size=15, color=C["ink"], align=PP_ALIGN.CENTER)
notes(s, """We also tested whether the poor Exp1 performance was mainly a tile artifact. Increasing tile size and overlap did increase recall in several cases, but precision and Dice got worse. So the model expanded more broadly, but not more accurately. This is important because it argues against a simple engineering explanation like insufficient context. Instead, it suggests that the mismatch is more fundamental than just tile size tuning.""")

# Slide 7
s = blank_slide(prs)
header(s, "TMA24: One-Slide Prompt Behavior Summary", "Supporting evidence from the earlier dataset")
panel(s, Inches(0.4), Inches(0.95), Inches(5.6), Inches(5.95), "Dataset + method")
add_bullets(s, [
    "Silicosis TMA dataset",
    "3 pseudo-region labels: mixed alveoli, granuloma border, hyalinized granuloma",
    "Pseudo-masks derived from mapped cell coordinates",
    "Key transfer test:",
    "  split the whole image into left half and right half",
    "  use only the left half to give prompts",
    "  score only on the right half",
    "  the prompted left region is excluded from evaluation",
], Inches(0.75), Inches(1.35), Inches(4.9), Inches(2.7), size=13.5)
add_image(s, FIG["tma24_mapping"], Inches(0.95), Inches(4.0), Inches(4.45), Inches(2.15))
panel(s, Inches(6.2), Inches(0.95), Inches(6.7), Inches(5.95), "Main result")
add_image(s, FIG["tma24_box_panel"], Inches(6.45), Inches(1.25), Inches(6.2), Inches(2.8))
add_bullets(s, [
    "Whole-image box prompt can segment the prompted region itself",
    "  hyalinized granuloma: Dice 0.711",
    "  mixed alveoli: Dice 0.653",
    "Text-only prompts were near zero",
    "Left-half -> right-half transfer stayed near zero",
    "Box, text, and box + text all failed to generalize",
    "Segment-everything / dense proposals did not fix it",
    "Random jitter / perturbation tests also did not rescue transfer",
], Inches(6.45), Inches(4.15), Inches(5.95), Inches(1.95), size=12.5)
notes(s, """I only want one slide on TMA24, but I do want to be clear about what we actually tested. We did not just try one prompt once. The key experiment was a left-to-right transfer setup. I split the whole image into a left half and a right half. The left half was used to give the model prompt information, such as text, box, or box plus text. The right half was never shown as a prompt region. When I scored the result, I only scored the right half, and I excluded the left prompted region from evaluation.

This is important because it turns TMA24 into a generalization test rather than a simple prompted segmentation demo. The result was that transfer was basically zero. I also tried multiple variants, including text, box, box plus text, dense or segment-everything style proposal search, and random perturbation or jitter tests. None of these rescued the transfer result.

So the main lesson from TMA24 is: SAM can segment the region you directly prompt, especially with a strong box. But it still does not learn a transferable semantic concept of the pathology region.""")

# Slide 8
s = blank_slide(prs)
header(s, "Why Can IAMSAM Use SAM Successfully?", "The task there is easier than ours")
panel(s, Inches(0.45), Inches(0.95), Inches(6.0), Inches(5.95), "IAMSAM setting")
add_bullets(s, [
    "SAM segments H&E based on morphology",
    "Two modes in the paper:",
    "  everything-mode: propose masks across the tissue image",
    "  prompt-mode: user draws rectangle boxes",
    "ST / gene expression is used after segmentation for downstream analysis",
    "So SAM works there because the segmentation task is morphology-based ROI extraction",
], Inches(0.8), Inches(1.35), Inches(5.1), Inches(2.1), size=14)
add_rect(s, Inches(0.8), Inches(4.25), Inches(5.0), Inches(1.35), RGBColor(0xF4, 0xF8, 0xFC), C["line"])
add_text(s, "Why that can work", Inches(1.0), Inches(4.45), Inches(1.8), Inches(0.2), size=15, bold=True, color=C["navy"])
add_text(s, "Because the task is ROI extraction with human guidance, not automatic semantic region segmentation driven by ST.", Inches(1.0), Inches(4.78), Inches(4.55), Inches(0.55), size=13, color=C["ink"])
panel(s, Inches(6.75), Inches(0.95), Inches(6.15), Inches(5.95), "Why our task is harder")
add_bullets(s, [
    "We ask the model to recover semantic pathology regions directly",
    "Examples: tumor, stroma, immune infiltration",
    "These are large, irregular, mixed, and expert-defined",
    "So the model must do more than follow an ROI boundary",
    "It has to infer a semantic tissue concept",
], Inches(7.1), Inches(1.35), Inches(5.1), Inches(2.2), size=14)
add_image(s, FIG["cross_summary"], Inches(7.2), Inches(4.05), Inches(5.15), Inches(1.55))
notes(s, """This slide is where I explain why SAM can work in the IAMSAM paper. The reason is not that SAM has solved semantic pathology segmentation in general. The reason is that the IAMSAM task is easier. SAM is used there for morphology-based ROI extraction. In everything-mode, it proposes masks across the tissue image. In prompt-mode, the user draws rectangle boxes, so the model already knows where to segment. Then the selected ROI is passed to downstream ST analysis.

So SAM works there because it is being used in the setting it is good at: cutting out morphologically distinct regions when the spatial cue is already available. Our task is harder because we want the model to recover semantic pathology regions directly, and eventually use ST data as an input that drives segmentation instead of only analyzing expression after segmentation.""")

# Slide 9
s = blank_slide(prs)
header(s, "What Is Actually Failing?", "Not just prompt weakness")
panel(s, Inches(0.45), Inches(0.95), Inches(6.2), Inches(5.95), "Diagnosis")
add_bullets(s, [
    "Prompt weakness is part of the problem",
    "  text prompts fail and single boxes are too coarse",
    "But prompt weakness is not the whole story",
    "  even oracle multipoint prompts are still weak",
    "So the deeper issue is likely SAM itself",
    "  SAM is good at object-like, boundary-driven segmentation",
    "  pathology semantic regions are not clean object masks",
], Inches(0.8), Inches(1.35), Inches(5.35), Inches(2.5), size=14)
add_rect(s, Inches(0.8), Inches(4.35), Inches(5.15), Inches(1.25), RGBColor(0xFE, 0xF5, 0xF5), C["line"])
add_text(s, "Bottom line", Inches(1.0), Inches(4.55), Inches(1.5), Inches(0.2), size=15, bold=True, color=C["red"])
add_text(s, "This looks more like a model-task mismatch than a prompt-engineering problem.", Inches(1.0), Inches(4.9), Inches(4.6), Inches(0.45), size=13, color=C["ink"])
panel(s, Inches(6.9), Inches(0.95), Inches(6.0), Inches(5.95), "Why pathology regions are hard for SAM")
add_bullets(s, [
    "They are fragmented and interleaved",
    "They often have weak or fuzzy visual boundaries",
    "They are defined by tissue state, not by object instance",
    "Neighboring regions can look locally similar but mean different biology",
], Inches(7.2), Inches(1.35), Inches(5.1), Inches(2.0), size=14)
add_image(s, FIG["exp1_geojson_on_he"], Inches(7.45), Inches(3.55), Inches(4.85), Inches(2.2))
notes(s, """This is the diagnosis slide. I want to be explicit here: I do not think the conclusion is simply that our prompts were bad. Prompt weakness is real, especially for text and coarse boxes, but that is not enough to explain the results. Even when we gave the model favorable multipoint prompts sampled from the ground-truth mask, the segmentation was still weak. That suggests the deeper issue is SAM itself, or more precisely the mismatch between what SAM is built for and what this task requires. SAM is strongest when there is an object-like region with a meaningful visual boundary. But pathology semantic regions are often fragmented, mixed, and defined by tissue state rather than by a clean object mask.""")

# Slide 10
s = blank_slide(prs)
header(s, "Where Does The Training Enter?", "Current training is upstream of SAM3, not end-to-end SAM fine-tuning")
panel(s, Inches(0.35), Inches(0.95), Inches(7.6), Inches(5.95), "Pipeline location")
add_image(s, FIG["sam3_intervention"], Inches(0.55), Inches(1.28), Inches(7.15), Inches(4.95))
panel(s, Inches(8.15), Inches(0.95), Inches(4.75), Inches(5.95), "What we are training now")
add_bullets(s, [
    "This corresponds mainly to Direction C",
    "  gene expression -> expression encoder -> shared embedding / prompt-side signal",
    "And partially to Direction B",
    "  replacing the image tower with GigaPath instead of generic ResNet50",
    "Important: current runs do not train SAM3 end-to-end yet",
    "  they train a separate dual-encoder before the SAM step",
    "After training, the embedding can be used for:",
    "  retrieval of similar regions",
    "  localization / heatmaps",
    "  later prompt-adapter experiments into SAM3",
], Inches(8.45), Inches(1.35), Inches(4.0), Inches(3.45), size=13.2)
add_rect(s, Inches(8.45), Inches(5.0), Inches(4.0), Inches(0.88), RGBColor(0xF4, 0xF8, 0xFC), C["line"])
add_text(s, "Training dataset", Inches(8.63), Inches(5.18), Inches(1.5), Inches(0.18), size=15, bold=True, color=C["navy"])
add_text(s, "VisiumHD Exp1 8 um bins paired with H&E crops: 420,645 non-background examples, 18,085 genes, 8 labels.", Inches(8.63), Inches(5.44), Inches(3.55), Inches(0.34), size=12.2, color=C["ink"])
notes(s, """This is the most important clarification for the training section. The model we trained is not yet SAM3 end-to-end. It enters upstream of SAM3. In the April 30 framework, this is mainly Direction C, and partly Direction B when I switch the image tower to GigaPath. Concretely, I take one H and E patch and its matched gene expression vector from VisiumHD Exp1, and train a dual-encoder so the two modalities land in a shared embedding space. That shared space can later support retrieval, localization, or a prompt adapter into SAM3, but the current training itself happens before the SAM segmentation stage.""")

# Slide 11
s = blank_slide(prs)
header(s, "What Loss Did We Optimize?", "Two training objectives, both on patch-expression pairs rather than masks")
panel(s, Inches(0.4), Inches(0.95), Inches(4.0), Inches(5.95), "Contrastive alignment")
add_bullets(s, [
    "Image tower: image patch -> encoder -> z_img",
    "Expression tower: expression vector -> MLP -> z_expr",
    "Loss: symmetric InfoNCE / NT-Xent",
    "  L = 1/2[CE(sim(z_img,z_expr)/tau) + CE(sim(z_expr,z_img)/tau)]",
    "Meaning:",
    "  matched patch-expression pairs are pulled together",
    "  mismatched pairs in the minibatch are pushed apart",
    "Used for: modality alignment without explicit mask supervision",
], Inches(0.7), Inches(1.35), Inches(3.4), Inches(3.35), size=13.0)
add_rect(s, Inches(0.72), Inches(4.92), Inches(3.35), Inches(0.85), RGBColor(0xF7, 0xF9, 0xFC), C["line"])
add_text(s, "Current limitation", Inches(0.95), Inches(5.08), Inches(1.45), Inches(0.18), size=15, bold=True, color=C["navy"])
add_text(s, "No explicit region-label supervision, so instance matching is a hard objective on this dataset.", Inches(0.95), Inches(5.34), Inches(2.9), Inches(0.28), size=12.0, color=C["ink"])
panel(s, Inches(4.65), Inches(0.95), Inches(4.0), Inches(5.95), "Classification + alignment")
add_bullets(s, [
    "Both towers predict the semantic region label",
    "Loss = CE(img_pred,label) + CE(expr_pred,label) + lambda * MSE(norm(z_img), norm(z_expr))",
    "In code: lambda = align_weight = 0.5",
    "Meaning:",
    "  CE(img): patch embedding should classify tissue region",
    "  CE(expr): expression embedding should classify tissue region",
    "  MSE align: matched embeddings should stay close",
    "This is easier to interpret than pure contrastive loss",
], Inches(4.95), Inches(1.35), Inches(3.35), Inches(3.35), size=13.0)
add_rect(s, Inches(4.95), Inches(4.92), Inches(3.1), Inches(0.85), RGBColor(0xF2, 0xFA, 0xF4), C["line"])
add_text(s, "What to retain", Inches(5.15), Inches(5.08), Inches(1.5), Inches(0.18), size=15, bold=True, color=C["green"])
add_text(s, "Keep the loss breakdown itself: total loss, CE_img, CE_expr, align, plus val acc_img / val acc_expr / val macro-F1.", Inches(5.15), Inches(5.34), Inches(2.55), Inches(0.34), size=12.0, color=C["ink"])
panel(s, Inches(8.9), Inches(0.95), Inches(4.0), Inches(5.95), "Training schedule")
add_bullets(s, [
    "Default training script has a two-phase schedule:",
    "  phase 1: freeze image backbone",
    "  phase 2: unfreeze backbone at lower LR",
    "Quick GigaPath screen last week:",
    "  total_epochs = 8, freeze_epochs = 8",
    "  so GigaPath stayed frozen for the whole quick screen",
    "  this was a feature-quality test, not full end-to-end fine-tuning",
    "Class imbalance handling in the quick screen:",
    "  balanced sampler + sqrt-inverse class weights",
], Inches(9.2), Inches(1.35), Inches(3.35), Inches(3.55), size=12.8)
notes(s, """This slide is where I would define the training objective clearly. There are really two training lines. The first is pure contrastive alignment with symmetric InfoNCE. The second is the classification-plus-alignment version, which is easier to interpret because I can separate image classification loss, expression classification loss, and the alignment term. I also want to say something subtle but important: the GigaPath quick screen was not full fine-tuning. In that sbatch run, freeze_epochs equaled total_epochs, so GigaPath stayed frozen the whole time. That means the quick screen is telling us about the quality of the frozen pathology backbone features, not about the full upper bound after fine-tuning.""")

# Slide 12
s = blank_slide(prs)
header(s, "Loss Curves And Key Metrics", "What to keep in the meeting version")
panel(s, Inches(0.35), Inches(0.95), Inches(8.2), Inches(5.95), "Observed training behavior")
add_image(s, FIG["training_curves"], Inches(0.55), Inches(1.28), Inches(7.8), Inches(5.25))
panel(s, Inches(8.75), Inches(0.95), Inches(4.15), Inches(5.95), "Numbers to retain")
add_bullets(s, [
    "Contrastive (ResNet50, 64 px):",
    "  best val loss 5.579 at epoch 1",
    "  val loss rose to 9.230 by epoch 80",
    "  readout: strong overfitting / weak objective match",
    "Classification long run:",
    "  best val expr acc 0.485 at epoch 2",
    "  best val img acc 0.455 at epoch 1",
    "  by epoch 60: val expr acc 0.350, val img acc 0.398",
    "GigaPath quick screen:",
    "  crop64: val img acc 0.726, val expr acc 0.552",
    "  crop128: val img acc 0.695, val expr acc 0.553",
], Inches(9.0), Inches(1.35), Inches(3.45), Inches(3.95), size=12.7)
add_rect(s, Inches(9.0), Inches(5.1), Inches(3.45), Inches(0.78), RGBColor(0xF2, 0xFA, 0xF4), C["line"])
add_text(s, "Bottom line", Inches(9.18), Inches(5.25), Inches(1.1), Inches(0.18), size=15, bold=True, color=C["green"])
add_text(s, "Keep the pipeline story, the exact loss definitions, the loss trend, and the best validation metrics. Those are the core results from training.", Inches(9.18), Inches(5.48), Inches(3.0), Inches(0.28), size=12.0, color=C["ink"])
notes(s, """This last training slide should be read very concretely. For the contrastive run, the train loss goes down but the validation loss steadily rises, so the model is memorizing or learning a notion of instance matching that does not transfer. For the long classification run, the same thing happens even though the objective is more interpretable. But the quick GigaPath screen changes the picture: the validation image accuracy jumps to about 0.73 with crop size 64 while expression-side accuracy stays around 0.55. So the message is not that training failed completely. The message is that generic objectives and generic backbones are weak, while pathology-specific image features look much more promising. In the meeting, I would definitely keep the loss definitions, the loss curves, and the best validation metrics because those are what make the training story feel real rather than speculative.""")

prs.save(OUTPUT_PPTX)
print(f"Saved PPTX to {OUTPUT_PPTX}")
