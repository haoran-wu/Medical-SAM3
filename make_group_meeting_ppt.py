"""
Group meeting PPT: SAM3 for Pathology Region Segmentation
12 slides with speaker notes.
"""

import os
import textwrap
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
import lxml.etree as etree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
OUT_DIR = BASE / "output" / "group_meeting_ppt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS = OUT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)

OUTPUT_PPTX = OUT_DIR / f"group_meeting_{date.today().strftime('%Y%m%d')}.pptx"

# Figure paths
FIG = {
    # VisiumHD Exp1
    "exp1_all_panels":    BASE / "output/visium_hd_exp1/final_sam3_medicalsam3_overlays/overview/exp1_sam3_vs_medicalsam3_all_8_panels.png",
    "exp1_all_overlays":  BASE / "output/visium_hd_exp1/final_sam3_medicalsam3_overlays/overview/exp1_sam3_vs_medicalsam3_all_8_prediction_overlays.png",
    # TMA24
    "tma24_legend":       BASE / "output/00_FINAL_tma24_example1_scale_0p55_shiftX_neg120_shiftY_620/combined_region_overlay_with_legend.png",
    "tma24_box_panel":    BASE / "output/group_meeting_ppt/assets/tma24_box_panel.png",
    "tma24_alveoli_box":  BASE / "output/hpc_tma24_whole_image_prompts/panels/01_mixed_alveoli.png",
    "tma24_granuloma_box":BASE / "output/hpc_tma24_whole_image_prompts/panels/03_hyalinized_granuloma.png",
    "tma24_border_box":   BASE / "output/hpc_tma24_whole_image_prompts/panels/02_granuloma_border.png",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
W, H = Inches(13.33), Inches(7.5)   # 16:9 slide

C = {
    "bg":      RGBColor(0xFF, 0xFF, 0xFF),
    "title":   RGBColor(0x1F, 0x35, 0x64),   # dark navy
    "accent":  RGBColor(0x2E, 0x75, 0xB6),   # mid blue
    "red":     RGBColor(0xC0, 0x00, 0x00),
    "green":   RGBColor(0x37, 0x86, 0x44),
    "gray":    RGBColor(0x59, 0x59, 0x59),
    "light":   RGBColor(0xD6, 0xE4, 0xF0),
    "white":   RGBColor(0xFF, 0xFF, 0xFF),
    "black":   RGBColor(0x00, 0x00, 0x00),
}

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)

def add_rect(slide, l, t, w, h, fill=None, line=None):
    shape = slide.shapes.add_shape(1, l, t, w, h)  # MSO_SHAPE_TYPE.RECTANGLE
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_text(slide, text, l, t, w, h, size=18, bold=False, color=None,
             align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color or C["black"]
    return txb

def add_bullets(slide, items, l, t, w, h, size=16, title=None, title_size=18):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.text = title
        p.font.bold = True
        p.font.size = Pt(title_size)
        p.font.color.rgb = C["title"]
        first = False
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        # indent level
        level = 0
        text  = item
        if item.startswith("  "):
            level = 1
            text  = item.strip()
        p.level = level
        p.text  = ("• " if level == 0 else "  – ") + text
        p.font.size = Pt(size if level == 0 else size - 1)
        p.font.color.rgb = C["gray"] if level else C["black"]

def add_image(slide, path, l, t, w, h=None):
    path = Path(path)
    if not path.exists():
        print(f"  [WARN] missing figure: {path.name}")
        add_rect(slide, l, t, w, h or Inches(3), fill=RGBColor(0xEE, 0xEE, 0xEE))
        add_text(slide, f"[{path.name}]", l, t, w, h or Inches(0.4),
                 size=10, color=C["gray"], align=PP_ALIGN.CENTER)
        return
    if h:
        slide.shapes.add_picture(str(path), l, t, w, h)
    else:
        slide.shapes.add_picture(str(path), l, t, w)

def set_notes(slide, text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text

def header_bar(slide, title, subtitle=None):
    """Dark navy header bar across the top."""
    bar_h = Inches(1.05)
    add_rect(slide, 0, 0, W, bar_h, fill=C["title"])
    add_text(slide, title, Inches(0.3), Inches(0.08), Inches(12.5), Inches(0.6),
             size=28, bold=True, color=C["white"])
    if subtitle:
        add_text(slide, subtitle, Inches(0.3), Inches(0.65), Inches(12.5), Inches(0.35),
                 size=14, color=C["light"])

def make_table_img(headers, rows, path, col_widths=None, title=None,
                   highlight_cols=None, figsize=None):
    """Render a table as a PNG with matplotlib."""
    ncols = len(headers)
    nrows = len(rows)
    if figsize is None:
        figsize = (ncols * 1.6, nrows * 0.38 + 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    # header style
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor("#1F3564")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    # row alternating + highlight
    for i in range(1, nrows + 1):
        for j in range(ncols):
            cell = tbl[i, j]
            cell.set_facecolor("#EBF3FA" if i % 2 == 0 else "white")
            if highlight_cols and j in highlight_cols:
                v = rows[i-1][j]
                try:
                    fv = float(v)
                    if fv >= 0.5:
                        cell.set_facecolor("#C6EFCE")
                    elif fv >= 0.3:
                        cell.set_facecolor("#FFEB9C")
                    elif fv <= 0.05:
                        cell.set_facecolor("#FFC7CE")
                except (ValueError, TypeError):
                    pass
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

def make_sam3_intervention_diagram(path):
    """
    Draw SAM3 architecture + colored arrows showing WHERE each future direction intervenes.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=10, lw=2, ec="white"):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold", multialignment="center")

    def arr(x1, y1, x2, y2, col="gray", lw=2, style="->"):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle=style, color=col, lw=lw))

    def label_arrow(x, y, text, color, dx=0, dy=0.35, ha="center"):
        ax.annotate(text, xy=(x, y), xytext=(x+dx, y+dy),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
            fontsize=9, color=color, fontweight="bold", ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.5))

    # ── Core SAM3 pipeline (center row, y≈2.5) ──────────────────────────────
    box(0.3,  2.1, 2.0, 0.9, "H&E Image\n(tile 1024px)", "#AAAAAA", tc="#222")
    arr(2.3, 2.55, 3.1, 2.55)

    box(3.1,  2.1, 2.5, 0.9, "Image\nEncoder (ViT)", "#1F3564")
    arr(5.6, 2.55, 6.3, 2.55)

    box(6.3,  2.1, 2.5, 0.9, "Prompt\nEncoder", "#1F3564")
    arr(8.8, 2.55, 9.5, 2.55)

    box(9.5,  2.1, 2.2, 0.9, "Mask\nDecoder", "#1F3564")
    arr(11.7, 2.55, 12.4, 2.55)

    box(12.4, 2.1, 1.3, 0.9, "Mask\nOutput", "#2E75B6")

    # Prompt inputs below prompt encoder
    box(5.7, 0.6, 1.1, 0.7, "Points\n/ Boxes", "#888", tc="#222", fs=8, lw=1)
    arr(6.25, 1.3, 7.0, 2.1, "gray", lw=1.5)
    box(6.95, 0.6, 1.0, 0.7, "Text\n(CLIP)", "#888", tc="#222", fs=8, lw=1)
    arr(7.45, 1.3, 7.45, 2.1, "gray", lw=1.5)
    box(8.1, 0.6, 1.1, 0.7, "Exemplar\nPatches", "#888", tc="#222", fs=8, lw=1)
    arr(8.65, 1.3, 7.8, 2.1, "gray", lw=1.5)

    ax.text(7.0, 0.15, "← existing prompt modalities →",
            ha="center", fontsize=8, color="gray", style="italic")

    # ── Direction A: Fine-tune SAM3 (orange, touches Image Encoder + Mask Decoder) ──
    rect_a1 = mpatches.FancyBboxPatch((3.05, 2.05), 2.6, 1.0,
        boxstyle="round,pad=0.05", fill=False, edgecolor="#C55A11", lw=2.5, linestyle="--")
    ax.add_patch(rect_a1)
    rect_a2 = mpatches.FancyBboxPatch((9.45, 2.05), 2.3, 1.0,
        boxstyle="round,pad=0.05", fill=False, edgecolor="#C55A11", lw=2.5, linestyle="--")
    ax.add_patch(rect_a2)
    ax.text(4.35, 3.25, "A. Supervised Fine-tune", ha="center",
            fontsize=9, color="#C55A11", fontweight="bold")
    ax.annotate("", xy=(4.35, 3.15), xytext=(4.35, 3.05),
        arrowprops=dict(arrowstyle="->", color="#C55A11", lw=1.5))
    ax.text(10.6, 3.25, "(same)", ha="center", fontsize=8, color="#C55A11")

    # ── Direction B: GigaPath backbone (blue, touches Image Encoder) ──
    rect_b = mpatches.FancyBboxPatch((3.05, 2.05), 2.6, 1.0,
        boxstyle="round,pad=0.05", fill=False, edgecolor="#2E75B6", lw=2, linestyle=":")
    ax.add_patch(rect_b)
    ax.text(4.35, 3.55, "B. GigaPath Backbone",
            ha="center", fontsize=9, color="#2E75B6", fontweight="bold")
    ax.annotate("", xy=(4.35, 3.45), xytext=(4.35, 3.25),
        arrowprops=dict(arrowstyle="->", color="#2E75B6", lw=1.5))

    # ── Direction C: Expression → Prompt Encoder (green, top arrow) ──
    ax.annotate("",
        xy=(7.55, 3.0), xytext=(7.55, 4.3),
        arrowprops=dict(arrowstyle="->", color="#375623", lw=2.5))
    box(6.2, 4.3, 2.7, 0.75,
        "Expression\nMLP Encoder", "#375623", fs=9)
    arr(3.5, 4.67, 6.2, 4.67, "#375623", lw=1.5)
    box(1.0, 4.3, 2.2, 0.75,
        "Gene Expression\n(18K genes)", "#6AA84F", fs=8)
    ax.text(7.55, 3.85, "C. Expression\n→ Prompt", ha="center",
            fontsize=9, color="#375623", fontweight="bold")

    # ── Direction D: RAG → Exemplar path (purple) ──
    ax.annotate("",
        xy=(8.65, 1.3), xytext=(8.65, 0.6),
        arrowprops=dict(arrowstyle="->", color="#7030A0", lw=2))
    box(7.7, 5.1, 3.3, 0.7,
        "Reference DB\n(annotated patches)", "#7030A0", fs=8)
    ax.annotate("",
        xy=(9.35, 5.1), xytext=(9.35, 4.4),
        arrowprops=dict(arrowstyle="->", color="#7030A0", lw=2))
    ax.text(11.3, 5.45, "D. RAG: retrieve similar\nannotated patches\nas exemplar prompts",
            ha="left", fontsize=9, color="#7030A0", fontweight="bold")
    ax.annotate("",
        xy=(9.35, 4.35), xytext=(9.35, 5.8),
        arrowprops=dict(arrowstyle="<-", color="#7030A0", lw=1.5))
    ax.text(8.6, 4.5, "Query\n(image emb.)",
            ha="center", fontsize=8, color="#7030A0")

    # ── Direction E: Replace entire model (red bracket) ──
    brace = mpatches.FancyBboxPatch((0.25, 1.95), 12.0, 1.2,
        boxstyle="round,pad=0.05", fill=False, edgecolor="#C00000", lw=1.5, linestyle="-.")
    ax.add_patch(brace)
    ax.text(12.55, 1.9, "E. Replace with\nCellViT / HoVerNet",
            ha="left", fontsize=9, color="#C00000", fontweight="bold")

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend_items = [
        mpatches.Patch(facecolor="#C55A11", label="A  Supervised fine-tune (Image Enc + Mask Dec)"),
        mpatches.Patch(facecolor="#2E75B6", label="B  GigaPath: replace Image Encoder"),
        mpatches.Patch(facecolor="#375623", label="C  Expression MLP → Prompt Encoder"),
        mpatches.Patch(facecolor="#7030A0", label="D  RAG: retrieve exemplars → Prompt Encoder"),
        mpatches.Patch(facecolor="#C00000", label="E  Replace entire SAM3 with pathology model"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8.5,
              framealpha=0.95, ncol=1, bbox_to_anchor=(0.0, -0.22))

    ax.set_title("SAM3 Architecture — Where Each Future Direction Intervenes",
                 fontsize=12, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def make_rag_diagram(path):
    """RAG-based prompting pipeline for SAM3."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.12", facecolor=fc, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold", multialignment="center")

    def arr(x1, y1, x2, y2, col="#555", lw=2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle="->", color=col, lw=lw))

    # ── Offline: build reference database ──────────────────────────────────
    ax.text(3.2, 4.7, "OFFLINE  (annotated training data)",
            ha="center", fontsize=10, color="#888", style="italic")

    box(0.2, 3.6, 2.0, 0.8, "Annotated\nH&E Patches", "#C55A11", fs=9)
    box(0.2, 2.5, 2.0, 0.8, "Matched\nExpression\nVectors", "#375623", fs=8)
    box(0.2, 1.5, 2.0, 0.8, "Region\nLabels", "#888888", tc="#222", fs=9)

    arr(2.2, 4.0, 3.0, 4.0)
    arr(2.2, 2.9, 3.0, 3.4)
    arr(2.2, 1.9, 3.0, 3.1)

    box(3.0, 3.0, 2.4, 1.2, "Dual-Encoder\n(train offline)\nimage + expr → 128d", "#1F3564", fs=8)
    arr(5.4, 3.6, 6.1, 3.6)

    box(6.1, 3.15, 2.2, 0.9, "Reference\nVector DB\n(FAISS / cosine)", "#7030A0", fs=8)

    ax.text(6.5, 2.75, "↑ built once,\nreused at inference", fontsize=8,
            color="#7030A0", ha="center")

    # ── Online: inference on new unannotated slide ───────────────────────────
    ax.add_patch(plt.Rectangle((0.1, 0.1), 12.8, 2.2, fill=False,
        edgecolor="#2E75B6", lw=1.5, linestyle="--"))
    ax.text(6.5, 0.0, "ONLINE  (new unannotated slide)",
            ha="center", fontsize=10, color="#2E75B6", style="italic")

    box(0.2, 0.9, 1.9, 0.9, "New H&E\nTile", "#C55A11", fs=9)
    arr(2.1, 1.35, 3.0, 1.35)

    box(3.0, 0.9, 2.0, 0.9, "Image\nEncoder", "#1F3564", fs=9)
    arr(5.0, 1.35, 5.7, 1.35)

    # Query arrow into DB
    ax.annotate("", xy=(7.2, 3.15), xytext=(5.85, 1.8),
        arrowprops=dict(arrowstyle="->", color="#7030A0", lw=1.8,
                        connectionstyle="arc3,rad=-0.2"))
    ax.text(5.6, 2.6, "query\n(cosine sim)", fontsize=8, color="#7030A0",
            ha="center")

    # Retrieved exemplars
    arr(8.3, 3.6, 9.0, 2.2, "#7030A0", lw=1.8)
    box(9.0, 0.9, 2.1, 0.9, "Top-K\nRetrieved\nExemplars", "#7030A0", fs=8)

    arr(5.7, 1.35, 9.0, 1.35, "#555", lw=1.5)
    box(5.7, 0.9, 3.1, 0.9, "Prompt\nEncoder\n(exemplar mode)", "#1F3564", fs=8)
    arr(8.8, 1.35, 9.0, 1.35)

    arr(11.1, 1.35, 11.7, 1.35)
    box(11.7, 0.9, 1.1, 0.9, "Mask\nDecoder", "#1F3564", fs=8)
    arr(12.8, 1.35, 13.0, 1.35)

    ax.set_title("RAG-Based Prompting: Retrieve Annotated Examples → Exemplar Prompts → SAM3",
                 fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def make_arch_diagram(path):
    """Draw the dual-encoder architecture diagram."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def box(x, y, w, h, text, fcolor="#2E75B6", tcolor="white", fs=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", facecolor=fcolor,
            edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                color=tcolor, fontsize=fs, fontweight="bold",
                multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    # Image tower
    box(0.2, 3.0, 2.0, 0.8, "H&E Patch\n(64×64 px)", "#C55A11")
    arrow(2.2, 3.4, 3.0, 3.4)
    box(3.0, 3.0, 2.2, 0.8, "Image Encoder\n(ResNet50 / GigaPath)", "#2E75B6")
    arrow(5.2, 3.4, 6.0, 3.4)
    box(6.0, 3.0, 1.8, 0.8, "Projection\nHead → 128d", "#2E75B6")
    arrow(7.8, 3.4, 8.6, 3.4)

    # Expression tower
    box(0.2, 1.2, 2.0, 0.8, "Gene Expression\n(18,085 genes)", "#375623")
    arrow(2.2, 1.6, 3.0, 1.6)
    box(3.0, 1.2, 2.2, 0.8, "MLP Encoder\n(18085→256→128)", "#375623")
    arrow(5.2, 1.6, 6.0, 1.6)
    box(6.0, 1.2, 1.8, 0.8, "Projection\nHead → 128d", "#375623")
    arrow(7.8, 1.6, 8.6, 1.6)

    # Shared space
    box(8.6, 1.9, 2.1, 1.8,
        "Shared\nEmbedding\nSpace\n(128-dim)", "#1F3564", fs=9)

    # Alignment
    ax.annotate("", xy=(6.9, 2.9), xytext=(6.9, 2.0),
        arrowprops=dict(arrowstyle="<->", color="#C00000", lw=2))
    ax.text(7.05, 2.45, "alignment\n(CE + MSE)", color="#C00000",
            fontsize=8, va="center")

    # Labels
    ax.text(1.2, 4.1, "Image Tower", ha="center", fontsize=10,
            color="#C55A11", fontweight="bold")
    ax.text(1.2, 0.8, "Expression Tower", ha="center", fontsize=10,
            color="#375623", fontweight="bold")
    ax.text(9.65, 3.9, "Downstream:", ha="center", fontsize=9,
            color="#555", fontstyle="italic")
    for i, txt in enumerate(["• Region retrieval",
                              "• Expression → SAM prompt",
                              "• Linear probe (Dice)"]):
        ax.text(9.65, 3.5 - i*0.35, txt, ha="center", fontsize=8,
                color="#333")

    ax.set_title("Proposed Dual-Encoder Architecture (Future Direction)",
                 fontsize=11, fontweight="bold", pad=6)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

def make_tile_schematic(path):
    """Sketch of the tile-based inference strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: tile grid on H&E
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8F0F0")
    # Draw pseudo H&E background
    bg = mpatches.FancyBboxPatch((0.1, 0.1), 5.8, 9.8, boxstyle="square",
        facecolor="#F0C8C8", edgecolor="gray", lw=1)
    ax.add_patch(bg)
    # Draw tile grid (3×5)
    colors = plt.cm.tab10.colors
    ci = 0
    for row in range(5):
        for col in range(3):
            x = 0.1 + col * (5.8/3)
            y = 0.1 + row * (9.8/5)
            rect = mpatches.FancyBboxPatch((x+0.05, y+0.05),
                5.8/3 - 0.1, 9.8/5 - 0.1,
                boxstyle="square",
                facecolor=colors[ci % 10] + (0.15,),
                edgecolor=colors[ci % 10], lw=1.5, linestyle="--")
            ax.add_patch(rect)
            ax.text(x + 5.8/6, y + 9.8/10, f"tile {row*3+col+1}",
                    ha="center", va="center", fontsize=6, color="black")
            ci += 1
    ax.set_title("28 tiles (1024×1024 px, 128 px overlap)", fontsize=9)
    ax.axis("off")

    # Right: point sampling
    ax2 = axes[1]
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_facecolor("#F0F0F0")
    # GT mask shape (irregular polygon)
    poly_x = [1.0, 2.5, 3.8, 4.0, 3.5, 2.0, 0.8, 1.0]
    poly_y = [1.5, 0.8, 1.2, 2.5, 3.8, 4.2, 3.5, 1.5]
    ax2.fill(poly_x, poly_y, color="#A8C8E8", alpha=0.5, label="GT mask")
    ax2.plot(poly_x, poly_y, "b-", lw=1.5)
    # Positive points (inside)
    np.random.seed(42)
    for _ in range(10):
        px, py = np.random.uniform(1.2, 3.5), np.random.uniform(1.5, 3.5)
        ax2.plot(px, py, "g*", ms=10, zorder=5)
    # Negative points (outside, near bbox)
    for px, py in [(0.3,0.5),(4.5,0.5),(4.6,3.0),(0.4,4.0),(2.5,4.8)]:
        ax2.plot(px, py, "rx", ms=10, mew=2, zorder=5)
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor="#A8C8E8", label="GT mask (region label)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="g", ms=10, label="10 positive points"),
        Line2D([0],[0], marker="x", color="r", ms=10, mew=2, label="5 negative points"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=7)
    ax2.set_title("Oracle prompt sampling per tile", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(2.5, 5.2, "⚠ Oracle: GT used for BOTH prompt & evaluation",
             ha="center", fontsize=8, color="red", fontweight="bold",
             transform=ax2.transData)

    plt.suptitle("Tile-Based SAM3 Inference Strategy", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

# ---------------------------------------------------------------------------
# Pre-generate asset images
# ---------------------------------------------------------------------------
print("Generating asset images...")

TILE_SCHEMATIC = ASSETS / "tile_schematic.png"
make_tile_schematic(TILE_SCHEMATIC)

ARCH_DIAGRAM = ASSETS / "dual_encoder_arch.png"
make_arch_diagram(ARCH_DIAGRAM)

SAM3_INTERVENTION = ASSETS / "sam3_intervention_diagram.png"
make_sam3_intervention_diagram(SAM3_INTERVENTION)

RAG_DIAGRAM_PATH = ASSETS / "rag_diagram.png"
make_rag_diagram(RAG_DIAGRAM_PATH)

# VisiumHD results table
EXP1_TABLE = ASSETS / "exp1_results_table.png"
make_table_img(
    headers=["Label", "SAM3\nDice", "MedSAM3\nDice", "SAM3\nPrecision",
             "MedSAM3\nPrecision", "SAM3\nRecall", "MedSAM3\nRecall"],
    rows=[
        ["Lung Bronchiola",            "0.080", "0.470", "0.050", "0.947", "0.203", "0.313"],
        ["Erythorocytes",              "0.074", "0.089", "0.039", "0.062", "0.666", "0.152"],
        ["Immune Infiltration",        "0.071", "0.162", "0.038", "0.126", "0.722", "0.229"],
        ["Lung Alveoli (normal adj.)", "0.229", "0.053", "0.177", "0.612", "0.324", "0.028"],
        ["Lung Vessels",               "0.273", "0.350", "0.167", "0.358", "0.752", "0.342"],
        ["Pigment",                    "0.002", "0.042", "0.001", "0.023", "0.117", "0.210"],
        ["Stroma",                     "0.382", "0.225", "0.259", "0.307", "0.726", "0.178"],
        ["Tumor",                      "0.351", "0.146", "0.239", "0.267", "0.658", "0.101"],
    ],
    path=EXP1_TABLE,
    highlight_cols=[1, 2],
    title="VisiumHD Exp1 — SAM3 vs MedicalSAM3 Oracle Multipoint Results",
    figsize=(12, 4.5),
)

# Tile size ablation table
TILE_TABLE = ASSETS / "tile_ablation_table.png"
make_table_img(
    headers=["Label", "1024 Dice", "1536 Dice", "1024 Recall", "1536 Recall",
             "1024 Precision", "1536 Precision"],
    rows=[
        ["Lung Bronchiola", "0.080", "0.079", "0.203", "0.994", "0.050", "0.041"],
        ["Lung Vessels",    "0.232", "0.176", "0.923", "0.810", "0.133", "0.099"],
        ["Stroma",          "0.382", "0.325", "0.726", "0.823", "0.259", "0.202"],
        ["Tumor",           "0.364", "0.323", "0.663", "0.801", "0.250", "0.202"],
    ],
    path=TILE_TABLE,
    highlight_cols=[1, 2],
    title="Tile Size Ablation (SAM3-base, oracle multipoint prompts)",
    figsize=(10, 2.8),
)

# TMA24 results table
TMA_TABLE = ASSETS / "tma24_results_table.png"
make_table_img(
    headers=["Model", "Prompt Type", "Label", "Dice"],
    rows=[
        ["Medical-SAM3", "Whole-image box", "Hyalinized granuloma", "0.711"],
        ["Base SAM3",    "Whole-image box", "Mixed alveoli",        "0.653"],
        ["Medical-SAM3", "Whole-image box", "Mixed alveoli",        "0.575"],
        ["Base SAM3",    "Whole-image box", "Granuloma border",     "0.398"],
        ["Medical-SAM3", "Whole-image box", "Granuloma border",     "0.392"],
        ["Any",          "Text prompt",     "All labels",           "≈ 0.000"],
        ["Base SAM3",    "Cross-region transfer", "All labels",     "≈ 0.000"],
        ["Base SAM3",    "Dense box (oracle best)", "Hyalinized",   "0.499"],
        ["Base SAM3",    "Dense box (oracle best)", "Alveoli",      "0.471"],
    ],
    path=TMA_TABLE,
    highlight_cols=[3],
    title="TMA24 (Silicosis) — SAM3 / MedicalSAM3 Results Summary",
    figsize=(9, 3.5),
)

# Cross-dataset summary table
SUMMARY_TABLE = ASSETS / "cross_dataset_summary.png"
make_table_img(
    headers=["Finding", "TMA24 (Silicosis)", "VisiumHD Exp1 (Lung Cancer)"],
    rows=[
        ["Text prompts",           "≈ 0.000",               "N/A (skipped)"],
        ["Cross-region transfer",  "≈ 0.000",               "N/A (not tested)"],
        ["Best oracle Dice",       "0.711 (box prompt)",    "0.470 (multipoint)"],
        ["Larger tile/context",    "N/A",                   "Dice ↓ (recall ↑, precision ↓)"],
        ["SAM3 behavior",          "Oversegmentation",      "Oversegmentation"],
        ["MedSAM3 behavior",       "More conservative",     "High precision, low recall"],
    ],
    path=SUMMARY_TABLE,
    figsize=(11, 3.0),
    title="Cross-Dataset Summary",
)

print("Assets generated.")

# ---------------------------------------------------------------------------
# Build Presentation
# ---------------------------------------------------------------------------
prs = new_prs()

# ============================================================
# SLIDE 1 — Title
# ============================================================
sl = blank_slide(prs)
add_rect(sl, 0, 0, W, H, fill=C["title"])
add_rect(sl, 0, Inches(2.5), W, Inches(2.8), fill=RGBColor(0x2E, 0x75, 0xB6))

add_text(sl, "SAM3 for Pathology Region Segmentation",
         Inches(0.5), Inches(2.65), Inches(12.3), Inches(1.0),
         size=36, bold=True, color=C["white"], align=PP_ALIGN.CENTER)
add_text(sl, "From TMA24 Silicosis to VisiumHD Lung Cancer",
         Inches(0.5), Inches(3.6), Inches(12.3), Inches(0.6),
         size=22, color=C["light"], align=PP_ALIGN.CENTER)
add_text(sl, f"Haoran Wu  ·  Yan Lab  ·  {date.today().strftime('%B %Y')}",
         Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.5),
         size=14, color=RGBColor(0xAA, 0xC4, 0xDF), align=PP_ALIGN.CENTER)

set_notes(sl, """\
Good morning / Good afternoon everyone. Today I'm going to walk you through the work I've been doing over the past few months on applying SAM3 — the Segment Anything Model — to pathology tissue region segmentation in the context of spatial transcriptomics.

The talk covers two datasets: TMA24, which is a silicosis TMA from our lab, and a 10x Genomics VisiumHD human lung cancer sample called Exp1. I'll go through the experimental methods in detail, show you the results, and then discuss what the findings mean and where I think we should go next.

I'll try to keep things honest — some of the results are more encouraging than others, and I want to make sure the interpretation is clear so we can have a good discussion at the end.
""")

# ============================================================
# SLIDE 2 — Background & Motivation
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Background & Motivation", "Why SAM3 for Spatial Transcriptomics?")

add_bullets(sl, [
    "Goal: automatically segment semantically-defined tissue regions from H&E, linked to spatial gene expression",
    "Why SAM3?  Foundation model for image segmentation — zero-shot, prompt-driven, pre-trained on 1B+ masks",
    "Related work — IAMSAM (Lee et al., Genome Biology 2024):",
    "  Uses SAM to segment H&E → user selects ROIs → downstream DEG / enrichment / cell-type analysis",
    "  Two modes: everything-mode (auto masks) or prompt-mode (user draws bounding boxes)",
    "  Key limitation: semi-automatic — requires human-in-the-loop; no gene expression to guide segmentation",
    "Our question: can SAM3 automatically recover pathology-defined semantic regions?",
    "  And ultimately: can gene expression replace or augment manual prompts?",
    "Two datasets tested: TMA24 (silicosis, 3 labels) and VisiumHD Exp1 (lung cancer, 8 labels)",
], Inches(0.35), Inches(1.15), Inches(8.0), Inches(5.9), size=15)

# IAMSAM figure placeholder (use paper figure if available, else note)
iamsam_fig = BASE / "IAMSAM_fig1.png"
if iamsam_fig.exists():
    add_image(sl, iamsam_fig, Inches(8.5), Inches(1.2), Inches(4.5))
else:
    add_rect(sl, Inches(8.5), Inches(1.2), Inches(4.5), Inches(4.5),
             fill=RGBColor(0xEE, 0xEE, 0xEE))
    add_text(sl, "IAMSAM\nFig. 1\n(Lee et al.,\nGenome Biology 2024)",
             Inches(8.6), Inches(2.5), Inches(4.3), Inches(1.5),
             size=12, color=C["gray"], align=PP_ALIGN.CENTER)

set_notes(sl, """\
So let me start with some motivation. The core problem we're trying to solve is: given an H&E-stained tissue image that comes paired with spatial transcriptomics data, can we automatically identify and delineate biologically meaningful tissue regions — like tumor, stroma, or immune infiltration — without requiring a pathologist to manually annotate every sample?

The reason we looked at SAM3 is that it's arguably the most capable general-purpose segmentation foundation model available. It was trained on over a billion masks from 11 million images, so it has very strong zero-shot segmentation abilities. The medical variant, MedicalSAM3, was further adapted on biomedical imaging data, so in theory it should understand tissue patterns better.

Now, a very relevant piece of prior work is IAMSAM, published in Genome Biology in 2024. This is a web tool that uses SAM to segment H&E images from 10x Visium spatial transcriptomics data. The user can either run SAM in an automatic everything-mode, or draw bounding boxes as prompts, and then the tool automatically extracts gene expression profiles from the segmented regions and runs downstream analysis like DEG testing and enrichment. It's a nice tool and it does work — but critically, it's semi-automatic. It still requires the user to look at the image and draw the boxes, or at least select from the automatically generated masks. And it doesn't use any gene expression information to actually guide the segmentation.

Our question is more ambitious: can we do this automatically, without human input, and eventually use the molecular signal itself to drive the segmentation? That's the direction we're heading toward, and today's results are an important step in understanding where the current bottlenecks are.
""")

# ============================================================
# SLIDE 3 — VisiumHD Exp1: Dataset
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "VisiumHD Exp1: Dataset", "10x Genomics VisiumHD Human Lung Cancer")

add_bullets(sl, [
    "H&E image: 3,524 × 6,000 px (tissue_hires_image.png)",
    "Expert GeoJSON annotations: 8 tissue region labels, 3,821 polygon features",
    "8 μm bin resolution: 448,109 in-tissue bins",
    "Gene count: 18,085 (full transcriptome — not a targeted panel)",
    "Priority labels for modeling: tumor, stroma, immune infiltration",
], Inches(0.35), Inches(1.15), Inches(5.8), Inches(3.5), size=16)

# Region bin count table (inline)
add_rect(sl, Inches(0.35), Inches(4.3), Inches(5.8), Inches(2.8),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
rows_data = [
    ("Tumor",                      "152,941"),
    ("Stroma",                     "152,765"),
    ("Lung vessels",               "29,982"),
    ("Lung alveoli (normal adj.)", "29,093"),
    ("Immune infiltration",        "22,184"),
    ("Lung Bronchiola",            "18,744"),
    ("Erythorocytes",              "14,250"),
    ("Pigment",                    "686"),
]
add_text(sl, "Region", Inches(0.45), Inches(4.35), Inches(3.5), Inches(0.3),
         size=11, bold=True, color=C["title"])
add_text(sl, "Bins", Inches(4.1), Inches(4.35), Inches(1.8), Inches(0.3),
         size=11, bold=True, color=C["title"], align=PP_ALIGN.RIGHT)
for i, (label, count) in enumerate(rows_data):
    y = Inches(4.65 + i * 0.27)
    bg = RGBColor(0xE8, 0xF2, 0xFC) if i % 2 == 0 else C["white"]
    add_rect(sl, Inches(0.35), y, Inches(5.8), Inches(0.27), fill=bg)
    add_text(sl, label, Inches(0.45), y, Inches(3.5), Inches(0.27), size=11)
    add_text(sl, count, Inches(4.1), y, Inches(1.9), Inches(0.27),
             size=11, align=PP_ALIGN.RIGHT)

# GeoJSON overlay image
geojson_img = FIG.get("exp1_all_panels")  # fallback to panels if no geojson overlay
# Try to find a combined geojson overlay first
for candidate in [
    BASE / "output/visium_hd_exp1/region_masks/overlays/combined_overlay.png",
    BASE / "output/visium_hd_exp1/region_masks/all_regions_overlay.png",
]:
    if candidate.exists():
        geojson_img = candidate
        break

add_image(sl, FIG["exp1_all_panels"], Inches(6.4), Inches(1.1), Inches(6.6))

set_notes(sl, """\
Let me tell you about the VisiumHD Exp1 dataset — this is the main new data I've been working with.

VisiumHD is 10x Genomics' newest spatial transcriptomics platform. Unlike the older Visium which has spots at 55 micron resolution, VisiumHD captures gene expression at 8 micron bin resolution — which is much closer to the resolution needed to distinguish individual cell types. Each 8-micron bin corresponds to roughly one to a few cells.

This specific sample is human lung cancer tissue. The H&E image is 3,524 by 6,000 pixels in the hires coordinate system. Critically for our work, the sample comes with expert GeoJSON annotations — these are polygon annotations made by a pathologist covering 8 distinct tissue region types, with 3,821 individual polygons total. You can see the annotation overlay on the right — the different colors represent different region types.

There are a total of 448,109 in-tissue bins at 8 micron resolution. The gene expression data contains 18,085 genes — this is the full transcriptome, which is much richer than a targeted panel. For comparison, our TMA24 Xenium data only has 479 genes.

The label distribution is interesting. Tumor and stroma are by far the largest, each with around 150,000 bins. Immune infiltration has about 22,000 bins. These three — tumor, stroma, and immune infiltration — are the priority labels because they're biologically most relevant for cancer biology, they're large enough to train on, and they're the most visually distinct in H&E.

I want to emphasize that these GeoJSON annotations are expert-drawn. They define regions not just based on visual appearance but based on pathological criteria — which is exactly why this is a hard problem for SAM, as we'll see.
""")

# ============================================================
# SLIDE 4 — VisiumHD Exp1: Experimental Methods
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "VisiumHD Exp1: Experimental Methods",
           "GeoJSON → Masks → Tile-Based SAM3 Oracle Baseline")

add_bullets(sl, [
    "Stage 1 — GeoJSON → Binary Masks:",
    "  Rasterize each polygon label onto tissue_hires_image.png coordinate space",
    "  Transform: x_hires = x_geojson × 0.13752  (no y-flip needed)",
    "  Output: one binary PNG per region label",
    "Stage 2–3 — Expression Pipeline:",
    "  Export 448K bin metadata (spatial position + region label) from parquet file",
    "  Normalize 18K-gene expression matrix: log1p → z-score per gene",
    "SAM3 Oracle Baseline (Slide 5):",
    "  Tile-based inference: 1024 × 1024 px tiles, 128 px overlap → 28 tiles",
    "  Per tile: sample 10 positive points from GT mask foreground, 5 negative outside",
    "  Run SAM3 / MedicalSAM3 per tile → stitch predictions → compute Dice vs GT",
    "  ⚠ Oracle design: GT mask used BOTH to generate prompts AND as evaluation target",
    "  → Measures upper-bound capability, NOT automatic region discovery",
], Inches(0.35), Inches(1.15), Inches(6.5), Inches(5.9), size=14)

add_image(sl, TILE_SCHEMATIC, Inches(6.7), Inches(1.1), Inches(6.4))

set_notes(sl, """\
Now let me walk through the experimental setup in detail, because I think it's really important to understand exactly what we're measuring here.

The first step is turning the GeoJSON annotations into binary masks we can use. We wrote a script that rasterizes each polygon onto the hires image coordinate space. The coordinate transform is straightforward — multiply by the scale factor 0.137 — and importantly, there's no y-axis flip needed, because Space Ranger uses image coordinates with the top-left as origin. We confirmed this by visually comparing our rendered masks against the official Space Ranger annotation overlay.

The output is one binary PNG per region label, same dimensions as the hires image. These masks are what we use both for prompt generation and for evaluation.

The pipeline also includes exporting the expression data — 448K bins, each with its hires x/y position, region label assignment, and 18K-dimensional log1p normalized expression vector. But that's more relevant to the future directions, so I won't dwell on it now.

For the SAM3 experiment, the main challenge is that this image is 3,524 by 6,000 pixels — way too large to feed into SAM3 directly at full resolution. SAM3 expects images around 1024 pixels on the longest side. If we just downscale the whole image to 1024 pixels, we lose a lot of fine-grained detail. So we do tile-based inference: we split the image into 1024 by 1024 pixel tiles with 128-pixel overlap, which gives us 28 tiles. Each tile is processed independently by SAM3, and then we stitch the predictions back together using overlap averaging.

Now, about the oracle design — this is the most important thing to understand about this experiment. For each tile, we take the ground truth binary mask, sample 10 random positive points from the foreground pixels, and 5 negative points from near the bounding box but outside the mask. We give these points to SAM3 as the prompt, and then compare SAM3's output mask against the same ground truth mask.

This is an oracle setup. We're using the answer to generate the question. This means we're measuring the best possible performance of SAM3 — if it can't recover the region even when you tell it exactly where the region is, that tells us something fundamental about the model's capabilities.

The diagram on the right shows this visually — the tile grid on the left, and on the right the point sampling strategy where green stars are positive points inside the GT mask and red crosses are negative points outside.
""")

# ============================================================
# SLIDE 5 — VisiumHD Exp1: Results
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "VisiumHD Exp1: SAM3 Oracle Baseline Results",
           "Even Oracle Prompts Fail to Reliably Recover Semantic Regions")

add_image(sl, EXP1_TABLE, Inches(0.2), Inches(1.1), Inches(8.5))

# Key finding boxes
add_rect(sl, Inches(8.8), Inches(1.15), Inches(4.3), Inches(1.3),
         fill=RGBColor(0xFF, 0xEB, 0x9C))
add_text(sl, "SAM3 (aggressive)\nHigh recall, low precision\n→ oversegmentation",
         Inches(8.9), Inches(1.2), Inches(4.1), Inches(1.2), size=13)

add_rect(sl, Inches(8.8), Inches(2.55), Inches(4.3), Inches(1.3),
         fill=RGBColor(0xC6, 0xEF, 0xCE))
add_text(sl, "MedicalSAM3 (conservative)\nHigh precision, low recall\n→ undersegmentation",
         Inches(8.9), Inches(2.6), Inches(4.1), Inches(1.2), size=13)

add_rect(sl, Inches(8.8), Inches(3.95), Inches(4.3), Inches(1.1),
         fill=RGBColor(0xFF, 0xC7, 0xCE))
add_text(sl, "Best Dice overall:\nStroma 0.382 (SAM3)\nBronchiola 0.470 (MedSAM3)",
         Inches(8.9), Inches(4.0), Inches(4.1), Inches(1.0), size=13)

add_image(sl, FIG["exp1_all_overlays"], Inches(0.15), Inches(5.3), Inches(12.8))

set_notes(sl, """\
Here are the results. The table shows all 8 region labels with their Dice, IoU, Precision, and Recall for both SAM3 and MedicalSAM3.

Let me point out the two contrasting patterns. SAM3-base tends to be aggressive — it has high recall, meaning it finds most of the pixels that are in the ground truth mask, but it also predicts a lot of pixels that are not in the mask, so precision is low. This leads to oversegmentation. You can see this pattern clearly for stroma and tumor, where SAM3 recall is 0.73 and 0.66 respectively, but precision is only around 0.25.

MedicalSAM3 behaves the opposite way — it's much more conservative. For Lung Bronchiola, MedSAM3 achieves precision of 0.947 — meaning almost everything it predicts is correct — but recall is only 0.313. It finds a small, high-confidence part of the bronchiola but misses most of it. This leads to undersegmentation.

The best Dice scores are stroma 0.382 from SAM3 and bronchiola 0.470 from MedicalSAM3. These are mediocre at best. For a typical medical image segmentation task you'd want Dice above 0.7 or 0.8.

And remember — these are oracle results. We're giving SAM3 points sampled directly from the ground truth mask. This is the best-case scenario for point-prompted SAM3 on this data, and the performance is still weak.

The prediction overlay grid at the bottom shows this visually. Each row is a different region label, left column is SAM3 and right is MedSAM3. The dark green is the prediction and you can see the block-like tile artifacts in many of the predictions.

The implication is clear: even with favorable prompts, SAM3-style segmentation doesn't map well onto pathology-defined semantic tissue regions. This is a fundamental observation, not just a tuning issue.
""")

# ============================================================
# SLIDE 6 — Tile Size Ablation
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "VisiumHD Exp1: Tile Size Ablation",
           "Larger Context Does Not Fix the Mismatch")

add_bullets(sl, [
    "Question: are tile boundary artifacts causing the weak results?",
    "Tested three tile sizes: 1024/128 overlap · 1536/256 · 2048/384",
    "4-label subset, SAM3-base, same oracle multipoint prompts",
    "Consistent pattern: larger tile → Recall ↑, Precision ↓, Dice ↓",
    "Interpretation: larger field of view encourages broader mask expansion,",
    "  not better semantic boundary understanding",
    "Conclusion: tile size is NOT the bottleneck",
], Inches(0.35), Inches(1.15), Inches(7.2), Inches(3.5), size=16)

add_image(sl, TILE_TABLE, Inches(0.3), Inches(4.55), Inches(7.5))

add_rect(sl, Inches(7.7), Inches(1.15), Inches(5.4), Inches(5.9),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
add_text(sl, "Why bigger tiles don't help:", Inches(7.85), Inches(1.3),
         Inches(5.1), Inches(0.4), size=14, bold=True, color=C["title"])
add_bullets(sl, [
    "SAM3 sees more context → expands the predicted mask further",
    "But semantic region boundaries aren't defined by visual extent — they're defined by pathology criteria",
    "Bronchiola: recall jumps from 0.20 → 0.99 with 1536 tiles, but Dice barely changes — it covers everything",
    "Model is predicting visual objects (e.g. airway lumen), not semantic labels (e.g. 'bronchiola as annotated by expert')",
], Inches(7.85), Inches(1.75), Inches(5.1), Inches(4.8), size=13)

set_notes(sl, """\
One obvious hypothesis for why SAM3 performs poorly is that the 1024-pixel tiles create artificial boundaries. When you cut the image into tiles, each tile is processed independently, and SAM3 doesn't know about the context outside that tile. This could lead to blocky artifacts and inconsistent predictions across tile boundaries.

So we ran a tile size ablation. We tested three tile sizes: 1024 with 128 overlap, 1536 with 256 overlap, and 2048 with 384 overlap, on a subset of 4 labels.

The result is shown in the table. Moving from 1024 to 1536 consistently decreases Dice — for stroma from 0.382 to 0.325, for tumor from 0.364 to 0.323. Recall tends to increase, but precision drops even more, so the net effect on Dice is negative.

The bronchiola case is particularly illuminating. At 1536, recall shoots up from 0.20 to 0.99 — SAM3 is now predicting almost the entire image as bronchiola. But Dice barely changes because precision collapses. The model isn't learning a better definition of bronchiola, it's just predicting more and more aggressively.

This tells us that tile size is not the fundamental problem. The core issue is that SAM-style segmentation is designed to find visually coherent objects — like a lumen, a gland, a nucleus. But pathology region labels like 'stroma' or 'tumor' are defined by expert criteria that go beyond visual coherence. Two pixels right next to each other might have very different region labels based on the pathologist's interpretation.

This is the model-task mismatch we keep coming back to.
""")

# ============================================================
# SLIDE 7 — TMA24: Dataset & Methods
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "TMA24 (Silicosis): Dataset & Experimental Methods",
           "Single H&E Image, 3 Region Labels, 4 Prompt Strategies")

add_image(sl, FIG["tma24_legend"], Inches(0.2), Inches(1.1), Inches(4.8))

add_bullets(sl, [
    "Dataset: Silicosis TMA, single H&E image (example1.jpg)",
    "3 region labels:",
    "  Mixed alveoli (normal-appearing lung)",
    "  Granuloma border (inflammatory border)",
    "  Hyalinized granuloma (dense fibrotic core)",
    "Ground truth: pseudo-masks from expert spatial alignment",
    "",
    "4 experimental strategies tested:",
], Inches(5.2), Inches(1.15), Inches(7.8), Inches(3.8), size=15)

# Strategy table
strategies = [
    ("Whole-image box prompt",     "Bounding box of GT region, full image input"),
    ("Text prompt",                "Label name as text query (e.g. 'mixed alveoli')"),
    ("Cross-region transfer",      "Prompt from left half → evaluate on right half"),
    ("Dense box proposals (oracle)","Grid of 256×256 boxes, oracle-best proposal selected"),
]
add_rect(sl, Inches(5.2), Inches(4.55), Inches(7.85), Inches(2.65),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
add_text(sl, "Strategy", Inches(5.3), Inches(4.6), Inches(3.0), Inches(0.3),
         size=11, bold=True, color=C["title"])
add_text(sl, "Description", Inches(8.4), Inches(4.6), Inches(4.5), Inches(0.3),
         size=11, bold=True, color=C["title"])
for i, (s, d) in enumerate(strategies):
    y = Inches(4.92 + i * 0.55)
    bg = RGBColor(0xEB, 0xF3, 0xFA) if i % 2 == 0 else C["white"]
    add_rect(sl, Inches(5.2), y, Inches(7.85), Inches(0.55), fill=bg)
    add_text(sl, s, Inches(5.3), y + Inches(0.05), Inches(3.0), Inches(0.5), size=11, bold=True)
    add_text(sl, d, Inches(8.4), y + Inches(0.05), Inches(4.5), Inches(0.5), size=11)

set_notes(sl, """\
Now let me shift to the TMA24 dataset, which is where this whole project started.

This is a silicosis TMA — tissue microarray — from our lab. Silicosis is an occupational lung disease caused by inhaling silica dust, and it leads to characteristic histological patterns including granuloma formation. We have a single H&E image, example1.jpg, with 3 annotated region labels. You can see the annotation overlay on the left — the pseudo-masks showing mixed alveoli, granuloma border, and hyalinized granuloma.

The ground truth here comes from expert spatial alignment of the pathologist annotations, so these are high-quality masks. There are three labels: mixed alveoli, which looks like relatively normal-appearing lung tissue; granuloma border, which is the inflammatory rim around the granuloma; and hyalinized granuloma, which is the dense fibrotic core.

For this dataset, we tried four different strategies. The whole-image box prompt is straightforward — take the bounding box of the target region, give it to SAM3 as a box prompt, and see what it segments. Text prompt uses the label name as a text query through SAM3's CLIP-based text encoder. Cross-region transfer is the most interesting generalization test — we prompt SAM3 using the left half of the image, and then evaluate whether it can find the same structure in the right half, without any additional prompts. And dense box proposals is an oracle-style sweep where we cover the image with a grid of 256×256 boxes and pick the one with the best Dice.

Each of these tests something different about SAM3's capabilities.
""")

# ============================================================
# SLIDE 8 — TMA24: Results
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "TMA24 (Silicosis): Results",
           "Box Prompts Work In-Region; Text ≈ 0; Cross-Region Transfer Fails")

add_image(sl, TMA_TABLE, Inches(0.2), Inches(1.1), Inches(7.2))

add_image(sl, FIG["tma24_granuloma_box"], Inches(7.5), Inches(1.1), Inches(2.7))
add_image(sl, FIG["tma24_alveoli_box"],   Inches(10.3), Inches(1.1), Inches(2.7))

add_text(sl, "MedSAM3, hyalinized granuloma, Dice=0.711",
         Inches(7.5), Inches(3.4), Inches(2.7), Inches(0.4), size=9, color=C["gray"])
add_text(sl, "SAM3, mixed alveoli, Dice=0.653",
         Inches(10.3), Inches(3.4), Inches(2.7), Inches(0.4), size=9, color=C["gray"])

add_rect(sl, Inches(0.2), Inches(4.8), Inches(12.9), Inches(2.4),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
add_bullets(sl, [
    "Box prompts: usable signal when prompted in the same region (up to Dice 0.711)",
    "Text prompts: Dice ≈ 0 for all labels — SAM3's CLIP text encoder has no histology prior",
    "Cross-region transfer: Dice ≈ 0 — the model cannot generalize spatially without new prompts",
    "Dense box oracle-best: finds partial structures (0.363–0.499) but requires oracle selection",
], Inches(0.35), Inches(4.9), Inches(12.6), Inches(2.2), size=15)

set_notes(sl, """\
Here are the TMA24 results. Let me go through the key findings one by one.

Starting with whole-image box prompts: this is the best-performing condition. MedicalSAM3 achieves Dice 0.711 on hyalinized granuloma, and SAM3-base gets 0.653 on mixed alveoli. These are actually decent numbers — not perfect, but usable. The granuloma result makes intuitive sense: hyalinized granuloma is a visually very distinct structure with sharp boundaries and a characteristic dense, pinkish appearance, so it's the kind of thing SAM is good at — it looks like a clear, coherent object.

Now, text prompts. Dice approximately zero for every label, every model. This is not surprising in retrospect — SAM's text encoder is CLIP, which was trained on natural image captions, not histopathology descriptions. When you ask it to find 'mixed alveoli' or 'granuloma border', it has no idea what those are in an H&E context. This rules out text as a useful modality for this type of data.

Cross-region transfer is the most important test for generalization, and the result is zero. When we prompt with the left half of the image and evaluate on the right half, SAM3 completely fails to transfer. This means the model is essentially doing interactive object segmentation — it finds the thing near the prompt, but it has no general concept of what 'granuloma border' looks like that it can apply elsewhere.

The dense box proposals are an oracle test — we're cheating by picking the best proposal. Even with oracle selection, the best we get is 0.499 for hyalinized granuloma. So even exhaustively sweeping the image with proposals, we can't reliably recover the ground truth regions.

The figures on the right show the best-case box prompt results for context.
""")

# ============================================================
# SLIDE 9 — Cross-Dataset Discussion
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Cross-Dataset Discussion",
           "Two Datasets, Same Fundamental Limitation")

# Left col
add_rect(sl, Inches(0.2), Inches(1.1), Inches(6.0), Inches(5.9),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
add_text(sl, "Consistent Findings (TMA24 + VisiumHD Exp1)",
         Inches(0.3), Inches(1.15), Inches(5.8), Inches(0.4),
         size=14, bold=True, color=C["title"])
add_bullets(sl, [
    "Text prompts: Dice ≈ 0 on both datasets",
    "Cross-region generalization: Dice ≈ 0",
    "Oracle prompts: weak Dice — even the upper bound is low",
    "SAM3: oversegments (high recall, low precision)",
    "MedSAM3: undersegments (high precision, low recall)",
    "Larger tile context: Dice does not improve",
], Inches(0.3), Inches(1.6), Inches(5.8), Inches(3.0), size=14)

add_text(sl, "Two Failure Modes Identified:",
         Inches(0.3), Inches(4.65), Inches(5.8), Inches(0.4),
         size=14, bold=True, color=C["red"])
add_bullets(sl, [
    "Prompt quality: sparse points/boxes are insufficient for large, irregular semantic regions",
    "Model-task mismatch: SAM 'object segmentation' ≠ pathology 'semantic tissue region'",
], Inches(0.3), Inches(5.1), Inches(5.8), Inches(1.7), size=13)

# Right col
add_rect(sl, Inches(6.4), Inches(1.1), Inches(6.7), Inches(5.9),
         fill=RGBColor(0xFF, 0xF5, 0xE6))
add_text(sl, "Why Does IAMSAM 'Work'?",
         Inches(6.5), Inches(1.15), Inches(6.5), Inches(0.4),
         size=14, bold=True, color=C["title"])
add_bullets(sl, [
    "IAMSAM uses human-drawn box prompts → SAM segments the visually obvious object",
    "No requirement for automatic semantic label recovery",
    "No cross-region generalization test — users draw boxes in each new image",
    "Our setting is strictly harder:",
    "  No human in the loop",
    "  Must recover pathologist-defined semantic labels automatically",
    "  Prompts must come from molecular data, not visual inspection",
    "",
    "IAMSAM shows SAM can do fine-grained segmentation when prompted well",
    "  → The architecture is capable; the bottleneck is prompt quality + semantic understanding",
], Inches(6.5), Inches(1.6), Inches(6.5), Inches(5.2), size=13)

set_notes(sl, """\
This slide is where I want to synthesize what we've learned across both datasets.

The pattern is remarkably consistent. In both TMA24 and VisiumHD Exp1, text prompts completely fail. Cross-region generalization is essentially zero in both cases. And even our oracle experiments, where we use the ground truth to generate the prompts, give us weak Dice scores.

Now, I want to be careful about how we interpret this in the context of IAMSAM, because you might reasonably ask: if IAMSAM works with SAM for spatial transcriptomics, why doesn't ours?

The key difference is that IAMSAM is semi-automatic. The user looks at the image, identifies the region of interest visually, and either draws a bounding box around it or selects from automatically generated masks. SAM then segments the visually obvious object defined by that prompt. This is a much easier task than what we're asking SAM3 to do.

In our setup, we want to automatically find pathology-defined semantic regions across the entire slide, without any human input. Cross-region generalization is the critical test, and that fails completely.

However — and this is important — IAMSAM's success does tell us that the SAM architecture is fundamentally capable. When it gets a good prompt that's close to the actual region, it can produce a reasonable segmentation. The problem is: where do the good prompts come from? In IAMSAM, they come from a human looking at the image. In our case, we want them to come from the molecular data.

I see two distinct bottlenecks. The first is prompt quality — we need prompts that carry semantic information about tissue biology, not just spatial location. The second is model-task mismatch — SAM was designed for object segmentation, not semantic tissue labeling. These are related but separate problems, and addressing them requires different approaches.
""")

# ============================================================
# SLIDE 10 — Future Directions: Overview
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Future Directions: Overview",
           "Five Ways to Intervene in the SAM3 Pipeline")

add_image(sl, SAM3_INTERVENTION, Inches(0.1), Inches(1.05), Inches(13.15))

set_notes(sl, """\
Now let me talk about where we go from here. The key insight from all the experiments is that there are really two separate problems: one is the prompt quality problem — how do we get good semantic prompts without a human in the loop — and the other is the model-task mismatch problem — even with good prompts, SAM may not understand pathology semantic regions.

This diagram shows the SAM3 pipeline from left to right: the Image Encoder takes a tile and produces image features, the Prompt Encoder takes points, boxes, text, or exemplar patches and converts them to prompt embeddings, and the Mask Decoder combines these to produce the output mask.

The five future directions I'm proposing each intervene at a different point in this pipeline.

Direction A — supervised fine-tuning — wraps the Image Encoder and Mask Decoder in orange dashed boxes. Fine-tuning these two components on our GeoJSON masks is the most direct way to ask: can this architecture learn our region labels at all when trained properly?

Direction B — GigaPath backbone — also wraps the Image Encoder, but the change is to replace the ViT weights with Prov-GigaPath, a pathology-domain vision transformer. This should give better H&E representations without changing the overall architecture.

Direction C — shown in green — is the gene expression path. An expression MLP encoder sits above the pipeline and feeds into the Prompt Encoder directly. Instead of a human drawing a box, the expression vector for a region generates the prompt embedding.

Direction D — shown in purple — is the RAG direction, which I'll explain in detail on the next slide. The basic idea is to build a reference database of annotated patches, and at inference time retrieve the most similar annotated examples as exemplar prompts.

Direction E — the red dashed rectangle around the entire pipeline — represents replacing SAM3 entirely with a pathology-specific segmentation model like CellViT or HoVerNet that was trained on histopathology annotations.

I'll now go through Directions C and D in more detail since those are the most novel parts of our plan.
""")

# ============================================================
# SLIDE 11 — Direction C: Gene Expression Alignment
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Direction C: Gene Expression → Prompt Encoder",
           "Dual-Encoder Alignment for Cross-Modal Retrieval & Prompting")

add_image(sl, ARCH_DIAGRAM, Inches(0.2), Inches(1.1), Inches(9.0))

add_rect(sl, Inches(9.3), Inches(1.1), Inches(3.85), Inches(5.9),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
add_text(sl, "Why this can work:",
         Inches(9.4), Inches(1.2), Inches(3.6), Inches(0.35),
         size=13, bold=True, color=C["title"])
add_bullets(sl, [
    "Gene expression encodes biological region identity — tumor marker genes ≠ stroma genes",
    "VisiumHD: 448K co-registered (H&E patch, expression) pairs — rich training signal",
    "IAMSAM shows SAM can segment well given good prompts — expression could provide those prompts",
    "18K genes carry much more discriminative power than a bounding box",
], Inches(9.4), Inches(1.6), Inches(3.6), Inches(2.4), size=12)

add_text(sl, "Proposed pipeline:",
         Inches(9.4), Inches(4.1), Inches(3.6), Inches(0.35),
         size=13, bold=True, color=C["title"])
add_bullets(sl, [
    "Train dual-encoder on (patch, expression) pairs with semantic region labels",
    "Expression query → retrieve morphologically matching image regions",
    "Project expression embedding → SAM prompt token via lightweight adapter",
    "Evaluate: retrieval recall@k by region + downstream Dice",
], Inches(9.4), Inches(4.5), Inches(3.6), Inches(2.5), size=12)

set_notes(sl, """\
Direction C is the gene expression alignment approach. The idea is to train a dual-encoder model that brings H&E patch embeddings and gene expression embeddings into a shared 128-dimensional space, so that a patch of tumor tissue and its matched gene expression vector are close together, while a patch of stroma and the same tumor vector are far apart.

The image tower takes a 64-pixel H&E patch — cropped from the hires image at the bin location — and runs it through a ResNet50 or GigaPath encoder followed by a projection head. The expression tower takes the full 18,085-dimensional log1p normalized expression vector and runs it through a three-layer MLP followed by a projection head. Both come out as 128-dimensional vectors.

Training uses two losses: cross-entropy classification loss to predict the semantic region label from both embeddings, and MSE alignment loss to explicitly pull matched pairs together. The combination ensures that the shared space is both semantically organized by region type and metric-aligned between modalities.

Once trained, this model lets us do several things. We can do cross-modal retrieval: take a reference expression profile for a cell type of interest, embed it, and find the nearest-neighbor image patches. We can generate heatmaps of where a molecular signature is most expressed in the tissue. And most ambitiously, we can project the expression embedding into SAM's prompt encoder token space using a small learned adapter, giving SAM a molecular-identity-aware prompt.

The 448,000 co-registered pairs from VisiumHD are a significant advantage. Most spatial transcriptomics datasets don't come with this kind of dense, expert-annotated pairing. We're sitting on a rich training signal.
""")

# ============================================================
# SLIDE 12 — Direction D: RAG-Based Prompting
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Direction D: RAG-Based Prompting",
           "Retrieve Annotated Examples → Exemplar Prompts → SAM3")

add_image(sl, RAG_DIAGRAM_PATH, Inches(0.1), Inches(1.05), Inches(9.3))

add_rect(sl, Inches(9.5), Inches(1.1), Inches(3.65), Inches(5.9),
         fill=RGBColor(0xF5, 0xF0, 0xFF))
add_text(sl, "RAG for SAM3 — Key Ideas:",
         Inches(9.6), Inches(1.2), Inches(3.4), Inches(0.38),
         size=13, bold=True, color=RGBColor(0x70, 0x30, 0xA0))
add_bullets(sl, [
    "Offline: build a reference vector DB from all annotated (patch, expression, label) triplets",
    "  Encode with dual-encoder → store FAISS index (cosine similarity)",
    "Online: at inference, encode new image tile → query DB → retrieve top-K similar annotated patches",
    "  Feed retrieved patches as exemplar prompts to SAM3 Prompt Encoder (PerSAM-style)",
    "  SAM3 then segments tiles matching the retrieved exemplar appearance",
    "Advantage: no retraining SAM3 needed; exemplar prompts carry real patch appearances",
    "Advantage: DB can be updated by adding new annotated slides — improves with more data",
    "Challenge: requires dual-encoder quality to be good enough for meaningful retrieval",
], Inches(9.6), Inches(1.65), Inches(3.4), Inches(4.8), size=11)

set_notes(sl, """\
Direction D is what I call RAG-based prompting, by analogy with Retrieval-Augmented Generation in language models. The idea is to build a reference database of annotated patches offline, and then at inference time retrieve the most similar annotated examples and use them as exemplar prompts for SAM3.

Let me walk through the diagram. On the top half, we have the offline phase. We take all our annotated training data — the H&E patches from VisiumHD, their matched gene expression vectors, and the region labels from the GeoJSON annotations. We encode all of these through the dual-encoder we trained in Direction C, producing 128-dimensional embedding vectors. We then store these in a FAISS vector database that supports fast cosine similarity search.

On the bottom half, we have the online inference phase. When we encounter a new unannotated slide, we take each image tile, encode it through the image encoder, and query the FAISS database for the top-K most similar annotated patches. These retrieved patches — with their labels — are then fed into SAM3's Prompt Encoder as exemplar prompts. This is analogous to the PerSAM approach, where you give SAM a few example images of the thing you want to segment rather than just points or boxes.

The key insight is that we're using the retrieval system to convert the hard problem of "where is the tumor?" into a series of comparison questions: "does this tile look like any of these known tumor patches?" The SAM3 architecture already supports exemplar-mode prompting, so the main engineering work is building and querying the reference database.

This has several practical advantages. We don't need to modify SAM3 at all — we're only adding a retrieval step in front of it. The database can be incrementally updated as we annotate more slides, so the system improves naturally over time. And because the retrieval is based on both image and expression similarity, the exemplars carry real biological information about what the target region looks like and feels like molecularly.

The main risk is that the dual-encoder retrieval quality needs to be good enough. If the embedding space is not well-organized by region type, we'll retrieve the wrong exemplars and confuse SAM3. So this direction depends on Direction C succeeding first.
""")

# ============================================================
# SLIDE 13 — Directions A / B / E: Better Backbone / Model
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Directions A, B, E: Better Backbone & Model",
           "Supervised Fine-Tuning · GigaPath · Pathology-Specific Segmentation")

opts = [
    ("A. Supervised SAM3 Fine-Tuning",
     "#C55A11",
     ["Fine-tune Image Encoder + Mask Decoder on GeoJSON masks",
      "Upper-bound baseline for SAM-based approaches",
      "Expected Dice 0.6–0.8 (based on similar work)",
      "Answers: can the SAM architecture learn these labels at all?"]),
    ("B. GigaPath Backbone",
     "#2E75B6",
     ["Replace ViT weights with Prov-GigaPath (1.5M WSI pretrained)",
      "Frozen GigaPath first → fine-tune if retrieval improves",
      "Pathology-domain features → better H&E representations",
      "Plug-in replacement for the Image Encoder"]),
    ("E. Pathology-Specific Segmentation Model",
     "#C00000",
     ["Replace entire SAM3 with CellViT / HoVerNet / HistoLytics",
      "Trained on pathology semantic annotations — understands 'stroma', 'tumor'",
      "Avoids SAM's object-segmentation bias entirely",
      "Best long-term option if SAM fine-tuning also underperforms"]),
]

for i, (title, color, bullets) in enumerate(opts):
    x = Inches(0.2 + i * 4.4)
    y = Inches(1.1)
    w, h = Inches(4.2), Inches(5.9)
    add_rect(sl, x, y, w, h, fill=RGBColor(0xF2, 0xF7, 0xFD))
    add_rect(sl, x, y, w, Inches(0.42), fill=RGBColor(*[int(color[k:k+2], 16) for k in (1,3,5)]))
    add_text(sl, title, x + Inches(0.1), y + Inches(0.04), w - Inches(0.2), Inches(0.35),
             size=13, bold=True, color=C["white"])
    for j, b in enumerate(bullets):
        add_text(sl, "• " + b, x + Inches(0.12), y + Inches(0.55 + j * 0.62),
                 w - Inches(0.25), Inches(0.62), size=12)

# Priority recommendation
add_rect(sl, Inches(0.2), Inches(6.4), Inches(12.9), Inches(0.78),
         fill=RGBColor(0xFF, 0xEB, 0x9C))
add_text(sl,
    "Recommended priority: A first (quick upper-bound, ~1 week) → B in parallel → E if A also underperforms.",
    Inches(0.35), Inches(6.45), Inches(12.5), Inches(0.65),
    size=13, bold=True, color=C["title"])

set_notes(sl, """\
These are the three remaining directions that focus on improving the model itself rather than the prompting strategy.

Direction A, supervised SAM3 fine-tuning, is the one I want to do first, because it answers a fundamental question quickly. If we take SAM3 and fine-tune it on our GeoJSON masks with supervised training — paired image tiles and binary masks — what Dice can we achieve? Based on similar work fine-tuning SAM on medical images, I'd expect somewhere between 0.6 and 0.8. If we get that, we know SAM's architecture can learn these labels when trained properly, and the gap between the oracle zero-shot performance and fine-tuned performance tells us exactly how much of the problem is the lack of domain-specific training. This experiment takes maybe one to two weeks on the HPC.

Direction B is replacing the image encoder with Prov-GigaPath. GigaPath is a pathology-specific vision transformer trained on 1.5 million whole-slide images from Providence Health System — it's arguably the best pathology foundation model available right now. The idea is to start by swapping it in as a frozen feature extractor in the dual-encoder alignment framework we described in Direction C, without fine-tuning it. If the retrieval quality improves significantly with frozen GigaPath features compared to ResNet50, that tells us a lot about the importance of domain-specific pretraining.

Direction E is the most radical option: abandon SAM3 entirely and switch to a model designed for histopathology semantic segmentation. Models like CellViT, HoVerNet, and HistoLytics were explicitly trained on annotated pathology data with labels like stroma, tumor, and immune regions. They understand pathology semantics in a way that SAM simply doesn't. The tradeoff is that these models are less general and less flexible, but if our goal is accurate tissue region segmentation and not generalizability, they might be the right tool.

My recommendation for the immediate next steps is to start A first — it's fast and informative — and run B in parallel as we wait for the alignment training results. We can make a decision on E based on whether A and B deliver usable Dice scores.
""")

# ============================================================
# SLIDE 14 — Summary
# ============================================================
sl = blank_slide(prs)
header_bar(sl, "Summary", "SAM-Style Segmentation ≠ Pathology Semantic Regions")

add_image(sl, SUMMARY_TABLE, Inches(0.2), Inches(1.1), Inches(9.5))

add_rect(sl, Inches(9.7), Inches(1.1), Inches(3.4), Inches(2.5),
         fill=RGBColor(0xFF, 0xEB, 0x9C))
add_text(sl, "Core Conclusion",
         Inches(9.8), Inches(1.15), Inches(3.2), Inches(0.35),
         size=13, bold=True, color=C["title"])
add_text(sl,
    "SAM-style promptable object segmentation "
    "does not reliably recover pathology-defined semantic tissue regions "
    "— consistent across two diseases, two datasets, and multiple prompt strategies.",
    Inches(9.8), Inches(1.55), Inches(3.2), Inches(1.9),
    size=12, color=C["black"])

add_rect(sl, Inches(0.2), Inches(4.7), Inches(12.9), Inches(2.55),
         fill=RGBColor(0xF2, 0xF7, 0xFD))
add_text(sl, "Next Steps:",
         Inches(0.35), Inches(4.75), Inches(12.5), Inches(0.35),
         size=14, bold=True, color=C["title"])
add_bullets(sl, [
    "A. Supervised SAM3 fine-tuning on GeoJSON masks → fast upper-bound test (~1 week)",
    "B. Prov-GigaPath backbone: replace ResNet50 in dual-encoder, eval frozen features",
    "C. Gene expression dual-encoder alignment on 448K pairs → cross-modal retrieval + expression → SAM prompt adapter",
    "D. RAG: build annotated patch DB from C → retrieve exemplars at inference → SAM3 exemplar prompts",
    "E. Pathology-specific segmentation (CellViT / HoVerNet) if A/B underperform",
], Inches(0.35), Inches(5.15), Inches(12.5), Inches(2.0), size=13)

set_notes(sl, """\
Let me wrap up.

The cross-dataset summary table captures the key findings. Across both TMA24 and VisiumHD Exp1, text prompts fail, cross-region transfer fails, and even oracle prompts give only moderate Dice at best. Larger context tiles don't help. SAM3 oversegments, MedicalSAM3 undersegments.

The core conclusion I want you to take away is: SAM-style promptable object segmentation does not reliably recover pathology-defined semantic tissue regions. This is not a surprise — SAM was trained on natural images to segment objects like cars and cups. Pathology labels like 'stroma' or 'immune infiltration' are defined by expert criteria that span many different visual textures. They're not objects in the SAM sense.

But this is informative, not discouraging. We now know precisely where the bottlenecks are, and we have a clear plan.

The five next steps I showed in slides 10 through 13: A is supervised SAM fine-tuning — do this first, it takes maybe a week and tells us the SAM upper bound. B is swapping in Prov-GigaPath as the image encoder backbone — low-risk, run in parallel with the alignment training. C is the gene expression dual-encoder alignment using 448,000 co-registered pairs — this is the main novel contribution that distinguishes our work from IAMSAM. D is the RAG pipeline that builds on C — offline reference database, online retrieval and exemplar prompts. E is switching to pathology-specific segmentation models if A through D don't get us to usable Dice scores.

I'm optimistic about C and D specifically, because no one has really used the full-transcriptome expression signal at VisiumHD resolution to guide segmentation before. The 18,000-gene by 448,000-bin paired dataset is a genuinely rich training signal.

Thank you. Happy to take questions — especially on the experimental setup details or the future direction choices.
""")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
prs.save(str(OUTPUT_PPTX))
print(f"\nSaved: {OUTPUT_PPTX}")
