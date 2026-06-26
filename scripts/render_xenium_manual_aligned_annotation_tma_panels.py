#!/usr/bin/env python3
"""Render final no-margin manually aligned panels for annotated Xenium Silica TMAs."""

from __future__ import annotations

import csv
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_xenium_manual_alignment_tool as manual_tool
import xenium_wholeslide_tif_he_translation_registration as ws_align


DEFAULT_ROOT = Path("output/xenium_silica_wholeslide_tif_he_translation_registration_20260624")
DEFAULT_OUT = DEFAULT_ROOT / "manual_accepted_alignment_annotation_tmas_20260624"
ANNOTATED_TMAS = [7, 24, 29, 30, 31, 34, 36, 39, 41, 42]
DENSE_FICTURE_MAX_DISTANCE_PX = 48.0
MEDIUM_FICTURE_BASE_RADIUS_PX = 1
MEDIUM_FICTURE_EXTRA_RADIUS_PX = 2
MEDIUM_FICTURE_EXTRA_SUBSAMPLE = 4

JUNE16_FACTOR_STYLE_PALETTE = {
    "F0 tumor epithelial": "#ffccff",
    "F1 stromal/smooth muscle": "#00ffff",
    "F2 epithelial/tumor subpopulation": "#ffff00",
    "F3 AT2": "#ff5400",
    "F4 lymphocytes": "#00ff54",
    "F5 epithelial/alveolar substate": "#5400ff",
    "F6 myeloid": "#aa00aa",
    "F7 airway epithelial": "#00aaaa",
    "F8 plasma/IgG": "#aaaa00",
    "F9 endothelial": "#ff007f",
    "F10 plasma/IgA": "#994c00",
    "F11 mixed/uncertain display": "#007300",
}

CELLTYPE_TO_JUNE16_FACTOR_STYLE = {
    "AT1": "F5 epithelial/alveolar substate",
    "AT2": "F3 AT2",
    "AberrantB": "F4 lymphocytes",
    "B": "F4 lymphocytes",
    "Basal": "F7 airway epithelial",
    "Ciliated": "F7 airway epithelial",
    "Club": "F7 airway epithelial",
    "DCmature": "F6 myeloid",
    "Endothelial": "F9 endothelial",
    "Fibroblast": "F1 stromal/smooth muscle",
    "Goblet": "F7 airway epithelial",
    "Langerhans": "F6 myeloid",
    "Macrophage": "F6 myeloid",
    "Mast": "F6 myeloid",
    "Megakaryocyte": "F6 myeloid",
    "Monocyte": "F6 myeloid",
    "Mucosal": "F7 airway epithelial",
    "Neutrophil": "F6 myeloid",
    "Pericyte": "F1 stromal/smooth muscle",
    "SmoothMuscle": "F1 stromal/smooth muscle",
    "T": "F4 lymphocytes",
    "TBSecretory": "F7 airway epithelial",
    "activeFib": "F1 stromal/smooth muscle",
    "activeFib_COL10A1": "F1 stromal/smooth muscle",
    "cDC1": "F6 myeloid",
    "mix": "F11 mixed/uncertain display",
    "profibMac": "F6 myeloid",
    "profibMac_CCL22": "F6 myeloid",
}


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def slide_for_tma(tma: int) -> str:
    return "770-1" if tma <= 24 else "771-1"


def clean_label(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "na", "none", "null"}:
        return ""
    return text


def june16_factor_style_label(celltype: str) -> str:
    return CELLTYPE_TO_JUNE16_FACTOR_STYLE.get(clean_label(celltype), "F11 mixed/uncertain display")


def load_manual_params(root: Path, slide: str, tma: int) -> dict:
    path = root / "manual_alignment_params" / f"{slide}_TMA{tma}_manual_alignment_live.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manual alignment params: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    shift = data.get("manual_translation_after_base_affine_px") or {}
    scale_xy = data.get("manual_scale_xy_after_base_affine") or {}
    fallback_scale = float(data.get("manual_scale_after_base_affine", 1.0))
    return {
        "path": str(path),
        "kind": data.get("kind"),
        "unit_id": data.get("unit_id"),
        "mode": data.get("mode"),
        "crop_bbox_xyxy_in_oriented_he_grid": data.get("crop_bbox_xyxy_in_oriented_he_grid"),
        "image_size_px": data.get("image_size_px"),
        "dx": float(shift.get("dx", 0.0)),
        "dy": float(shift.get("dy", 0.0)),
        "sx": float(scale_xy.get("sx", fallback_scale)),
        "sy": float(scale_xy.get("sy", fallback_scale)),
        "raw": data,
    }


def apply_manual_residual(local_xy: np.ndarray, width: int, height: int, params: dict) -> np.ndarray:
    pts = local_xy.astype(float).copy()
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=float)
    pts[:, 0] = (pts[:, 0] - center[0]) * params["sx"] + center[0] + params["dx"]
    pts[:, 1] = (pts[:, 1] - center[1]) * params["sy"] + center[1] + params["dy"]
    return pts


def points_for_tma(rows: list[dict[str, str]], manifest: dict, crop_bbox: list[int], width: int, height: int, params: dict) -> np.ndarray:
    morph_meta = manifest["morphology"]
    affine = np.array(manifest["selected_transform"]["affine_xy_from_morphology_to_oriented_he"], dtype=float)
    morph_xy = ws_align.point_coordinates(rows, morph_meta)
    he_xy = ws_align.apply_affine_to_points(morph_xy, affine)
    local_xy = he_xy - np.array([float(crop_bbox[0]), float(crop_bbox[1])], dtype=float)
    return apply_manual_residual(local_xy, width, height, params)


def paint_points(
    base: Image.Image,
    xy: np.ndarray,
    labels: list[str],
    palette: dict[str, str],
    *,
    radius: int,
    alpha: int,
    default_color: str | None = None,
) -> Image.Image:
    out = base.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for (x, y), label in zip(xy, labels):
        color = palette.get(label, default_color)
        if color is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if -radius <= xi < out.width + radius and -radius <= yi < out.height + radius:
            draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=(*rgb_tuple(color), alpha))
    return Image.alpha_composite(out, overlay)


def make_celltype_map(size: tuple[int, int], xy: np.ndarray, labels: list[str]) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 255))
    factor_labels = [june16_factor_style_label(label) for label in labels]
    return paint_points(canvas, xy, factor_labels, JUNE16_FACTOR_STYLE_PALETTE, radius=1, alpha=255).convert("RGB")


def make_medium_celltype_map(size: tuple[int, int], xy: np.ndarray, labels: list[str]) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 255))
    factor_labels = [june16_factor_style_label(label) for label in labels]
    base = paint_points(
        canvas,
        xy,
        factor_labels,
        JUNE16_FACTOR_STYLE_PALETTE,
        radius=MEDIUM_FICTURE_BASE_RADIUS_PX,
        alpha=255,
    )
    xy_arr = xy.astype(float)
    valid = np.isfinite(xy_arr[:, 0]) & np.isfinite(xy_arr[:, 1])
    # Lightly thicken only a deterministic subset so the map is just above sparse.
    subset = np.zeros(len(xy_arr), dtype=bool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx):
        coords = np.floor(xy_arr[valid_idx]).astype(int)
        subset[valid_idx] = ((coords[:, 0] + coords[:, 1]) % MEDIUM_FICTURE_EXTRA_SUBSAMPLE) == 0
    if np.any(subset):
        base = paint_points(
            base,
            xy_arr[subset],
            [factor_labels[i] for i in np.where(subset)[0]],
            JUNE16_FACTOR_STYLE_PALETTE,
            radius=MEDIUM_FICTURE_EXTRA_RADIUS_PX,
            alpha=255,
        )
    return base.convert("RGB")


def make_tissue_mask_from_he(he: Image.Image) -> np.ndarray:
    """Mask non-background H&E pixels so dense interpolation stays inside tissue."""
    arr = np.asarray(he.convert("RGB"), dtype=np.uint8)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    chroma = maxc - minc
    # White/gray slide background has high brightness and very low chroma.
    tissue = ~((maxc > 238) & (chroma < 18))
    mask = Image.fromarray((tissue.astype(np.uint8) * 255), mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    return np.asarray(mask) > 0


def make_full_dense_celltype_map(
    he: Image.Image,
    xy: np.ndarray,
    labels: list[str],
    *,
    max_distance_px: float = DENSE_FICTURE_MAX_DISTANCE_PX,
) -> Image.Image:
    """Render a full dense Xenium FICTURE-style map by nearest-cell interpolation."""
    from scipy.spatial import cKDTree

    width, height = he.size
    valid_idx: list[int] = []
    colors: list[tuple[int, int, int]] = []
    for idx, ((x, y), label) in enumerate(zip(xy, labels)):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        color = JUNE16_FACTOR_STYLE_PALETTE.get(june16_factor_style_label(label))
        if color is None:
            continue
        if -max_distance_px <= x < width + max_distance_px and -max_distance_px <= y < height + max_distance_px:
            valid_idx.append(idx)
            colors.append(rgb_tuple(color))

    out = np.zeros((height, width, 3), dtype=np.uint8)
    if not valid_idx:
        return Image.fromarray(out, mode="RGB")

    tissue = make_tissue_mask_from_he(he)
    yy, xx = np.nonzero(tissue)
    if len(xx) == 0:
        return Image.fromarray(out, mode="RGB")

    pts = xy[np.array(valid_idx, dtype=int)].astype(float)
    tree = cKDTree(pts)
    query_xy = np.column_stack([xx.astype(float), yy.astype(float)])
    dist, nearest = tree.query(query_xy, k=1)
    keep = dist <= max_distance_px
    color_arr = np.asarray(colors, dtype=np.uint8)
    out[yy[keep], xx[keep], :] = color_arr[nearest[keep]]
    return Image.fromarray(out, mode="RGB")


def make_gt_overlay(he: Image.Image, xy: np.ndarray, labels: list[str]) -> Image.Image:
    base = he.convert("RGBA")
    base = paint_points(base, xy, ["" for _ in labels], {}, radius=1, alpha=35, default_color="#b8b8b8")
    base = paint_points(base, xy, labels, ws_align.base.GT_PALETTE, radius=2, alpha=225)
    return base.convert("RGB")


def make_gt_mask_only(size: tuple[int, int], xy: np.ndarray, labels: list[str]) -> Image.Image:
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    canvas = paint_points(canvas, xy, ["" for _ in labels], {}, radius=1, alpha=50, default_color="#c8c8c8")
    canvas = paint_points(canvas, xy, labels, ws_align.base.GT_PALETTE, radius=2, alpha=245)
    return canvas.convert("RGB")


def make_annotation_overlay_with_legend(
    overlay: Image.Image,
    annotation_counts: dict[str, int],
    *,
    title: str = "new_annotation",
) -> Image.Image:
    """Append a compact color legend to an annotation-on-H&E image."""
    legend_w = 330
    pad = 18
    row_h = 28
    width, height = overlay.size
    out = Image.new("RGB", (width + legend_w, height), "white")
    out.paste(overlay.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    x0 = width
    draw.rectangle((x0, 0, width + legend_w - 1, height - 1), fill="#ffffff", outline="#cbd5e1")
    title_font = font(16)
    small_font = font(13)
    tiny_font = font(11)
    draw.text((x0 + pad, pad), title, fill="#172033", font=title_font)
    draw.text((x0 + pad, pad + 22), "tissue annotation on H&E", fill="#64748b", font=tiny_font)
    y = pad + 52
    for label, count in sorted(annotation_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        color = ws_align.base.GT_PALETTE.get(label, "#999999")
        draw.rounded_rectangle((x0 + pad, y + 4, x0 + pad + 18, y + 22), radius=3, fill=rgb_tuple(color))
        label_text = label if len(label) <= 27 else label[:24] + "..."
        draw.text((x0 + pad + 28, y), label_text, fill="#1f2937", font=small_font)
        draw.text((x0 + pad + 28, y + 15), f"{count:,} cells", fill="#64748b", font=tiny_font)
        y += row_h
        if y > height - 28:
            draw.text((x0 + pad, y), "...", fill="#64748b", font=small_font)
            break
    return out


def thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = image.convert("RGB").copy()
    out.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(out, ((size[0] - out.width) // 2, (size[1] - out.height) // 2))
    return canvas


def make_panel(
    he: Image.Image,
    cell_map: Image.Image,
    gt_overlay: Image.Image,
    gt_mask: Image.Image,
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    tile_w, tile_h = 420, 430
    header_h = 58
    gap = 12
    panels = [
        (he, "H&E no-margin ROI", "manual accepted crop"),
        (cell_map, "Medium-density FICTURE-style map", "June16 RGB remap + local cell expansion"),
        (gt_overlay, "Tissue annotation overlay", "new_annotation on H&E"),
        (gt_mask, "Tissue annotation only", "background H&E removed"),
    ]
    width = len(panels) * tile_w + (len(panels) + 1) * gap
    height = tile_h + header_h + 2 * gap + 52
    canvas = Image.new("RGB", (width, height), "#edf1f6")
    draw = ImageDraw.Draw(canvas)
    title_font = font(18)
    small_font = font(12)
    draw.text((gap, 10), title, fill="#172033", font=title_font)
    draw.text((gap, 34), subtitle, fill="#526173", font=small_font)
    top = 52
    for idx, (image, panel_title, panel_subtitle) in enumerate(panels):
        x = gap + idx * (tile_w + gap)
        y = top
        draw.rectangle((x, y, x + tile_w, y + header_h + tile_h), fill="white", outline="#cbd5e1")
        draw.text((x + 12, y + 10), panel_title, fill="#1f2937", font=title_font)
        draw.text((x + 12, y + 34), panel_subtitle, fill="#64748b", font=small_font)
        thumb = thumbnail(image, (tile_w, tile_h))
        canvas.paste(thumb, (x, y + header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def count_labels(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def render_tma(root: Path, out_root: Path, tma: int) -> dict:
    slide = slide_for_tma(tma)
    manifest = manual_tool.read_manifest(root, slide)
    params = load_manual_params(root, slide, tma)
    config = manual_tool.ensure_assets(root, slide, tma=tma, pad_px=0, options=[])
    he_path = root / config["he"]
    he = Image.open(he_path).convert("RGB")
    if params["crop_bbox_xyxy_in_oriented_he_grid"] != config["crop_bbox_xyxy_in_oriented_he_grid"]:
        raise RuntimeError(f"TMA{tma} manual params do not match current no-margin crop bbox")
    if params["image_size_px"] != {"width": he.width, "height": he.height}:
        raise RuntimeError(f"TMA{tma} manual params do not match current no-margin crop size")

    rows = ws_align.base.read_tma_frame(tma)
    xy = points_for_tma(
        rows,
        manifest,
        config["crop_bbox_xyxy_in_oriented_he_grid"],
        he.width,
        he.height,
        params,
    )
    cell_labels = [row.get(ws_align.base.CELLTYPE_COL, "") for row in rows]
    gt_labels = [clean_label(row.get(ws_align.base.GT_COL, "")) for row in rows]
    gt_non_na = sum(1 for label in gt_labels if label)

    image_dir = out_root / "images"
    he_out = image_dir / f"TMA{tma:02d}_{slide}_manual_he.png"
    cell_out = image_dir / f"TMA{tma:02d}_{slide}_manual_ficture_style_celltype.png"
    cell_sparse_out = image_dir / f"TMA{tma:02d}_{slide}_manual_ficture_style_celltype_sparse_points.png"
    cell_full_dense_out = image_dir / f"TMA{tma:02d}_{slide}_manual_ficture_style_celltype_full_dense_nearest.png"
    gt_overlay_out = image_dir / f"TMA{tma:02d}_{slide}_manual_annotation_overlay.png"
    gt_overlay_legend_out = image_dir / f"TMA{tma:02d}_{slide}_manual_annotation_overlay_with_legend.png"
    gt_mask_out = image_dir / f"TMA{tma:02d}_{slide}_manual_annotation_mask_only.png"
    panel_out = image_dir / f"TMA{tma:02d}_{slide}_manual_accepted_panel.png"
    image_dir.mkdir(parents=True, exist_ok=True)

    cell_sparse_map = make_celltype_map(he.size, xy, cell_labels)
    cell_map = make_medium_celltype_map(he.size, xy, cell_labels)
    cell_full_dense_map = make_full_dense_celltype_map(he, xy, cell_labels)
    gt_overlay = make_gt_overlay(he, xy, gt_labels)
    gt_mask = make_gt_mask_only(he.size, xy, gt_labels)
    annotation_counts = count_labels(gt_labels)
    gt_overlay_with_legend = make_annotation_overlay_with_legend(gt_overlay, annotation_counts)
    he.save(he_out)
    cell_map.save(cell_out)
    cell_sparse_map.save(cell_sparse_out)
    cell_full_dense_map.save(cell_full_dense_out)
    gt_overlay.save(gt_overlay_out)
    gt_overlay_with_legend.save(gt_overlay_legend_out)
    gt_mask.save(gt_mask_out)

    confidence = "low_confidence" if tma == 42 else ("borderline" if tma == 31 else "accepted")
    title = f"TMA{tma} | {slide} | manual accepted alignment"
    subtitle = (
        f"dx={params['dx']:.1f}, dy={params['dy']:.1f}, "
        f"sx={params['sx']:.4f}, sy={params['sy']:.4f}; "
        f"{gt_non_na:,}/{len(rows):,} cells have tissue annotation; status={confidence}"
    )
    make_panel(he, cell_map, gt_overlay, gt_mask, title, subtitle, panel_out)

    return {
        "tma": tma,
        "slide": slide,
        "status": confidence,
        "n_cells": len(rows),
        "n_annotated_cells": gt_non_na,
        "annotation_counts": annotation_counts,
        "manual_alignment": {
            "dx": params["dx"],
            "dy": params["dy"],
            "sx": params["sx"],
            "sy": params["sy"],
            "source_param": params["path"],
            "source_kind": params["kind"],
            "crop_bbox_xyxy_in_oriented_he_grid": params["crop_bbox_xyxy_in_oriented_he_grid"],
            "image_size_px": params["image_size_px"],
        },
        "outputs": {
            "he": str(he_out.relative_to(out_root)),
            "ficture_style_celltype": str(cell_out.relative_to(out_root)),
            "ficture_style_celltype_sparse_points": str(cell_sparse_out.relative_to(out_root)),
            "ficture_style_celltype_full_dense_nearest": str(cell_full_dense_out.relative_to(out_root)),
            "annotation_overlay": str(gt_overlay_out.relative_to(out_root)),
            "annotation_overlay_with_legend": str(gt_overlay_legend_out.relative_to(out_root)),
            "annotation_mask_only": str(gt_mask_out.relative_to(out_root)),
            "panel": str(panel_out.relative_to(out_root)),
        },
    }


def write_index(records: list[dict], out_path: Path) -> None:
    rows = []
    cards = []
    for rec in records:
        align = rec["manual_alignment"]
        counts = "; ".join(f"{k}:{v}" for k, v in sorted(rec["annotation_counts"].items(), key=lambda kv: (-kv[1], kv[0])))
        rows.append(
            f"<tr><td>TMA{rec['tma']}</td><td>{html.escape(rec['slide'])}</td><td>{html.escape(rec['status'])}</td>"
            f"<td>{rec['n_annotated_cells']:,}/{rec['n_cells']:,}</td>"
            f"<td>{align['dx']:.1f}</td><td>{align['dy']:.1f}</td><td>{align['sx']:.4f}</td><td>{align['sy']:.4f}</td>"
            f"<td>{html.escape(counts)}</td></tr>"
        )
        cards.append(
            f"""
            <section class="card" id="tma{rec['tma']}">
              <h2>TMA{rec['tma']} <span>{html.escape(rec['slide'])} | {html.escape(rec['status'])}</span></h2>
              <p class="params">Manual alignment: dx={align['dx']:.1f}, dy={align['dy']:.1f}, sx={align['sx']:.4f}, sy={align['sy']:.4f}. Annotated cells: {rec['n_annotated_cells']:,}/{rec['n_cells']:,}.</p>
              <div class="image-grid">
                <figure>
                  <img src="{html.escape(rec['outputs']['he'])}" alt="TMA{rec['tma']} H&E">
                  <figcaption>H&amp;E no-margin ROI</figcaption>
                </figure>
                <figure>
                  <img src="{html.escape(rec['outputs']['ficture_style_celltype'])}" alt="TMA{rec['tma']} FICTURE-style cell-type map">
                  <figcaption>Medium-density FICTURE-style cell-type map, June16 RGB remap</figcaption>
                </figure>
                <figure>
                  <img src="{html.escape(rec['outputs']['annotation_overlay_with_legend'])}" alt="TMA{rec['tma']} annotation overlay with legend">
                  <figcaption>Annotation overlay on H&amp;E with legend</figcaption>
                </figure>
              </div>
            </section>
            """
        )
    out_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Manual Accepted Alignment Annotated TMAs</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f7fb;color:#172033}}
header{{background:#111827;color:white;padding:24px 30px}}
main{{padding:22px 30px 50px}}
table{{border-collapse:collapse;width:100%;background:white;border:1px solid #d8dee8;margin-bottom:22px}}
th,td{{padding:8px 10px;border-bottom:1px solid #e5e9f0;text-align:left;font-size:13px;vertical-align:top}}
th{{background:#edf1f6}}
.card{{background:white;border:1px solid #d7dee8;border-radius:8px;margin:18px 0;padding:16px}}
.card h2{{margin:0 0 12px;font-size:20px}} .card h2 span{{font-size:14px;color:#64748b}}
.params{{margin:0 0 12px;color:#526173;font-size:13px}}
.image-grid{{display:grid;grid-template-columns:1fr 1fr 1.35fr;gap:14px;align-items:start}}
figure{{margin:0}}
figcaption{{font-size:12px;color:#526173;margin-top:6px}}
.card img{{display:block;width:100%;height:auto;border:1px solid #d8dee8;background:white}}
code{{background:#edf1f6;padding:2px 5px;border-radius:4px}}
@media(max-width:1200px){{.image-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
<h1>Manual Accepted Alignment for Annotated TMAs</h1>
<p><b>Experiment Design.</b> Goal: show the 10 tissue-annotated Xenium Silica TMAs after manual no-margin alignment. Each TMA has three views: H&amp;E morphology, a medium-density FICTURE-style Xenium cell-type map, and the tissue annotation overlay on H&amp;E with a legend. FICTURE-style here means a false-color map from source-provided Xenium <code>cellTypeFibSim</code> labels, not official VisiumHD FICTURE factors. To make the visual style closer to the June16 VisiumHD/FICTURE figure without overfilling the tissue, Xenium cell types are remapped to the June16 factor RGB palette and each cell is locally expanded by a small radius. The original sparse point map and the previous full nearest-cell dense map are still saved separately.</p>
</header>
<main>
<table><thead><tr><th>TMA</th><th>Slide</th><th>Status</th><th>Annotated cells</th><th>dx</th><th>dy</th><th>sx</th><th>sy</th><th>new_annotation counts</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
{''.join(cards)}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    root = DEFAULT_ROOT.resolve()
    out_root = DEFAULT_OUT.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    records = [render_tma(root, out_root, tma) for tma in ANNOTATED_TMAS]
    manifest = {
        "alignment_policy": "manual residual alignment accepted for annotated no-margin TMAs",
        "ficture_style_policy": {
            "displayed_middle_panel": "medium-density local expansion of source-provided Xenium cellTypeFibSim labels",
            "color_policy": "cellTypeFibSim labels remapped to June16 VisiumHD/FICTURE factor-style RGB buckets before rendering",
            "not_official_ficture": True,
            "medium_base_radius_px": MEDIUM_FICTURE_BASE_RADIUS_PX,
            "medium_extra_radius_px": MEDIUM_FICTURE_EXTRA_RADIUS_PX,
            "medium_extra_subsample": MEDIUM_FICTURE_EXTRA_SUBSAMPLE,
            "max_nearest_cell_distance_px": DENSE_FICTURE_MAX_DISTANCE_PX,
            "sparse_point_maps_saved": True,
            "full_dense_nearest_maps_saved": True,
            "june16_factor_style_palette": JUNE16_FACTOR_STYLE_PALETTE,
            "celltype_to_june16_factor_style": CELLTYPE_TO_JUNE16_FACTOR_STYLE,
        },
        "warning": "TMA42 is rendered with manual parameters but marked low_confidence from QC.",
        "root": str(root),
        "out_root": str(out_root),
        "annotated_tmas": ANNOTATED_TMAS,
        "records": records,
    }
    (out_root / "manual_accepted_alignment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(records, out_root / "index.html")
    print(json.dumps({"out_root": str(out_root), "n_records": len(records), "index": str(out_root / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
