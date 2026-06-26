#!/usr/bin/env python3
"""Build FICTURE-driven high-resolution Xenium Silica TMA overview.

The previous high-resolution overview kept the old manual H&E crop bbox and
resized the molecule-level FICTURE image into that canvas.  This script instead
uses the molecule-level FICTURE footprint to expand the H&E crop, then warps the
K=12 FICTURE map into the same expanded canvas.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

import render_xenium_manual_aligned_annotation_tma_panels as manual_panels
import xenium_wholeslide_tif_he_translation_registration as ws_align
import build_xenium_silica_continuous_gt_masks as gt_masks
import serve_xenium_manual_alignment_tool as manual_tool
from build_xenium_silica_highres_he_crops import grid_bbox_to_raw_bbox
from build_xenium_silica_highres_tma_overview import (
    annotation_legend_html,
    load_factor_legend,
    make_contact_sheet,
    make_original_slide_previews,
    rel,
    source_slide_html,
)

AAAI_ROOT = ROOT / "output/aaai_xenium_silica_20260623"
REG_ROOT = ROOT / "output/xenium_silica_wholeslide_tif_he_translation_registration_20260624"
RAW_XENIUM_ROOT = Path("/Volumes/Disk/ST_annotated_datasets/tmp_Xenium")
ANNOTATION_ROOT = Path("/Volumes/Disk/ST_annotated_datasets/Xenium_Silica/with_annotation")
FICTURE_SLIM_ROOT = (
    AAAI_ROOT
    / "xenium_true_molecule_ficture_sync_20260626"
    / "xenium_silica_true_molecule_ficture_slim_20260626"
)
FICTURE_ROOT = FICTURE_SLIM_ROOT / "ksweep"
INPUT_MANIFEST_ROOT = FICTURE_SLIM_ROOT / "input_manifests"
OUT_ROOT = AAAI_ROOT / "tma_sample_overview_ficturedriven_highres_20260626"
HIGHRES_ROOT = AAAI_ROOT / "highres_he_crops_ficturedriven_20260626"
GUARD_PAD_OLD_HE_PX = 30
REGION_CLOSE_RADIUS_PX = 8


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_metadata_native_to_old_he(rows: list[dict[str, str]], local_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    src: list[list[float]] = []
    dst: list[list[float]] = []
    for row, xy in zip(rows, local_xy):
        try:
            x = float(row["coordinateX"])
            y = float(row["coordinateY"])
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y) or not np.all(np.isfinite(xy)):
            continue
        src.append([x, y, 1.0])
        dst.append([float(xy[0]), float(xy[1])])
    if len(src) < 6:
        raise RuntimeError(f"Only {len(src)} metadata points available for native->H&E fit")
    design = np.asarray(src, dtype=float)
    target = np.asarray(dst, dtype=float)
    ax = np.linalg.lstsq(design, target[:, 0], rcond=None)[0]
    ay = np.linalg.lstsq(design, target[:, 1], rcond=None)[0]
    affine = np.asarray([[ax[0], ax[1], ax[2]], [ay[0], ay[1], ay[2]]], dtype=float)
    pred = np.column_stack([design @ ax, design @ ay])
    residual = np.abs(pred - target)
    return affine, {
        "fit_basis": "metadata coordinateX/coordinateY to old manual H&E crop",
        "n_fit_points": int(len(src)),
        "residual_median_xy_px": [float(v) for v in np.median(residual, axis=0)],
        "residual_max_xy_px": [float(v) for v in residual.max(axis=0)],
        "affine_native_xy_to_old_he": affine.tolist(),
    }


def apply_affine(xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return xy @ affine[:, :2].T + affine[:, 2]


def nonblack_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    yy, xx = np.where(arr.max(axis=2) > 8)
    if len(xx) == 0:
        return (0, 0, image.width, image.height)
    return (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)


def ficture_nonblack_bbox_in_old_he(
    ficture: Image.Image,
    input_manifest: dict,
    native_to_old_he: np.ndarray,
) -> tuple[list[float], dict]:
    rng = input_manifest["output_coordinate_range"]
    xmin, xmax = float(rng["xmin"]), float(rng["xmax"])
    ymin, ymax = float(rng["ymin"]), float(rng["ymax"])
    sx = (xmax - xmin) / float(ficture.width)
    sy = (ymax - ymin) / float(ficture.height)
    x0, y0, x1, y1 = nonblack_bbox(ficture)
    native_corners = np.asarray(
        [
            [xmin + x0 * sx, ymin + y0 * sy],
            [xmin + x1 * sx, ymin + y0 * sy],
            [xmin + x1 * sx, ymin + y1 * sy],
            [xmin + x0 * sx, ymin + y1 * sy],
        ],
        dtype=float,
    )
    local = apply_affine(native_corners, native_to_old_he)
    bbox = [float(local[:, 0].min()), float(local[:, 1].min()), float(local[:, 0].max()), float(local[:, 1].max())]
    return bbox, {
        "source_nonblack_bbox_px": [x0, y0, x1, y1],
        "source_size_px": [ficture.width, ficture.height],
        "source_pixel_to_xenium_scale_xy": [sx, sy],
        "xenium_nonblack_bbox_xyxy": [
            float(native_corners[:, 0].min()),
            float(native_corners[:, 1].min()),
            float(native_corners[:, 0].max()),
            float(native_corners[:, 1].max()),
        ],
        "old_he_local_nonblack_bbox_xyxy": bbox,
    }


def expanded_old_he_bbox(old_w: int, old_h: int, ficture_bbox: list[float]) -> list[float]:
    x0 = min(0.0, ficture_bbox[0]) - GUARD_PAD_OLD_HE_PX
    y0 = min(0.0, ficture_bbox[1]) - GUARD_PAD_OLD_HE_PX
    x1 = max(float(old_w), ficture_bbox[2]) + GUARD_PAD_OLD_HE_PX
    y1 = max(float(old_h), ficture_bbox[3]) + GUARD_PAD_OLD_HE_PX
    return [x0, y0, x1, y1]


def crop_oriented_highres(slide: ws_align.base.SlideConfig, raw_bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = raw_bbox
    arr = ws_align.base.he_memmap(slide)
    raw_crop = np.asarray(arr[y0:y1, x0:x1, :]).copy()
    oriented = ws_align.roi_align.orient_he_crop(raw_crop, ws_align.ORIENTATION)
    return Image.fromarray(oriented)


def warp_ficture_to_expanded_canvas(
    ficture: Image.Image,
    input_manifest: dict,
    native_to_old_he: np.ndarray,
    expanded_bbox: list[float],
    out_size_lowres: tuple[int, int],
) -> Image.Image:
    rng = input_manifest["output_coordinate_range"]
    xmin, xmax = float(rng["xmin"]), float(rng["xmax"])
    ymin, ymax = float(rng["ymin"]), float(rng["ymax"])
    src_sx = (xmax - xmin) / float(ficture.width)
    src_sy = (ymax - ymin) / float(ficture.height)
    matrix = native_to_old_he[:, :2]
    offset = native_to_old_he[:, 2]
    inv = np.linalg.inv(matrix)
    expanded_origin = np.asarray([expanded_bbox[0], expanded_bbox[1]], dtype=float)
    base = inv @ (expanded_origin - offset)
    a = inv[0, 0] / src_sx
    b = inv[0, 1] / src_sx
    c = (base[0] - xmin) / src_sx
    d = inv[1, 0] / src_sy
    e = inv[1, 1] / src_sy
    f = (base[1] - ymin) / src_sy
    return ficture.transform(
        out_size_lowres,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.NEAREST,
        fillcolor=(0, 0, 0),
    ).convert("RGB")


def draw_original_gt_to_expanded_canvas(
    *,
    tma: int,
    raw_slide: str,
    native_to_old_he: np.ndarray,
    expanded_bbox: list[float],
    out_size: tuple[int, int],
    out_dir: Path,
) -> tuple[Image.Image, Image.Image, dict[str, int], dict[str, object]]:
    annotated = gt_masks.read_annotated_metadata(ANNOTATION_ROOT, tma, gt_masks.GT_COL)
    cells_path = RAW_XENIUM_ROOT / raw_slide / "cells.parquet"
    cells = gt_masks.pd.read_parquet(cells_path, columns=["cell_id", "x_centroid", "y_centroid"])
    cells["cell_id"] = cells["cell_id"].astype(str)
    matched, match_stats = gt_masks.match_annotation_to_cells(
        annotated,
        cells,
        annotation_col=gt_masks.GT_COL,
        max_distance=10.0,
    )
    boundaries = gt_masks.load_boundaries_for_cells(RAW_XENIUM_ROOT, raw_slide, matched["_xenium_cell_id"])
    id_to_label = {str(row["_xenium_cell_id"]): str(row[gt_masks.GT_COL]) for _, row in matched.iterrows()}

    point_mask = Image.new("RGB", out_size, "white")
    point_draw = ImageDraw.Draw(point_mask)
    point_counts = {str(label): int(count) for label, count in gt_masks.Counter(annotated[gt_masks.GT_COL].tolist()).items()}
    origin = np.asarray([expanded_bbox[0], expanded_bbox[1]], dtype=float)
    annotation_xy = annotated[["coordinateX", "coordinateY"]].to_numpy(dtype=float)
    point_xy = apply_affine(annotation_xy, native_to_old_he) - origin
    for (x, y), label in zip(point_xy, annotated[gt_masks.GT_COL].tolist()):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        rgb = gt_masks.rgb_tuple(ws_align.base.GT_PALETTE.get(str(label), "#999999"))
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        radius = 2
        if -radius <= xi < out_size[0] + radius and -radius <= yi < out_size[1] + radius:
            point_draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=rgb)

    cell_mask = Image.new("RGB", out_size, "white")
    cell_draw = ImageDraw.Draw(cell_mask)
    counts: dict[str, int] = {}
    per_class_masks: dict[str, Image.Image] = {}
    per_class_draws: dict[str, ImageDraw.ImageDraw] = {}
    per_class_dir = out_dir / "original_gt_per_class_masks"
    per_class_dir.mkdir(parents=True, exist_ok=True)

    for cell_id, group in boundaries.groupby("cell_id", sort=False):
        label = id_to_label.get(str(cell_id))
        if not label:
            continue
        native_xy = group[["vertex_x", "vertex_y"]].to_numpy(dtype=float)
        old_xy = apply_affine(native_xy, native_to_old_he)
        expanded_xy = old_xy - origin
        finite = np.isfinite(expanded_xy).all(axis=1)
        poly = [(float(x), float(y)) for x, y in expanded_xy[finite]]
        if len(poly) < 3:
            continue
        rgb = gt_masks.rgb_tuple(ws_align.base.GT_PALETTE.get(label, "#999999"))
        cell_draw.polygon(poly, fill=rgb)
        if label not in per_class_masks:
            per_class_masks[label] = Image.new("L", out_size, 0)
            per_class_draws[label] = ImageDraw.Draw(per_class_masks[label])
        per_class_draws[label].polygon(poly, fill=255)
        counts[label] = counts.get(label, 0) + 1

    region_mask, region_meta = build_shape_preserving_continuous_region_mask(per_class_masks, out_size)

    per_class_outputs: dict[str, dict[str, str]] = {}
    for label, mask in sorted(per_class_masks.items()):
        slug = gt_masks.slugify(label)
        binary_path = per_class_dir / f"TMA{tma:02d}_{slug}_binary.png"
        color_path = per_class_dir / f"TMA{tma:02d}_{slug}_color.png"
        mask.save(binary_path)
        color_img = Image.new("RGB", out_size, "white")
        arr = np.asarray(mask, dtype=np.uint8) > 0
        color_arr = np.asarray(color_img).copy()
        color_arr[arr] = gt_masks.rgb_tuple(ws_align.base.GT_PALETTE.get(label, "#999999"))
        Image.fromarray(color_arr, "RGB").save(color_path)
        per_class_outputs[label] = {
            "binary_mask_abs": str(binary_path),
            "color_mask_abs": str(color_path),
        }

    source_meta = {
        "gt_source": "original annotation rows rebuilt directly for the FICTURE-driven crop",
        "annotation_csv": str(ANNOTATION_ROOT / f"TMA{tma}.csv"),
        "annotation_column": gt_masks.GT_COL,
        "cell_centroids": str(cells_path),
        "cell_boundaries": str(RAW_XENIUM_ROOT / raw_slide / "cell_boundaries.parquet"),
        "original_annotation_policy": "draw original new_annotation rows as sparse colored points in the expanded FICTURE-driven H&E canvas",
        "continuous_region_policy": (
            "nearest original annotation coordinate to Xenium cell centroid, fill matching Xenium cell_boundaries polygon, "
            f"then apply a small label-wise closing radius {REGION_CLOSE_RADIUS_PX}px plus hole filling; "
            "no final dilation or outward growth step; overlaps are resolved by nearest original filled-cell mask"
        ),
        "match_stats": match_stats,
        "point_counts": point_counts,
        "filled_cell_counts": counts,
        "continuous_region_meta": region_meta,
        "n_boundary_vertices_loaded": int(len(boundaries)),
        "per_class_outputs": per_class_outputs,
    }
    return point_mask, region_mask, point_counts, source_meta


def disk_structure(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def build_shape_preserving_continuous_region_mask(
    per_class_masks: dict[str, Image.Image],
    out_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, object]]:
    labels = sorted(per_class_masks)
    width, height = out_size
    if not labels:
        return Image.new("RGB", out_size, "white"), {"region_area_px": {}}

    close_structure = disk_structure(REGION_CLOSE_RADIUS_PX)
    region_area_px: dict[str, int] = {}
    original_area_px: dict[str, int] = {}
    candidates: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}

    for label in labels:
        original = np.asarray(per_class_masks[label], dtype=np.uint8) > 0
        original_area_px[label] = int(original.sum())
        if not np.any(original):
            continue
        region = ndimage.binary_closing(original, structure=close_structure)
        region = ndimage.binary_fill_holes(region)
        candidates[label] = region
        distances[label] = ndimage.distance_transform_edt(~original)
        region_area_px[label] = int(region.sum())

    assignment = np.full((height, width), -1, dtype=np.int16)
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    for idx, label in enumerate(labels):
        region = candidates.get(label)
        if region is None:
            continue
        dist = distances[label]
        update = region & (dist < best_distance)
        assignment[update] = idx
        best_distance[update] = dist[update]

    out = Image.new("RGB", out_size, "white")
    arr = np.asarray(out).copy()
    for idx, label in enumerate(labels):
        arr[assignment == idx] = gt_masks.rgb_tuple(ws_align.base.GT_PALETTE.get(label, "#999999"))
    meta = {
        "region_generation": "shape-preserving small closing plus hole filling",
        "region_close_radius_px": REGION_CLOSE_RADIUS_PX,
        "region_dilate_radius_px": 0,
        "hole_fill": True,
        "original_cell_boundary_area_px": original_area_px,
        "region_area_px": region_area_px,
    }
    return Image.fromarray(arr, "RGB"), meta


def alpha_blend_annotation_on_he(he: Image.Image, mask: Image.Image) -> Image.Image:
    resized = mask.convert("RGB").resize(he.size, Image.Resampling.NEAREST)
    white = Image.new("RGB", he.size, "white")
    nonwhite = ImageChops.difference(resized, white).convert("L").point(lambda v: 0 if v < 8 else 1)
    color_rgba = resized.convert("RGBA")
    color_rgba.putalpha(nonwhite.point(lambda v: 115 if v else 0))
    return Image.alpha_composite(he.convert("RGBA"), color_rgba).convert("RGB")


def make_panel_preview(he: Image.Image, fic: Image.Image, out_path: Path) -> None:
    tile_w, tile_h = 520, 450
    header_h = 36
    canvas = Image.new("RGB", (tile_w * 2, tile_h), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)
    for idx, (title, image) in enumerate(
        [
            ("FICTURE-driven high-res H&E", he),
            ("K=12 molecule FICTURE aligned", fic),
        ]
    ):
        x = idx * tile_w
        draw.rectangle((x, 0, x + tile_w - 1, tile_h - 1), fill="white", outline="#cbd5e1")
        draw.text((x + 12, 8), title, fill="#0f172a", font=title_font)
        thumb = image.copy()
        thumb.thumbnail((tile_w - 24, tile_h - header_h - 12), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (tile_w - thumb.width) // 2, header_h + 6))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key not in {"factor_legend", "annotation_counts"}:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def pretty_factor_legend_html(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>No FICTURE factor legend found.</p>"
    parts = ["<div class='factor-cards'>"]
    for row in rows:
        celltype = row.get("celltype") or row.get("compartment") or "unlabeled"
        genes = row.get("top_genes") or ""
        parts.append(
            "<div class='factor-card'>"
            f"<span class='factor-swatch' style='background:{html.escape(row['hex'])}'></span>"
            "<div>"
            f"<div class='factor-title'>F{html.escape(row['factor'])} <span>{html.escape(row['hex'])}</span></div>"
            f"<div class='factor-type'>{html.escape(celltype)}</div>"
            f"<div class='factor-genes'>{html.escape(genes)}</div>"
            "</div></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    image_dir = OUT_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    highres_dir = HIGHRES_ROOT / "images"
    highres_dir.mkdir(parents=True, exist_ok=True)

    source_slide_records = make_original_slide_previews(OUT_ROOT)
    zip_sizes: dict[str, int] = {}
    with zipfile.ZipFile(ws_align.base.HE_ZIP) as zf:
        for slide in ws_align.base.SLIDES.values():
            zip_sizes[slide.zip_member] = int(zf.getinfo(slide.zip_member).file_size)

    records: list[dict] = []
    for tma in [7, 24, 29, 30, 31, 34, 36, 39, 41, 42]:
        tma_name = f"TMA{tma:02d}"
        slide_id = manual_panels.slide_for_tma(tma)
        sample = f"{tma_name}_{slide_id}"
        slide = ws_align.base.SLIDES[slide_id]
        params = manual_panels.load_manual_params(REG_ROOT, slide_id, tma)
        old_bbox = params["crop_bbox_xyxy_in_oriented_he_grid"]
        old_w = int(old_bbox[2] - old_bbox[0])
        old_h = int(old_bbox[3] - old_bbox[1])
        annotated_df = gt_masks.read_annotated_metadata(ANNOTATION_ROOT, tma, gt_masks.GT_COL)
        rows = annotated_df.astype(str).to_dict(orient="records")
        reg_manifest = manual_tool.read_manifest(REG_ROOT, slide_id)
        old_local_cell_xy = manual_panels.points_for_tma(rows, reg_manifest, old_bbox, old_w, old_h, params)
        native_to_old_he, affine_meta = fit_metadata_native_to_old_he(rows, old_local_cell_xy)

        sample_dir = FICTURE_ROOT / sample / "hex_12_k12"
        ficture_path = sample_dir / "h12.k12.cmap48.pixel.png"
        ficture_native = Image.open(ficture_path).convert("RGB")
        input_manifest = read_json(INPUT_MANIFEST_ROOT / sample / "manifest.json")
        ficture_bbox, ficture_bbox_meta = ficture_nonblack_bbox_in_old_he(
            ficture_native,
            input_manifest,
            native_to_old_he,
        )
        expanded_bbox = expanded_old_he_bbox(old_w, old_h, ficture_bbox)
        expanded_low_w = int(math.ceil(expanded_bbox[2] - expanded_bbox[0]))
        expanded_low_h = int(math.ceil(expanded_bbox[3] - expanded_bbox[1]))
        expanded_grid_bbox = [
            old_bbox[0] + expanded_bbox[0],
            old_bbox[1] + expanded_bbox[1],
            old_bbox[0] + expanded_bbox[2],
            old_bbox[1] + expanded_bbox[3],
        ]
        grid_path = REG_ROOT / f"{slide_id}_wholeslide_rotated_he_grid.png"
        grid_size = Image.open(grid_path).size
        raw_bbox, oriented_raw_bbox = grid_bbox_to_raw_bbox(expanded_grid_bbox, slide=slide, grid_size_wh=grid_size)
        he_high = crop_oriented_highres(slide, raw_bbox)
        he_path = highres_dir / f"{sample}_he_ficturedriven_highres.png"
        he_high.save(he_path)

        ficture_low = warp_ficture_to_expanded_canvas(
            ficture_native,
            input_manifest,
            native_to_old_he,
            expanded_bbox,
            (expanded_low_w, expanded_low_h),
        )
        ficture_high = ficture_low.resize(he_high.size, Image.Resampling.NEAREST)
        ficture_high_path = image_dir / f"{sample}_ficture_k12_cmap48_ficturedriven_highres.png"
        ficture_high.save(ficture_high_path)

        raw_slide = f"slide{slide_id}"
        original_gt_low, continuous_region_low, gt_counts, gt_source_meta = draw_original_gt_to_expanded_canvas(
            tma=tma,
            raw_slide=raw_slide,
            native_to_old_he=native_to_old_he,
            expanded_bbox=expanded_bbox,
            out_size=(expanded_low_w, expanded_low_h),
            out_dir=image_dir,
        )
        original_gt_mask_path = image_dir / f"{sample}_gt_original_points_mask_only.png"
        original_gt_mask_high = original_gt_low.resize(he_high.size, Image.Resampling.NEAREST)
        original_gt_mask_high.save(original_gt_mask_path)
        original_gt_overlay = alpha_blend_annotation_on_he(he_high, original_gt_low)
        original_gt_overlay_path = image_dir / f"{sample}_gt_original_points_overlay.png"
        original_gt_overlay.save(original_gt_overlay_path)

        continuous_gt_mask_path = image_dir / f"{sample}_gt_continuous_region_mask_only.png"
        continuous_gt_mask_high = continuous_region_low.resize(he_high.size, Image.Resampling.NEAREST)
        continuous_gt_mask_high.save(continuous_gt_mask_path)
        continuous_gt_overlay = alpha_blend_annotation_on_he(he_high, continuous_region_low)
        continuous_gt_overlay_path = image_dir / f"{sample}_gt_continuous_region_overlay.png"
        continuous_gt_overlay.save(continuous_gt_overlay_path)

        preview_path = image_dir / f"{sample}_picturedriven_he_ficture_preview.jpg"
        make_panel_preview(he_high, ficture_high, preview_path)

        record = {
            "tma": tma_name,
            "tma_num": tma,
            "slide": slide_id,
            "he_abs": str(he_path),
            "he_rel": rel(he_path, OUT_ROOT),
            "he_size": f"{he_high.width}x{he_high.height}",
            "old_he_size": f"{old_w}x{old_h}",
            "expanded_lowres_size": f"{expanded_low_w}x{expanded_low_h}",
            "expanded_old_he_bbox_xyxy": json.dumps([round(v, 3) for v in expanded_bbox]),
            "ficture_bbox_in_old_he_xyxy": json.dumps([round(v, 3) for v in ficture_bbox]),
            "expanded_grid_bbox_xyxy": json.dumps([round(v, 3) for v in expanded_grid_bbox]),
            "raw_he_bbox_unrotated_xyxy": json.dumps(raw_bbox),
            "oriented_raw_bbox_xyxy": json.dumps([round(v, 3) for v in oriented_raw_bbox]),
            "raw_he_member": slide.zip_member,
            "raw_he_member_px_unrotated": f"{slide.he_shape[1]}x{slide.he_shape[0]}",
            "raw_he_member_bytes": zip_sizes[slide.zip_member],
            "ficture_native_abs": str(ficture_path),
            "ficture_native_rel": rel(ficture_path, OUT_ROOT),
            "ficture_native_size": f"{ficture_native.width}x{ficture_native.height}",
            "ficture_high_abs": str(ficture_high_path),
            "ficture_high_rel": rel(ficture_high_path, OUT_ROOT),
            "gt_original_overlay_abs": str(original_gt_overlay_path),
            "gt_original_overlay_rel": rel(original_gt_overlay_path, OUT_ROOT),
            "gt_original_mask_abs": str(original_gt_mask_path),
            "gt_original_mask_rel": rel(original_gt_mask_path, OUT_ROOT),
            "gt_continuous_overlay_abs": str(continuous_gt_overlay_path),
            "gt_continuous_overlay_rel": rel(continuous_gt_overlay_path, OUT_ROOT),
            "gt_continuous_mask_abs": str(continuous_gt_mask_path),
            "gt_continuous_mask_rel": rel(continuous_gt_mask_path, OUT_ROOT),
            "preview_panel_abs": str(preview_path),
            "preview_panel_rel": rel(preview_path, OUT_ROOT),
            "factor_legend": load_factor_legend(sample_dir),
            "annotation_counts": gt_counts,
            "gt_source_meta": gt_source_meta,
            "native_to_old_he_affine_meta": affine_meta,
            "ficture_bbox_meta": ficture_bbox_meta,
            "crop_policy": "union(old H&E manual crop, FICTURE K=12 nonblack footprint) + 30 old-grid-pixel guard pad",
        }
        records.append(record)
        print(tma_name, record["old_he_size"], "->", record["expanded_lowres_size"], "->", record["he_size"])

    records.sort(key=lambda r: r["tma_num"])
    make_contact_sheet(records, OUT_ROOT / "figuredriven_highres_tma_overview_contact_sheet.jpg")

    slim_records = []
    for rec in records:
        slim = {k: v for k, v in rec.items() if k not in {"factor_legend", "annotation_counts"}}
        slim["annotation_counts"] = rec["annotation_counts"]
        slim_records.append(slim)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(slim_records, indent=2), encoding="utf-8")
    write_csv(OUT_ROOT / "manifest.csv", records)
    (HIGHRES_ROOT / "highres_he_crop_manifest.json").write_text(json.dumps(slim_records, indent=2), encoding="utf-8")
    write_csv(HIGHRES_ROOT / "highres_he_crop_manifest.csv", records)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Xenium Silica FICTURE-Driven High-Resolution TMA Overview</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#f1f5f9;color:#111827}",
        "h1{font-size:24px;margin:0 0 8px} h2{font-size:20px;margin:28px 0 8px} h3{font-size:15px;margin:10px 0 6px}",
        ".note{background:white;border:1px solid #d7dee8;padding:14px 16px;margin:14px 0 22px;line-height:1.45}",
        ".source-section{background:white;border:1px solid #cbd5e1;margin:12px 0 22px;padding:14px}",
        ".source-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}",
        ".row{background:white;border:1px solid #cbd5e1;margin:22px 0;padding:14px}",
        ".primary-grid{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1.05fr) minmax(280px,.75fr);gap:14px;align-items:start}",
        ".gt-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}",
        "figure{margin:0;border:1px solid #d7dee8;background:#fff}",
        "figcaption{font-size:13px;padding:8px 10px;border-bottom:1px solid #e5e7eb;font-weight:700;color:#334155}",
        ".caption-sub{font-weight:400;color:#64748b}",
        "img.panel{width:100%;height:auto;display:block}",
        ".meta{font-size:13px;color:#475569;line-height:1.45;margin:8px 0 12px}",
        ".legend-panel{border:1px solid #d7dee8;background:#fbfdff;padding:12px}",
        ".legend-panel h3{margin-top:0}",
        ".factor-cards{display:grid;grid-template-columns:1fr;gap:8px;max-height:640px;overflow:auto;padding-right:2px}",
        ".factor-card{display:grid;grid-template-columns:34px 1fr;gap:8px;border:1px solid #e2e8f0;background:white;padding:8px;align-items:start}",
        ".factor-swatch{width:28px;height:28px;border:1px solid #334155;display:inline-block}",
        ".factor-title{font-size:13px;font-weight:800;color:#111827}.factor-title span{font-weight:500;color:#64748b}",
        ".factor-type{font-size:12px;color:#334155;margin-top:2px}.factor-genes{font-size:11px;color:#64748b;margin-top:3px;line-height:1.25}",
        "details.gt-details{border:1px solid #d7dee8;background:#f8fafc;margin-top:12px;padding:0}",
        "details.gt-details summary{cursor:pointer;font-weight:800;padding:10px 12px;background:#eef2f7;color:#172033}",
        "details.gt-details[open] summary{border-bottom:1px solid #d7dee8}",
        ".details-body{padding:12px}",
        "table.legend{border-collapse:collapse;width:100%;font-size:12px}",
        "table.legend th,table.legend td{border:1px solid #e5e7eb;padding:5px 6px;text-align:left;vertical-align:top}",
        ".swatch{display:inline-block;width:18px;height:18px;border:1px solid #475569;margin-right:6px;vertical-align:middle}",
        ".ann-counts{font-size:12px;display:flex;flex-wrap:wrap;gap:7px}",
        ".ann-counts span{border:1px solid #d1d5db;border-radius:4px;background:#f8fafc;padding:5px 7px}",
        ".muted{color:#64748b}",
        "code{background:#f8fafc;border:1px solid #e5e7eb;border-radius:3px;padding:1px 4px}",
        "a{color:#075985;text-decoration:none} a:hover{text-decoration:underline}",
        "</style></head><body>",
        "<h1>Xenium Silica FICTURE-Driven High-Resolution TMA Overview</h1>",
        source_slide_html(source_slide_records),
        "<div class='note'><b>Experiment design.</b> Goal: inspect each TMA sample using a crop driven by the complete K=12 molecule-level FICTURE footprint. The H&amp;E panel is cropped from the original full-resolution H&amp;E TIFF. The FICTURE panel is warped into the same expanded H&amp;E canvas using Xenium transcript coordinates, not resized blindly. Ground-truth annotation is hidden in collapsible QC sections below each sample and is evaluation-only, not model input.</div>",
        f"<p><a href='{html.escape(rel(OUT_ROOT / 'figuredriven_highres_tma_overview_contact_sheet.jpg', OUT_ROOT))}'>Contact sheet</a> | <a href='manifest.csv'>Manifest CSV</a></p>",
    ]
    for rec in records:
        html_parts.extend(
            [
                f"<section class='row'><h2>{html.escape(rec['tma'])} | slide {html.escape(rec['slide'])}</h2>",
                "<div class='meta'>"
                f"Crop policy: <code>{html.escape(rec['crop_policy'])}</code>. "
                f"Old H&amp;E bbox canvas: <code>{html.escape(rec['old_he_size'])}</code>; expanded canvas: <code>{html.escape(rec['expanded_lowres_size'])}</code>; high-res H&amp;E: <code>{html.escape(rec['he_size'])}</code>. "
                f"FICTURE bbox in old H&amp;E coordinates: <code>{html.escape(rec['ficture_bbox_in_old_he_xyxy'])}</code>. "
                f"Raw member: <code>{html.escape(rec['raw_he_member'])}</code>. "
                f"Raw bbox: <code>{html.escape(rec['raw_he_bbox_unrotated_xyxy'])}</code>."
                "</div>",
                "<div class='primary-grid'>",
                f"<figure><figcaption>FICTURE-driven high-resolution H&amp;E ROI</figcaption><a href='{html.escape(rec['he_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['he_rel'])}'></a></figure>",
                f"<figure><figcaption>K=12 molecule-level FICTURE, coordinate-aligned</figcaption><a href='{html.escape(rec['ficture_high_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['ficture_high_rel'])}'></a></figure>",
                "<aside class='legend-panel'><h3>FICTURE factor legend</h3>",
                pretty_factor_legend_html(rec["factor_legend"]),
                "</aside></div>",
                "<details class='gt-details'><summary>GT annotation on H&amp;E background: original vs continuous region</summary><div class='details-body'>",
                f"<p class='muted'>Original GT is the raw <code>new_annotation</code> cell-level points from the source CSV. Continuous region GT is generated from those same annotations by matching to Xenium cell boundaries, applying a small closing radius of {REGION_CLOSE_RADIUS_PX}px to make the boundary continuous, and filling enclosed holes. There is no final dilation step.</p>",
                "<div class='gt-grid'>",
                f"<figure><figcaption>Original GT annotation points on H&amp;E</figcaption><a href='{html.escape(rec['gt_original_overlay_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['gt_original_overlay_rel'])}'></a></figure>",
                f"<figure><figcaption>Continuous GT region on H&amp;E</figcaption><a href='{html.escape(rec['gt_continuous_overlay_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['gt_continuous_overlay_rel'])}'></a></figure>",
                "</div><h3>Annotation labels</h3>",
                annotation_legend_html(rec["annotation_counts"]),
                "</div></details>",
                "<details class='gt-details'><summary>GT annotation mask only: original vs continuous region</summary><div class='details-body'>",
                "<div class='gt-grid'>",
                f"<figure><figcaption>Original GT points, blank background</figcaption><a href='{html.escape(rec['gt_original_mask_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['gt_original_mask_rel'])}'></a></figure>",
                f"<figure><figcaption>Continuous GT region mask, blank background</figcaption><a href='{html.escape(rec['gt_continuous_mask_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['gt_continuous_mask_rel'])}'></a></figure>",
                "</div></div></details>",
                f"<p class='muted'>Two-panel H&amp;E/FICTURE preview: <a href='{html.escape(rec['preview_panel_rel'])}'>open</a>.</p>",
                "</section>",
            ]
        )
    html_parts.append("</body></html>")
    (OUT_ROOT / "index.html").write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Wrote {len(records)} FICTURE-driven high-res TMA rows to {OUT_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
