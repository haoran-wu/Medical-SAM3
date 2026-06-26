#!/usr/bin/env python3
"""Whole-slide translation-first morphology.ome.tif to H&E alignment.

This aligns an entire Xenium Silica morphology.ome.tif slide to the matching
whole-slide H&E image after H&E rotate-90 plus left-right mirror. The only local
registration transform accepted by default is translation; scale is reported as
a diagnostic and used only if it clears an explicit improvement threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import feature, measure, morphology

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_xenium_silica_global_aligned_nomargin as base
import render_xenium_silica_local_translation_nomargin as localreg
import xenium_tif_he_translation_registration as roi_align


LOCAL_DATA = Path("/Volumes/Disk/ST_annotated_datasets")
base.HE_ZIP = LOCAL_DATA / "Xenium_Silica" / "OneDrive_1_4-24-2026.zip"
base.XENIUM_ROOT = LOCAL_DATA / "Xenium_Silica"
localreg.RAW_XENIUM_ROOT = LOCAL_DATA / "tmp_Xenium"

OUT_ROOT = Path("output/xenium_silica_wholeslide_tif_he_translation_registration_20260624")
MORPH_LEVEL = 4
ORIENTATION = "rot90_ccw_lr_mirror"
SCALE_ACCEPT_DELTA = 0.035
CORE_GRID_COLS = 3
CORE_GRID_ROWS = 8
CORE_HE_THRESHOLD = 247
CORE_HE_SATURATION_THRESHOLD = 6
CORE_MORPH_THRESHOLD = 1
CORE_HE_RADIUS_PAD = 1.08
CORE_MORPH_RADIUS_PAD = 1.12


def clean_slide_id(value: str) -> str:
    return value.strip()


def load_he_oriented_grid(slide: base.SlideConfig, target_shape_yx: tuple[int, int]) -> tuple[np.ndarray, dict]:
    target_h, target_w = target_shape_yx
    he_h, he_w = slide.he_shape[:2]
    # After rot90+LR mirror, H&E dimensions become height=he_w, width=he_h.
    stride_y = he_w / float(target_h)
    stride_x = he_h / float(target_w)
    stride = max(1, int(round(max(stride_y, stride_x))))
    arr = base.he_memmap(slide)
    thumb = np.asarray(arr[::stride, ::stride, :]).copy()
    oriented = roi_align.orient_he_crop(thumb, ORIENTATION)
    he_grid = np.asarray(Image.fromarray(oriented).resize((target_w, target_h), Image.Resampling.BILINEAR))
    return he_grid, {
        "orientation": ORIENTATION,
        "he_read_stride_raw_px": int(stride),
        "he_raw_shape_yx": [int(he_h), int(he_w)],
        "he_oriented_sample_shape_yx": [int(oriented.shape[0]), int(oriented.shape[1])],
        "target_morphology_shape_yx": [int(target_h), int(target_w)],
        "dimension_resample_note": "H&E is resized only to compare on the morphology level grid; local transform remains translation-first.",
    }


def multi_component_largest_filter(mask: np.ndarray, min_size: int) -> np.ndarray:
    clean = morphology.remove_small_objects(mask.astype(bool), min_size)
    lab = measure.label(clean)
    if lab.max() == 0:
        return clean
    props = measure.regionprops(lab)
    # Keep all sizeable TMA cores, not just the largest one.
    areas = np.array([p.area for p in props], dtype=float)
    floor = max(float(min_size), float(np.percentile(areas, 20)) * 0.25)
    keep_labels = {p.label for p in props if p.area >= floor}
    return np.isin(lab, list(keep_labels))


def centers_to_edges(centers: np.ndarray, max_value: int) -> np.ndarray:
    centers = np.array(sorted(float(v) for v in centers), dtype=float)
    if len(centers) < 2:
        raise ValueError("Need at least two centers to infer slot edges")
    edges = np.zeros(len(centers) + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = max(0.0, centers[0] - (centers[1] - centers[0]) / 2.0)
    edges[-1] = min(float(max_value), centers[-1] + (centers[-1] - centers[-2]) / 2.0)
    return edges


def he_core_footprint_seed(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    mean = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    return ((mean < CORE_HE_THRESHOLD) & (saturation > CORE_HE_SATURATION_THRESHOLD)) | (mean < 238)


def morphology_core_footprint_seed(gray: np.ndarray) -> np.ndarray:
    return ndi.gaussian_filter(gray.astype(np.float32), 2.0) > CORE_MORPH_THRESHOLD


def ellipse_footprint_from_seed(
    seed: np.ndarray,
    *,
    radius_pad: float,
    grid_cols: int = CORE_GRID_COLS,
    grid_rows: int = CORE_GRID_ROWS,
    min_pixels: int = 20,
    radius_floor: float = 0.88,
    radius_ceil: float = 1.25,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """Represent each TMA punch by an ellipse footprint on the full-slide grid."""
    dilated = ndi.binary_dilation(seed.astype(bool), iterations=2)
    x_projection = ndi.gaussian_filter1d(dilated.sum(axis=0).astype(float), 35)
    y_projection = ndi.gaussian_filter1d(dilated.sum(axis=1).astype(float), 35)
    x_centers = base.weighted_kmeans_1d(x_projection, grid_cols)
    y_centers = base.weighted_kmeans_1d(y_projection, grid_rows)
    x_edges = centers_to_edges(x_centers, seed.shape[1])
    y_edges = centers_to_edges(y_centers, seed.shape[0])

    records: list[list[float]] = []
    radii: list[tuple[float, float]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = int(x_edges[col])
            x1 = int(x_edges[col + 1])
            y0 = int(y_edges[row])
            y1 = int(y_edges[row + 1])
            yy, xx = np.where(seed[y0:y1, x0:x1])
            if len(xx) < min_pixels:
                continue
            gx = x0 + xx
            gy = y0 + yy
            cx = (float(gx.min()) + float(gx.max()) + 1.0) / 2.0
            cy = (float(gy.min()) + float(gy.max()) + 1.0) / 2.0
            rx = (float(gx.max()) - float(gx.min()) + 1.0) * 0.5 * radius_pad
            ry = (float(gy.max()) - float(gy.min()) + 1.0) * 0.5 * radius_pad
            records.append([float(col), float(row), cx, cy, rx, ry, float(len(xx))])
            radii.append((rx, ry))

    if not records:
        raise RuntimeError("No TMA core footprint records were detected")

    median_rx = float(np.median([v[0] for v in radii]))
    median_ry = float(np.median([v[1] for v in radii]))
    out = np.zeros(seed.shape, dtype=bool)
    result: dict[tuple[int, int], dict] = {}
    for col_f, row_f, cx, cy, rx, ry, n_pixels in records:
        rx = min(max(rx, median_rx * radius_floor), median_rx * radius_ceil)
        ry = min(max(ry, median_ry * radius_floor), median_ry * radius_ceil)
        col = int(col_f)
        row = int(row_f)
        x0 = max(0, int(cx - rx - 3))
        x1 = min(seed.shape[1], int(cx + rx + 4))
        y0 = max(0, int(cy - ry - 3))
        y1 = min(seed.shape[0], int(cy + ry + 4))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        out[y0:y1, x0:x1] |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        result[(col, row)] = {
            "col": col,
            "row": row,
            "cx": float(cx),
            "cy": float(cy),
            "rx": float(rx),
            "ry": float(ry),
            "seed_pixels": int(n_pixels),
        }
    return out, result


def fit_affine_xy(moving_records: dict[tuple[int, int], dict], fixed_records: dict[tuple[int, int], dict]) -> tuple[np.ndarray, dict]:
    keys = sorted(set(moving_records) & set(fixed_records), key=lambda key: (key[1], key[0]))
    if len(keys) < 6:
        raise RuntimeError(f"Only {len(keys)} matched core centers were found; cannot fit a stable whole-slide affine")
    moving_xy = np.array([[moving_records[key]["cx"], moving_records[key]["cy"]] for key in keys], dtype=float)
    fixed_xy = np.array([[fixed_records[key]["cx"], fixed_records[key]["cy"]] for key in keys], dtype=float)
    design = np.column_stack([moving_xy[:, 0], moving_xy[:, 1], np.ones(len(keys), dtype=float)])
    ax = np.linalg.lstsq(design, fixed_xy[:, 0], rcond=None)[0]
    ay = np.linalg.lstsq(design, fixed_xy[:, 1], rcond=None)[0]
    affine_xy = np.array([[ax[0], ax[1], ax[2]], [ay[0], ay[1], ay[2]]], dtype=float)
    predicted = np.column_stack([design @ ax, design @ ay])
    residual = np.abs(predicted - fixed_xy)
    return affine_xy, {
        "matched_core_count": int(len(keys)),
        "matched_slots": [{"col": int(col), "row": int(row)} for col, row in keys],
        "center_residual_mean_xy_px": [float(v) for v in residual.mean(axis=0)],
        "center_residual_max_xy_px": [float(v) for v in residual.max(axis=0)],
    }


def affine_inverse_params(affine_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix_xy = np.asarray(affine_xy[:, :2], dtype=float)
    offset_xy = np.asarray(affine_xy[:, 2], dtype=float)
    inv_xy = np.linalg.inv(matrix_xy)
    inv_offset_xy = -inv_xy @ offset_xy
    # scipy.ndimage expects output coordinates as (y, x) and returns input (y, x).
    matrix_yx = np.array([[inv_xy[1, 1], inv_xy[1, 0]], [inv_xy[0, 1], inv_xy[0, 0]]], dtype=float)
    offset_yx = np.array([inv_offset_xy[1], inv_offset_xy[0]], dtype=float)
    return matrix_yx, offset_yx


def warp_affine_array(arr: np.ndarray, affine_xy: np.ndarray, order: int) -> np.ndarray:
    matrix_yx, offset_yx = affine_inverse_params(affine_xy)
    if arr.ndim == 2:
        return ndi.affine_transform(
            arr.astype(float),
            matrix_yx,
            offset=offset_yx,
            output_shape=arr.shape,
            order=order,
            mode="constant",
            cval=0,
        )
    channels = [
        ndi.affine_transform(
            arr[:, :, idx].astype(float),
            matrix_yx,
            offset=offset_yx,
            output_shape=arr.shape[:2],
            order=order,
            mode="constant",
            cval=0,
        )
        for idx in range(arr.shape[2])
    ]
    return np.stack(channels, axis=2)


def warp_affine_mask(mask: np.ndarray, affine_xy: np.ndarray) -> np.ndarray:
    return warp_affine_array(mask.astype(np.float32), affine_xy, order=0) > 0.5


def warp_affine_image_u8(gray: np.ndarray, affine_xy: np.ndarray) -> np.ndarray:
    return roi_align.normalize_uint8(warp_affine_array(gray.astype(np.float32), affine_xy, order=1))


def apply_affine_to_points(xy: np.ndarray, affine_xy: np.ndarray) -> np.ndarray:
    matrix_xy = np.asarray(affine_xy[:, :2], dtype=float)
    offset_xy = np.asarray(affine_xy[:, 2], dtype=float)
    return xy.astype(float) @ matrix_xy.T + offset_xy


def core_footprint_affine_alignment(he_rgb: np.ndarray, morph_u8: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    he_seed = he_core_footprint_seed(he_rgb)
    morph_seed = morphology_core_footprint_seed(morph_u8)
    he_core, he_records = ellipse_footprint_from_seed(he_seed, radius_pad=CORE_HE_RADIUS_PAD)
    morph_core, morph_records = ellipse_footprint_from_seed(morph_seed, radius_pad=CORE_MORPH_RADIUS_PAD)
    affine_xy, fit_metrics = fit_affine_xy(morph_records, he_records)
    aligned_morph_core = warp_affine_mask(morph_core, affine_xy)
    dice = float((2.0 * (aligned_morph_core & he_core).sum()) / (float(aligned_morph_core.sum() + he_core.sum()) + 1e-6))
    leakage = float(
        (aligned_morph_core & ~ndi.binary_dilation(he_core, iterations=12)).sum()
        / (float(aligned_morph_core.sum()) + 1e-6)
    )
    he_texture_mask = he_wholeslide_mask(he_rgb)
    morph_texture_mask = morphology_wholeslide_mask(morph_u8)
    aligned_morph_texture = warp_affine_mask(morph_texture_mask, affine_xy)
    texture_dice = float(
        (2.0 * (aligned_morph_texture & he_texture_mask).sum())
        / (float(aligned_morph_texture.sum() + he_texture_mask.sum()) + 1e-6)
    )
    texture_leakage = float(
        (aligned_morph_texture & ~morphology.binary_dilation(he_texture_mask, morphology.disk(12))).sum()
        / (float(aligned_morph_texture.sum()) + 1e-6)
    )
    metrics = {
        "transform_type": "global_affine_from_3x8_tma_core_footprints",
        "affine_xy_from_morphology_to_oriented_he": affine_xy.tolist(),
        "matrix_terms": {
            "x_from_x": float(affine_xy[0, 0]),
            "x_from_y": float(affine_xy[0, 1]),
            "x_offset": float(affine_xy[0, 2]),
            "y_from_x": float(affine_xy[1, 0]),
            "y_from_y": float(affine_xy[1, 1]),
            "y_offset": float(affine_xy[1, 2]),
        },
        "core_footprint_dice": dice,
        "core_footprint_leakage": leakage,
        "texture_tissue_dice_after_affine": texture_dice,
        "texture_tissue_leakage_after_affine": texture_leakage,
        "he_core_footprint_pixels": int(he_core.sum()),
        "morphology_core_footprint_pixels": int(morph_core.sum()),
        "morphology_core_footprint_pixels_after_affine": int(aligned_morph_core.sum()),
        "seed_parameters": {
            "grid_cols": CORE_GRID_COLS,
            "grid_rows": CORE_GRID_ROWS,
            "he_threshold": CORE_HE_THRESHOLD,
            "he_saturation_threshold": CORE_HE_SATURATION_THRESHOLD,
            "morphology_threshold": CORE_MORPH_THRESHOLD,
            "he_radius_pad": CORE_HE_RADIUS_PAD,
            "morphology_radius_pad": CORE_MORPH_RADIUS_PAD,
        },
        **fit_metrics,
    }
    return metrics, he_core, morph_core, aligned_morph_core


def he_wholeslide_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean = arr.mean(axis=2)
    mask = (mean < 247) & ~((r > 245) & (g > 245) & (b > 245))
    mask &= ((r - g > 2) | (b - g > 0) | (mean < 232))
    radius = max(2, min(mask.shape) // 850)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(radius))
    mask = multi_component_largest_filter(mask, min_size=max(800, mask.size // 8000))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def morphology_wholeslide_mask(gray: np.ndarray) -> np.ndarray:
    arr = gray.astype(np.float32)
    smooth = ndi.gaussian_filter(arr, 2.0)
    vals = smooth[smooth > 0]
    if vals.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    mask = smooth > np.percentile(vals, 20)
    radius = max(2, min(mask.shape) // 900)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(radius))
    mask = multi_component_largest_filter(mask, min_size=max(800, mask.size // 8000))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def build_wholeslide_features(he_rgb: np.ndarray, morph_u8: np.ndarray) -> tuple[dict, dict, float]:
    he_mask = he_wholeslide_mask(he_rgb)
    morph_mask = morphology_wholeslide_mask(morph_u8)
    he_sig = roi_align.normalize01(0.72 * he_mask.astype(np.float32) + 0.28 * roi_align.he_tissue_signal(he_rgb))
    mo_sig = roi_align.normalize01(0.72 * morph_mask.astype(np.float32) + 0.28 * roi_align.morph_signal(morph_u8))
    he_sig_ds, scale = roi_align.resize_for_analysis(he_sig, order=1)
    he_mask_ds = roi_align.resize_for_analysis(he_mask, order=0)[0].astype(bool)
    mo_sig_ds = roi_align.resize_for_analysis(mo_sig, order=1)[0].astype(np.float32)
    mo_mask_ds = roi_align.resize_for_analysis(morph_mask, order=0)[0].astype(bool)
    he_edge = roi_align.boundary(he_mask_ds, radius=2) | feature.canny(he_sig_ds, sigma=1.2, low_threshold=0.05, high_threshold=0.18)
    mo_edge = roi_align.boundary(mo_mask_ds, radius=2) | feature.canny(mo_sig_ds, sigma=1.2, low_threshold=0.05, high_threshold=0.18)
    he = {
        "signal": he_sig_ds.astype(np.float32),
        "mask": he_mask_ds,
        "edge": he_edge,
        "soft_edge": roi_align.normalize01(ndi.gaussian_filter(he_edge.astype(np.float32), 2.0)),
        "safe_mask": morphology.binary_dilation(he_mask_ds, morphology.disk(8)),
    }
    moving = {
        "signal": mo_sig_ds.astype(np.float32),
        "mask": mo_mask_ds,
        "edge": mo_edge,
    }
    return he, moving, scale


def full_grid_metrics(he_rgb: np.ndarray, morph_u8: np.ndarray, shift_yx: tuple[float, float], scale: float) -> dict:
    he_mask = he_wholeslide_mask(he_rgb)
    morph_mask = morphology_wholeslide_mask(morph_u8)
    shifted = roi_align.transform_moving_mask(morph_mask, shift_yx, scale)
    dice = float((2.0 * (shifted & he_mask).sum()) / (float(shifted.sum() + he_mask.sum()) + 1e-6))
    leakage = float((shifted & ~morphology.binary_dilation(he_mask, morphology.disk(12))).sum() / (float(shifted.sum()) + 1e-6))
    return {
        "full_grid_dice": dice,
        "full_grid_leakage": leakage,
        "he_mask_pixels": int(he_mask.sum()),
        "morphology_mask_pixels_after_transform": int(shifted.sum()),
    }


def apply_point_transform(xy: np.ndarray, shape_yx: tuple[int, int], shift_yx: tuple[float, float], scale: float) -> np.ndarray:
    pts = xy.copy().astype(float)
    if abs(scale - 1.0) > 1e-6:
        h, w = shape_yx
        center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=float)
        pts = (pts - center) * scale + center
    pts[:, 0] += shift_yx[1]
    pts[:, 1] += shift_yx[0]
    return pts


def overlay_morph_on_he_affine(he_rgb: np.ndarray, morph_u8: np.ndarray, affine_xy: np.ndarray) -> Image.Image:
    shifted = warp_affine_image_u8(morph_u8, affine_xy)
    he = he_rgb.astype(np.float32)
    mask = shifted > np.percentile(shifted[shifted > 0], 35) if np.any(shifted > 0) else shifted > 0
    out = he.copy()
    color_arr = np.zeros_like(out)
    color_arr[:, :, 1] = 255
    color_arr[:, :, 2] = 160
    alpha = np.clip(shifted.astype(np.float32) / 255.0, 0, 0.55)
    alpha = alpha * mask.astype(np.float32)
    out = out * (1 - alpha[:, :, None]) + color_arr * alpha[:, :, None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def footprint_mask_overlay(he_rgb: np.ndarray, he_core: np.ndarray, aligned_morph_core: np.ndarray) -> Image.Image:
    base_img = Image.fromarray(he_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    arr = np.zeros((he_core.shape[0], he_core.shape[1], 4), dtype=np.uint8)
    overlap = he_core & aligned_morph_core
    he_only = he_core & ~aligned_morph_core
    morph_only = aligned_morph_core & ~he_core
    arr[overlap] = (40, 180, 120, 95)
    arr[he_only] = (230, 40, 110, 105)
    arr[morph_only] = (0, 205, 230, 110)
    overlay = Image.fromarray(arr, "RGBA")
    return Image.alpha_composite(base_img, overlay).convert("RGB")


def load_slide_rows(slide: base.SlideConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tma in range(slide.first_tma, slide.last_tma + 1):
        try:
            rows.extend(base.read_tma_frame(tma))
        except FileNotFoundError:
            continue
    return rows


def point_coordinates(rows: list[dict[str, str]], meta: dict) -> np.ndarray:
    coords = np.array([[float(row["coordinateX"]), float(row["coordinateY"])] for row in rows], dtype=float)
    return np.column_stack([coords[:, 0] * meta["scale_x_px_per_coord"], coords[:, 1] * meta["scale_y_px_per_coord"]])


def paint_points(canvas: Image.Image, xy: np.ndarray, labels: list[str], palette: dict[str, str], alpha: int, radius: int) -> Image.Image:
    out = canvas.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for (x, y), label in zip(xy, labels):
        color = palette.get(label)
        if color is None or not np.isfinite(x) or not np.isfinite(y):
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if -radius <= xi < out.width + radius and -radius <= yi < out.height + radius:
            draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=(*base.rgb_tuple(color), alpha))
    return Image.alpha_composite(out, overlay)


def write_slide_triptych(he: Image.Image, cell_map: Image.Image, annotation: Image.Image, out_path: Path) -> None:
    roi_align.write_panel(
        [
            (he, "Whole-slide rotated H&E", "rotate 90 + left-right mirror"),
            (cell_map, "Whole-slide FICTURE-style cell type", "Xenium cellTypeFibSim using whole-slide tif-to-H&E transform"),
            (annotation, "Whole-slide annotation overlay", "new_annotation using same transform"),
        ],
        out_path,
        tile_size=(520, 700),
    )


def run_slide(histo: str, out_root: Path) -> dict:
    histo = clean_slide_id(histo)
    slide = base.SLIDES[histo]
    out_root.mkdir(parents=True, exist_ok=True)
    morph_meta = roi_align.load_morphology_meta(histo, MORPH_LEVEL)
    morph_shape = tuple(int(v) for v in morph_meta["shape_yx"])
    he_grid, he_info = load_he_oriented_grid(slide, morph_shape)
    morph_grid = roi_align.load_morphology_crop(morph_meta, [0, 0, morph_shape[1], morph_shape[0]])

    he_features, moving_features, analysis_scale = build_wholeslide_features(he_grid, morph_grid)
    translation = roi_align.search_translation(he_features, moving_features)
    scale_diag = roi_align.scale_diagnostic(he_features, moving_features)
    translation_shift_full = (float(translation["shift_yx"][0] / analysis_scale), float(translation["shift_yx"][1] / analysis_scale))
    translation_full_metrics = full_grid_metrics(he_grid, morph_grid, translation_shift_full, 1.0)
    accepted_scale = 1.0
    if scale_diag["best"]["scale"] != 1.0 and scale_diag["best"]["score"] > translation["score"] + SCALE_ACCEPT_DELTA:
        accepted_scale = float(scale_diag["best"]["scale"])
    chosen = translation if accepted_scale == 1.0 else scale_diag["best"]
    shift_full = (float(chosen["shift_yx"][0] / analysis_scale), float(chosen["shift_yx"][1] / analysis_scale))
    metrics = {
        **chosen,
        "analysis_scale": float(analysis_scale),
        "alignment_grid_shift_yx": [shift_full[0], shift_full[1]],
        "alignment_grid_shift_xy": [shift_full[1], shift_full[0]],
        **full_grid_metrics(he_grid, morph_grid, shift_full, accepted_scale),
    }
    core_metrics, he_core, _morph_core, aligned_morph_core = core_footprint_affine_alignment(he_grid, morph_grid)
    affine_xy = np.array(core_metrics["affine_xy_from_morphology_to_oriented_he"], dtype=float)

    translation_overlay = roi_align.overlay_morph_on_he(he_grid, morph_grid, translation_shift_full, 1.0)
    scale_best = scale_diag["best"]
    scale_overlay = roi_align.overlay_morph_on_he(
        he_grid,
        morph_grid,
        (float(scale_best["shift_yx"][0] / analysis_scale), float(scale_best["shift_yx"][1] / analysis_scale)),
        float(scale_best["scale"]),
    )
    affine_overlay = overlay_morph_on_he_affine(he_grid, morph_grid, affine_xy)
    footprint_overlay = footprint_mask_overlay(he_grid, he_core, aligned_morph_core)

    he_img = Image.fromarray(he_grid).convert("RGB")
    morph_img = Image.fromarray(morph_grid).convert("RGB")
    qc_path = out_root / f"{histo}_wholeslide_tif_he_alignment_qc.png"
    roi_align.write_panel(
        [
            (he_img, "Whole-slide rotated H&E", f"grid {he_grid.shape[1]}x{he_grid.shape[0]}"),
            (morph_img, "Whole-slide morphology.ome.tif", f"level {MORPH_LEVEL}; grid {morph_grid.shape[1]}x{morph_grid.shape[0]}"),
            (
                scale_overlay,
                "Texture-mask baseline",
                f"scale={accepted_scale:.2f}; texture Dice={metrics['full_grid_dice']:.3f}",
            ),
            (
                affine_overlay,
                "Core-footprint affine overlay",
                f"core Dice={core_metrics['core_footprint_dice']:.3f}; leakage={core_metrics['core_footprint_leakage']:.3f}",
            ),
            (
                footprint_overlay,
                "Core-footprint QC mask",
                "green=overlap; magenta=H&E only; cyan=morphology only",
            ),
        ],
        qc_path,
        tile_size=(520, 700),
    )

    rows = load_slide_rows(slide)
    xy = apply_affine_to_points(point_coordinates(rows, morph_meta), affine_xy)
    cell_canvas = Image.new("RGBA", (morph_shape[1], morph_shape[0]), (255, 255, 255, 255))
    cell_map = paint_points(cell_canvas, xy, [row.get(base.CELLTYPE_COL, "") for row in rows], base.CELLTYPE_PALETTE, alpha=225, radius=1).convert("RGB")
    he_cell_overlay = paint_points(he_img, xy, [row.get(base.CELLTYPE_COL, "") for row in rows], base.CELLTYPE_PALETTE, alpha=95, radius=1).convert("RGB")
    annotation_overlay = paint_points(he_img, xy, [base.clean_label(row.get(base.GT_COL, "")) for row in rows], base.GT_PALETTE, alpha=230, radius=2).convert("RGB")

    he_path = out_root / f"{histo}_wholeslide_rotated_he_grid.png"
    morph_path = out_root / f"{histo}_wholeslide_morphology_grid.png"
    cell_path = out_root / f"{histo}_wholeslide_ficture_style_celltype_map.png"
    cell_overlay_path = out_root / f"{histo}_wholeslide_he_celltype_overlay.png"
    annotation_path = out_root / f"{histo}_wholeslide_annotation_overlay.png"
    triptych_path = out_root / f"{histo}_wholeslide_he_ficture_annotation_triptych.png"
    footprint_path = out_root / f"{histo}_wholeslide_core_footprint_alignment_mask.png"
    he_img.save(he_path)
    morph_img.save(morph_path)
    cell_map.save(cell_path)
    he_cell_overlay.save(cell_overlay_path)
    annotation_overlay.save(annotation_path)
    footprint_overlay.save(footprint_path)
    write_slide_triptych(he_img, cell_map, annotation_overlay, triptych_path)

    translation_only_pass = bool(translation_full_metrics["full_grid_dice"] >= 0.80 and translation_full_metrics["full_grid_leakage"] <= 0.12)
    selected_alignment_pass = bool(core_metrics["core_footprint_dice"] >= 0.95 and core_metrics["core_footprint_leakage"] <= 0.05)
    manifest = {
        "histo_slide": histo,
        "alignment_pass": selected_alignment_pass,
        "translation_only_pass": translation_only_pass,
        "inputs": {
            "he_zip": str(base.HE_ZIP),
            "morphology_ome_tif": morph_meta["path"],
            "xenium_root": str(base.XENIUM_ROOT),
        },
        "he_grid": he_info,
        "morphology": morph_meta,
        "selected_transform": {
            "orientation": ORIENTATION,
            "morphology_level": MORPH_LEVEL,
            "local_transform_policy": "rot90_ccw_lr_mirror plus one whole-slide global affine fitted from 3x8 TMA core footprints; no per-TMA local deformation",
            "transform_type": core_metrics["transform_type"],
            "affine_xy_from_morphology_to_oriented_he": core_metrics["affine_xy_from_morphology_to_oriented_he"],
            "matrix_terms": core_metrics["matrix_terms"],
            "metrics": core_metrics,
            "texture_mask_scale_translation_baseline": {
                "scale": accepted_scale,
                "dx_alignment_grid_px": shift_full[1],
                "dy_alignment_grid_px": shift_full[0],
                "metrics": metrics,
            },
            "translation_only": {
                "scale": 1.0,
                "dx_alignment_grid_px": translation_shift_full[1],
                "dy_alignment_grid_px": translation_shift_full[0],
                "metrics": {
                    **translation,
                    "analysis_scale": float(analysis_scale),
                    "alignment_grid_shift_yx": [translation_shift_full[0], translation_shift_full[1]],
                    "alignment_grid_shift_xy": [translation_shift_full[1], translation_shift_full[0]],
                    **translation_full_metrics,
                },
            },
            "scale_diagnostic": scale_diag,
        },
        "n_cells_rendered": len(rows),
        "outputs": {
            "qc_panel": str(qc_path),
            "rotated_he_grid": str(he_path),
            "morphology_grid": str(morph_path),
            "ficture_style_celltype_map": str(cell_path),
            "he_celltype_overlay": str(cell_overlay_path),
            "annotation_overlay": str(annotation_path),
            "core_footprint_overlay": str(footprint_path),
            "triptych": str(triptych_path),
        },
    }
    manifest_path = out_root / f"{histo}_wholeslide_tif_to_he_transform.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", default="771-1", help="Comma-separated histo slide ids, e.g. 770-1,771-1")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    records = []
    for histo in [v.strip() for v in args.slides.split(",") if v.strip()]:
        manifest = run_slide(histo, args.out_root)
        st = manifest["selected_transform"]
        records.append(
            {
                "histo_slide": histo,
                "alignment_pass": manifest["alignment_pass"],
                "transform_type": st["transform_type"],
                "affine_xy_from_morphology_to_oriented_he": st["affine_xy_from_morphology_to_oriented_he"],
                "core_footprint_dice": st["metrics"]["core_footprint_dice"],
                "core_footprint_leakage": st["metrics"]["core_footprint_leakage"],
                "texture_tissue_dice_after_affine": st["metrics"]["texture_tissue_dice_after_affine"],
                "texture_tissue_leakage_after_affine": st["metrics"]["texture_tissue_leakage_after_affine"],
                "manifest": str(args.out_root / f"{histo}_wholeslide_tif_to_he_transform.json"),
                "qc_panel": manifest["outputs"]["qc_panel"],
                "triptych": manifest["outputs"]["triptych"],
            }
        )
    summary = {"records": records}
    summary_path = args.out_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(args.out_root), "batch_summary": str(summary_path), "records": records}, indent=2))


if __name__ == "__main__":
    main()
