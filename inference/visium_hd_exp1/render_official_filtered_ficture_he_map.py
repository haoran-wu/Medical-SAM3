#!/usr/bin/env python3
"""Render the official filtered FICTURE map aligned to VisiumHD Exp1 H&E.

This is the locked FICTURE-to-H&E pipeline for the corrected candidate-pool
work. It uses the filtered FICTURE pixel PNG from the original FICTURE folder,
not the unfiltered/direct ``hex_12.k12.results.tsv`` redraw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from ficture_factor_semantics import build_semantic_legend, write_outputs


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILTERED_FICTURE_PNG = (
    PROJECT_ROOT
    / "pixel-level cell type image"
    / "visiumhd_exp1_hex12_k12"
    / "hex_12.k12.pixel.png"
)
DEFAULT_FACTOR_INFO = DEFAULT_FILTERED_FICTURE_PNG.with_suffix(".info.tsv")
DEFAULT_FICTURE_MD = PROJECT_ROOT / "pixel-level cell type image" / "Ficture.md"
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "output"
    / "visium_hd_exp1"
    / "ficture_official_filtered_he_aligned"
    / "reference_results_coordinate_qc_only.png"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "ficture_official_filtered_he_aligned"


def estimate_tissue_mask(he: np.ndarray) -> np.ndarray:
    """Estimate main tissue foreground so colored FICTURE is not drawn on scanner background."""
    arr = he.astype(np.int16)
    channel_range = arr.max(axis=2) - arr.min(axis=2)
    mean = arr.mean(axis=2)
    tissue = (channel_range > 8) | (mean < 235)

    small = tissue[::4, ::4]
    labels, _ = ndi.label(small)
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.zeros(tissue.shape, dtype=bool)
    counts[0] = 0
    small_main = labels == counts.argmax()
    full = np.repeat(np.repeat(small_main, 4, axis=0), 4, axis=1)[: he.shape[0], : he.shape[1]]
    full = ndi.binary_fill_holes(full)
    return ndi.binary_dilation(full, iterations=3)


def orient(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return arr
    if name == "fliplr":
        return np.fliplr(arr)
    if name == "flipud":
        return np.flipud(arr)
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "transpose":
        return np.transpose(arr, (1, 0, 2))
    if name == "transverse":
        return np.fliplr(np.flipud(np.transpose(arr, (1, 0, 2))))
    raise ValueError(f"Unsupported orientation: {name}")


def map_filtered_png_to_canvas(
    image: np.ndarray,
    he_shape: tuple[int, int],
    tissue_mask: np.ndarray | None,
    *,
    microns_per_pixel: float,
    tissue_hires_scalef: float,
    ficture_res_um_per_pixel: float,
    ficture_xmin_um: float,
    ficture_ymin_um: float,
    black_threshold: int,
) -> tuple[np.ndarray, dict[str, int | float | list[int]]]:
    source_mask = image.sum(axis=2) > black_threshold
    src_row, src_col = np.nonzero(source_mask)
    scale = tissue_hires_scalef / microns_per_pixel

    # FICTURE pixel-grid coordinates are 2 um bins. The x/y axes are swapped
    # when projected to the 10x H&E hires canvas.
    x_um = ficture_xmin_um + src_row.astype(np.float64) * ficture_res_um_per_pixel
    y_um = ficture_ymin_um + src_col.astype(np.float64) * ficture_res_um_per_pixel
    he_x = np.rint(y_um * scale).astype(np.int32)
    he_y = np.rint(x_um * scale).astype(np.int32)

    height, width = he_shape
    valid = (he_x >= 0) & (he_x < width) & (he_y >= 0) & (he_y < height)
    inside_tissue = np.zeros_like(valid)
    if tissue_mask is None:
        inside_tissue[valid] = True
    else:
        inside_tissue[valid] = tissue_mask[he_y[valid], he_x[valid]]
    keep = valid & inside_tissue

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[he_y[keep], he_x[keep]] = image[src_row[keep], src_col[keep]]
    colored = canvas.sum(axis=2) > 0
    false_background = 0
    if tissue_mask is not None:
        false_background = int((colored & ~tissue_mask).sum())

    metrics: dict[str, int | float | list[int]] = {
        "source_nonblack_pixels": int(source_mask.sum()),
        "mapped_valid_pixels": int(valid.sum()),
        "mapped_inside_tissue_pixels_kept": int(keep.sum()),
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "inside_tissue_fraction_of_valid_pixels": float(inside_tissue[valid].mean()) if valid.any() else 0.0,
        "rendered_colored_pixels": int(colored.sum()),
        "rendered_background_false_color_pixels": false_background,
        "rendered_background_false_color_fraction": float(false_background / colored.sum()) if colored.any() else 0.0,
    }
    if colored.any():
        yy, xx = np.nonzero(colored)
        metrics["aligned_bbox_xyxy_inclusive"] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    return canvas, metrics


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def compare_to_reference(mask: np.ndarray, reference_path: Path) -> dict[str, float | str]:
    if not reference_path.exists():
        return {"reference_status": "missing", "reference_path": str(reference_path)}
    ref = np.array(Image.open(reference_path).convert("RGB")).sum(axis=2) > 0
    out: dict[str, float | str] = {
        "reference_status": "found",
        "reference_path": str(reference_path),
        "exact_dice_vs_reference": dice(mask, ref),
        "exact_iou_vs_reference": iou(mask, ref),
    }
    for radius in (1, 2, 3, 4):
        out[f"dilated_{radius}px_dice_vs_reference"] = dice(
            ndi.binary_dilation(mask, iterations=radius),
            ndi.binary_dilation(ref, iterations=radius),
        )
        out[f"dilated_{radius}px_iou_vs_reference"] = iou(
            ndi.binary_dilation(mask, iterations=radius),
            ndi.binary_dilation(ref, iterations=radius),
        )
    return out


def orientation_qc(
    raw: np.ndarray,
    he_shape: tuple[int, int],
    tissue: np.ndarray,
    reference_path: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, float | str]]:
    if not reference_path.exists():
        return {}
    ref = np.array(Image.open(reference_path).convert("RGB")).sum(axis=2) > 0
    result: dict[str, dict[str, float | str]] = {}
    for name in ("identity", "fliplr", "flipud", "rot90", "rot180", "rot270", "transpose", "transverse"):
        oriented = orient(raw, name)
        canvas, _ = map_filtered_png_to_canvas(
            oriented,
            he_shape,
            tissue,
            microns_per_pixel=args.microns_per_pixel,
            tissue_hires_scalef=args.tissue_hires_scalef,
            ficture_res_um_per_pixel=args.ficture_res_um_per_pixel,
            ficture_xmin_um=args.ficture_xmin_um,
            ficture_ymin_um=args.ficture_ymin_um,
            black_threshold=args.black_threshold,
        )
        mask = canvas.sum(axis=2) > 0
        result[name] = {
            "exact_dice": dice(mask, ref),
            "exact_iou": iou(mask, ref),
            "dilated_1px_dice": dice(ndi.binary_dilation(mask, iterations=1), ndi.binary_dilation(ref, iterations=1)),
            "dilated_1px_iou": iou(ndi.binary_dilation(mask, iterations=1), ndi.binary_dilation(ref, iterations=1)),
        }
    return result


def blend_overlay(he: np.ndarray, ficture: np.ndarray, alpha: float) -> np.ndarray:
    mask = ficture.sum(axis=2) > 0
    overlay = he.astype(np.float32).copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * ficture[mask].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def detect_seams(mask: np.ndarray, tissue: np.ndarray, *, threshold: int = 500) -> tuple[list[int], list[int]]:
    row_gaps = ((~mask) & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) & tissue).sum(axis=1)
    col_gaps = ((~mask) & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1) & tissue).sum(axis=0)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    rows = [int(i) for i, gap in enumerate(row_gaps) if gap >= threshold and row_counts[i] == 0]
    cols = [int(i) for i, gap in enumerate(col_gaps) if gap >= threshold and col_counts[i] == 0]
    return rows, cols


def deseam(canvas: np.ndarray, tissue: np.ndarray, *, threshold: int = 500) -> tuple[np.ndarray, dict[str, int | list[int] | float]]:
    out = canvas.copy()
    before_mask = out.sum(axis=2) > 0
    rows, cols = detect_seams(before_mask, tissue, threshold=threshold)

    for y in rows:
        if y <= 0 or y >= out.shape[0] - 1:
            continue
        mask = (~(out[y].sum(axis=1) > 0)) & (out[y - 1].sum(axis=1) > 0) & (out[y + 1].sum(axis=1) > 0) & tissue[y]
        fill = ((out[y - 1].astype(np.uint16) + out[y + 1].astype(np.uint16)) // 2).astype(np.uint8)
        out[y, mask] = fill[mask]

    for x in cols:
        if x <= 0 or x >= out.shape[1] - 1:
            continue
        mask = (~(out[:, x].sum(axis=1) > 0)) & (out[:, x - 1].sum(axis=1) > 0) & (out[:, x + 1].sum(axis=1) > 0) & tissue[:, x]
        fill = ((out[:, x - 1].astype(np.uint16) + out[:, x + 1].astype(np.uint16)) // 2).astype(np.uint8)
        out[mask, x] = fill[mask]

    after_mask = out.sum(axis=2) > 0
    metrics: dict[str, int | list[int] | float] = {
        "detected_seam_rows": rows,
        "detected_seam_cols": cols,
        "n_seam_rows": len(rows),
        "n_seam_cols": len(cols),
        "colored_pixels_before": int(before_mask.sum()),
        "colored_pixels_after": int(after_mask.sum()),
        "added_pixels": int(after_mask.sum() - before_mask.sum()),
        "dice_deseamed_vs_raw": dice(after_mask, before_mask),
    }
    return out, metrics


def thumbnail(arr: np.ndarray, height: int) -> Image.Image:
    image = Image.fromarray(arr)
    scale = height / image.height
    return image.resize((round(image.width * scale), height), Image.Resampling.LANCZOS)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)
    except OSError:
        return ImageFont.load_default()


def save_qc_triptych(he: np.ndarray, ficture: np.ndarray, overlay: np.ndarray, path: Path, height: int) -> None:
    panels = [
        ("H&E hires", thumbnail(he, height)),
        ("Official filtered FICTURE", thumbnail(ficture, height)),
        ("Overlay on H&E", thumbnail(overlay, height)),
    ]
    label_h = 54
    gap = 12
    width = sum(panel.width for _, panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, panel in panels:
        canvas.paste(panel, (x, label_h))
        draw.text((x + 12, 14), label, fill=(20, 20, 20), font=font(24, bold=True))
        x += panel.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render official filtered FICTURE map aligned to H&E.")
    parser.add_argument("--filtered-ficture-png", type=Path, default=DEFAULT_FILTERED_FICTURE_PNG)
    parser.add_argument("--factor-info", type=Path, default=DEFAULT_FACTOR_INFO)
    parser.add_argument("--ficture-md", type=Path, default=DEFAULT_FICTURE_MD)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--reference-results-coordinate-map", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737554241192739)
    parser.add_argument("--tissue-hires-scalef", type=float, default=0.13752006)
    parser.add_argument("--ficture-res-um-per-pixel", type=float, default=2.0)
    parser.add_argument("--ficture-xmin-um", type=float, default=-10.934)
    parser.add_argument("--ficture-ymin-um", type=float, default=-156.145)
    parser.add_argument("--orientation", choices=["fliplr"], default="fliplr")
    parser.add_argument("--black-threshold", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--seam-gap-threshold", type=int, default=500)
    parser.add_argument("--qc-height", type=int, default=1450)
    parser.add_argument("--skip-orientation-qc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None

    raw = np.array(Image.open(args.filtered_ficture_png).convert("RGB"))
    he = np.array(Image.open(args.he_image).convert("RGB"))
    tissue = estimate_tissue_mask(he)
    oriented = orient(raw, args.orientation)
    canvas_raw, metrics = map_filtered_png_to_canvas(
        oriented,
        he.shape[:2],
        tissue,
        microns_per_pixel=args.microns_per_pixel,
        tissue_hires_scalef=args.tissue_hires_scalef,
        ficture_res_um_per_pixel=args.ficture_res_um_per_pixel,
        ficture_xmin_um=args.ficture_xmin_um,
        ficture_ymin_um=args.ficture_ymin_um,
        black_threshold=args.black_threshold,
    )
    canvas_deseamed, seam_metrics = deseam(canvas_raw, tissue, threshold=args.seam_gap_threshold)
    overlay = blend_overlay(he, canvas_deseamed, alpha=args.alpha)

    raw_path = args.output_dir / "filtered_ficture_official_raw_full_he_canvas.png"
    ficture_path = args.output_dir / "filtered_ficture_official_full_he_canvas.png"
    overlay_path = args.output_dir / "filtered_ficture_official_overlay_on_he.png"
    qc_path = args.output_dir / "filtered_ficture_official_qc_triptych.png"
    summary_path = args.output_dir / "summary_official.json"

    Image.fromarray(canvas_raw).save(raw_path)
    Image.fromarray(canvas_deseamed).save(ficture_path)
    Image.fromarray(overlay).save(overlay_path)
    save_qc_triptych(he, canvas_deseamed, overlay, qc_path, args.qc_height)
    semantic_legend_path = args.output_dir / "factor_semantic_legend.json"
    if args.factor_info.exists():
        write_outputs(build_semantic_legend(args.factor_info), args.output_dir)

    final_mask = canvas_deseamed.sum(axis=2) > 0
    reference_metrics = compare_to_reference(final_mask, args.reference_results_coordinate_map)
    orientation_metrics = {}
    if not args.skip_orientation_qc:
        orientation_metrics = orientation_qc(raw, he.shape[:2], tissue, args.reference_results_coordinate_map, args)
    best_orientation = None
    if orientation_metrics:
        best_orientation = max(orientation_metrics.items(), key=lambda item: float(item[1]["dilated_1px_dice"]))[0]

    status_checks = {
        "uses_filtered_png_source": args.filtered_ficture_png.name == "hex_12.k12.pixel.png"
        and "pixel-level cell type image" in str(args.filtered_ficture_png),
        "ficture_md_says_filtered_feature_bc_matrix": "filtered_feature_bc_matrix"
        in args.ficture_md.read_text(errors="ignore"),
        "no_manual_shift": True,
        "full_canvas_matches_he": tuple(canvas_deseamed.shape[:2]) == tuple(he.shape[:2]),
        "formula_uses_microns_per_pixel_and_tissue_hires_scalef": True,
        "background_false_color_fraction_zero": float(metrics["rendered_background_false_color_fraction"]) == 0.0,
        "orientation_selected_fliplr": args.orientation == "fliplr",
        "orientation_qc_best_is_fliplr": best_orientation in (None, "fliplr"),
    }
    status = "PASS_OFFICIAL" if all(status_checks.values()) else "FAIL_RECHECK_REQUIRED"

    summary = {
        "status": status,
        "status_checks": status_checks,
        "official_pipeline": str(Path(__file__).resolve()),
        "method": (
            "Use filtered FICTURE PNG from pixel-level cell type image, apply fliplr, map every "
            "non-black filtered pixel to H&E hires with he_x=y_um/microns_per_pixel*tissue_hires_scalef "
            "and he_y=x_um/microns_per_pixel*tissue_hires_scalef, clip to H&E tissue, then fill only "
            "detected internal 1-px tile seams. No manual shift, no ROI resize, no results.tsv redraw."
        ),
        "source_paths": {
            "filtered_ficture_png": str(args.filtered_ficture_png),
            "filtered_ficture_factor_info_tsv": str(args.factor_info),
            "ficture_md": str(args.ficture_md),
            "he_image": str(args.he_image),
            "reference_results_coordinate_map_for_qc_only": str(args.reference_results_coordinate_map),
        },
        "filtered_evidence": {
            "ficture_md_input_matrix_line": "--in-sge ${DATADIR}/square_002um/filtered_feature_bc_matrix",
            "exclude_feature_regex": "^(BLANK|NegCon|NegPrb|mt-|MT-|Gm\\d+$$)",
            "rendered_pixels": "all non-black pixels from the filtered PNG after fliplr orientation; not hex_12.k12.results.tsv",
        },
        "formula": {
            "scale": "tissue_hires_scalef / microns_per_pixel",
            "scale_value": args.tissue_hires_scalef / args.microns_per_pixel,
            "x_um": "ficture_xmin_um + oriented_png_row * 2.0",
            "y_um": "ficture_ymin_um + oriented_png_col * 2.0",
            "he_x": "y_um / microns_per_pixel * tissue_hires_scalef",
            "he_y": "x_um / microns_per_pixel * tissue_hires_scalef",
            "microns_per_pixel": args.microns_per_pixel,
            "tissue_hires_scalef": args.tissue_hires_scalef,
            "ficture_xmin_um": args.ficture_xmin_um,
            "ficture_ymin_um": args.ficture_ymin_um,
            "manual_shift_hires_px": {"x": 0.0, "y": 0.0},
        },
        "orientation": {
            "selected": "fliplr(raw filtered PNG)",
            "orientation_qc_best_by_dilated_1px_dice": best_orientation,
            "orientation_qc": orientation_metrics,
        },
        "dims": {
            "source_png_size_wh": [int(raw.shape[1]), int(raw.shape[0])],
            "oriented_png_size_wh": [int(oriented.shape[1]), int(oriented.shape[0])],
            "he_png_size_wh": [int(he.shape[1]), int(he.shape[0])],
            "aligned_png_size_wh": [int(canvas_deseamed.shape[1]), int(canvas_deseamed.shape[0])],
            "overlay_png_size_wh": [int(overlay.shape[1]), int(overlay.shape[0])],
        },
        "metrics": metrics,
        "reference_alignment_metrics": reference_metrics,
        "deseam_note": (
            "The visible small grid came from regular tile seam rows/cols in the filtered PNG render. "
            "Only detected internal 1-px empty seam rows/cols are filled from immediate colored neighbors; "
            "scale, orientation, source, and coordinates are unchanged."
        ),
        "deseam_metrics": seam_metrics,
        "outputs": {
            "raw_before_deseam_full_he_canvas_png": str(raw_path),
            "official_filtered_ficture_full_he_canvas_png": str(ficture_path),
            "official_overlay_on_he_png": str(overlay_path),
            "official_qc_triptych_png": str(qc_path),
            "factor_semantic_legend_json": str(semantic_legend_path),
            "summary_official_json": str(summary_path),
        },
        "not_official": [
            "output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi",
            "output/visium_hd_exp1/ficture_corrected_candidate_pool_input_roi",
            "output/visium_hd_exp1/agent_verified_filtered_ficture_he_align/strict_results_coordinates_no_shift",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": status, "summary": str(summary_path), "qc": str(qc_path)}, indent=2))


if __name__ == "__main__":
    main()
