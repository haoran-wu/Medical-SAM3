#!/usr/bin/env python3
"""Render FICTURE factor assignments back onto the VisiumHD Exp1 H&E image.

This uses the FICTURE pixel PNG as a factor-assignment grid, not as an image
to paste. Factor colors are decoded from the accompanying ``*.info.tsv`` file,
then grid coordinates are converted to H&E hires pixels using the spatula
coordinate convention documented in ``pixel-level cell type image/Ficture.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FICTURE_GRID = PROJECT_ROOT / "pixel-level cell type image" / "visiumhd_exp1_hex12_k12" / "hex_12.k12.pixel.png"
DEFAULT_FACTOR_INFO = PROJECT_ROOT / "pixel-level cell type image" / "visiumhd_exp1_hex12_k12" / "hex_12.k12.pixel.info.tsv"
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "ficture_molecular_he_overlay"


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB value: {value}")
    return tuple(parts)


def load_factor_info(path: Path) -> pd.DataFrame:
    info = pd.read_csv(path, sep="\t")
    info["rgb_tuple"] = info["RGB"].map(parse_rgb)
    return info


def decode_factor_grid(image: np.ndarray, palette: np.ndarray, max_color_distance: float) -> np.ndarray:
    """Assign every non-black grid pixel to the nearest factor palette color."""
    flat = image.reshape(-1, 3).astype(np.int32)
    d2 = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    min_dist = np.sqrt(d2.min(axis=1))
    black = (flat == 0).all(axis=1)
    assignment = np.full(flat.shape[0], -1, dtype=np.int16)
    assignment[(~black) & (min_dist <= max_color_distance)] = nearest[(~black) & (min_dist <= max_color_distance)]
    return assignment.reshape(image.shape[:2])


def estimate_tissue_mask(he: np.ndarray) -> np.ndarray:
    """Estimate the main H&E tissue foreground and reject pale scanner background."""
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
    small = labels == counts.argmax()
    full = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)[: he.shape[0], : he.shape[1]]
    full = ndi.binary_fill_holes(full)
    return ndi.binary_dilation(full, iterations=3)


def load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def save_legend_preview(
    preview: Image.Image,
    info: pd.DataFrame,
    palette: np.ndarray,
    factors: np.ndarray,
    factor_masks: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    legend_width = 720
    canvas = Image.new("RGB", (preview.width + legend_width, preview.height), (255, 255, 255))
    canvas.paste(preview, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    text_font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
    small_font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 15)

    x0 = preview.width + 28
    y = 28
    draw.text((x0, y), "FICTURE factors on H&E", fill=(20, 20, 20), font=title_font)
    y += 44
    draw.text((x0, y), "Coordinate-level redraw from factor grid, not PNG paste", fill=(70, 70, 70), font=small_font)
    y += 34

    for _, row in info.iterrows():
        palette_idx = int(np.where(factors == int(row["Factor"]))[0][0])
        rgb = tuple(int(v) for v in palette[palette_idx])
        mask = factor_masks.get(palette_idx)
        pixels = int(mask.sum()) if mask is not None else 0
        genes = str(row["TopGene_specific"]).split(", ")[:4]
        draw.rectangle((x0, y, x0 + 26, y + 26), fill=rgb, outline=(0, 0, 0))
        draw.text(
            (x0 + 38, y - 2),
            f"F{int(row['Factor'])}  w={float(row['Weight']):.3f}  px={pixels:,}",
            fill=(20, 20, 20),
            font=text_font,
        )
        draw.text((x0 + 38, y + 22), ", ".join(genes), fill=(75, 75, 75), font=small_font)
        y += 58

    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FICTURE factors onto the VisiumHD Exp1 H&E image.")
    parser.add_argument("--ficture-grid", type=Path, default=DEFAULT_FICTURE_GRID)
    parser.add_argument("--factor-info", type=Path, default=DEFAULT_FACTOR_INFO)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-color-distance", type=float, default=10.0)
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737554241192739)
    parser.add_argument("--tissue-hires-scalef", type=float, default=0.13752006)
    parser.add_argument("--ficture-res-um-per-pixel", type=float, default=2.0)
    parser.add_argument("--ficture-xmin-um", type=float, default=-10.934)
    parser.add_argument("--ficture-ymin-um", type=float, default=-156.145)
    parser.add_argument("--alpha", type=float, default=0.44)
    parser.add_argument("--shift-x-px", type=float, default=0.0, help="Extra H&E hires-pixel x shift after coordinate mapping.")
    parser.add_argument("--shift-y-px", type=float, default=0.0, help="Extra H&E hires-pixel y shift after coordinate mapping.")
    parser.add_argument("--preview-height", type=int, default=1400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Image.MAX_IMAGE_PIXELS = None
    he = np.array(Image.open(args.he_image).convert("RGB"))
    grid = np.array(Image.open(args.ficture_grid).convert("RGB"))
    info = load_factor_info(args.factor_info)
    palette = np.array(info["rgb_tuple"].tolist(), dtype=np.int32)
    factors = info["Factor"].astype(int).to_numpy()

    assignment = decode_factor_grid(grid, palette, args.max_color_distance)
    src_y, src_x = np.nonzero(assignment >= 0)
    src_factor_idx = assignment[src_y, src_x]

    scale = args.tissue_hires_scalef / args.microns_per_pixel
    he_x = (args.ficture_xmin_um + src_y.astype(np.float64) * args.ficture_res_um_per_pixel) * scale
    he_y = (args.ficture_ymin_um + src_x.astype(np.float64) * args.ficture_res_um_per_pixel) * scale
    xi = np.rint(he_x + args.shift_x_px).astype(np.int32)
    yi = np.rint(he_y + args.shift_y_px).astype(np.int32)

    valid = (xi >= 0) & (xi < he.shape[1]) & (yi >= 0) & (yi < he.shape[0])
    tissue = estimate_tissue_mask(he)
    inside = np.zeros_like(valid)
    inside[valid] = tissue[yi[valid], xi[valid]]
    keep = valid & inside

    label_image = np.full(he.shape[:2], -1, dtype=np.int16)
    label_image[yi[keep], xi[keep]] = src_factor_idx[keep]

    factor_masks: dict[int, np.ndarray] = {}
    overlay = he.copy().astype(np.float32)
    for palette_idx, rgb in enumerate(palette):
        mask = label_image == palette_idx
        if not mask.any():
            continue
        mask = ndi.binary_dilation(mask, iterations=1) & tissue
        factor_masks[palette_idx] = mask
        overlay[mask] = (1.0 - args.alpha) * overlay[mask] + args.alpha * rgb.astype(np.float32)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    full_path = args.output_dir / "ficture_factor_molecular_overlay_on_he_full.png"
    preview_path = args.output_dir / "ficture_factor_molecular_overlay_on_he_preview.png"
    legend_path = args.output_dir / "ficture_factor_molecular_overlay_on_he_with_legend.png"
    Image.fromarray(overlay).save(full_path)

    preview = Image.fromarray(overlay)
    preview_scale = args.preview_height / preview.height
    preview = preview.resize((round(preview.width * preview_scale), args.preview_height), Image.Resampling.LANCZOS)
    preview.save(preview_path)
    save_legend_preview(preview, info, palette, factors, factor_masks, legend_path)

    summary = {
        "method": "Decode FICTURE factor grid colors, map grid coordinates to H&E hires coordinates, clip to tissue, redraw factors on H&E.",
        "not_used_as_direct_image_paste": True,
        "ficture_grid": str(args.ficture_grid),
        "factor_info": str(args.factor_info),
        "he_image": str(args.he_image),
        "coordinate_formula": {
            "source": "u=FICTURE PNG column, v=FICTURE PNG row",
            "selected_mapping": "he_x=(ficture_xmin_um + v*res_um)*tissue_hires_scalef/microns_per_pixel; he_y=(ficture_ymin_um + u*res_um)*tissue_hires_scalef/microns_per_pixel",
            "extra_shift_hires_px": {"x": args.shift_x_px, "y": args.shift_y_px},
        },
        "source_assigned_pixels": int((assignment >= 0).sum()),
        "mapped_valid_pixels": int(valid.sum()),
        "mapped_inside_tissue_pixels": int(keep.sum()),
        "valid_fraction": float(valid.mean()),
        "inside_tissue_fraction_of_valid": float(inside[valid].mean()) if valid.any() else 0.0,
        "output_full": str(full_path),
        "output_preview": str(preview_path),
        "output_legend": str(legend_path),
    }
    (args.output_dir / "ficture_molecular_overlay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
