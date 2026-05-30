#!/usr/bin/env python3
"""
Render approximate label masks from cell-center coordinates in TMA24.csv.

This is a first-pass workflow:
- map coordinateX/coordinateY into image pixel space with min-max scaling
- draw each labeled cell as a small filled disk
- export per-label masks and overlays for quick spatial sanity checks
"""

import argparse
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "tma24" / "TMA24.csv"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "examples" / "legacy_examples" / "tma24" / "example1.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "00_FINAL_tma24_example1_scale_0p55_shiftX_neg120_shiftY_620"

PALETTE: List[Tuple[int, int, int]] = [
    (230, 57, 70),
    (42, 157, 143),
    (106, 76, 147),
    (244, 162, 97),
    (29, 53, 87),
    (233, 196, 106),
    (69, 123, 157),
]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def transform_coordinates(
    x_values: pd.Series,
    y_values: pd.Series,
    mode: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply a coordinate-system transform before mapping into image space.

    Modes:
    - identity: x=coordinateX, y=coordinateY
    - x_y_y_negx: x=coordinateY, y=-coordinateX
    - x_negy_y_x: x=-coordinateY, y=coordinateX
    - x_negy_y_negx: x=-coordinateY, y=-coordinateX
    """
    x_values = x_values.astype(float)
    y_values = y_values.astype(float)

    if mode == "identity":
        return x_values, y_values
    if mode == "x_y_y_negx":
        return y_values, -x_values
    if mode == "x_negy_y_x":
        return -y_values, x_values
    if mode == "x_negy_y_negx":
        return -y_values, -x_values
    raise ValueError(f"Unsupported coordinate transform: {mode}")


def map_values_to_pixels(values: pd.Series, size: int) -> np.ndarray:
    """Linearly map a coordinate column into pixel indices."""
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.zeros(len(values), dtype=np.int32)
    scaled = (values.astype(float) - vmin) / (vmax - vmin)
    return np.rint(scaled * (size - 1)).astype(np.int32)


def estimate_tissue_mask(image: Image.Image, intensity_threshold: int = 240, border_trim: int = 5, downsample: int = 8) -> np.ndarray:
    """
    Estimate tissue foreground as the largest connected non-background component.

    This is more robust than using the full image extents because whole-slide exports often include
    edge artifacts and sparse colored specks outside the main tissue region.
    """
    rgb = np.array(image.convert("RGB"))
    tissue_candidate = (rgb.mean(axis=2) < intensity_threshold).astype(np.uint8)
    if border_trim > 0:
        tissue_candidate[:border_trim, :] = 0
        tissue_candidate[-border_trim:, :] = 0
        tissue_candidate[:, :border_trim] = 0
        tissue_candidate[:, -border_trim:] = 0

    small = tissue_candidate[::downsample, ::downsample]
    height_s, width_s = small.shape
    visited = np.zeros((height_s, width_s), dtype=bool)
    best_component: List[Tuple[int, int]] = []

    for y in range(height_s):
        for x in range(width_s):
            if small[y, x] and not visited[y, x]:
                queue = deque([(y, x)])
                visited[y, x] = True
                component: List[Tuple[int, int]] = []

                while queue:
                    cy, cx = queue.popleft()
                    component.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height_s and 0 <= nx < width_s and small[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                if len(component) > len(best_component):
                    best_component = component

    if not best_component:
        raise ValueError("Unable to detect a tissue component in the image.")

    small_mask = np.zeros_like(small, dtype=np.uint8)
    ys, xs = zip(*best_component)
    small_mask[np.array(ys), np.array(xs)] = 1

    full_mask = np.repeat(np.repeat(small_mask, downsample, axis=0), downsample, axis=1)
    full_mask = full_mask[: rgb.shape[0], : rgb.shape[1]]
    return full_mask.astype(np.uint8)


def compute_foreground_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Foreground mask is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def map_values_to_bbox(values: pd.Series, min_pixel: int, max_pixel: int) -> np.ndarray:
    """Linearly map values into a target pixel interval."""
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.full(len(values), min_pixel, dtype=np.int32)
    scaled = (values.astype(float) - vmin) / (vmax - vmin)
    return np.rint(min_pixel + scaled * (max_pixel - min_pixel)).astype(np.int32)


def scale_bbox_around_center(
    bbox: Tuple[int, int, int, int],
    scale: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Shrink or expand a bbox around its center, then clip to image bounds."""
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half_w = max(1.0, width * scale / 2.0)
    half_h = max(1.0, height * scale / 2.0)
    img_w, img_h = image_size
    sx_min = max(0, int(round(cx - half_w)))
    sy_min = max(0, int(round(cy - half_h)))
    sx_max = min(img_w - 1, int(round(cx + half_w)))
    sy_max = min(img_h - 1, int(round(cy + half_h)))
    return sx_min, sy_min, sx_max, sy_max


def shift_bbox(
    bbox: Tuple[int, int, int, int],
    shift_x: int,
    shift_y: int,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Translate a bbox while preserving its size as much as possible."""
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    img_w, img_h = image_size

    x_min += int(shift_x)
    x_max += int(shift_x)
    y_min += int(shift_y)
    y_max += int(shift_y)

    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > img_w - 1:
        overflow = x_max - (img_w - 1)
        x_min = max(0, x_min - overflow)
        x_max = img_w - 1
    if y_max > img_h - 1:
        overflow = y_max - (img_h - 1)
        y_min = max(0, y_min - overflow)
        y_max = img_h - 1

    x_max = min(img_w - 1, x_min + width)
    y_max = min(img_h - 1, y_min + height)
    return x_min, y_min, x_max, y_max


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def draw_disk_mask(points: np.ndarray, image_size: Tuple[int, int], radius: int) -> np.ndarray:
    """Draw filled disks around mapped cell centers."""
    width, height = image_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return (np.array(mask_img) > 0).astype(np.uint8)


def ensure_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def make_region_like_mask(
    disk_mask: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    connect_size: int,
    smooth_radius: float,
    threshold: int,
    downsample_factor: int,
) -> np.ndarray:
    """
    Convert sparse cell disks into a smoother pseudo-region mask.

    Steps:
    - dilate to connect nearby cells
    - close small gaps
    - blur and threshold for softer region edges
    - clip back to tissue
    """
    mask_img = Image.fromarray(disk_mask.astype(np.uint8) * 255, mode="L")
    tissue_img = Image.fromarray(tissue_mask.astype(np.uint8) * 255, mode="L")
    connect_size = ensure_odd(connect_size)
    downsample_factor = max(1, int(downsample_factor))

    if downsample_factor > 1:
        small_size = (
            max(1, mask_img.size[0] // downsample_factor),
            max(1, mask_img.size[1] // downsample_factor),
        )
        mask_img = mask_img.resize(small_size, Image.Resampling.NEAREST)
        tissue_img = tissue_img.resize(small_size, Image.Resampling.NEAREST)

    small_connect_size = ensure_odd(max(3, round(connect_size / downsample_factor)))
    small_blur_radius = max(0.1, smooth_radius / downsample_factor)

    dilated = mask_img.filter(ImageFilter.MaxFilter(size=small_connect_size))
    closed = dilated.filter(ImageFilter.MaxFilter(size=small_connect_size)).filter(ImageFilter.MinFilter(size=small_connect_size))
    smoothed = closed.filter(ImageFilter.GaussianBlur(radius=small_blur_radius))
    region = (np.array(smoothed) >= threshold).astype(np.uint8)
    if downsample_factor > 1:
        region = np.array(
            Image.fromarray(region.astype(np.uint8) * 255, mode="L").resize(
                tissue_mask.shape[::-1], Image.Resampling.NEAREST
            )
        ) >= 128
        region = region.astype(np.uint8)
    return (region & tissue_mask.astype(np.uint8)).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_overlay(overlay: np.ndarray, path: Path) -> None:
    Image.fromarray(overlay).save(path)


def save_all_cells_preview(image: np.ndarray, mapped_df: pd.DataFrame, path: Path) -> None:
    preview = Image.fromarray(image).convert("RGBA")
    dots = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dots)
    for x, y in mapped_df[["image_x", "image_y"]].itertuples(index=False):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0, 115))
    Image.alpha_composite(preview, dots).convert("RGB").save(path)


def save_tissue_preview(image: np.ndarray, tissue_mask: np.ndarray, path: Path) -> None:
    overlay = image.astype(np.float32).copy()
    mask_bool = tissue_mask.astype(bool)
    color = np.array([0, 200, 255], dtype=np.float32)
    overlay[mask_bool] = 0.75 * overlay[mask_bool] + 0.25 * color
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def save_combined_overlay(image: np.ndarray, label_masks: List[dict], path: Path) -> None:
    canvas = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for item in label_masks:
        color = item["color"]
        mask = Image.fromarray(item["mask"] * 255).convert("L")
        layer = Image.new("RGBA", canvas.size, color + (0,))
        layer.putalpha(mask.point(lambda p: 110 if p > 0 else 0))
        overlay = Image.alpha_composite(overlay, layer)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(path)


def load_font(size: int, bold: bool = False):
    font_name = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except OSError:
        return ImageFont.load_default()


def save_combined_overlay_with_legend(
    image: np.ndarray,
    label_masks: List[dict],
    path: Path,
    title: str,
) -> None:
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for item in label_masks:
        color = item["color"]
        mask = Image.fromarray(item["mask"] * 255).convert("L")
        layer = Image.new("RGBA", base.size, color + (0,))
        layer.putalpha(mask.point(lambda p: 110 if p > 0 else 0))
        overlay = Image.alpha_composite(overlay, layer)
    combined = Image.alpha_composite(base, overlay)

    legend_width = 700
    row_height = 96
    top_margin = 32
    bottom_margin = 32
    title_block = 90
    canvas_height = max(combined.size[1], top_margin + title_block + len(label_masks) * row_height + bottom_margin)

    canvas = Image.new("RGBA", (combined.size[0] + legend_width, canvas_height), (255, 255, 255, 255))
    image_y = (canvas_height - combined.size[1]) // 2
    canvas.alpha_composite(combined, (0, image_y))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    text_font = load_font(24, bold=False)
    small_font = load_font(20, bold=False)

    x0 = combined.size[0] + 30
    y = top_margin
    draw.text((x0, y), title, fill=(25, 25, 25), font=title_font)
    y += 48
    draw.text((x0, y), "Each color corresponds to one label.", fill=(80, 80, 80), font=small_font)
    y += 42

    for idx, item in enumerate(label_masks, start=1):
        row_y = y + (idx - 1) * row_height
        color = item["color"]
        label = item["label"]
        coverage = float(item["mask"].mean()) * 100.0
        draw.rounded_rectangle((x0, row_y, x0 + 34, row_y + 34), radius=6, fill=color + (255,), outline=(40, 40, 40))
        draw.text((x0 + 50, row_y - 2), f"{idx}. {label}", fill=(20, 20, 20), font=text_font)
        draw.text((x0 + 50, row_y + 34), f"coverage: {coverage:.2f}%", fill=(90, 90, 90), font=small_font)

    canvas.convert("RGB").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate approximate label masks from TMA24 cell centers.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotation-column", type=str, default="new_annotation")
    parser.add_argument("--x-column", type=str, default="coordinateX")
    parser.add_argument("--y-column", type=str, default="coordinateY")
    parser.add_argument(
        "--coordinate-transform",
        type=str,
        default="x_negy_y_negx",
        choices=["identity", "x_y_y_negx", "x_negy_y_x", "x_negy_y_negx"],
        help="Transform applied before min-max mapping into the tissue bounding box.",
    )
    parser.add_argument("--radius", type=int, default=12, help="Disk radius in image pixels for each cell center.")
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=0.55,
        help="Scale the tissue bbox around its center before mapping coordinates. Values <1 shrink inward.",
    )
    parser.add_argument("--bbox-shift-x", type=int, default=-120, help="Horizontal shift in pixels applied to the mapping bbox.")
    parser.add_argument("--bbox-shift-y", type=int, default=620, help="Vertical shift in pixels applied to the mapping bbox.")
    parser.add_argument("--region-connect-size", type=int, default=61, help="Odd kernel size used to connect nearby cell disks.")
    parser.add_argument("--region-blur-radius", type=float, default=9.0, help="Gaussian blur radius for region-like masks.")
    parser.add_argument("--region-threshold", type=int, default=32, help="Threshold applied after smoothing the region-like mask.")
    parser.add_argument("--region-downsample-factor", type=int, default=4, help="Downsample factor for faster region-like smoothing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)
    width, height = image.size
    tissue_mask = estimate_tissue_mask(image)
    tissue_bbox = compute_foreground_bbox(tissue_mask)
    mapping_bbox = scale_bbox_around_center(tissue_bbox, args.bbox_scale, image.size)
    mapping_bbox = shift_bbox(mapping_bbox, args.bbox_shift_x, args.bbox_shift_y, image.size)

    missing = [col for col in [args.annotation_column, args.x_column, args.y_column] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work_df = df[[args.annotation_column, args.x_column, args.y_column]].copy()
    work_df = work_df.dropna(subset=[args.annotation_column, args.x_column, args.y_column]).copy()
    transformed_x, transformed_y = transform_coordinates(
        work_df[args.x_column],
        work_df[args.y_column],
        args.coordinate_transform,
    )
    work_df["transformed_x"] = transformed_x
    work_df["transformed_y"] = transformed_y
    x_min_px, y_min_px, x_max_px, y_max_px = mapping_bbox
    work_df["image_x"] = map_values_to_bbox(work_df["transformed_x"], x_min_px, x_max_px)
    work_df["image_y"] = map_values_to_bbox(work_df["transformed_y"], y_min_px, y_max_px)
    inside_tissue = tissue_mask[work_df["image_y"].to_numpy(), work_df["image_x"].to_numpy()] > 0
    work_df["inside_tissue"] = inside_tissue
    work_df = work_df[work_df["inside_tissue"]].copy()

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    region_masks_dir = output_dir / "region_masks"
    region_overlays_dir = output_dir / "region_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    region_masks_dir.mkdir(parents=True, exist_ok=True)
    region_overlays_dir.mkdir(parents=True, exist_ok=True)

    save_all_cells_preview(image_np, work_df, output_dir / "all_cells_preview.png")
    save_tissue_preview(image_np, tissue_mask, output_dir / "tissue_mask_preview.png")

    summary = {
        "csv_path": str(args.csv_path),
        "image_path": str(args.image_path),
        "annotation_column": args.annotation_column,
        "x_column": args.x_column,
        "y_column": args.y_column,
        "coordinate_transform": args.coordinate_transform,
        "mapping_method": "transform coordinates, min-max scale them into the estimated tissue bounding box, then clip to tissue foreground",
        "tissue_bbox_xyxy": tissue_bbox,
        "mapping_bbox_xyxy": mapping_bbox,
        "bbox_scale": args.bbox_scale,
        "bbox_shift_x": args.bbox_shift_x,
        "bbox_shift_y": args.bbox_shift_y,
        "radius": args.radius,
        "region_connect_size": args.region_connect_size,
        "region_blur_radius": args.region_blur_radius,
        "region_threshold": args.region_threshold,
        "region_downsample_factor": args.region_downsample_factor,
        "n_labeled_cells_after_tissue_clip": int(len(work_df)),
        "labels": [],
    }

    label_masks: List[dict] = []
    region_label_masks: List[dict] = []
    labels = sorted(work_df[args.annotation_column].unique())
    for idx, label in enumerate(labels):
        color = PALETTE[idx % len(PALETTE)]
        label_df = work_df[work_df[args.annotation_column] == label]
        points = label_df[["image_x", "image_y"]].to_numpy(dtype=np.int32)
        disk_mask = draw_disk_mask(points, image.size, args.radius)
        disk_mask = (disk_mask & tissue_mask).astype(np.uint8)
        region_mask = make_region_like_mask(
            disk_mask,
            tissue_mask,
            connect_size=args.region_connect_size,
            smooth_radius=args.region_blur_radius,
            threshold=args.region_threshold,
            downsample_factor=args.region_downsample_factor,
        )
        overlay = make_overlay(image_np, disk_mask, color)
        region_overlay = make_overlay(image_np, region_mask, color)

        stem = f"{idx + 1:02d}_{slugify(label)}"
        mask_path = masks_dir / f"{stem}.png"
        overlay_path = overlays_dir / f"{stem}.png"
        region_mask_path = region_masks_dir / f"{stem}.png"
        region_overlay_path = region_overlays_dir / f"{stem}.png"
        save_mask(disk_mask, mask_path)
        save_overlay(overlay, overlay_path)
        save_mask(region_mask, region_mask_path)
        save_overlay(region_overlay, region_overlay_path)

        label_masks.append({"label": label, "mask": disk_mask, "color": color})
        region_label_masks.append({"label": label, "mask": region_mask, "color": color})
        summary["labels"].append(
            {
                "label": label,
                "n_cells": int(len(label_df)),
                "disk_mask_path": str(mask_path),
                "disk_overlay_path": str(overlay_path),
                "region_mask_path": str(region_mask_path),
                "region_overlay_path": str(region_overlay_path),
                "disk_positive_pixels": int(disk_mask.sum()),
                "disk_coverage_ratio": float(disk_mask.mean()),
                "region_positive_pixels": int(region_mask.sum()),
                "region_coverage_ratio": float(region_mask.mean()),
                "color_rgb": color,
            }
        )

    save_combined_overlay(image_np, label_masks, output_dir / "combined_label_overlay.png")
    save_combined_overlay(image_np, region_label_masks, output_dir / "combined_region_overlay.png")
    save_combined_overlay_with_legend(
        image_np,
        label_masks,
        output_dir / "combined_label_overlay_with_legend.png",
        title="Cell Disk Masks on One H&E",
    )
    save_combined_overlay_with_legend(
        image_np,
        region_label_masks,
        output_dir / "combined_region_overlay_with_legend.png",
        title="Region-like Pseudo-masks on One H&E",
    )
    mapped_preview = work_df.rename(columns={args.annotation_column: "label"})
    mapped_preview.to_csv(output_dir / "mapped_cells.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote output to: {output_dir}")
    print(f"Labels: {labels}")
    print(f"Mapped labeled cells: {len(work_df)}")


if __name__ == "__main__":
    main()
