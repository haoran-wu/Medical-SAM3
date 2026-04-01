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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "tma24" / "TMA24.csv"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "example1.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "tma24_example1"

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


def map_values_to_pixels(values: pd.Series, size: int) -> np.ndarray:
    """Linearly map a coordinate column into pixel indices."""
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.zeros(len(values), dtype=np.int32)
    scaled = (values.astype(float) - vmin) / (vmax - vmin)
    return np.rint(scaled * (size - 1)).astype(np.int32)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate approximate label masks from TMA24 cell centers.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotation-column", type=str, default="new_annotation")
    parser.add_argument("--x-column", type=str, default="coordinateX")
    parser.add_argument("--y-column", type=str, default="coordinateY")
    parser.add_argument("--radius", type=int, default=12, help="Disk radius in image pixels for each cell center.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)
    width, height = image.size

    missing = [col for col in [args.annotation_column, args.x_column, args.y_column] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work_df = df[[args.annotation_column, args.x_column, args.y_column]].copy()
    work_df = work_df.dropna(subset=[args.annotation_column, args.x_column, args.y_column]).copy()
    work_df["image_x"] = map_values_to_pixels(work_df[args.x_column], width)
    work_df["image_y"] = map_values_to_pixels(work_df[args.y_column], height)

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    save_all_cells_preview(image_np, work_df, output_dir / "all_cells_preview.png")

    summary = {
        "csv_path": str(args.csv_path),
        "image_path": str(args.image_path),
        "annotation_column": args.annotation_column,
        "x_column": args.x_column,
        "y_column": args.y_column,
        "mapping_method": "independent min-max scaling of coordinate columns into image pixel space",
        "radius": args.radius,
        "labels": [],
    }

    label_masks: List[dict] = []
    labels = sorted(work_df[args.annotation_column].unique())
    for idx, label in enumerate(labels):
        color = PALETTE[idx % len(PALETTE)]
        label_df = work_df[work_df[args.annotation_column] == label]
        points = label_df[["image_x", "image_y"]].to_numpy(dtype=np.int32)
        mask = draw_disk_mask(points, image.size, args.radius)
        overlay = make_overlay(image_np, mask, color)

        stem = f"{idx + 1:02d}_{slugify(label)}"
        mask_path = masks_dir / f"{stem}.png"
        overlay_path = overlays_dir / f"{stem}.png"
        save_mask(mask, mask_path)
        save_overlay(overlay, overlay_path)

        label_masks.append({"label": label, "mask": mask, "color": color})
        summary["labels"].append(
            {
                "label": label,
                "n_cells": int(len(label_df)),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "mask_positive_pixels": int(mask.sum()),
                "coverage_ratio": float(mask.mean()),
                "color_rgb": color,
            }
        )

    save_combined_overlay(image_np, label_masks, output_dir / "combined_label_overlay.png")
    mapped_preview = work_df.rename(columns={args.annotation_column: "label"})
    mapped_preview.to_csv(output_dir / "mapped_cells.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote output to: {output_dir}")
    print(f"Labels: {labels}")
    print(f"Mapped labeled cells: {len(work_df)}")


if __name__ == "__main__":
    main()
