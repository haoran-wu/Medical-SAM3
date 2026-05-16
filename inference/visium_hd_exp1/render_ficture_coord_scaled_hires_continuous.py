#!/usr/bin/env python3
"""Render FICTURE factor assignments directly into H&E hires coordinates.

This uses the Space Ranger scale factors rather than resizing an already
rendered PNG. Each FICTURE unit is drawn as a small hexagon so the result looks
like a continuous factor map, while still using the coordinate formula:

    he_x = ficture_y_um / microns_per_pixel * tissue_hires_scalef
    he_y = ficture_x_um / microns_per_pixel * tissue_hires_scalef
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3")
FICTURE = Path("/nfs/roberts/project/pi_xy48/hw646/FICTURE/punkst_runs/VisiumHD_Exp1_hex12_k12_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--ficture-dir", type=Path, default=FICTURE)
    parser.add_argument(
        "--he-image",
        type=Path,
        default=PROJECT / "output/visium_hd_exp1/assets/tissue_hires_image.png",
    )
    parser.add_argument(
        "--full-summary",
        type=Path,
        default=PROJECT / "output/visium_hd_exp1/sam3_local_region_summary/region_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi",
    )
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737554241192739)
    parser.add_argument("--tissue-hires-scalef", type=float, default=0.13752006)
    parser.add_argument("--hex-size-um", type=float, default=6.92820323027551)
    parser.add_argument("--radius-scale", type=float, default=1.05)
    parser.add_argument("--margin-px", type=int, default=30)
    return parser.parse_args()


def read_palette(path: Path) -> dict[int, tuple[int, int, int]]:
    palette: dict[int, tuple[int, int, int]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            palette[int(row["Factor"])] = tuple(int(v) for v in row["RGB"].split(","))
    return palette


def get_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def resize_panel(image: Image.Image, title: str, max_h: int, resample: int) -> Image.Image:
    scale = max_h / image.height
    panel = image.resize((max(1, int(image.width * scale)), max_h), resample)
    pad = 60
    canvas = Image.new("RGB", (panel.width, panel.height + pad), "white")
    canvas.paste(panel, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill="black", font=get_font(24))
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    he = Image.open(args.he_image).convert("RGB")
    width, height = he.size
    scale = args.tissue_hires_scalef / args.microns_per_pixel
    radius_px = max(3, int(round(args.hex_size_um * scale * args.radius_scale)))

    palette = read_palette(args.ficture_dir / "hex_12.k12.pixel.info.tsv")
    factor_ids = sorted(palette)
    centers: list[tuple[int, int, int]] = []
    total = 0

    with (args.ficture_dir / "hex_12.k12.results.tsv").open() as fh:
        factor_cols: list[int] = []
        for line in fh:
            if not line.strip():
                continue
            if line.startswith("#"):
                header = line[1:].strip().split("\t")
                factor_cols = [i for i, col in enumerate(header) if col.isdigit()]
                continue
            parts = line.rstrip("\n").split("\t")
            total += 1
            try:
                x_um = float(parts[1])
                y_um = float(parts[2])
                best_i = 0
                best_v = -1.0
                for j, idx in enumerate(factor_cols):
                    value = float(parts[idx])
                    if value > best_v:
                        best_v = value
                        best_i = j
            except Exception:
                continue
            px = int(round(y_um * scale))
            py = int(round(x_um * scale))
            if 0 <= px < width and 0 <= py < height:
                centers.append((px, py, factor_ids[best_i]))

    if not centers:
        raise RuntimeError("No FICTURE centers mapped into the H&E hires canvas.")

    xs = [x for x, _, _ in centers]
    ys = [y for _, y, _ in centers]
    bbox = [
        max(0, min(xs) - radius_px - args.margin_px),
        max(0, min(ys) - radius_px - args.margin_px),
        min(width, max(xs) + radius_px + args.margin_px + 1),
        min(height, max(ys) + radius_px + args.margin_px + 1),
    ]
    crop_w = bbox[2] - bbox[0]
    crop_h = bbox[3] - bbox[1]

    roi = Image.new("RGB", (crop_w, crop_h), (0, 0, 0))
    factor_roi = Image.new("I", (crop_w, crop_h), -1)
    draw = ImageDraw.Draw(roi)
    factor_draw = ImageDraw.Draw(factor_roi)
    angles = [math.radians(60 * k + 30) for k in range(6)]
    for px, py, factor in centers:
        cx = px - bbox[0]
        cy = py - bbox[1]
        poly = [(cx + radius_px * math.cos(a), cy + radius_px * math.sin(a)) for a in angles]
        draw.polygon(poly, fill=palette[factor])
        factor_draw.polygon(poly, fill=int(factor))

    he_roi = he.crop(tuple(bbox))
    mask = roi.convert("L").point(lambda value: 255 if value > 0 else 0)
    blend = Image.blend(he_roi, roi, 0.48)
    overlay = he_roi.copy()
    overlay.paste(blend, (0, 0), mask)

    full = Image.new("RGB", (width, height), (0, 0, 0))
    full.paste(roi, (bbox[0], bbox[1]))

    roi_path = args.output_dir / "ficture_coord_scaled_hires_continuous_roi_rgb.png"
    full_path = args.output_dir / "ficture_coord_scaled_hires_continuous_full_canvas_rgb.png"
    he_roi_path = args.output_dir / "he_roi_matching_coord_scaled_continuous_ficture_coverage.png"
    overlay_path = args.output_dir / "ficture_coord_scaled_hires_continuous_overlay_roi.png"
    factor_roi_path = args.output_dir / "ficture_coord_scaled_hires_continuous_roi_factor_index.npy"
    roi.save(roi_path)
    full.save(full_path)
    he_roi.save(he_roi_path)
    overlay.save(overlay_path)
    try:
        import numpy as np

        np.save(factor_roi_path, np.array(factor_roi, dtype=np.int16))
    except Exception as exc:
        raise RuntimeError(f"Could not save factor-index ROI array: {factor_roi_path}") from exc

    panels = [
        resize_panel(he_roi, "H&E crop from coordinate coverage", 1000, Image.Resampling.LANCZOS),
        resize_panel(
            roi,
            f"Continuous FICTURE using tissue_hires_scalef (hex radius {radius_px}px)",
            1000,
            Image.Resampling.NEAREST,
        ),
        resize_panel(overlay, "Overlay check", 1000, Image.Resampling.LANCZOS),
    ]
    qc = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + 24 * (len(panels) - 1), max(panel.height for panel in panels)),
        "white",
    )
    xoff = 0
    for panel in panels:
        qc.paste(panel, (xoff, 0))
        xoff += panel.width + 24
    qc_path = args.output_dir / "coord_scaled_hires_continuous_qc_threeway.png"
    qc.save(qc_path)

    full_summary = json.loads(args.full_summary.read_text())
    mask_dir = args.output_dir / "cropped_annotation_masks"
    mask_dir.mkdir(exist_ok=True)
    labels = []
    for i, item in enumerate(full_summary["labels"], start=1):
        source = Path(item["mask_path"])
        if not source.exists():
            source = args.project / source
        mask_crop = Image.open(source).convert("L").crop(tuple(bbox))
        mask_array = mask_crop.point(lambda value: 255 if value > 127 else 0)
        target = mask_dir / f"{i:02d}_{item['slug']}_target_coord_scaled_continuous_roi.png"
        mask_array.save(target)
        labels.append(
            {
                "label": item["label"],
                "slug": item["slug"],
                "mask_path": str(target),
                "area_pixels": int(sum(1 for value in mask_array.getdata() if value > 0)),
            }
        )

    summary = {
        "image_path": str(roi_path),
        "full_canvas_image_path": str(full_path),
        "he_roi_image_path": str(he_roi_path),
        "crop_bbox_xyxy_exclusive": bbox,
        "crop_size_wh": [crop_w, crop_h],
        "source_note": "FICTURE factors rendered as continuous hex footprints from hex_12.k12.results.tsv using tissue_hires_scalef and microns_per_pixel.",
        "labels": labels,
    }
    summary_path = args.output_dir / "region_summary_coord_scaled_hires_continuous_roi_hpc.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    local_summary = json.loads(json.dumps(summary).replace(str(args.project) + "/", ""))
    (args.output_dir / "region_summary_coord_scaled_hires_continuous_roi.json").write_text(
        json.dumps(local_summary, indent=2)
    )

    render_summary = {
        "formula": "he_x = y / microns_per_pixel * tissue_hires_scalef; he_y = x / microns_per_pixel * tissue_hires_scalef",
        "microns_per_pixel": args.microns_per_pixel,
        "tissue_hires_scalef": args.tissue_hires_scalef,
        "scale": scale,
        "hex_size_um": args.hex_size_um,
        "draw_radius_px": radius_px,
        "n_units_total": total,
        "n_units_in_he_canvas": len(centers),
        "coverage_bbox_xyxy_exclusive": bbox,
        "he_size_wh": [width, height],
        "outputs": {
            "roi": str(roi_path),
            "full_canvas": str(full_path),
            "he_roi": str(he_roi_path),
            "overlay_roi": str(overlay_path),
            "factor_roi_npy": str(factor_roi_path),
            "qc": str(qc_path),
            "region_summary_hpc": str(summary_path),
        },
    }
    (args.output_dir / "coord_scaled_hires_continuous_summary.json").write_text(
        json.dumps(render_summary, indent=2)
    )
    print(json.dumps(render_summary, indent=2))


if __name__ == "__main__":
    main()
