#!/usr/bin/env python3
"""Build a blind full-resolution tissue support mask from the registered H&E."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-max-side", type=int, default=2048)
    parser.add_argument("--minimum-component-fraction", type=float, default=0.0002)
    return parser.parse_args()


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    with Image.open(args.he_image) as handle:
        full_size = handle.size
        scale = min(1.0, args.work_max_side / max(full_size))
        work_size = (
            max(1, int(round(full_size[0] * scale))),
            max(1, int(round(full_size[1] * scale))),
        )
        image = np.asarray(handle.convert("RGB").resize(work_size, Image.Resampling.BILINEAR))

    rgb = image.astype(np.float32)
    intensity = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    tissue = (rgb.min(axis=2) < 244) & ((chroma >= 7) | (intensity < 226))
    tissue = ndi.binary_closing(tissue, structure=disk(2))

    labels, count = ndi.label(tissue)
    sizes = np.bincount(labels.ravel())
    minimum = max(100, int(round(tissue.size * args.minimum_component_fraction)))
    keep = sizes >= minimum
    keep[0] = False
    tissue = keep[labels]

    labels, count = ndi.label(tissue)
    support = np.zeros_like(tissue, dtype=bool)
    for label in range(1, count + 1):
        support |= ndi.binary_fill_holes(labels == label)
    support = ndi.binary_dilation(support, structure=disk(4))

    full = Image.fromarray(support.astype(np.uint8) * 255).resize(full_size, Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full.save(args.output)
    summary = {
        "method": "registered_he_color_threshold_components_v1",
        "source": str(args.he_image),
        "full_size": list(full_size),
        "work_size": list(work_size),
        "work_max_side": args.work_max_side,
        "minimum_component_fraction": args.minimum_component_fraction,
        "tissue_rule": "min_rgb_below_244_and_chroma_at_least_7_or_intensity_below_226",
        "closing_disk_radius": 2,
        "fill_holes_per_component": True,
        "dilation_disk_radius": 4,
        "full_resolution_resampling": "nearest",
        "minimum_component_pixels_at_work_size": minimum,
        "retained_component_count": int(count),
        "support_fraction_at_work_size": float(support.mean()),
        "selection_used_annotation": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
