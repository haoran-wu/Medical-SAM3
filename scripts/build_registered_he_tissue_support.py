#!/usr/bin/env python3
"""Create the blind tissue-support mask used to filter SAM candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if len(xx) < 3:
        return mask.copy()
    points = np.column_stack([xx, yy])
    hull = ConvexHull(points)
    canvas = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in points[hull.vertices]], fill=1)
    return np.asarray(canvas, dtype=bool)


def registered_he_tissue_support(registered_he: np.ndarray) -> np.ndarray:
    """Keep the tissue body and its enclosed blank spaces."""
    image = registered_he.astype(np.float32)
    intensity = image.mean(axis=2)
    chroma = image.max(axis=2) - image.min(axis=2)
    raw = (image.min(axis=2) < 244) & ((chroma >= 7) | (intensity < 226))
    raw = ndi.binary_closing(raw, structure=disk(5))
    labels, count = ndi.label(raw)
    if count == 0:
        return np.zeros_like(raw, dtype=bool)
    sizes = np.bincount(labels.ravel())
    minimum = max(300, int(sizes[1:].max() * 0.002))
    keep = sizes >= minimum
    keep[0] = False
    tissue = keep[labels]
    support = convex_hull_mask(tissue)
    return ndi.binary_dilation(support, structure=disk(12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with Image.open(args.he_image) as image:
        he = np.asarray(image.convert("RGB"))
    support = registered_he_tissue_support(he)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(support.astype(np.uint8) * 255).save(args.output)
    print(f"Saved {args.output} with {int(support.sum()):,} supported pixels")


if __name__ == "__main__":
    main()
