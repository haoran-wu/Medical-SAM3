#!/usr/bin/env python3
"""Build structured FICTURE text summaries for candidate masks.

The output is intended for VLM prompts where H&E images are primary and FICTURE
is supplied as text evidence instead of a raw color image. Masks are resized to
the official ROI size using the same nearest-neighbor convention used by the
candidate source-pool evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


GROUPS = {
    "alveolar_AT2_F3_F5": [3, 5],
    "tumor_epithelial_F0_F2": [0, 2],
    "stroma_endothelial_F1_F9": [1, 9],
    "airway_epithelial_F7": [7],
    "immune_F4_F6_F8_F10_F11": [4, 6, 8, 10, 11],
}


def load_legend(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["Factor"] = int(row["Factor"])
        row["rgb_tuple"] = tuple(map(int, re.findall(r"\d+", row["RGB"])))
    return sorted(rows, key=lambda row: row["Factor"])


def factor_distribution(pixels: np.ndarray, legend: list[dict]) -> np.ndarray:
    if len(pixels) == 0:
        return np.zeros(len(legend), dtype=float)
    valid = pixels.sum(axis=1) > 0
    pixels = pixels[valid]
    if len(pixels) == 0:
        return np.zeros(len(legend), dtype=float)
    colors = np.array([row["rgb_tuple"] for row in legend], dtype=np.int16)
    pix = pixels.astype(np.int16, copy=False)
    distances = ((pix[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(legend)).astype(float)
    return counts / counts.sum() if counts.sum() else counts


def top_factor_text(props: np.ndarray, legend: list[dict], n: int = 5) -> str:
    order = np.argsort(-props)[:n]
    parts: list[str] = []
    for idx in order:
        value = float(props[idx])
        if value < 0.005:
            continue
        row = legend[idx]
        major = row.get("Major Compartment") or row.get("compartment") or ""
        cell_type = row.get("cell type") or row.get("Celltype2") or ""
        parts.append(f"F{row['Factor']} {value * 100:.1f}%: RGB {row['RGB']}; {major}; {cell_type}")
    return "; ".join(parts) if parts else "no dominant non-background FICTURE color"


def group_text(props: np.ndarray) -> str:
    parts: list[str] = []
    for name, factors in GROUPS.items():
        value = sum(float(props[factor]) for factor in factors if factor < len(props))
        parts.append(f"{name} {value * 100:.1f}%")
    return "; ".join(parts)


def expanded_bbox(bbox: list[int], width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * scale / 2.0)
    half_h = max(1.0, (y2 - y1) * scale / 2.0)
    return (
        max(0, int(round(cx - half_w))),
        max(0, int(round(cy - half_h))),
        min(width, int(round(cx + half_w))),
        min(height, int(round(cy + half_h))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-csv", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--legend-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--local-scale", type=float, default=2.0)
    args = parser.parse_args()

    legend = load_legend(args.legend_csv)
    ficture_img = Image.open(args.ficture_image).convert("RGB")
    ficture = np.array(ficture_img, dtype=np.uint8)
    width, height = ficture_img.size
    rows = list(csv.DictReader(args.hidden_csv.open(newline="")))
    output: list[dict] = []
    for row in rows:
        mask = Image.open(row["mask_path"]).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_arr = np.array(mask) > 0
        inside = factor_distribution(ficture[mask_arr], legend)
        bbox = json.loads(row["candidate_bbox_xyxy"])
        lx1, ly1, lx2, ly2 = expanded_bbox(bbox, width, height, args.local_scale)
        local_pixels = ficture[ly1:ly2, lx1:lx2].reshape(-1, 3)
        local = factor_distribution(local_pixels, legend)
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_uid": row["candidate_uid"],
                "inside_group_summary": group_text(inside),
                "inside_top_factors": top_factor_text(inside, legend),
                "local_group_summary": group_text(local),
                "local_top_factors": top_factor_text(local, legend),
                "local_bbox_xyxy": json.dumps([lx1, ly1, lx2, ly2]),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        fieldnames = [
            "candidate_id",
            "candidate_uid",
            "inside_group_summary",
            "inside_top_factors",
            "local_group_summary",
            "local_top_factors",
            "local_bbox_xyxy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
