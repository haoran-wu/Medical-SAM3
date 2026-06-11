#!/usr/bin/env python3
"""Build a VLM pool with spatial FICTURE semantics instead of raw factor colors.

This keeps the useful part of FICTURE for VLMs: where cell-type-derived
signals occur relative to the highlighted candidate. It removes the raw
factor palette, which previous local VLM runs tended to treat as an ordinary
color image.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image


FACTOR_GROUPS = {
    "tumor_like_epithelial": [0, 2],
    "stromal_mesenchymal": [1],
    "alveolar_epithelial": [3, 5],
    "immune": [4, 6, 8, 10, 11],
    "airway_epithelial": [7],
    "endothelial": [9],
}

GROUP_COLORS = {
    "tumor_like_epithelial": (160, 80, 210),
    "stromal_mesenchymal": (40, 120, 220),
    "alveolar_epithelial": (245, 150, 45),
    "immune": (40, 180, 90),
    "airway_epithelial": (20, 180, 190),
    "endothelial": (220, 45, 100),
}

GROUP_DESCRIPTIONS = {
    "tumor_like_epithelial": "tumor-like epithelial signal",
    "stromal_mesenchymal": "stromal / mesenchymal signal",
    "alveolar_epithelial": "AT2 / alveolar epithelial signal",
    "immune": "immune-cell signal",
    "airway_epithelial": "airway epithelial signal",
    "endothelial": "endothelial / vascular signal",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_rgb(text: str) -> tuple[int, int, int]:
    values = [int(item) for item in re.findall(r"\d+", str(text))]
    if len(values) != 3:
        raise ValueError(f"Cannot parse RGB from {text!r}")
    return values[0], values[1], values[2]


def load_factor_rgb(legend_csv: Path) -> dict[int, tuple[int, int, int]]:
    out: dict[int, tuple[int, int, int]] = {}
    for row in read_csv(legend_csv):
        out[int(row["Factor"])] = parse_rgb(row["RGB"])
    return out


def factor_to_group() -> dict[int, str]:
    out: dict[int, str] = {}
    for group, factors in FACTOR_GROUPS.items():
        for factor in factors:
            out[factor] = group
    return out


def semantic_recolor(
    image_path: Path,
    factor_rgb: dict[int, tuple[int, int, int]],
    tolerance: float,
    exclude_groups: set[str],
) -> Image.Image:
    """Map raw FICTURE factor colors to a smaller semantic palette.

    Non-factor pixels are kept as a light grayscale background. This keeps the
    spatial layout readable but prevents raw factor colors from acting like an
    accidental label shortcut.
    """

    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    # Lighten non-factor pixels so the semantic signal is visually dominant.
    bg = np.clip(gray.astype(np.float32) * 0.55 + 115, 0, 255).astype(np.uint8)
    out = np.stack([bg, bg, bg], axis=-1)

    factor_items = sorted(factor_rgb.items())
    colors = np.asarray([rgb for _, rgb in factor_items], dtype=np.int16)
    # Exact color matching is too brittle because crops may contain antialiasing.
    # Nearest factor color within tolerance is a safer local remapping.
    diff = arr[..., None, :] - colors[None, None, :, :]
    dist = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=-1))
    nearest = np.argmin(dist, axis=-1)
    min_dist = np.min(dist, axis=-1)

    # Avoid mapping ordinary H&E pink/gray or gray-blur pixels into factor groups.
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    chroma = maxc - minc
    is_factor_like = (min_dist <= tolerance) & (chroma >= 35) & (maxc >= 80)

    f2g = factor_to_group()
    for factor_index, (factor, _) in enumerate(factor_items):
        group = f2g[factor]
        if group in exclude_groups:
            continue
        mask = is_factor_like & (nearest == factor_index)
        out[mask] = GROUP_COLORS[group]
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def legend_text(exclude_groups: set[str]) -> str:
    lines = ["Semantic FICTURE map legend:"]
    for group, color in GROUP_COLORS.items():
        if group in exclude_groups:
            continue
        lines.append(f"- RGB {color}: {GROUP_DESCRIPTIONS[group]}")
    if "tumor_like_epithelial" in exclude_groups:
        lines.append(
            "- Tumor-like epithelial signal is intentionally not colored in this map; "
            "tumor should be judged from H&E morphology."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--legend-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=80.0)
    parser.add_argument(
        "--exclude-groups",
        nargs="*",
        default=[],
        choices=sorted(GROUP_COLORS),
        help="Semantic groups to keep as gray background instead of coloring.",
    )
    args = parser.parse_args()

    source_root = args.source_pool.parent
    out_dir = args.output_dir
    semantic_dir = out_dir / "semantic_ficture_maps"
    semantic_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.source_pool)
    factor_rgb = load_factor_rgb(args.legend_csv)
    exclude_groups = set(args.exclude_groups)
    legend = legend_text(exclude_groups)
    out_rows: list[dict[str, str]] = []
    for row in rows:
        uid = row["candidate_uid"]
        piece_in = source_root / row["image2_rel"]
        local_in = source_root / row["image4_rel"]
        piece_out = semantic_dir / f"{uid}_semantic_ficture_piece.png"
        local_out = semantic_dir / f"{uid}_semantic_ficture_local.png"
        semantic_recolor(piece_in, factor_rgb, args.tolerance, exclude_groups).save(piece_out)
        semantic_recolor(local_in, factor_rgb, args.tolerance, exclude_groups).save(local_out)

        out = dict(row)
        out["image_cue"] = (
            "H&E images show morphology and candidate location. Semantic FICTURE maps preserve "
            "where simplified cell-type signals occur, but they are not raw FICTURE colors."
        )
        out["semantic_ficture_legend_text"] = legend
        # Keep original numbered order meaningful for the runner.
        out["image1_rel"] = row["image1_rel"]  # H&E piece
        out["image2_rel"] = str(piece_out.relative_to(out_dir))  # semantic FICTURE piece
        out["image3_rel"] = row["image3_rel"]  # H&E local context
        out["image4_rel"] = str(local_out.relative_to(out_dir))  # semantic FICTURE local context
        out["image5_rel"] = row["image5_rel"]  # H&E ROI locator
        out["image_sequence"] = (
            "1=H&E piece; 2=semantic FICTURE piece; 3=H&E local context; "
            "4=semantic FICTURE local context; 5=H&E ROI locator"
        )
        out_rows.append(out)

    write_csv(out_dir / "public_vlm_requests.csv", out_rows)
    # Symlink or copy the source image directories so existing relative paths resolve.
    for dirname in ["candidate_pair_crops", "context_images"]:
        target = source_root / dirname
        link = out_dir / dirname
        if not link.exists():
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                pass
    truth = source_root / "hidden_candidate_truth.csv"
    if truth.exists() and not (out_dir / "hidden_candidate_truth.csv").exists():
        try:
            (out_dir / "hidden_candidate_truth.csv").symlink_to(truth)
        except OSError:
            pass
    (out_dir / "semantic_ficture_map_legend.txt").write_text(legend + "\n", encoding="utf-8")
    print(out_dir / "public_vlm_requests.csv")


if __name__ == "__main__":
    main()
