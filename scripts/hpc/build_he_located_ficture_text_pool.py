#!/usr/bin/env python3
"""Build H&E-located / FICTURE-informed VLM requests.

The output pool shows the VLM only H&E images for location and morphology.
FICTURE is converted into per-candidate structured text computed from the
same candidate mask and a local ring around it.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional speedup
    ndi = None


GROUPS = {
    "airway_epithelial": [7],
    "endothelial": [9],
    "AT2_alveolar_epithelial": [3, 5],
    "stromal_mesenchymal": [1],
    "immune": [4, 6, 8, 10, 11],
    "tumor_like_epithelial": [0, 2],
}

DISPLAY_NAMES = {
    "airway_epithelial": "airway epithelial cells",
    "endothelial": "endothelial cells",
    "AT2_alveolar_epithelial": "AT2 / alveolar epithelial cells",
    "stromal_mesenchymal": "stromal / mesenchymal cells",
    "immune": "immune cells",
    "tumor_like_epithelial": "tumor-like epithelial cells",
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
    values = [int(item) for item in re.findall(r"\d+", text)]
    if len(values) != 3:
        raise ValueError(f"Cannot parse RGB from {text!r}")
    return tuple(values)  # type: ignore[return-value]


def load_factor_rgb(legend_csv: Path) -> dict[int, tuple[int, int, int]]:
    rows = read_csv(legend_csv)
    out: dict[int, tuple[int, int, int]] = {}
    for row in rows:
        factor = int(row["Factor"])
        out[factor] = parse_rgb(row["RGB"])
    return out


def mask_for_factor(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return (
        (rgb[..., 0] == color[0])
        & (rgb[..., 1] == color[1])
        & (rgb[..., 2] == color[2])
    )


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if ndi is not None:
        return ndi.binary_dilation(mask, iterations=radius)
    # PIL MaxFilter is dependency-light and good enough for a local context ring.
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    size = max(3, radius * 2 + 1)
    if size % 2 == 0:
        size += 1
    return np.asarray(img.filter(ImageFilter.MaxFilter(size=size))) > 0


def fractions_for_region(
    region: np.ndarray,
    factor_masks: dict[int, np.ndarray],
) -> tuple[dict[str, float], int]:
    valid = np.zeros(region.shape, dtype=bool)
    for factor_mask in factor_masks.values():
        valid |= factor_mask
    denom_mask = region & valid
    denom = int(denom_mask.sum())
    values: dict[str, float] = {}
    if denom == 0:
        return {key: 0.0 for key in GROUPS}, 0
    for group, factors in GROUPS.items():
        group_mask = np.zeros(region.shape, dtype=bool)
        for factor in factors:
            group_mask |= factor_masks[factor]
        values[group] = float((denom_mask & group_mask).sum() / denom)
    return values, denom


def enrichment_word(delta: float) -> str:
    if delta >= 0.15:
        return "strongly enriched"
    if delta >= 0.06:
        return "enriched"
    if delta <= -0.15:
        return "strongly depleted"
    if delta <= -0.06:
        return "depleted"
    return "similar"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_summary_text(
    inside: dict[str, float],
    around: dict[str, float],
    inside_pixels: int,
    around_pixels: int,
    spatial_grid_text: str = "",
) -> str:
    inside_lines = "\n".join(
        f"- {DISPLAY_NAMES[key]}: {format_percent(inside[key])}" for key in GROUPS
    )
    around_lines = "\n".join(
        f"- {DISPLAY_NAMES[key]}: {format_percent(around[key])}" for key in GROUPS
    )
    enrichment_lines = "\n".join(
        f"- {DISPLAY_NAMES[key]}: {enrichment_word(inside[key] - around[key])}"
        for key in GROUPS
    )
    spatial_block = (
        f"\nInside-mask spatial layout from FICTURE:\n{spatial_grid_text}\n"
        if spatial_grid_text
        else ""
    )
    return f"""FICTURE-derived information for the highlighted candidate:

Inside this candidate mask:
{inside_lines}

Around this candidate:
{around_lines}

Enrichment inside the candidate compared with nearby context:
{enrichment_lines}
{spatial_block}

Quality note: inside summary used {inside_pixels} FICTURE-colored pixels; around summary used {around_pixels} FICTURE-colored pixels.

Use H&E morphology and H&E location/context as the primary evidence. Use this FICTURE text only as molecular/cell-type supporting evidence. Do not interpret any FICTURE color image, because no FICTURE image is provided."""


def row_key(row: dict[str, str]) -> str:
    return "__".join(
        [
            row.get("label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            str(row.get("candidate_id", "")),
        ]
    )


def resolve_mask(row: dict[str, str], local_mask_root: Path | None) -> Path:
    if local_mask_root is not None:
        candidate_uid = row["candidate_uid"]
        candidate = local_mask_root / f"{candidate_uid}.png"
        if candidate.exists():
            return candidate
    path = Path(row["mask_path"])
    if not path.exists():
        raise FileNotFoundError(f"Mask not found for {row.get('candidate_uid')}: {path}")
    return path


def build_spatial_grid_text(
    mask: np.ndarray,
    factor_masks: dict[int, np.ndarray],
    grid_size: int = 3,
) -> str:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return ""
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if y1 <= y0 or x1 <= x0:
        return ""
    row_names = ["upper", "middle", "lower"]
    col_names = ["left", "center", "right"]
    y_edges = np.linspace(y0, y1, grid_size + 1).astype(int)
    x_edges = np.linspace(x0, x1, grid_size + 1).astype(int)
    lines: list[str] = []
    for yi in range(grid_size):
        for xi in range(grid_size):
            region = np.zeros(mask.shape, dtype=bool)
            region[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]] = True
            region &= mask
            values, denom = fractions_for_region(region, factor_masks)
            name = f"{row_names[yi]}-{col_names[xi]}"
            if denom == 0:
                lines.append(f"- {name}: no FICTURE-colored pixels inside this part of the mask")
                continue
            top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:2]
            desc = ", ".join(f"{DISPLAY_NAMES[key]} {format_percent(value)}" for key, value in top)
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--ficture-roi-rgb", type=Path, required=True)
    parser.add_argument("--legend-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--local-mask-root", type=Path)
    parser.add_argument("--ring-radius", type=int, default=128)
    parser.add_argument("--image-mode", choices=["he_piece_only", "he_three_image"], default="he_piece_only")
    parser.add_argument("--include-spatial-grid", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.pool_csv)
    rgb = np.asarray(Image.open(args.ficture_roi_rgb).convert("RGB"))
    factor_rgb = load_factor_rgb(args.legend_csv)
    factor_masks = {factor: mask_for_factor(rgb, color) for factor, color in factor_rgb.items()}

    out_rows: list[dict[str, str]] = []
    for row in rows:
        mask = load_mask(resolve_mask(row, args.local_mask_root))
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask/FICTURE shape mismatch for {row['candidate_uid']}: {mask.shape} vs {rgb.shape[:2]}")
        ring = dilate_mask(mask, args.ring_radius) & ~mask
        inside, inside_pixels = fractions_for_region(mask, factor_masks)
        around, around_pixels = fractions_for_region(ring, factor_masks)
        spatial_grid = build_spatial_grid_text(mask, factor_masks) if args.include_spatial_grid else ""

        out = dict(row)
        out["row_key"] = row_key(row)
        out["ficture_summary_text"] = build_summary_text(
            inside,
            around,
            inside_pixels,
            around_pixels,
            spatial_grid,
        )
        out["ficture_text_mode"] = (
            "inside_around_enrichment_spatial_grid_no_ficture_image"
            if args.include_spatial_grid
            else "inside_around_enrichment_no_ficture_image"
        )
        out["image_cue"] = (
            "Only H&E image inputs are shown. The highlighted/sharp candidate region is the object to classify. "
            "FICTURE is provided only as structured text computed from the same candidate mask."
        )
        out["image1_rel"] = row["he_crop_rel"]
        # If a future remote pool includes local/locator H&E columns, preserve them.
        if args.image_mode == "he_three_image":
            if row.get("he_local_context_rel"):
                out["image2_rel"] = row["he_local_context_rel"]
            if row.get("he_roi_locator_rel"):
                out["image3_rel"] = row["he_roi_locator_rel"]
        out_rows.append(out)

    write_csv(args.out_csv, out_rows)
    print(args.out_csv)


if __name__ == "__main__":
    main()
