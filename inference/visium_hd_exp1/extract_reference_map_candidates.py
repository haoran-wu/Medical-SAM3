#!/usr/bin/env python3
"""Extract region candidates directly from a reference-colored pixel map.

The reference map is not an H&E image. It is already a spatial factor/cell-type
assignment rendered as colors, so the most faithful candidates are connected
components in factor space rather than SAM3 free-form masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology


DEFAULT_MAP = Path(
    "/Users/haoranwu/Desktop/visiumhd_exp1_pixel_cell_type_result/reference_aligned/"
    "hex_12.k12.pixel.rot90.reference_color_mapped.png"
)
DEFAULT_INFO = Path("/Users/haoranwu/Desktop/visiumhd_exp1_pixel_cell_type_result/hex_12.k12.pixel.info.tsv")
DEFAULT_OUT = PROJECT_ROOT / "output" / "visium_hd_exp1" / "reference_map_candidates" / "hex12_k12"


def parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB value: {value}")
    return tuple(parts)


def load_factor_info(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: List[Dict[str, object]] = []
        for row in reader:
            row["Factor"] = int(row["Factor"])
            row["RGB_tuple"] = parse_rgb(str(row["RGB"]))
            row["Weight"] = float(row["Weight"])
            row["PostUMI"] = int(row["PostUMI"])
            rows.append(row)
    return rows


def nearest_palette_assignment(
    image: np.ndarray,
    palette: Sequence[Tuple[int, int, int]],
    max_color_distance: float,
) -> np.ndarray:
    """Assign each non-background pixel to the nearest listed palette color."""
    arr = image.astype(np.int32)
    flat = arr.reshape(-1, 3)
    palette_arr = np.array(palette, dtype=np.int32)
    # Squared distance to each factor color; 12 factors is small enough to do exactly.
    d2 = ((flat[:, None, :] - palette_arr[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1).astype(np.int16)
    min_dist = np.sqrt(d2.min(axis=1))
    is_black = (flat == 0).all(axis=1)
    assigned = nearest
    assigned[(min_dist > max_color_distance) | is_black] = -1
    return assigned.reshape(image.shape[:2])


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_overlay(base: np.ndarray, masks: Sequence[np.ndarray], colors: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    overlay = base.copy().astype(np.float32)
    for mask, color in zip(masks, colors):
        if mask.sum() == 0:
            continue
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract connected reference-map factor candidates.")
    parser.add_argument("--map-path", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--info-path", type=Path, default=DEFAULT_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-color-distance", type=float, default=8.0)
    parser.add_argument("--min-component-area", type=int, default=1000)
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--top-components-per-factor", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "candidate_masks"
    panels_dir = args.output_dir / "factor_panels"
    masks_dir.mkdir(exist_ok=True)
    panels_dir.mkdir(exist_ok=True)

    image = np.array(Image.open(args.map_path).convert("RGB"))
    factors = load_factor_info(args.info_path)
    palette = [row["RGB_tuple"] for row in factors]
    assignment = nearest_palette_assignment(image, palette, args.max_color_distance)

    candidates: List[Dict[str, object]] = []
    factor_masks: List[np.ndarray] = []
    factor_colors: List[Tuple[int, int, int]] = []
    structure = np.ones((3, 3), dtype=bool)

    for palette_idx, row in enumerate(factors):
        factor = int(row["Factor"])
        rgb = tuple(int(v) for v in row["RGB_tuple"])
        factor_mask = assignment == palette_idx
        if args.close_radius > 0:
            factor_mask = morphology.binary_closing(factor_mask, morphology.disk(args.close_radius))
        factor_mask = morphology.remove_small_objects(factor_mask, min_size=args.min_component_area)
        factor_masks.append(factor_mask)
        factor_colors.append(rgb)

        labels, n_labels = ndi.label(factor_mask, structure=structure)
        component_records: List[Dict[str, object]] = []
        for component_id in range(1, n_labels + 1):
            component = labels == component_id
            area = int(component.sum())
            if area < args.min_component_area:
                continue
            x0, y0, x1, y1 = bbox_from_mask(component)
            component_records.append(
                {
                    "factor": factor,
                    "component_id": component_id,
                    "area": area,
                    "bbox_xyxy": [x0, y0, x1, y1],
                }
            )

        component_records.sort(key=lambda item: int(item["area"]), reverse=True)
        for rank, record in enumerate(component_records[: args.top_components_per_factor], start=1):
            component = labels == int(record["component_id"])
            mask_path = masks_dir / f"factor_{factor:02d}_rank_{rank:02d}.png"
            Image.fromarray((component.astype(np.uint8) * 255)).save(mask_path)
            record.update(
                {
                    "rank_within_factor": rank,
                    "rgb": list(rgb),
                    "weight": float(row["Weight"]),
                    "mask_path": str(mask_path.relative_to(args.output_dir)),
                    "top_genes_specific": str(row["TopGene_specific"]).split(", ")[:10],
                }
            )
            candidates.append(record)

        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        axes[0].imshow(image)
        axes[0].set_title("Reference map")
        axes[0].axis("off")
        axes[1].imshow(factor_mask, cmap="gray")
        axes[1].set_title(f"Factor {factor} mask, components kept={len(component_records)}")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(panels_dir / f"factor_{factor:02d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    overlay = make_overlay(image, factor_masks, factor_colors)
    Image.fromarray(overlay).save(args.output_dir / "factor_mask_overlay.png")

    candidates.sort(key=lambda item: int(item["area"]), reverse=True)
    with (args.output_dir / "candidate_components.csv").open("w", newline="") as f:
        fieldnames = [
            "factor",
            "rank_within_factor",
            "area",
            "bbox_xyxy",
            "rgb",
            "weight",
            "mask_path",
            "top_genes_specific",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            writer.writerow({key: item[key] for key in fieldnames})

    report = {
        "map_path": str(args.map_path),
        "info_path": str(args.info_path),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "max_color_distance": args.max_color_distance,
        "min_component_area": args.min_component_area,
        "close_radius": args.close_radius,
        "n_factors": len(factors),
        "n_candidates": len(candidates),
        "factors": [
            {
                "factor": int(row["Factor"]),
                "rgb": list(row["RGB_tuple"]),
                "weight": float(row["Weight"]),
                "top_genes_specific": str(row["TopGene_specific"]).split(", ")[:10],
                "n_candidates": sum(1 for item in candidates if int(item["factor"]) == int(row["Factor"])),
            }
            for row in factors
        ],
    }
    (args.output_dir / "candidate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
