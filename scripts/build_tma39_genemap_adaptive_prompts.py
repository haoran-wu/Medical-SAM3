#!/usr/bin/env python3
"""Build annotation-free GeneMap prompts whose boxes follow stable expression regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tma", default="TMA39")
    parser.add_argument("--module-npz", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-core-bins", type=int, default=2)
    parser.add_argument("--context-bins", type=float, default=1.0)
    parser.add_argument("--variant-name", default="context1")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(mask: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    result = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    result = ndi.binary_fill_holes(result)
    return np.logical_and(result, tissue)


def filtered_labels(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    output = np.zeros_like(labels, dtype=np.int16)
    next_label = 0
    for label in range(1, count + 1):
        component = labels == label
        if int(component.sum()) < minimum_size:
            continue
        next_label += 1
        output[component] = next_label
    return output


def pixel_edges(length: int, bins: int) -> np.ndarray:
    return np.rint(np.linspace(0, length, bins + 1)).astype(np.int64)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    data = np.load(args.module_npz)
    modules = data["raw_modules"].astype(np.float32)
    tissue = data["tissue"].astype(bool)
    with Image.open(args.reference_image) as image:
        width, height = image.size

    grid_h, grid_w = modules.shape[1:]
    x_edges = pixel_edges(width, grid_w)
    y_edges = pixel_edges(height, grid_h)
    context_x = int(round(float(np.median(np.diff(x_edges))) * args.context_bins))
    context_y = int(round(float(np.median(np.diff(y_edges))) * args.context_bins))

    region_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    box_peak_rows: list[dict[str, Any]] = []
    grown_labels = np.zeros_like(modules, dtype=np.int16)
    threshold_rows: list[dict[str, Any]] = []

    for module_index, module in enumerate(modules):
        values = module[tissue]
        threshold_1 = float(np.quantile(values, 0.99))
        threshold_2p5 = float(np.quantile(values, 0.975))
        threshold_5 = float(np.quantile(values, 0.95))

        core = clean(np.logical_and(module >= threshold_1, tissue), tissue)
        middle = clean(np.logical_and(module >= threshold_2p5, tissue), tissue)
        support = clean(np.logical_and(module >= threshold_5, tissue), tissue)
        # Preserve the nested relationship after morphology.
        middle = np.logical_or(middle, core)
        support = np.logical_or(support, middle)

        core_labels = filtered_labels(core, args.minimum_core_bins)
        support_labels, support_count = ndi.label(
            support, structure=np.ones((3, 3), dtype=bool)
        )
        next_output_label = 0

        for support_label in range(1, support_count + 1):
            support_component = support_labels == support_label
            core_ids = [
                int(value)
                for value in np.unique(core_labels[support_component])
                if int(value) > 0
            ]
            if not core_ids:
                continue

            if len(core_ids) == 1:
                assigned = {core_ids[0]: support_component}
            else:
                distances = np.stack(
                    [ndi.distance_transform_edt(core_labels != core_id) for core_id in core_ids]
                )
                nearest = np.argmin(distances, axis=0)
                assigned = {
                    core_id: np.logical_and(support_component, nearest == index)
                    for index, core_id in enumerate(core_ids)
                }

            for core_id in core_ids:
                grown = assigned[core_id]
                core_component = core_labels == core_id
                if not np.any(grown) or not np.any(core_component):
                    continue
                next_output_label += 1
                grown_labels[module_index, grown] = next_output_label

                yy, xx = np.nonzero(grown)
                core_y, core_x = np.nonzero(core_component)
                core_scores = module[core_y, core_x]
                peak_index = int(np.argmax(core_scores))
                peak_x_bin = int(core_x[peak_index])
                peak_y_bin = int(core_y[peak_index])

                raw_x1 = int(x_edges[int(xx.min())])
                raw_y1 = int(y_edges[int(yy.min())])
                raw_x2 = int(x_edges[int(xx.max()) + 1])
                raw_y2 = int(y_edges[int(yy.max()) + 1])
                box_x1 = max(0, raw_x1 - context_x)
                box_y1 = max(0, raw_y1 - context_y)
                box_x2 = min(width - 1, raw_x2 + context_x)
                box_y2 = min(height - 1, raw_y2 + context_y)
                point_x = int(round((peak_x_bin + 0.5) * width / grid_w))
                point_y = int(round((peak_y_bin + 0.5) * height / grid_h))
                if not (box_x1 <= point_x <= box_x2 and box_y1 <= point_y <= box_y2):
                    raise RuntimeError("Peak point fell outside its adaptive box")

                prompt_id = f"adaptive_m{module_index + 1:02d}_r{next_output_label:03d}"
                base = {
                    "prompt_id": prompt_id,
                    "gene_module": module_index + 1,
                    "core_area_gene_bins": int(core_component.sum()),
                    "grown_area_gene_bins": int(grown.sum()),
                    "core_threshold_top_percent": 1.0,
                    "middle_threshold_top_percent": 2.5,
                    "extent_threshold_top_percent": 5.0,
                    "context_gene_bins": args.context_bins,
                    "box_x1": box_x1,
                    "box_y1": box_y1,
                    "box_x2": box_x2,
                    "box_y2": box_y2,
                    "selection_used_annotation": False,
                }
                box_rows.append(dict(base))
                box_peak_rows.append(dict(base, point_x=point_x, point_y=point_y))
                region_rows.append(
                    dict(
                        base,
                        peak_point_x=point_x,
                        peak_point_y=point_y,
                        raw_extent_box_x1=raw_x1,
                        raw_extent_box_y1=raw_y1,
                        raw_extent_box_x2=raw_x2,
                        raw_extent_box_y2=raw_y2,
                        box_width=box_x2 - box_x1 + 1,
                        box_height=box_y2 - box_y1 + 1,
                        box_area_fraction=(box_x2 - box_x1 + 1)
                        * (box_y2 - box_y1 + 1)
                        / (width * height),
                        support_component_core_count=len(core_ids),
                    )
                )

        threshold_rows.append(
            {
                "gene_module": module_index + 1,
                "top1_threshold": threshold_1,
                "top2p5_threshold": threshold_2p5,
                "top5_threshold": threshold_5,
                "adaptive_region_count": next_output_label,
                "selection_used_annotation": False,
            }
        )

    box_name = f"adaptive_boxes_{args.variant_name}.csv"
    box_peak_name = f"adaptive_box_peak_{args.variant_name}.csv"
    write_csv(args.output / "adaptive_regions.csv", region_rows)
    write_csv(args.output / box_name, box_rows)
    write_csv(args.output / box_peak_name, box_peak_rows)
    write_csv(args.output / "module_thresholds.csv", threshold_rows)
    labels_path = args.output / "adaptive_region_labels.npz"
    np.savez_compressed(labels_path, labels=grown_labels)

    widths = np.array([int(row["box_width"]) for row in region_rows])
    heights = np.array([int(row["box_height"]) for row in region_rows])
    areas = np.array([float(row["box_area_fraction"]) for row in region_rows])
    manifest = {
        "tma": args.tma,
        "purpose": "Use GeneMap to place region-sized SAM prompts without annotation.",
        "method": [
            "Use the strongest 1% of tissue pixels as location cores.",
            "Confirm that each core remains inside connected areas at the strongest 2.5% and 5% thresholds.",
            "Use the strongest-5% connected extent to size the box.",
            "When several cores join at 5%, divide the shared extent by nearest core.",
            f"Add {args.context_bins:g} GeneMap grid bins around the extent as image context.",
            "Test the same box both alone and with its highest-expression point.",
        ],
        "prompt_count": len(region_rows),
        "variant_name": args.variant_name,
        "module_count": int(len(modules)),
        "module_prompt_counts": {
            str(index): sum(int(row["gene_module"]) == index for row in region_rows)
            for index in range(1, len(modules) + 1)
        },
        "full_image_size": [width, height],
        "gene_grid_shape": [grid_h, grid_w],
        "context_pixels": [context_x, context_y],
        "box_width_pixels": {
            "minimum": int(widths.min()),
            "median": float(np.median(widths)),
            "maximum": int(widths.max()),
        },
        "box_height_pixels": {
            "minimum": int(heights.min()),
            "median": float(np.median(heights)),
            "maximum": int(heights.max()),
        },
        "box_area_fraction": {
            "minimum": float(areas.min()),
            "median": float(np.median(areas)),
            "maximum": float(areas.max()),
        },
        "selection_used_annotation": False,
        "annotation_files_read": [],
        "files": {},
    }
    for path in (
        args.output / "adaptive_regions.csv",
        args.output / box_name,
        args.output / box_peak_name,
        args.output / "module_thresholds.csv",
        labels_path,
    ):
        manifest["files"][path.name] = sha256(path)
    manifest_path = args.output / "adaptive_prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "adaptive_prompt_manifest.sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
