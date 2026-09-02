#!/usr/bin/env python3
"""Build the audited 44 GeneMap box-plus-point prompts for TMA39."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


EDGE_STRUCTURE = np.array(
    [[False, True, False], [True, True, True], [False, True, False]],
    dtype=bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tma", default="TMA39")
    parser.add_argument("--old-labels-npz", type=Path, required=True)
    parser.add_argument("--module-npz", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--old-regions-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-shared-edge-bins", type=int, default=3)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=44,
        help="Expected prompt count; use 0 when applying the fixed rule to a new TMA.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_edges(length: int, bins: int) -> np.ndarray:
    return np.rint(np.linspace(0, length, bins + 1)).astype(np.int64)


def connected_region_labels(
    labels: np.ndarray, minimum_shared_edge_bins: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    connected = np.zeros_like(labels, dtype=np.int16)
    audit: list[dict[str, Any]] = []
    for module_index, current in enumerate(labels):
        current_count = int(current.max())
        parents = list(range(current_count + 1))

        def find(region_id: int) -> int:
            while parents[region_id] != region_id:
                parents[region_id] = parents[parents[region_id]]
                region_id = parents[region_id]
            return region_id

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        contacts: list[dict[str, int | bool]] = []
        for first in range(1, current_count + 1):
            expanded = ndimage.binary_dilation(current == first, structure=EDGE_STRUCTURE)
            for second in range(first + 1, current_count + 1):
                shared = int(np.logical_and(expanded, current == second).sum())
                if shared <= 0:
                    continue
                joined = shared >= minimum_shared_edge_bins
                contacts.append(
                    {
                        "old_region_1": first,
                        "old_region_2": second,
                        "shared_edge_bins": shared,
                        "joined": joined,
                    }
                )
                if joined:
                    union(first, second)

        root_to_output: dict[int, int] = {}
        old_to_new: dict[int, int] = {}
        for current_id in range(1, current_count + 1):
            root = find(current_id)
            if root not in root_to_output:
                root_to_output[root] = len(root_to_output) + 1
            new_id = root_to_output[root]
            old_to_new[current_id] = new_id
            connected[module_index][current == current_id] = new_id

        groups = []
        for new_id in range(1, len(root_to_output) + 1):
            members = [old for old, output in old_to_new.items() if output == new_id]
            groups.append(
                {
                    "new_region": new_id,
                    "old_regions": members,
                    "old_region_names": "+".join(f"R{value}" for value in members),
                }
            )
        audit.append(
            {
                "module": module_index + 1,
                "old_region_count": current_count,
                "new_region_count": len(root_to_output),
                "groups": groups,
                "contacts": contacts,
            }
        )
    return connected, audit


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    old_labels = np.load(args.old_labels_npz)["labels"].astype(np.int16)
    module_data = np.load(args.module_npz)
    raw_modules = module_data["raw_modules"].astype(np.float32)
    if raw_modules.shape != old_labels.shape:
        raise RuntimeError("Module score grid and old region labels have different shapes")
    with Image.open(args.reference_image) as image:
        width, height = image.size
    grid_h, grid_w = old_labels.shape[1:]
    x_edges = pixel_edges(width, grid_w)
    y_edges = pixel_edges(height, grid_h)

    old_rows = read_csv(args.old_regions_csv)
    old_by_module_region: dict[tuple[int, int], dict[str, str]] = {}
    for row in old_rows:
        prompt_id = row["prompt_id"]
        region_number = int(prompt_id.rsplit("_r", 1)[1])
        old_by_module_region[(int(row["gene_module"]), region_number)] = row

    labels, audit = connected_region_labels(old_labels, args.minimum_shared_edge_bins)
    region_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for module_index, module_labels in enumerate(labels):
        module_number = module_index + 1
        scores = raw_modules[module_index]
        module_audit = audit[module_index]
        group_by_id = {int(row["new_region"]): row for row in module_audit["groups"]}
        for new_id in range(1, int(module_labels.max()) + 1):
            region = module_labels == new_id
            yy, xx = np.nonzero(region)
            if not len(xx):
                raise RuntimeError(f"Empty merged region: module {module_number}, R{new_id}")
            region_scores = scores[yy, xx]
            peak_index = int(np.argmax(region_scores))
            peak_y_bin, peak_x_bin = int(yy[peak_index]), int(xx[peak_index])
            x1 = int(x_edges[int(xx.min())])
            y1 = int(y_edges[int(yy.min())])
            x2 = int(x_edges[int(xx.max()) + 1])
            y2 = int(y_edges[int(yy.max()) + 1])
            point_x = int(round((peak_x_bin + 0.5) * width / grid_w))
            point_y = int(round((peak_y_bin + 0.5) * height / grid_h))
            if not (x1 <= point_x <= x2 and y1 <= point_y <= y2):
                raise RuntimeError("Peak point fell outside merged region box")

            members = list(group_by_id[new_id]["old_regions"])
            old_member_rows = [old_by_module_region[(module_number, int(value))] for value in members]
            prompt_id = f"sustained_m{module_number:02d}_r{new_id:03d}"
            common = {
                "prompt_id": prompt_id,
                "source_prompt_id": prompt_id,
                "gene_module": module_number,
                "old_region_ids": "+".join(f"R{value}" for value in members),
                "old_region_count": len(members),
                "grown_area_gene_bins": int(region.sum()),
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "selection_used_annotation": False,
            }
            box_rows.append(dict(common))
            prompt_rows.append(dict(common, point_x=point_x, point_y=point_y))
            region_rows.append(
                dict(
                    common,
                    peak_point_x=point_x,
                    peak_point_y=point_y,
                    peak_score=float(scores[peak_y_bin, peak_x_bin]),
                    box_width=x2 - x1 + 1,
                    box_height=y2 - y1 + 1,
                    box_area_fraction=(x2 - x1 + 1) * (y2 - y1 + 1) / (width * height),
                    summed_old_core_area_gene_bins=sum(
                        int(row["core_area_gene_bins"]) for row in old_member_rows
                    ),
                )
            )
            group_rows.append(
                {
                    "gene_module": module_number,
                    "new_region": f"R{new_id}",
                    "old_regions": "+".join(f"R{value}" for value in members),
                    "old_region_count": len(members),
                    "merged": len(members) > 1,
                    "selection_used_annotation": False,
                }
            )

    if args.expected_count > 0 and len(prompt_rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} prompts, generated {len(prompt_rows)}"
        )
    if len({row["prompt_id"] for row in prompt_rows}) != len(prompt_rows):
        raise RuntimeError("Prompt identifiers are not unique")

    prompt_path = args.output / "adaptive_box_peak_sustained_edge3_context0.csv"
    box_path = args.output / "adaptive_boxes_sustained_edge3_context0.csv"
    region_path = args.output / "adaptive_regions.csv"
    group_path = args.output / "region_merge_audit.csv"
    labels_path = args.output / "adaptive_region_labels.npz"
    write_csv(prompt_path, prompt_rows)
    write_csv(box_path, box_rows)
    write_csv(region_path, region_rows)
    write_csv(group_path, group_rows)
    np.savez_compressed(labels_path, labels=labels)

    module_counts = {
        str(module): sum(int(row["gene_module"]) == module for row in prompt_rows)
        for module in range(1, labels.shape[0] + 1)
    }
    manifest = {
        "tma": args.tma,
        "purpose": "Generate GeneMap box-plus-point prompts with the fixed sustained-edge rule.",
        "prompt_rule": (
            "Within each GeneMap module, join old regions only when they share at least "
            f"{args.minimum_shared_edge_bins} edge-adjacent GeneMap bins. Draw one tight box "
            "around each resulting region and place one positive point at its highest module score."
        ),
        "minimum_shared_edge_bins": args.minimum_shared_edge_bins,
        "old_prompt_count": int(sum(int(item.max()) for item in old_labels)),
        "prompt_count": len(prompt_rows),
        "module_prompt_counts": module_counts,
        "full_image_size": [width, height],
        "gene_grid_shape": [grid_h, grid_w],
        "selection_used_annotation": False,
        "annotation_files_read": [],
        "inputs": {
            "old_labels_npz": str(args.old_labels_npz),
            "module_npz": str(args.module_npz),
            "reference_image": str(args.reference_image),
            "old_regions_csv": str(args.old_regions_csv),
        },
        "files": {},
    }
    for path in (prompt_path, box_path, region_path, group_path, labels_path):
        manifest["files"][path.name] = sha256(path)
    manifest_path = args.output / "adaptive_prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "adaptive_prompt_manifest.sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
