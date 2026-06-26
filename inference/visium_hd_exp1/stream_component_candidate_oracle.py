#!/usr/bin/env python3
"""Memory-light component-wise candidate oracle.

This scans full Bouchet candidate pools one mask at a time and keeps only the
top-k metric rows for each annotation component. Unlike
`componentwise_candidate_oracle.py`, it does not keep full mask arrays for the
top-k candidates in memory, so it is suitable for larger top-k searches on
Slurm compute nodes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


LABEL_TO_MASK = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
    "immune infiltration": "03_immune_infiltration_target_roi.png",
}


@dataclass(frozen=True)
class Component:
    component_id: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["immune_infiltration"], choices=sorted(LABEL_TO_MASK))
    parser.add_argument("--candidate-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-component-area", type=int, default=256)
    parser.add_argument("--top-k-per-component", type=int, default=100)
    parser.add_argument("--max-candidates-per-root", type=int, default=0)
    return parser.parse_args()


def load_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if expected_size is not None and image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def split_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[Component]]:
    labeled, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndi.find_objects(labeled)
    components: list[Component] = []
    kept = np.zeros_like(labeled, dtype=np.int32)
    next_id = 1
    for original_id, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        component_mask = labeled[sl] == original_id
        area = int(component_mask.sum())
        if area < min_area:
            continue
        y0, y1 = int(sl[0].start), int(sl[0].stop)
        x0, x1 = int(sl[1].start), int(sl[1].stop)
        kept[sl][component_mask] = next_id
        components.append(Component(next_id, area, x0, y0, x1, y1))
        next_id += 1
    components.sort(key=lambda item: item.area, reverse=True)
    remapped = np.zeros_like(kept)
    remapped_components: list[Component] = []
    for new_id, component in enumerate(components, start=1):
        remapped[kept == component.component_id] = new_id
        remapped_components.append(
            Component(new_id, component.area, component.x0, component.y0, component.x1, component.y1)
        )
    return remapped, remapped_components


def metrics_from_counts(tp: int, pred_area: int, gt_area: int) -> dict[str, float]:
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return {"dice": dice, "precision": precision, "recall": recall}


def candidate_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("candidate_", "") if stem.startswith("candidate_") else stem


def parse_candidate_roots(entries: list[str]) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--candidate-root must be NAME=PATH, got {entry}")
        name, value = entry.split("=", 1)
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        roots.append((name, path))
    return roots


def push_top(top_rows: list[dict[str, object]], row: dict[str, object], top_k: int) -> None:
    top_rows.append(row)
    top_rows.sort(key=lambda item: float(item["dice_vs_component"]), reverse=True)
    del top_rows[top_k:]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_size = Image.open(args.he_image).size
    roots = parse_candidate_roots(args.candidate_root)

    component_summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    full_best_rows: list[dict[str, object]] = []

    for label in args.labels:
        gt = load_mask(args.annotation_dir / LABEL_TO_MASK[label], expected_size)
        component_map, components = split_components(gt, args.min_component_area)
        gt_area = int(gt.sum())
        component_masks = [component_map == component.component_id for component in components]
        component_crops = [
            mask[component.y0 : component.y1, component.x0 : component.x1]
            for mask, component in zip(component_masks, components)
        ]
        for component in components:
            component_summary_rows.append(
                {
                    "class": label,
                    "component": f"C{component.component_id}",
                    "area_px": component.area,
                    "area_%_of_GT": 100 * component.area / gt_area if gt_area else 0.0,
                    "bbox_xyxy": f"{component.x0},{component.y0},{component.x1},{component.y1}",
                }
            )

        for source, root in roots:
            top_by_component: dict[str, list[dict[str, object]]] = {
                f"C{component.component_id}": [] for component in components
            }
            best_full: dict[str, object] | None = None
            paths = sorted(root.rglob("candidate_masks/candidate_*.png"))
            if args.max_candidates_per_root > 0:
                paths = paths[: args.max_candidates_per_root]
            for idx, path in enumerate(paths, start=1):
                mask = load_mask(path, expected_size)
                pred_area = int(mask.sum())
                full_tp = int(mask[gt].sum())
                full_m = metrics_from_counts(full_tp, pred_area, gt_area)
                setting = path.parent.parent.name if path.parent.name == "candidate_masks" else path.parent.name
                candidate_id = candidate_id_from_path(path)
                if best_full is None or full_m["dice"] > float(best_full["dice"]):
                    best_full = {
                        "class": label,
                        "source": source,
                        "dice": full_m["dice"],
                        "precision": full_m["precision"],
                        "recall": full_m["recall"],
                        "setting": setting,
                        "candidate": candidate_id,
                        "mask_path": str(path),
                    }
                for component, component_crop, component_mask in zip(components, component_crops, component_masks):
                    pred_crop = mask[component.y0 : component.y1, component.x0 : component.x1]
                    tp = int(pred_crop[component_crop].sum())
                    m = metrics_from_counts(tp, pred_area, component.area)
                    row = {
                        "class": label,
                        "source": source,
                        "component": f"C{component.component_id}",
                        "dice_vs_component": m["dice"],
                        "precision_vs_component": m["precision"],
                        "recall_vs_component": m["recall"],
                        "setting": setting,
                        "candidate": candidate_id,
                        "resized_from": "",
                        "mask_path": str(path),
                    }
                    push_top(top_by_component[f"C{component.component_id}"], row, args.top_k_per_component)
                if idx % 1000 == 0:
                    best_dice = float(best_full["dice"]) if best_full else 0.0
                    print(f"{label} {source}: scanned {idx}/{len(paths)} masks; best full Dice={best_dice:.4f}", flush=True)
            if best_full:
                full_best_rows.append(best_full)
            for component_id, rows in top_by_component.items():
                for rank, row in enumerate(rows, start=1):
                    row = dict(row)
                    row["rank"] = rank
                    candidate_rows.append(row)

    component_df = pd.DataFrame(component_summary_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df[
            [
                "class",
                "source",
                "component",
                "rank",
                "dice_vs_component",
                "precision_vs_component",
                "recall_vs_component",
                "setting",
                "candidate",
                "resized_from",
                "mask_path",
            ]
        ]
    component_df.to_csv(args.out_dir / "gt_component_summary.csv", index=False)
    candidates_df.to_csv(args.out_dir / "component_top_candidates.csv", index=False)
    pd.DataFrame(full_best_rows).to_csv(args.out_dir / "full_single_best_by_source.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "annotation_dir": str(args.annotation_dir),
                "he_image": str(args.he_image),
                "labels": args.labels,
                "candidate_roots": args.candidate_root,
                "min_component_area": args.min_component_area,
                "top_k_per_component": args.top_k_per_component,
                "max_candidates_per_root": args.max_candidates_per_root,
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
