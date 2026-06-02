#!/usr/bin/env python3
"""Diagnose under-covered annotation components and try targeted recall unions.

This is used after a broad-class component union, especially immune
infiltration, when a visually meaningful annotation region is still missed. It
does not rescan the full mask pool directly; it consumes the full-pool
component oracle CSV produced on Bouchet and searches candidate additions from
that oracle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


@dataclass(frozen=True)
class Component:
    component_id: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class TargetPolicy:
    name: str
    max_adds: int
    min_delta_recall: float
    min_incremental_precision: float
    max_precision_drop: float
    rank_max: int


POLICIES = [
    TargetPolicy("targeted_balanced", 6, 0.002, 0.12, 0.040, 50),
    TargetPolicy("targeted_recall_push", 10, 0.001, 0.06, 0.080, 75),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-mask", type=Path, required=True)
    parser.add_argument("--current-mask", type=Path, required=True)
    parser.add_argument("--component-candidates", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-component-area", type=int, default=256)
    parser.add_argument("--important-area", type=int, default=1000)
    parser.add_argument("--low-recall", type=float, default=0.35)
    parser.add_argument("--left-max-frac", type=float, default=0.60)
    parser.add_argument("--bottom-min-frac", type=float, default=0.45)
    parser.add_argument("--target-components", default="", help="Comma-separated component IDs such as C15,C20.")
    parser.add_argument("--rank-max", type=int, default=0, help="Override candidate rank cutoff for a single custom policy.")
    parser.add_argument("--max-adds", type=int, default=0, help="Override max additions for a single custom policy.")
    parser.add_argument("--min-delta-recall", type=float, default=-1.0)
    parser.add_argument("--min-incremental-precision", type=float, default=-1.0)
    parser.add_argument("--max-precision-drop", type=float, default=-1.0)
    return parser.parse_args()


def load_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if expected_size is not None and image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


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


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "pred_area": float(pred_area),
        "gt_area": float(gt_area),
    }


def candidate_stats(candidate: np.ndarray, current: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    before = metrics(current, gt)
    proposed = np.logical_or(current, candidate)
    after = metrics(proposed, gt)
    new_pixels = np.logical_and(candidate, ~current)
    new_area = int(new_pixels.sum())
    new_tp = int(np.logical_and(new_pixels, gt).sum())
    return {
        **{f"new_{key}": value for key, value in after.items()},
        "delta_dice": after["dice"] - before["dice"],
        "delta_precision": after["precision"] - before["precision"],
        "delta_recall": after["recall"] - before["recall"],
        "incremental_precision": new_tp / new_area if new_area else 0.0,
        "new_area": float(new_area),
        "new_tp": float(new_tp),
    }


def row_id(row: pd.Series | dict[str, object]) -> str:
    return (
        f"{row.get('source', '')}_{row.get('component', '')}_rank{row.get('rank', '')}_"
        f"{row.get('setting', '')}_candidate_{row.get('candidate', '')}"
    )


def component_rows(
    component_map: np.ndarray,
    components: list[Component],
    current: np.ndarray,
    gt_area: int,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component in components:
        comp_mask = component_map == component.component_id
        covered = int(np.logical_and(comp_mask, current).sum())
        cx = 0.5 * (component.x0 + component.x1)
        cy = 0.5 * (component.y0 + component.y1)
        is_left_bottom = cx <= args.left_max_frac * width and cy >= args.bottom_min_frac * height
        component_recall = covered / component.area if component.area else 0.0
        rows.append(
            {
                "component": f"C{component.component_id}",
                "area_px": component.area,
                "area_%_of_GT": 100 * component.area / gt_area if gt_area else 0.0,
                "bbox_xyxy": f"{component.x0},{component.y0},{component.x1},{component.y1}",
                "centroid_x": cx,
                "centroid_y": cy,
                "component_recall_by_current_union": component_recall,
                "covered_px": covered,
                "missed_px": component.area - covered,
                "is_left_bottom": is_left_bottom,
                "is_important_low_recall": component.area >= args.important_area and component_recall < args.low_recall,
            }
        )
    return rows


def prepare_candidate_rows(df: pd.DataFrame, target_components: set[str], policy: TargetPolicy) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for _, row in df.iterrows():
        component = str(row.get("component", ""))
        if component not in target_components:
            continue
        if int(row.get("rank", 999999)) > policy.rank_max:
            continue
        path = str(row.get("mask_path", ""))
        if not path or path in seen_paths or not Path(path).exists():
            continue
        clean = row.to_dict()
        clean["row_id"] = row_id(clean)
        rows.append(clean)
        seen_paths.add(path)
    return rows


def greedy_targeted(
    rows: list[dict[str, object]],
    current: np.ndarray,
    gt: np.ndarray,
    baseline: dict[str, float],
    policy: TargetPolicy,
    expected_size: tuple[int, int],
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, float]]:
    selected: list[dict[str, object]] = []
    union = current.copy()
    used: set[str] = set()
    for _ in range(policy.max_adds):
        best: tuple[float, dict[str, object], np.ndarray, dict[str, float]] | None = None
        for row in rows:
            rid = str(row["row_id"])
            if rid in used:
                continue
            candidate = load_mask(Path(str(row["mask_path"])), expected_size)
            stats = candidate_stats(candidate, union, gt)
            if stats["delta_recall"] < policy.min_delta_recall:
                continue
            if stats["incremental_precision"] < policy.min_incremental_precision:
                continue
            if stats["new_precision"] < baseline["precision"] - policy.max_precision_drop:
                continue
            proposed = np.logical_or(union, candidate)
            score = (
                1.5 * stats["delta_recall"]
                + 0.25 * max(stats["delta_dice"], 0.0)
                + 0.05 * stats["incremental_precision"]
            )
            if best is None or score > best[0]:
                best = (score, row, proposed, stats)
        if best is None:
            break
        _, row, union, stats = best
        used.add(str(row["row_id"]))
        selected.append({**row, **stats})
    return selected, union, metrics(union, gt)


def overlay_components(base: Image.Image, gt: np.ndarray, current: np.ndarray, target: np.ndarray, title: str) -> Image.Image:
    rgba = base.convert("RGBA")
    arr = np.zeros((*gt.shape, 4), dtype=np.uint8)
    arr[np.logical_and(gt, current)] = (0, 180, 80, 110)
    arr[np.logical_and(gt, ~current)] = (255, 0, 0, 145)
    arr[target] = (255, 210, 0, 145)
    arr[np.logical_and(target, ~gt)] = (255, 120, 0, 120)
    out = Image.alpha_composite(rgba, Image.fromarray(arr, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 1350), 38), fill=(255, 255, 255, 230))
    draw.text((10, 11), title + " | green=covered GT, red=missed GT, yellow/orange=target components", fill=(0, 0, 0, 255))
    return out.convert("RGB")


def overlay_prediction(base: Image.Image, gt: np.ndarray, pred: np.ndarray, title: str) -> Image.Image:
    rgba = base.convert("RGBA")
    arr = np.zeros((*gt.shape, 4), dtype=np.uint8)
    arr[np.logical_and(gt, pred)] = (0, 180, 80, 125)
    arr[np.logical_and(~gt, pred)] = (0, 90, 255, 120)
    arr[np.logical_and(gt, ~pred)] = (255, 0, 0, 140)
    out = Image.alpha_composite(rgba, Image.fromarray(arr, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 1250), 38), fill=(255, 255, 255, 230))
    draw.text((10, 11), title + " | green=TP, blue=extra, red=missed", fill=(0, 0, 0, 255))
    return out.convert("RGB")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_mask(args.gt_mask)
    expected_size = (gt.shape[1], gt.shape[0])
    current = load_mask(args.current_mask, expected_size)
    he = Image.open(args.he_image).convert("RGB")
    component_map, components = split_components(gt, args.min_component_area)
    base_m = metrics(current, gt)
    rows = component_rows(component_map, components, current, int(gt.sum()), expected_size[0], expected_size[1], args)
    component_df = pd.DataFrame(rows)
    component_df.to_csv(args.out_dir / "component_current_coverage.csv", index=False)

    if args.target_components.strip():
        target_components = {
            item.strip()
            for item in args.target_components.split(",")
            if item.strip()
        }
    else:
        target_components = set(
            component_df[
                (component_df["is_important_low_recall"])
                | ((component_df["is_left_bottom"]) & (component_df["component_recall_by_current_union"] < 0.60))
            ]["component"].astype(str)
        )
    target_mask = np.isin(component_map, [int(comp[1:]) for comp in target_components])
    overlay_components(
        he,
        gt,
        current,
        target_mask,
        f"under-covered targets: {len(target_components)} components",
    ).save(args.out_dir / "undercovered_components_on_he.png")

    candidates_df = pd.read_csv(args.component_candidates)
    summary_rows: list[dict[str, object]] = [
        {
            "policy": "current",
            "selected_count": 0,
            "Dice": base_m["dice"],
            "Precision": base_m["precision"],
            "Recall": base_m["recall"],
            "delta_dice": 0.0,
            "delta_precision": 0.0,
            "delta_recall": 0.0,
            "target_components": ";".join(sorted(target_components)),
            "selected": "",
        }
    ]
    selected_rows: list[dict[str, object]] = []
    policies = POLICIES
    if any(
        [
            args.rank_max > 0,
            args.max_adds > 0,
            args.min_delta_recall >= 0,
            args.min_incremental_precision >= 0,
            args.max_precision_drop >= 0,
        ]
    ):
        policies = [
            TargetPolicy(
                "custom_targeted",
                args.max_adds if args.max_adds > 0 else 6,
                args.min_delta_recall if args.min_delta_recall >= 0 else 0.001,
                args.min_incremental_precision if args.min_incremental_precision >= 0 else 0.08,
                args.max_precision_drop if args.max_precision_drop >= 0 else 0.060,
                args.rank_max if args.rank_max > 0 else 25,
            )
        ]

    for policy in policies:
        candidate_rows = prepare_candidate_rows(candidates_df, target_components, policy)
        selected, union, final_m = greedy_targeted(candidate_rows, current, gt, base_m, policy, expected_size)
        save_mask(union, args.out_dir / f"{policy.name}_union_mask.png")
        overlay_prediction(
            he,
            gt,
            union,
            f"{policy.name}: Dice {final_m['dice']:.3f}, Precision {final_m['precision']:.3f}, Recall {final_m['recall']:.3f}",
        ).save(args.out_dir / f"{policy.name}_union_on_he.png")
        summary_rows.append(
            {
                "policy": policy.name,
                "selected_count": len(selected),
                "Dice": final_m["dice"],
                "Precision": final_m["precision"],
                "Recall": final_m["recall"],
                "delta_dice": final_m["dice"] - base_m["dice"],
                "delta_precision": final_m["precision"] - base_m["precision"],
                "delta_recall": final_m["recall"] - base_m["recall"],
                "target_components": ";".join(sorted(target_components)),
                "selected": "; ".join(str(row["row_id"]) for row in selected),
            }
        )
        for row in selected:
            row = dict(row)
            row["policy"] = policy.name
            selected_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "targeted_union_summary.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.out_dir / "targeted_selected_candidates.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "gt_mask": str(args.gt_mask),
                "current_mask": str(args.current_mask),
                "component_candidates": str(args.component_candidates),
                "he_image": str(args.he_image),
                "baseline": base_m,
                "target_components": sorted(target_components),
                "policies": [policy.__dict__ for policy in POLICIES],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
