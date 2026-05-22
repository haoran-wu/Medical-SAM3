#!/usr/bin/env python3
"""Finalize a merged SAM3 candidate directory from saved masks/metadata.

This is a fast recovery path for chunked candidate pools. It assumes
candidate_masks/, candidate_metadata.csv, and candidate_union_mask.png already
exist, then computes label metrics with bbox filtering and writes
candidate_report.json without rerunning SAM or global NMS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT_OVERRIDE", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(PROJECT_ROOT / "inference" / "visium_hd_exp1"))

from sam3_baseline_utils import load_binary_mask, parse_label_filter, resolve_mask_path, select_label_records, slugify  # noqa: E402
from sam3_inference import resize_mask  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    return resize_mask(mask.astype(np.uint8), (h, w)).astype(bool)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_intersection(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, int, int]:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def metrics_from_counts(inter: int, pred_sum: int, gt_sum: int) -> Dict[str, float]:
    union = pred_sum + gt_sum - inter
    return {
        "dice": 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0,
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_sum if pred_sum else 0.0,
        "recall": inter / gt_sum if gt_sum else 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }


def mask_metrics(mask: np.ndarray, gt: np.ndarray, pred_sum: int | None = None) -> Dict[str, float]:
    if pred_sum is None:
        pred_sum = int(mask.sum())
    gt_sum = int(gt.sum())
    inter = int(np.logical_and(mask, gt).sum())
    return metrics_from_counts(inter, pred_sum, gt_sum)


def read_metadata(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def chunk_manifest(chunk_root: Path | None) -> List[dict]:
    if chunk_root is None or not chunk_root.exists():
        return []
    rows = []
    for report_path in sorted(chunk_root.glob("**/candidate_report.json")):
        report = load_json(report_path)
        rows.append(
            {
                "chunk_dir": str(report_path.parent),
                "prompt_start": report.get("prompt_start"),
                "prompt_end": report.get("prompt_end"),
                "n_candidates_after_nms": report.get("n_candidates_after_nms"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--chunk-root", type=Path, default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--checkpoint", default="/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/checkpoints/Medical-SAM3/checkpoint.pt")
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--resize-scale", type=float, default=0.61557)
    parser.add_argument("--grid-step", type=int, default=24)
    parser.add_argument("--border", type=int, default=12)
    parser.add_argument("--n-total-prompts", type=int, default=3600)
    parser.add_argument("--nms-iou", type=float, default=0.97)
    args = parser.parse_args()

    metadata_path = args.output_dir / "candidate_metadata.csv"
    masks_dir = args.output_dir / "candidate_masks"
    union_path = args.output_dir / "candidate_union_mask.png"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not union_path.exists():
        raise FileNotFoundError(union_path)

    rows = read_metadata(metadata_path)
    union_mask = load_mask(union_path)
    shape_hw = union_mask.shape
    image_h, image_w = shape_hw
    image_area = image_h * image_w
    union_pixels = int(union_mask.sum())

    summary = load_json(args.summary_path)
    records = select_label_records(summary["labels"], include_labels=parse_label_filter(args.labels))
    gt_items = []
    for _, item in records:
        label = str(item["label"])
        slug = str(item.get("slug", slugify(label)))
        gt = resize_bool(load_binary_mask(resolve_mask_path(PROJECT_ROOT, args.summary_path, str(item["mask_path"]))), shape_hw)
        gt_items.append(
            {
                "label": label,
                "slug": slug,
                "mask": gt,
                "bbox": bbox_from_mask(gt),
                "area": int(gt.sum()),
                "best_metrics": metrics_from_counts(0, 0, int(gt.sum())),
                "best_candidate_index": None,
                "best_point_xy": None,
                "best_candidate_pixels": 0,
                "union_metrics": mask_metrics(union_mask, gt, union_pixels),
            }
        )

    for idx, row in enumerate(rows):
        mask_path = masks_dir / f"candidate_{int(row['candidate_id']):03d}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        pred_area = int(float(row.get("area") or 0))
        cand_bbox = [
            int(float(row["bbox_x1"])),
            int(float(row["bbox_y1"])),
            int(float(row["bbox_x2"])),
            int(float(row["bbox_y2"])),
        ]
        mask = None
        for gt_item in gt_items:
            gt_area = gt_item["area"]
            current_best = float(gt_item["best_metrics"]["dice"])
            max_inter = min(pred_area, gt_area)
            if metrics_from_counts(max_inter, pred_area, gt_area)["dice"] <= current_best:
                continue
            x1, y1, x2, y2 = bbox_intersection(cand_bbox, gt_item["bbox"])
            if x2 <= x1 or y2 <= y1:
                continue
            bbox_inter_area = (x2 - x1) * (y2 - y1)
            if metrics_from_counts(min(bbox_inter_area, max_inter), pred_area, gt_area)["dice"] <= current_best:
                continue
            if mask is None:
                mask = load_mask(mask_path)
                if mask.shape != shape_hw:
                    mask = resize_bool(mask, shape_hw)
            inter = int(np.logical_and(mask[y1:y2, x1:x2], gt_item["mask"][y1:y2, x1:x2]).sum())
            cand_metrics = metrics_from_counts(inter, pred_area, gt_area)
            if cand_metrics["dice"] > current_best:
                gt_item["best_metrics"] = cand_metrics
                gt_item["best_candidate_index"] = int(row["candidate_id"])
                point_x = row.get("point_x")
                point_y = row.get("point_y")
                gt_item["best_point_xy"] = [
                    int(float(point_x)),
                    int(float(point_y)),
                ] if point_x not in ("", None) and point_y not in ("", None) else None
                gt_item["best_candidate_pixels"] = pred_area
        if (idx + 1) % 250 == 0 or idx + 1 == len(rows):
            print(f"metric scan processed {idx + 1}/{len(rows)}", flush=True)

    labels = []
    for gt_item in gt_items:
        labels.append(
            {
                "label": gt_item["label"],
                "slug": gt_item["slug"],
                "target_pixels": gt_item["area"],
                "union_metrics": gt_item["union_metrics"],
                "best_candidate_metrics": gt_item["best_metrics"],
                "best_candidate_index": gt_item["best_candidate_index"],
                "best_point_xy": gt_item["best_point_xy"],
                "best_candidate_pixels": gt_item["best_candidate_pixels"],
            }
        )

    chunks = chunk_manifest(args.chunk_root)
    report = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": args.checkpoint,
        "device": "cuda",
        "max_side": args.max_side,
        "resize_scale": args.resize_scale,
        "image_shape": [int(image_h), int(image_w)],
        "proposal_mode": "point",
        "text_prompt": None,
        "prompt_start": 0,
        "prompt_end": args.n_total_prompts,
        "n_total_prompts": args.n_total_prompts,
        "n_processed_prompts": args.n_total_prompts,
        "skip_label_eval": False,
        "grid_step": args.grid_step,
        "border": args.border,
        "n_points": args.n_total_prompts,
        "box_size": 192,
        "box_stride": 96,
        "n_boxes": 0,
        "n_empty": None,
        "n_rejected_small": None,
        "n_rejected_large": None,
        "n_raw_candidates": sum(int(row.get("n_candidates_after_nms") or 0) for row in chunks) if chunks else len(rows),
        "n_candidates_after_nms": len(rows),
        "nms_iou": args.nms_iou,
        "min_area": None,
        "max_area": None,
        "union_pixels": union_pixels,
        "union_fraction": float(union_pixels / image_area),
        "merged_from_chunks": chunks,
        "labels": labels,
    }
    out = args.output_dir / "candidate_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved candidate report: {out}", flush=True)
    for item in labels:
        m = item["best_candidate_metrics"]
        print(
            f"{item['label']}: Dice={m['dice']:.3f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} "
            f"best_idx={item['best_candidate_index']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
