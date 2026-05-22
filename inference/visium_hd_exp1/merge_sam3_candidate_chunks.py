#!/usr/bin/env python3
"""Merge chunked SAM3 candidate-proposal runs into one candidate_report.

The chunk jobs use the same image, checkpoint, prompt recipe, and ROI, but each
processes only a slice of the prompt list. This script loads their saved masks,
applies one global NMS, evaluates the merged pool against the annotation masks,
and writes the same artifact shape as run_sam3_candidate_proposals.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT_OVERRIDE", str(Path(__file__).resolve().parent.parent.parent)))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image

sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(PROJECT_ROOT / "inference" / "visium_hd_exp1"))

from run_sam3_candidate_proposals import (  # noqa: E402
    candidate_overlay,
    draw_label_panel,
    metrics_to_dict,
    save_candidate_contact_sheet,
)
from sam3_baseline_utils import (  # noqa: E402
    ensure_dirs,
    load_binary_mask,
    make_overlay,
    parse_label_filter,
    resolve_mask_path,
    save_mask,
    select_label_records,
    slugify,
)
from sam3_inference import resize_mask  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resize_rgb(image: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return np.array(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(np.uint8)
    return resize_mask(mask.astype(np.uint8), (h, w)).astype(np.uint8)


def read_candidate_metadata(path: Path) -> Dict[int, dict]:
    with path.open(newline="") as handle:
        return {int(row["candidate_id"]): row for row in csv.DictReader(handle)}


def bbox_intersection_area(a: Sequence[int], b: Sequence[int]) -> int:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def fast_nms_candidates(candidates: List[dict], iou_threshold: float) -> List[dict]:
    """Exact mask NMS with cheap area/bbox filters before pixel intersections."""
    if iou_threshold <= 0 or not candidates:
        return candidates
    ordered = sorted(candidates, key=lambda item: int(item["area"]), reverse=True)
    kept: List[dict] = []
    for idx, cand in enumerate(ordered, start=1):
        cand_area = int(cand["area"])
        cand_bbox = cand.get("bbox_xyxy") or [0, 0, 0, 0]
        keep = True
        for prev in kept:
            prev_area = int(prev["area"])
            small = min(cand_area, prev_area)
            large = max(cand_area, prev_area)
            if large <= 0 or small / large < iou_threshold:
                continue
            # If even the bounding boxes cannot contain enough overlapping
            # pixels for this IoU threshold, the masks cannot suppress.
            required_inter = iou_threshold * (cand_area + prev_area) / (1.0 + iou_threshold)
            if bbox_intersection_area(cand_bbox, prev.get("bbox_xyxy") or [0, 0, 0, 0]) < required_inter:
                continue
            inter = np.logical_and(cand["mask"], prev["mask"]).sum()
            union = cand_area + prev_area - inter
            if union > 0 and float(inter / union) >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(cand)
        if idx % 250 == 0 or idx == len(ordered):
            print(f"global NMS processed {idx}/{len(ordered)}; kept={len(kept)}", flush=True)
    return kept


def resolve_chunk_dirs(values: Sequence[Path]) -> List[Path]:
    dirs: List[Path] = []
    for value in values:
        if value.is_dir() and (value / "candidate_report.json").exists():
            dirs.append(value)
        elif value.is_dir():
            dirs.extend(sorted(path.parent for path in value.glob("**/candidate_report.json")))
        else:
            raise FileNotFoundError(value)
    unique: List[Path] = []
    seen = set()
    for path in dirs:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def candidate_from_chunk(chunk_dir: Path, report: dict, cid: int, row: dict, shape_hw: Tuple[int, int]) -> dict:
    mask_path = chunk_dir / "candidate_masks" / f"candidate_{cid:03d}.png"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    mask = resize_bool(mask, shape_hw).astype(np.uint8)
    bbox = [
        int(float(row["bbox_x1"])),
        int(float(row["bbox_y1"])),
        int(float(row["bbox_x2"])),
        int(float(row["bbox_y2"])),
    ]
    prompt_box = None
    if row.get("prompt_box_x1") not in ("", None):
        prompt_box = [
            int(float(row["prompt_box_x1"])),
            int(float(row["prompt_box_y1"])),
            int(float(row["prompt_box_x2"])),
            int(float(row["prompt_box_y2"])),
        ]
    point = None
    if row.get("point_x") not in ("", None):
        point = [int(float(row["point_x"])), int(float(row["point_y"]))]
    return {
        "prompt_type": row.get("prompt_type", report.get("proposal_mode", "")),
        "prompt_index": int(float(row.get("prompt_index") or -1)),
        "point_xy": point,
        "prompt_box_xyxy": prompt_box,
        "area": int(mask.sum()),
        "bbox_xyxy": bbox,
        "mask": mask,
        "chunk_dir": str(chunk_dir),
        "chunk_candidate_id": cid,
    }


def write_candidate_csv(candidates: Sequence[dict], path: Path) -> None:
    fieldnames = [
        "candidate_id",
        "prompt_index",
        "prompt_type",
        "point_x",
        "point_y",
        "prompt_box_x1",
        "prompt_box_y1",
        "prompt_box_x2",
        "prompt_box_y2",
        "area",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "chunk_dir",
        "chunk_candidate_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, cand in enumerate(candidates):
            bbox = cand.get("bbox_xyxy") or [None, None, None, None]
            point = cand.get("point_xy") or [None, None]
            prompt_box = cand.get("prompt_box_xyxy") or [None, None, None, None]
            writer.writerow(
                {
                    "candidate_id": idx,
                    "prompt_index": cand.get("prompt_index"),
                    "prompt_type": cand.get("prompt_type"),
                    "point_x": point[0],
                    "point_y": point[1],
                    "prompt_box_x1": prompt_box[0],
                    "prompt_box_y1": prompt_box[1],
                    "prompt_box_x2": prompt_box[2],
                    "prompt_box_y2": prompt_box[3],
                    "area": int(cand["area"]),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "chunk_dir": cand.get("chunk_dir"),
                    "chunk_candidate_id": cand.get("chunk_candidate_id"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, action="append", required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--nms-iou", type=float, default=0.97)
    parser.add_argument("--save-mask-limit", type=int, default=0, help="0 saves all merged candidate masks.")
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    args = parser.parse_args()

    chunk_dirs = resolve_chunk_dirs(args.chunk_dir)
    if not chunk_dirs:
        raise SystemExit("No chunk candidate_report.json files found.")

    first_report = load_json(chunk_dirs[0] / "candidate_report.json")
    shape_hw = tuple(int(v) for v in first_report["image_shape"])
    image = resize_rgb(np.array(Image.open(args.image_path).convert("RGB")), shape_hw)
    image_h, image_w = shape_hw
    image_area = image_h * image_w

    raw_candidates: List[dict] = []
    chunk_manifest = []
    for chunk_dir in chunk_dirs:
        report = load_json(chunk_dir / "candidate_report.json")
        if tuple(int(v) for v in report["image_shape"]) != shape_hw:
            raise SystemExit(f"Chunk image_shape mismatch: {chunk_dir}")
        metadata = read_candidate_metadata(chunk_dir / "candidate_metadata.csv")
        available_ids = sorted(
            int(path.stem.split("_")[-1]) for path in (chunk_dir / "candidate_masks").glob("candidate_*.png")
        )
        for cid in available_ids:
            raw_candidates.append(candidate_from_chunk(chunk_dir, report, cid, metadata[cid], shape_hw))
        chunk_manifest.append(
            {
                "chunk_dir": str(chunk_dir),
                "prompt_start": report.get("prompt_start"),
                "prompt_end": report.get("prompt_end"),
                "n_candidates_after_nms": report.get("n_candidates_after_nms"),
                "n_saved_masks": len(available_ids),
            }
        )

    candidates = fast_nms_candidates(raw_candidates, args.nms_iou)
    candidates = sorted(candidates, key=lambda item: int(item["area"]), reverse=True)

    masks_dir = args.output_dir / "candidate_masks"
    panels_dir = args.output_dir / "label_panels"
    ensure_dirs((args.output_dir, masks_dir, panels_dir))

    union_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    for cand in candidates:
        union_mask = np.logical_or(union_mask, cand["mask"]).astype(np.uint8)

    save_mask(union_mask, args.output_dir / "candidate_union_mask.png")
    Image.fromarray(make_overlay(image, union_mask, (0, 220, 255), alpha=0.45)).save(
        args.output_dir / "candidate_union_overlay.png"
    )
    Image.fromarray(candidate_overlay(image, candidates[: max(1, min(len(candidates), 320))])).save(
        args.output_dir / "candidate_colored_overlay.png"
    )
    save_candidate_contact_sheet(image, candidates, args.output_dir / "candidate_contact_sheet.png", args.contact_sheet_limit)

    save_limit = len(candidates) if args.save_mask_limit <= 0 else min(len(candidates), args.save_mask_limit)
    for idx, cand in enumerate(candidates[:save_limit]):
        save_mask(cand["mask"], masks_dir / f"candidate_{idx:03d}.png")
    write_candidate_csv(candidates, args.output_dir / "candidate_metadata.csv")

    summary = load_json(args.summary_path)
    records = select_label_records(summary["labels"], include_labels=parse_label_filter(args.labels))
    gt_masks = [
        resize_bool(load_binary_mask(resolve_mask_path(PROJECT_ROOT, args.summary_path, str(item["mask_path"]))), shape_hw)
        for _, item in records
    ]

    label_results = []
    for (_, item), gt_mask in zip(records, gt_masks):
        label = str(item["label"])
        slug = str(item.get("slug", slugify(label)))
        best_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        best_metrics = metrics_to_dict(best_mask, gt_mask)
        best_idx = None
        best_point = None
        for cand_idx, cand in enumerate(candidates):
            cand_metrics = metrics_to_dict(cand["mask"], gt_mask)
            if cand_metrics["dice"] > best_metrics["dice"]:
                best_metrics = cand_metrics
                best_mask = cand["mask"]
                best_idx = cand_idx
                best_point = cand.get("point_xy")
        union_metrics = metrics_to_dict(union_mask, gt_mask)
        label_results.append(
            {
                "label": label,
                "slug": slug,
                "target_pixels": int(gt_mask.sum()),
                "union_metrics": union_metrics,
                "best_candidate_metrics": best_metrics,
                "best_candidate_index": best_idx,
                "best_point_xy": list(best_point) if best_point is not None else None,
                "best_candidate_pixels": int(best_mask.sum()),
            }
        )
        draw_label_panel(
            image=image,
            label=label,
            gt_mask=gt_mask,
            union_mask=union_mask,
            best_mask=best_mask,
            best_point=tuple(best_point) if best_point is not None else None,
            union_metrics=union_metrics,
            best_metrics=best_metrics,
            output_path=panels_dir / f"{slug}.png",
        )

    report = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": first_report.get("checkpoint"),
        "device": first_report.get("device"),
        "max_side": first_report.get("max_side"),
        "resize_scale": first_report.get("resize_scale"),
        "image_shape": [int(image_h), int(image_w)],
        "proposal_mode": first_report.get("proposal_mode"),
        "text_prompt": first_report.get("text_prompt"),
        "prompt_start": min((row.get("prompt_start") or 0) for row in chunk_manifest),
        "prompt_end": max((row.get("prompt_end") or 0) for row in chunk_manifest),
        "n_total_prompts": first_report.get("n_total_prompts"),
        "n_processed_prompts": sum((row.get("prompt_end") or 0) - (row.get("prompt_start") or 0) for row in chunk_manifest),
        "grid_step": first_report.get("grid_step"),
        "border": first_report.get("border"),
        "n_points": first_report.get("n_points"),
        "box_size": first_report.get("box_size"),
        "box_stride": first_report.get("box_stride"),
        "n_boxes": first_report.get("n_boxes"),
        "n_empty": None,
        "n_rejected_small": None,
        "n_rejected_large": None,
        "n_raw_candidates": len(raw_candidates),
        "n_candidates_after_nms": len(candidates),
        "nms_iou": args.nms_iou,
        "min_area": first_report.get("min_area"),
        "max_area": first_report.get("max_area"),
        "union_pixels": int(union_mask.sum()),
        "union_fraction": float(union_mask.sum() / image_area),
        "merged_from_chunks": chunk_manifest,
        "labels": label_results,
    }
    (args.output_dir / "candidate_report.json").write_text(json.dumps(report, indent=2))
    print(f"Merged {len(chunk_dirs)} chunks")
    print(f"Raw candidates={len(raw_candidates)}; after global NMS={len(candidates)}")
    print(f"Saved candidate report: {args.output_dir / 'candidate_report.json'}")


if __name__ == "__main__":
    main()
