#!/usr/bin/env python3
"""Run TMA39 box+point prompts while preserving every eligible SAM mask.

Candidate selection is annotation-free.  For every prompt, the runner keeps all
unique masks that contain the positive point and pass the established tissue and
area gates.  It then writes three frozen selections for posthoc comparison:

1. point_constrained_sam_score: highest SAM score after enforcing the point.
2. genemap_support_f1: closest spatial agreement with the prompt's GeneMap region.
3. support_recall_then_sam: highest SAM score among masks close to the best
   GeneMap-region recall for that prompt.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi


PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT_OVERRIDE", str(Path(__file__).resolve().parents[1]))
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from inference.sam3_inference import SAM3Model, normalize_bbox, normalize_points
from sam3.model.box_ops import box_xywh_to_cxcywh


SELECTION_METHODS = (
    "point_constrained_sam_score",
    "genemap_support_f1",
    "support_recall_then_sam",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--foreground-mask-path", type=Path, required=True)
    parser.add_argument("--prompts-csv", type=Path, required=True)
    parser.add_argument("--region-labels-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", choices=("ficture", "he"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-area-frac", type=float, default=0.00002)
    parser.add_argument("--max-area-frac", type=float, default=0.20)
    parser.add_argument("--expected-prompts", type=int, default=44)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_binary(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {shape}")
    return mask


def pixel_edges(length: int, bins: int) -> np.ndarray:
    return np.rint(np.linspace(0, length, bins + 1)).astype(np.int64)


def support_mask_from_labels(
    labels: np.ndarray,
    module_number: int,
    region_number: int,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    module_labels = labels[module_number - 1]
    region_grid = module_labels == region_number
    if not region_grid.any():
        raise RuntimeError(
            f"GeneMap support missing for module {module_number}, region {region_number}"
        )
    grid_h, grid_w = region_grid.shape
    x_edges = pixel_edges(width, grid_w)
    y_edges = pixel_edges(height, grid_h)
    support = np.zeros(shape, dtype=bool)
    for grid_y, grid_x in np.argwhere(region_grid):
        support[
            y_edges[grid_y] : y_edges[grid_y + 1],
            x_edges[grid_x] : x_edges[grid_x + 1],
        ] = True
    return support


def threshold_stability(logits: np.ndarray, offset: float) -> float:
    low = logits > -offset
    high = logits > offset
    denominator = int(low.sum())
    return float(high.sum() / denominator) if denominator else 0.0


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def candidate_features(
    mask: np.ndarray,
    support: np.ndarray,
    prompt_box: tuple[int, int, int, int],
) -> dict[str, Any]:
    area = int(mask.sum())
    support_area = int(support.sum())
    shared = int(np.logical_and(mask, support).sum())
    x1, y1, x2, y2 = prompt_box
    inside_box = int(mask[y1:y2, x1:x2].sum())
    box_area = max(1, (x2 - x1) * (y2 - y1))
    bbox = tight_bbox(mask)
    crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    components, component_count = ndi.label(crop)
    sizes = np.bincount(components.ravel())[1:]
    largest_fraction = float(sizes.max() / area) if sizes.size else 0.0
    return {
        "area_pixels": area,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "support_area_pixels": support_area,
        "support_shared_pixels": shared,
        "support_precision": shared / max(1, area),
        "support_recall": shared / max(1, support_area),
        "support_f1": 2 * shared / max(1, area + support_area),
        "mask_fraction_inside_prompt_box": inside_box / max(1, area),
        "prompt_box_fraction_covered": inside_box / box_area,
        "component_count": int(component_count),
        "largest_component_fraction": largest_fraction,
    }


def add_box_point_prompt(
    sam3: SAM3Model,
    inference_state: dict,
    box: tuple[int, int, int, int],
    point: tuple[int, int],
    image_shape: tuple[int, int],
) -> dict:
    sam3.processor.reset_all_prompts(inference_state)
    if "language_features" not in inference_state["backbone_out"]:
        text_outputs = sam3.processor.model.backbone.forward_text(
            ["visual"], device=sam3.device
        )
        inference_state["backbone_out"].update(text_outputs)
    if "geometric_prompt" not in inference_state:
        inference_state["geometric_prompt"] = sam3.processor.model._get_dummy_prompt()

    image_h, image_w = image_shape
    x1, y1, x2, y2 = box
    box_xywh = torch.tensor(
        [x1, y1, x2 - x1, y2 - y1], dtype=torch.float32
    ).view(1, 4)
    normalized_box = normalize_bbox(
        box_xywh_to_cxcywh(box_xywh), image_w, image_h
    ).to(device=sam3.device, dtype=torch.float32).view(1, 1, 4)
    inference_state["geometric_prompt"].append_boxes(
        normalized_box,
        torch.ones((1, 1), device=sam3.device, dtype=torch.bool),
    )
    normalized_point = normalize_points([point], image_w, image_h)
    inference_state["geometric_prompt"].append_points(
        torch.tensor(
            normalized_point, device=sam3.device, dtype=torch.float32
        ).view(1, 1, 2),
        torch.ones((1, 1), device=sam3.device, dtype=torch.long),
    )
    return sam3.processor._forward_grounding(inference_state)


def save_packed_prompt_masks(
    path: Path,
    masks: list[np.ndarray],
    rows: list[dict[str, Any]],
) -> None:
    bboxes: list[tuple[int, int, int, int]] = []
    shapes: list[tuple[int, int]] = []
    offsets = [0]
    packed_parts: list[np.ndarray] = []
    for mask, row in zip(masks, rows):
        bbox = (
            int(row["bbox_x1"]),
            int(row["bbox_y1"]),
            int(row["bbox_x2"]),
            int(row["bbox_y2"]),
        )
        crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        packed = np.packbits(crop.reshape(-1), bitorder="little")
        bboxes.append(bbox)
        shapes.append(crop.shape)
        packed_parts.append(packed)
        offsets.append(offsets[-1] + len(packed))
    concatenated = (
        np.concatenate(packed_parts) if packed_parts else np.empty(0, dtype=np.uint8)
    )
    np.savez_compressed(
        path,
        bboxes=np.asarray(bboxes, dtype=np.int32),
        shapes=np.asarray(shapes, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        packed=concatenated,
        image_shape=np.asarray(masks[0].shape if masks else (0, 0), dtype=np.int32),
    )


def select_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    point_score = max(
        rows,
        key=lambda row: (
            float(row["sam_score"]),
            float(row["threshold_stability_offset0p5"]),
            float(row["support_f1"]),
        ),
    )
    support_f1 = max(
        rows,
        key=lambda row: (
            float(row["support_f1"]),
            float(row["largest_component_fraction"]),
            float(row["threshold_stability_offset0p5"]),
            float(row["sam_score"]),
        ),
    )
    best_recall = max(float(row["support_recall"]) for row in rows)
    recall_floor = 0.90 * best_recall
    recall_eligible = [
        row for row in rows if float(row["support_recall"]) >= recall_floor
    ]
    recall_then_score = max(
        recall_eligible,
        key=lambda row: (
            float(row["sam_score"]),
            float(row["support_f1"]),
            float(row["threshold_stability_offset0p5"]),
        ),
    )
    return {
        "point_constrained_sam_score": point_score,
        "genemap_support_f1": support_f1,
        "support_recall_then_sam": recall_then_score,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    packed_dir = args.output_dir / "all_eligible_internal_masks_packed"
    packed_dir.mkdir()
    selected_dirs = {}
    for method in SELECTION_METHODS:
        selected_dirs[method] = args.output_dir / f"selected_{method}"
        selected_dirs[method].mkdir()

    image = load_rgb(args.image_path)
    image_shape = image.shape[:2]
    foreground = load_binary(args.foreground_mask_path, image_shape)
    labels = np.load(args.region_labels_npz)["labels"].astype(np.int16)
    prompts = read_csv(args.prompts_csv)
    if len(prompts) != args.expected_prompts:
        raise RuntimeError(
            f"Expected {args.expected_prompts} prompts, found {len(prompts)}"
        )
    min_area = int(round(args.min_area_frac * image_shape[0] * image_shape[1]))
    max_area = int(round(args.max_area_frac * image_shape[0] * image_shape[1]))

    sam3 = SAM3Model(
        confidence_threshold=0.1,
        checkpoint_path=str(args.checkpoint),
        device=args.device,
    )
    inference_state = sam3.encode_image(image)
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prompt_audit: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        prompt_id = prompt["prompt_id"]
        module_number = int(prompt["gene_module"])
        region_number = int(prompt_id.rsplit("_r", 1)[1])
        box = tuple(
            int(prompt[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2")
        )
        point = (int(prompt["point_x"]), int(prompt["point_y"]))
        support = support_mask_from_labels(
            labels, module_number, region_number, image_shape
        )
        state = add_box_point_prompt(sam3, inference_state, box, point, image_shape)
        raw_masks = state.get("masks")
        raw_scores = state.get("scores")
        model_count = int(len(raw_masks)) if raw_masks is not None else 0
        eligible_masks: list[np.ndarray] = []
        eligible_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        rejection_counts = {
            "not_point_consistent": 0,
            "empty_after_foreground_clip": 0,
            "area_filter": 0,
            "duplicate": 0,
        }

        if raw_masks is not None and raw_scores is not None:
            px, py = point
            for model_index in range(len(raw_masks)):
                logits_tensor = raw_masks[model_index]
                while logits_tensor.ndim > 2 and logits_tensor.shape[0] == 1:
                    logits_tensor = logits_tensor[0]
                if logits_tensor.ndim != 2:
                    raise ValueError(
                        f"Unexpected SAM mask shape: {tuple(logits_tensor.shape)}"
                    )
                logits = logits_tensor.detach().float().cpu().numpy()
                raw_mask = logits > 0
                if not raw_mask[py, px]:
                    rejection_counts["not_point_consistent"] += 1
                    continue
                mask = np.logical_and(raw_mask, foreground)
                if not mask[py, px] or not mask.any():
                    rejection_counts["empty_after_foreground_clip"] += 1
                    continue
                area = int(mask.sum())
                if area < min_area or area > max_area:
                    rejection_counts["area_filter"] += 1
                    continue
                digest = hashlib.sha1(
                    np.packbits(mask, axis=None, bitorder="little").tobytes()
                ).hexdigest()
                if digest in seen:
                    rejection_counts["duplicate"] += 1
                    continue
                seen.add(digest)
                score = float(
                    raw_scores[model_index].detach().float().cpu().reshape(-1)[0]
                )
                row = {
                    "source": args.source,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "gene_module": module_number,
                    "region_number": region_number,
                    "model_index": int(model_index),
                    "sam_score": score,
                    "threshold_stability_offset0p5": threshold_stability(logits, 0.5),
                    "threshold_stability_offset1p0": threshold_stability(logits, 1.0),
                    "selection_used_annotation": False,
                }
                row.update(candidate_features(mask, support, box))
                row["eligible_index"] = len(eligible_rows)
                eligible_masks.append(mask)
                eligible_rows.append(row)

        if not eligible_rows:
            prompt_audit.append(
                {
                    "source": args.source,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "gene_module": module_number,
                    "model_mask_count": model_count,
                    "eligible_unique_mask_count": 0,
                    **rejection_counts,
                    "no_selected_mask_reason": (
                        "no mask survived the fixed positive-point, foreground, area, and duplicate checks"
                    ),
                    "selection_used_annotation": False,
                }
            )
            print(
                f"[{prompt_index + 1:02d}/{len(prompts)}] {args.source} {prompt_id}: "
                f"model={model_count} eligible_unique=0; recorded without a selected candidate",
                flush=True,
            )
            del state, raw_masks, raw_scores, eligible_masks, support
            gc.collect()
            continue
        save_packed_prompt_masks(
            packed_dir / f"{prompt_id}.npz", eligible_masks, eligible_rows
        )
        selected = select_rows(eligible_rows)
        for method, chosen in selected.items():
            eligible_index = int(chosen["eligible_index"])
            output_name = f"candidate_{prompt_index:04d}_{prompt_id}.png"
            Image.fromarray(np.uint8(eligible_masks[eligible_index]) * 255).save(
                selected_dirs[method] / output_name,
                optimize=True,
            )
            selected_rows.append(
                {
                    "selection_method": method,
                    "selected_mask_path": str(
                        Path(f"selected_{method}") / output_name
                    ),
                    **chosen,
                }
            )
        all_rows.extend(eligible_rows)
        prompt_audit.append(
            {
                "source": args.source,
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "gene_module": module_number,
                "model_mask_count": model_count,
                "eligible_unique_mask_count": len(eligible_rows),
                **rejection_counts,
                "selection_used_annotation": False,
            }
        )
        print(
            f"[{prompt_index + 1:02d}/{len(prompts)}] {args.source} {prompt_id}: "
            f"model={model_count} eligible_unique={len(eligible_rows)}",
            flush=True,
        )
        del state, raw_masks, raw_scores, eligible_masks, support
        gc.collect()

    write_csv(args.output_dir / "all_internal_mask_features.csv", all_rows)
    write_csv(args.output_dir / "selected_candidates.csv", selected_rows)
    write_csv(args.output_dir / "prompt_audit.csv", prompt_audit)
    manifest = {
        "source": args.source,
        "image_path": str(args.image_path),
        "image_shape": list(image_shape),
        "checkpoint": str(args.checkpoint),
        "prompts_csv": str(args.prompts_csv),
        "region_labels_npz": str(args.region_labels_npz),
        "foreground_mask_path": str(args.foreground_mask_path),
        "prompt_count": len(prompts),
        "image_encoding_count": 1,
        "inference_calls_per_prompt": 1,
        "total_inference_calls": len(prompts),
        "inference_entrypoint": "sam3.processor._forward_grounding",
        "total_model_masks": sum(row["model_mask_count"] for row in prompt_audit),
        "total_eligible_unique_masks": len(all_rows),
        "selected_prompt_count": len(
            {
                int(row["prompt_index"])
                for row in selected_rows
                if row["selection_method"] == "genemap_support_f1"
            }
        ),
        "prompts_without_eligible_mask": sum(
            int(row["eligible_unique_mask_count"]) == 0 for row in prompt_audit
        ),
        "selection_methods": list(SELECTION_METHODS),
        "selection_used_annotation": False,
        "positive_point_required": True,
        "foreground_clipping": True,
        "min_area_frac": args.min_area_frac,
        "max_area_frac": args.max_area_frac,
        "packed_mask_format": {
            "one_npz_per_prompt": True,
            "arrays": ["bboxes", "shapes", "offsets", "packed", "image_shape"],
            "bitorder": "little",
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
