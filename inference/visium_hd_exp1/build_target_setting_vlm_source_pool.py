#!/usr/bin/env python3
"""Build a focused VLM source pool from one SAM setting.

This is used for class-specific ablations where the question is not "search
all settings", but "can a ranker recover the best candidates from one strong
candidate generator?"  The script writes the same public/hidden CSV contract
used by the paired crop renderers and local VLM runners.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


CLASS_INFO = {
    "bronchiola": {
        "label": "lung_bronchiola",
        "display": "bronchiola",
        "annotation": "01_lung_bronchiola_target_roi.png",
        "description": "bronchiolar airway tissue, airway-like lumen, epithelial lining",
    },
    "alveoli": {
        "label": "lung_alveoli_normal_adjacent",
        "display": "alveoli",
        "annotation": "04_lung_alveoli_normal_adjacent_target_roi.png",
        "description": "alveolar lung parenchyma, open air spaces, thin septa",
    },
    "vessels": {
        "label": "lung_vessels",
        "display": "vessels",
        "annotation": "05_lung_vessels_target_roi.png",
        "description": "blood vessel or vascular wall, lumen-like vascular structure",
    },
    "tumor": {
        "label": "tumor",
        "display": "tumor",
        "annotation": "08_tumor_target_roi.png",
        "description": "malignant epithelial tumor region",
    },
    "stroma": {
        "label": "stroma",
        "display": "stroma",
        "annotation": "07_stroma_target_roi.png",
        "description": "stromal or mesenchymal tissue, collagen, fibroblast, smooth-muscle-like tissue",
    },
    "immune_infiltration": {
        "label": "immune_infiltration",
        "display": "immune infiltration",
        "annotation": "03_immune_infiltration_target_roi.png",
        "description": "immune-cell-rich region, small round-cell aggregates",
    },
}


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    inter = int(np.logical_and(candidate, target).sum())
    union = int(np.logical_or(candidate, target).sum())
    cand = int(candidate.sum())
    targ = int(target.sum())
    precision = inter / cand if cand else 0.0
    recall = inter / targ if targ else 0.0
    dice = 2 * inter / (cand + targ) if (cand + targ) else 0.0
    iou = inter / union if union else 0.0
    return {"dice": dice, "precision": precision, "recall": recall, "iou": iou}


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def candidate_id(path: Path) -> int:
    match = re.search(r"candidate_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse candidate id from {path}")
    return int(match.group(1))


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(mask)
        return labels.astype(np.int32, copy=False), int(count)
    except Exception:
        try:
            import cv2  # type: ignore

            count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            return labels.astype(np.int32, copy=False), int(count - 1)
        except Exception as exc:
            raise RuntimeError("Need scipy.ndimage or cv2 for connected components") from exc


def component_metrics(candidate: np.ndarray, annotation: np.ndarray, min_area: int) -> dict[str, object]:
    comp_labels, count = label_components(annotation)
    best: dict[str, object] = {
        "component_best_dice": 0.0,
        "component_best_precision": 0.0,
        "component_best_recall": 0.0,
        "matched_annotation_component_id": "",
        "matched_annotation_component_pixels": "",
        "matched_annotation_component_bbox_xyxy": "",
        "annotation_component_count": count,
    }
    cand_pixels = int(candidate.sum())
    if cand_pixels == 0:
        return best
    for comp_id in range(1, count + 1):
        comp = comp_labels == comp_id
        comp_area = int(comp.sum())
        if comp_area < min_area:
            continue
        inter = int(np.logical_and(candidate, comp).sum())
        precision = inter / cand_pixels if cand_pixels else 0.0
        recall = inter / comp_area if comp_area else 0.0
        dice = 2 * inter / (cand_pixels + comp_area) if (cand_pixels + comp_area) else 0.0
        if dice > float(best["component_best_dice"]):
            best.update(
                {
                    "component_best_dice": dice,
                    "component_best_precision": precision,
                    "component_best_recall": recall,
                    "matched_annotation_component_id": comp_id,
                    "matched_annotation_component_pixels": comp_area,
                    "matched_annotation_component_bbox_xyxy": list(bbox(comp)),
                }
            )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-class", choices=sorted(CLASS_INFO), required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--source", default="he")
    parser.add_argument("--run", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--min-component-area", type=int, default=25)
    args = parser.parse_args()

    info = CLASS_INFO[args.target_class]
    he = Image.open(args.he_image).convert("RGB")
    ficture = Image.open(args.ficture_image).convert("RGB")
    if he.size != ficture.size:
        raise SystemExit(f"H&E/FICTURE size mismatch: {he.size} vs {ficture.size}")
    annotation = read_mask(args.annotation_dir / str(info["annotation"]), he.size)

    mask_paths = sorted(args.mask_dir.glob("candidate_*.png"), key=candidate_id)
    if not mask_paths:
        raise SystemExit(f"No candidate_*.png masks found under {args.mask_dir}")

    public_rows: list[dict[str, object]] = []
    hidden_rows: list[dict[str, object]] = []
    for path in mask_paths:
        cid = candidate_id(path)
        mask = read_mask(path, he.size)
        met = metrics(mask, annotation)
        comp_met = component_metrics(mask, annotation, args.min_component_area)
        uid = f"{info['display'].replace(' ', '_')}__{args.source}__{args.run}__{args.setting}__{cid:03d}"
        base = {
            "candidate_uid": uid,
            "label": info["label"],
            "display": info["display"],
            "sample_bucket": "SINGLE_SETTING",
            "source": args.source,
            "run": args.run,
            "setting": args.setting,
            "candidate_id": cid,
            "mask_path": str(path),
            "target_description": info["description"],
            "he_crop_rel": "",
            "ficture_crop_rel": "",
            "prompt_text": "",
            "image_cue": "",
        }
        public_rows.append(base)
        hidden = {
            **base,
            "dice": met["dice"],
            "precision": met["precision"],
            "recall": met["recall"],
            "full_dice": met["dice"],
            "full_precision": met["precision"],
            "full_recall": met["recall"],
            "full_iou": met["iou"],
            "hidden_dice": met["dice"],
            "hidden_precision": met["precision"],
            "hidden_recall": met["recall"],
            "hidden_iou": met["iou"],
            "candidate_pixels": int(mask.sum()),
            "candidate_bbox_xyxy": list(bbox(mask)),
            **comp_met,
        }
        hidden_rows.append(hidden)

    public_fields = [
        "candidate_uid",
        "label",
        "display",
        "sample_bucket",
        "source",
        "run",
        "setting",
        "candidate_id",
        "mask_path",
        "target_description",
        "he_crop_rel",
        "ficture_crop_rel",
        "prompt_text",
        "image_cue",
    ]
    hidden_fields = public_fields + [
        "dice",
        "precision",
        "recall",
        "full_dice",
        "full_precision",
        "full_recall",
        "full_iou",
        "hidden_dice",
        "hidden_precision",
        "hidden_recall",
        "hidden_iou",
        "component_best_dice",
        "component_best_precision",
        "component_best_recall",
        "matched_annotation_component_id",
        "matched_annotation_component_pixels",
        "matched_annotation_component_bbox_xyxy",
        "annotation_component_count",
        "candidate_pixels",
        "candidate_bbox_xyxy",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "public_vlm_requests.csv", public_rows, public_fields)
    write_csv(args.output_dir / "hidden_candidate_truth.csv", hidden_rows, hidden_fields)

    best_full = max(hidden_rows, key=lambda row: float(row["full_dice"]))
    best_comp = max(hidden_rows, key=lambda row: float(row["component_best_dice"]))
    write_csv(
        args.output_dir / "pool_summary.csv",
        [
            {
                "target_class": args.target_class,
                "source": args.source,
                "run": args.run,
                "setting": args.setting,
                "candidate_count": len(hidden_rows),
                "best_full_candidate_id": best_full["candidate_id"],
                "best_full_dice": best_full["full_dice"],
                "best_full_precision": best_full["full_precision"],
                "best_full_recall": best_full["full_recall"],
                "best_component_candidate_id": best_comp["candidate_id"],
                "best_component_dice": best_comp["component_best_dice"],
                "best_component_precision": best_comp["component_best_precision"],
                "best_component_recall": best_comp["component_best_recall"],
            }
        ],
        [
            "target_class",
            "source",
            "run",
            "setting",
            "candidate_count",
            "best_full_candidate_id",
            "best_full_dice",
            "best_full_precision",
            "best_full_recall",
            "best_component_candidate_id",
            "best_component_dice",
            "best_component_precision",
            "best_component_recall",
        ],
    )
    print(f"Wrote {len(hidden_rows)} candidates to {args.output_dir}")
    print(
        "Best full:",
        best_full["candidate_id"],
        best_full["full_dice"],
        best_full["full_precision"],
        best_full["full_recall"],
    )
    print(
        "Best component:",
        best_comp["candidate_id"],
        best_comp["component_best_dice"],
        best_comp["component_best_precision"],
        best_comp["component_best_recall"],
    )


if __name__ == "__main__":
    main()
