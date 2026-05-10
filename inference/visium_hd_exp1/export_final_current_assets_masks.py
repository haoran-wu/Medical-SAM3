#!/usr/bin/env python3
"""Export the current-assets Dice>0.8 masks at original H&E resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "dice08_experiments" / "final_fullslide_1536_multiscale_et260"
HE_PATH = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "final_deliverables" / "current_assets_dice08_masks"
TARGET_DIRS = [
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks",
]

LABEL_ORDER = [
    "lung_bronchiola",
    "erythorocytes",
    "immune_infiltration",
    "lung_alveoli_normal_adjacent",
    "lung_vessels",
    "pigment",
    "stroma",
    "tumor",
]

LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "lung_bronchiola": (31, 119, 180),
    "erythorocytes": (255, 127, 14),
    "immune_infiltration": (44, 160, 44),
    "lung_alveoli_normal_adjacent": (148, 103, 189),
    "lung_vessels": (140, 86, 75),
    "pigment": (127, 127, 127),
    "stroma": (188, 189, 34),
    "tumor": (23, 190, 207),
}


def load_binary(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)
    ) > 127


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    return {
        "dice": 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0,
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_sum if pred_sum else 0.0,
        "recall": inter / gt_sum if gt_sum else 0.0,
        "pred_pixels": float(pred_sum),
        "gt_pixels": float(gt_sum),
    }


def load_targets(shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    for target_dir in TARGET_DIRS:
        for path in target_dir.glob("*_target.png"):
            label = path.name.split("_", 1)[1].removesuffix("_target.png")
            mask = load_binary(path)
            if mask.shape != shape_hw:
                mask = resize_mask(mask, shape_hw)
            targets[label] = mask
    return targets


def overlay(image: np.ndarray, masks: Dict[str, np.ndarray], alpha: float = 0.42) -> np.ndarray:
    out = image.astype(np.float32).copy()
    for label in LABEL_ORDER:
        mask = masks.get(label)
        if mask is None:
            continue
        color = np.array(LABEL_COLORS[label], dtype=np.float32)
        idx = mask.astype(bool)
        out[idx] = (1.0 - alpha) * out[idx] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def make_preview(image: np.ndarray, masks: Dict[str, np.ndarray], path: Path) -> None:
    short_h = 1800
    scale = short_h / image.shape[0]
    preview_shape = (short_h, int(round(image.shape[1] * scale)))
    image_small = np.array(Image.fromarray(image).resize((preview_shape[1], preview_shape[0]), Image.Resampling.BILINEAR))
    small_masks = {label: resize_mask(mask, preview_shape) for label, mask in masks.items()}
    out = overlay(image_small, small_masks, alpha=0.44)
    canvas = Image.fromarray(out)
    draw = ImageDraw.Draw(canvas)
    x0, y0 = 18, 18
    for label in LABEL_ORDER:
        if label not in small_masks:
            continue
        color = LABEL_COLORS[label]
        draw.rectangle([x0, y0, x0 + 18, y0 + 18], fill=color)
        draw.text((x0 + 26, y0 - 1), label, fill=(0, 0, 0))
        y0 += 28
    canvas.save(path)


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_dir = OUTPUT_DIR / "masks_original_resolution"
    nonoverlap_mask_dir = OUTPUT_DIR / "masks_original_resolution_nonoverlap"
    overlay_dir = OUTPUT_DIR / "overlays_original_resolution"
    mask_dir.mkdir(parents=True, exist_ok=True)
    nonoverlap_mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    he = np.array(Image.open(HE_PATH).convert("RGB"))
    original_shape = he.shape[:2]
    targets = load_targets(original_shape)

    masks: Dict[str, np.ndarray] = {}
    rows = []
    for label in LABEL_ORDER:
        source = SOURCE_DIR / "masks" / f"{label}_prediction.png"
        if not source.exists():
            continue
        mask = load_binary(source)
        mask = resize_mask(mask, original_shape)
        masks[label] = mask
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{label}_prediction_original_resolution.png")
        Image.fromarray(overlay(he, {label: mask}, alpha=0.45)).save(
            overlay_dir / f"{label}_overlay_original_resolution.png"
        )
        row = {"label": label}
        if label in targets:
            row.update(metrics(mask, targets[label]))
        rows.append(row)

    all_overlay = overlay(he, masks, alpha=0.42)
    Image.fromarray(all_overlay).save(OUTPUT_DIR / "all_labels_overlay_original_resolution.png")
    make_preview(he, masks, OUTPUT_DIR / "all_labels_overlay_preview_with_legend.png")

    label_index = np.zeros(original_shape, dtype=np.uint8)
    for idx, label in enumerate(LABEL_ORDER, start=1):
        if label in masks:
            label_index[masks[label]] = idx
    Image.fromarray(label_index).save(OUTPUT_DIR / "multiclass_label_index_original_resolution.png")

    nonoverlap_masks: Dict[str, np.ndarray] = {}
    nonoverlap_rows = []
    for idx, label in enumerate(LABEL_ORDER, start=1):
        if label not in masks:
            continue
        mask = label_index == idx
        nonoverlap_masks[label] = mask
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            nonoverlap_mask_dir / f"{label}_prediction_original_resolution_nonoverlap.png"
        )
        row = {"label": label}
        if label in targets:
            row.update(metrics(mask, targets[label]))
        nonoverlap_rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "original_resolution_metrics.csv", index=False)
    nonoverlap_metrics_df = pd.DataFrame(nonoverlap_rows)
    nonoverlap_metrics_df.to_csv(OUTPUT_DIR / "original_resolution_nonoverlap_metrics.csv", index=False)
    summary = {
        "source_dir": str(SOURCE_DIR),
        "he_path": str(HE_PATH),
        "output_dir": str(OUTPUT_DIR),
        "original_shape_hw": list(original_shape),
        "labels": list(masks.keys()),
        "primary_mask_dir": str(nonoverlap_mask_dir),
        "notes": "Masks are trained from current available H&E + FICTURE hard factors + existing GeoJSON targets, then exported at original H&E resolution.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Current-assets Dice 0.8 Mask Deliverable", ""]
    lines.append(f"Original H&E shape: `{original_shape[0]} x {original_shape[1]}` pixels.")
    lines.append("")
    lines.append("## Original-resolution metrics")
    lines.append("")
    lines.append("Primary masks are non-overlapping masks exported from `multiclass_label_index_original_resolution.png`.")
    lines.append("")
    lines.append("| label | Dice | IoU | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in nonoverlap_metrics_df.sort_values("dice", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dice']:.3f} | {row['iou']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    lines += [
        "",
        "## Important Scope",
        "",
        "This is the best current deliverable using only files already present locally: H&E, FICTURE hard factor image/metadata, and existing GeoJSON-derived target masks.",
        "It is suitable as an Exp1 slide reconstruction/mask package. It is not claimed as a validated cross-slide model.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")
    print(nonoverlap_metrics_df.sort_values("dice", ascending=False).to_string(index=False))
    print(f"Wrote deliverable to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
