#!/usr/bin/env python3
"""Refine molecular prior components with SAM3 box prompts.

This is the first heavy stage after `run_molecular_prior_experiments.py`.
It takes FICTURE-derived H&E components, selects candidates by marker evidence,
uses their bounding boxes as SAM3 prompts, and evaluates the resulting masks
against the available GeoJSON-derived target masks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam3_baseline_utils import load_binary_mask, make_overlay, metrics_to_dict, normalize_prediction, resize_image_and_masks, save_mask, slugify
from sam3_inference import SAM3Model


DEFAULT_PRIOR_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80"
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_TARGET_MASK_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_component_sam3_refine" / "base_sam3_dx-60_dy80"


def load_targets(mask_dir: Path) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    for path in sorted(mask_dir.glob("*_target.png")):
        label = path.name.split("_", 1)[1].removesuffix("_target.png")
        targets[label] = load_binary_mask(path)
    return targets


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    margin_fraction: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h, w = image_shape
    dx = int(round((x1 - x0) * margin_fraction))
    dy = int(round((y1 - y0) * margin_fraction))
    return max(0, x0 - dx), max(0, y0 - dy), min(w - 1, x1 + dx), min(h - 1, y1 + dy)


def scale_bbox(
    bbox: Tuple[int, int, int, int],
    scale: float,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    out = (
        int(round(x0 * scale)),
        int(round(y0 * scale)),
        int(round(x1 * scale)),
        int(round(y1 * scale)),
    )
    return max(0, out[0]), max(0, out[1]), min(w - 1, out[2]), min(h - 1, out[3])


def load_marker_scores(annotation_path: Path) -> Dict[int, Dict[str, float]]:
    annotations = pd.read_csv(annotation_path)
    scores: Dict[int, Dict[str, float]] = {}
    for _, row in annotations.iterrows():
        scores[int(row["factor"])] = json.loads(str(row["label_scores_json"]))
    return scores


def select_components(
    components: pd.DataFrame,
    marker_scores: Dict[int, Dict[str, float]],
    labels: List[str],
    top_factors_per_label: int,
    top_components_per_factor: int,
) -> pd.DataFrame:
    rows = []
    for label in labels:
        factor_rank = sorted(
            ((factor, scores.get(label, 0.0)) for factor, scores in marker_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        keep_factors = {factor for factor, score in factor_rank[:top_factors_per_label] if score > 0}
        if not keep_factors:
            keep_factors = {factor_rank[0][0]}
        for factor in keep_factors:
            subset = components[components["factor"].astype(int) == int(factor)].copy()
            subset["target_label"] = label
            subset["marker_score_for_label"] = marker_scores.get(int(factor), {}).get(label, 0.0)
            subset = subset.sort_values("area", ascending=False).head(top_components_per_factor)
            rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_panel(
    image: np.ndarray,
    label: str,
    component_mask: np.ndarray,
    sam3_mask: np.ndarray,
    target: np.ndarray,
    bbox: Tuple[int, int, int, int],
    component_metrics: Dict[str, float],
    sam3_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        ("Molecular prior", component_mask, (0, 220, 255)),
        ("SAM3 refined", sam3_mask, (255, 180, 0)),
        ("Target", target, (230, 57, 70)),
    ]
    for ax, (title, mask, color) in zip(axes[:3], panels):
        ax.imshow(make_overlay(image, mask, color))
        ax.set_title(title)
        ax.axis("off")
    axes[3].imshow(image)
    x0, y0, x1, y1 = bbox
    axes[3].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=2))
    axes[3].set_title(
        f"{label}\nprior Dice={component_metrics['dice']:.3f}; SAM3 Dice={sam3_metrics['dice']:.3f}"
    )
    axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 refinement on molecular prior components.")
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--target-mask-dir", type=Path, default=DEFAULT_TARGET_MASK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "SAM3-base" / "sam3.pt"))
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--labels", type=str, default="immune_infiltration,stroma,tumor")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--top-factors-per-label", type=int, default=2)
    parser.add_argument("--top-components-per-factor", type=int, default=3)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    panels_dir = args.output_dir / "panels"
    for path in (masks_dir, overlays_dir, panels_dir):
        path.mkdir(parents=True, exist_ok=True)

    components_path = args.prior_dir / "candidate_components_he.csv"
    annotations_path = args.prior_dir / "factor_annotation.csv"
    if not components_path.exists():
        raise FileNotFoundError(f"Missing components: {components_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotations_path}")

    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    image_full = np.array(Image.open(args.he_image).convert("RGB"))
    targets_full = load_targets(args.target_mask_dir)
    target_masks = [targets_full[label] for label in labels if label in targets_full]
    target_labels = [label for label in labels if label in targets_full]
    image, resized_targets, resize_scale = resize_image_and_masks(
        image_full,
        target_masks,
        None if args.max_side <= 0 else args.max_side,
    )
    targets = dict(zip(target_labels, resized_targets))

    components = pd.read_csv(components_path)
    marker_scores = load_marker_scores(annotations_path)
    selected = select_components(
        components,
        marker_scores,
        target_labels,
        args.top_factors_per_label,
        args.top_components_per_factor,
    )

    print("=" * 72)
    print("Molecular component -> SAM3 box refinement")
    print("=" * 72)
    print(f"Prior dir: {args.prior_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image shape: {image.shape[0]} x {image.shape[1]} (scale={resize_scale:.4f})")
    print(f"Selected components: {len(selected)}")
    print(f"Output: {args.output_dir}")

    if args.device == "cpu" and torch.backends.mps.is_available():
        # SAM3's geometry encoder calls pin_memory() before moving a small scale
        # tensor. On Apple/MPS PyTorch this can create a CPU/MPS storage conflict
        # even when the model itself is on CPU.
        torch.Tensor.pin_memory = lambda self, *unused_args, **unused_kwargs: self

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    inference_state = sam3.encode_image(image)

    rows = []
    for _, row in selected.iterrows():
        label = str(row["target_label"])
        component_id = str(row["component_id"])
        stem = f"{slugify(label)}_{component_id}"
        component_mask_full = load_binary_mask(args.prior_dir / str(row["mask_path"]))
        component_mask = resize_image_and_masks(image_full, [component_mask_full], None if args.max_side <= 0 else args.max_side)[1][0]
        target = targets[label]
        bbox_full = (int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"]))
        bbox = expand_bbox(scale_bbox(bbox_full, resize_scale, image.shape[:2]), image.shape[:2], args.bbox_margin)
        pred = normalize_prediction(sam3.predict_box(inference_state, bbox, image.shape[:2]), image.shape[:2])

        component_metrics = metrics_to_dict(component_mask, target)
        sam3_metrics = metrics_to_dict(pred, target)
        save_mask(pred, masks_dir / f"{stem}_sam3_refined.png")
        save_mask(component_mask, masks_dir / f"{stem}_molecular_prior.png")
        Image.fromarray(make_overlay(image, pred, (255, 180, 0))).save(overlays_dir / f"{stem}_sam3_refined.png")
        save_panel(
            image,
            label,
            component_mask,
            pred,
            target,
            bbox,
            component_metrics,
            sam3_metrics,
            panels_dir / f"{stem}.png",
        )
        rows.append(
            {
                "label": label,
                "component_id": component_id,
                "factor": int(row["factor"]),
                "marker_score_for_label": float(row["marker_score_for_label"]),
                "component_area_fullres": int(row["area"]),
                "bbox_fullres": json.dumps(list(bbox_full)),
                "bbox_inference": json.dumps(list(bbox)),
                **{f"prior_{key}": value for key, value in component_metrics.items()},
                **{f"sam3_{key}": value for key, value in sam3_metrics.items()},
                "panel_path": str((panels_dir / f"{stem}.png").relative_to(args.output_dir)),
            }
        )
        print(
            f"  {label:22s} {component_id:18s} "
            f"prior Dice={component_metrics['dice']:.3f} SAM3 Dice={sam3_metrics['dice']:.3f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "molecular_component_sam3_refine.csv", index=False)
    best = (
        results.sort_values(["label", "sam3_dice"], ascending=[True, False])
        .groupby("label", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(args.output_dir / "molecular_component_sam3_refine_best.csv", index=False)
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "prior_dir": str(args.prior_dir),
                "he_image": str(args.he_image),
                "target_mask_dir": str(args.target_mask_dir),
                "output_dir": str(args.output_dir),
                "checkpoint": args.checkpoint,
                "resize_scale": resize_scale,
                "image_shape": list(image.shape[:2]),
                "labels": target_labels,
                "n_selected_components": int(len(selected)),
                "bbox_margin": args.bbox_margin,
            },
            indent=2,
        )
    )
    lines = ["# Molecular Component SAM3 Refinement", ""]
    lines.append("| label | component | factor | prior Dice | SAM3 Dice | SAM3 Recall | SAM3 Precision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in best.iterrows():
        lines.append(
            f"| {row['label']} | {row['component_id']} | {int(row['factor'])} | "
            f"{row['prior_dice']:.3f} | {row['sam3_dice']:.3f} | "
            f"{row['sam3_recall']:.3f} | {row['sam3_precision']:.3f} |"
        )
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(best[["label", "component_id", "factor", "prior_dice", "sam3_dice", "sam3_recall", "sam3_precision"]].to_string(index=False))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
