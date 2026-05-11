#!/usr/bin/env python3
"""Refine teacher-mask components with SAM3 box prompts.

Teacher masks such as unified pixel-classifier outputs are used only to define
component boxes and a local prior context. The delivered prediction remains a
union of SAM3-refined masks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam3_baseline_utils import ensure_dirs, make_overlay, metrics_to_dict, normalize_prediction, save_mask
from sam3_inference import SAM3Model


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

DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_local_region_summary" / "region_summary.json"


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(image.resize((w, h), Image.Resampling.NEAREST)) > 127


def morph_filter(mask: np.ndarray, erode_radius: int, dilate_radius: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    if erode_radius > 0:
        image = image.filter(ImageFilter.MinFilter(2 * erode_radius + 1))
    if dilate_radius > 0:
        image = image.filter(ImageFilter.MaxFilter(2 * dilate_radius + 1))
    return np.array(image) > 127


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return Path.cwd() / path_text


def load_targets(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    targets = {}
    for item in summary["labels"]:
        label = item.get("slug") or slugify(item["label"])
        if label in LABEL_ORDER:
            targets[label] = resize_bool(read_mask(resolve_path(item["mask_path"], summary_path.parent)), shape_hw)
    return targets


def find_teacher_masks(roots: Iterable[Path], variant_filter: str) -> List[Tuple[str, str, Path]]:
    patterns = [
        ("unified_best", "{label}_unified_best_mask.png"),
        ("unified_argmax", "{label}_unified_argmax_mask.png"),
        ("best_ensemble", "{label}_best_ensemble_mask.png"),
        ("nonoverlap_ensemble", "{label}_nonoverlap_ensemble_mask.png"),
        ("micro_classifier", "{label}_micro_classifier_best.png"),
    ]
    rows: List[Tuple[str, str, Path]] = []
    for root in roots:
        mask_root = root / "masks" if (root / "masks").exists() else root
        for label_name in LABEL_ORDER:
            for variant, template in patterns:
                if variant_filter != "all" and variant != variant_filter:
                    continue
                path = mask_root / template.format(label=label_name)
                if path.exists():
                    rows.append((label_name, f"{root.parent.name}:{variant}", path))
    return rows


def estimate_tissue_mask(image: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(image / 255.0)
    gray = rgb2gray(image / 255.0)
    tissue = (hsv[:, :, 1] > 0.045) & (gray < 0.94)
    tissue = remove_small_objects(tissue, min_size=max(64, image.size // 10000))
    return binary_closing(tissue, disk(5)).astype(bool)


def bbox_from_mask(mask: np.ndarray, margin: float) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    x0 -= w * margin
    x1 += w * margin
    y0 -= h * margin
    y1 += h * margin
    height, width = mask.shape
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(width - 1, int(round(x1))),
        min(height - 1, int(round(y1))),
    )


def keep_component_near_prior(
    pred_mask: np.ndarray,
    prior_mask: np.ndarray,
    tissue_mask: Optional[np.ndarray],
    prior_dilate_radius: int,
) -> np.ndarray:
    refined = pred_mask.astype(bool)
    if tissue_mask is not None:
        refined = refined & tissue_mask
    if not prior_mask.any() or not refined.any():
        return refined.astype(np.uint8)
    context = binary_dilation(prior_mask.astype(bool), disk(max(1, prior_dilate_radius)))
    labeled = label(refined)
    best_label = 0
    best_overlap = 0
    for region in regionprops(labeled):
        component = labeled == region.label
        overlap = int(np.logical_and(component, context).sum())
        if overlap > best_overlap:
            best_label = int(region.label)
            best_overlap = overlap
    if best_label == 0:
        return np.logical_and(refined, context).astype(np.uint8)
    return np.logical_and(labeled == best_label, context).astype(np.uint8)


def select_components(mask: np.ndarray, min_area: int, max_components: int) -> List[np.ndarray]:
    clean = remove_small_objects(mask.astype(bool), min_size=max(1, min_area // 2))
    labeled = label(clean)
    components = []
    for region in regionprops(labeled):
        if int(region.area) >= min_area:
            components.append((int(region.area), labeled == region.label))
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components[:max_components]]


def draw_panel(
    image: np.ndarray,
    label_name: str,
    bbox: Tuple[int, int, int, int],
    teacher_component: np.ndarray,
    sam_mask: np.ndarray,
    target_mask: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image)
    x0, y0, x1, y1 = bbox
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
    axes[0].set_title(f"{label_name}\nbox prompt")
    axes[1].imshow(make_overlay(image, teacher_component, (0, 220, 255), alpha=0.50))
    axes[1].set_title("teacher component")
    axes[2].imshow(make_overlay(image, sam_mask, (255, 180, 0), alpha=0.50))
    axes[2].set_title("SAM refined")
    axes[3].imshow(make_overlay(image, target_mask, (230, 57, 70), alpha=0.45))
    axes[3].set_title("annotation target")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def resize_image(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0 or max(image.shape[:2]) <= max_side:
        return image, 1.0
    scale = max_side / float(max(image.shape[:2]))
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    return np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)), scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine teacher mask components with SAM3.")
    parser.add_argument("--teacher-root", type=Path, action="append", required=True)
    parser.add_argument("--teacher-variant", choices=["all", "unified_best", "unified_argmax", "best_ensemble", "nonoverlap_ensemble", "micro_classifier"], default="unified_best")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--max-side", type=int, default=4096)
    parser.add_argument("--min-component-area", type=int, default=500)
    parser.add_argument("--max-components-per-label", type=int, default=8)
    parser.add_argument("--box-margin", type=float, default=0.04)
    parser.add_argument("--prior-dilate-radius", type=int, default=64)
    parser.add_argument("--teacher-erode", type=int, default=0)
    parser.add_argument("--teacher-dilate", type=int, default=0)
    parser.add_argument("--clip-to-tissue", action="store_true")
    parser.add_argument("--labels", default=",".join(LABEL_ORDER))
    args = parser.parse_args()

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    panels_dir = output_dir / "panels"
    ensure_dirs((output_dir, masks_dir, panels_dir))

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    image_full = np.array(Image.open(args.he_image).convert("RGB"))
    image, scale = resize_image(image_full, args.max_side)
    shape_hw = image.shape[:2]
    targets = load_targets(args.summary_path, shape_hw)
    tissue_mask = estimate_tissue_mask(image) if args.clip_to_tissue else None

    teacher_entries = [
        entry for entry in find_teacher_masks(args.teacher_root, args.teacher_variant) if entry[0] in labels
    ]
    if not teacher_entries:
        raise ValueError("No teacher masks found.")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    state = sam3.encode_image(image)

    rows: List[Dict[str, object]] = []
    union_rows: List[Dict[str, object]] = []
    for label_name, teacher_variant, teacher_path in teacher_entries:
        if label_name not in targets:
            continue
        teacher = resize_bool(read_mask(teacher_path), shape_hw)
        teacher = morph_filter(teacher, args.teacher_erode, args.teacher_dilate)
        components = select_components(teacher, args.min_component_area, args.max_components_per_label)
        if not components:
            continue
        sam_union = np.zeros(shape_hw, dtype=bool)
        teacher_union = np.zeros(shape_hw, dtype=bool)
        for rank, component in enumerate(components, start=1):
            bbox = bbox_from_mask(component, args.box_margin)
            if bbox is None:
                continue
            pred = normalize_prediction(sam3.predict_box(state, bbox, shape_hw), shape_hw)
            pred = keep_component_near_prior(pred, component, tissue_mask, args.prior_dilate_radius).astype(bool)
            sam_union |= pred
            teacher_union |= component
            stem = f"{label_name}_{slugify(teacher_variant)}_r{rank:02d}"
            save_mask(pred, masks_dir / f"{stem}_sam.png")
            if rank <= 6:
                draw_panel(image, label_name, bbox, component, pred, targets[label_name], panels_dir / f"{stem}.png")
            target_metrics = metrics_to_dict(pred, targets[label_name])
            teacher_metrics = metrics_to_dict(pred, component)
            rows.append(
                {
                    "label": label_name,
                    "teacher_variant": teacher_variant,
                    "teacher_path": str(teacher_path),
                    "component_rank": rank,
                    "component_pixels": int(component.sum()),
                    "sam_pixels": int(pred.sum()),
                    "bbox_xyxy": json.dumps(list(bbox)),
                    **{f"target_{key}": value for key, value in target_metrics.items()},
                    **{f"teacher_{key}": value for key, value in teacher_metrics.items()},
                }
            )
        save_mask(sam_union, masks_dir / f"{label_name}_{slugify(teacher_variant)}_sam_union.png")
        save_mask(teacher_union, masks_dir / f"{label_name}_{slugify(teacher_variant)}_teacher_union.png")
        union_metrics = metrics_to_dict(sam_union, targets[label_name])
        teacher_union_metrics = metrics_to_dict(sam_union, teacher_union)
        union_rows.append(
            {
                "label": label_name,
                "teacher_variant": teacher_variant,
                "teacher_path": str(teacher_path),
                "n_components": len(components),
                "sam_pixels": int(sam_union.sum()),
                "teacher_pixels": int(teacher_union.sum()),
                **{f"target_{key}": value for key, value in union_metrics.items()},
                **{f"teacher_{key}": value for key, value in teacher_union_metrics.items()},
            }
        )

    comp_df = pd.DataFrame(rows)
    union_df = pd.DataFrame(union_rows)
    comp_df.to_csv(output_dir / "teacher_component_sam3_components.csv", index=False)
    union_df.to_csv(output_dir / "teacher_component_sam3_by_label.csv", index=False)
    if not union_df.empty:
        best = union_df.sort_values("target_dice", ascending=False).groupby("label", as_index=False).head(1)
        best.to_csv(output_dir / "teacher_component_sam3_best_by_label.csv", index=False)
        lines = ["# Teacher Component SAM3 Refinement", ""]
        lines += ["| label | Dice | precision | recall | teacher fit | SAM pixels | teacher |", "|---|---:|---:|---:|---:|---:|---|"]
        for _, row in best.sort_values("target_dice", ascending=False).iterrows():
            lines.append(
                f"| {row['label']} | {row['target_dice']:.3f} | {row['target_precision']:.3f} | "
                f"{row['target_recall']:.3f} | {row['teacher_dice']:.3f} | {int(row['sam_pixels'])} | {row['teacher_variant']} |"
            )
        (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
        print(best.sort_values("target_dice", ascending=False).to_string(index=False))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
