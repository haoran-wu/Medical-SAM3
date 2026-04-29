#!/usr/bin/env python3
"""
Shared helpers for Visium HD Exp1 SAM3 baselines.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from metrics import compute_all_metrics
from sam3_inference import resize_mask


DATA_DIR = Path("/nfs/roberts/project/pi_xy48/hw646/Exp1")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unlabeled"


def load_binary_mask(path: Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def parse_label_filter(label_arg: Optional[str]) -> Optional[set[str]]:
    if not label_arg:
        return None
    labels = [part.strip() for part in label_arg.split(",")]
    labels = [label for label in labels if label]
    return set(labels) if labels else None


def resolve_mask_path(project_root: Path, summary_path: Path, asset_path: str) -> Path:
    candidate = Path(asset_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate

    fallback = summary_path.parent / candidate.name
    if fallback.exists():
        return fallback

    return project_candidate


def select_label_records(
    labels: Sequence[Dict[str, object]],
    include_labels: Optional[set[str]] = None,
) -> List[Tuple[int, Dict[str, object]]]:
    selected: List[Tuple[int, Dict[str, object]]] = []
    for idx, item in enumerate(labels):
        label = str(item["label"])
        if include_labels is not None and label not in include_labels:
            continue
        selected.append((idx, item))
    return selected


def resize_image_and_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    max_side: Optional[int],
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    if max_side is None:
        return image, masks, 1.0

    image_h, image_w = image.shape[:2]
    current_max = max(image_h, image_w)
    if current_max <= max_side:
        return image, masks, 1.0

    scale = max_side / float(current_max)
    new_h = max(1, int(round(image_h * scale)))
    new_w = max(1, int(round(image_w * scale)))
    resized_image = np.array(
        Image.fromarray(image).resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    )
    resized_masks = [resize_mask(mask.astype(np.uint8), (new_h, new_w)).astype(np.uint8) for mask in masks]
    return resized_image, resized_masks, scale


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    margin_fraction: float,
) -> Tuple[int, int, int, int]:
    if margin_fraction <= 0:
        return bbox

    x_min, y_min, x_max, y_max = bbox
    height, width = image_shape
    box_w = x_max - x_min
    box_h = y_max - y_min
    dx = int(round(box_w * margin_fraction))
    dy = int(round(box_h * margin_fraction))
    return (
        max(0, x_min - dx),
        max(0, y_min - dy),
        min(width - 1, x_max + dx),
        min(height - 1, y_max + dy),
    )


def metrics_to_dict(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    m = compute_all_metrics(pred, gt)
    return {
        "dice": float(m.dice),
        "iou": float(m.iou),
        "precision": float(m.precision),
        "recall": float(m.recall),
        "psnr": float(m.psnr),
        "ssim": float(m.ssim),
    }


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    base = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def normalize_prediction(pred: Optional[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
    if pred is None:
        return np.zeros(target_shape, dtype=np.uint8)
    if pred.shape != target_shape:
        pred = resize_mask(pred, target_shape)
    return pred.astype(np.uint8)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
