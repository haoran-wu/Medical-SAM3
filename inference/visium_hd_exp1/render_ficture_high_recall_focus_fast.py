#!/usr/bin/env python3
"""Render higher-recall FICTURE candidate comparisons for focus classes."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/visium_hd_exp1/ficture_high_recall_focus_fast"
HE_PATH = PROJECT_ROOT / "output/visium_hd_exp1/ficture_candidate_pool_input_roi/he_roi_matching_ficture_coverage.png"

CONFIG = {
    "lung_bronchiola": {
        "display": "bronchiola",
        "old": (0.731, 0.827, 0.655),
        "dirs": [
            "results/visium_hd_exp1/ficture_candidate_pool_roi_light_point/ficture_roi_lpt_11737392/base_point80_m1536_p700",
            "results/visium_hd_exp1/ficture_candidate_pool_roi/ficture_roi_all_11735105/base_ficture_roi_boxes_b384_s96",
            "results/visium_hd_exp1/ficture_candidate_pool_roi_box_sweep/ficture_roi_box_11737389/base_box512_s160_m1536",
        ],
    },
    "lung_alveoli_normal_adjacent": {
        "display": "alveoli",
        "old": (0.678, 0.861, 0.560),
        "dirs": [
            "results/visium_hd_exp1/ficture_candidate_pool_roi/ficture_roi_all_11735105/base_ficture_roi_boxes_b384_s96",
            "results/visium_hd_exp1/ficture_candidate_pool_roi_box_sweep/ficture_roi_box_11737389/base_box512_s160_m1536",
            "results/visium_hd_exp1/ficture_candidate_pool_roi_box_sweep/ficture_roi_box_11737389/base_box384_s128_m1536",
        ],
    },
    "lung_vessels": {
        "display": "vessels",
        "old": (0.629, 0.857, 0.497),
        "dirs": [
            "results/visium_hd_exp1/ficture_candidate_pool_roi_box_sweep/ficture_roi_box_11737389/base_box512_s160_m1536",
            "results/visium_hd_exp1/ficture_candidate_pool_roi_box_sweep/ficture_roi_box_11737389/base_box384_s128_m1536",
            "results/visium_hd_exp1/ficture_candidate_pool_roi/ficture_roi_all_11735105/base_ficture_roi_boxes_b384_s96",
        ],
    },
}


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape == (h, w):
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype("uint8") * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


def resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if image.shape[:2] == (h, w):
        return image
    return np.array(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, int]:
    tp = np.logical_and(pred, target).sum()
    pred_pixels = pred.sum()
    target_pixels = target.sum()
    precision = tp / pred_pixels if pred_pixels else 0.0
    recall = tp / target_pixels if target_pixels else 0.0
    dice = 2 * tp / (pred_pixels + target_pixels) if pred_pixels + target_pixels else 1.0
    return float(dice), float(precision), float(recall), int(pred_pixels)


def resolve_path(path_value: str, parent: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if (parent / path).exists():
        return parent / path
    return PROJECT_ROOT / path


def method_label(setting: str) -> str:
    model = "Medical-SAM3" if setting.startswith("medical") else "SAM"
    match = re.search(r"point(\d+)", setting)
    if match:
        return f"{model} on FICTURE image, point prompts ({match.group(1)} px spacing)"
    match = re.search(r"(?:box|b)(\d+)_s(\d+)", setting)
    if match:
        return f"{model} on FICTURE image, box prompts ({match.group(1)} px box, {match.group(2)} px stride)"
    return f"{model} on FICTURE image"


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype("float32").copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype="float32")
    return np.clip(out, 0, 255).astype("uint8")


def solid(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.full((*mask.shape, 3), 255, dtype="uint8")
    out[mask] = np.array(color, dtype="uint8")
    return out


def render(display: str, he: np.ndarray, candidate: np.ndarray, annotation: np.ndarray, row: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(24, 8))
    panels = [
        (overlay(he, candidate, (0, 112, 255)), "Candidate on H&E", "blue"),
        (solid(candidate, (0, 112, 255)), "Candidate only", "blue"),
        (overlay(he, annotation, (0, 185, 95)), "Annotation on H&E", "green"),
        (solid(annotation, (0, 185, 95)), "Annotation only", "green"),
        (he, "H&E only", "black"),
    ]
    for ax, (image, title, color) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, color=color, loc="left")
        ax.axis("off")
    fig.suptitle(
        f"{display}: higher-recall candidate comparison\n"
        f"Dice {row['dice']:.3f} | Precision {row['precision']:.3f} | Recall {row['recall']:.3f}\n"
        f"{row['method']}",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (OUTPUT_DIR / "five_panel").mkdir(parents=True, exist_ok=True)
    he_orig = load_rgb(HE_PATH)
    rows: list[dict] = []

    for label, config in CONFIG.items():
        display = config["display"]
        old_dice, old_precision, old_recall = config["old"]
        items: list[dict] = []
        for rel_dir in config["dirs"]:
            run_dir = PROJECT_ROOT / rel_dir
            report = json.loads((run_dir / "candidate_report.json").read_text())
            shape = tuple(report["image_shape"])
            summary_path = Path(report["summary_path"])
            summary = json.loads(summary_path.read_text())
            target_item = next(item for item in summary["labels"] if item["label"] == label)
            target_path = resolve_path(target_item["mask_path"], summary_path.parent)
            target = resize_bool(load_mask(target_path), shape)
            for candidate_path in sorted((run_dir / "candidate_masks").glob("candidate_*.png")):
                candidate = resize_bool(load_mask(candidate_path), shape)
                dice, precision, recall, pixels = compute_metrics(candidate, target)
                items.append(
                    {
                        "label": label,
                        "display": display,
                        "old_dice": old_dice,
                        "old_precision": old_precision,
                        "old_recall": old_recall,
                        "dice": dice,
                        "precision": precision,
                        "recall": recall,
                        "pixels": pixels,
                        "method": method_label(run_dir.name),
                        "idx": int(candidate_path.stem.split("_")[-1]),
                        "dir": str(run_dir),
                        "candidate_path": str(candidate_path),
                        "target_path": str(target_path),
                        "shape": shape,
                    }
                )

        pool = [
            item
            for item in items
            if item["recall"] > old_recall + 0.03
            and item["dice"] >= old_dice - 0.12
            and item["precision"] >= old_precision - 0.20
        ]
        if not pool:
            pool = [
                item
                for item in items
                if item["recall"] > old_recall + 0.03
                and item["dice"] >= old_dice - 0.20
                and item["precision"] >= old_precision - 0.35
            ]
        if not pool:
            pool = [max(items, key=lambda item: item["dice"])]

        chosen = max(pool, key=lambda item: (item["recall"], item["dice"], item["precision"]))
        rows.append(chosen)
        shape = chosen["shape"]
        he = resize_rgb(he_orig, shape)
        candidate = resize_bool(load_mask(Path(chosen["candidate_path"])), shape)
        annotation = resize_bool(load_mask(Path(chosen["target_path"])), shape)
        render(display, he, candidate, annotation, chosen, OUTPUT_DIR / "five_panel" / f"{label}_high_recall_5panel.png")

    table_path = OUTPUT_DIR / "high_recall_focus_fast_table.csv"
    fields = [
        "display",
        "old_dice",
        "old_precision",
        "old_recall",
        "dice",
        "precision",
        "recall",
        "pixels",
        "method",
        "idx",
        "dir",
        "candidate_path",
    ]
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])

    for row in rows:
        print(
            row["display"],
            f"old={row['old_dice']:.3f}/{row['old_precision']:.3f}/{row['old_recall']:.3f}",
            f"new={row['dice']:.3f}/{row['precision']:.3f}/{row['recall']:.3f}",
            row["method"],
            "idx",
            row["idx"],
        )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
