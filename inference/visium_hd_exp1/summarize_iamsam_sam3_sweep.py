#!/usr/bin/env python3
"""Summarize IAMSAM-style SAM3 candidate sweeps.

The candidate generator reports union recall and best single-candidate Dice.
IAMSAM, however, lets users choose one or multiple SAM masks as an ROI. This
script adds a multi-candidate oracle: for each annotated label, greedily select
candidate masks that improve Dice against the target. It is an upper-bound
diagnostic for whether the SAM candidate pool contains useful H&E morphology
pieces before any non-oracle ranker is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


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


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


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


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def greedy_oracle(
    candidates: List[np.ndarray],
    gt: np.ndarray,
    *,
    max_masks: int,
    min_delta: float,
) -> Tuple[np.ndarray, List[int], Dict[str, float]]:
    selected: List[int] = []
    current = np.zeros_like(gt, dtype=bool)
    current_metrics = metrics(current, gt)
    used = np.zeros(len(candidates), dtype=bool)

    for _ in range(max_masks):
        best_idx = -1
        best_mask = current
        best_metrics = current_metrics
        for idx, cand in enumerate(candidates):
            if used[idx]:
                continue
            trial = np.logical_or(current, cand)
            trial_metrics = metrics(trial, gt)
            if trial_metrics["dice"] > best_metrics["dice"]:
                best_idx = idx
                best_mask = trial
                best_metrics = trial_metrics
        if best_idx < 0 or best_metrics["dice"] - current_metrics["dice"] < min_delta:
            break
        selected.append(best_idx)
        used[best_idx] = True
        current = best_mask
        current_metrics = best_metrics
    return current, selected, current_metrics


def summarize_run(run_dir: Path, out_dir: Path, max_masks: int, min_delta: float) -> List[Dict[str, object]]:
    report_path = run_dir / "candidate_report.json"
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text())
    image = np.array(Image.open(report["image_path"]).convert("RGB"))
    shape = tuple(report["image_shape"])
    image = np.array(Image.fromarray(image).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))

    candidate_paths = sorted((run_dir / "candidate_masks").glob("candidate_*.png"))
    candidates = [load_mask(path) for path in candidate_paths]
    rows: List[Dict[str, object]] = []
    panel_dir = out_dir / run_dir.name / "oracle_panels"
    mask_dir = out_dir / run_dir.name / "oracle_masks"
    panel_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for item in report["labels"]:
        label = item["label"]
        slug = item.get("slug", label)
        gt = load_mask(Path(next(label_item["mask_path"] for label_item in json.loads(Path(report["summary_path"]).read_text())["labels"] if label_item["label"] == label)))
        if gt.shape != shape:
            gt = resize_mask(gt, shape)
        pred, selected, oracle_metrics = greedy_oracle(candidates, gt, max_masks=max_masks, min_delta=min_delta)
        union_metrics = item["union_metrics"]
        best_metrics = item["best_candidate_metrics"]
        color = LABEL_COLORS.get(slug, (0, 220, 255))
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / f"{slug}_oracle_union.png")
        canvas = np.concatenate(
            [
                overlay(image, gt, (255, 140, 0), alpha=0.48),
                overlay(image, pred, color, alpha=0.48),
            ],
            axis=1,
        )
        Image.fromarray(canvas).save(panel_dir / f"{slug}_target_vs_oracle.png")
        rows.append(
            {
                "run": run_dir.name,
                "label": label,
                "slug": slug,
                "n_candidates": len(candidates),
                "selected_candidate_indices": ";".join(str(idx) for idx in selected),
                "n_selected": len(selected),
                "best_single_dice": best_metrics["dice"],
                "best_single_recall": best_metrics["recall"],
                "all_candidate_union_recall": union_metrics["recall"],
                "all_candidate_union_dice": union_metrics["dice"],
                **{f"oracle_{k}": v for k, v in oracle_metrics.items()},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IAMSAM-style SAM3 candidate sweeps.")
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-masks", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=0.002)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for report_path in sorted(args.sweep_root.glob("*/candidate_report.json")):
        rows.extend(summarize_run(report_path.parent, args.output_dir, args.max_masks, args.min_delta))

    csv_path = args.output_dir / "iamsam_sam3_oracle_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_lines = ["# IAMSAM-style SAM3 Candidate Sweep Summary", ""]
    md_lines.append("| run | label | best single Dice | oracle multi-mask Dice | oracle recall | selected |")
    md_lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        md_lines.append(
            f"| {row['run']} | {row['label']} | {row['best_single_dice']:.3f} | "
            f"{row['oracle_dice']:.3f} | {row['oracle_recall']:.3f} | {row['n_selected']} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
