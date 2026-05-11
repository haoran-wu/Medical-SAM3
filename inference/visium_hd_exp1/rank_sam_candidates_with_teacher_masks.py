#!/usr/bin/env python3
"""Select SAM candidate masks using teacher masks as priors.

The final prediction in this script is a union of SAM candidate masks. Teacher
masks from ensemble/pixel classifiers are used only to rank/select SAM masks,
so this tests a "SAM as final boundary generator" variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


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


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * radius + 1))) > 127


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(2 * radius + 1))) > 127


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


def teacher_score(pred: np.ndarray, teacher: np.ndarray, objective: str, beta: float) -> float:
    values = metrics(pred, teacher)
    if objective == "dice":
        return values["dice"]
    if objective == "precision":
        return values["precision"]
    if objective == "recall":
        return values["recall"]
    if objective == "fbeta":
        precision = values["precision"]
        recall = values["recall"]
        beta2 = beta * beta
        denom = beta2 * precision + recall
        return (1.0 + beta2) * precision * recall / denom if denom else 0.0
    raise ValueError(f"Unknown objective: {objective}")


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return Path.cwd() / path_text


def find_runs(roots: Iterable[Path]) -> List[Path]:
    runs: List[Path] = []
    for root in roots:
        if (root / "candidate_report.json").exists():
            runs.append(root)
        else:
            runs.extend(path.parent for path in sorted(root.glob("**/candidate_report.json")))
    return sorted(set(runs))


def find_teacher_masks(root: Path, variant_name: str) -> List[Tuple[str, str, Path]]:
    mask_root = root / "masks" if (root / "masks").exists() else root
    rows: List[Tuple[str, str, Path]] = []
    patterns = [
        ("best_ensemble", "{label}_best_ensemble_mask.png"),
        ("nonoverlap_ensemble", "{label}_nonoverlap_ensemble_mask.png"),
        ("micro_classifier", "{label}_micro_classifier_best.png"),
        ("hed_micro", "{label}_hed_micro_best.png"),
        ("unified_best", "{label}_unified_best_mask.png"),
        ("unified_argmax", "{label}_unified_argmax_mask.png"),
    ]
    for label in LABEL_ORDER:
        for default_variant, template in patterns:
            path = mask_root / template.format(label=label)
            if path.exists():
                rows.append((label, variant_name or default_variant, path))
    return rows


def greedy_select(
    candidates: Sequence[np.ndarray],
    teacher: np.ndarray,
    max_masks: int,
    min_delta: float,
    max_overlap: float,
    objective: str,
    beta: float,
) -> Tuple[np.ndarray, List[int], float]:
    current = np.zeros_like(teacher, dtype=bool)
    selected: List[int] = []
    used = np.zeros(len(candidates), dtype=bool)
    current_score = teacher_score(current, teacher, objective, beta)
    for _ in range(max_masks):
        best_idx = -1
        best_mask = current
        best_score = current_score
        for idx, cand in enumerate(candidates):
            if used[idx]:
                continue
            if current.any():
                overlap = np.logical_and(cand, current).sum() / max(1, cand.sum())
                if overlap > max_overlap:
                    continue
            trial = np.logical_or(current, cand)
            score = teacher_score(trial, teacher, objective, beta)
            if score > best_score:
                best_idx = idx
                best_mask = trial
                best_score = score
        if best_idx < 0 or best_score - current_score < min_delta:
            break
        used[best_idx] = True
        selected.append(best_idx)
        current = best_mask
        current_score = best_score
    return current, selected, current_score


def load_targets(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    targets = {}
    for item in summary["labels"]:
        label = item.get("slug") or slugify(item["label"])
        if label in LABEL_ORDER:
            targets[label] = resize_bool(read_mask(resolve_path(item["mask_path"], summary_path.parent)), shape_hw)
    return targets


def process_run(
    run_dir: Path,
    teacher_entries: Sequence[Tuple[str, str, Path]],
    max_masks_values: Sequence[int],
    min_delta: float,
    max_overlap: float,
    objective: str,
    beta: float,
    teacher_erode: int,
    teacher_dilate: int,
) -> List[Dict[str, object]]:
    report = json.loads((run_dir / "candidate_report.json").read_text())
    shape_hw = tuple(int(v) for v in report["image_shape"])
    summary_path = Path(report["summary_path"])
    if not summary_path.exists():
        summary_path = Path.cwd() / report["summary_path"]
    targets = load_targets(summary_path, shape_hw)
    candidate_paths = sorted((run_dir / "candidate_masks").glob("candidate_*.png"))
    candidates = [read_mask(path) for path in candidate_paths]
    rows: List[Dict[str, object]] = []
    if not candidates:
        return rows
    for label, teacher_variant, teacher_path in teacher_entries:
        if label not in targets:
            continue
        teacher = resize_bool(read_mask(teacher_path), shape_hw)
        if teacher_erode:
            teacher = erode(teacher, teacher_erode)
        if teacher_dilate:
            teacher = dilate(teacher, teacher_dilate)
        if not teacher.any():
            continue
        for max_masks in max_masks_values:
            pred, selected, teacher_fit = greedy_select(candidates, teacher, max_masks, min_delta, max_overlap, objective, beta)
            gt_score = metrics(pred, targets[label])
            teacher_score = metrics(pred, teacher)
            rows.append(
                {
                    "run": run_dir.name,
                    "label": label,
                    "teacher_variant": teacher_variant,
                    "teacher_path": str(teacher_path),
                    "objective": objective,
                    "beta": beta,
                    "teacher_erode": teacher_erode,
                    "teacher_dilate": teacher_dilate,
                    "max_masks": max_masks,
                    "n_selected": len(selected),
                    "selected_candidate_indices": ";".join(str(i) for i in selected),
                    "teacher_fit_dice": teacher_fit,
                    "teacher_precision": teacher_score["precision"],
                    "teacher_recall": teacher_score["recall"],
                    **gt_score,
                }
            )
    return rows


def write_outputs(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        (output_dir / "README.md").write_text("# Teacher-guided SAM Selection\n\nNo rows.\n")
        return
    df.to_csv(output_dir / "teacher_sam_selection_summary.csv", index=False)
    best = df.sort_values("dice", ascending=False).groupby("label", as_index=False).head(1)
    best.to_csv(output_dir / "teacher_sam_selection_best_by_label.csv", index=False)
    lines = ["# Teacher-guided SAM Selection", "", "Final predictions are unions of SAM candidate masks.", ""]
    lines += ["| label | Dice | precision | recall | teacher fit | run | teacher | max masks | selected |", "|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for _, row in best.sort_values("dice", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['teacher_fit_dice']:.3f} | {row['run']} | {row['teacher_variant']} | {int(row['max_masks'])} | {int(row['n_selected'])} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(best.sort_values("dice", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank/select SAM candidate masks with teacher masks.")
    parser.add_argument("--candidate-root", type=Path, action="append", required=True)
    parser.add_argument("--teacher-root", type=Path, action="append", required=True)
    parser.add_argument("--teacher-variant", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-masks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20, 32])
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--max-overlap", type=float, default=0.85)
    parser.add_argument("--selection-objective", choices=["dice", "fbeta", "precision", "recall"], default="dice")
    parser.add_argument("--fbeta", type=float, default=1.0)
    parser.add_argument("--teacher-erode", type=int, default=0)
    parser.add_argument("--teacher-dilate", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_entries: List[Tuple[str, str, Path]] = []
    for root in args.teacher_root:
        variant = args.teacher_variant or root.name
        teacher_entries.extend(find_teacher_masks(root, variant))
    rows: List[Dict[str, object]] = []
    for run in find_runs(args.candidate_root):
        print(f"Processing {run}", flush=True)
        rows.extend(
            process_run(
                run,
                teacher_entries,
                args.max_masks,
                args.min_delta,
                args.max_overlap,
                args.selection_objective,
                args.fbeta,
                args.teacher_erode,
                args.teacher_dilate,
            )
        )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
