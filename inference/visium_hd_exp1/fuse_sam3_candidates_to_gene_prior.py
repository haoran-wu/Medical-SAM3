#!/usr/bin/env python3
"""Fuse SAM3 candidates to gene-derived FICTURE prior masks.

This is a stricter gene+H&E+SAM fusion than candidate ranking. For each label:

1. Convert FICTURE factor labels plus factor_annotation.csv into a molecular
   prior score image.
2. Threshold that score image into a label-specific prior mask.
3. Greedily choose SAM3 masks that approximate the prior mask.
4. Evaluate both raw SAM-selected unions and prior-clipped unions against the
   annotated target when available.

The selection target is molecular evidence, not the annotation. The annotation
is used only for calibration on this single labeled sample.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


THRESHOLDS = [0.20, 0.35, 0.50, 0.65]


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


def resize_labels(labels: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if labels.shape == (h, w):
        return labels
    return np.array(Image.fromarray(labels.astype(np.int16)).resize((w, h), Image.Resampling.NEAREST)).astype(np.int16)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(img.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 127


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


def load_gene_factor_scores(path: Path, n_factors: int) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor = int(row["factor"])
            if factor < 0 or factor >= n_factors:
                continue
            label_scores = json.loads(row.get("label_scores_json") or "{}")
            for label, score in label_scores.items():
                slug = slugify(label)
                scores.setdefault(slug, np.zeros(n_factors, dtype=np.float64))
                scores[slug][factor] = max(scores[slug][factor], float(score))
    for slug, vec in list(scores.items()):
        if vec.max() > 0:
            scores[slug] = vec / vec.max()
    return scores


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return path


def prior_score_image(factors: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = np.zeros(factors.shape, dtype=np.float32)
    valid = factors >= 0
    out[valid] = scores[factors[valid]]
    return out


def greedy_to_prior(
    candidates: Sequence[np.ndarray],
    prior: np.ndarray,
    *,
    max_masks: int,
    min_delta: float,
) -> Tuple[np.ndarray, List[int], Dict[str, float]]:
    current = np.zeros_like(prior, dtype=bool)
    current_metrics = metrics(current, prior)
    selected: List[int] = []
    used = np.zeros(len(candidates), dtype=bool)
    for _ in range(max_masks):
        best_idx = -1
        best_mask = current
        best_metrics = current_metrics
        for idx, cand in enumerate(candidates):
            if used[idx]:
                continue
            trial = np.logical_or(current, cand)
            trial_metrics = metrics(trial, prior)
            if trial_metrics["dice"] > best_metrics["dice"]:
                best_idx = idx
                best_mask = trial
                best_metrics = trial_metrics
        if best_idx < 0 or best_metrics["dice"] - current_metrics["dice"] < min_delta:
            break
        used[best_idx] = True
        selected.append(best_idx)
        current = best_mask
        current_metrics = best_metrics
    return current, selected, current_metrics


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def summarize_one_run(
    run_dir: Path,
    output_dir: Path,
    factor_labels_full: np.ndarray,
    gene_scores: Dict[str, np.ndarray],
    max_masks: int,
    min_delta: float,
    prior_dilate: int,
) -> List[Dict[str, object]]:
    report = json.loads((run_dir / "candidate_report.json").read_text())
    shape_hw = tuple(int(v) for v in report["image_shape"])
    factors = resize_labels(factor_labels_full, shape_hw)
    summary_path = Path(report["summary_path"])
    if not summary_path.exists():
        summary_path = Path.cwd() / report["summary_path"]
    summary = json.loads(summary_path.read_text())
    image_path = Path(report["image_path"])
    if not image_path.exists():
        image_path = Path.cwd() / report["image_path"]
    image = np.array(Image.open(image_path).convert("RGB").resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR))

    candidate_paths = sorted((run_dir / "candidate_masks").glob("candidate_*.png"))
    candidates = [read_mask(path) for path in candidate_paths]
    rows: List[Dict[str, object]] = []
    panel_dir = output_dir / "panels" / run_dir.name
    mask_dir = output_dir / "masks" / run_dir.name
    panel_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for item in summary["labels"]:
        label = item["label"]
        slug = item.get("slug") or slugify(label)
        if slug not in gene_scores:
            continue
        target = resize_bool(read_mask(resolve_path(item["mask_path"], summary_path.parent)), shape_hw)
        score_img = prior_score_image(factors, gene_scores[slug])
        for threshold in THRESHOLDS:
            prior = score_img >= threshold
            if not prior.any():
                continue
            prior_context = dilate(prior, prior_dilate)
            selected_union, selected, prior_fit = greedy_to_prior(
                candidates,
                prior,
                max_masks=max_masks,
                min_delta=min_delta,
            )
            clipped = np.logical_and(selected_union, prior_context)
            strict_clipped = np.logical_and(selected_union, prior)
            all_union = np.logical_or.reduce(candidates) if candidates else np.zeros_like(prior)
            all_clipped = np.logical_and(all_union, prior_context)
            variants = {
                "gene_prior_only": prior,
                "sam_greedy_to_prior": selected_union,
                "sam_greedy_prior_context_clip": clipped,
                "sam_greedy_prior_strict_clip": strict_clipped,
                "all_sam_prior_context_clip": all_clipped,
            }
            for variant, pred in variants.items():
                row = {
                    "run": run_dir.name,
                    "label": label,
                    "slug": slug,
                    "threshold": threshold,
                    "variant": variant,
                    "n_selected": len(selected) if "greedy" in variant else 0,
                    "selected_candidate_indices": ";".join(str(idx) for idx in selected) if "greedy" in variant else "",
                    "prior_fit_dice": prior_fit["dice"],
                    **metrics(pred, target),
                }
                rows.append(row)
            best_variant, best_pred = max(variants.items(), key=lambda kv: metrics(kv[1], target)["dice"])
            Image.fromarray(best_pred.astype(np.uint8) * 255).save(mask_dir / f"{slug}_thr{threshold:.2f}_{best_variant}.png")
            canvas = np.concatenate(
                [
                    overlay(image, target, (255, 140, 0), 0.48),
                    overlay(image, prior, (255, 0, 180), 0.42),
                    overlay(image, best_pred, (0, 220, 255), 0.46),
                ],
                axis=1,
            )
            Image.fromarray(canvas).save(panel_dir / f"{slug}_thr{threshold:.2f}_target_prior_best.png")
    return rows


def find_runs(roots: Iterable[Path]) -> List[Path]:
    runs: List[Path] = []
    for root in roots:
        if (root / "candidate_report.json").exists():
            runs.append(root)
        else:
            runs.extend(path.parent for path in sorted(root.glob("**/candidate_report.json")))
    return runs


def write_summary(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "gene_prior_sam3_fusion_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["run"]), str(row["label"]))
        if key not in best or float(row["dice"]) > float(best[key]["dice"]):
            best[key] = dict(row)
    best_path = output_dir / "gene_prior_sam3_fusion_best_by_run_label.csv"
    if best:
        with best_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(next(iter(best.values())).keys()))
            writer.writeheader()
            writer.writerows(best.values())
    lines = ["# Gene-prior SAM3 Fusion", "", "| run | label | Dice | variant | threshold | selected | recall | precision |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in sorted(best.values(), key=lambda r: (str(r["run"]), str(r["label"]))):
        lines.append(
            f"| {row['run']} | {row['label']} | {float(row['dice']):.3f} | {row['variant']} | "
            f"{float(row['threshold']):.2f} | {row['n_selected']} | {float(row['recall']):.3f} | {float(row['precision']):.3f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse SAM3 candidates to gene-derived FICTURE prior masks.")
    parser.add_argument("--candidate-root", type=Path, action="append", required=True)
    parser.add_argument("--factor-image", type=Path, required=True)
    parser.add_argument("--factor-annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-masks", type=int, default=32)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--prior-dilate", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    factors = np.load(args.factor_image)
    n_factors = int(factors[factors >= 0].max()) + 1 if np.any(factors >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    rows: List[Dict[str, object]] = []
    for run in find_runs(args.candidate_root):
        print(f"Fusing {run}")
        rows.extend(summarize_one_run(run, args.output_dir, factors, gene_scores, args.max_masks, args.min_delta, args.prior_dilate))
    write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
