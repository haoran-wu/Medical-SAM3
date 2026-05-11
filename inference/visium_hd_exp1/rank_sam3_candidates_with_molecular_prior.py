#!/usr/bin/env python3
"""Rank SAM3 candidate masks with FICTURE molecular priors and H&E features.

This script turns the current Visium HD Exp1 setup into the SaLIP/IAMSAM-style
pipeline we actually need:

1. SAM3/Medical-SAM3 proposes many H&E-aligned candidate masks.
2. FICTURE factor labels projected into H&E coordinates provide molecular
   composition for each candidate.
3. The single annotated sample supplies label prototypes for factor makeup,
   H&E color, texture, and size.
4. Multiple non-oracle rankers select one or several SAM candidates per label.

The reported Dice is a calibration diagnostic on the one annotated sample, not
a claim that the same Dice is known for unannotated samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


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

VARIANTS = [
    "factor_dot",
    "factor_cosine",
    "factor_js",
    "factor_loglik",
    "factor_top2",
    "factor_purity_top2",
    "gene_factor_score",
    "gene_factor_score_purity",
    "he_color_gaussian",
    "he_color_cosine",
    "boundary_gradient",
    "vmf_factor_color",
    "combo_equal",
    "combo_gene_he_sam",
    "combo_gene_annotation_he",
]


@dataclass
class LabelPrototype:
    label: str
    slug: str
    target: np.ndarray
    factor_hist: np.ndarray
    top_factors: np.ndarray
    rgb_mean: np.ndarray
    rgb_std: np.ndarray
    rgb_unit: np.ndarray
    gradient_mean: float
    area_fraction: float
    gene_factor_scores: np.ndarray


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)
    ) > 127


def resize_labels(labels: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if labels.shape == (h, w):
        return labels
    return np.array(
        Image.fromarray(labels.astype(np.int16)).resize((w, h), Image.Resampling.NEAREST)
    ).astype(np.int16)


def resize_image(image: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return np.array(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))


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


def safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec, dtype=np.float64)
    return vec.astype(np.float64) / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    au = safe_unit(a)
    bu = safe_unit(b)
    return float(np.dot(au, bu))


def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(np.where(p > 0, p * np.log((p + 1e-12) / (m + 1e-12)), 0.0)))
    kl_qm = float(np.sum(np.where(q > 0, q * np.log((q + 1e-12) / (m + 1e-12)), 0.0)))
    js = 0.5 * (kl_pm + kl_qm)
    return 1.0 - min(1.0, js / math.log(2.0))


def label_hist(factor_labels: np.ndarray, mask: np.ndarray, n_factors: int) -> np.ndarray:
    values = factor_labels[mask & (factor_labels >= 0)]
    hist = np.bincount(values.astype(np.int64), minlength=n_factors).astype(np.float64)
    hist += 1e-3
    return hist / hist.sum()


def load_gene_factor_scores(path: Path | None, n_factors: int) -> Dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
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
                if slug not in scores:
                    scores[slug] = np.zeros(n_factors, dtype=np.float64)
                scores[slug][factor] = max(scores[slug][factor], float(score))
    for slug, vec in list(scores.items()):
        if vec.max() > 0:
            scores[slug] = vec / vec.max()
    return scores


def gradient_image(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    grad = np.sqrt(gx * gx + gy * gy)
    denom = np.percentile(grad, 99.0)
    if denom > 0:
        grad = np.clip(grad / denom, 0, 1)
    return grad.astype(np.float32)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return path


def load_label_prototypes(
    summary_path: Path,
    image: np.ndarray,
    factor_labels: np.ndarray,
    shape_hw: Tuple[int, int],
    gene_factor_scores: Dict[str, np.ndarray] | None = None,
) -> Dict[str, LabelPrototype]:
    summary = json.loads(summary_path.read_text())
    image = resize_image(image, shape_hw)
    factors = resize_labels(factor_labels, shape_hw)
    grad = gradient_image(image)
    n_factors = int(factors[factors >= 0].max()) + 1 if np.any(factors >= 0) else 1
    prototypes: Dict[str, LabelPrototype] = {}

    for item in summary["labels"]:
        label = item["label"]
        slug = item.get("slug") or slugify(label)
        mask_path = resolve_path(item["mask_path"], summary_path.parent)
        target = resize_bool(read_mask(mask_path), shape_hw)
        hist = label_hist(factors, target, n_factors)
        top_factors = np.argsort(hist)[::-1][:2]
        pixels = image[target]
        if pixels.size:
            rgb_mean = pixels.mean(axis=0).astype(np.float64)
            rgb_std = np.maximum(pixels.std(axis=0).astype(np.float64), 1.0)
        else:
            rgb_mean = np.zeros(3, dtype=np.float64)
            rgb_std = np.ones(3, dtype=np.float64)
        boundary = mask_boundary(target)
        gene_scores = None
        if gene_factor_scores is not None:
            gene_scores = gene_factor_scores.get(slug)
        if gene_scores is None:
            gene_scores = np.zeros(n_factors, dtype=np.float64)
        prototypes[slug] = LabelPrototype(
            label=label,
            slug=slug,
            target=target,
            factor_hist=hist,
            top_factors=top_factors,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            rgb_unit=safe_unit(rgb_mean),
            gradient_mean=float(grad[boundary].mean()) if boundary.any() else 0.0,
            area_fraction=float(target.mean()),
            gene_factor_scores=gene_scores,
        )
    return prototypes


def candidate_features(
    mask: np.ndarray,
    image: np.ndarray,
    factor_labels: np.ndarray,
    grad: np.ndarray,
    n_factors: int,
) -> Dict[str, object]:
    if not mask.any():
        rgb_mean = np.zeros(3, dtype=np.float64)
        rgb_std = np.ones(3, dtype=np.float64)
    else:
        pixels = image[mask]
        rgb_mean = pixels.mean(axis=0).astype(np.float64)
        rgb_std = np.maximum(pixels.std(axis=0).astype(np.float64), 1.0)
    hist = label_hist(factor_labels, mask, n_factors)
    boundary = mask_boundary(mask)
    return {
        "factor_hist": hist,
        "top_factor": int(np.argmax(hist)),
        "factor_purity": float(hist.max()),
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "rgb_unit": safe_unit(rgb_mean),
        "boundary_gradient": float(grad[boundary].mean()) if boundary.any() else 0.0,
        "area_fraction": float(mask.mean()),
    }


def score_candidate(features: Dict[str, object], proto: LabelPrototype, variant: str) -> float:
    fh = features["factor_hist"]
    rgb_mean = features["rgb_mean"]
    rgb_std = features["rgb_std"]
    area_fraction = max(float(features["area_fraction"]), 1e-12)
    proto_area = max(proto.area_fraction, 1e-12)
    area_score = math.exp(-abs(math.log(area_fraction / proto_area)))
    top2_overlap = float(np.asarray(fh)[proto.top_factors].sum())
    factor_cos = cosine(fh, proto.factor_hist)
    factor_dot = float(np.dot(fh, proto.factor_hist))
    factor_js = js_similarity(fh, proto.factor_hist)
    factor_loglik = float(np.sum(proto.factor_hist * np.log(np.asarray(fh) + 1e-9)))
    factor_loglik = 1.0 / (1.0 + math.exp(-0.6 * (factor_loglik + 2.0)))
    gene_factor_score = float(np.dot(fh, proto.gene_factor_scores)) if proto.gene_factor_scores.size else 0.0
    color_z = float(np.mean(np.abs((rgb_mean - proto.rgb_mean) / proto.rgb_std)))
    color_gaussian = math.exp(-0.5 * color_z)
    color_cos = cosine(features["rgb_unit"], proto.rgb_unit)
    boundary = float(features["boundary_gradient"])
    boundary_match = math.exp(-abs(boundary - proto.gradient_mean))
    purity = float(features["factor_purity"])
    vmf_vec_c = np.concatenate(
        [
            safe_unit(np.asarray(fh)) * 1.7,
            safe_unit(rgb_mean / 255.0) * 0.8,
            np.array([boundary, area_score], dtype=np.float64) * 0.5,
        ]
    )
    vmf_vec_p = np.concatenate(
        [
            safe_unit(proto.factor_hist) * 1.7,
            safe_unit(proto.rgb_mean / 255.0) * 0.8,
            np.array([proto.gradient_mean, 1.0], dtype=np.float64) * 0.5,
        ]
    )
    vmf_factor_color = cosine(vmf_vec_c, vmf_vec_p)

    if variant == "factor_dot":
        return factor_dot
    if variant == "factor_cosine":
        return factor_cos
    if variant == "factor_js":
        return factor_js
    if variant == "factor_loglik":
        return factor_loglik
    if variant == "factor_top2":
        return top2_overlap
    if variant == "factor_purity_top2":
        return top2_overlap * (0.5 + 0.5 * purity)
    if variant == "gene_factor_score":
        return gene_factor_score
    if variant == "gene_factor_score_purity":
        return gene_factor_score * (0.5 + 0.5 * purity)
    if variant == "he_color_gaussian":
        return color_gaussian
    if variant == "he_color_cosine":
        return color_cos
    if variant == "boundary_gradient":
        return 0.55 * boundary_match + 0.45 * area_score
    if variant == "vmf_factor_color":
        return vmf_factor_color
    if variant == "combo_equal":
        return 0.25 * factor_cos + 0.25 * factor_js + 0.20 * color_gaussian + 0.15 * boundary_match + 0.15 * area_score
    if variant == "combo_gene_he_sam":
        return 0.34 * factor_js + 0.22 * factor_loglik + 0.18 * color_gaussian + 0.14 * boundary_match + 0.12 * area_score
    if variant == "combo_gene_annotation_he":
        return 0.40 * gene_factor_score + 0.20 * factor_js + 0.18 * color_gaussian + 0.12 * boundary_match + 0.10 * area_score
    raise KeyError(variant)


def select_ranked_union(
    scored: Sequence[Tuple[float, int, np.ndarray]],
    *,
    top_k: int,
    max_overlap: float,
) -> Tuple[np.ndarray, List[int]]:
    selected: List[int] = []
    current: np.ndarray | None = None
    for score, idx, mask in sorted(scored, key=lambda item: item[0], reverse=True):
        if top_k and len(selected) >= top_k:
            break
        if current is not None:
            inter = float(np.logical_and(current, mask).sum())
            denom = float(mask.sum())
            if denom > 0 and inter / denom > max_overlap:
                continue
            current = np.logical_or(current, mask)
        else:
            current = mask.copy()
        selected.append(idx)
    if current is None:
        current = np.zeros_like(scored[0][2], dtype=bool)
    return current, selected


def summarize_one_run(
    run_dir: Path,
    out_dir: Path,
    factor_image: np.ndarray,
    gene_factor_scores: Dict[str, np.ndarray],
    top_ks: Sequence[int],
    max_overlap: float,
    save_panels: bool,
) -> List[Dict[str, object]]:
    report_path = run_dir / "candidate_report.json"
    report = json.loads(report_path.read_text())
    shape_hw = tuple(int(v) for v in report["image_shape"])
    image_path = Path(report["image_path"])
    if not image_path.exists():
        image_path = Path.cwd() / report["image_path"]
    image = resize_image(np.array(Image.open(image_path).convert("RGB")), shape_hw)
    factors = resize_labels(factor_image, shape_hw)
    grad = gradient_image(image)
    n_factors = int(factors[factors >= 0].max()) + 1 if np.any(factors >= 0) else 1
    summary_path = Path(report["summary_path"])
    if not summary_path.exists():
        summary_path = Path.cwd() / report["summary_path"]
    prototypes = load_label_prototypes(summary_path, image, factor_image, shape_hw, gene_factor_scores)

    candidate_paths = sorted((run_dir / "candidate_masks").glob("candidate_*.png"))
    masks = [read_mask(path) for path in candidate_paths]
    features = [candidate_features(mask, image, factors, grad, n_factors) for mask in masks]
    rows: List[Dict[str, object]] = []
    mask_out = out_dir / "selected_masks" / run_dir.name
    panel_out = out_dir / "selected_panels" / run_dir.name
    mask_out.mkdir(parents=True, exist_ok=True)
    panel_out.mkdir(parents=True, exist_ok=True)

    for proto in prototypes.values():
        label_scores: Dict[str, List[Tuple[float, int, np.ndarray]]] = {}
        for variant in VARIANTS:
            scored = [
                (score_candidate(feat, proto, variant), idx, masks[idx])
                for idx, feat in enumerate(features)
                if masks[idx].any()
            ]
            label_scores[variant] = scored
            for top_k in top_ks:
                pred, selected = select_ranked_union(scored, top_k=top_k, max_overlap=max_overlap)
                row_metrics = metrics(pred, proto.target)
                rows.append(
                    {
                        "run": run_dir.name,
                        "label": proto.label,
                        "slug": proto.slug,
                        "variant": variant,
                        "top_k": top_k,
                        "n_selected": len(selected),
                        "selected_candidate_indices": ";".join(str(x) for x in selected),
                        **row_metrics,
                    }
                )

        if save_panels:
            best = max(
                [r for r in rows if r["run"] == run_dir.name and r["slug"] == proto.slug],
                key=lambda r: float(r["dice"]),
            )
            best_scored = label_scores[str(best["variant"])]
            best_pred, _ = select_ranked_union(
                best_scored,
                top_k=int(best["top_k"]),
                max_overlap=max_overlap,
            )
            color = LABEL_COLORS.get(proto.slug, (0, 220, 255))
            Image.fromarray(best_pred.astype(np.uint8) * 255).save(mask_out / f"{proto.slug}_best_ranked.png")
            canvas = np.concatenate(
                [
                    overlay(image, proto.target, (255, 140, 0), alpha=0.48),
                    overlay(image, best_pred, color, alpha=0.48),
                ],
                axis=1,
            )
            Image.fromarray(canvas).save(panel_out / f"{proto.slug}_target_vs_best_ranked.png")
    return rows


def write_summary(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "molecular_sam3_ranked_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    best_by_label: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["run"]), str(row["label"]))
        if key not in best_by_label or float(row["dice"]) > float(best_by_label[key]["dice"]):
            best_by_label[key] = dict(row)

    best_csv = output_dir / "molecular_sam3_ranked_best_by_run_label.csv"
    if best_by_label:
        with best_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(next(iter(best_by_label.values())).keys()))
            writer.writeheader()
            writer.writerows(best_by_label.values())

    lines = [
        "# Molecular-prior SAM3 Candidate Ranking",
        "",
        "Calibration on the single annotated Visium HD Exp1 sample. The rankers use FICTURE factor composition, H&E color/texture, area priors, and SAM candidate masks.",
        "",
        "| run | label | best Dice | variant | top_k | selected | recall | precision |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(best_by_label.values(), key=lambda r: (str(r["run"]), str(r["label"]))):
        lines.append(
            f"| {row['run']} | {row['label']} | {float(row['dice']):.3f} | {row['variant']} | "
            f"{row['top_k']} | {row['n_selected']} | {float(row['recall']):.3f} | {float(row['precision']):.3f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {best_csv}")


def find_runs(roots: Iterable[Path]) -> List[Path]:
    runs: List[Path] = []
    for root in roots:
        if (root / "candidate_report.json").exists():
            runs.append(root)
        else:
            runs.extend(path.parent for path in sorted(root.glob("**/candidate_report.json")))
    seen = set()
    unique = []
    for run in runs:
        resolved = run.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(run)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank SAM3 candidate masks with molecular priors.")
    parser.add_argument("--candidate-root", type=Path, action="append", required=True)
    parser.add_argument("--factor-image", type=Path, required=True)
    parser.add_argument(
        "--factor-annotation",
        type=Path,
        default=None,
        help="Optional FICTURE factor_annotation.csv containing gene-derived label_scores_json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20, 32])
    parser.add_argument("--max-overlap", type=float, default=0.72)
    parser.add_argument("--skip-panels", action="store_true", help="Compute tables only; avoid writing heavy mask/panel PNGs.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    factor_image = np.load(args.factor_image)
    n_factors = int(factor_image[factor_image >= 0].max()) + 1 if np.any(factor_image >= 0) else 1
    factor_annotation = args.factor_annotation
    if factor_annotation is None:
        candidate = args.factor_image.parent / "factor_annotation.csv"
        factor_annotation = candidate if candidate.exists() else None
    gene_factor_scores = load_gene_factor_scores(factor_annotation, n_factors)
    rows: List[Dict[str, object]] = []
    runs = find_runs(args.candidate_root)
    if not runs:
        raise SystemExit("No candidate_report.json files found.")
    for run_dir in runs:
        print(f"Ranking {run_dir}")
        rows.extend(
            summarize_one_run(
                run_dir,
                args.output_dir,
                factor_image,
                gene_factor_scores,
                args.top_ks,
                args.max_overlap,
                save_panels=not args.skip_panels,
            )
        )
    write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
