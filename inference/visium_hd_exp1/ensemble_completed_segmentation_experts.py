#!/usr/bin/env python3
"""Fuse completed Visium HD Exp1 segmentation experts.

The preceding sweeps show a class split: superpixels are strongest for
bronchiola and small factor-heavy classes, molecular SAM ranking is strongest
for alveoli/vessels, and gene-prior/SAM fusion is strongest for tumor/stroma.
This script treats those outputs as candidate experts and searches simple
label-wise consensus rules on the single annotated calibration sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HE = PROJECT_ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/sam3_local_region_summary/region_summary.json"
DEFAULT_FACTOR = PROJECT_ROOT / "output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/ficture_factor_label_image_he.npy"
DEFAULT_ANNOT = PROJECT_ROOT / "output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results/visium_hd_exp1"

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

COLORS = {
    "lung_bronchiola": (31, 119, 180),
    "erythorocytes": (255, 127, 14),
    "immune_infiltration": (44, 160, 44),
    "lung_alveoli_normal_adjacent": (148, 103, 189),
    "lung_vessels": (140, 86, 75),
    "pigment": (127, 127, 127),
    "stroma": (188, 189, 34),
    "tumor": (23, 190, 207),
}

NONOVERLAP_PRIORITY = [
    "pigment",
    "erythorocytes",
    "immune_infiltration",
    "lung_bronchiola",
    "lung_vessels",
    "lung_alveoli_normal_adjacent",
    "tumor",
    "stroma",
]


@dataclass
class ExpertMask:
    label: str
    method: str
    source: str
    variant: str
    mask: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.method}:{self.source}:{self.variant}"


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def max_side_shape(shape_hw: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    h, w = shape_hw
    scale = min(1.0, float(max_side) / max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def resize_image(image: np.ndarray, shape_hw: Tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return np.array(Image.fromarray(image).resize((w, h), resample))


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


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 127


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(radius * 2 + 1))) > 127


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


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
    out = {}
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        out[slug] = resize_bool(read_mask(resolve_path(item["mask_path"], summary_path.parent)), shape_hw)
    return out


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


def prior_score_image(factors: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = np.zeros(factors.shape, dtype=np.float32)
    valid = factors >= 0
    out[valid] = scores[factors[valid]]
    return out


def index_candidate_runs(results_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for report in (results_root / "sam3_iamsam_retrial").glob("**/candidate_report.json"):
        out[report.parent.name] = report.parent
    return out


def selected_indices(row: pd.Series) -> List[int]:
    text = str(row.get("selected_candidate_indices", "") or "")
    if not text or text.lower() == "nan":
        return []
    return [int(x) for x in re.split(r"[;,]", text) if x.strip().isdigit()]


def load_candidate_masks(
    run_dir: Path,
    shape_hw: Tuple[int, int],
    cache: Dict[str, List[np.ndarray]],
) -> List[np.ndarray]:
    key = str(run_dir)
    if key in cache:
        return cache[key]
    masks = []
    for path in sorted((run_dir / "candidate_masks").glob("candidate_*.png")):
        masks.append(resize_bool(read_mask(path), shape_hw))
    cache[key] = masks
    return masks


def parse_dilate_from_source(source: str, default: int = 12) -> int:
    match = re.search(r"_d(\d+)_", source)
    return int(match.group(1)) if match else default


def collect_superpixel(results_root: Path, shape_hw: Tuple[int, int]) -> List[ExpertMask]:
    experts: List[ExpertMask] = []
    for mask_path in sorted((results_root / "superpixel_gene_he").glob("*/masks/*_superpixel_mask.png")):
        label = mask_path.name.replace("_superpixel_mask.png", "")
        if label not in LABEL_ORDER:
            continue
        experts.append(
            ExpertMask(
                label=label,
                method="superpixel",
                source=mask_path.parents[1].name,
                variant="mask",
                mask=resize_bool(read_mask(mask_path), shape_hw),
            )
        )
    return experts


def collect_hed_micro(results_root: Path, shape_hw: Tuple[int, int]) -> List[ExpertMask]:
    experts: List[ExpertMask] = []
    for mask_path in sorted((results_root / "hed_micro_sweep").glob("*/masks/*_hed_micro_best.png")):
        label = mask_path.name.replace("_hed_micro_best.png", "")
        if label not in LABEL_ORDER:
            continue
        experts.append(
            ExpertMask(
                label=label,
                method="hed_micro",
                source=mask_path.parents[1].name,
                variant="mask",
                mask=resize_bool(read_mask(mask_path), shape_hw),
            )
        )
    return experts


def collect_micro_classifier(results_root: Path, shape_hw: Tuple[int, int]) -> List[ExpertMask]:
    experts: List[ExpertMask] = []
    for mask_path in sorted((results_root / "micro_classifier_sweep").glob("*/masks/*_micro_classifier_best.png")):
        label = mask_path.name.replace("_micro_classifier_best.png", "")
        if label not in LABEL_ORDER:
            continue
        experts.append(
            ExpertMask(
                label=label,
                method="micro_classifier",
                source=mask_path.parents[1].name,
                variant="mask",
                mask=resize_bool(read_mask(mask_path), shape_hw),
            )
        )
    return experts


def collect_molecular_rank(
    results_root: Path,
    shape_hw: Tuple[int, int],
    run_index: Dict[str, Path],
    mask_cache: Dict[str, List[np.ndarray]],
) -> List[ExpertMask]:
    experts: List[ExpertMask] = []
    for csv_path in sorted((results_root / "molecular_sam3_ranking").glob("*/molecular_sam3_ranked_best_by_run_label.csv")):
        source = csv_path.parent.name
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            label = str(row["slug"] if "slug" in row else slugify(row["label"]))
            run = str(row["run"])
            if label not in LABEL_ORDER or run not in run_index:
                continue
            masks = load_candidate_masks(run_index[run], shape_hw, mask_cache)
            chosen = [masks[i] for i in selected_indices(row) if i < len(masks)]
            if not chosen:
                continue
            pred = np.logical_or.reduce(chosen)
            experts.append(ExpertMask(label, "molecular_rank", source, f"{run}:{row['variant']}:top{row['top_k']}", pred))
    return experts


def collect_gene_fusion(
    results_root: Path,
    shape_hw: Tuple[int, int],
    run_index: Dict[str, Path],
    factors_full: np.ndarray,
    gene_scores: Dict[str, np.ndarray],
    mask_cache: Dict[str, List[np.ndarray]],
) -> List[ExpertMask]:
    experts: List[ExpertMask] = []
    factors = resize_labels(factors_full, shape_hw)
    for csv_path in sorted((results_root / "gene_prior_sam3_fusion").glob("*/gene_prior_sam3_fusion_best_by_run_label.csv")):
        source = csv_path.parent.name
        prior_dilate = parse_dilate_from_source(source)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            label = str(row["slug"] if "slug" in row else slugify(row["label"]))
            run = str(row["run"])
            if label not in LABEL_ORDER or label not in gene_scores or run not in run_index:
                continue
            masks = load_candidate_masks(run_index[run], shape_hw, mask_cache)
            prior = prior_score_image(factors, gene_scores[label]) >= float(row["threshold"])
            prior_context = dilate(prior, prior_dilate)
            variant = str(row["variant"])
            if variant == "gene_prior_only":
                pred = prior
            elif variant == "all_sam_prior_context_clip":
                pred = np.logical_and(np.logical_or.reduce(masks), prior_context) if masks else np.zeros(shape_hw, bool)
            else:
                chosen = [masks[i] for i in selected_indices(row) if i < len(masks)]
                union = np.logical_or.reduce(chosen) if chosen else np.zeros(shape_hw, bool)
                if variant == "sam_greedy_prior_context_clip":
                    pred = np.logical_and(union, prior_context)
                elif variant == "sam_greedy_prior_strict_clip":
                    pred = np.logical_and(union, prior)
                else:
                    pred = union
            experts.append(ExpertMask(label, "gene_fusion", source, f"{run}:{variant}:thr{float(row['threshold']):.2f}", pred))
    return experts


def consensus_candidates(experts: Sequence[ExpertMask], target: np.ndarray) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    scored = [(expert, metrics(expert.mask, target)) for expert in experts]
    scored.sort(key=lambda item: item[1]["dice"], reverse=True)
    candidates: List[Tuple[str, np.ndarray, Dict[str, float]]] = []
    for expert, score in scored[:30]:
        candidates.append((f"single|{expert.name}", expert.mask, score))
    for k in [2, 3, 4, 5, 8, 12]:
        top = [expert.mask for expert, _ in scored[:k]]
        if len(top) < 2:
            continue
        stack = np.stack(top, axis=0).astype(np.float32)
        for threshold in [1, 2, max(2, k // 2), k]:
            threshold = min(threshold, k)
            pred = stack.sum(axis=0) >= threshold
            candidates.append((f"vote_top{k}_thr{threshold}", pred, metrics(pred, target)))
        weighted = np.zeros_like(stack[0])
        denom = 0.0
        for mask, (_, score) in zip(top, scored[:k]):
            weight = max(float(score["dice"]), 1e-3)
            weighted += mask * weight
            denom += weight
        for thr in [0.25, 0.35, 0.45, 0.55, 0.65]:
            pred = (weighted / max(denom, 1e-6)) >= thr
            candidates.append((f"weighted_top{k}_thr{thr:.2f}", pred, metrics(pred, target)))
    more: List[Tuple[str, np.ndarray, Dict[str, float]]] = []
    for name, pred, _ in candidates:
        for radius in [1, 2, 4]:
            closed = close_mask(pred, radius)
            more.append((f"{name}|close{radius}", closed, metrics(closed, target)))
    candidates.extend(more)
    return candidates


def make_overlay(image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    out = image.astype(np.float32).copy()
    for label in LABEL_ORDER:
        mask = masks.get(label)
        if mask is None:
            continue
        color = np.array(COLORS[label], dtype=np.float32)
        out[mask] = out[mask] * 0.52 + color * 0.48
    return np.clip(out, 0, 255).astype(np.uint8)


def enforce_non_overlap(chosen: Dict[str, Tuple[str, np.ndarray, Dict[str, float]]]) -> Dict[str, np.ndarray]:
    claimed = np.zeros(next(iter(chosen.values()))[1].shape, dtype=bool)
    out: Dict[str, np.ndarray] = {}
    priority = [label for label in NONOVERLAP_PRIORITY if label in chosen]
    priority.extend(label for label in chosen if label not in priority)
    for label in priority:
        pred = chosen[label][1] & ~claimed
        out[label] = pred
        claimed |= pred
    return out


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape_hw = max_side_shape(image_full.shape[:2], args.max_side)
    image = resize_image(image_full, shape_hw, Image.Resampling.BILINEAR)
    targets = load_targets(args.summary_path, shape_hw)
    factors = np.load(args.factor_image)
    n_factors = int(factors[factors >= 0].max()) + 1 if np.any(factors >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    run_index = index_candidate_runs(args.results_root)

    experts = []
    experts.extend(collect_superpixel(args.results_root, shape_hw))
    experts.extend(collect_hed_micro(args.results_root, shape_hw))
    experts.extend(collect_micro_classifier(args.results_root, shape_hw))
    print(f"Loaded {len(experts)} superpixel/HED/micro-classifier experts", flush=True)
    mask_cache: Dict[str, List[np.ndarray]] = {}
    experts.extend(collect_molecular_rank(args.results_root, shape_hw, run_index, mask_cache))
    print(f"Loaded {len(experts)} experts after molecular rank; cached {len(mask_cache)} SAM runs", flush=True)
    experts.extend(collect_gene_fusion(args.results_root, shape_hw, run_index, factors, gene_scores, mask_cache))
    print(f"Loaded {len(experts)} total experts; cached {len(mask_cache)} SAM runs", flush=True)

    rows = []
    chosen: Dict[str, Tuple[str, np.ndarray, Dict[str, float]]] = {}
    for label in LABEL_ORDER:
        label_experts = [expert for expert in experts if expert.label == label]
        if not label_experts or label not in targets:
            continue
        candidates = consensus_candidates(label_experts, targets[label])
        candidates.sort(key=lambda item: item[2]["dice"], reverse=True)
        chosen[label] = candidates[0]
        for rank, (name, _, score) in enumerate(candidates[:50], start=1):
            rows.append({"label": label, "rank": rank, "candidate": name, **score})

    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    non_overlap = enforce_non_overlap(chosen)
    final_rows = []
    for label in LABEL_ORDER:
        if label not in chosen:
            continue
        name, pred, score = chosen[label]
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / f"{label}_best_ensemble_mask.png")
        Image.fromarray(non_overlap[label].astype(np.uint8) * 255).save(mask_dir / f"{label}_nonoverlap_ensemble_mask.png")
        final_rows.append({"label": label, "candidate": name, **score, **{f"nonoverlap_{k}": v for k, v in metrics(non_overlap[label], targets[label]).items()}})
    pd.DataFrame(rows).to_csv(args.output_dir / "ensemble_candidate_scores.csv", index=False)
    final = pd.DataFrame(final_rows)
    final.to_csv(args.output_dir / "ensemble_best_by_label.csv", index=False)
    Image.fromarray(make_overlay(image, {label: chosen[label][1] for label in chosen})).save(args.output_dir / "ensemble_overlay.png")
    Image.fromarray(make_overlay(image, non_overlap)).save(args.output_dir / "ensemble_nonoverlap_overlay.png")

    lines = ["# Completed Expert Ensemble", ""]
    lines.append(f"Experts discovered: {len(experts)}")
    lines += ["", "| label | Dice | non-overlap Dice | candidate |", "|---|---:|---:|---|"]
    for _, row in final.sort_values("dice", ascending=False).iterrows():
        lines.append(f"| {row['label']} | {row['dice']:.3f} | {row['nonoverlap_dice']:.3f} | {row['candidate']} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(final.sort_values("dice", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse completed Visium HD Exp1 segmentation experts.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--factor-image", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--factor-annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
