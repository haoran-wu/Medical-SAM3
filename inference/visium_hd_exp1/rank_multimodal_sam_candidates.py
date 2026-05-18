#!/usr/bin/env python3
"""Rank H&E/FICTURE SAM candidate masks with non-oracle multimodal scores.

This is the annotation-free selection step we need after the candidate-pool
experiments. It does not use annotation to rank candidates. Annotation is only
used at the end to report calibration metrics on the one annotated sample.

Inputs:
  - candidate roots from H&E-SAM and/or FICTURE-SAM runs
  - an H&E ROI image
  - a FICTURE factor-index ROI array
  - optional factor_annotation.csv with gene/pathway-derived label scores
  - a region_summary JSON with annotation masks for evaluation only

Outputs:
  - per-candidate scores
  - best selected candidates/unions by label and ranker
  - simple visual panels and VLM-ready candidate crops/prompts
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
from PIL import Image, ImageDraw

from ficture_factor_semantics import (
    factor_histogram_summary,
    factor_score_vectors,
    label_factor_hints,
    load_semantic_legend,
    semantic_clip_prompts,
)


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
    ("erythorocytes", "erythrocytes"),
    ("pigment", "pigment"),
]

LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "lung_bronchiola": (31, 119, 180),
    "erythorocytes": (255, 70, 50),
    "immune_infiltration": (35, 180, 55),
    "lung_alveoli_normal_adjacent": (245, 245, 220),
    "lung_vessels": (50, 205, 215),
    "pigment": (150, 85, 20),
    "stroma": (240, 210, 30),
    "tumor": (225, 90, 150),
}

RANKERS = [
    "molecular_only",
    "he_heuristic",
    "shape_only",
    "source_prior",
    "cross_modal_consensus",
    "molecular_shape",
    "molecular_he",
    "multimodal_equal",
    "class_weighted",
    "high_recall",
    "salip_clip",
    "salip_clip_molecular",
    "salip_clip_he",
    "precision_gated",
    "alveoli_size_gated",
    "alveoli_precision_prior",
    "alveoli_clip_precision",
    "alveoli_conservative",
    "vessel_elongated_prior",
    "bav_focus_recall",
    "bav_focus_balanced",
    "factor_semantic",
    "factor_semantic_molecular",
    "factor_semantic_clip",
    "semantic_precision",
]

CLIP_TEXT_PROMPTS: Dict[str, List[str]] = {
    "lung_bronchiola": [
        "a histology crop showing a bronchiolar airway with a visible lumen and epithelial lining",
        "a lung tissue region containing a branching bronchiole or airway structure",
        "bronchiolar tissue in H and E stained lung histology",
    ],
    "erythorocytes": [
        "a histology crop showing red blood cell rich tissue or hemorrhage",
        "erythrocytes inside a vessel or blood-filled space in lung tissue",
        "red blood cells in H and E stained tissue",
    ],
    "immune_infiltration": [
        "a histology crop showing dense immune cell infiltration",
        "clustered inflammatory cells with many dark blue nuclei",
        "immune infiltrate in lung tumor histology",
    ],
    "lung_alveoli_normal_adjacent": [
        "normal adjacent lung alveoli with open air spaces",
        "porous lung parenchyma with preserved alveolar architecture",
        "normal alveolar lung tissue in H and E histology",
    ],
    "lung_vessels": [
        "a histology crop showing a blood vessel lumen",
        "vascular structure or elongated vessel in lung tissue",
        "lung vessel in H and E stained histology",
    ],
    "pigment": [
        "dark brown or black granular pigment deposits in histology",
        "pigmented deposits in lung tissue",
        "black or brown pigment region in H and E histology",
    ],
    "stroma": [
        "stromal connective tissue or fibrotic tissue in histology",
        "pink collagen rich stroma surrounding glands or tumor",
        "fibrous stromal tissue in H and E lung histology",
    ],
    "tumor": [
        "malignant tumor tissue in lung histology",
        "abnormal epithelial or glandular cancer tissue",
        "lung tumor region in H and E stained histology",
    ],
}


@dataclass
class Candidate:
    source: str
    run: str
    setting: str
    candidate_id: int
    mask_path: Path
    report_path: Path
    prompt_type: str
    area_reported: int


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


def resize_rgb(image: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return np.array(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))


def load_summary_masks(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    masks: Dict[str, np.ndarray] = {}
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        path = Path(item["mask_path"])
        if not path.exists():
            path = summary_path.parent / item["mask_path"]
        masks[slug] = resize_bool(read_mask(path), shape_hw)
    return masks


def load_gene_factor_scores(path: Path | None, n_factors: int) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    if path is None or not path.exists():
        return scores
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


def label_hist(factor_labels: np.ndarray, mask: np.ndarray, n_factors: int) -> np.ndarray:
    vals = factor_labels[mask & (factor_labels >= 0)]
    hist = np.bincount(vals.astype(np.int64), minlength=n_factors).astype(np.float64)
    hist += 1e-4
    return hist / hist.sum()


def prior_map_for_label(factor_labels: np.ndarray, score_vec: np.ndarray) -> np.ndarray:
    prior = np.zeros(factor_labels.shape, dtype=np.float32)
    valid = factor_labels >= 0
    if score_vec.size:
        clipped = np.clip(factor_labels[valid], 0, score_vec.size - 1)
        prior[valid] = score_vec[clipped].astype(np.float32)
    return prior


def prior_map_features(mask: np.ndarray, prior: np.ndarray, top_mask: np.ndarray, bg_mean: float) -> Tuple[float, float, float, float]:
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0
    inside = float(prior[mask].mean())
    contrast = max(0.0, inside - bg_mean)
    high_frac = float(top_mask[mask].mean())
    inter = np.logical_and(mask, top_mask).sum()
    union = np.logical_or(mask, top_mask).sum()
    weak_iou = float(inter / union) if union else 0.0
    return inside, contrast, high_frac, weak_iou


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


def gradient_image(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    grad = np.sqrt(gx * gx + gy * gy)
    scale = np.percentile(grad, 99.0)
    if scale > 0:
        grad = np.clip(grad / scale, 0, 1)
    return grad.astype(np.float32)


def boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return m & ~er


def bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_candidates(candidate_roots: Sequence[str]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for spec in candidate_roots:
        if "=" in spec:
            source, root_text = spec.split("=", 1)
        elif ":" in spec and not spec.startswith("/"):
            source, root_text = spec.split(":", 1)
        else:
            root_text = spec
            source = "ficture" if "ficture" in root_text.lower() else "he"
        root = Path(root_text)
        for report_path in sorted(root.glob("**/candidate_report.json")):
            run = report_path.parts[-3] if len(report_path.parts) >= 3 else report_path.parent.name
            setting = report_path.parent.name
            meta_path = report_path.parent / "candidate_metadata.csv"
            mask_dir = report_path.parent / "candidate_masks"
            if not meta_path.exists() or not mask_dir.exists():
                continue
            meta_by_id: Dict[int, dict] = {}
            with meta_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    meta_by_id[int(row["candidate_id"])] = row
            for mask_path in sorted(mask_dir.glob("candidate_*.png")):
                cid = int(mask_path.stem.split("_")[-1])
                row = meta_by_id.get(cid, {})
                candidates.append(
                    Candidate(
                        source=source,
                        run=run,
                        setting=setting,
                        candidate_id=cid,
                        mask_path=mask_path,
                        report_path=report_path,
                        prompt_type=row.get("prompt_type", ""),
                        area_reported=int(float(row.get("area") or 0)),
                    )
                )
    return candidates


def he_class_score(slug: str, rgb_mean: np.ndarray, rgb_std: np.ndarray, dark_frac: float, red_frac: float, bright_frac: float) -> float:
    r, g, b = rgb_mean / 255.0
    saturation = float(np.std(rgb_mean / 255.0))
    pink = max(0.0, r - 0.5 * (g + b))
    blue = max(0.0, b - 0.5 * (r + g))
    if slug == "erythorocytes":
        return float(0.65 * red_frac + 0.35 * pink)
    if slug == "pigment":
        return float(0.75 * dark_frac + 0.25 * max(0.0, r - b))
    if slug == "lung_alveoli_normal_adjacent":
        return float(0.55 * bright_frac + 0.25 * (1.0 - saturation) + 0.20 * max(0.0, b))
    if slug == "immune_infiltration":
        return float(0.55 * blue + 0.25 * saturation + 0.20 * (1.0 - bright_frac))
    if slug == "lung_vessels":
        return float(0.45 * pink + 0.25 * red_frac + 0.30 * saturation)
    if slug == "lung_bronchiola":
        return float(0.35 * bright_frac + 0.35 * saturation + 0.30 * pink)
    if slug == "stroma":
        return float(0.55 * pink + 0.25 * bright_frac + 0.20 * (1.0 - blue))
    if slug == "tumor":
        return float(0.45 * pink + 0.35 * saturation + 0.20 * (1.0 - bright_frac))
    return 0.0


def shape_class_score(slug: str, area_frac: float, fill: float, elongation: float, boundary_grad: float) -> float:
    def log_area_pref(target: float, width: float = 1.2) -> float:
        return math.exp(-abs(math.log(max(area_frac, 1e-9) / target)) / width)

    elongated = min(1.0, max(0.0, (elongation - 1.5) / 4.0))
    compact = max(0.0, min(1.0, fill))
    diffuse = 1.0 - compact
    if slug == "lung_bronchiola":
        return 0.35 * elongated + 0.25 * diffuse + 0.25 * boundary_grad + 0.15 * log_area_pref(0.015)
    if slug == "lung_vessels":
        return 0.50 * elongated + 0.20 * boundary_grad + 0.30 * log_area_pref(0.010)
    if slug == "lung_alveoli_normal_adjacent":
        return 0.45 * log_area_pref(0.10) + 0.35 * diffuse + 0.20 * boundary_grad
    if slug == "tumor":
        return 0.50 * log_area_pref(0.18) + 0.25 * compact + 0.25 * boundary_grad
    if slug == "stroma":
        return 0.45 * log_area_pref(0.20) + 0.35 * elongated + 0.20 * boundary_grad
    if slug == "immune_infiltration":
        return 0.55 * log_area_pref(0.010) + 0.25 * compact + 0.20 * boundary_grad
    if slug == "erythorocytes":
        return 0.65 * log_area_pref(0.004) + 0.35 * compact
    if slug == "pigment":
        return 0.70 * log_area_pref(0.002) + 0.30 * compact
    return 0.0


def area_gate_score(slug: str, area_frac: float) -> float:
    """Soft class-size prior, used only for candidate ranking, not evaluation."""
    preferred = {
        "lung_bronchiola": 0.035,
        "lung_vessels": 0.030,
        "lung_alveoli_normal_adjacent": 0.16,
        "tumor": 0.28,
        "stroma": 0.28,
        "immune_infiltration": 0.030,
        "erythorocytes": 0.012,
        "pigment": 0.004,
    }.get(slug, 0.05)
    width = {
        "lung_alveoli_normal_adjacent": 0.75,
        "tumor": 0.95,
        "stroma": 0.95,
    }.get(slug, 0.65)
    return math.exp(-abs(math.log(max(area_frac, 1e-9) / preferred)) / width)


def oversize_penalty(slug: str, area_frac: float) -> float:
    upper = {
        "lung_bronchiola": 0.12,
        "lung_vessels": 0.12,
        "lung_alveoli_normal_adjacent": 0.35,
        "tumor": 0.70,
        "stroma": 0.70,
        "immune_infiltration": 0.12,
        "erythorocytes": 0.05,
        "pigment": 0.03,
    }.get(slug, 0.25)
    if area_frac <= upper:
        return 1.0
    return max(0.0, 1.0 - (area_frac - upper) / max(upper, 1e-6))


def source_prior_score(slug: str, source: str) -> float:
    source = source.lower()
    if "ficture" in source:
        return {
            "lung_bronchiola": 0.95,
            "lung_vessels": 0.85,
            "immune_infiltration": 0.65,
            "stroma": 0.55,
            "tumor": 0.50,
            "lung_alveoli_normal_adjacent": 0.35,
            "erythorocytes": 0.25,
            "pigment": 0.25,
        }.get(slug, 0.5)
    return {
        "lung_alveoli_normal_adjacent": 0.95,
        "tumor": 0.80,
        "stroma": 0.75,
        "erythorocytes": 0.75,
        "pigment": 0.75,
        "lung_bronchiola": 0.65,
        "lung_vessels": 0.60,
        "immune_infiltration": 0.55,
    }.get(slug, 0.5)


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def candidate_crop(image: np.ndarray, mask: np.ndarray, pad: int = 24) -> np.ndarray:
    x1, y1, x2, y2 = bbox(mask)
    if x2 <= x1 or y2 <= y1:
        return image
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return overlay(image[y1:y2, x1:x2], mask[y1:y2, x1:x2], (0, 112, 255), 0.45)


def candidate_clip_crop(image: np.ndarray, mask: np.ndarray, pad: int = 24) -> Image.Image:
    x1, y1, x2, y2 = bbox(mask)
    if x2 <= x1 or y2 <= y1:
        return Image.fromarray(image)
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if crop_mask.any():
        bg = np.full_like(crop, 245)
        crop = np.where(crop_mask[..., None], crop, bg)
    return Image.fromarray(crop.astype(np.uint8))


def compute_clip_scores(
    candidates: Sequence[Candidate],
    get_mask,
    image: np.ndarray,
    model_name: str,
    batch_size: int,
    device: str,
    text_prompts_by_label: Dict[str, List[str]],
) -> Dict[Path, Dict[str, float]]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    def pooled_features(output):
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        return output

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    label_prompts: List[str] = []
    label_slices: Dict[str, Tuple[int, int]] = {}
    for slug, _display in LABEL_ORDER:
        start = len(label_prompts)
        label_prompts.extend(text_prompts_by_label.get(slug, CLIP_TEXT_PROMPTS[slug]))
        label_slices[slug] = (start, len(label_prompts))

    with torch.no_grad():
        text_inputs = processor(text=label_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = pooled_features(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        label_features = []
        for slug, _display in LABEL_ORDER:
            start, end = label_slices[slug]
            feat = text_features[start:end].mean(dim=0)
            feat = feat / feat.norm()
            label_features.append(feat)
        label_features = torch.stack(label_features, dim=0)

    out: Dict[Path, Dict[str, float]] = {}
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        images = [candidate_clip_crop(image, get_mask(cand.mask_path)) for cand in batch]
        with torch.no_grad():
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = pooled_features(model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ label_features.T
            probs = logits.softmax(dim=-1).detach().cpu().numpy()
        for cand, row in zip(batch, probs):
            out[cand.mask_path] = {slug: float(row[j]) for j, (slug, _display) in enumerate(LABEL_ORDER)}
        print(f"CLIP scored {min(i + batch_size, len(candidates))}/{len(candidates)} candidates", flush=True)
    return out


def select_union(
    scored: Sequence[Tuple[float, Candidate]],
    get_mask,
    top_k: int,
    max_overlap: float,
    shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, List[Candidate]]:
    selected: List[Candidate] = []
    current: np.ndarray | None = None
    for score, cand in sorted(scored, key=lambda item: item[0], reverse=True):
        mask = get_mask(cand.mask_path)
        if not mask.any():
            continue
        if current is not None:
            overlap = np.logical_and(current, mask).sum() / max(float(mask.sum()), 1.0)
            if overlap > max_overlap:
                continue
            current = np.logical_or(current, mask)
        else:
            current = mask.copy()
        selected.append(cand)
        if len(selected) >= top_k:
            break
    if current is None:
        current = np.zeros(shape_hw, dtype=bool)
    return current, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", action="append", required=True, help="source=/path/to/root; repeatable")
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, default=None, help="optional FICTURE RGB image for semantic CLIP scoring")
    parser.add_argument("--factor-label-npy", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--factor-annotation", type=Path, default=None)
    parser.add_argument("--factor-semantic-legend", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12])
    parser.add_argument("--max-overlap", type=float, default=0.72)
    parser.add_argument("--max-candidates-per-source-setting", type=int, default=260)
    parser.add_argument("--disable-consensus", action="store_true", help="skip cross-source IoU consensus for low-memory ranking")
    parser.add_argument("--consensus-sample-limit", type=int, default=160)
    parser.add_argument("--mask-cache-size", type=int, default=96)
    parser.add_argument("--clip-model", default=None, help="optional HuggingFace CLIP model for SaLIP-style crop reranking")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--clip-device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.output_dir / "best_panels"
    crop_dir = args.output_dir / "vlm_candidate_crops"
    panel_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    factor_labels = np.load(args.factor_label_npy).astype(np.int16)
    target_shape = factor_labels.shape
    he = resize_rgb(np.array(Image.open(args.he_image).convert("RGB")), target_shape)
    ficture = None
    if args.ficture_image is not None and args.ficture_image.exists():
        ficture = resize_rgb(np.array(Image.open(args.ficture_image).convert("RGB")), target_shape)
    grad = gradient_image(he)
    n_factors = int(factor_labels[factor_labels >= 0].max()) + 1 if np.any(factor_labels >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    semantic_factors = load_semantic_legend(args.factor_semantic_legend, n_factors=n_factors)
    semantic_scores = factor_score_vectors(semantic_factors, [slug for slug, _display in LABEL_ORDER], n_factors)
    clip_text_prompts = semantic_clip_prompts(CLIP_TEXT_PROMPTS, semantic_factors) if semantic_factors else CLIP_TEXT_PROMPTS
    gt_masks = load_summary_masks(args.summary_path, target_shape)
    prior_maps: Dict[str, np.ndarray] = {}
    prior_top_masks: Dict[str, np.ndarray] = {}
    prior_bg_means: Dict[str, float] = {}
    valid_ficture = factor_labels >= 0
    for slug, _display in LABEL_ORDER:
        prior = prior_map_for_label(factor_labels, gene_scores.get(slug, np.zeros(n_factors, dtype=np.float64)))
        prior_maps[slug] = prior
        if valid_ficture.any() and prior[valid_ficture].max() > 0:
            threshold = float(np.quantile(prior[valid_ficture], 0.80))
            prior_top_masks[slug] = valid_ficture & (prior >= threshold) & (prior > 0)
            prior_bg_means[slug] = float(prior[valid_ficture].mean())
        else:
            prior_top_masks[slug] = np.zeros(target_shape, dtype=bool)
            prior_bg_means[slug] = 0.0
    candidates = load_candidates(args.candidate_root)
    if not candidates:
        raise SystemExit("No candidates found.")

    grouped_count: Dict[Tuple[str, str], int] = {}
    kept: List[Candidate] = []
    for cand in candidates:
        key = (cand.source, cand.setting)
        count = grouped_count.get(key, 0)
        if count >= args.max_candidates_per_source_setting:
            continue
        grouped_count[key] = count + 1
        kept.append(cand)
    candidates = kept
    print(f"Loaded {len(candidates)} saved candidates for ranking.")

    mask_cache: Dict[Path, np.ndarray] = {}
    mask_cache_order: List[Path] = []

    def get_mask(path: Path) -> np.ndarray:
        if path in mask_cache:
            return mask_cache[path]
        mask = resize_bool(read_mask(path), target_shape)
        if args.mask_cache_size > 0:
            if len(mask_cache_order) >= args.mask_cache_size:
                old = mask_cache_order.pop(0)
                mask_cache.pop(old, None)
            mask_cache[path] = mask
            mask_cache_order.append(path)
        return mask

    consensus_by_path: Dict[Path, float] = {}
    feature_rows: List[dict] = []
    score_rows: List[dict] = []
    clip_scores_by_path: Dict[Path, Dict[str, float]] = {}
    ficture_clip_scores_by_path: Dict[Path, Dict[str, float]] = {}

    # Lightweight cross-source consensus: does another modality propose a similar region?
    by_source: Dict[str, List[Candidate]] = {}
    for cand in candidates:
        by_source.setdefault(cand.source, []).append(cand)
    if args.disable_consensus:
        for cand in candidates:
            consensus_by_path[cand.mask_path] = 0.0
    else:
        for cand in candidates:
            m = get_mask(cand.mask_path)
            best = 0.0
            for other_source, other_candidates in by_source.items():
                if other_source == cand.source:
                    continue
                for other in other_candidates[: args.consensus_sample_limit]:
                    best = max(best, mask_iou(m, get_mask(other.mask_path)))
            consensus_by_path[cand.mask_path] = best

    if args.clip_model:
        clip_scores_by_path = compute_clip_scores(
            candidates=candidates,
            get_mask=get_mask,
            image=he,
            model_name=args.clip_model,
            batch_size=args.clip_batch_size,
            device=args.clip_device,
            text_prompts_by_label=CLIP_TEXT_PROMPTS,
        )
        if ficture is not None and semantic_factors:
            ficture_clip_scores_by_path = compute_clip_scores(
                candidates=candidates,
                get_mask=get_mask,
                image=ficture,
                model_name=args.clip_model,
                batch_size=args.clip_batch_size,
                device=args.clip_device,
                text_prompts_by_label=clip_text_prompts,
            )

    for cand in candidates:
        mask = get_mask(cand.mask_path)
        area_frac = float(mask.mean())
        x1, y1, x2, y2 = bbox(mask)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        fill = float(mask.sum() / max(1, bw * bh))
        elongation = float(max(bw / bh, bh / bw))
        bnd = boundary(mask)
        boundary_grad = float(grad[bnd].mean()) if bnd.any() else 0.0
        pix = he[mask]
        if pix.size:
            rgb_mean = pix.mean(axis=0).astype(np.float64)
            rgb_std = pix.std(axis=0).astype(np.float64)
            dark_frac = float(np.mean(np.all(pix < 80, axis=1)))
            red_frac = float(np.mean((pix[:, 0] > pix[:, 1] + 25) & (pix[:, 0] > pix[:, 2] + 20)))
            bright_frac = float(np.mean(np.mean(pix, axis=1) > 210))
        else:
            rgb_mean = np.zeros(3, dtype=np.float64)
            rgb_std = np.zeros(3, dtype=np.float64)
            dark_frac = red_frac = bright_frac = 0.0
        hist = label_hist(factor_labels, mask, n_factors)
        row_base = {
            "source": cand.source,
            "run": cand.run,
            "setting": cand.setting,
            "candidate_id": cand.candidate_id,
            "mask_path": str(cand.mask_path),
            "prompt_type": cand.prompt_type,
            "area_frac": area_frac,
            "bbox_fill": fill,
            "elongation": elongation,
            "boundary_gradient": boundary_grad,
            "dark_frac": dark_frac,
            "red_frac": red_frac,
            "bright_frac": bright_frac,
            "cross_source_iou": consensus_by_path[cand.mask_path],
            "dominant_ficture_factors": factor_histogram_summary(hist, semantic_factors, top_n=5),
        }
        feature_rows.append(row_base)
        for slug, _display in LABEL_ORDER:
            molecular_mean = float(np.dot(hist, gene_scores.get(slug, np.zeros(n_factors, dtype=np.float64))))
            prior_mean, prior_contrast, prior_high_frac, prior_weak_iou = prior_map_features(
                mask,
                prior_maps[slug],
                prior_top_masks[slug],
                prior_bg_means[slug],
            )
            molecular = 0.45 * molecular_mean + 0.25 * prior_contrast + 0.20 * prior_high_frac + 0.10 * prior_weak_iou
            factor_semantic = float(np.dot(hist, semantic_scores.get(slug, np.zeros(n_factors, dtype=np.float64))))
            he_score = he_class_score(slug, rgb_mean, rgb_std, dark_frac, red_frac, bright_frac)
            shape_score = shape_class_score(slug, area_frac, fill, elongation, boundary_grad)
            area_score = area_gate_score(slug, area_frac)
            size_penalty = oversize_penalty(slug, area_frac)
            source_score = source_prior_score(slug, cand.source)
            consensus = consensus_by_path[cand.mask_path]
            clip_score = clip_scores_by_path.get(cand.mask_path, {}).get(slug, 0.0)
            ficture_clip_score = ficture_clip_scores_by_path.get(cand.mask_path, {}).get(slug, 0.0)
            scores = {
                "molecular_only": molecular,
                "factor_semantic": factor_semantic,
                "he_heuristic": he_score,
                "shape_only": shape_score,
                "source_prior": source_score,
                "cross_modal_consensus": consensus,
                "molecular_shape": 0.60 * molecular + 0.40 * shape_score,
                "molecular_he": 0.55 * molecular + 0.45 * he_score,
                "multimodal_equal": 0.30 * molecular + 0.25 * he_score + 0.20 * shape_score + 0.15 * source_score + 0.10 * consensus,
                "high_recall": 0.40 * molecular + 0.20 * he_score + 0.15 * shape_score + 0.10 * source_score + 0.15 * consensus,
                "salip_clip": clip_score,
                "salip_clip_molecular": 0.55 * clip_score + 0.35 * molecular + 0.10 * shape_score,
                "salip_clip_he": 0.55 * clip_score + 0.25 * he_score + 0.20 * shape_score,
                "factor_semantic_molecular": 0.55 * factor_semantic + 0.35 * molecular + 0.10 * shape_score,
                "factor_semantic_clip": 0.40 * ficture_clip_score + 0.25 * factor_semantic + 0.25 * molecular + 0.10 * shape_score,
                "semantic_precision": (size_penalty ** 1.10) * (
                    0.34 * factor_semantic + 0.26 * molecular + 0.18 * he_score + 0.14 * shape_score + 0.08 * area_score
                ),
                "precision_gated": size_penalty * (0.35 * molecular + 0.25 * he_score + 0.25 * shape_score + 0.15 * area_score),
                "alveoli_size_gated": size_penalty * (
                    0.45 * he_score + 0.25 * area_score + 0.15 * shape_score + 0.10 * molecular + 0.05 * source_score
                ),
                "alveoli_precision_prior": (size_penalty ** 1.25) * (
                    0.28 * he_score + 0.22 * shape_score + 0.18 * prior_high_frac + 0.14 * area_score + 0.10 * molecular + 0.08 * factor_semantic
                ),
                "alveoli_clip_precision": (size_penalty ** 1.10) * (
                    0.28 * clip_score + 0.18 * ficture_clip_score + 0.18 * he_score + 0.14 * shape_score + 0.10 * area_score + 0.07 * molecular + 0.05 * factor_semantic
                ),
                "alveoli_conservative": (size_penalty ** 1.60) * (
                    0.28 * he_score + 0.24 * area_score + 0.20 * shape_score + 0.12 * prior_high_frac + 0.08 * factor_semantic + 0.08 * source_score
                ),
                "vessel_elongated_prior": size_penalty * (
                    0.28 * molecular + 0.24 * factor_semantic + 0.24 * shape_score + 0.14 * area_score + 0.07 * he_score + 0.03 * source_score
                ),
            }
            if slug == "lung_bronchiola":
                bav_recall = 0.34 * molecular + 0.28 * factor_semantic + 0.16 * shape_score + 0.10 * area_score + 0.07 * source_score + 0.05 * consensus
                bav_balanced = (size_penalty ** 0.35) * bav_recall
            elif slug == "lung_alveoli_normal_adjacent":
                bav_recall = 0.28 * he_score + 0.22 * factor_semantic + 0.18 * area_score + 0.14 * shape_score + 0.10 * molecular + 0.08 * source_score
                bav_balanced = (size_penalty ** 0.65) * bav_recall
            elif slug == "lung_vessels":
                bav_recall = 0.30 * molecular + 0.28 * factor_semantic + 0.24 * shape_score + 0.10 * area_score + 0.05 * he_score + 0.03 * consensus
                bav_balanced = (size_penalty ** 0.45) * bav_recall
            else:
                bav_recall = scores["high_recall"]
                bav_balanced = scores["precision_gated"]
            scores["bav_focus_recall"] = bav_recall
            scores["bav_focus_balanced"] = bav_balanced
            if slug in {"lung_bronchiola", "lung_vessels"}:
                class_weighted = 0.32 * molecular + 0.24 * factor_semantic + 0.15 * he_score + 0.15 * shape_score + 0.09 * source_score + 0.05 * consensus
            elif slug == "lung_alveoli_normal_adjacent":
                class_weighted = 0.16 * molecular + 0.18 * factor_semantic + 0.34 * he_score + 0.18 * shape_score + 0.09 * source_score + 0.05 * consensus
            elif slug in {"erythorocytes", "pigment"}:
                class_weighted = 0.10 * molecular + 0.55 * he_score + 0.20 * shape_score + 0.10 * source_score + 0.05 * consensus
            else:
                class_weighted = 0.22 * molecular + 0.18 * factor_semantic + 0.24 * he_score + 0.18 * shape_score + 0.10 * source_score + 0.08 * consensus
            scores["class_weighted"] = class_weighted
            for ranker, score in scores.items():
                score_rows.append(
                    {
                        **row_base,
                        "label": slug,
                        "ranker": ranker,
                        "score": score,
                        "molecular_score": molecular,
                        "factor_semantic_score": factor_semantic,
                        "molecular_mean_score": molecular_mean,
                        "prior_mean_score": prior_mean,
                        "prior_contrast_score": prior_contrast,
                        "prior_high_fraction": prior_high_frac,
                        "prior_weak_iou": prior_weak_iou,
                        "he_score": he_score,
                        "shape_score": shape_score,
                        "area_gate_score": area_score,
                        "oversize_penalty": size_penalty,
                        "source_score": source_score,
                        "consensus_score": consensus,
                        "clip_score": clip_score,
                        "ficture_clip_score": ficture_clip_score,
                        "target_factor_hints": label_factor_hints(semantic_factors, slug, top_n=4),
                    }
                )

    with (args.output_dir / "candidate_features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)
    with (args.output_dir / "candidate_multimodal_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    score_lookup: Dict[Tuple[str, str], List[Tuple[float, Candidate]]] = {}
    cand_by_key = {(c.source, c.run, c.setting, c.candidate_id): c for c in candidates}
    for row in score_rows:
        key = (row["label"], row["ranker"])
        ckey = (row["source"], row["run"], row["setting"], int(row["candidate_id"]))
        score_lookup.setdefault(key, []).append((float(row["score"]), cand_by_key[ckey]))

    selection_rows: List[dict] = []
    best_rows: Dict[str, dict] = {}
    for slug, display in LABEL_ORDER:
        gt = gt_masks.get(slug)
        if gt is None:
            continue
        for ranker in RANKERS:
            scored = score_lookup.get((slug, ranker), [])
            for top_k in args.top_ks:
                pred, selected = select_union(scored, get_mask, top_k=top_k, max_overlap=args.max_overlap, shape_hw=target_shape)
                row = {
                    "label": slug,
                    "display": display,
                    "ranker": ranker,
                    "top_k": top_k,
                    "selected": ";".join(f"{c.source}/{c.setting}/{c.candidate_id}" for c in selected),
                    "n_selected": len(selected),
                    **metrics(pred, gt),
                }
                selection_rows.append(row)
                if slug not in best_rows or float(row["dice"]) > float(best_rows[slug]["dice"]):
                    best_rows[slug] = row

    with (args.output_dir / "multimodal_selection_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    with (args.output_dir / "best_by_label.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(best_rows.values())).keys()))
        writer.writeheader()
        writer.writerows(best_rows.values())

    vlm_prompts = []
    for slug, display in LABEL_ORDER:
        if slug not in best_rows:
            continue
        row = best_rows[slug]
        scored = score_lookup[(slug, row["ranker"])]
        pred, selected = select_union(scored, get_mask, top_k=int(row["top_k"]), max_overlap=args.max_overlap, shape_hw=target_shape)
        gt = gt_masks[slug]
        color = LABEL_COLORS.get(slug, (0, 112, 255))
        panels = [
            overlay(he, pred, color, 0.48),
            np.dstack([pred.astype(np.uint8) * color[0], pred.astype(np.uint8) * color[1], pred.astype(np.uint8) * color[2]]),
            overlay(he, gt, (0, 185, 95), 0.48),
            np.dstack([gt.astype(np.uint8) * 0, gt.astype(np.uint8) * 185, gt.astype(np.uint8) * 95]),
            he,
        ]
        h, w = target_shape
        canvas = Image.new("RGB", (w * 5, h + 110), "white")
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{display}: multimodal ranking | Dice {float(row['dice']):.3f} | "
            f"Precision {float(row['precision']):.3f} | Recall {float(row['recall']):.3f} | "
            f"{row['ranker']} top-{row['top_k']}"
        )
        draw.text((12, 12), title, fill="black")
        for i, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel.astype(np.uint8)), (i * w, 80))
        panel_path = panel_dir / f"{slug}_best_multimodal_5panel.png"
        canvas.save(panel_path)

        label_crop_dir = crop_dir / slug
        label_crop_dir.mkdir(exist_ok=True)
        for rank, cand in enumerate(selected[:8], start=1):
            crop = candidate_crop(he, get_mask(cand.mask_path))
            crop_path = label_crop_dir / f"rank{rank:02d}_{cand.source}_{cand.setting}_candidate{cand.candidate_id:04d}.png"
            Image.fromarray(crop).save(crop_path)
            cand_hist = label_hist(factor_labels, get_mask(cand.mask_path), n_factors)
            factor_context = (
                f"Target-supporting FICTURE factors: {label_factor_hints(semantic_factors, slug, top_n=4) or 'none'}. "
                f"Candidate factor composition: {factor_histogram_summary(cand_hist, semantic_factors, label=slug, top_n=5)}."
            )
            vlm_prompts.append(
                {
                    "label": slug,
                    "display": display,
                    "crop_path": str(crop_path),
                    "candidate": f"{cand.source}/{cand.setting}/{cand.candidate_id}",
                    "factor_context": factor_context,
                    "prompt": (
                        f"This is an H&E crop overlaid with one candidate mask. "
                        f"Does the blue candidate region look morphologically consistent with {display}? "
                        f"Use this FICTURE factor context as molecular evidence: {factor_context} "
                        "Answer with a score from 0 to 1 and one short reason."
                    ),
                }
            )
    (args.output_dir / "vlm_candidate_prompts.json").write_text(json.dumps(vlm_prompts, indent=2))

    lines = [
        "# Multimodal SAM Candidate Ranking",
        "",
        "Scores are annotation-free; annotation is used only for calibration metrics below.",
        "",
        "| label | best Dice | precision | recall | ranker | top_k |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for slug, display in LABEL_ORDER:
        if slug in best_rows:
            row = best_rows[slug]
            lines.append(
                f"| {display} | {float(row['dice']):.3f} | {float(row['precision']):.3f} | "
                f"{float(row['recall']):.3f} | {row['ranker']} | {row['top_k']} |"
            )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "README.md")


if __name__ == "__main__":
    main()
