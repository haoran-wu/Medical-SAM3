#!/usr/bin/env python3
"""Build the Jun09 skill-based component ranker report.

This is a local, reproducible ablation line.  It uses the existing 167-piece
compact pool first, computes interpretable features, writes VLM-shaped score
outputs, assembles selected pieces, and renders a collaborator-facing report.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage import measure


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
CLEAN_ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/VisiumHD-Segmentation-MaskSelection")

POOL_SOURCE = CLEAN_ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs/compact167_reverse_blur_pool"
LOCAL_CLUSTER_MASKS = (
    CLEAN_ROOT
    / "output/visium_hd_exp1/final_deliverables/Jun03_medpt24_clustered_iou_component_aware_union"
    / "cluster_component_scores/iou_0.90/cluster_masks"
)
OUT_BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"

SUMMARY_OFFICIAL = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
SEMANTIC_LEGEND = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json"
PROMPT_LEGEND_CSV = ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_prompt_legend_from_html.csv"
FACTOR_INDEX = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_factor_index.npy"
ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"

CLIP_LARGE_HE = (
    CLEAN_ROOT
    / "outputs/June6_Clip_FICTURE_HE_Semantic_Latent_Ranker/model_openai_clip-vit-large-patch14"
    / "score_outputs/HE_CLIP_rank/per_candidate_predictions.csv"
)
FICTURE_MARKER_PRIOR = (
    CLEAN_ROOT
    / "outputs/June6_Clip_FICTURE_HE_Semantic_Latent_Ranker"
    / "marker_prior_score_outputs/FICTURE_composition_prior_rankcalibrated/per_candidate_predictions.csv"
)
ZERO_SHOT_VLM = (
    ROOT
    / "output/visium_hd_exp1/final_deliverables/Jun08_piece_top1_candidate_scores"
    / "all_candidate_predicted_scores_qwen3vl32b_internvl35_38b.csv"
)

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
CLASS_DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}
LONG_TO_CLASS = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "immune infiltration": "immune_infiltration",
}
CLASS_TO_ANNOTATION = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}
SEMANTIC_CLASS_KEYS = {
    "bronchiola": "lung_bronchiola",
    "alveoli": "lung_alveoli_normal_adjacent",
    "vessels": "lung_vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_key(row: dict[str, str]) -> str:
    if row.get("row_key"):
        return row["row_key"]
    return "__".join(
        [
            row.get("target_label") or row.get("label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            str(row.get("candidate_id", "")),
        ]
    )


def class_key(value: str) -> str:
    return LONG_TO_CLASS.get(value, value)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return np.array(img) > 0


def load_mask_resized(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return np.array(img) > 0


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    return {
        "precision": tp / pred_area if pred_area else 0.0,
        "recall": tp / gt_area if gt_area else 0.0,
        "dice": (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0,
    }


def verify_data_gate() -> dict[str, object]:
    summary = json.loads(SUMMARY_OFFICIAL.read_text())
    he_size = Image.open(HE_ROI).size
    ficture_size = Image.open(FICTURE_ROI).size
    factor = np.load(FACTOR_INDEX, mmap_mode="r")
    status = {
        "official_status": summary.get("status"),
        "official_crop_bbox": summary.get("crop_bbox_xyxy"),
        "he_roi_size": f"{he_size[0]} x {he_size[1]}",
        "ficture_roi_size": f"{ficture_size[0]} x {ficture_size[1]}",
        "factor_index_shape": str(tuple(factor.shape)),
        "pool_source": str(POOL_SOURCE),
    }
    if summary.get("status") != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE gate failed: {summary.get('status')}")
    if he_size != (3144, 3327) or ficture_size != (3144, 3327):
        raise SystemExit(f"Unexpected ROI size: H&E {he_size}, FICTURE {ficture_size}")
    if tuple(factor.shape) != (3327, 3144):
        raise SystemExit(f"Unexpected factor index shape: {factor.shape}")
    return status


def corrected_mask_path(candidate_uid: str, old_path: str) -> Path:
    local = LOCAL_CLUSTER_MASKS / f"{candidate_uid}.png"
    if local.exists():
        return local
    fallback = Path(old_path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No local mask for {candidate_uid}: {local}")


def build_corrected_pool() -> tuple[Path, list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    public_rows = read_csv(POOL_SOURCE / "public_vlm_requests.csv")
    hidden_rows = read_csv(POOL_SOURCE / "hidden_candidate_truth.csv")
    if len(public_rows) != 167 or len(hidden_rows) != 167:
        raise SystemExit(f"Expected 167 public/hidden rows, got {len(public_rows)} / {len(hidden_rows)}")
    crop_out = OUT_BASE / "corrected_pool/candidate_pair_crops"
    crop_out.mkdir(parents=True, exist_ok=True)
    corrected_public: list[dict[str, str]] = []
    corrected_hidden: list[dict[str, str]] = []
    mask_by_key: dict[str, str] = {}
    for pub, hid in zip(public_rows, hidden_rows):
        uid = pub["candidate_uid"]
        mask_path = corrected_mask_path(uid, pub.get("mask_path") or hid.get("mask_path", ""))
        key = row_key(pub)
        mask_by_key[key] = str(mask_path)
        pub = dict(pub)
        hid = dict(hid)
        pub["mask_path"] = str(mask_path)
        hid["mask_path"] = str(mask_path)
        for col in ["he_crop_rel", "ficture_crop_rel"]:
            if pub.get(col):
                src = POOL_SOURCE / pub[col]
                dst = crop_out / src.name
                if src.exists() and not dst.exists():
                    shutil.copyfile(src, dst)
                pub[col] = str(Path("candidate_pair_crops") / dst.name)
                hid[col] = pub[col]
        corrected_public.append(pub)
        corrected_hidden.append(hid)
    pool_dir = OUT_BASE / "corrected_pool"
    write_csv(pool_dir / "public_vlm_requests.csv", corrected_public)
    write_csv(pool_dir / "hidden_candidate_truth.csv", corrected_hidden)
    return pool_dir, corrected_public, corrected_hidden, mask_by_key


def load_legend() -> tuple[dict[int, dict[str, str]], dict[str, np.ndarray]]:
    prompt_rows = read_csv(PROMPT_LEGEND_CSV)
    prompt_legend: dict[int, dict[str, str]] = {}
    for row in prompt_rows:
        factor = int(row["Factor"])
        prompt_legend[factor] = {
            "rgb": row.get("RGB", ""),
            "major": row.get("Major Compartment", ""),
            "cell_type": row.get("cell type", ""),
        }
    semantic = json.loads(SEMANTIC_LEGEND.read_text())
    n_factor = max(int(item["factor"]) for item in semantic["factors"]) + 1
    vectors = {key: np.zeros(n_factor, dtype=float) for key in CLASS_KEYS}
    for item in semantic["factors"]:
        factor = int(item["factor"])
        scores = item.get("target_scores") or {}
        for c in CLASS_KEYS:
            vectors[c][factor] = float(scores.get(SEMANTIC_CLASS_KEYS[c], 0.0))
    return prompt_legend, vectors


def factor_group_scores(hist: np.ndarray, prompt_legend: dict[int, dict[str, str]]) -> dict[str, float]:
    groups = {
        "frac_airway_epithelial": 0.0,
        "frac_endothelial": 0.0,
        "frac_immune": 0.0,
        "frac_stroma": 0.0,
        "frac_epithelial_tumor": 0.0,
        "frac_at2": 0.0,
    }
    for factor, frac in enumerate(hist):
        info = prompt_legend.get(factor, {})
        text = f"{info.get('major','')} {info.get('cell_type','')}".lower()
        if "airway" in text or "club" in text or "goblet" in text or "ciliated" in text:
            groups["frac_airway_epithelial"] += float(frac)
        if "endothelial" in text:
            groups["frac_endothelial"] += float(frac)
        if "immune" in text or "lymphocyte" in text or "macrophage" in text or "plasma" in text:
            groups["frac_immune"] += float(frac)
        if "stromal" in text or "fibroblast" in text or "smooth muscle" in text:
            groups["frac_stroma"] += float(frac)
        if "tumor" in text or "adenocarcinoma" in text or "malignant" in text:
            groups["frac_epithelial_tumor"] += float(frac)
        if "at2" in text or "alveolar type ii" in text:
            groups["frac_at2"] += float(frac)
    return groups


def composition_summary(hist: np.ndarray, prompt_legend: dict[int, dict[str, str]], top_n: int = 5) -> str:
    parts: list[str] = []
    for factor in np.argsort(hist)[::-1][:top_n]:
        frac = float(hist[factor])
        if frac <= 0:
            continue
        info = prompt_legend.get(int(factor), {})
        parts.append(
            f"Color {factor}: {frac:.3f}; RGB {info.get('rgb','')}; "
            f"compartment: {info.get('major','')}; cell type: {info.get('cell_type','')}"
        )
    return " | ".join(parts)


def percentile_scores(raw_rows: list[dict[str, object]], keys: list[str], prefix: str) -> dict[str, dict[str, int]]:
    values = {key: [float(row[f"{prefix}_{key}_raw"]) for row in raw_rows] for key in keys}
    out: dict[str, dict[str, int]] = {}
    for row in raw_rows:
        scores: dict[str, int] = {}
        for key in keys:
            vals = values[key]
            value = float(row[f"{prefix}_{key}_raw"])
            below = sum(1 for x in vals if x < value)
            equal = sum(1 for x in vals if x == value)
            scores[key] = int(round(100 * (below + 0.5 * equal) / max(1, len(vals))))
        out[str(row["row_key"])] = scores
    return out


def compute_features(public_rows: list[dict[str, str]], hidden_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    size = Image.open(HE_ROI).size
    factor_index = np.load(FACTOR_INDEX)
    prompt_legend, semantic_vectors = load_legend()
    mask_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for pub, hid in zip(public_rows, hidden_rows):
        key = row_key(pub)
        mask_path = Path(pub["mask_path"])
        if str(mask_path) not in mask_cache:
            mask_cache[str(mask_path)] = load_mask(mask_path, size)
        mask = mask_cache[str(mask_path)]
        area = int(mask.sum())
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        if props:
            largest = max(props, key=lambda p: p.area)
            minr, minc, maxr, maxc = largest.bbox
            bbox_h = maxr - minr
            bbox_w = maxc - minc
            solidity = float(largest.solidity or 0.0)
            eccentricity = float(largest.eccentricity or 0.0)
            perimeter = float(largest.perimeter or 0.0)
            centroid_y, centroid_x = largest.centroid
        else:
            bbox_h = bbox_w = 0
            solidity = eccentricity = perimeter = centroid_x = centroid_y = 0.0
        inside_vals = factor_index[mask]
        valid_inside = inside_vals[inside_vals >= 0]
        hist = np.bincount(valid_inside.astype(int), minlength=12).astype(float)
        hist = hist / hist.sum() if hist.sum() else hist
        ring = ndi.binary_dilation(mask, iterations=18) & ~mask
        ring_vals = factor_index[ring]
        valid_ring = ring_vals[ring_vals >= 0]
        ring_hist = np.bincount(valid_ring.astype(int), minlength=12).astype(float)
        ring_hist = ring_hist / ring_hist.sum() if ring_hist.sum() else ring_hist
        ficture_raw = {
            f"ficture_inside_{c}_raw": float(hist[: len(vec)] @ vec)
            for c, vec in semantic_vectors.items()
        }
        ficture_ring_raw = {
            f"ficture_ring_{c}_raw": float(ring_hist[: len(vec)] @ vec)
            for c, vec in semantic_vectors.items()
        }
        shape_raw = {
            "shape_bronchiola_raw": 0.42 * eccentricity + 0.20 * (1 - solidity) + 0.18 * math.log1p(area) / 16.0,
            "shape_alveoli_raw": 0.36 * (1 - solidity) + 0.30 * math.log1p(area) / 16.0 + 0.10 * (1 - eccentricity),
            "shape_vessels_raw": 0.48 * eccentricity + 0.25 * (1 - solidity) + 0.12 * min(1.0, area / 120000.0),
            "shape_tumor_raw": 0.50 * solidity + 0.35 * math.log1p(area) / 16.0,
            "shape_stroma_raw": 0.40 * math.log1p(area) / 16.0 + 0.28 * eccentricity + 0.15 * solidity,
            "shape_immune_infiltration_raw": 0.45 * min(1.0, area / 60000.0) + 0.25 * (1 - solidity) + 0.15 * perimeter / max(1.0, area),
        }
        group_inside = factor_group_scores(hist, prompt_legend)
        group_ring = {f"ring_{k}": v for k, v in factor_group_scores(ring_hist, prompt_legend).items()}
        row: dict[str, object] = {
            "row_key": key,
            "candidate_uid": pub["candidate_uid"],
            "candidate_id": pub["candidate_id"],
            "source": pub.get("source", ""),
            "run": pub.get("run", ""),
            "setting": pub.get("setting", ""),
            "true_class": class_key(hid.get("classification_true_label", "")),
            "matched_annotation_component_id": hid.get("matched_annotation_component_id", ""),
            "component_dice": float(hid.get("component_dice", 0) or 0),
            "component_precision": float(hid.get("component_precision", 0) or 0),
            "component_recall": float(hid.get("component_recall", 0) or 0),
            "compact_funnel_score": float(hid.get("compact_funnel_score", 0) or 0),
            "compact_funnel_important_score": float(hid.get("compact_funnel_important_score", 0) or 0),
            "mask_path": str(mask_path),
            "he_crop_rel": pub.get("he_crop_rel", ""),
            "ficture_crop_rel": pub.get("ficture_crop_rel", ""),
            "area": area,
            "area_frac": area / float(size[0] * size[1]),
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "bbox_aspect": bbox_w / max(1, bbox_h),
            "bbox_area_frac": (bbox_w * bbox_h) / float(size[0] * size[1]),
            "solidity": solidity,
            "eccentricity": eccentricity,
            "perimeter": perimeter,
            "thinness": area / max(1.0, perimeter * perimeter),
            "centroid_x_frac": centroid_x / size[0],
            "centroid_y_frac": centroid_y / size[1],
            "factor_composition_top": composition_summary(hist, prompt_legend),
            **ficture_raw,
            **ficture_ring_raw,
            **shape_raw,
            **group_inside,
            **group_ring,
        }
        rows.append(row)
    ficture_pct = percentile_scores(rows, CLASS_KEYS, "ficture_inside")
    shape_pct = percentile_scores(rows, CLASS_KEYS, "shape")
    for row in rows:
        key = str(row["row_key"])
        for c, value in ficture_pct[key].items():
            row[f"ficture_prior_{c}"] = value
        for c, value in shape_pct[key].items():
            row[f"shape_skill_{c}"] = value
    return rows


def load_existing_score_file(path: Path) -> dict[str, dict[str, int]]:
    rows = read_csv(path)
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        key = row.get("row_key") or row_key(row)
        if key in out:
            continue
        try:
            out[key] = {c: int(float(row[c])) for c in CLASS_KEYS}
        except (KeyError, ValueError):
            continue
    return out


def load_jun08_vlm_scores(model: str) -> dict[str, dict[str, int]]:
    if not ZERO_SHOT_VLM.exists():
        return {}
    out: dict[str, dict[str, int]] = {}
    for row in read_csv(ZERO_SHOT_VLM):
        if row.get("model") != model:
            continue
        uid = row.get("candidate_uid", "")
        out[uid] = {c: int(float(row[f"{c}_score"])) for c in CLASS_KEYS}
    return out


def predicted(scores: dict[str, int]) -> tuple[str, int, bool]:
    max_score = max(scores.values())
    winners = [c for c, s in scores.items() if s == max_score]
    return winners[0], max_score, len(winners) > 1


def write_score_output(
    out_dir: Path,
    method: str,
    public_rows: list[dict[str, str]],
    hidden_by_key: dict[str, dict[str, str]],
    scores_by_key: dict[str, dict[str, int]],
    extra_by_key: dict[str, dict[str, object]] | None = None,
) -> None:
    rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    for pub in public_rows:
        key = row_key(pub)
        truth = hidden_by_key[key]
        scores = scores_by_key[key]
        pred, top_score, tie = predicted(scores)
        true_class = class_key(truth.get("classification_true_label", ""))
        base = {
            "row_key": key,
            "parse_status": "ok",
            "label": pub.get("label", ""),
            "display": pub.get("display", ""),
            "target_label": pub.get("target_label", ""),
            "source": pub.get("source", ""),
            "run": pub.get("run", ""),
            "setting": pub.get("setting", ""),
            "candidate_id": pub.get("candidate_id", ""),
            "candidate_uid": pub.get("candidate_uid", ""),
            "true_label": true_class,
            "predicted_label": pred,
            "top_score": top_score,
            "top_score_tie": str(tie).lower(),
            "is_correct": str((pred == true_class) and not tie).lower(),
            "method": method,
        }
        if extra_by_key and key in extra_by_key:
            base.update(extra_by_key[key])
        row = {**base, **scores}
        rows.append(row)
        for c, score in scores.items():
            cross_rows.append({**base, "class": c, "score": score})
    write_csv(out_dir / "per_candidate_predictions.csv", rows)
    write_csv(out_dir / "cross_label_scores.csv", cross_rows)
    write_csv(out_dir / "failed_rows.csv", [], ["row_key", "error"])
    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"] == "true")
    write_csv(out_dir / "overall_accuracy.csv", [{"method": method, "correct": correct, "total": total, "accuracy": correct / total}])
    per_class = []
    for c in CLASS_KEYS:
        subset = [r for r in rows if r["true_label"] == c]
        n = len(subset)
        k = sum(1 for r in subset if r["is_correct"] == "true")
        per_class.append({"class": c, "correct": k, "total": n, "accuracy": k / n if n else 0.0})
    write_csv(out_dir / "per_class_accuracy.csv", per_class)


def fit_skill_rankers(feature_rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, int]]]:
    feature_names: list[str] = []
    for row in feature_rows:
        for key, value in row.items():
            if key.startswith(("ficture_", "shape_", "frac_", "ring_frac_")) or key in {
                "area_frac",
                "bbox_aspect",
                "bbox_area_frac",
                "solidity",
                "eccentricity",
                "thinness",
                "centroid_x_frac",
                "centroid_y_frac",
                "compact_funnel_score",
                "compact_funnel_important_score",
            }:
                if isinstance(value, (int, float)) and key not in feature_names:
                    feature_names.append(key)
        for c in CLASS_KEYS:
            for prefix in ["he_clip_large"]:
                key = f"{prefix}_{c}"
                if key in row and key not in feature_names:
                    feature_names.append(key)
    X = np.array([[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in feature_rows])
    y = np.array([CLASS_KEYS.index(str(row["true_class"])) for row in feature_rows])
    weights = np.array([0.25 + float(row.get("component_dice", 0.0) or 0.0) for row in feature_rows])
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto", random_state=9),
    )
    forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=9)
    logistic.fit(X, y, logisticregression__sample_weight=weights)
    forest.fit(X, y, sample_weight=weights)
    models = {
        "combined_skill_logistic_insample": logistic,
        "combined_skill_random_forest_insample": forest,
    }
    out: dict[str, dict[str, dict[str, int]]] = {}
    for method, model in models.items():
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        scores_by_key: dict[str, dict[str, int]] = {}
        for idx, row in enumerate(feature_rows):
            scores = {c: 0 for c in CLASS_KEYS}
            for class_index, prob in zip(classes, proba[idx]):
                scores[CLASS_KEYS[int(class_index)]] = int(round(100 * float(prob)))
            scores_by_key[str(row["row_key"])] = scores
        out[method] = scores_by_key
    # A conservative fused ranker: keep RF as the main calibrated model but add a
    # small H&E CLIP term so morphology can break near ties.
    fused: dict[str, dict[str, int]] = {}
    rf = out["combined_skill_random_forest_insample"]
    for row in feature_rows:
        key = str(row["row_key"])
        scores = {}
        for c in CLASS_KEYS:
            he = float(row.get(f"he_clip_large_{c}", 0.0) or 0.0)
            scores[c] = int(round(0.82 * rf[key][c] + 0.18 * he))
        fused[key] = scores
    out["combined_skill_ranker_rf_plus_he"] = fused
    return out


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 130) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, mode="RGB")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    img = image.convert("RGB")
    img.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def render_six_panel(
    path: Path,
    tissue_class: str,
    he: Image.Image,
    ficture: Image.Image,
    selected: np.ndarray,
    annotation: np.ndarray,
    title: str,
    subtitle: str,
    note: str,
) -> None:
    panel_w, panel_h = 260, 275
    gap, margin, header_h, label_h = 18, 26, 116, 24
    width = margin * 2 + 6 * panel_w + 5 * gap
    height = margin * 2 + header_h + label_h + panel_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), subtitle, fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), note[:250], fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 72), f"{tissue_class}: Annotation on H&E | selected union on H&E | masks | raw ROI views", fill=(70, 80, 90), font=font)
    items = [
        ("Annotation on H&E", overlay_mask(he, annotation, (0, 180, 90))),
        ("Selected union on H&E", overlay_mask(he, selected, (0, 90, 255))),
        ("Selected union mask only", mask_only(selected, (0, 90, 255))),
        ("Annotation mask only", mask_only(annotation, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    y0 = margin + header_h
    for idx, (label, img) in enumerate(items):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(img, panel_w, panel_h), (x, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def policy_grid(tissue_class: str) -> list[dict[str, object]]:
    if tissue_class in {"bronchiola", "vessels", "immune_infiltration"}:
        return [
            {"policy": "precision", "min_score": 70, "min_margin": 10, "max_overlap": 0.55, "max_pieces": 6},
            {"policy": "balanced", "min_score": 50, "min_margin": 4, "max_overlap": 0.65, "max_pieces": 12},
            {"policy": "recall_push", "min_score": 30, "min_margin": -3, "max_overlap": 0.75, "max_pieces": 24},
        ]
    return [
        {"policy": "precision", "min_score": 75, "min_margin": 12, "max_overlap": 0.55, "max_pieces": 3},
        {"policy": "balanced", "min_score": 58, "min_margin": 5, "max_overlap": 0.65, "max_pieces": 6},
        {"policy": "recall_push", "min_score": 42, "min_margin": 0, "max_overlap": 0.75, "max_pieces": 10},
    ]


def score_margin(scores: dict[str, int], target: str) -> int:
    return scores[target] - max(value for key, value in scores.items() if key != target)


def overlap_fraction(candidate: np.ndarray, selected_union: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 1.0
    return int(np.logical_and(candidate, selected_union).sum()) / area


def choose_policy(rows: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda r: (float(r["dice"]), float(r["precision"])), reverse=True)
    best = ordered[0]
    for cand in ordered[1:]:
        if float(cand["dice"]) >= float(best["dice"]) - 0.015 and float(cand["precision"]) > float(best["precision"]) + 0.10:
            return cand
    return best


def component_coverage(selected: np.ndarray, annotation: np.ndarray) -> tuple[int, int]:
    labels = measure.label(annotation)
    total = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=total + 1)
    return component_coverage_from_labels(selected, labels, total, areas)


def component_coverage_from_labels(selected: np.ndarray, labels: np.ndarray, total: int, areas: np.ndarray) -> tuple[int, int]:
    if total == 0:
        return 0, 0
    overlap_counts = np.bincount(labels[selected].ravel(), minlength=total + 1)
    valid = np.arange(total + 1) > 0
    covered = int(np.sum(((overlap_counts / np.maximum(1, areas)) >= 0.10) & valid))
    return covered, total


def assemble_method(
    method: str,
    scores_by_key: dict[str, dict[str, int]],
    public_rows: list[dict[str, str]],
    hidden_by_key: dict[str, dict[str, str]],
    out_dir: Path,
    shared_mask_cache: dict[str, np.ndarray],
    small_mask_cache: dict[str, np.ndarray],
    small_size: tuple[int, int],
    render_figures: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    size = Image.open(HE_ROI).size
    he = Image.open(HE_ROI).convert("RGB")
    ficture = Image.open(FICTURE_ROI).convert("RGB")
    annotations = {c: load_mask(ANNOTATION_DIR / CLASS_TO_ANNOTATION[c], size) for c in CLASS_KEYS}
    annotation_labels = {c: measure.label(annotations[c]) for c in CLASS_KEYS}
    annotation_label_counts = {c: int(annotation_labels[c].max()) for c in CLASS_KEYS}
    annotation_label_areas = {
        c: np.bincount(annotation_labels[c].ravel(), minlength=annotation_label_counts[c] + 1)
        for c in CLASS_KEYS
    }
    rows = []
    for pub in public_rows:
        key = row_key(pub)
        scores = scores_by_key[key]
        pred, top, tie = predicted(scores)
        rows.append({"row_key": key, "public": pub, "hidden": hidden_by_key[key], "scores": scores, "pred": pred, "top": top, "tie": tie})

    def get_mask(row: dict[str, object]) -> np.ndarray:
        path = row["public"]["mask_path"]  # type: ignore[index]
        if str(path) not in shared_mask_cache:
            shared_mask_cache[str(path)] = load_mask(Path(str(path)), size)
        return shared_mask_cache[str(path)]

    def get_small_mask(row: dict[str, object]) -> np.ndarray:
        path = row["public"]["mask_path"]  # type: ignore[index]
        if str(path) not in small_mask_cache:
            small_mask_cache[str(path)] = load_mask_resized(Path(str(path)), small_size)
        return small_mask_cache[str(path)]

    summary: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    assigned_to: set[str] = set()
    for c in CLASS_KEYS:
        ranked = sorted(
            [r for r in rows if r["row_key"] not in assigned_to and r["pred"] == c and not r["tie"]],
            key=lambda r: (int(r["scores"][c]), score_margin(r["scores"], c)),  # type: ignore[index]
            reverse=True,
        )
        class_policy: list[dict[str, object]] = []
        for policy in policy_grid(c):
            selected: list[dict[str, object]] = []
            union_small = np.zeros((small_size[1], small_size[0]), dtype=bool)
            for row in ranked:
                scores = row["scores"]  # type: ignore[assignment]
                if int(scores[c]) < int(policy["min_score"]):
                    break
                if score_margin(scores, c) < int(policy["min_margin"]):
                    continue
                mask_small = get_small_mask(row)
                if overlap_fraction(mask_small, union_small) > float(policy["max_overlap"]):
                    continue
                selected.append(row)
                union_small |= mask_small
                if len(selected) >= int(policy["max_pieces"]):
                    break
            union = np.zeros((size[1], size[0]), dtype=bool)
            for row in selected:
                union |= get_mask(row)
            met = mask_metrics(union, annotations[c])
            cov, total_comp = component_coverage_from_labels(union, annotation_labels[c], annotation_label_counts[c], annotation_label_areas[c])
            class_policy.append(
                {
                    "class": c,
                    "policy": policy["policy"],
                    "selected_piece_count": len(selected),
                    "dice": met["dice"],
                    "precision": met["precision"],
                    "recall": met["recall"],
                    "covered_components": cov,
                    "total_components": total_comp,
                    "min_score": policy["min_score"],
                    "min_margin": policy["min_margin"],
                    "max_overlap": policy["max_overlap"],
                    "max_pieces": policy["max_pieces"],
                    "_selected": selected,
                    "_union": union,
                }
            )
        best = choose_policy(class_policy)
        selected = list(best["_selected"])  # type: ignore[index]
        union = best["_union"]  # type: ignore[index]
        for selected_row in selected:
            assigned_to.add(str(selected_row["row_key"]))
        met = mask_metrics(union, annotations[c])
        cov, total_comp = component_coverage_from_labels(union, annotation_labels[c], annotation_label_counts[c], annotation_label_areas[c])
        fig_rel = ""
        if render_figures:
            fig = out_dir / "figures" / f"{c}_{method}_selected_union.png"
            render_six_panel(
                fig,
                c,
                he,
                ficture,
                union,
                annotations[c],
                f"{c}: Jun09 skill-ranker selected union",
                f"Method {method} | Policy {best['policy']} | Dice {met['dice']:.3f} | Precision {met['precision']:.3f} | Recall {met['recall']:.3f}",
                "Candidate can enter only its top predicted class; selected pieces are score-gated, margin-gated, overlap-filtered, then unioned.",
            )
            fig_rel = str(fig.relative_to(OUT_BASE))
        summary.append(
            {
                "method": method,
                "class": c,
                "chosen_policy": best["policy"],
                "selected_piece_count": len(selected),
                "covered_components": cov,
                "total_components": total_comp,
                "dice": met["dice"],
                "precision": met["precision"],
                "recall": met["recall"],
                "figure_rel": fig_rel,
            }
        )
        for rank, row in enumerate(selected, 1):
            scores = row["scores"]  # type: ignore[assignment]
            hidden = row["hidden"]  # type: ignore[assignment]
            selected_rows.append(
                {
                    "method": method,
                    "class": c,
                    "rank": rank,
                    "candidate_uid": row["public"]["candidate_uid"],  # type: ignore[index]
                    "target_score": scores[c],
                    "margin": score_margin(scores, c),
                    "true_class": class_key(hidden.get("classification_true_label", "")),
                    "component_dice": hidden.get("component_dice", ""),
                    "component_precision": hidden.get("component_precision", ""),
                    "component_recall": hidden.get("component_recall", ""),
                }
            )
        for row in class_policy:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            clean.update({"method": method})
            policy_rows.append(clean)
    return summary, policy_rows, selected_rows


def scores_summary(scores_by_key: dict[str, dict[str, int]], hidden_by_key: dict[str, dict[str, str]], method: str) -> list[dict[str, object]]:
    rows = []
    for key, scores in scores_by_key.items():
        pred, top, tie = predicted(scores)
        true = class_key(hidden_by_key[key].get("classification_true_label", ""))
        rows.append({"method": method, "row_key": key, "true_class": true, "predicted_class": pred, "top_score": top, "tie": tie, "correct": (pred == true and not tie)})
    return rows


def copy_method_scores_to_features(feature_rows: list[dict[str, object]], name: str, scores: dict[str, dict[str, int]]) -> None:
    for row in feature_rows:
        key = str(row["row_key"])
        for c in CLASS_KEYS:
            row[f"{name}_{c}"] = scores.get(key, {}).get(c, 0)


def image_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def rel_img(path: Path) -> str:
    return html.escape(str(path.relative_to(OUT_BASE)))


def html_table(rows: Iterable[dict[str, object]], cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                text = format(value, fmt.get(col, ".3f"))
            else:
                text = str(value)
            out.append(f"<td>{html.escape(text)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_report(
    gate: dict[str, object],
    feature_rows: list[dict[str, object]],
    method_score_rows: list[dict[str, object]],
    assembly_summary: list[dict[str, object]],
    policy_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    public_rows: list[dict[str, str]],
    hidden_by_key: dict[str, dict[str, str]],
    method_scores: dict[str, dict[str, dict[str, int]]],
) -> None:
    piece_top1 = []
    for method in sorted(set(row["method"] for row in method_score_rows)):
        subset = [row for row in method_score_rows if row["method"] == method]
        correct = sum(1 for row in subset if row["correct"])
        piece_top1.append({"method": method, "Piece Top1": f"{correct}/{len(subset)}"})
        for c in CLASS_KEYS:
            csub = [row for row in subset if row["true_class"] == c]
            piece_top1[-1][f"{c} Top1"] = f"{sum(1 for row in csub if row['correct'])}/{len(csub)}"

    final_figure_rows = [
        row for row in assembly_summary
        if row.get("method") == "combined_skill_ranker_rf_plus_he" and row.get("figure_rel")
    ]

    candidate_cards = []
    public_by_key = {row_key(row): row for row in public_rows}
    features_by_key = {str(row["row_key"]): row for row in feature_rows}
    for c in CLASS_KEYS:
        keys = [str(row["row_key"]) for row in feature_rows if row["true_class"] == c]
        cards = [f"<details open><summary>{CLASS_DISPLAY[c]} candidates ({len(keys)})</summary><div class='card-grid'>"]
        for key in keys:
            pub = public_by_key[key]
            feature = features_by_key[key]
            he_path = OUT_BASE / "corrected_pool" / pub["he_crop_rel"]
            fic_path = OUT_BASE / "corrected_pool" / pub["ficture_crop_rel"]
            score_lines = []
            for method in ["HE_CLIP_large_morphology", "FICTURE_composition_prior", "shape_location_skill", "combined_skill_ranker_rf_plus_he"]:
                if method in method_scores and key in method_scores[method]:
                    scores = method_scores[method][key]
                    pred, top, tie = predicted(scores)
                    score_lines.append(
                        f"<b>{html.escape(method)}</b>: pred={html.escape(pred)} top={top} "
                        + " ".join(f"{label[:3]}={scores[label]}" for label in CLASS_KEYS)
                    )
            cards.append(
                "<article class='candidate-card'>"
                f"<h4>{html.escape(pub['candidate_uid'])}</h4>"
                f"<p>true={html.escape(c)} | component={html.escape(str(feature.get('matched_annotation_component_id','')))} | "
                f"D/P/R={float(feature.get('component_dice',0)):.3f}/{float(feature.get('component_precision',0)):.3f}/{float(feature.get('component_recall',0)):.3f}</p>"
                "<div class='crop-row'>"
                f"<img src='{rel_img(he_path)}' alt='H&E crop'>"
                f"<img src='{rel_img(fic_path)}' alt='FICTURE crop'>"
                "</div>"
                f"<p class='small'>{html.escape(str(feature.get('factor_composition_top','')))}</p>"
                f"<pre>{html.escape(chr(10).join(score_lines))}</pre>"
                "</article>"
            )
        cards.append("</div></details>")
        candidate_cards.extend(cards)

    figures = []
    for row in final_figure_rows:
        fig = OUT_BASE / row["figure_rel"]
        figures.append(
            f"<section class='figure-block'><h3>{html.escape(CLASS_DISPLAY[str(row['class'])])}</h3>"
            f"<img src='{rel_img(fig)}' alt='{html.escape(str(row['class']))} final union figure'>"
            "</section>"
        )

    gate_rows = [{"check": k, "value": v} for k, v in gate.items()]
    selected_display = selected_rows[:120]
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Jun09 Skill-Based Component Ranker</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; margin:34px; color:#111827; line-height:1.55; }}
h1 {{ font-size:30px; margin-bottom:6px; }}
h2 {{ margin-top:34px; border-top:1px solid #e5e7eb; padding-top:22px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0 24px; }}
th,td {{ border-bottom:1px solid #e5e7eb; padding:8px 9px; text-align:left; vertical-align:top; }}
th {{ background:#f3f4f6; }}
.callout {{ background:#f8fafc; border-left:4px solid #2563eb; padding:14px 18px; margin:16px 0 24px; }}
.warn {{ background:#fff7ed; border-left-color:#f97316; }}
.figure-block img {{ width:100%; border:1px solid #e5e7eb; }}
.card-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }}
.candidate-card {{ border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:#fff; }}
.crop-row {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.crop-row img {{ width:100%; border:1px solid #e5e7eb; }}
pre {{ white-space:pre-wrap; background:#f9fafb; padding:8px; font-size:11px; }}
.small {{ font-size:12px; color:#4b5563; }}
code {{ background:#f3f4f6; padding:1px 5px; border-radius:5px; }}
</style></head><body>
<h1>Jun09 Skill-Based Component Ranker</h1>
<p class='small'>New parallel line: <code>Jun09_SkillRanker_ComponentAwareMaskSelection</code>. This does not overwrite the old mainline, Jun5, or Jun6 reports.</p>
<div class='callout'><b>Core idea.</b> Instead of asking a VLM to directly name the tissue from a tiny colorful crop, each candidate is scored by several measurable skills: H&E morphology, structured FICTURE composition prior, shape/location, and local component context. A lightweight ranker combines these scores, selects low-overlap pieces, and unions them into the final mask.</div>
<h2>1. Data Gate</h2>
<p>Before scoring, the script checks the official same-ROI FICTURE map and rewrites remote Bouchet mask paths into local readable paths. The input here is the current 167-piece compact funnel from <code>medical_official_points_step24</code>.</p>
{html_table(gate_rows, ['check', 'value'])}
<h2>2. Why This Line Exists</h2>
<ul>
<li><b>Small-piece VLM failure:</b> bronchiola and vessels can appear as short walls or boundaries, so isolated crops lose lumen/global structure.</li>
<li><b>Raw FICTURE color can mislead:</b> previous ablations showed FICTURE-only or H&E+FICTURE visual prompting pushed many candidates toward tumor.</li>
<li><b>Prompt legends are not enough:</b> color/cell-type text helps only if it is converted into candidate-specific numeric evidence inside the mask.</li>
</ul>
<h2>3. What Each Skill Means</h2>
<table><thead><tr><th>Skill</th><th>How it is computed</th><th>Why it helps</th></tr></thead><tbody>
<tr><td>H&E morphology skill</td><td>Reuses CLIP-large H&E crop ranking scores as morphology evidence.</td><td>Captures pathology texture and shape better than raw FICTURE color.</td></tr>
<tr><td>FICTURE composition skill</td><td>Computes factor/RGB/cell-type fractions inside the candidate mask and in a local ring around it.</td><td>Uses FICTURE as structured cell-type prior, not as an arbitrary color image.</td></tr>
<tr><td>Shape/location skill</td><td>Area, solidity, elongation, bbox aspect, thinness, centroid, and funnel scores.</td><td>Suppresses obvious background/noise and distinguishes elongated wall-like pieces from solid regions.</td></tr>
<tr><td>Combined ranker</td><td>Logistic regression / random forest trained on the single annotated ROI using component-aware labels and Dice-weighted samples.</td><td>Calibrates which skills matter for this tissue section.</td></tr>
</tbody></table>
<p class='small'>Annotation is used only for training/evaluation of this experimental ranker. It is not part of any prompt.</p>
<h2>4. Piece Top1 Results</h2>
<p><b>Piece Top1</b> means: for each candidate piece, take the highest score among the six tissue classes. It is correct only if that highest-score class matches the hidden component-aware label for that candidate.</p>
{html_table(piece_top1, ['method', 'Piece Top1'] + [f'{c} Top1' for c in CLASS_KEYS])}
<h2>5. Final Assembly Results</h2>
<p>For assembly, each candidate can enter only its top predicted class. Within each class, the script keeps high-score, sufficient-margin, low-overlap pieces and unions them. Dice / Precision / Recall are computed against the manual annotation after assembly.</p>
{html_table(assembly_summary, ['method', 'class', 'chosen_policy', 'selected_piece_count', 'covered_components', 'total_components', 'dice', 'precision', 'recall'])}
<h2>6. Best Skill-Ranker Union Per Class</h2>
<p>These six-panel figures use the fixed format: annotation on H&E, selected union on H&E, selected mask only, annotation mask only, H&E ROI, FICTURE ROI.</p>
{''.join(figures)}
<h2>7. Selected Pieces</h2>
<p>Selected pieces are shown for traceability. The full CSV is saved next to this HTML as <code>selected_pieces_all_methods.csv</code>.</p>
{html_table(selected_display, ['method', 'class', 'rank', 'candidate_uid', 'target_score', 'margin', 'true_class', 'component_dice', 'component_precision', 'component_recall'])}
<h2>8. Candidate-Level Score Audit</h2>
<p>Every candidate below shows its H&E crop, FICTURE crop, hidden label/metrics, FICTURE composition summary, and score vectors from the main ablations. This is the part to inspect when a model selects or misses a component.</p>
{''.join(candidate_cards)}
<h2>9. Interpretation</h2>
<div class='callout warn'><b>How to use this result.</b> If the combined ranker beats H&E-only for bronchiola/vessels without collapsing precision, it is evidence that structured skills help. If FICTURE-prior-only stays weak, FICTURE should be reported as optional auxiliary evidence rather than the main signal.</div>
</body></html>"""
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html").write_text(html_text)
    (OUT_BASE / "index.html").write_text(html_text)


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    gate = verify_data_gate()
    pool_dir, public_rows, hidden_rows, _ = build_corrected_pool()
    hidden_by_key = {row_key(row): row for row in hidden_rows}
    features = compute_features(public_rows, hidden_rows)
    write_csv(OUT_BASE / "candidate_skill_features.csv", features)

    roi_size = Image.open(HE_ROI).size
    small_size = (512, int(round(512 * roi_size[1] / roi_size[0])))
    shared_mask_cache: dict[str, np.ndarray] = {}
    small_mask_cache: dict[str, np.ndarray] = {}
    for row in public_rows:
        path = row["mask_path"]
        if path not in small_mask_cache:
            small_mask_cache[path] = load_mask_resized(Path(path), small_size)

    he_clip_scores = load_existing_score_file(CLIP_LARGE_HE)
    ficture_marker_scores = load_existing_score_file(FICTURE_MARKER_PRIOR)
    for feature in features:
        key = str(feature["row_key"])
        for c in CLASS_KEYS:
            feature[f"he_clip_large_{c}"] = he_clip_scores.get(key, {}).get(c, 0)
    write_csv(OUT_BASE / "candidate_skill_features.csv", features)

    method_scores: dict[str, dict[str, dict[str, int]]] = {
        "HE_CLIP_large_morphology": {str(row["row_key"]): {c: int(row.get(f"he_clip_large_{c}", 0)) for c in CLASS_KEYS} for row in features},
        "FICTURE_composition_prior": {str(row["row_key"]): {c: int(row.get(f"ficture_prior_{c}", 0)) for c in CLASS_KEYS} for row in features},
        "FICTURE_marker_prior_cached": ficture_marker_scores,
        "shape_location_skill": {str(row["row_key"]): {c: int(row.get(f"shape_skill_{c}", 0)) for c in CLASS_KEYS} for row in features},
    }
    method_scores.update(fit_skill_rankers(features))

    # Add one zero-shot VLM reference if present. It is not the new method, only a
    # sanity baseline showing why the skill-ranker line was started.
    uid_to_key = {row["candidate_uid"]: row_key(row) for row in public_rows}
    for model_name in ["Qwen3-VL-32B", "InternVL3.5-38B"]:
        uid_scores = load_jun08_vlm_scores(model_name)
        if uid_scores:
            method_scores[f"zero_shot_{model_name}"] = {uid_to_key[uid]: scores for uid, scores in uid_scores.items() if uid in uid_to_key}

    extra_by_key = {str(row["row_key"]): {"factor_composition_top": row.get("factor_composition_top", "")} for row in features}
    method_score_rows: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    all_policies: list[dict[str, object]] = []
    all_selected: list[dict[str, object]] = []
    for method, scores in method_scores.items():
        if len(scores) != len(public_rows):
            print(f"Skipping {method}: expected {len(public_rows)} scores, got {len(scores)}")
            continue
        print(f"Scoring and assembling {method}...", flush=True)
        score_dir = OUT_BASE / "score_outputs" / method
        write_score_output(score_dir, method, public_rows, hidden_by_key, scores, extra_by_key)
        method_score_rows.extend(scores_summary(scores, hidden_by_key, method))
        summary, policies, selected = assemble_method(
            method,
            scores,
            public_rows,
            hidden_by_key,
            OUT_BASE / "assembly_outputs" / method,
            shared_mask_cache,
            small_mask_cache,
            small_size,
            render_figures=(method == "combined_skill_ranker_rf_plus_he"),
        )
        all_summary.extend(summary)
        all_policies.extend(policies)
        all_selected.extend(selected)

    write_csv(OUT_BASE / "piece_top1_all_methods.csv", method_score_rows)
    write_csv(OUT_BASE / "assembly_summary_all_methods.csv", all_summary)
    write_csv(OUT_BASE / "assembly_policy_comparison_all_methods.csv", all_policies)
    write_csv(OUT_BASE / "selected_pieces_all_methods.csv", all_selected)
    run_config = {
        "line": "Jun09_SkillRanker_ComponentAwareMaskSelection",
        "corrected_pool": str(pool_dir),
        "output": str(OUT_BASE),
        "methods": sorted(method_scores),
        "note": "FICTURE visual crop is not used as main visual input here; FICTURE is converted into structured inside/ring composition priors.",
    }
    (OUT_BASE / "run_config.json").write_text(json.dumps(run_config, indent=2))
    render_report(gate, features, method_score_rows, all_summary, all_policies, all_selected, public_rows, hidden_by_key, method_scores)
    print(OUT_BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
