#!/usr/bin/env python3
"""Ask a local VLM to directly ground tissue classes, then evaluate masks.

This tests the "VLM draws the mask" direction in two practical forms:
  1. Direct VLM mask: ask the VLM for boxes/polygons and rasterize them.
  2. Grounded-SAM-style selection: use VLM boxes to select existing SAM masks.

Annotation is used only at the end for calibration metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent))
from vlm_candidate_judge import (  # noqa: E402
    LABEL_DESCRIPTIONS,
    LABEL_ORDER,
    bbox,
    load_gene_factor_scores,
    load_summary_masks,
    load_vlm,
    metrics,
    overlay,
    prior_map_for_label,
    render_prior_rgb,
    resize_bool,
    resize_rgb,
    slugify,
    vlm_generate,
)
from ficture_factor_semantics import label_factor_hints, legend_text, load_semantic_legend  # noqa: E402


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def parse_grounding(text: str) -> List[dict]:
    match = re.search(r"\[.*\]|\{.*\}", text, flags=re.S)
    if not match:
        return []
    raw = match.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, dict):
        for key in ["regions", "boxes", "objects", "annotations"]:
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    return []


def norm_to_px_box(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    vals = [float(x) for x in box[:4]]
    if max(vals) <= 1.5:
        x1, y1, x2, y2 = vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h
    elif max(vals) <= 1000:
        x1, y1, x2, y2 = vals[0] / 1000 * w, vals[1] / 1000 * h, vals[2] / 1000 * w, vals[3] / 1000 * h
    else:
        x1, y1, x2, y2 = vals
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return (
        max(0, min(w, int(round(x1)))),
        max(0, min(h, int(round(y1)))),
        max(0, min(w, int(round(x2)))),
        max(0, min(h, int(round(y2)))),
    )


def norm_to_px_polygon(points: Sequence[Sequence[float]], w: int, h: int) -> List[Tuple[int, int]]:
    out = []
    flat = []
    for p in points:
        if isinstance(p, (int, float)):
            flat.append(float(p))
        else:
            flat.extend(float(x) for x in p[:2])
    if not flat:
        return out
    if max(flat) <= 1.5:
        mode = "unit"
    elif max(flat) <= 1000:
        mode = "permille"
    else:
        mode = "pixel"
    for i in range(0, len(flat) - 1, 2):
        if mode == "unit":
            x, y = flat[i] * w, flat[i + 1] * h
        elif mode == "permille":
            x, y = flat[i] / 1000 * w, flat[i + 1] / 1000 * h
        else:
            x, y = flat[i], flat[i + 1]
        out.append((max(0, min(w, int(round(x)))), max(0, min(h, int(round(y))))))
    return out


def rasterize_regions(regions: List[dict], shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    h, w = shape_hw
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    boxes = []
    for r in regions:
        box = r.get("bbox") or r.get("bbox_2d") or r.get("box") or r.get("rectangle")
        poly = r.get("polygon") or r.get("points") or r.get("segmentation")
        if box:
            px = norm_to_px_box(box, w, h)
            if px[2] > px[0] and px[3] > px[1]:
                draw.rectangle(px, fill=255)
                boxes.append(px)
        if poly:
            pts = norm_to_px_polygon(poly, w, h)
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
                boxes.append(bbox(np.array(img) > 127))
    return np.array(img) > 127, boxes


def box_mask(box: Tuple[int, int, int, int], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    out = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = True
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_candidate_rows(score_csvs: Sequence[Path], labels: Sequence[str], rankers: Sequence[str], per_label: int) -> Dict[str, List[dict]]:
    rows_by_label: Dict[str, Dict[str, dict]] = {label: {} for label in labels}
    ranker_set = set(rankers)
    for score_csv in score_csvs:
        with score_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] not in rows_by_label or row["ranker"] not in ranker_set:
                    continue
                key = "|".join([row["source"], row["run"], row["setting"], row["candidate_id"]])
                old = rows_by_label[row["label"]].get(key)
                if old is None or float(row["score"]) > float(old["score"]):
                    rows_by_label[row["label"]][key] = row
    out: Dict[str, List[dict]] = {}
    for label, items in rows_by_label.items():
        rows = list(items.values())
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        out[label] = rows[:per_label]
    return out


def select_candidates_by_grounding(rows: Sequence[dict], regions_mask: np.ndarray, shape_hw: Tuple[int, int], top_k: int, max_overlap: float) -> Tuple[np.ndarray, List[dict]]:
    selected = []
    current = np.zeros(shape_hw, dtype=bool)
    scored = []
    for row in rows:
        mask = resize_bool(read_mask(Path(row["mask_path"])), shape_hw)
        overlap = mask_iou(mask, regions_mask)
        score = 0.65 * overlap + 0.20 * float(row.get("score") or 0) + 0.15 * float(row.get("molecular_score") or 0)
        scored.append((score, row, mask))
    for score, row, mask in sorted(scored, key=lambda x: x[0], reverse=True):
        if not mask.any():
            continue
        if current.any():
            ov = np.logical_and(current, mask).sum() / max(float(mask.sum()), 1.0)
            if ov > max_overlap:
                continue
        current |= mask
        selected.append(row)
        if len(selected) >= top_k:
            break
    return current, selected


def make_context_images(he: np.ndarray, ficture: np.ndarray, prior_rgb: np.ndarray, max_side: int = 1100) -> List[Image.Image]:
    imgs = []
    for arr in [he, ficture, prior_rgb]:
        img = Image.fromarray(arr.astype(np.uint8))
        scale = min(1.0, max_side / max(img.size))
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.BILINEAR)
        imgs.append(img)
    return imgs


def prompt_for_label(label: str, factor_context: str = "") -> Tuple[str, str]:
    display = dict(LABEL_ORDER)[label]
    desc = LABEL_DESCRIPTIONS[label]
    system = "You are a pathology visual grounding assistant. Return only valid JSON."
    prompt = f"""
You are given three aligned full-ROI images from the same lung tissue:
Image 1: H&E histology.
Image 2: FICTURE/spatial transcriptomic factor map.
Image 3: class-specific gene/FICTURE prior heatmap where warm colors indicate stronger molecular support.

Target class: {display}
Definition: {desc}

FICTURE interpretation context:
{factor_context}

Task: directly mark where this target tissue class is located. Prefer recall, but avoid clearly unrelated tissue.
Return JSON only as a list of regions. Use normalized coordinates from 0 to 1000 relative to the image:
[
  {{"bbox_2d": [x1, y1, x2, y2], "confidence": 0.0-1.0, "reason": "short"}},
  ...
]
If a polygon is more appropriate, use {{"polygon": [[x,y], ...], "confidence": 0.0-1.0}}.
Return at most 8 regions.
"""
    return system, prompt.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-csv", type=Path, action="append", required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--factor-label-npy", type=Path, required=True)
    parser.add_argument("--factor-annotation", type=Path, required=True)
    parser.add_argument("--factor-semantic-legend", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-label", type=int, default=80)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20])
    parser.add_argument("--max-overlap", type=float, default=0.88)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--rankers", nargs="+", default=["molecular_only", "class_weighted", "bav_focus_recall", "bav_focus_balanced", "salip_clip", "precision_gated"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.output_dir / "best_panels"
    panel_dir.mkdir(exist_ok=True)

    factor_labels = np.load(args.factor_label_npy).astype(np.int16)
    shape = factor_labels.shape
    he = resize_rgb(np.array(Image.open(args.he_image).convert("RGB")), shape)
    ficture = resize_rgb(np.array(Image.open(args.ficture_image).convert("RGB")), shape)
    n_factors = int(factor_labels[factor_labels >= 0].max()) + 1 if np.any(factor_labels >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    semantic_factors = load_semantic_legend(args.factor_semantic_legend, n_factors=n_factors)
    full_legend_text = legend_text(semantic_factors) if semantic_factors else "No factor semantic legend was provided."
    gt_masks = load_summary_masks(args.summary_path, shape)
    labels = [slug for slug, _ in LABEL_ORDER]
    candidates = load_candidate_rows(args.score_csv, labels, args.rankers, args.per_label)
    processor, model = load_vlm(args.model, args.device)

    grounding_rows = []
    selection_rows = []
    best_rows = {}
    for label, display in LABEL_ORDER:
        prior_rgb = render_prior_rgb(prior_map_for_label(factor_labels, gene_scores.get(label, np.zeros(n_factors))))
        factor_context = (
            f"Target-supporting factors: {label_factor_hints(semantic_factors, label, top_n=4) or 'none'}\n"
            f"Full color legend:\n{full_legend_text}"
        )
        system, prompt = prompt_for_label(label, factor_context)
        raw = vlm_generate(processor, model, args.device, make_context_images(he, ficture, prior_rgb), system, prompt, args.max_new_tokens)
        regions = parse_grounding(raw)
        direct_mask, boxes = rasterize_regions(regions, shape)
        grounding_rows.append({
            "label": label,
            "display": display,
            "raw_response": raw,
            "n_regions": len(regions),
            "boxes_json": json.dumps(boxes),
        })
        gt = gt_masks[label]
        direct_row = {
            "label": label,
            "display": display,
            "ranker": "vlm_direct_raster",
            "top_k": 0,
            "selected": "vlm_boxes_or_polygons",
            "n_selected": len(boxes),
            **metrics(direct_mask, gt),
        }
        selection_rows.append(direct_row)
        best_rows[label] = direct_row
        for top_k in args.top_ks:
            pred, selected = select_candidates_by_grounding(candidates.get(label, []), direct_mask, shape, top_k, args.max_overlap)
            row = {
                "label": label,
                "display": display,
                "ranker": "vlm_grounded_candidate",
                "top_k": top_k,
                "selected": ";".join(f"{r['source']}/{r['setting']}/{r['candidate_id']}" for r in selected),
                "n_selected": len(selected),
                **metrics(pred, gt),
            }
            selection_rows.append(row)
            if float(row["dice"]) > float(best_rows[label]["dice"]):
                best_rows[label] = row

        color = (0, 185, 255)
        best = best_rows[label]
        if best["ranker"] == "vlm_direct_raster":
            best_mask = direct_mask
        else:
            best_mask, _ = select_candidates_by_grounding(candidates.get(label, []), direct_mask, shape, int(best["top_k"]), args.max_overlap)
        panels = [
            overlay(he, direct_mask, (255, 140, 0), 0.45),
            overlay(ficture, direct_mask, (255, 140, 0), 0.45),
            overlay(he, best_mask, color, 0.45),
            overlay(he, gt, (0, 185, 95), 0.48),
            he,
        ]
        h, w = shape
        canvas = Image.new("RGB", (w * 5, h + 110), "white")
        draw = ImageDraw.Draw(canvas)
        title = f"{display}: VLM direct grounding | Dice {float(best['dice']):.3f} | P {float(best['precision']):.3f} | R {float(best['recall']):.3f} | {best['ranker']} top-{best['top_k']}"
        draw.text((12, 12), title, fill="black")
        for i, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel.astype(np.uint8)), (i * w, 80))
        canvas.save(panel_dir / f"{label}_vlm_direct_5panel.png")

    with (args.output_dir / "vlm_grounding_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grounding_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grounding_rows)
    with (args.output_dir / "vlm_direct_selection_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    with (args.output_dir / "best_by_label.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(best_rows.values())).keys()))
        writer.writeheader()
        writer.writerows(best_rows.values())

    lines = [
        "# VLM Direct Grounding",
        "",
        f"Model: `{args.model}`",
        "",
        "| label | best Dice | precision | recall | ranker | top_k |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for label, display in LABEL_ORDER:
        row = best_rows[label]
        lines.append(f"| {display} | {float(row['dice']):.3f} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {row['ranker']} | {row['top_k']} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
