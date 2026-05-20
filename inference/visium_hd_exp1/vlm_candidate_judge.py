#!/usr/bin/env python3
"""Use local VLMs to judge SAM/Medical-SAM3 candidate masks.

The VLM is used only as an annotation-free candidate judge. It sees three
modalities for each candidate:
  1. H&E crop with the candidate mask highlighted
  2. FICTURE/factor-map crop with the same candidate highlighted
  3. class-specific gene/FICTURE prior crop with the same candidate highlighted

Annotation masks are loaded only after scoring, to calibrate Dice/precision/recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ficture_factor_semantics import (
    factor_histogram_summary,
    label_factor_hints,
    legend_text,
    load_semantic_legend,
)


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]

LABEL_DESCRIPTIONS = {
    "lung_bronchiola": "bronchiolar airway tissue: airway-like lumen, epithelial lining, branching or folded bronchiole morphology",
    "lung_alveoli_normal_adjacent": "normal adjacent alveoli: porous lung parenchyma, preserved open air spaces, not solid tumor or broad stroma",
    "lung_vessels": "lung vessel tissue: vascular lumen or elongated blood-vessel structure, sometimes containing red blood cells",
    "tumor": "tumor tissue: malignant epithelial/tumor region, denser atypical cellular areas, not normal alveoli or vessel lumen",
    "stroma": "stroma: connective/desmoplastic supporting tissue, fibrous matrix and broad stromal bands, not epithelial tumor nests",
    "immune_infiltration": "immune infiltration: lymphocyte/macrophage-rich inflammatory infiltrate, small dense immune cells or immune aggregates",
}

LABEL_COLORS = {
    "lung_bronchiola": (31, 119, 180),
    "lung_alveoli_normal_adjacent": (245, 245, 220),
    "lung_vessels": (50, 205, 215),
    "tumor": (23, 190, 207),
    "stroma": (188, 189, 34),
    "immune_infiltration": (44, 160, 44),
}


@dataclass
class Candidate:
    key: str
    label: str
    source: str
    run: str
    setting: str
    candidate_id: int
    mask_path: Path
    base_score: float
    molecular_score: float
    factor_semantic_score: float
    he_score: float
    shape_score: float
    area_frac: float


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


def bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return m & ~er


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


def load_summary_masks(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    masks: Dict[str, np.ndarray] = {}
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        path = Path(item["mask_path"])
        if not path.exists():
            path = summary_path.parent / item["mask_path"]
        if slug in {s for s, _ in LABEL_ORDER}:
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


def prior_map_for_label(factor_labels: np.ndarray, score_vec: np.ndarray) -> np.ndarray:
    prior = np.zeros(factor_labels.shape, dtype=np.float32)
    valid = factor_labels >= 0
    if score_vec.size:
        clipped = np.clip(factor_labels[valid], 0, score_vec.size - 1)
        prior[valid] = score_vec[clipped].astype(np.float32)
    return prior


def render_prior_rgb(prior: np.ndarray) -> np.ndarray:
    p = prior.astype(np.float32)
    if p.max() > p.min():
        p = (p - p.min()) / (p.max() - p.min())
    rgb = np.zeros((*p.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * p, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(80 * (1 - p), 0, 80).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1 - p), 0, 255).astype(np.uint8)
    return rgb


def crop_triplet(
    he: np.ndarray,
    ficture: np.ndarray,
    prior_rgb: np.ndarray,
    mask: np.ndarray,
    label: str,
    pad: int = 48,
    max_side: int = 768,
) -> List[Image.Image]:
    x1, y1, x2, y2 = bbox(mask)
    h, w = mask.shape
    if x2 <= x1 or y2 <= y1:
        x1 = y1 = 0
        x2, y2 = w, h
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    color = LABEL_COLORS.get(label, (0, 112, 255))
    panels = [
        overlay(he, mask, color, 0.45)[y1:y2, x1:x2],
        overlay(ficture, mask, color, 0.45)[y1:y2, x1:x2],
        overlay(prior_rgb, mask, (0, 180, 255), 0.50)[y1:y2, x1:x2],
    ]
    out = []
    for panel in panels:
        img = Image.fromarray(panel)
        scale = min(1.0, max_side / max(img.size))
        if scale < 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.BILINEAR)
        out.append(img)
    return out


def load_candidates(score_csvs: Sequence[Path], labels: Sequence[str], per_label: int, rankers: Sequence[str]) -> List[Candidate]:
    by_key: Dict[str, Candidate] = {}
    ranker_set = set(rankers)
    label_set = set(labels)
    for score_csv in score_csvs:
        with score_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] not in label_set or row["ranker"] not in ranker_set:
                    continue
                key = "|".join([row["label"], row["source"], row["run"], row["setting"], row["candidate_id"]])
                score = float(row["score"])
                existing = by_key.get(key)
                if existing is not None and existing.base_score >= score:
                    continue
                by_key[key] = Candidate(
                    key=key,
                    label=row["label"],
                    source=row["source"],
                    run=row["run"],
                    setting=row["setting"],
                    candidate_id=int(row["candidate_id"]),
                    mask_path=Path(row["mask_path"]),
                    base_score=score,
                    molecular_score=float(row.get("molecular_score") or 0.0),
                    factor_semantic_score=float(row.get("factor_semantic_score") or 0.0),
                    he_score=float(row.get("he_score") or 0.0),
                    shape_score=float(row.get("shape_score") or 0.0),
                    area_frac=float(row.get("area_frac") or 0.0),
                )
    selected: List[Candidate] = []
    for label in labels:
        candidates = [c for c in by_key.values() if c.label == label]
        candidates.sort(key=lambda c: c.base_score, reverse=True)
        selected.extend(candidates[:per_label])
    return selected


def build_messages(label: str, candidate: Candidate, factor_context: str = "") -> Tuple[str, str]:
    desc = LABEL_DESCRIPTIONS[label]
    system = (
        "You are a careful pathology image judge. You do not draw masks. "
        "You only score whether a proposed candidate mask matches the requested lung tissue class."
    )
    prompt = f"""
You are given three aligned crops for one candidate mask:
Image 1: H&E histology crop with the candidate mask highlighted.
Image 2: FICTURE spatial transcriptomic factor-map crop with the same mask highlighted.
Image 3: class-specific gene/FICTURE prior crop; warmer colors mean stronger molecular support, same mask highlighted.

Target tissue class: {label}
Meaning: {desc}

Candidate metadata:
- source: {candidate.source}
- prompt setting: {candidate.setting}
- approximate area fraction: {candidate.area_frac:.4f}
- non-VLM molecular score: {candidate.molecular_score:.3f}
- non-VLM factor semantic score: {candidate.factor_semantic_score:.3f}
- non-VLM H&E heuristic score: {candidate.he_score:.3f}
- non-VLM shape score: {candidate.shape_score:.3f}

FICTURE interpretation context:
{factor_context}

Score this candidate for the target class. Prefer masks that cover the real target tissue while avoiding unrelated tissue.
Return only one valid JSON object with numeric values between 0 and 1.
Use this exact key format, but estimate your own values:
{{"score": 0.73, "recall": 0.68, "precision": 0.82, "reason": "short reason"}}
"""
    return system, prompt.strip()


def trim_factor_context(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars].rstrip()
    return head + "\n[Factor legend truncated for this VLM context window.]"


def parse_json_score(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        nums = re.findall(r"0?\.\d+|1\.0|1|0", text)
        score = float(nums[0]) if nums else 0.0
        return {"score": score, "recall": score, "precision": score, "reason": text[:160]}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"score": 0.0, "recall": 0.0, "precision": 0.0, "reason": text[:160]}
    for k in ["score", "recall", "precision"]:
        try:
            data[k] = max(0.0, min(1.0, float(data.get(k, 0.0))))
        except Exception:
            data[k] = 0.0
    data["reason"] = str(data.get("reason", ""))[:240]
    return data


def load_vlm(model_name: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return processor, model


def vlm_generate(processor, model, device: str, images: List[Image.Image], system: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()


def select_union(scored: Sequence[Tuple[float, Candidate]], get_mask, top_k: int, max_overlap: float, shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, List[Candidate]]:
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
    parser.add_argument("--per-label", type=int, default=24)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12])
    parser.add_argument("--max-overlap", type=float, default=0.82)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-factor-context-chars", type=int, default=2400)
    parser.add_argument("--rankers", nargs="+", default=["molecular_only", "class_weighted", "bav_focus_recall", "bav_focus_balanced", "salip_clip", "precision_gated"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "vlm_triplets"
    panel_dir = args.output_dir / "best_panels"
    crop_dir.mkdir(exist_ok=True)
    panel_dir.mkdir(exist_ok=True)

    factor_labels = np.load(args.factor_label_npy).astype(np.int16)
    target_shape = factor_labels.shape
    he = resize_rgb(np.array(Image.open(args.he_image).convert("RGB")), target_shape)
    ficture = resize_rgb(np.array(Image.open(args.ficture_image).convert("RGB")), target_shape)
    n_factors = int(factor_labels[factor_labels >= 0].max()) + 1 if np.any(factor_labels >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    semantic_factors = load_semantic_legend(args.factor_semantic_legend, n_factors=n_factors)
    full_legend_text = legend_text(semantic_factors) if semantic_factors else "No factor semantic legend was provided."
    priors = {
        slug: render_prior_rgb(prior_map_for_label(factor_labels, gene_scores.get(slug, np.zeros(n_factors))))
        for slug, _display in LABEL_ORDER
    }
    gt_masks = load_summary_masks(args.summary_path, target_shape)
    focus_labels = [slug for slug, _display in LABEL_ORDER]
    candidates = load_candidates(args.score_csv, focus_labels, args.per_label, args.rankers)
    if not candidates:
        raise SystemExit("No candidates selected for VLM judging.")
    print(f"Loaded {len(candidates)} candidates for VLM judging.", flush=True)

    mask_cache: Dict[Path, np.ndarray] = {}

    def get_mask(path: Path) -> np.ndarray:
        if path not in mask_cache:
            mask_cache[path] = resize_bool(read_mask(path), target_shape)
        return mask_cache[path]

    processor, model = load_vlm(args.model, args.device)

    score_rows: List[dict] = []
    for i, cand in enumerate(candidates, start=1):
        mask = get_mask(cand.mask_path)
        images = crop_triplet(he, ficture, priors[cand.label], mask, cand.label)
        vals = factor_labels[mask & (factor_labels >= 0)]
        if vals.size:
            hist = np.bincount(vals.astype(np.int64), minlength=n_factors).astype(np.float64)
            hist = hist / max(hist.sum(), 1.0)
            composition = factor_histogram_summary(hist, semantic_factors, label=cand.label, top_n=5)
        else:
            composition = "candidate mask contains no valid FICTURE factor pixels"
        factor_context = (
            f"Target-supporting factors: {label_factor_hints(semantic_factors, cand.label, top_n=4) or 'none'}\n"
            f"Candidate mask factor composition: {composition}\n"
            f"Full color legend:\n{full_legend_text}"
        )
        factor_context = trim_factor_context(factor_context, args.max_factor_context_chars)
        system, prompt = build_messages(cand.label, cand, factor_context)
        raw = vlm_generate(processor, model, args.device, images, system, prompt, args.max_new_tokens)
        parsed = parse_json_score(raw)
        fused = (
            0.50 * float(parsed["score"])
            + 0.18 * cand.molecular_score
            + 0.14 * cand.factor_semantic_score
            + 0.10 * cand.shape_score
            + 0.08 * cand.he_score
        )
        row = {
            "label": cand.label,
            "source": cand.source,
            "run": cand.run,
            "setting": cand.setting,
            "candidate_id": cand.candidate_id,
            "mask_path": str(cand.mask_path),
            "base_score": cand.base_score,
            "factor_semantic_score": cand.factor_semantic_score,
            "vlm_score": parsed["score"],
            "vlm_recall": parsed["recall"],
            "vlm_precision": parsed["precision"],
            "vlm_fused_score": fused,
            "reason": parsed["reason"],
            "raw_response": raw,
        }
        score_rows.append(row)
        if i <= 12:
            label_dir = crop_dir / cand.label
            label_dir.mkdir(exist_ok=True)
            for j, img in enumerate(images, start=1):
                img.save(label_dir / f"{i:03d}_{cand.source}_{cand.setting}_{cand.candidate_id}_img{j}.png")
        print(f"VLM scored {i}/{len(candidates)} {cand.label} {cand.source}/{cand.setting}/{cand.candidate_id}: {parsed['score']}", flush=True)

    fieldnames = list(score_rows[0].keys())
    with (args.output_dir / "vlm_candidate_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    cand_by_key = {c.key: c for c in candidates}
    by_label: Dict[str, List[Tuple[float, Candidate]]] = {}
    by_label_fused: Dict[str, List[Tuple[float, Candidate]]] = {}
    for row in score_rows:
        key = "|".join([row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"])])
        cand = cand_by_key[key]
        by_label.setdefault(row["label"], []).append((float(row["vlm_score"]), cand))
        by_label_fused.setdefault(row["label"], []).append((float(row["vlm_fused_score"]), cand))

    selection_rows: List[dict] = []
    best_rows: Dict[str, dict] = {}
    for slug, display in LABEL_ORDER:
        gt = gt_masks.get(slug)
        if gt is None:
            continue
        for ranker_name, scored_lookup in [("vlm_only", by_label), ("vlm_fused", by_label_fused)]:
            scored = scored_lookup.get(slug, [])
            for top_k in args.top_ks:
                pred, selected = select_union(scored, get_mask, top_k, args.max_overlap, target_shape)
                row = {
                    "label": slug,
                    "display": display,
                    "ranker": ranker_name,
                    "top_k": top_k,
                    "selected": ";".join(f"{c.source}/{c.setting}/{c.candidate_id}" for c in selected),
                    "n_selected": len(selected),
                    **metrics(pred, gt),
                }
                selection_rows.append(row)
                if slug not in best_rows or float(row["dice"]) > float(best_rows[slug]["dice"]):
                    best_rows[slug] = row

    with (args.output_dir / "vlm_selection_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    with (args.output_dir / "best_by_label.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(best_rows.values())).keys()))
        writer.writeheader()
        writer.writerows(best_rows.values())

    for slug, display in LABEL_ORDER:
        if slug not in best_rows:
            continue
        row = best_rows[slug]
        scored_lookup = by_label_fused if row["ranker"] == "vlm_fused" else by_label
        pred, selected = select_union(scored_lookup.get(slug, []), get_mask, int(row["top_k"]), args.max_overlap, target_shape)
        gt = gt_masks[slug]
        color = LABEL_COLORS[slug]
        panels = [
            overlay(he, pred, color, 0.48),
            overlay(ficture, pred, color, 0.48),
            overlay(priors[slug], pred, (0, 180, 255), 0.50),
            overlay(he, gt, (0, 185, 95), 0.48),
            he,
        ]
        h, w = target_shape
        canvas = Image.new("RGB", (w * 5, h + 110), "white")
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{display}: VLM judge | Dice {float(row['dice']):.3f} | "
            f"Precision {float(row['precision']):.3f} | Recall {float(row['recall']):.3f} | "
            f"{row['ranker']} top-{row['top_k']}"
        )
        draw.text((12, 12), title, fill="black")
        for i, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel.astype(np.uint8)), (i * w, 80))
        canvas.save(panel_dir / f"{slug}_vlm_5panel.png")

    lines = [
        "# VLM Candidate Judge",
        "",
        f"Model: `{args.model}`",
        "",
        "| label | best Dice | precision | recall | ranker | top_k |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for slug, display in LABEL_ORDER:
        if slug in best_rows:
            row = best_rows[slug]
            lines.append(f"| {display} | {float(row['dice']):.3f} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {row['ranker']} | {row['top_k']} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
