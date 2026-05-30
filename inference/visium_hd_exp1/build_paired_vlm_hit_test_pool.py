#!/usr/bin/env python3
"""Build a paired H&E/FICTURE VLM retrieval hit-test pool.

For each target class, this script samples candidate masks from three buckets:

- GOOD: high-Dice candidates for the same target class
- MID: middle-Dice candidates for the same target class
- BAD: candidates that were good for another class but poor for this target

Each selected candidate gets two aligned crop images:

1. H&E crop with the candidate mask highlighted
2. official FICTURE factor-color crop with the same candidate mask highlighted

Ground-truth metrics are written only to the hidden truth file and are not
included in the VLM request text.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ficture_factor_semantics import factor_histogram_summary, label_factor_hints, load_semantic_legend


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
    "lung_vessels": "lung vessel tissue: vascular lumen or elongated blood-vessel structure",
    "tumor": "tumor tissue: malignant epithelial/tumor region with atypical cellular areas",
    "stroma": "stroma: connective or desmoplastic supporting tissue, fibrous matrix and stromal bands",
    "immune_infiltration": "immune infiltration: lymphocyte or macrophage rich inflammatory infiltrate",
}


@dataclass(frozen=True)
class ReportBest:
    source: str
    run: str
    setting: str
    own_label: str
    candidate_id: int
    mask_path: Path
    own_dice: float
    own_precision: float
    own_recall: float

    @property
    def uid(self) -> str:
        safe = [self.source, self.run, self.setting, f"{self.candidate_id:03d}"]
        return "__".join(part.replace("/", "_") for part in safe)


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def parse_candidate_roots(items: Sequence[str]) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected source=/path mapping, got {item}")
        source, path = item.split("=", 1)
        roots[source] = Path(path)
    return roots


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.array(image) > 127


def resize_rgb(image: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape_hw:
        return image
    return np.array(Image.fromarray(image).resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR))


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    inter = float(np.logical_and(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = float(np.logical_or(pred, gt).sum())
    return {
        "dice": 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0,
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_sum if pred_sum else 0.0,
        "recall": inter / gt_sum if gt_sum else 0.0,
        "area_frac": pred_sum / pred.size,
    }


def load_gt_masks(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    wanted = {slug for slug, _display in LABEL_ORDER}
    masks: Dict[str, np.ndarray] = {}
    fallback_dir = summary_path.parent / "cropped_annotation_masks"
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        if slug not in wanted:
            continue
        raw = Path(item["mask_path"])
        for path in [raw, summary_path.parent / item["mask_path"], fallback_dir / raw.name]:
            if path.exists():
                masks[slug] = read_mask(path, shape_hw)
                break
        else:
            raise FileNotFoundError(f"Could not resolve target mask for {slug}")
    missing = sorted(wanted - set(masks))
    if missing:
        raise SystemExit(f"Missing target masks: {missing}")
    return masks


def iter_reports(root: Path) -> Iterable[Path]:
    for report_path in sorted(root.glob("**/candidate_report.json")):
        if "candidate_masks" in report_path.parts:
            continue
        yield report_path


def parse_report_location(root: Path, report_path: Path) -> Tuple[str, str]:
    rel = report_path.relative_to(root)
    if len(rel.parts) == 2:
        return root.name, rel.parts[0]
    if len(rel.parts) >= 3:
        return rel.parts[0], rel.parts[1]
    raise ValueError(f"Cannot parse report location: {report_path}")


def iter_report_bests(source: str, root: Path) -> Iterable[ReportBest]:
    for report_path in iter_reports(root):
        obj = json.loads(report_path.read_text())
        run, setting = parse_report_location(root, report_path)
        mask_dir = report_path.parent / "candidate_masks"
        for item in obj.get("labels", []):
            slug = item.get("slug") or slugify(item.get("label", ""))
            if slug not in {label for label, _display in LABEL_ORDER}:
                continue
            candidate_id = int(item["best_candidate_index"])
            mask_path = mask_dir / f"candidate_{candidate_id:03d}.png"
            if not mask_path.exists():
                continue
            m = item.get("best_candidate_metrics") or {}
            yield ReportBest(
                source=source,
                run=run,
                setting=setting,
                own_label=slug,
                candidate_id=candidate_id,
                mask_path=mask_path,
                own_dice=float(m.get("dice", 0.0)),
                own_precision=float(m.get("precision", 0.0)),
                own_recall=float(m.get("recall", 0.0)),
            )


def load_score_lookup(path: Path | None, ranker_name: str) -> Dict[Tuple[str, str, str, str, str], float]:
    lookup: Dict[Tuple[str, str, str, str, str], float] = {}
    if path is None or not path.exists():
        return lookup
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"]))
            try:
                lookup[key] = float(row["score"])
            except Exception:
                lookup[key] = 0.0
    return lookup


def choose_good(rows: List[dict], n: int) -> List[dict]:
    return sorted(rows, key=lambda row: float(row["hidden_dice"]), reverse=True)[:n]


def choose_mid(rows: List[dict], n: int, exclude: set[Tuple[str, str, str, str]]) -> List[dict]:
    pool = [row for row in rows if row_key(row) not in exclude]
    if not pool:
        return []
    values = sorted(float(row["hidden_dice"]) for row in pool)
    target = values[len(values) // 2]
    return sorted(pool, key=lambda row: abs(float(row["hidden_dice"]) - target))[:n]


def choose_bad(rows: List[dict], n: int, exclude: set[Tuple[str, str, str, str]]) -> List[dict]:
    pool = [row for row in rows if row_key(row) not in exclude]
    return sorted(pool, key=lambda row: float(row["hidden_dice"]))[:n]


def row_key(row: dict) -> Tuple[str, str, str, str]:
    return (row["source"], row["run"], row["setting"], str(row["candidate_id"]))


def bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def overlay_crop(image: np.ndarray, mask: np.ndarray, pad: int = 72, max_side: int = 768) -> Image.Image:
    x1, y1, x2, y2 = bbox(mask)
    h, w = mask.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2].astype(np.float32).copy()
    crop_mask = mask[y1:y2, x1:x2]
    color = np.array([0, 145, 255], dtype=np.float32)
    crop[crop_mask] = 0.55 * crop[crop_mask] + 0.45 * color
    img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.BILINEAR)
    return img


def image_tag(path: str, alt: str) -> str:
    return f'<img src="{html.escape(path)}" alt="{html.escape(alt)}">'


def write_html(output_dir: Path, public_rows: List[dict], per_bucket: int) -> None:
    per_label = per_bucket * 3
    rows_html = []
    for label_slug, display in LABEL_ORDER:
        rows_html.append(f"<h2>{html.escape(display)}</h2>")
        rows_html.append("<div class='grid'>")
        for row in [r for r in public_rows if r["label"] == label_slug]:
            rows_html.append(
                "<article>"
                f"<h3>{html.escape(row['sample_bucket'])}: {html.escape(row['source'])} / {html.escape(row['setting'])} / {html.escape(row['candidate_id'])}</h3>"
                "<div class='imgs'>"
                + image_tag(row["he_crop_rel"], "H&E crop")
                + image_tag(row["ficture_crop_rel"], "FICTURE crop")
                + "</div>"
                f"<p>Target class: <b>{html.escape(row['display'])}</b>. The model sees these two images plus the text prompt; hidden Dice is not shown to the model.</p>"
                f"<details><summary>Prompt text</summary><pre>{html.escape(row['prompt_text'])}</pre></details>"
                "</article>"
            )
        rows_html.append("</div>")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Paired H&E/FICTURE VLM hit-test pool</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #20242a; }}
h1 {{ margin-bottom: 0.2rem; }}
.note {{ max-width: 980px; line-height: 1.55; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
article {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: white; }}
.imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; align-items: start; }}
img {{ width: 100%; height: auto; border: 1px solid #eee; background: #fafafa; }}
pre {{ white-space: pre-wrap; font-size: 12px; }}
</style>
</head>
<body>
<h1>VLM paired-image retrieval hit-test pool</h1>
<div class="note">
<p><b>实验设计：</b>每个类别最多 {per_label} 个候选 mask：{per_bucket} 个 GOOD、{per_bucket} 个 MID、{per_bucket} 个 BAD。每个候选给模型两张对齐 crop：左边是 H&E，右边是同一个 mask 投到官方 FICTURE 彩色图上。模型还会收到目标类别和 FICTURE 颜色/细胞类型/marker gene 的文字提示。人工 annotation/Dice 只在模型打完分后用于评价，不放进 prompt。</p>
<p><b>GOOD/MID/BAD：</b>GOOD 是同类候选里 Dice 高的；MID 是同类候选里中等的；BAD 是别的类别里表现好、但对当前目标类别很差的候选。</p>
</div>
{''.join(rows_html)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", action="append", required=True, help="source=/path/to/root")
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--factor-label-npy", type=Path, default=None)
    parser.add_argument("--factor-semantic-legend", type=Path, default=None)
    parser.add_argument("--source-clip-csv", type=Path, default=None)
    parser.add_argument("--semantic-clip-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--bad-min-own-dice", type=float, default=0.20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "candidate_pair_crops"
    image_dir.mkdir(exist_ok=True)

    ficture = np.array(Image.open(args.ficture_image).convert("RGB"))
    shape_hw = ficture.shape[:2]
    he = resize_rgb(np.array(Image.open(args.he_image).convert("RGB")), shape_hw)
    gt_masks = load_gt_masks(args.summary_path, shape_hw)

    factors = load_semantic_legend(args.factor_semantic_legend, n_factors=12) if args.factor_semantic_legend else []
    factor_labels = np.load(args.factor_label_npy).astype(np.int16) if args.factor_label_npy and args.factor_label_npy.exists() else None
    source_clip = load_score_lookup(args.source_clip_csv, "source_image_salip_clip")
    semantic_clip = load_score_lookup(args.semantic_clip_csv, "openai_clip_ficture_semantic")

    roots = parse_candidate_roots(args.candidate_root)
    records: List[ReportBest] = []
    for source, root in roots.items():
        records.extend(iter_report_bests(source, root))
    if not records:
        raise SystemExit("No candidate report bests found.")

    mask_cache: Dict[Path, np.ndarray] = {}

    def get_mask(path: Path) -> np.ndarray:
        if path not in mask_cache:
            mask_cache[path] = read_mask(path, shape_hw)
        return mask_cache[path]

    all_rows: Dict[str, List[dict]] = {slug: [] for slug, _display in LABEL_ORDER}
    for record in records:
        pred = get_mask(record.mask_path)
        for target, display in LABEL_ORDER:
            m = metrics(pred, gt_masks[target])
            key = (target, record.source, record.run, record.setting, str(record.candidate_id))
            all_rows[target].append(
                {
                    "label": target,
                    "display": display,
                    "source": record.source,
                    "run": record.run,
                    "setting": record.setting,
                    "candidate_id": str(record.candidate_id),
                    "candidate_uid": record.uid,
                    "mask_path": str(record.mask_path),
                    "candidate_original_best_label": record.own_label,
                    "candidate_original_best_dice": f"{record.own_dice:.9f}",
                    "hidden_dice": f"{m['dice']:.9f}",
                    "hidden_iou": f"{m['iou']:.9f}",
                    "hidden_precision": f"{m['precision']:.9f}",
                    "hidden_recall": f"{m['recall']:.9f}",
                    "area_frac": f"{m['area_frac']:.9f}",
                    "source_image_clip_score": f"{source_clip.get(key, 0.0):.9f}",
                    "openai_clip_semantic_score": f"{semantic_clip.get(key, 0.0):.9f}",
                }
            )

    selected: List[dict] = []
    for target, _display in LABEL_ORDER:
        rows = all_rows[target]
        same = [row for row in rows if row["candidate_original_best_label"] == target]
        cross = [
            row
            for row in rows
            if row["candidate_original_best_label"] != target
            and float(row["candidate_original_best_dice"]) >= args.bad_min_own_dice
        ]
        seen: set[Tuple[str, str, str, str]] = set()
        for bucket, bucket_rows in [
            ("GOOD", choose_good(same, args.per_bucket)),
        ]:
            for row in bucket_rows:
                if row_key(row) in seen:
                    continue
                copy = dict(row)
                copy["sample_bucket"] = bucket
                selected.append(copy)
                seen.add(row_key(row))
        for bucket, bucket_rows in [
            ("MID", choose_mid(same, args.per_bucket, seen)),
            ("BAD", choose_bad(cross, args.per_bucket, seen)),
        ]:
            for row in bucket_rows:
                if row_key(row) in seen:
                    continue
                copy = dict(row)
                copy["sample_bucket"] = bucket
                selected.append(copy)
                seen.add(row_key(row))

    # Render one H&E/FICTURE image pair per unique candidate mask.
    rendered: Dict[str, Tuple[str, str, str]] = {}
    for row in selected:
        uid = row["candidate_uid"]
        if uid in rendered:
            row["he_crop_rel"], row["ficture_crop_rel"], row["mask_path"] = rendered[uid]
            continue
        mask_path = Path(row["mask_path"])
        mask = get_mask(mask_path)
        he_name = f"{uid}_he.png"
        fic_name = f"{uid}_ficture.png"
        overlay_crop(he, mask).save(image_dir / he_name)
        overlay_crop(ficture, mask).save(image_dir / fic_name)
        rendered[uid] = (f"candidate_pair_crops/{he_name}", f"candidate_pair_crops/{fic_name}", str(mask_path))
        row["he_crop_rel"], row["ficture_crop_rel"], row["mask_path"] = rendered[uid]

    for row in selected:
        label = row["label"]
        mask = get_mask(Path(row["mask_path"]))
        factor_context = label_factor_hints(factors, label, top_n=4) if factors else ""
        composition = ""
        if factor_labels is not None and factors:
            vals = factor_labels[mask & (factor_labels >= 0)]
            if vals.size:
                hist = np.bincount(vals.astype(np.int64), minlength=max(12, int(vals.max()) + 1)).astype(np.float64)
                hist = hist / max(hist.sum(), 1.0)
                composition = factor_histogram_summary(hist, factors, label=label, top_n=5)
        row["target_description"] = LABEL_DESCRIPTIONS[label]
        row["target_factor_hints"] = factor_context
        row["candidate_factor_composition"] = composition or "no valid FICTURE factor pixels inside this candidate"
        row["prompt_text"] = (
            f"Target tissue class: {row['display']} ({label}).\n"
            f"Plain meaning: {LABEL_DESCRIPTIONS[label]}.\n"
            "You see two aligned images for the same candidate mask: H&E crop and FICTURE factor-color crop. "
            "The blue overlay is the candidate mask to judge.\n"
            f"FICTURE color/cell-type/gene hints for this target: {row['target_factor_hints'] or 'not available'}.\n"
            f"Candidate FICTURE color composition: {row['candidate_factor_composition']}.\n"
            "Score whether the highlighted mask is a good candidate for the target class. "
            "Return JSON only: {\"score\": 0.0-1.0, \"precision\": 0.0-1.0, \"recall\": 0.0-1.0, \"reason\": \"short reason\"}."
        )

    hidden_fields = [
        "label",
        "display",
        "sample_bucket",
        "source",
        "run",
        "setting",
        "candidate_id",
        "candidate_uid",
        "mask_path",
        "candidate_original_best_label",
        "candidate_original_best_dice",
        "hidden_dice",
        "hidden_iou",
        "hidden_precision",
        "hidden_recall",
        "area_frac",
        "source_image_clip_score",
        "openai_clip_semantic_score",
    ]
    public_fields = [
        "label",
        "display",
        "sample_bucket",
        "source",
        "run",
        "setting",
        "candidate_id",
        "candidate_uid",
        "mask_path",
        "he_crop_rel",
        "ficture_crop_rel",
        "target_description",
        "target_factor_hints",
        "candidate_factor_composition",
        "prompt_text",
        "source_image_clip_score",
        "openai_clip_semantic_score",
    ]
    with (args.output_dir / "hidden_candidate_truth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hidden_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in hidden_fields} for row in selected)
    with (args.output_dir / "public_vlm_requests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in public_fields} for row in selected)

    summary_rows = []
    for label, display in LABEL_ORDER:
        label_rows = [row for row in selected if row["label"] == label]
        for bucket in ["GOOD", "MID", "BAD"]:
            bucket_rows = [row for row in label_rows if row["sample_bucket"] == bucket]
            if not bucket_rows:
                continue
            dices = [float(row["hidden_dice"]) for row in bucket_rows]
            summary_rows.append(
                {
                    "label": label,
                    "display": display,
                    "bucket": bucket,
                    "n": len(bucket_rows),
                    "min_hidden_dice": f"{min(dices):.6f}",
                    "median_hidden_dice": f"{np.median(dices):.6f}",
                    "max_hidden_dice": f"{max(dices):.6f}",
                }
            )
    with (args.output_dir / "hit_test_pool_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    write_html(args.output_dir, selected, args.per_bucket)
    readme = [
        "# Paired H&E/FICTURE VLM Hit-Test Pool",
        "",
        "This is a small direct-retrieval test pool, not a final segmentation result.",
        f"Each of the six labels has up to {args.per_bucket} GOOD, {args.per_bucket} MID, and {args.per_bucket} BAD candidate masks when enough examples exist.",
        "Each candidate row has two images: H&E crop and official FICTURE crop with the same mask highlighted.",
        "Hidden Dice/Precision/Recall are used only after model scoring for evaluation.",
    ]
    (args.output_dir / "README_中文.md").write_text("\n".join(readme) + "\n")
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
