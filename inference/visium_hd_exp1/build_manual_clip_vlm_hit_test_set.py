#!/usr/bin/env python3
"""Build a small manual official-FICTURE CLIP/VLM candidate hit-test set.

The goal is not to find a new best segmentation. It is to test whether CLIP and
VLM can do the same retrieval task on the same candidate list: rank good masks
above bad masks inside the official FICTURE candidate pool.

Candidates are sampled from candidate_report.json best-candidate entries so we
avoid scanning every mask in the pool. For each target class we include:

- high: strong candidates for that class,
- mid: weaker candidates for that class,
- bad_cross: candidates that were good for another class but should be bad here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.array(image) > 127


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


def load_gt_masks(summary_path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    summary = json.loads(summary_path.read_text())
    wanted = {slug for slug, _display in LABEL_ORDER}
    masks: Dict[str, np.ndarray] = {}
    shape_hw: Tuple[int, int] | None = None
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        if slug not in wanted:
            continue
        path = Path(item["mask_path"])
        if not path.exists():
            path = summary_path.parent / item["mask_path"]
        mask = np.array(Image.open(path).convert("L")) > 127
        shape_hw = mask.shape if shape_hw is None else shape_hw
        masks[slug] = mask
    if shape_hw is None:
        raise SystemExit(f"No official target masks found in {summary_path}")
    missing = sorted(wanted - set(masks))
    if missing:
        raise SystemExit(f"Missing target masks in {summary_path}: {missing}")
    return masks, shape_hw


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
    def key(self) -> Tuple[str, str, str, int]:
        return (self.source, self.run, self.setting, self.candidate_id)


def iter_report_bests(candidate_root: Path) -> Iterable[ReportBest]:
    for report_path in sorted(candidate_root.glob("ficture_official_*/**/candidate_report.json")):
        obj = json.loads(report_path.read_text())
        run = report_path.parent.parent.name
        setting = report_path.parent.name
        mask_dir = report_path.parent / "candidate_masks"
        for label_item in obj.get("labels", []):
            slug = label_item.get("slug") or slugify(label_item.get("label", ""))
            if slug not in {label for label, _display in LABEL_ORDER}:
                continue
            idx = int(label_item["best_candidate_index"])
            mask_path = mask_dir / f"candidate_{idx:03d}.png"
            if not mask_path.exists():
                continue
            m = label_item["best_candidate_metrics"]
            yield ReportBest(
                source="ficture",
                run=run,
                setting=setting,
                own_label=slug,
                candidate_id=idx,
                mask_path=mask_path,
                own_dice=float(m.get("dice", 0.0)),
                own_precision=float(m.get("precision", 0.0)),
                own_recall=float(m.get("recall", 0.0)),
            )


def load_clip_lookup(score_csvs: List[Path]) -> Dict[Tuple[str, str, str, int], dict]:
    lookup: Dict[Tuple[str, str, str, int], dict] = {}
    preferred_rankers = [
        "salip_clip",
        "factor_semantic_clip",
        "salip_clip_molecular",
        "multimodal_equal",
        "molecular_only",
    ]
    ranker_priority = {name: i for i, name in enumerate(preferred_rankers)}
    best_priority: Dict[Tuple[str, str, str, int], int] = {}
    for score_csv in score_csvs:
        if not score_csv.exists():
            continue
        with score_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    key = (
                        row["label"],
                        row["run"],
                        row["setting"],
                        int(row["candidate_id"]),
                    )
                except Exception:
                    continue
                priority = ranker_priority.get(row.get("ranker", ""), 999)
                if key in lookup and priority >= best_priority.get(key, 999):
                    continue
                lookup[key] = row
                best_priority[key] = priority
    return lookup


def choose_by_score(rows: List[dict], n: int, reverse: bool = True) -> List[dict]:
    out: List[dict] = []
    seen: set[Tuple[str, str, str, int]] = set()
    for row in sorted(rows, key=lambda r: float(r["hidden_dice"]), reverse=reverse):
        key = (row["source"], row["run"], row["setting"], int(row["candidate_id"]))
        if key in seen:
            continue
        out.append(row)
        seen.add(key)
        if len(out) >= n:
            break
    return out


def choose_mid(rows: List[dict], n: int) -> List[dict]:
    if not rows:
        return []
    values = sorted(float(row["hidden_dice"]) for row in rows)
    target = values[len(values) // 2]
    out: List[dict] = []
    seen: set[Tuple[str, str, str, int]] = set()
    for row in sorted(rows, key=lambda r: abs(float(r["hidden_dice"]) - target)):
        key = (row["source"], row["run"], row["setting"], int(row["candidate_id"]))
        if key in seen:
            continue
        out.append(row)
        seen.add(key)
        if len(out) >= n:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-score-csv", type=Path, action="append", default=[])
    parser.add_argument("--high-per-label", type=int, default=2)
    parser.add_argument("--mid-per-label", type=int, default=2)
    parser.add_argument("--bad-per-label", type=int, default=2)
    parser.add_argument("--uniform-ranker", default="manual_hit_test_uniform")
    args = parser.parse_args()

    gt_masks, shape_hw = load_gt_masks(args.summary_path)
    report_bests = list(iter_report_bests(args.candidate_root))
    if not report_bests:
        raise SystemExit(f"No candidate report bests found under {args.candidate_root}")

    mask_cache: Dict[Path, np.ndarray] = {}

    def get_mask(path: Path) -> np.ndarray:
        if path not in mask_cache:
            mask_cache[path] = read_mask(path, shape_hw)
        return mask_cache[path]

    all_rows: Dict[str, List[dict]] = {slug: [] for slug, _display in LABEL_ORDER}
    for record in report_bests:
        pred = get_mask(record.mask_path)
        for target, display in LABEL_ORDER:
            m = metrics(pred, gt_masks[target])
            all_rows[target].append(
                {
                    "label": target,
                    "display": display,
                    "source": record.source,
                    "run": record.run,
                    "setting": record.setting,
                    "candidate_id": str(record.candidate_id),
                    "mask_path": str(record.mask_path),
                    "candidate_original_best_label": record.own_label,
                    "candidate_original_best_dice": f"{record.own_dice:.9f}",
                    "hidden_dice": f"{m['dice']:.9f}",
                    "hidden_precision": f"{m['precision']:.9f}",
                    "hidden_recall": f"{m['recall']:.9f}",
                    "area_frac": f"{m['area_frac']:.9f}",
                }
            )

    clip_lookup = load_clip_lookup(args.clip_score_csv)
    selected: List[dict] = []
    for target, _display in LABEL_ORDER:
        rows = all_rows[target]
        same_label = [row for row in rows if row["candidate_original_best_label"] == target]
        cross_label = [
            row
            for row in rows
            if row["candidate_original_best_label"] != target
            and float(row["candidate_original_best_dice"]) >= 0.20
        ]
        for bucket, bucket_rows in [
            ("high_same_label", choose_by_score(same_label, args.high_per_label, reverse=True)),
            ("mid_same_label", choose_mid(same_label, args.mid_per_label)),
            ("bad_cross_label", choose_by_score(cross_label, args.bad_per_label, reverse=False)),
        ]:
            for row in bucket_rows:
                copy = dict(row)
                copy["sample_bucket"] = bucket
                key = (copy["label"], copy["run"], copy["setting"], int(copy["candidate_id"]))
                clip_row = clip_lookup.get(key, {})
                for field in [
                    "clip_score",
                    "ficture_clip_score",
                    "molecular_score",
                    "factor_semantic_score",
                    "he_score",
                    "shape_score",
                    "dominant_ficture_factors",
                    "target_factor_hints",
                ]:
                    copy[field] = clip_row.get(field, "")
                selected.append(copy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth_fields = [
        "label",
        "display",
        "source",
        "run",
        "setting",
        "candidate_id",
        "mask_path",
        "sample_bucket",
        "candidate_original_best_label",
        "candidate_original_best_dice",
        "hidden_dice",
        "hidden_precision",
        "hidden_recall",
        "area_frac",
        "clip_score",
        "ficture_clip_score",
        "molecular_score",
        "factor_semantic_score",
        "he_score",
        "shape_score",
        "dominant_ficture_factors",
        "target_factor_hints",
    ]
    with (args.output_dir / "hidden_candidate_truth.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=truth_fields)
        writer.writeheader()
        writer.writerows(selected)

    score_fields = [
        "source",
        "run",
        "setting",
        "candidate_id",
        "mask_path",
        "prompt_type",
        "area_frac",
        "bbox_fill",
        "elongation",
        "boundary_gradient",
        "dark_frac",
        "red_frac",
        "bright_frac",
        "cross_source_iou",
        "dominant_ficture_factors",
        "label",
        "ranker",
        "score",
        "molecular_score",
        "factor_semantic_score",
        "he_score",
        "shape_score",
        "clip_score",
        "ficture_clip_score",
        "target_factor_hints",
    ]
    with (args.output_dir / "candidate_multimodal_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        for row in selected:
            base = {
                "source": row["source"],
                "run": row["run"],
                "setting": row["setting"],
                "candidate_id": row["candidate_id"],
                "mask_path": row["mask_path"],
                "prompt_type": "",
                "area_frac": row["area_frac"],
                "bbox_fill": "0.0",
                "elongation": "0.0",
                "boundary_gradient": "0.0",
                "dark_frac": "0.0",
                "red_frac": "0.0",
                "bright_frac": "0.0",
                "cross_source_iou": "0.0",
                "dominant_ficture_factors": row.get("dominant_ficture_factors", ""),
                "label": row["label"],
                "molecular_score": row.get("molecular_score") or "0.0",
                "factor_semantic_score": row.get("factor_semantic_score") or "0.0",
                "he_score": row.get("he_score") or "0.0",
                "shape_score": row.get("shape_score") or "0.0",
                "clip_score": row.get("clip_score") or "0.0",
                "ficture_clip_score": row.get("ficture_clip_score") or "0.0",
                "target_factor_hints": row.get("target_factor_hints", ""),
            }
            clip = float(row.get("clip_score") or 0.0)
            ficture_clip = float(row.get("ficture_clip_score") or 0.0)
            mean_clip = (clip + ficture_clip) / 2.0 if clip or ficture_clip else 0.0
            for ranker, score in [
                (args.uniform_ranker, 0.5),
                ("clip_he", clip),
                ("clip_ficture", ficture_clip),
                ("clip_mean", mean_clip),
            ]:
                out = dict(base)
                out["ranker"] = ranker
                out["score"] = f"{score:.9f}"
                writer.writerow(out)

    summary = []
    for target, display in LABEL_ORDER:
        rows = [row for row in selected if row["label"] == target]
        best = max(rows, key=lambda row: float(row["hidden_dice"]))
        summary.append(
            {
                "label": target,
                "display": display,
                "n_candidates": len(rows),
                "best_hidden_dice": best["hidden_dice"],
                "best_hidden_precision": best["hidden_precision"],
                "best_hidden_recall": best["hidden_recall"],
                "best_sample": f"{best['source']}/{best['setting']}/{best['candidate_id']}",
            }
        )
    with (args.output_dir / "hit_test_pool_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    readme = [
        "# Manual CLIP/VLM Candidate Pool Hit-Test",
        "",
        "This is a small official FICTURE candidate-pool retrieval test.",
        "It uses candidate_report.json best-candidate entries, not deprecated roots.",
        "",
        "Each label has high_same_label, mid_same_label, and bad_cross_label examples.",
        "The hidden truth columns are used only for evaluation after CLIP/VLM scoring.",
        "",
        f"Uniform VLM ranker name: `{args.uniform_ranker}`",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
