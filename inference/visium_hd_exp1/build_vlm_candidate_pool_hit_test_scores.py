#!/usr/bin/env python3
"""Build an official FICTURE VLM candidate-pool retrieval hit-test set.

This creates a candidate_multimodal_scores.csv shaped like the ranking outputs,
but the score is intentionally uninformative. The hidden truth file stores each
candidate's Dice/precision/recall so a VLM reranker can later be evaluated by
whether high-Dice candidates are retrieved into top-k.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.array(image) > 127


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    inter = float(np.logical_and(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    precision = inter / pred_sum if pred_sum else 0.0
    recall = inter / gt_sum if gt_sum else 0.0
    dice = 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0
    return {"dice": dice, "precision": precision, "recall": recall, "area_frac": pred_sum / pred.size}


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def load_gt_masks(summary_path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    summary = json.loads(summary_path.read_text())
    masks: Dict[str, np.ndarray] = {}
    shape_hw: Tuple[int, int] | None = None
    labels = {slug for slug, _display in LABEL_ORDER}
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        if slug not in labels:
            continue
        path = Path(item["mask_path"])
        mask = np.array(Image.open(path).convert("L")) > 127
        shape_hw = mask.shape if shape_hw is None else shape_hw
        masks[slug] = mask
    if shape_hw is None:
        raise SystemExit(f"No target masks found in {summary_path}")
    return masks, shape_hw


def iter_candidate_masks(report_path: Path) -> Iterable[Tuple[int, Path]]:
    mask_dir = report_path.parent / "candidate_masks"
    for path in sorted(mask_dir.glob("candidate_*.png")):
        try:
            candidate_id = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        yield candidate_id, path


def select_rows(rows: List[dict], per_label: int, top_n: int, mid_n: int, low_n: int) -> List[dict]:
    rows = sorted(rows, key=lambda row: float(row["hidden_dice"]), reverse=True)
    selected: List[dict] = []
    seen = set()

    def add(items: Iterable[dict], bucket: str) -> None:
        for row in items:
            key = (row["run"], row["setting"], row["candidate_id"])
            if key in seen:
                continue
            copy = dict(row)
            copy["sample_bucket"] = bucket
            selected.append(copy)
            seen.add(key)
            if len(selected) >= per_label:
                return

    add(rows[:top_n], "high_dice")
    if len(selected) < per_label and rows:
        middle = sorted(rows, key=lambda row: abs(float(row["hidden_dice"]) - 0.20))
        add(middle[:mid_n], "middle_dice")
    if len(selected) < per_label:
        low = sorted(rows, key=lambda row: float(row["hidden_dice"]))
        add(low[:low_n], "low_dice")
    if len(selected) < per_label:
        add(rows, "fill")
    return selected[:per_label]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=36)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--mid-n", type=int, default=12)
    parser.add_argument("--low-n", type=int, default=12)
    parser.add_argument("--ranker", default="vlm_hit_test")
    args = parser.parse_args()

    gt_masks, shape_hw = load_gt_masks(args.summary_path)
    by_label: Dict[str, List[dict]] = {slug: [] for slug, _display in LABEL_ORDER}

    reports = sorted(args.candidate_root.glob("ficture_official_*/**/candidate_report.json"))
    if not reports:
        raise SystemExit(f"No official candidate_report.json found under {args.candidate_root}")

    for report_path in reports:
        run = report_path.parent.parent.name
        setting = report_path.parent.name
        for candidate_id, mask_path in iter_candidate_masks(report_path):
            pred = read_mask(mask_path, shape_hw)
            for slug, display in LABEL_ORDER:
                m = metrics(pred, gt_masks[slug])
                by_label[slug].append(
                    {
                        "label": slug,
                        "display": display,
                        "source": "ficture",
                        "run": run,
                        "setting": setting,
                        "candidate_id": str(candidate_id),
                        "mask_path": str(mask_path),
                        "hidden_dice": f"{m['dice']:.9f}",
                        "hidden_precision": f"{m['precision']:.9f}",
                        "hidden_recall": f"{m['recall']:.9f}",
                        "area_frac": f"{m['area_frac']:.9f}",
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: List[dict] = []
    for slug, _display in LABEL_ORDER:
        selected.extend(select_rows(by_label[slug], args.per_label, args.top_n, args.mid_n, args.low_n))

    score_fields = [
        "label",
        "source",
        "run",
        "setting",
        "candidate_id",
        "mask_path",
        "ranker",
        "score",
        "molecular_score",
        "factor_semantic_score",
        "he_score",
        "shape_score",
        "area_frac",
    ]
    truth_fields = [
        "label",
        "display",
        "source",
        "run",
        "setting",
        "candidate_id",
        "mask_path",
        "sample_bucket",
        "hidden_dice",
        "hidden_precision",
        "hidden_recall",
        "area_frac",
    ]
    with (args.output_dir / "candidate_multimodal_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "label": row["label"],
                    "source": row["source"],
                    "run": row["run"],
                    "setting": row["setting"],
                    "candidate_id": row["candidate_id"],
                    "mask_path": row["mask_path"],
                    "ranker": args.ranker,
                    "score": "0.5",
                    "molecular_score": "0.0",
                    "factor_semantic_score": "0.0",
                    "he_score": "0.0",
                    "shape_score": "0.0",
                    "area_frac": row["area_frac"],
                }
            )
    with (args.output_dir / "hidden_candidate_truth.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=truth_fields)
        writer.writeheader()
        writer.writerows(selected)

    summary_rows = []
    for slug, display in LABEL_ORDER:
        label_rows = [row for row in selected if row["label"] == slug]
        best = max(label_rows, key=lambda row: float(row["hidden_dice"]))
        summary_rows.append(
            {
                "label": slug,
                "display": display,
                "n_candidates": len(label_rows),
                "best_hidden_dice": best["hidden_dice"],
                "best_hidden_precision": best["hidden_precision"],
                "best_hidden_recall": best["hidden_recall"],
                "best_source": f"{best['source']}/{best['setting']}/{best['candidate_id']}",
            }
        )
    with (args.output_dir / "hit_test_pool_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(args.output_dir)


if __name__ == "__main__":
    main()
