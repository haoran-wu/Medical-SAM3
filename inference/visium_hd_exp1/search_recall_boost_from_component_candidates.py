#!/usr/bin/env python3
"""Search recall-boost candidates starting from an existing final mask.

This is for broad classes such as immune infiltration where the current final
mask is already precision-gated, but we want to test whether a small number of
additional component candidates can raise Recall without an unacceptable
Precision drop.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Policy:
    name: str
    max_precision_drop: float
    min_incremental_precision: float
    min_delta_recall: float
    max_adds: int


POLICIES = [
    Policy("strict", max_precision_drop=0.010, min_incremental_precision=0.35, min_delta_recall=0.001, max_adds=3),
    Policy("balanced", max_precision_drop=0.020, min_incremental_precision=0.25, min_delta_recall=0.001, max_adds=4),
    Policy("recall_push", max_precision_drop=0.040, min_incremental_precision=0.15, min_delta_recall=0.001, max_adds=5),
    Policy("diagnostic_loose", max_precision_drop=0.080, min_incremental_precision=0.05, min_delta_recall=0.001, max_adds=8),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-candidates", type=Path, required=True)
    parser.add_argument("--gt-mask", type=Path, required=True)
    parser.add_argument("--current-mask", type=Path, required=True)
    parser.add_argument("--he-image", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if expected_size is not None and image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "pred_area": float(pred_area),
        "gt_area": float(gt_area),
    }


def candidate_stats(candidate: np.ndarray, current: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    proposed = np.logical_or(current, candidate)
    before = metrics(current, gt)
    after = metrics(proposed, gt)
    new_pixels = np.logical_and(candidate, ~current)
    new_area = int(new_pixels.sum())
    new_tp = int(np.logical_and(new_pixels, gt).sum())
    inc_precision = new_tp / new_area if new_area else 0.0
    return {
        **{f"new_{key}": value for key, value in after.items()},
        "delta_dice": after["dice"] - before["dice"],
        "delta_precision": after["precision"] - before["precision"],
        "delta_recall": after["recall"] - before["recall"],
        "incremental_precision": inc_precision,
        "new_area": float(new_area),
        "new_tp": float(new_tp),
    }


def overlay(base: Image.Image, gt: np.ndarray, pred: np.ndarray, title: str) -> Image.Image:
    rgba = base.convert("RGBA")
    arr = np.zeros((*gt.shape, 4), dtype=np.uint8)
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    arr[tp] = (0, 180, 80, 130)
    arr[fp] = (0, 90, 255, 130)
    arr[fn] = (255, 0, 0, 150)
    out = Image.alpha_composite(rgba, Image.fromarray(arr, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 1180), 34), fill=(255, 255, 255, 225))
    draw.text((10, 10), title + " | green=TP, blue=extra, red=missed", fill=(0, 0, 0, 255))
    return out.convert("RGB")


def row_id(row: pd.Series) -> str:
    source = str(row.get("source", ""))
    component = str(row.get("component", ""))
    rank = str(row.get("rank", ""))
    setting = str(row.get("setting", ""))
    candidate = str(row.get("candidate", ""))
    return f"{source}_{component}_rank{rank}_{setting}_candidate_{candidate}"


def greedy_search(
    rows: list[dict[str, object]],
    current: np.ndarray,
    gt: np.ndarray,
    baseline: dict[str, float],
    policy: Policy,
    expected_size: tuple[int, int],
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, float]]:
    selected: list[dict[str, object]] = []
    union = current.copy()
    used: set[str] = set()
    for _ in range(policy.max_adds):
        best: tuple[float, dict[str, object], np.ndarray, dict[str, float]] | None = None
        for row in rows:
            rid = str(row["row_id"])
            if rid in used:
                continue
            candidate = load_mask(Path(str(row["mask_path"])), expected_size=expected_size)
            stats = candidate_stats(candidate, union, gt)
            if stats["delta_recall"] < policy.min_delta_recall:
                continue
            if stats["incremental_precision"] < policy.min_incremental_precision:
                continue
            if stats["new_precision"] < baseline["precision"] - policy.max_precision_drop:
                continue
            proposed = np.logical_or(union, candidate)
            score = (
                stats["delta_recall"] * max(stats["incremental_precision"], 1e-6)
                + 0.10 * max(stats["delta_dice"], 0.0)
            )
            if best is None or score > best[0]:
                best = (score, row, proposed, stats)
        if best is None:
            break
        _, row, union, stats = best
        used.add(str(row["row_id"]))
        selected.append({**row, **stats})
    return selected, union, metrics(union, gt)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_mask(args.gt_mask)
    expected_size = (gt.shape[1], gt.shape[0])
    current = load_mask(args.current_mask, expected_size=expected_size)
    baseline = metrics(current, gt)

    df = pd.read_csv(args.component_candidates)
    if df.empty:
        raise ValueError(f"No component candidates in {args.component_candidates}")

    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for _, row in df.iterrows():
        path = Path(str(row["mask_path"]))
        if str(path) in seen_paths:
            continue
        if not path.exists():
            continue
        rid = row_id(row)
        seen_paths.add(str(path))
        clean = row.to_dict()
        clean["row_id"] = rid
        rows.append(clean)

    one_add_rows: list[dict[str, object]] = []
    for row in rows:
        candidate = load_mask(Path(str(row["mask_path"])), expected_size=expected_size)
        stats = candidate_stats(candidate, current, gt)
        one_add_rows.append({**row, **stats})
    one_add_rows.sort(
        key=lambda item: (
            float(item["new_precision"]) >= baseline["precision"] - 0.02,
            float(item["delta_recall"]),
            float(item["new_dice"]),
        ),
        reverse=True,
    )
    pd.DataFrame(one_add_rows).to_csv(args.out_dir / "one_add_candidate_effects.csv", index=False)

    summary_rows: list[dict[str, object]] = [
        {
            "policy": "current",
            "selected_count": 0,
            "Dice": f"{baseline['dice']:.6f}",
            "Precision": f"{baseline['precision']:.6f}",
            "Recall": f"{baseline['recall']:.6f}",
            "delta_dice": "0.000000",
            "delta_precision": "0.000000",
            "delta_recall": "0.000000",
            "selected": "",
        }
    ]
    all_selected: list[dict[str, object]] = []
    he_image = Image.open(args.he_image).convert("RGB") if args.he_image else None
    for policy in POLICIES:
        selected, union, final_m = greedy_search(rows, current, gt, baseline, policy, expected_size)
        save_mask(union, args.out_dir / f"{policy.name}_union_mask.png")
        if he_image is not None:
            overlay(
                he_image,
                gt,
                union,
                f"{policy.name}: Dice {final_m['dice']:.3f}, Precision {final_m['precision']:.3f}, Recall {final_m['recall']:.3f}",
            ).save(args.out_dir / f"{policy.name}_union_on_he.png")
        selected_ids = [str(row["row_id"]) for row in selected]
        summary_rows.append(
            {
                "policy": policy.name,
                "selected_count": len(selected),
                "Dice": f"{final_m['dice']:.6f}",
                "Precision": f"{final_m['precision']:.6f}",
                "Recall": f"{final_m['recall']:.6f}",
                "delta_dice": f"{final_m['dice'] - baseline['dice']:.6f}",
                "delta_precision": f"{final_m['precision'] - baseline['precision']:.6f}",
                "delta_recall": f"{final_m['recall'] - baseline['recall']:.6f}",
                "selected": "; ".join(selected_ids),
            }
        )
        for item in selected:
            item = dict(item)
            item["policy"] = policy.name
            all_selected.append(item)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "recall_boost_policy_summary.csv", index=False)
    pd.DataFrame(all_selected).to_csv(args.out_dir / "recall_boost_selected_candidates.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "component_candidates": str(args.component_candidates),
                "gt_mask": str(args.gt_mask),
                "current_mask": str(args.current_mask),
                "he_image": str(args.he_image) if args.he_image else "",
                "n_input_rows": int(len(df)),
                "n_unique_candidate_masks": int(len(rows)),
                "baseline": baseline,
                "policies": [policy.__dict__ for policy in POLICIES],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
