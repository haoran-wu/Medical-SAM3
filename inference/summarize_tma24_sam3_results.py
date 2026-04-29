#!/usr/bin/env python3
"""
Summarize local TMA24 SAM3 experiment outputs into CSV and Markdown tables.

The script reads ignored `output/**/experiment_summary.json` files and writes a
compact summary for slide/table use. It does not require model weights.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_EXPERIMENTS = [
    ("medical_sam3_whole_image", "output/hpc_tma24_whole_image_prompts/experiment_summary.json"),
    ("base_sam3_whole_image", "output/hpc_tma24_base_sam3_whole_image/experiment_summary.json"),
    ("base_sam3_left_to_right_box", "output/hpc_tma24_base_sam3_left_right_box/experiment_summary.json"),
    ("base_sam3_dense_points_step64", "output/hpc_tma24_base_sam3_dense_points_step64/experiment_summary.json"),
    ("base_sam3_dense_boxes_256_stride128", "output/hpc_tma24_base_sam3_dense_boxes_256_stride128/experiment_summary.json"),
]


def metric_block(row: Dict[str, object], *names: str) -> Optional[Dict[str, float]]:
    for name in names:
        value = row.get(name)
        if isinstance(value, dict):
            return value  # type: ignore[return-value]
    return None


def get_metric(metrics: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if not metrics:
        return None
    value = metrics.get(key)
    return float(value) if value is not None else None


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def load_rows(name: str, path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text())
    rows: List[Dict[str, object]] = []
    for label_row in data.get("labels", []):
        label = label_row.get("label", "")

        if "box_prompt_metrics" in label_row or "box_text_prompt_metrics" in label_row:
            box = metric_block(label_row, "box_prompt_metrics")
            text = metric_block(label_row, "text_prompt_metrics")
            joint = metric_block(label_row, "box_text_prompt_metrics", "joint_box_text_metrics")
            rows.append(
                {
                    "experiment": name,
                    "mode": "direct_prompt",
                    "label": label,
                    "box_dice": get_metric(box, "dice"),
                    "text_dice": get_metric(text, "dice"),
                    "box_text_dice": get_metric(joint, "dice"),
                    "box_iou": get_metric(box, "iou"),
                    "text_iou": get_metric(text, "iou"),
                    "box_text_iou": get_metric(joint, "iou"),
                    "box_recall": get_metric(box, "recall"),
                    "text_recall": get_metric(text, "recall"),
                    "box_text_recall": get_metric(joint, "recall"),
                    "best_prompt": label_row.get("box_xyxy"),
                }
            )
            continue

        if "prompt_metrics_heldout" in label_row or "box_prompt_metrics_right_half" in label_row:
            prompt = metric_block(label_row, "prompt_metrics_heldout", "box_prompt_metrics_right_half")
            text = metric_block(label_row, "text_metrics_heldout", "text_prompt_metrics_right_half")
            joint = metric_block(label_row, "joint_metrics_heldout", "joint_box_text_metrics_right_half")
            rows.append(
                {
                    "experiment": name,
                    "mode": "cross_region",
                    "label": label,
                    "box_dice": get_metric(prompt, "dice"),
                    "text_dice": get_metric(text, "dice"),
                    "box_text_dice": get_metric(joint, "dice"),
                    "box_iou": get_metric(prompt, "iou"),
                    "text_iou": get_metric(text, "iou"),
                    "box_text_iou": get_metric(joint, "iou"),
                    "box_recall": get_metric(prompt, "recall"),
                    "text_recall": get_metric(text, "recall"),
                    "box_text_recall": get_metric(joint, "recall"),
                    "best_prompt": label_row.get("prompt_bboxes_xyxy") or label_row.get("left_bboxes_xyxy"),
                }
            )
            continue

        if "best_single_point_metrics" in label_row:
            union = metric_block(label_row, "union_metrics")
            best = metric_block(label_row, "best_single_point_metrics")
            rows.append(
                {
                    "experiment": name,
                    "mode": "dense_points",
                    "label": label,
                    "union_dice": get_metric(union, "dice"),
                    "union_recall": get_metric(union, "recall"),
                    "best_dice": get_metric(best, "dice"),
                    "best_iou": get_metric(best, "iou"),
                    "best_recall": get_metric(best, "recall"),
                    "best_prompt": label_row.get("best_point_xy"),
                }
            )
            continue

        if "best_single_box_metrics" in label_row:
            union = metric_block(label_row, "union_metrics")
            best = metric_block(label_row, "best_single_box_metrics")
            rows.append(
                {
                    "experiment": name,
                    "mode": "dense_boxes",
                    "label": label,
                    "union_dice": get_metric(union, "dice"),
                    "union_recall": get_metric(union, "recall"),
                    "best_dice": get_metric(best, "dice"),
                    "best_iou": get_metric(best, "iou"),
                    "best_recall": get_metric(best, "recall"),
                    "best_prompt": label_row.get("best_box_xyxy"),
                }
            )
    return rows


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    headers = [
        "experiment",
        "mode",
        "label",
        "box_dice",
        "text_dice",
        "box_text_dice",
        "union_dice",
        "best_dice",
        "best_prompt",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value) if value is not None else "")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TMA24 SAM3 experiment outputs.")
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--csv-path", type=Path, default=Path("output/tma24_sam3_results_summary.csv"))
    parser.add_argument("--md-path", type=Path, default=Path("output/tma24_sam3_results_summary.md"))
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for name, relative_path in DEFAULT_EXPERIMENTS:
        rows.extend(load_rows(name, args.output_root / relative_path))

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.md_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "mode",
        "label",
        "box_dice",
        "text_dice",
        "box_text_dice",
        "box_iou",
        "text_iou",
        "box_text_iou",
        "box_recall",
        "text_recall",
        "box_text_recall",
        "union_dice",
        "union_recall",
        "best_dice",
        "best_iou",
        "best_recall",
        "best_prompt",
    ]
    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    write_markdown(rows, args.md_path)
    print(f"Wrote {len(rows)} rows to {args.csv_path}")
    print(f"Wrote Markdown table to {args.md_path}")


if __name__ == "__main__":
    main()
