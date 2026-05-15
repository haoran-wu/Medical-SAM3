#!/usr/bin/env python3
"""Summarize and visualize SAM3-refined reference-map candidates."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_genes(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not value:
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def candidate_score(item: Dict[str, Any]) -> float:
    metrics = item.get("metrics_vs_resized_reference_component") or {}
    dice = float(metrics.get("dice", 0.0))
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    pixels = max(1, int(item.get("prediction_pixels", 0)))
    size_bonus = min(0.15, math.log10(pixels) / 40.0)
    tiny_penalty = 0.35 if pixels < 250 else 0.0
    return dice + 0.15 * recall + 0.10 * precision + size_bonus - tiny_penalty


def load_report(output_dir: Path) -> Dict[str, Any]:
    report_path = output_dir / "refine_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    return json.loads(report_path.read_text())


def write_ranked_csv(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    path = output_dir / "ranked_sam3_refined_candidates.csv"
    fieldnames = [
        "rank",
        "score",
        "factor",
        "rank_within_factor",
        "dice_vs_coordinate_prior",
        "iou_vs_coordinate_prior",
        "precision_vs_coordinate_prior",
        "recall_vs_coordinate_prior",
        "prediction_pixels",
        "mapped_bbox_xyxy",
        "top_genes_specific",
        "mask_path",
        "panel_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(rows, start=1):
            metrics = item.get("metrics_vs_resized_reference_component") or {}
            genes = parse_genes(item.get("top_genes_specific"))[:6]
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{candidate_score(item):.4f}",
                    "factor": item.get("factor"),
                    "rank_within_factor": item.get("rank_within_factor"),
                    "dice_vs_coordinate_prior": f"{float(metrics.get('dice', 0.0)):.4f}",
                    "iou_vs_coordinate_prior": f"{float(metrics.get('iou', 0.0)):.4f}",
                    "precision_vs_coordinate_prior": f"{float(metrics.get('precision', 0.0)):.4f}",
                    "recall_vs_coordinate_prior": f"{float(metrics.get('recall', 0.0)):.4f}",
                    "prediction_pixels": int(item.get("prediction_pixels", 0)),
                    "mapped_bbox_xyxy": item.get("mapped_bbox_xyxy"),
                    "top_genes_specific": ", ".join(genes),
                    "mask_path": item.get("mask_path"),
                    "panel_path": item.get("panel_path"),
                }
            )
    return path


def make_overview(output_dir: Path, rows: List[Dict[str, Any]], max_panels: int) -> Path:
    selected = rows[:max_panels]
    n = len(selected)
    if n == 0:
        raise ValueError("No candidates to visualize.")
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for idx, item in enumerate(selected):
        ax = axes_arr.ravel()[idx]
        panel_path = output_dir / str(item["panel_path"])
        image = np.array(Image.open(panel_path).convert("RGB"))
        ax.imshow(image)
        metrics = item.get("metrics_vs_resized_reference_component") or {}
        genes = ", ".join(parse_genes(item.get("top_genes_specific"))[:3])
        ax.set_title(
            f"#{idx + 1} Factor {item['factor']} | Dice {float(metrics.get('dice', 0.0)):.2f} | "
            f"R {float(metrics.get('recall', 0.0)):.2f}\n{genes}",
            fontsize=10,
        )
    fig.tight_layout()
    path = output_dir / "top_sam3_refined_candidate_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def crop_bounds_from_item(item: Dict[str, Any], shape: tuple[int, int], margin: float = 0.55) -> tuple[int, int, int, int]:
    bbox = [int(v) for v in item.get("mapped_bbox_xyxy", [0, 0, shape[1] - 1, shape[0] - 1])]
    x0, y0, x1, y1 = bbox
    w = max(24, x1 - x0)
    h = max(24, y1 - y0)
    pad = int(round(max(w, h) * margin))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(shape[1] - 1, x1 + pad)
    y1 = min(shape[0] - 1, y1 + pad)
    return x0, y0, x1, y1


def load_resized_target(report: Dict[str, Any], output_dir: Path) -> np.ndarray:
    target_path = Path(str(report["target_image"]))
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path
    image = Image.open(target_path).convert("RGB")
    target_w, target_h = [int(v) for v in report["target_size"]]
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.array(image)


def make_zoom_overview(
    output_dir: Path,
    report: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_panels: int,
) -> Path:
    selected = rows[:max_panels]
    image = load_resized_target(report, output_dir)
    n = len(selected)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.6 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes_arr.ravel():
        ax.axis("off")

    for idx, item in enumerate(selected):
        ax = axes_arr.ravel()[idx]
        mask = np.array(Image.open(output_dir / str(item["mask_path"])).convert("L")) > 127
        x0, y0, x1, y1 = crop_bounds_from_item(item, image.shape[:2])
        crop = image[y0 : y1 + 1, x0 : x1 + 1].copy().astype(np.float32)
        mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
        color = np.array([0, 220, 255], dtype=np.float32)
        crop[mask_crop] = 0.45 * crop[mask_crop] + 0.55 * color
        ax.imshow(np.clip(crop, 0, 255).astype(np.uint8))
        bx0, by0, bx1, by1 = [int(v) for v in item["mapped_bbox_xyxy"]]
        ax.add_patch(
            plt.Rectangle(
                (bx0 - x0, by0 - y0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                edgecolor="yellow",
                linewidth=1.8,
            )
        )
        metrics = item.get("metrics_vs_resized_reference_component") or {}
        genes = ", ".join(parse_genes(item.get("top_genes_specific"))[:3])
        ax.set_title(
            f"#{idx + 1} F{item['factor']} r{item['rank_within_factor']} | "
            f"Dice {float(metrics.get('dice', 0.0)):.2f} | R {float(metrics.get('recall', 0.0)):.2f}\n{genes}",
            fontsize=10,
        )
    fig.suptitle("Top SAM3-refined H&E candidates, zoomed around mapped reference regions", fontsize=15, y=0.995)
    fig.tight_layout()
    path = output_dir / "top_sam3_refined_candidate_zoom_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    csv_path: Path,
    overview_path: Path,
    zoom_path: Path,
) -> Path:
    path = output_dir / "sam3_refine_summary.md"
    lines = [
        "# SAM3 Reference Candidate Refinement Summary",
        "",
        "This run maps smoothed FICTURE/reference-map components to real H&E coordinates, prompts SAM3 on the H&E image, and ranks the refined masks by agreement with the coordinate prior.",
        "",
        f"- Ranked table: `{csv_path.name}`",
        f"- Overview image: `{overview_path.name}`",
        f"- Zoomed overview image: `{zoom_path.name}`",
        "",
        "## Top Candidates",
        "",
        "| Rank | Factor | Dice | Recall | Pixels | Marker genes |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for idx, item in enumerate(rows[:10], start=1):
        metrics = item.get("metrics_vs_resized_reference_component") or {}
        genes = ", ".join(parse_genes(item.get("top_genes_specific"))[:5])
        lines.append(
            f"| {idx} | {item['factor']} | {float(metrics.get('dice', 0.0)):.3f} | "
            f"{float(metrics.get('recall', 0.0)):.3f} | {int(item.get('prediction_pixels', 0))} | {genes} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SAM3-refined candidates.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top-panels", type=int, default=12)
    args = parser.parse_args()

    report = load_report(args.output_dir)
    rows = list(report.get("candidates", []))
    rows.sort(key=candidate_score, reverse=True)
    csv_path = write_ranked_csv(args.output_dir, rows)
    overview_path = make_overview(args.output_dir, rows, args.top_panels)
    zoom_path = make_zoom_overview(args.output_dir, report, rows, args.top_panels)
    md_path = write_markdown(args.output_dir, rows, csv_path, overview_path, zoom_path)
    print(
        json.dumps(
            {"csv": str(csv_path), "overview": str(overview_path), "zoom_overview": str(zoom_path), "report": str(md_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
