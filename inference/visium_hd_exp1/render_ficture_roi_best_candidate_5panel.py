#!/usr/bin/env python3
"""Render best single-candidate ROI FICTURE visuals.

This script summarizes completed ROI FICTURE SAM/Medical-SAM3 candidate reports
and creates one 5-panel figure per tissue class:
candidate on H&E ROI, candidate only, annotation on H&E ROI, annotation only,
and H&E ROI only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(PROJECT_ROOT / "inference" / "visium_hd_exp1"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam3_inference import SAM3Model, resize_mask


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


def load_mask(path: Path) -> np.ndarray:
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


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask.astype(bool)] = (1.0 - alpha) * out[mask.astype(bool)] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def solid_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return out


def quantile(values: List[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def resolve_path(path_value: str, summary_parent: Path) -> Path:
    p = Path(path_value)
    if p.exists():
        return p
    rel = summary_parent / p
    if rel.exists():
        return rel
    cwd_rel = PROJECT_ROOT / p
    if cwd_rel.exists():
        return cwd_rel
    return p


def read_reports(roots: Iterable[Path]) -> List[Tuple[Path, dict]]:
    reports: List[Tuple[Path, dict]] = []
    for root in roots:
        for path in sorted(root.glob("**/candidate_report.json")):
            try:
                reports.append((path, json.loads(path.read_text())))
            except Exception as exc:
                print(f"Skipping {path}: {exc}", file=sys.stderr)
    return reports


def load_candidate_from_saved(run_dir: Path, best_idx: int) -> np.ndarray | None:
    path = run_dir / "candidate_masks" / f"candidate_{best_idx:03d}.png"
    if path.exists():
        return load_mask(path)
    return None


def reconstruct_candidate(report: dict, report_path: Path, best_idx: int, output_path: Path) -> np.ndarray:
    metadata_path = report_path.parent / "candidate_metadata.csv"
    with metadata_path.open() as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if int(r["candidate_id"]) == best_idx), None)
    if row is None:
        raise RuntimeError(f"candidate_id={best_idx} not found in {metadata_path}")

    shape_hw = tuple(report["image_shape"])
    image = np.array(Image.open(report["image_path"]).convert("RGB"))
    image = resize_rgb(image, shape_hw)
    model = SAM3Model(confidence_threshold=0.1, checkpoint_path=report["checkpoint"], device="cuda")
    state = model.encode_image(image)

    if row["prompt_type"] == "point":
        point = (int(float(row["point_x"])), int(float(row["point_y"])))
        pred = model.predict_points(state, [point], [1], shape_hw)
    else:
        box = (
            int(float(row["prompt_box_x1"])),
            int(float(row["prompt_box_y1"])),
            int(float(row["prompt_box_x2"])),
            int(float(row["prompt_box_y2"])),
        )
        pred = model.predict_box(state, box, shape_hw)
    if pred is None:
        raise RuntimeError(f"Could not reconstruct candidate {best_idx} for {report_path}")
    if pred.shape != shape_hw:
        pred = resize_mask(pred.astype(np.uint8), shape_hw).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred.astype(np.uint8) * 255).save(output_path)
    return pred.astype(bool)


def method_label(setting: str) -> str:
    model = "Medical-SAM3" if setting.startswith("medical") else "SAM"
    if "point" in setting:
        spacing = re.search(r"point(\d+)", setting)
        if spacing:
            return f"{model} on FICTURE ROI, point prompts ({spacing.group(1)} px spacing)"
        return f"{model} on FICTURE ROI, point prompts"
    if "box" in setting:
        box_size = re.search(r"(?:box|b)(\d+)_s(\d+)", setting)
        if box_size:
            return f"{model} on FICTURE ROI, box prompts ({box_size.group(1)} px box, {box_size.group(2)} px stride)"
        return f"{model} on FICTURE ROI, box prompts"
    return f"{model} on FICTURE ROI"


def render_panel(
    label_display: str,
    he: np.ndarray,
    candidate: np.ndarray,
    annotation: np.ndarray,
    metrics: Dict[str, float],
    method: str,
    output_path: Path,
) -> None:
    cand_color = (0, 112, 255)
    ann_color = (0, 185, 95)
    fig, axes = plt.subplots(1, 5, figsize=(24, 8))
    panels = [
        (overlay(he, candidate, cand_color), "Candidate on H&E ROI", "blue"),
        (solid_mask(candidate, cand_color), "Candidate only", "blue"),
        (overlay(he, annotation, ann_color), "Annotation on H&E ROI", "green"),
        (solid_mask(annotation, ann_color), "Annotation only", "green"),
        (he, "H&E ROI only", "black"),
    ]
    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, color=color, loc="left")
        ax.axis("off")
    fig.suptitle(
        f"{label_display}: ROI FICTURE candidate comparison\n"
        f"Dice {metrics['dice']:.3f} | Precision {metrics['precision']:.3f} | Recall {metrics['recall']:.3f}\n"
        f"{method}",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        required=True,
        help="Root directory containing candidate_report.json files. Can be repeated.",
    )
    parser.add_argument(
        "--he-roi",
        type=Path,
        default=PROJECT_ROOT / "output/visium_hd_exp1/ficture_candidate_pool_input_roi/he_roi_matching_ficture_coverage.png",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "best_candidate_masks").mkdir(parents=True, exist_ok=True)
    reports = read_reports(args.root)
    if not reports:
        raise SystemExit("No candidate_report.json files found.")

    by_label: Dict[str, List[dict]] = {}
    for report_path, report in reports:
        for label_item in report["labels"]:
            metrics = label_item["best_candidate_metrics"]
            by_label.setdefault(label_item["label"], []).append(
                {
                    "report_path": report_path,
                    "report": report,
                    "label_item": label_item,
                    "dice": float(metrics["dice"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "setting": report_path.parent.name,
                    "job": report_path.parts[-3],
                    "best_idx": label_item["best_candidate_index"],
                    "n_candidates": report["n_candidates_after_nms"],
                }
            )

    he_orig = np.array(Image.open(args.he_roi).convert("RGB"))
    rows: List[dict] = []
    for label_key, display in LABEL_ORDER:
        items = by_label[label_key]
        best = max(items, key=lambda item: item["dice"])
        report = best["report"]
        report_path = best["report_path"]
        shape_hw = tuple(report["image_shape"])
        he = resize_rgb(he_orig, shape_hw)

        summary_path = Path(report["summary_path"])
        summary = json.loads(summary_path.read_text())
        mask_path = next(item["mask_path"] for item in summary["labels"] if item["label"] == label_key)
        annotation = resize_bool(load_mask(resolve_path(mask_path, summary_path.parent)), shape_hw)

        best_idx = int(best["best_idx"])
        candidate = load_candidate_from_saved(report_path.parent, best_idx)
        if candidate is None:
            candidate = reconstruct_candidate(
                report,
                report_path,
                best_idx,
                args.output_dir / "best_candidate_masks" / f"{label_key}.png",
            )
        else:
            candidate = resize_bool(candidate, shape_hw)
            Image.fromarray(candidate.astype(np.uint8) * 255).save(
                args.output_dir / "best_candidate_masks" / f"{label_key}.png"
            )

        method = method_label(best["setting"])
        render_panel(
            display,
            he,
            candidate,
            annotation,
            {"dice": best["dice"], "precision": best["precision"], "recall": best["recall"]},
            method,
            args.output_dir / "five_panel" / f"{label_key}_5panel.png",
        )

        dice_values = [item["dice"] for item in items]
        rows.append(
            {
                "tissue class": display,
                "best Dice": best["dice"],
                "Precision": best["precision"],
                "Recall": best["recall"],
                "25%": quantile(dice_values, 0.25),
                "median": quantile(dice_values, 0.50),
                "75%": quantile(dice_values, 0.75),
                "best-performing method": method,
                "best report": str(best["report_path"]),
                "best candidate index": best_idx,
                "n candidate masks": best["n_candidates"],
            }
        )

    csv_path = args.output_dir / "roi_ficture_candidate_pool_best_single_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# ROI FICTURE Candidate Mask Performance", ""]
    md_lines.append(
        "| tissue class | best Dice | Precision | Recall | 25% | median | 75% | best-performing method |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        md_lines.append(
            f"| {row['tissue class']} | {row['best Dice']:.3f} | {row['Precision']:.3f} | "
            f"{row['Recall']:.3f} | {row['25%']:.3f} | {row['median']:.3f} | "
            f"{row['75%']:.3f} | {row['best-performing method']} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(md_lines) + "\n")
    print(args.output_dir)
    print(csv_path)


if __name__ == "__main__":
    main()
