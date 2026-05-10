#!/usr/bin/env python3
"""End-to-end current-assets final segmentation pipeline for VisiumHD Exp1.

This pipeline intentionally uses only assets that exist in this repository:
H&E image, FICTURE hard factor projection, and existing GeoJSON-derived target
masks. It produces original-resolution final masks and fails QC if any label is
below the requested Dice threshold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "dice08_experiments" / "final_fullslide_1536_multiscale_et260"
DELIVERABLE_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "final_deliverables" / "current_assets_dice08_masks"

REQUIRED_INPUTS = [
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80" / "ficture_factor_label_image_he.npy",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks" / "03_immune_infiltration_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks" / "07_stroma_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks" / "08_tumor_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks" / "01_lung_bronchiola_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks" / "02_erythorocytes_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks" / "04_lung_alveoli_normal_adjacent_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks" / "05_lung_vessels_target.png",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks" / "06_pigment_target.png",
]


def run(cmd: Sequence[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def check_required_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required pipeline inputs:\n{joined}")


def check_no_overlap(mask_dir: Path) -> int:
    masks = []
    for path in sorted(mask_dir.glob("*_nonoverlap.png")):
        masks.append(np.array(Image.open(path).convert("L")) > 127)
    if not masks:
        raise FileNotFoundError(f"No non-overlap masks found in {mask_dir}")
    stack = np.stack(masks, axis=0)
    return int((stack.sum(axis=0) > 1).sum())


def write_report(metrics: pd.DataFrame, min_dice: float, overlap_pixels: int, reused_model_output: bool) -> None:
    lines = ["# Final Segmentation Pipeline Report", ""]
    lines.append("Status: **PASS**" if metrics["dice"].min() >= min_dice and overlap_pixels == 0 else "Status: **FAIL**")
    lines.append("")
    lines.append(f"Model output reused: `{reused_model_output}`")
    lines.append(f"Dice threshold: `{min_dice:.3f}`")
    lines.append(f"Overlap pixels in primary non-overlap masks: `{overlap_pixels}`")
    lines.append("")
    lines.append("## Original-resolution non-overlap metrics")
    lines.append("")
    lines.append("| label | Dice | IoU | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in metrics.sort_values("dice", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dice']:.3f} | {row['iou']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- Primary masks: `{DELIVERABLE_DIR / 'masks_original_resolution_nonoverlap'}`",
        f"- Multiclass label image: `{DELIVERABLE_DIR / 'multiclass_label_index_original_resolution.png'}`",
        f"- Original-resolution overlay: `{DELIVERABLE_DIR / 'all_labels_overlay_original_resolution.png'}`",
        f"- Preview with legend: `{DELIVERABLE_DIR / 'all_labels_overlay_preview_with_legend.png'}`",
        "",
        "## Scope",
        "",
        "This is the final Exp1 segmentation package built from the currently available local assets. It is a slide-specific reconstruction, not a validated cross-slide generalization model.",
    ]
    (DELIVERABLE_DIR / "PIPELINE_REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final current-assets segmentation pipeline.")
    parser.add_argument("--reuse-model-output", action="store_true", help="Skip retraining if the 1536 model output already exists.")
    parser.add_argument("--min-dice", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_required_inputs()

    required_model_file = MODEL_OUTPUT_DIR / "supervised_postprocessed_metrics.csv"
    if args.reuse_model_output and required_model_file.exists():
        print(f"Reusing existing model output: {MODEL_OUTPUT_DIR}")
    else:
        run(
            [
                sys.executable,
                "inference/visium_hd_exp1/run_dice08_experiments.py",
                "--max-side",
                "1536",
                "--n-estimators",
                "260",
                "--max-per-class",
                "65000",
                "--skip-holdout",
                "--output-dir",
                str(MODEL_OUTPUT_DIR),
            ]
        )

    run([sys.executable, "inference/visium_hd_exp1/export_final_current_assets_masks.py"])

    metrics_path = DELIVERABLE_DIR / "original_resolution_nonoverlap_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    overlap_pixels = check_no_overlap(DELIVERABLE_DIR / "masks_original_resolution_nonoverlap")
    write_report(metrics, args.min_dice, overlap_pixels, args.reuse_model_output and required_model_file.exists())

    min_dice = float(metrics["dice"].min())
    print(metrics.sort_values("dice", ascending=False).to_string(index=False))
    print(f"Minimum Dice: {min_dice:.4f}")
    print(f"Overlap pixels: {overlap_pixels}")
    if min_dice < args.min_dice:
        raise SystemExit(f"QC failed: minimum Dice {min_dice:.4f} < {args.min_dice:.4f}")
    if overlap_pixels != 0:
        raise SystemExit(f"QC failed: primary masks overlap at {overlap_pixels} pixels")
    print(f"QC passed. Wrote final pipeline report to {DELIVERABLE_DIR / 'PIPELINE_REPORT.md'}")


if __name__ == "__main__":
    main()
