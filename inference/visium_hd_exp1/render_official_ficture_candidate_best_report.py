#!/usr/bin/env python3
"""Render official FICTURE candidate-pool best-by-Dice report.

This is the official deliverable renderer for the filtered, H&E-aligned
FICTURE map. It refuses to run unless the official FICTURE map reports
PASS_OFFICIAL, then summarizes candidate_report.json files by selecting the
highest-Dice single candidate for each tissue class.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT_OVERRIDE", str(Path(__file__).resolve().parent.parent.parent))
)
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(PROJECT_ROOT / "inference" / "visium_hd_exp1"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

OFFICIAL_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
OFFICIAL_INPUT_DIR = PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
OFFICIAL_CANDIDATE_ROOT = PROJECT_ROOT / "results/visium_hd_exp1/ficture_official_filtered_candidate_pool"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/visium_hd_exp1/official_ficture_candidate_best_report"
METHOD_MODALITY_LABEL = "FICTURE ROI"
REPORT_TITLE = "Official FICTURE Candidate Best By Dice"
ARTIFACT_PREFIX = "official_ficture_candidate"

REQUIRED_STATUS_CHECKS = [
    "uses_filtered_png_source",
    "ficture_md_says_filtered_feature_bc_matrix",
    "no_manual_shift",
    "full_canvas_matches_he",
    "formula_uses_microns_per_pixel_and_tissue_hires_scalef",
    "background_false_color_fraction_zero",
    "orientation_selected_fliplr",
    "orientation_qc_best_is_fliplr",
]

FORBIDDEN_PATH_FRAGMENTS = [
    "ficture_coord_scaled_hires",
    "ficture_corrected_candidate_pool_input_roi",
    "agent_verified_filtered_ficture_he_align",
    "deprecated_ficture_pre_official_20260517",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(image.resize((w, h), Image.Resampling.NEAREST)) > 127


def resize_rgb(image: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return np.array(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
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


def resolve_path(path_value: str | None, base: Path | None = None) -> Path:
    if path_value is None:
        raise ValueError("Cannot resolve an empty path")
    p = Path(path_value)
    if p.exists():
        return p
    if base is not None and (base / p).exists():
        return base / p
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    return p


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_no_forbidden_paths(paths: Iterable[Path]) -> None:
    bad = []
    for path in paths:
        text = str(path)
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in text:
                bad.append(text)
    if bad:
        raise SystemExit("Refusing deprecated/non-official FICTURE path(s):\n" + "\n".join(sorted(set(bad))))


def official_preflight(summary_path: Path, input_dir: Path, candidate_root: Path) -> dict:
    check_no_forbidden_paths([summary_path, input_dir, candidate_root])
    summary = load_json(summary_path)
    if summary.get("status") != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {summary.get('status')}")

    checks = summary.get("status_checks", {})
    failed = [key for key in REQUIRED_STATUS_CHECKS if not checks.get(key)]
    if failed:
        raise SystemExit(f"Official FICTURE preflight failed checks: {failed}")

    formula = summary.get("formula", {})
    if formula.get("microns_per_pixel") != 0.2737554241192739:
        raise SystemExit("microns_per_pixel does not match official value.")
    if formula.get("tissue_hires_scalef") != 0.13752006:
        raise SystemExit("tissue_hires_scalef does not match official value.")
    if formula.get("manual_shift_hires_px", {}).get("x") != 0.0 or formula.get("manual_shift_hires_px", {}).get("y") != 0.0:
        raise SystemExit("Manual shift is not zero.")

    he_path = input_dir / "he_roi_matching_official_ficture_coverage.png"
    ficture_path = input_dir / "ficture_official_filtered_roi_rgb.png"
    factor_path = input_dir / "ficture_official_filtered_roi_factor_index.npy"
    region_summary_path = input_dir / "region_summary_official_filtered_roi.json"
    for path in [he_path, ficture_path, factor_path, region_summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    he_size = Image.open(he_path).size
    ficture_size = Image.open(ficture_path).size
    factor_shape = np.load(factor_path, mmap_mode="r").shape
    if he_size != (3144, 3327):
        raise SystemExit(f"Unexpected official H&E ROI size: {he_size}; expected (3144, 3327)")
    if ficture_size != he_size:
        raise SystemExit(f"H&E/FICTURE ROI size mismatch: {he_size} vs {ficture_size}")
    if factor_shape != (he_size[1], he_size[0]):
        raise SystemExit(f"Factor index shape mismatch: {factor_shape} vs {(he_size[1], he_size[0])}")

    return {
        "summary": summary,
        "he_path": he_path,
        "ficture_path": ficture_path,
        "factor_path": factor_path,
        "region_summary_path": region_summary_path,
        "roi_size_wh": he_size,
        "factor_shape": factor_shape,
    }


def read_reports(roots: Iterable[Path]) -> List[Tuple[Path, dict]]:
    reports: List[Tuple[Path, dict]] = []
    for root in roots:
        check_no_forbidden_paths([root])
        for path in sorted(root.glob("**/candidate_report.json")):
            reports.append((path, load_json(path)))
    return reports


def output_ref(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def method_label(setting: str) -> str:
    model = "Medical-SAM3" if setting.startswith("medical") else "SAM"
    point_step = re.search(r"(?:points?_step|point)(\d+)", setting)
    if "point" in setting and point_step:
        return f"{model} on {METHOD_MODALITY_LABEL}, point prompts ({point_step.group(1)} px spacing)"
    if "point" in setting:
        return f"{model} on {METHOD_MODALITY_LABEL}, point prompts"
    box = re.search(r"(?:box|b)(\d+)_s(\d+)", setting)
    if box:
        return f"{model} on {METHOD_MODALITY_LABEL}, box prompts ({box.group(1)} px box, {box.group(2)} px stride)"
    if "box" in setting:
        return f"{model} on {METHOD_MODALITY_LABEL}, box prompts"
    return f"{model} on {METHOD_MODALITY_LABEL}"


def candidate_mask_path(run_dir: Path, candidate_idx: int) -> Path:
    return run_dir / "candidate_masks" / f"candidate_{candidate_idx:03d}.png"


def read_candidate_metadata_row(report_path: Path, candidate_idx: int) -> dict:
    metadata_path = report_path.parent / "candidate_metadata.csv"
    with metadata_path.open() as f:
        for row in csv.DictReader(f):
            if int(row["candidate_id"]) == candidate_idx:
                return row
    raise RuntimeError(f"candidate_id={candidate_idx} not found in {metadata_path}")


class Reconstructor:
    def __init__(self, device: str | None):
        self.device = device
        self._cache: Dict[Tuple[str, str, Tuple[int, int]], Tuple[object, dict]] = {}

    def reconstruct(self, report: dict, report_path: Path, candidate_idx: int, output_path: Path) -> np.ndarray:
        from sam3_inference import SAM3Model, resize_mask

        row = read_candidate_metadata_row(report_path, candidate_idx)
        shape_hw = tuple(int(v) for v in report["image_shape"])
        image_path = resolve_path(report["image_path"])
        checkpoint = str(resolve_path(report["checkpoint"])) if report.get("checkpoint") else None
        key = (str(image_path), checkpoint or "", shape_hw)
        if key not in self._cache:
            image = np.array(Image.open(image_path).convert("RGB"))
            image = resize_rgb(image, shape_hw)
            model = SAM3Model(confidence_threshold=0.1, checkpoint_path=checkpoint, device=self.device)
            state = model.encode_image(image)
            self._cache[key] = (model, state)
        model, state = self._cache[key]

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
            raise RuntimeError(f"Could not reconstruct candidate {candidate_idx} for {report_path}")
        if pred.shape != shape_hw:
            pred = resize_mask(pred.astype(np.uint8), shape_hw).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred.astype(np.uint8) * 255).save(output_path)
        return pred.astype(bool)


def load_or_reconstruct_candidate(
    report: dict,
    report_path: Path,
    candidate_idx: int,
    out_mask_path: Path,
    reconstructor: Reconstructor | None,
) -> np.ndarray:
    saved_path = candidate_mask_path(report_path.parent, candidate_idx)
    if saved_path.exists():
        mask = load_mask(saved_path)
        shape_hw = tuple(int(v) for v in report["image_shape"])
        mask = resize_bool(mask, shape_hw)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out_mask_path)
        return mask
    if reconstructor is None:
        raise RuntimeError(
            f"Best candidate mask is not saved: {saved_path}. "
            "Rerun with --allow-reconstruct-missing on a machine with SAM3 available."
        )
    return reconstructor.reconstruct(report, report_path, candidate_idx, out_mask_path)


def render_six_panel(
    display: str,
    he: np.ndarray,
    ficture: np.ndarray,
    candidate: np.ndarray,
    annotation: np.ndarray,
    metrics: Dict[str, float],
    method: str,
    output_path: Path,
) -> None:
    cand_color = (0, 112, 255)
    ann_color = (0, 185, 95)
    panels = [
        (overlay(he, candidate, cand_color), "Candidate on H&E ROI", "blue"),
        (solid_mask(candidate, cand_color), "Candidate only", "blue"),
        (overlay(he, annotation, ann_color), "Annotation on H&E ROI", "green"),
        (solid_mask(annotation, ann_color), "Annotation only", "green"),
        (he, "H&E ROI only", "black"),
        (ficture, "FICTURE map", "black"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(32, 8))
    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color=color, loc="left")
        ax.axis("off")

    fig.suptitle(
        f"{display}: ROI FICTURE candidate comparison\n"
        f"Dice {metrics['dice']:.3f} | Precision {metrics['precision']:.3f} | Recall {metrics['recall']:.3f}\n"
        f"{method}",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86], w_pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_table_png(rows: List[dict], output_path: Path) -> None:
    columns = ["tissue class", "best Dice", "Precision", "Recall", "25%", "median", "75%", "best-performing method"]
    widths = [0.16, 0.09, 0.09, 0.08, 0.07, 0.08, 0.07, 0.36]
    row_heights = []
    wrapped_methods = []
    for row in rows:
        wrapped = textwrap.fill(row["best-performing method"], width=42)
        wrapped_methods.append(wrapped)
        row_heights.append(0.56 + 0.34 * wrapped.count("\n"))

    total_height = 0.85 + sum(row_heights)
    fig_w = 16.5
    fig_h = max(4.8, total_height * 0.62)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    x_positions = np.cumsum([0] + widths)
    y = total_height - 0.42
    for idx, (x0, col, width) in enumerate(zip(x_positions[:-1], columns, widths)):
        if 1 <= idx <= 6:
            ax.text(x0 + width / 2, y, col, ha="center", va="center", fontsize=13, fontweight="bold", color="#202124")
        else:
            ax.text(x0 + 0.005, y, col, ha="left", va="center", fontsize=15, fontweight="bold", color="#202124")
    ax.hlines(y - 0.32, 0, 1, colors="#d9d9d9", linewidth=1.2)

    y -= 0.75
    for row, wrapped, height in zip(rows, wrapped_methods, row_heights):
        values = [
            row["tissue class"],
            f"{row['best Dice']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['25%']:.3f}",
            f"{row['median']:.3f}",
            f"{row['75%']:.3f}",
            wrapped,
        ]
        for idx, (x0, width, value) in enumerate(zip(x_positions[:-1], widths, values)):
            ha = "right" if 1 <= idx <= 6 else "left"
            x = x0 + width - 0.012 if ha == "right" else x0 + 0.005
            ax.text(x, y, value, ha=ha, va="top", fontsize=14, color="#2b2f33", linespacing=1.15)
        ax.hlines(y - height + 0.08, 0, 1, colors="#e3e3e3", linewidth=1.0)
        y -= height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_rows(reports: List[Tuple[Path, dict]]) -> List[dict]:
    label_names = {key for key, _ in LABEL_ORDER}
    by_label: Dict[str, List[dict]] = {key: [] for key in label_names}

    for report_path, report in reports:
        check_no_forbidden_paths([report_path])
        setting = report_path.parent.name
        job_root = report_path.parents[1].name
        for label_item in report.get("labels", []):
            label = label_item.get("label") or label_item.get("slug")
            if label not in by_label:
                continue
            metrics = label_item.get("best_candidate_metrics", {})
            by_label[label].append(
                {
                    "report_path": report_path,
                    "report": report,
                    "label_item": label_item,
                    "dice": float(metrics["dice"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "iou": float(metrics.get("iou", 0.0)),
                    "psnr": float(metrics.get("psnr", 0.0)),
                    "ssim": float(metrics.get("ssim", 0.0)),
                    "setting": setting,
                    "job_root": job_root,
                    "best_idx": int(label_item["best_candidate_index"]),
                    "n_candidates": int(report.get("n_candidates_after_nms", 0)),
                }
            )

    rows: List[dict] = []
    for label_key, display in LABEL_ORDER:
        items = by_label.get(label_key) or []
        if not items:
            raise SystemExit(f"No official candidate metrics found for label: {label_key}")
        best = max(items, key=lambda item: item["dice"])
        dice_values = [item["dice"] for item in items]
        rows.append(
            {
                "label": label_key,
                "tissue class": display,
                "best Dice": best["dice"],
                "Precision": best["precision"],
                "Recall": best["recall"],
                "IoU": best["iou"],
                "PSNR": best["psnr"],
                "SSIM": best["ssim"],
                "25%": quantile(dice_values, 0.25),
                "median": quantile(dice_values, 0.50),
                "75%": quantile(dice_values, 0.75),
                "best-performing method": method_label(best["setting"]),
                "setting": best["setting"],
                "job_root": best["job_root"],
                "best_candidate_index": best["best_idx"],
                "n_candidates": best["n_candidates"],
                "report_path": best["report_path"],
                "report": best["report"],
                "label_item": best["label_item"],
                "all_dice_count": len(dice_values),
            }
        )
    return rows


def write_table_files(rows: List[dict], output_dir: Path, preflight: dict, report_count: int, unique_settings: int) -> None:
    csv_fields = [
        "tissue class",
        "best Dice",
        "Precision",
        "Recall",
        "25%",
        "median",
        "75%",
        "best-performing method",
        "setting",
        "job_root",
        "best_candidate_index",
        "n_candidates",
        "all_dice_count",
        "report_path",
        "figure_path",
        "candidate_mask_path",
    ]
    csv_path = output_dir / f"{ARTIFACT_PREFIX}_best_by_dice.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in csv_fields})

    md_lines = [
        f"# {REPORT_TITLE}",
        "",
        f"PASS_OFFICIAL: {preflight['summary']['status']}; ROI: {preflight['roi_size_wh'][0]} x {preflight['roi_size_wh'][1]}; "
        f"factor index: {tuple(preflight['factor_shape'])}; reports: {report_count}; unique settings: {unique_settings}.",
        "",
        "| tissue class | best Dice | Precision | Recall | 25% | median | 75% | best-performing method |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['tissue class']} | {row['best Dice']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | "
            f"{row['25%']:.3f} | {row['median']:.3f} | {row['75%']:.3f} | {row['best-performing method']} |"
        )
    (output_dir / f"{ARTIFACT_PREFIX}_best_by_dice.md").write_text("\n".join(md_lines) + "\n")
    render_table_png(rows, output_dir / f"{ARTIFACT_PREFIX}_best_by_dice_table.png")


def render_figures(rows: List[dict], preflight: dict, output_dir: Path, reconstructor: Reconstructor | None) -> None:
    he_orig = np.array(Image.open(preflight["he_path"]).convert("RGB"))
    ficture_orig = np.array(Image.open(preflight["ficture_path"]).convert("RGB"))
    region_summary = load_json(preflight["region_summary_path"])
    masks_by_label = {item["label"]: resolve_path(item["mask_path"]) for item in region_summary["labels"]}

    figure_dir = output_dir / "six_panel"
    mask_dir = output_dir / "best_candidate_masks"
    for idx, row in enumerate(rows, start=1):
        label = row["label"]
        report = row["report"]
        report_path = row["report_path"]
        shape_hw = tuple(int(v) for v in report["image_shape"])
        he = resize_rgb(he_orig, shape_hw)
        ficture = resize_rgb(ficture_orig, shape_hw)
        annotation = resize_bool(load_mask(masks_by_label[label]), shape_hw)
        mask_path = mask_dir / f"{idx:02d}_{label}_best_candidate_mask.png"
        candidate = load_or_reconstruct_candidate(
            report,
            report_path,
            int(row["best_candidate_index"]),
            mask_path,
            reconstructor,
        )
        figure_path = figure_dir / f"{idx:02d}_{label}_{ARTIFACT_PREFIX}_best_by_dice_6panel.png"
        render_six_panel(
            row["tissue class"],
            he,
            ficture,
            candidate,
            annotation,
            {"dice": row["best Dice"], "precision": row["Precision"], "recall": row["Recall"]},
            row["best-performing method"],
            figure_path,
        )
        row["figure_path"] = output_ref(figure_path)
        row["candidate_mask_path"] = output_ref(mask_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, action="append", default=None)
    parser.add_argument("--official-summary", type=Path, default=OFFICIAL_SUMMARY)
    parser.add_argument("--official-input-dir", type=Path, default=OFFICIAL_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-unique-settings", type=int, default=48)
    parser.add_argument("--allow-reconstruct-missing", action="store_true")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--method-modality-label", default="FICTURE ROI")
    parser.add_argument("--report-title", default="Official FICTURE Candidate Best By Dice")
    parser.add_argument("--artifact-prefix", default="official_ficture_candidate")
    args = parser.parse_args()

    global METHOD_MODALITY_LABEL, REPORT_TITLE, ARTIFACT_PREFIX
    METHOD_MODALITY_LABEL = args.method_modality_label
    REPORT_TITLE = args.report_title
    ARTIFACT_PREFIX = args.artifact_prefix

    args.official_summary = project_path(args.official_summary)
    args.official_input_dir = project_path(args.official_input_dir)
    args.output_dir = project_path(args.output_dir)
    roots = [project_path(root) for root in (args.candidate_root or [OFFICIAL_CANDIDATE_ROOT])]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preflight = official_preflight(args.official_summary, args.official_input_dir, roots[0])
    reports = read_reports(roots)
    if not reports:
        raise SystemExit("No candidate_report.json files found.")
    unique_settings = len({path.parent.name for path, _ in reports})
    if unique_settings < args.min_unique_settings:
        raise SystemExit(f"Only found {unique_settings} unique official settings; expected at least {args.min_unique_settings}.")

    rows = build_rows(reports)
    reconstructor = Reconstructor(args.device) if args.allow_reconstruct_missing else None
    render_figures(rows, preflight, args.output_dir, reconstructor)
    write_table_files(rows, args.output_dir, preflight, len(reports), unique_settings)

    manifest = {
        "status": "PASS_OFFICIAL_CANDIDATE_BEST_RENDERED",
        "selection_rule": "best single candidate per tissue class by maximum Dice across official candidate reports",
        "official_summary": str(args.official_summary),
        "official_input_dir": str(args.official_input_dir),
        "candidate_roots": [str(root) for root in roots],
        "report_count": len(reports),
        "unique_settings": unique_settings,
        "outputs": {
            "table_csv": str(args.output_dir / f"{ARTIFACT_PREFIX}_best_by_dice.csv"),
            "table_md": str(args.output_dir / f"{ARTIFACT_PREFIX}_best_by_dice.md"),
            "table_png": str(args.output_dir / f"{ARTIFACT_PREFIX}_best_by_dice_table.png"),
            "six_panel_dir": str(args.output_dir / "six_panel"),
        },
        "rows": [
            {
                "label": row["label"],
                "display": row["tissue class"],
                "dice": row["best Dice"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "setting": row["setting"],
                "candidate": row["best_candidate_index"],
                "figure_path": row.get("figure_path"),
            }
            for row in rows
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"WROTE {args.output_dir}")
    print(f"reports={len(reports)} unique_settings={unique_settings}")
    for row in rows:
        print(
            f"{row['tissue class']}: Dice={row['best Dice']:.3f} "
            f"P={row['Precision']:.3f} R={row['Recall']:.3f} "
            f"{row['setting']} cand={row['best_candidate_index']}"
        )


if __name__ == "__main__":
    main()
