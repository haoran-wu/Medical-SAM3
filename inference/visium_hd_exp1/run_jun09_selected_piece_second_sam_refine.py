#!/usr/bin/env python3
"""Second-SAM refinement for Jun09 selected component pieces.

The selected piece masks are used only as spatial prompts.  For every selected
piece, this script turns the piece into a box prompt, asks SAM3/Medical-SAM3 to
segment inside that local box, keeps the SAM component near the original piece,
then unions the refined pieces per tissue class.

This tests a SaLIP-style hypothesis: a small piece can localize the right
structure even when the piece mask itself is too fragmented for the final mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


ROOT = PROJECT_ROOT
DEFAULT_BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
DEFAULT_SELECTED = DEFAULT_BASE / "refined_diagnostics/refined_policy_selected_pieces.csv"
DEFAULT_HIDDEN = DEFAULT_BASE / "corrected_pool/hidden_candidate_truth.csv"
DEFAULT_HE = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
DEFAULT_FICTURE = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"
DEFAULT_ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
DEFAULT_OUT = DEFAULT_BASE / "second_sam_refinement"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
CLASS_TO_ANNOTATION = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 127


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def bbox_from_mask(mask: np.ndarray, margin: float) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    x0 -= w * margin
    x1 += w * margin
    y0 -= h * margin
    y1 += h * margin
    height, width = mask.shape
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(width - 1, int(round(x1))),
        min(height - 1, int(round(y1))),
    )


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    pa = int(pred.sum())
    ga = int(gt.sum())
    precision = tp / pa if pa else 0.0
    recall = tp / ga if ga else 0.0
    dice = 2 * tp / (pa + ga) if pa + ga else 0.0
    return {"dice": dice, "precision": precision, "recall": recall}


def keep_component_near_prior(pred_mask: np.ndarray, prior_mask: np.ndarray, prior_dilate_radius: int) -> np.ndarray:
    pred = pred_mask.astype(bool)
    if not pred.any() or not prior_mask.any():
        return pred
    context = binary_dilation(prior_mask.astype(bool), disk(max(1, prior_dilate_radius)))
    labeled = label(pred)
    best_label = 0
    best_overlap = 0
    for region in regionprops(labeled):
        component = labeled == region.label
        overlap = int(np.logical_and(component, context).sum())
        if overlap > best_overlap:
            best_label = int(region.label)
            best_overlap = overlap
    if best_label == 0:
        return np.logical_and(pred, context)
    return np.logical_and(labeled == best_label, context)


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 130) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, mode="RGB")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    img = image.convert("RGB")
    img.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def render_six_panel(
    out_path: Path,
    tissue_class: str,
    he: Image.Image,
    ficture: Image.Image,
    selected_mask: np.ndarray,
    annotation: np.ndarray,
    title: str,
    subtitle: str,
    note: str,
    selected_label: str = "Second-SAM union",
) -> None:
    panel_w, panel_h = 260, 275
    gap = 18
    margin = 26
    header_h = 112
    title_h = 24
    items = [
        ("Annotation on H&E", overlay_mask(he, annotation, (0, 180, 90))),
        (f"{selected_label} on H&E", overlay_mask(he, selected_mask, (0, 90, 255))),
        (f"{selected_label} mask only", mask_only(selected_mask, (0, 90, 255))),
        ("Annotation mask only", mask_only(annotation, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    width = margin * 2 + panel_w * 6 + gap * 5
    height = margin * 2 + header_h + title_h + panel_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), subtitle, fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), note[:260], fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 72), f"{tissue_class}: second-SAM refinement check", fill=(70, 80, 90), font=font)
    y0 = margin + header_h
    for idx, (label_text, image) in enumerate(items):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label_text, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(image, panel_w, panel_h), (x, y0 + title_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-pieces-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--hidden-truth-csv", type=Path, default=DEFAULT_HIDDEN)
    parser.add_argument("--he-roi", type=Path, default=DEFAULT_HE)
    parser.add_argument("--ficture-roi", type=Path, default=DEFAULT_FICTURE)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--labels", default=",".join(CLASS_KEYS))
    parser.add_argument("--box-margin", type=float, default=0.08)
    parser.add_argument("--prior-dilate-radius", type=int, default=96)
    parser.add_argument("--max-pieces-per-class", type=int, default=0)
    parser.add_argument(
        "--dry-run-prompts-only",
        action="store_true",
        help="Validate selected pieces, prompt boxes, and preview unions without loading SAM3.",
    )
    args = parser.parse_args()
    if not args.dry_run_prompts_only and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --dry-run-prompts-only is set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    figures_dir = args.output_dir / "figures"
    panels_dir = args.output_dir / "piece_panels"
    masks_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    panels_dir.mkdir(exist_ok=True)

    he = Image.open(args.he_roi).convert("RGB")
    ficture = Image.open(args.ficture_roi).convert("RGB")
    if he.size != ficture.size:
        raise SystemExit(f"H&E/FICTURE ROI size mismatch: {he.size} vs {ficture.size}")
    shape_hw = (he.height, he.width)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    annotations = {
        c: load_mask(args.annotation_dir / CLASS_TO_ANNOTATION[c], he.size)
        for c in labels
    }

    selected = [row for row in read_csv(args.selected_pieces_csv) if row["class"] in labels]
    selected_base = args.selected_pieces_csv.parent
    if args.max_pieces_per_class:
        limited: list[dict[str, str]] = []
        per_class_counts = {c: 0 for c in labels}
        for row in selected:
            c = row["class"]
            if per_class_counts[c] >= args.max_pieces_per_class:
                continue
            limited.append(row)
            per_class_counts[c] += 1
        selected = limited
    hidden_by_uid = {row["candidate_uid"]: row for row in read_csv(args.hidden_truth_csv)}
    if not selected:
        raise SystemExit("No selected pieces found for requested labels.")

    sam3 = None
    state = None
    normalize_prediction = None
    if not args.dry_run_prompts_only:
        from sam3_baseline_utils import normalize_prediction as _normalize_prediction
        from sam3_inference import SAM3Model

        normalize_prediction = _normalize_prediction
        sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
        state = sam3.encode_image(np.array(he))

    per_piece_rows: list[dict[str, object]] = []
    prompt_preview_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for tissue_class in labels:
        class_rows = [row for row in selected if row["class"] == tissue_class]
        if not class_rows:
            continue
        piece_union = np.zeros(shape_hw, dtype=bool)
        sam_union = np.zeros(shape_hw, dtype=bool)
        for row in class_rows:
            uid = row["candidate_uid"]
            hidden = hidden_by_uid.get(uid)
            candidate_mask_path = row.get("mask_path") or ""
            candidate_mask = Path(candidate_mask_path) if candidate_mask_path else None
            if candidate_mask is not None and not candidate_mask.is_absolute():
                candidate_mask = selected_base / candidate_mask
            if candidate_mask is None or not candidate_mask.exists():
                if hidden is None:
                    raise SystemExit(f"Missing hidden truth and selected mask_path for selected candidate: {uid}")
                candidate_mask = Path(hidden["mask_path"])
            if not candidate_mask.exists():
                raise SystemExit(
                    f"Selected candidate mask does not exist for {uid}: {candidate_mask}. "
                    "For remote runs, use a selected-pieces CSV with mask_path relative to that CSV."
                )
            prior = load_mask(candidate_mask, he.size)
            box = bbox_from_mask(prior, args.box_margin)
            if box is None:
                continue
            piece_union |= prior
            prior_metrics = metrics(prior, annotations[tissue_class])
            prompt_preview_rows.append(
                {
                    "class": tissue_class,
                    "candidate_uid": uid,
                    "rank": row.get("rank", ""),
                    "target_score": row.get("target_score", ""),
                    "margin": row.get("margin", ""),
                    "bbox_xyxy": json.dumps(list(box)),
                    "prior_pixels": int(prior.sum()),
                    "prior_dice": f"{prior_metrics['dice']:.6f}",
                    "prior_precision": f"{prior_metrics['precision']:.6f}",
                    "prior_recall": f"{prior_metrics['recall']:.6f}",
                    "mask_path": str(candidate_mask),
                }
            )
            if args.dry_run_prompts_only:
                continue
            if sam3 is None or state is None or normalize_prediction is None:
                raise RuntimeError("SAM3 model was not initialized.")
            pred = normalize_prediction(sam3.predict_box(state, box, shape_hw), shape_hw).astype(bool)
            refined = keep_component_near_prior(pred, prior, args.prior_dilate_radius)
            sam_union |= refined
            save_mask(refined.astype(np.uint8), masks_dir / f"{tissue_class}_{uid}_second_sam.png")
            per_piece_metrics = metrics(refined, annotations[tissue_class])
            per_piece_rows.append(
                {
                    "class": tissue_class,
                    "candidate_uid": uid,
                    "rank": row.get("rank", ""),
                    "target_score": row.get("target_score", ""),
                    "margin": row.get("margin", ""),
                    "bbox_xyxy": json.dumps(list(box)),
                    "prior_pixels": int(prior.sum()),
                    "second_sam_pixels": int(refined.sum()),
                    "prior_dice": f"{prior_metrics['dice']:.6f}",
                    "prior_precision": f"{prior_metrics['precision']:.6f}",
                    "prior_recall": f"{prior_metrics['recall']:.6f}",
                    "second_sam_dice": f"{per_piece_metrics['dice']:.6f}",
                    "second_sam_precision": f"{per_piece_metrics['precision']:.6f}",
                    "second_sam_recall": f"{per_piece_metrics['recall']:.6f}",
                    "mask_path": str(masks_dir / f"{tissue_class}_{uid}_second_sam.png"),
                }
            )
        save_mask(piece_union.astype(np.uint8), masks_dir / f"{tissue_class}_selected_piece_union.png")
        preview_metrics = metrics(piece_union, annotations[tissue_class])
        preview_path = figures_dir / f"{tissue_class}_selected_piece_prompt_preview.png"
        render_six_panel(
            preview_path,
            tissue_class,
            he,
            ficture,
            piece_union,
            annotations[tissue_class],
            f"{tissue_class}: selected-piece prompt preview",
            (
                f"Selected piece union Dice {preview_metrics['dice']:.3f} | "
                f"Precision {preview_metrics['precision']:.3f} | Recall {preview_metrics['recall']:.3f}"
            ),
            (
                f"{len(class_rows)} selected pieces will become box prompts; "
                f"this preview is before any second-SAM refinement."
            ),
            selected_label="Selected piece union",
        )
        prompt_preview_rows.append(
            {
                "class": tissue_class,
                "candidate_uid": "__CLASS_UNION__",
                "rank": "",
                "target_score": "",
                "margin": "",
                "bbox_xyxy": "",
                "prior_pixels": int(piece_union.sum()),
                "prior_dice": f"{preview_metrics['dice']:.6f}",
                "prior_precision": f"{preview_metrics['precision']:.6f}",
                "prior_recall": f"{preview_metrics['recall']:.6f}",
                "mask_path": str(masks_dir / f"{tissue_class}_selected_piece_union.png"),
                "figure_rel": str(preview_path.relative_to(args.output_dir)),
            }
        )
        if args.dry_run_prompts_only:
            summary_rows.append(
                {
                    "class": tissue_class,
                    "selected_piece_count": len(class_rows),
                    "prompt_preview_dice": f"{preview_metrics['dice']:.6f}",
                    "prompt_preview_precision": f"{preview_metrics['precision']:.6f}",
                    "prompt_preview_recall": f"{preview_metrics['recall']:.6f}",
                    "figure_rel": str(preview_path.relative_to(args.output_dir)),
                }
            )
            continue
        save_mask(sam_union.astype(np.uint8), masks_dir / f"{tissue_class}_second_sam_union.png")
        prior_union_metrics = metrics(piece_union, annotations[tissue_class])
        sam_union_metrics = metrics(sam_union, annotations[tissue_class])
        fig_path = figures_dir / f"{tissue_class}_second_sam_union.png"
        render_six_panel(
            fig_path,
            tissue_class,
            he,
            ficture,
            sam_union,
            annotations[tissue_class],
            f"{tissue_class}: selected-piece second-SAM refinement",
            (
                f"Second-SAM Dice {sam_union_metrics['dice']:.3f} | Precision {sam_union_metrics['precision']:.3f} | "
                f"Recall {sam_union_metrics['recall']:.3f}"
            ),
            (
                f"Prior piece union D/P/R {prior_union_metrics['dice']:.3f}/"
                f"{prior_union_metrics['precision']:.3f}/{prior_union_metrics['recall']:.3f}; "
                f"{len(class_rows)} selected pieces converted to box prompts."
            ),
        )
        summary_rows.append(
            {
                "class": tissue_class,
                "selected_piece_count": len(class_rows),
                "prior_union_dice": f"{prior_union_metrics['dice']:.6f}",
                "prior_union_precision": f"{prior_union_metrics['precision']:.6f}",
                "prior_union_recall": f"{prior_union_metrics['recall']:.6f}",
                "second_sam_dice": f"{sam_union_metrics['dice']:.6f}",
                "second_sam_precision": f"{sam_union_metrics['precision']:.6f}",
                "second_sam_recall": f"{sam_union_metrics['recall']:.6f}",
                "delta_dice": f"{sam_union_metrics['dice'] - prior_union_metrics['dice']:.6f}",
                "delta_precision": f"{sam_union_metrics['precision'] - prior_union_metrics['precision']:.6f}",
                "delta_recall": f"{sam_union_metrics['recall'] - prior_union_metrics['recall']:.6f}",
                "figure_rel": str(fig_path.relative_to(args.output_dir)),
            }
        )

    if args.dry_run_prompts_only:
        write_csv(args.output_dir / "second_sam_prompt_preview_summary.csv", summary_rows)
    else:
        write_csv(args.output_dir / "second_sam_piece_results.csv", per_piece_rows)
        write_csv(args.output_dir / "second_sam_summary.csv", summary_rows)
    write_csv(args.output_dir / "second_sam_prompt_preview_rows.csv", prompt_preview_rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "selected_pieces_csv": str(args.selected_pieces_csv),
                "hidden_truth_csv": str(args.hidden_truth_csv),
                "he_roi": str(args.he_roi),
                "ficture_roi": str(args.ficture_roi),
                "annotation_dir": str(args.annotation_dir),
                "checkpoint": args.checkpoint,
                "labels": labels,
                "box_margin": args.box_margin,
                "prior_dilate_radius": args.prior_dilate_radius,
                "max_pieces_per_class": args.max_pieces_per_class,
                "dry_run_prompts_only": args.dry_run_prompts_only,
            },
            indent=2,
        )
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
