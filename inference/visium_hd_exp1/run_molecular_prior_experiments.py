#!/usr/bin/env python3
"""Run molecular-prior experiments for VisiumHD Exp1.

This script covers the first local pass of the four proposed directions:

1. FICTURE-prior masks/components in H&E coordinates.
2. Molecular reranking of existing SAM3 proposal masks.
3. A lightweight gene-marker ranking proxy for image-expression retrieval.
4. A SaLIP-style hybrid score using text-prompt candidate order plus molecular
   evidence.

It intentionally avoids loading SAM3. The heavy refinement stage can consume
the exported component masks/boxes after this script verifies that the molecular
priors are sensible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects

from render_ficture_factor_overlay_on_he import (
    decode_factor_grid,
    estimate_tissue_mask,
    load_factor_info,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FICTURE_GRID = PROJECT_ROOT / "pixel-level cell type image" / "visiumhd_exp1_hex12_k12" / "hex_12.k12.pixel.png"
DEFAULT_FACTOR_INFO = PROJECT_ROOT / "pixel-level cell type image" / "visiumhd_exp1_hex12_k12" / "hex_12.k12.pixel.info.tsv"
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80"
DEFAULT_SAM3_PROPOSAL_ROOT = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_candidate_proposals" / "labelwise_text_box_11232554"
DEFAULT_TARGET_MASK_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks"


LABEL_MARKERS: Dict[str, set[str]] = {
    "tumor": {
        "CEACAM5",
        "CEACAM6",
        "SPINK1",
        "NAPSA",
        "WFDC2",
        "SLC34A2",
        "MUC1",
        "KRT8",
        "IFI6",
        "ECM1",
        "SPP1",
    },
    "stroma": {
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "ACTA2",
        "TAGLN",
        "MYL9",
        "MYH11",
        "MGP",
        "DCN",
        "SPARC",
        "VIM",
    },
    "immune_infiltration": {
        "CD74",
        "LYZ",
        "CXCR4",
        "IL7R",
        "TRAC",
        "MS4A1",
        "IGKC",
        "IGHG1",
        "IGHM",
        "JCHAIN",
        "CD68",
        "C1QA",
        "C1QB",
        "C1QC",
        "APOE",
        "SPP1",
    },
    "lung_bronchiola": {
        "SCGB1A1",
        "SCGB3A1",
        "BPIFB1",
        "MUC5B",
        "MUC5AC",
        "CAPS",
        "TPPP3",
        "FOXJ1",
        "WFDC2",
    },
    "lung_vessels": {
        "TAGLN",
        "MYL9",
        "MYH11",
        "ACTA2",
        "MGP",
        "A2M",
        "DES",
        "COL3A1",
    },
    "lung_alveoli_normal_adjacent": {
        "SFTPA1",
        "SFTPA2",
        "SFTPB",
        "SFTPC",
        "LPCAT1",
        "NAPSA",
        "AGER",
        "AQP5",
        "SLC34A2",
    },
    "erythorocytes": {"HBA1", "HBA2", "HBB", "ALAS2", "HBD"},
    "pigment": {"SPP1", "APOE", "CHIT1", "GPNMB", "CD68", "FTL", "FTH1", "CTSB"},
}


def slugify(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unlabeled"


def parse_gene_list(value: object, limit: Optional[int] = None) -> List[str]:
    genes = [item.strip() for item in str(value).split(",") if item.strip()]
    return genes[:limit] if limit is not None else genes


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    precision = inter / pred_sum if pred_sum else 0.0
    recall = inter / gt_sum if gt_sum else 0.0
    dice = 2 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0
    iou = inter / union if union else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "intersection": float(inter),
        "pred_pixels": float(pred_sum),
        "gt_pixels": float(gt_sum),
    }


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)) > 127


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask.astype(bool)] = (1.0 - alpha) * out[mask.astype(bool)] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_label_image(
    he_shape: Tuple[int, int],
    grid: np.ndarray,
    info: pd.DataFrame,
    *,
    max_color_distance: float,
    microns_per_pixel: float,
    tissue_hires_scalef: float,
    ficture_res_um_per_pixel: float,
    ficture_xmin_um: float,
    ficture_ymin_um: float,
    shift_x_px: float,
    shift_y_px: float,
    tissue_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    palette = np.array(info["rgb_tuple"].tolist(), dtype=np.int32)
    assignment = decode_factor_grid(grid, palette, max_color_distance)
    src_y, src_x = np.nonzero(assignment >= 0)
    src_factor_idx = assignment[src_y, src_x]

    scale = tissue_hires_scalef / microns_per_pixel
    he_x = (ficture_xmin_um + src_y.astype(np.float64) * ficture_res_um_per_pixel) * scale
    he_y = (ficture_ymin_um + src_x.astype(np.float64) * ficture_res_um_per_pixel) * scale
    xi = np.rint(he_x + shift_x_px).astype(np.int32)
    yi = np.rint(he_y + shift_y_px).astype(np.int32)

    height, width = he_shape
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    inside = np.zeros_like(valid)
    inside[valid] = tissue_mask[yi[valid], xi[valid]]
    keep = valid & inside

    label_image = np.full((height, width), -1, dtype=np.int16)
    label_image[yi[keep], xi[keep]] = src_factor_idx[keep]
    return label_image, {
        "source_assigned_pixels": float((assignment >= 0).sum()),
        "mapped_valid_pixels": float(valid.sum()),
        "mapped_inside_tissue_pixels": float(keep.sum()),
        "valid_fraction": float(valid.mean()),
        "inside_tissue_fraction_of_valid": float(inside[valid].mean()) if valid.any() else 0.0,
    }


def infer_factor_annotation(row: pd.Series) -> Dict[str, object]:
    genes = set(parse_gene_list(row["TopGene_specific"]))
    label_scores = {}
    for label_name, markers in LABEL_MARKERS.items():
        label_scores[label_name] = len(genes & markers) / max(1, len(markers))
    best_label = max(label_scores, key=label_scores.get)
    sorted_scores = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "factor": int(row["Factor"]),
        "rgb": row["RGB"],
        "weight": float(row["Weight"]),
        "post_umi": int(row["PostUMI"]),
        "provisional_label": best_label,
        "provisional_score": float(label_scores[best_label]),
        "label_scores_json": json.dumps(dict(sorted_scores), sort_keys=True),
        "top_genes_specific": ", ".join(parse_gene_list(row["TopGene_specific"], 20)),
        "top_genes_pval": ", ".join(parse_gene_list(row["TopGene_pval"], 20)),
    }


def save_factor_masks_and_components(
    he: np.ndarray,
    label_image: np.ndarray,
    info: pd.DataFrame,
    outdir: Path,
    *,
    min_component_area: int,
    close_radius: int,
    top_components_per_factor: int,
) -> Tuple[List[Dict[str, object]], Dict[int, np.ndarray], pd.DataFrame]:
    masks_dir = outdir / "factor_masks_he"
    overlays_dir = outdir / "factor_overlays_he"
    components_dir = outdir / "factor_components"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    components_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.DataFrame([infer_factor_annotation(row) for _, row in info.iterrows()])
    annotations.to_csv(outdir / "factor_annotation.csv", index=False)

    factor_masks: Dict[int, np.ndarray] = {}
    component_rows: List[Dict[str, object]] = []
    palette = np.array(info["rgb_tuple"].tolist(), dtype=np.int32)
    factors = info["Factor"].astype(int).to_numpy()
    structure = np.ones((3, 3), dtype=bool)

    for palette_idx, row in info.iterrows():
        factor = int(row["Factor"])
        rgb = tuple(int(v) for v in row["rgb_tuple"])
        mask = label_image == palette_idx
        if close_radius > 0:
            mask = binary_closing(mask, disk(close_radius))
        mask = remove_small_objects(mask, min_size=max(1, min_component_area // 4))
        factor_masks[factor] = mask.astype(bool)

        Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / f"factor_{factor:02d}.png")
        Image.fromarray(make_overlay(he, mask, rgb, alpha=0.45)).save(overlays_dir / f"factor_{factor:02d}_overlay.png")

        labeled, n_labels = ndi.label(mask, structure=structure)
        records = []
        for component_id in range(1, n_labels + 1):
            component = labeled == component_id
            area = int(component.sum())
            if area < min_component_area:
                continue
            ys, xs = np.nonzero(component)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cx = float(xs.mean())
            cy = float(ys.mean())
            records.append((area, component_id, component, x0, y0, x1, y1, cx, cy))
        records.sort(reverse=True, key=lambda item: item[0])

        ann = annotations[annotations["factor"] == factor].iloc[0].to_dict()
        for rank, (area, component_id, component, x0, y0, x1, y1, cx, cy) in enumerate(
            records[:top_components_per_factor], start=1
        ):
            mask_name = f"factor_{factor:02d}_rank_{rank:02d}.png"
            Image.fromarray(component.astype(np.uint8) * 255).save(components_dir / mask_name)
            component_rows.append(
                {
                    "component_id": f"factor_{factor:02d}_rank_{rank:02d}",
                    "factor": factor,
                    "rank_within_factor": rank,
                    "area": area,
                    "bbox_x0": x0,
                    "bbox_y0": y0,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "rgb": row["RGB"],
                    "weight": float(row["Weight"]),
                    "provisional_label": ann["provisional_label"],
                    "provisional_score": ann["provisional_score"],
                    "top_genes_specific": ann["top_genes_specific"],
                    "mask_path": str((components_dir / mask_name).relative_to(outdir)),
                }
            )

    components_df = pd.DataFrame(component_rows)
    components_df.to_csv(outdir / "candidate_components_he.csv", index=False)

    overlay = he.copy().astype(np.float32)
    for _, row in info.iterrows():
        factor = int(row["Factor"])
        rgb = np.array(row["rgb_tuple"], dtype=np.float32)
        mask = factor_masks.get(factor)
        if mask is not None and mask.any():
            overlay[mask] = 0.56 * overlay[mask] + 0.44 * rgb
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(outdir / "all_factor_masks_overlay.png")
    return component_rows, factor_masks, annotations


def load_target_masks(mask_dir: Path) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    if not mask_dir.exists():
        return targets
    for path in mask_dir.glob("*_target.png"):
        name = path.name
        # 08_tumor_target.png -> tumor
        label_name = name.split("_", 1)[1].removesuffix("_target.png")
        targets[label_name] = np.array(Image.open(path).convert("L")) > 127
    return targets


def evaluate_method_a(
    outdir: Path,
    factor_masks: Dict[int, np.ndarray],
    components: Sequence[Dict[str, object]],
    targets: Dict[str, np.ndarray],
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    component_masks: Dict[str, np.ndarray] = {}
    for comp in components:
        mask_path = outdir / str(comp["mask_path"])
        component_masks[str(comp["component_id"])] = np.array(Image.open(mask_path).convert("L")) > 127

    for label_name, target in targets.items():
        for factor, mask in factor_masks.items():
            metrics = mask_metrics(mask, target)
            ann = annotations[annotations["factor"] == factor].iloc[0]
            rows.append(
                {
                    "method": "A_factor_prior_mask",
                    "label": label_name,
                    "candidate_id": f"factor_{factor:02d}",
                    "factor": factor,
                    "provisional_label": ann["provisional_label"],
                    "ranking_score": float(metrics["dice"]),
                    **{f"metric_{k}": v for k, v in metrics.items()},
                }
            )
        for comp in components:
            mask = component_masks[str(comp["component_id"])]
            metrics = mask_metrics(mask, target)
            rows.append(
                {
                    "method": "A_component_prior_mask",
                    "label": label_name,
                    "candidate_id": comp["component_id"],
                    "factor": int(comp["factor"]),
                    "provisional_label": comp["provisional_label"],
                    "ranking_score": float(metrics["dice"]),
                    **{f"metric_{k}": v for k, v in metrics.items()},
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["label", "method", "ranking_score"], ascending=[True, True, False]).to_csv(
            outdir / "method_a_prior_vs_targets.csv", index=False
        )
    return df


def load_sam3_candidate_masks(label_dir: Path, image_shape: Tuple[int, int]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    metadata_path = label_dir / "candidate_metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    for mask_path in sorted((label_dir / "candidate_masks").glob("candidate_*.png")):
        candidate_id = int(mask_path.stem.split("_")[-1])
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        row = metadata[metadata["candidate_id"] == candidate_id].iloc[0].to_dict() if not metadata.empty and (metadata["candidate_id"] == candidate_id).any() else {}
        candidates.append({"candidate_id": candidate_id, "mask": mask, "metadata": row})
    return candidates


def marker_score_for_label(factor: int, label_name: str, annotations: pd.DataFrame) -> float:
    row = annotations[annotations["factor"] == factor]
    if row.empty:
        return 0.0
    scores = json.loads(str(row.iloc[0]["label_scores_json"]))
    return float(scores.get(label_name, 0.0))


def evaluate_methods_bcd(
    outdir: Path,
    proposal_root: Path,
    factor_masks: Dict[int, np.ndarray],
    targets: Dict[str, np.ndarray],
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not proposal_root.exists():
        return pd.DataFrame()

    for label_dir in sorted(path for path in proposal_root.iterdir() if path.is_dir()):
        label_name = label_dir.name
        candidates = load_sam3_candidate_masks(label_dir, next(iter(factor_masks.values())).shape)
        if not candidates:
            continue
        candidate_shape = candidates[0]["mask"].shape
        resized_factor_masks = {
            factor: resize_mask(mask, candidate_shape) if mask.shape != candidate_shape else mask
            for factor, mask in factor_masks.items()
        }
        target = targets.get(label_name)
        if target is not None and target.shape != candidate_shape:
            target_eval = resize_mask(target, candidate_shape)
        else:
            target_eval = target

        for cand in candidates:
            mask = cand["mask"]
            area = int(mask.sum())
            if area == 0:
                continue
            overlaps = {}
            marker_weighted_overlap = 0.0
            best_factor = None
            best_overlap = -1.0
            for factor, fmask in resized_factor_masks.items():
                frac = float(np.logical_and(mask, fmask).sum() / area)
                overlaps[factor] = frac
                marker_weighted_overlap += frac * marker_score_for_label(factor, label_name, annotations)
                if frac > best_overlap:
                    best_overlap = frac
                    best_factor = factor
            # Existing labelwise_text_box candidates are already in text-prompt order. Use inverse
            # candidate id as a weak SaLIP-style text score proxy.
            text_proxy = 1.0 / (1.0 + float(cand["candidate_id"]))
            molecular_score = marker_weighted_overlap
            hybrid_score = 0.35 * text_proxy + 0.65 * molecular_score
            metrics = mask_metrics(mask, target_eval) if target_eval is not None else {}
            for method, score in (
                ("B_molecular_rerank_sam3_proposals", molecular_score),
                ("C_marker_retrieval_proxy", marker_weighted_overlap),
                ("D_hybrid_text_proxy_plus_molecular", hybrid_score),
            ):
                rows.append(
                    {
                        "method": method,
                        "label": label_name,
                        "candidate_id": cand["candidate_id"],
                        "best_overlap_factor": best_factor,
                        "best_factor_overlap": best_overlap,
                        "ranking_score": score,
                        "text_proxy_score": text_proxy,
                        "molecular_score": molecular_score,
                        **{f"metric_{k}": v for k, v in metrics.items()},
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["label", "method", "ranking_score"], ascending=[True, True, False]).to_csv(
            outdir / "methods_bcd_sam3_candidate_reranking.csv", index=False
        )
    return df


def write_summary_tables(outdir: Path, method_a: pd.DataFrame, method_bcd: pd.DataFrame) -> None:
    lines = ["# Molecular Prior Experiment Results", ""]
    if not method_a.empty:
        lines += ["## Method A: FICTURE Prior Masks/Components", ""]
        for (label_name, method), group in method_a.groupby(["label", "method"]):
            top = group.sort_values("ranking_score", ascending=False).head(5)
            lines.append(f"### {label_name} / {method}")
            lines.append("")
            lines.append("| candidate | factor | provisional label | Dice | IoU | Recall | Precision |")
            lines.append("|---|---:|---|---:|---:|---:|---:|")
            for _, row in top.iterrows():
                lines.append(
                    f"| {row['candidate_id']} | {int(row['factor'])} | {row['provisional_label']} | "
                    f"{row['metric_dice']:.3f} | {row['metric_iou']:.3f} | "
                    f"{row['metric_recall']:.3f} | {row['metric_precision']:.3f} |"
                )
            lines.append("")
    if not method_bcd.empty:
        lines += ["## Methods B/C/D: Reranking Existing SAM3 Candidates", ""]
        for (label_name, method), group in method_bcd.groupby(["label", "method"]):
            top = group.sort_values("ranking_score", ascending=False).head(5)
            lines.append(f"### {label_name} / {method}")
            lines.append("")
            lines.append("| candidate | score | best factor | factor overlap | Dice | Recall |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for _, row in top.iterrows():
                dice = row.get("metric_dice", math.nan)
                recall = row.get("metric_recall", math.nan)
                lines.append(
                    f"| {int(row['candidate_id'])} | {row['ranking_score']:.3f} | "
                    f"{int(row['best_overlap_factor']) if not pd.isna(row['best_overlap_factor']) else ''} | "
                    f"{row['best_factor_overlap']:.3f} | {dice:.3f} | {recall:.3f} |"
                )
            lines.append("")
    (outdir / "RESULTS.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run molecular-prior SAM3 experiments.")
    parser.add_argument("--ficture-grid", type=Path, default=DEFAULT_FICTURE_GRID)
    parser.add_argument("--factor-info", type=Path, default=DEFAULT_FACTOR_INFO)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sam3-proposal-root", type=Path, default=DEFAULT_SAM3_PROPOSAL_ROOT)
    parser.add_argument("--target-mask-dir", type=Path, default=DEFAULT_TARGET_MASK_DIR)
    parser.add_argument("--shift-x-px", type=float, default=-60.0)
    parser.add_argument("--shift-y-px", type=float, default=80.0)
    parser.add_argument("--max-color-distance", type=float, default=10.0)
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737554241192739)
    parser.add_argument("--tissue-hires-scalef", type=float, default=0.13752006)
    parser.add_argument("--ficture-res-um-per-pixel", type=float, default=2.0)
    parser.add_argument("--ficture-xmin-um", type=float, default=-10.934)
    parser.add_argument("--ficture-ymin-um", type=float, default=-156.145)
    parser.add_argument("--min-component-area", type=int, default=2500)
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--top-components-per-factor", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Image.MAX_IMAGE_PIXELS = None
    he = np.array(Image.open(args.he_image).convert("RGB"))
    grid = np.array(Image.open(args.ficture_grid).convert("RGB"))
    info = load_factor_info(args.factor_info)
    tissue_mask = estimate_tissue_mask(he)
    label_image, mapping_stats = build_label_image(
        he.shape[:2],
        grid,
        info,
        max_color_distance=args.max_color_distance,
        microns_per_pixel=args.microns_per_pixel,
        tissue_hires_scalef=args.tissue_hires_scalef,
        ficture_res_um_per_pixel=args.ficture_res_um_per_pixel,
        ficture_xmin_um=args.ficture_xmin_um,
        ficture_ymin_um=args.ficture_ymin_um,
        shift_x_px=args.shift_x_px,
        shift_y_px=args.shift_y_px,
        tissue_mask=tissue_mask,
    )
    np.save(args.output_dir / "ficture_factor_label_image_he.npy", label_image)
    Image.fromarray(((label_image >= 0).astype(np.uint8) * 255)).save(args.output_dir / "molecular_support_mask_he.png")

    components, factor_masks, annotations = save_factor_masks_and_components(
        he,
        label_image,
        info,
        args.output_dir,
        min_component_area=args.min_component_area,
        close_radius=args.close_radius,
        top_components_per_factor=args.top_components_per_factor,
    )
    targets = load_target_masks(args.target_mask_dir)
    method_a = evaluate_method_a(args.output_dir, factor_masks, components, targets, annotations)
    method_bcd = evaluate_methods_bcd(args.output_dir, args.sam3_proposal_root, factor_masks, targets, annotations)
    write_summary_tables(args.output_dir, method_a, method_bcd)

    summary = {
        "ficture_grid": str(args.ficture_grid),
        "factor_info": str(args.factor_info),
        "he_image": str(args.he_image),
        "output_dir": str(args.output_dir),
        "shift_x_px": args.shift_x_px,
        "shift_y_px": args.shift_y_px,
        "mapping_stats": mapping_stats,
        "n_factor_masks": len(factor_masks),
        "n_components": len(components),
        "target_labels": sorted(targets.keys()),
        "method_a_rows": int(len(method_a)),
        "method_bcd_rows": int(len(method_bcd)),
    }
    (args.output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
