#!/usr/bin/env python3
"""Build candidate-pool input ROI from the official filtered FICTURE map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OFFICIAL_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "ficture_official_filtered_he_aligned"
DEFAULT_FICTURE_FULL = DEFAULT_OFFICIAL_DIR / "filtered_ficture_official_full_he_canvas.png"
DEFAULT_OFFICIAL_SUMMARY = DEFAULT_OFFICIAL_DIR / "summary_official.json"
DEFAULT_HE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_FACTOR_INFO = (
    PROJECT_ROOT
    / "pixel-level cell type image"
    / "visiumhd_exp1_hex12_k12"
    / "hex_12.k12.pixel.info.tsv"
)
DEFAULT_LABEL_SUMMARY = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_local_region_summary" / "region_summary.json"
DEFAULT_OUT = PROJECT_ROOT / "output" / "visium_hd_exp1" / "ficture_official_filtered_candidate_pool_input_roi"
DEFAULT_BBOX = (75, 40, 3219, 3367)
HPC_PROJECT = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3")


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def to_hpc(path: Path) -> str:
    return str(HPC_PROJECT / project_relative(path))


def crop_image(src: Path, bbox: tuple[int, int, int, int], dst: Path) -> Image.Image:
    img = Image.open(src).convert("RGB")
    cropped = img.crop(bbox)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dst)
    return cropped


def crop_mask(src: Path, bbox: tuple[int, int, int, int], dst: Path) -> int:
    mask = Image.open(src).convert("L").crop(bbox)
    arr = np.array(mask) > 0
    out = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return int(arr.sum())


def make_overlay(he_roi: Image.Image, ficture_roi: Image.Image, alpha: float = 0.55) -> Image.Image:
    he = np.array(he_roi.convert("RGB")).astype(np.float32)
    fic = np.array(ficture_roi.convert("RGB")).astype(np.float32)
    mask = fic.sum(axis=2) > 0
    out = he.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * fic[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def load_factor_palette(path: Path) -> tuple[np.ndarray, np.ndarray]:
    factors: list[int] = []
    colors: list[tuple[int, int, int]] = []
    with path.open() as handle:
        header = handle.readline().strip().split("\t")
        factor_idx = header.index("Factor")
        rgb_idx = header.index("RGB")
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            factors.append(int(fields[factor_idx]))
            colors.append(tuple(int(v) for v in fields[rgb_idx].split(",")))
    if not factors:
        raise ValueError(f"No FICTURE factor colors found in {path}")
    return np.asarray(factors, dtype=np.int16), np.asarray(colors, dtype=np.int16)


def factor_index_from_rgb(ficture_roi: Image.Image, factor_info: Path, dst: Path) -> dict[str, int | float | list[int] | str]:
    """Decode official FICTURE ROI colors to factor ids for ranking priors.

    The official ROI image is the PASS_OFFICIAL filtered FICTURE canvas crop.
    Most colors are exact palette colors from hex_12.k12.pixel.info.tsv; a tiny
    number of pixels may come from the approved one-pixel seam fill, so those
    are assigned to the nearest official palette color.
    """
    factors, palette = load_factor_palette(factor_info)
    rgb = np.asarray(ficture_roi.convert("RGB"), dtype=np.uint8)
    nonblack = rgb.sum(axis=2) > 0
    out = np.full(rgb.shape[:2], -1, dtype=np.int16)
    if nonblack.any():
        colors = rgb[nonblack].astype(np.int16)
        unique_colors, inverse = np.unique(colors, axis=0, return_inverse=True)
        diff = unique_colors[:, None, :] - palette[None, :, :]
        dist = np.sum(diff.astype(np.int32) ** 2, axis=2)
        nearest = np.argmin(dist, axis=1)
        mapped = factors[nearest]
        out[nonblack] = mapped[inverse]
        exact = dist[np.arange(dist.shape[0]), nearest] == 0
        color_counts = np.bincount(inverse, minlength=len(unique_colors))
        exact_pixels = int(color_counts[exact].sum())
        nearest_only_colors = int((~exact).sum())
    else:
        unique_colors = np.empty((0, 3), dtype=np.int16)
        exact_pixels = 0
        nearest_only_colors = 0
    np.save(dst, out)
    valid = out >= 0
    return {
        "factor_index_path": project_relative(dst),
        "factor_info_tsv": project_relative(factor_info),
        "shape_hw": [int(out.shape[0]), int(out.shape[1])],
        "dtype": str(out.dtype),
        "background_pixels": int((~valid).sum()),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()) if out.size else 0.0,
        "unique_rgb_colors_seen": int(len(unique_colors)),
        "nearest_palette_unique_colors": nearest_only_colors,
        "exact_palette_pixel_fraction": float(exact_pixels / valid.sum()) if valid.any() else 1.0,
        "factor_ids_present": [int(v) for v in np.unique(out[valid]).tolist()] if valid.any() else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official filtered FICTURE candidate-pool ROI input.")
    parser.add_argument("--official-ficture-full", type=Path, default=DEFAULT_FICTURE_FULL)
    parser.add_argument("--official-summary", type=Path, default=DEFAULT_OFFICIAL_SUMMARY)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--factor-info", type=Path, default=DEFAULT_FACTOR_INFO)
    parser.add_argument("--label-summary", type=Path, default=DEFAULT_LABEL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--crop-bbox", type=int, nargs=4, default=DEFAULT_BBOX, metavar=("X1", "Y1", "X2", "Y2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bbox = tuple(int(v) for v in args.crop_bbox)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop bbox: {bbox}")

    official_summary = json.loads(args.official_summary.read_text())
    if official_summary.get("status") != "PASS_OFFICIAL":
        raise RuntimeError(f"Official FICTURE summary is not PASS_OFFICIAL: {args.official_summary}")

    ficture_roi_path = args.output_dir / "ficture_official_filtered_roi_rgb.png"
    he_roi_path = args.output_dir / "he_roi_matching_official_ficture_coverage.png"
    overlay_path = args.output_dir / "ficture_official_filtered_overlay_roi.png"
    ficture_roi = crop_image(args.official_ficture_full, bbox, ficture_roi_path)
    he_roi = crop_image(args.he_image, bbox, he_roi_path)
    make_overlay(he_roi, ficture_roi).save(overlay_path)
    factor_index_path = args.output_dir / "ficture_official_filtered_roi_factor_index.npy"
    factor_index_metrics = factor_index_from_rgb(ficture_roi, args.factor_info, factor_index_path)

    if ficture_roi.size != he_roi.size:
        raise RuntimeError(f"FICTURE ROI {ficture_roi.size} and H&E ROI {he_roi.size} differ")
    if factor_index_metrics["shape_hw"] != [ficture_roi.size[1], ficture_roi.size[0]]:
        raise RuntimeError(f"Factor index shape does not match FICTURE ROI: {factor_index_metrics}")

    label_summary = json.loads(args.label_summary.read_text())
    labels = []
    mask_dir = args.output_dir / "cropped_annotation_masks"
    for idx, item in enumerate(label_summary["labels"], start=1):
        src_mask = Path(item["mask_path"])
        dst_mask = mask_dir / f"{idx:02d}_{item['slug']}_target_roi.png"
        area = crop_mask(src_mask, bbox, dst_mask)
        labels.append(
            {
                "label": item["label"],
                "slug": item["slug"],
                "mask_path": project_relative(dst_mask),
                "area_pixels": area,
            }
        )

    summary = {
        "image_path": project_relative(ficture_roi_path),
        "factor_index_path": project_relative(factor_index_path),
        "full_canvas_image_path": project_relative(args.official_ficture_full),
        "he_roi_image_path": project_relative(he_roi_path),
        "overlay_path": project_relative(overlay_path),
        "official_summary": project_relative(args.official_summary),
        "crop_bbox_xyxy_exclusive": list(bbox),
        "crop_size_wh": [x2 - x1, y2 - y1],
        "factor_index": factor_index_metrics,
        "labels": labels,
        "source": {
            "pipeline": project_relative(Path(__file__)),
            "filtered_ficture_full_canvas": project_relative(args.official_ficture_full),
            "filtered_source_png": official_summary["source_paths"]["filtered_ficture_png"],
            "formula": official_summary["formula"],
            "orientation": official_summary["orientation"]["selected"],
            "status": official_summary["status"],
        },
        "note": (
            "Official candidate-pool input: crop from PASS_OFFICIAL filtered FICTURE full canvas. "
            "Uses original full-slide annotation masks cropped to the unchanged candidate-pool bbox."
        ),
    }
    local_summary_path = args.output_dir / "region_summary_official_filtered_roi.json"
    local_summary_path.write_text(json.dumps(summary, indent=2))

    hpc_summary = json.loads(json.dumps(summary))
    hpc_summary["image_path"] = to_hpc(ficture_roi_path)
    hpc_summary["factor_index_path"] = to_hpc(factor_index_path)
    hpc_summary["factor_index"]["factor_index_path"] = to_hpc(factor_index_path)
    hpc_summary["factor_index"]["factor_info_tsv"] = to_hpc(args.factor_info)
    hpc_summary["full_canvas_image_path"] = to_hpc(args.official_ficture_full)
    hpc_summary["he_roi_image_path"] = to_hpc(he_roi_path)
    hpc_summary["overlay_path"] = to_hpc(overlay_path)
    hpc_summary["official_summary"] = to_hpc(args.official_summary)
    for item in hpc_summary["labels"]:
        item["mask_path"] = to_hpc(PROJECT_ROOT / item["mask_path"])
    hpc_summary_path = args.output_dir / "region_summary_official_filtered_roi_hpc.json"
    hpc_summary_path.write_text(json.dumps(hpc_summary, indent=2))

    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official Filtered FICTURE Candidate-Pool Input",
                "",
                "Generated by `inference/visium_hd_exp1/build_official_ficture_candidate_pool_input.py`.",
                "",
                "This directory is the official input for rerunning FICTURE-map candidate pools.",
                "It is cropped from `output/visium_hd_exp1/ficture_official_filtered_he_aligned/filtered_ficture_official_full_he_canvas.png`.",
                "",
                f"Crop bbox xyxy exclusive: `{list(bbox)}`.",
                f"ROI size: `{x2 - x1} x {y2 - y1}`.",
                "",
                "The source FICTURE summary must be `PASS_OFFICIAL`.",
                "",
            ]
        )
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir),
                "image": str(ficture_roi_path),
                "summary": str(local_summary_path),
                "hpc_summary": str(hpc_summary_path),
                "crop_size_wh": [x2 - x1, y2 - y1],
                "n_labels": len(labels),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
