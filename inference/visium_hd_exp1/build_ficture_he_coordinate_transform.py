#!/usr/bin/env python3
"""Build a coordinate transform from FICTURE pixel-map space to 10x H&E hires space.

The FICTURE/punkst pixel map is rendered from spatial coordinates in microns.
The 10x spatial files store the same locations as full-resolution H&E pixels.
This script makes that relationship explicit so reference-map candidates can be
mapped to H&E without fitting a loose tissue-outline affine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, warp


DEFAULT_FICTURE_MAP = Path("/Users/haoranwu/Desktop/visiumhd_exp1_pixel_cell_type_result/hex_12.k12.pixel.png")
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_OUT = PROJECT_ROOT / "output" / "visium_hd_exp1" / "registration_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FICTURE pixel-map to H&E hires transform.")
    parser.add_argument("--ficture-map", type=Path, default=DEFAULT_FICTURE_MAP)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-prefix", default="ficture_coordinate")
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737554241192739)
    parser.add_argument("--tissue-hires-scale", type=float, default=0.13752006)
    parser.add_argument("--ficture-res-um-per-pixel", type=float, default=2.0)
    parser.add_argument("--ficture-xmin-um", type=float, default=-10.934)
    parser.add_argument("--ficture-ymin-um", type=float, default=-156.145)
    parser.add_argument(
        "--source-rotation",
        choices=["none", "rot90"],
        default="none",
        help="Coordinate convention of the FICTURE map. Use rot90 for hex_12.k12.pixel.rot90*.png files.",
    )
    parser.add_argument("--flip-x", action="store_true", help="Mirror mapped H&E x coordinates across the target image width.")
    parser.add_argument("--flip-y", action="store_true", help="Mirror mapped H&E y coordinates across the target image height.")
    parser.add_argument("--shift-x-px", type=float, default=0.0, help="Extra x translation in target H&E pixels after flips.")
    parser.add_argument("--shift-y-px", type=float, default=0.0, help="Extra y translation in target H&E pixels after flips.")
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def build_matrix(args: argparse.Namespace, source_width: int, target_width: int, target_height: int) -> np.ndarray:
    scale = args.tissue_hires_scale / args.microns_per_pixel
    step_x = args.ficture_res_um_per_pixel * scale
    step_y = args.ficture_res_um_per_pixel * scale
    x_offset = args.ficture_ymin_um * scale
    y_offset = args.ficture_xmin_um * scale
    if args.source_rotation == "rot90":
        # np.rot90(raw, 1): raw_col = raw_width - 1 - rot_row, raw_row = rot_col.
        # H&E x follows the original raw column/Y coordinate; H&E y follows raw row/X.
        matrix = np.array(
            [
                [0.0, -step_x, step_x * (source_width - 1) + x_offset],
                [step_y, 0.0, y_offset],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        matrix[0, 2] += args.shift_x_px
        matrix[1, 2] += args.shift_y_px
        return matrix
    if args.flip_x:
        step_x = -step_x
        x_offset = float(target_width - 1) - x_offset
    if args.flip_y:
        step_y = -step_y
        y_offset = float(target_height - 1) - y_offset
    x_offset += args.shift_x_px
    y_offset += args.shift_y_px
    return np.array(
        [
            [step_x, 0.0, x_offset],
            [0.0, step_y, y_offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_overlay(ficture: np.ndarray, he: np.ndarray, matrix: np.ndarray, alpha: float) -> np.ndarray:
    transform = AffineTransform(matrix=matrix)
    warped = warp(
        ficture.astype(np.float32),
        inverse_map=transform.inverse,
        output_shape=he.shape[:2],
        order=0,
        preserve_range=True,
    ).astype(np.uint8)
    mask = warped.sum(axis=2) > 12
    overlay = he.copy().astype(np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * warped[mask].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ficture = np.array(Image.open(args.ficture_map).convert("RGB"))
    he = np.array(Image.open(args.he_image).convert("RGB"))
    raw_source_width = ficture.shape[1]
    if args.source_rotation == "rot90":
        # The pre-rotated image width equals raw height; raw width is therefore
        # the pre-rotated image height.
        raw_source_width = ficture.shape[0]
    matrix = build_matrix(args, source_width=raw_source_width, target_width=he.shape[1], target_height=he.shape[0])

    json_path = args.output_dir / f"{args.output_prefix}_ref_to_he.json"
    payload = {
        "description": "FICTURE pixel-map pixel coordinates to 10x tissue_hires_image pixel coordinates.",
        "coordinate_formula": {
            "source_pixel": ["u = FICTURE map column", "v = FICTURE map row"],
            "ficture_um": [
                "Y_um = ficture_ymin_um + u * ficture_res_um_per_pixel",
                "X_um = ficture_xmin_um + v * ficture_res_um_per_pixel",
            ],
            "he_hires_pixel": [
                "he_x = Y_um / microns_per_pixel * tissue_hires_scale",
                "he_y = X_um / microns_per_pixel * tissue_hires_scale",
            ],
        },
        "ficture_map": str(args.ficture_map),
        "he_image": str(args.he_image),
        "source_size": [int(ficture.shape[1]), int(ficture.shape[0])],
        "target_size": [int(he.shape[1]), int(he.shape[0])],
        "source_rotation": args.source_rotation,
        "microns_per_pixel": args.microns_per_pixel,
        "tissue_hires_scale": args.tissue_hires_scale,
        "ficture_res_um_per_pixel": args.ficture_res_um_per_pixel,
        "ficture_xmin_um": args.ficture_xmin_um,
        "ficture_ymin_um": args.ficture_ymin_um,
        "flip_x": bool(args.flip_x),
        "flip_y": bool(args.flip_y),
        "shift_x_px": args.shift_x_px,
        "shift_y_px": args.shift_y_px,
        "full_matrix_ref_to_he": matrix.tolist(),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    overlay = make_overlay(ficture, he, matrix, args.alpha)
    overlay_path = args.output_dir / f"{args.output_prefix}_overlay_check.png"
    Image.fromarray(overlay).save(overlay_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    axes[0].imshow(he)
    axes[0].set_title("H&E hires image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("FICTURE reference map over H&E using coordinate transform")
    axes[1].axis("off")
    fig.tight_layout()
    panel_path = args.output_dir / f"{args.output_prefix}_overlay_panel.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"transform": str(json_path), "overlay": str(overlay_path), "panel": str(panel_path)}, indent=2))


if __name__ == "__main__":
    main()
