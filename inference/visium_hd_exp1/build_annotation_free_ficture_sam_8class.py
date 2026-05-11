#!/usr/bin/env python3
"""Build an annotation-free 8-class proxy labelmap from FICTURE-guided SAM outputs.

This script deliberately does not train on, read, or optimize against manual
annotations. It converts factor-level SAM-refined components into the same
visual output format as the supervised/diagnostic best-by-label figures:
individual class masks, a mask-only labelmap, an H&E overlay, and a records CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "visium_hd_exp1"
    / "sam3_reference_candidate_refine"
    / "he_ficturecoord_quality_top12_hi4096_11240236"
)
DEFAULT_HE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "no_training_best_8class"

LABEL_ORDER = [
    "lung_bronchiola",
    "erythorocytes",
    "immune_infiltration",
    "lung_alveoli_normal_adjacent",
    "lung_vessels",
    "pigment",
    "stroma",
    "tumor",
]

PAINT_ORDER = [
    "pigment",
    "erythorocytes",
    "lung_bronchiola",
    "lung_vessels",
    "immune_infiltration",
    "lung_alveoli_normal_adjacent",
    "tumor",
    "stroma",
]

COLORS: Dict[str, Tuple[int, int, int]] = {
    "lung_bronchiola": (31, 119, 180),
    "erythorocytes": (255, 127, 14),
    "immune_infiltration": (44, 160, 44),
    "lung_alveoli_normal_adjacent": (148, 103, 189),
    "lung_vessels": (140, 86, 75),
    "pigment": (127, 127, 127),
    "stroma": (188, 189, 34),
    "tumor": (23, 190, 207),
}

PRETTY = {
    "lung_bronchiola": "Lung Bronchiola",
    "erythorocytes": "erythorocytes",
    "immune_infiltration": "immune infiltration",
    "lung_alveoli_normal_adjacent": "lung alveoli normal adjacent",
    "lung_vessels": "lung vessels",
    "pigment": "pigment",
    "stroma": "stroma",
    "tumor": "tumor",
}


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    scale = min(float(max_side) / max(image.size), 1.0)
    size = (int(round(image.size[0] * scale)), int(round(image.size[1] * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def read_mask(run_dir: Path, rel_path: str, size: Tuple[int, int]) -> np.ndarray:
    path = run_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.array(mask) > 127


def union_masks(run_dir: Path, rel_paths: Iterable[str], size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    out = np.zeros((height, width), dtype=bool)
    for rel_path in rel_paths:
        out |= read_mask(run_dir, rel_path, size)
    return out


def build_tissue_mask(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    tissue = ((r < 245) | (g < 240) | (b < 245)) & ~((r > 238) & (g > 235) & (b > 235))
    return np.array(Image.fromarray(tissue.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 127


def build_he_rule_masks(image: Image.Image, tissue: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = np.array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    pigment = ((r < 95) & (g < 80) & (b < 95) & ((r.astype(int) + g.astype(int) + b.astype(int)) < 230) & tissue)
    pigment = np.array(Image.fromarray(pigment.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 127

    erythrocytes = (
        (r > 150)
        & (g < 115)
        & (b < 150)
        & ((r.astype(int) - g.astype(int)) > 50)
        & ((r.astype(int) - b.astype(int)) > 20)
        & tissue
    )
    erythrocytes = np.array(
        Image.fromarray(erythrocytes.astype(np.uint8) * 255)
        .filter(ImageFilter.MinFilter(3))
        .filter(ImageFilter.MaxFilter(5))
    ) > 127
    return {"pigment": pigment, "erythorocytes": erythrocytes}


def build_class_masks(run_dir: Path, image: Image.Image) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    size = image.size
    masks: Dict[str, np.ndarray] = {}
    rules: Dict[str, str] = {}

    masks["lung_bronchiola"] = union_masks(run_dir, ["masks/011_factor_07_rank_03_sam3_refined.png"], size)
    rules["lung_bronchiola"] = "F7 SCGB1A1/SCGB3A1/BPIFB1 SAM-refined component"

    masks["lung_alveoli_normal_adjacent"] = union_masks(
        run_dir,
        [
            "masks/004_factor_03_rank_01_sam3_refined.png",
            "masks/007_factor_03_rank_02_sam3_refined.png",
            "masks/008_factor_03_rank_03_sam3_refined.png",
        ],
        size,
    )
    rules["lung_alveoli_normal_adjacent"] = "F3 SFTPB/SFTPC/LPCAT1 SAM-refined components"

    masks["lung_vessels"] = union_masks(run_dir, ["masks/001_factor_01_rank_02_sam3_refined.png"], size)
    rules["lung_vessels"] = "F1 MYL9/TAGLN/MYH11/ACTA2 local SAM-refined component"

    stroma_base = read_mask(run_dir, "masks/009_factor_01_rank_01_sam3_refined.png", size)
    masks["stroma"] = stroma_base & ~masks["lung_vessels"]
    rules["stroma"] = "Broad F1 MYL9/TAGLN/MYH11/ACTA2 SAM-refined component minus vessel component"

    masks["immune_infiltration"] = union_masks(
        run_dir,
        ["masks/000_factor_06_rank_03_sam3_refined.png", "masks/005_factor_08_rank_05_sam3_refined.png"],
        size,
    )
    rules["immune_infiltration"] = "F6 SPP1/LYZ/FTL plus F8 IG genes SAM-refined components"

    masks["tumor"] = union_masks(
        run_dir,
        [
            "masks/003_factor_00_rank_01_sam3_refined.png",
            "masks/006_factor_00_rank_03_sam3_refined.png",
            "masks/002_factor_02_rank_03_sam3_refined.png",
            "masks/010_factor_02_rank_04_sam3_refined.png",
        ],
        size,
    )
    rules["tumor"] = "F0/F2 CEACAM5/SPINK1/NAPSA epithelial/tumor-like SAM-refined components"

    tissue = build_tissue_mask(image)
    he_masks = build_he_rule_masks(image, tissue)
    masks.update(he_masks)
    rules["pigment"] = "H&E unsupervised dark-pixel threshold clipped to tissue"
    rules["erythorocytes"] = "H&E unsupervised eosin/RBC color threshold clipped to tissue"
    return masks, rules


def make_label_map(masks: Dict[str, np.ndarray]) -> np.ndarray:
    height, width = next(iter(masks.values())).shape
    label_map = np.zeros((height, width), dtype=np.uint8)
    for label in PAINT_ORDER:
        label_map[(masks[label]) & (label_map == 0)] = LABEL_ORDER.index(label) + 1
    return label_map


def load_fonts() -> Tuple[ImageFont.ImageFont | None, ImageFont.ImageFont | None, ImageFont.ImageFont | None]:
    try:
        return (
            ImageFont.truetype("DejaVuSans.ttf", 30),
            ImageFont.truetype("DejaVuSans.ttf", 20),
            ImageFont.truetype("DejaVuSans.ttf", 16),
        )
    except Exception:
        return None, None, None


def add_legend(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    width, height = image.size
    legend_width = 560
    canvas = Image.new("RGB", (width + legend_width, height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font, subtitle_font, small_font = load_fonts()
    x, y = width + 26, 32
    draw.text((x, y), title, fill=(0, 0, 0), font=title_font)
    y += 50
    draw.text((x, y), subtitle, fill=(0, 0, 0), font=subtitle_font)
    y += 36
    draw.text((x, y), "No annotation training or Dice selection", fill=(0, 0, 0), font=small_font)
    y += 38
    for label in LABEL_ORDER:
        draw.rectangle((x, y + 3, x + 27, y + 29), fill=COLORS[label])
        draw.text((x + 40, y), PRETTY[label], fill=(0, 0, 0), font=small_font)
        y += 31
    return canvas


def render_outputs(image: Image.Image, masks: Dict[str, np.ndarray], output_dir: Path, rules: Dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = make_label_map(masks)

    for label, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{label}_no_training_mask_preview.png")

    label_rgb = np.ones((*label_map.shape, 3), dtype=np.uint8) * 255
    for idx, label in enumerate(LABEL_ORDER, start=1):
        label_rgb[label_map == idx] = COLORS[label]
    add_legend(Image.fromarray(label_rgb), "No-training 8-class mask", "FICTURE prior + SAM refine").save(
        output_dir / "no_training_8class_labelmap_preview.png"
    )

    base = np.array(image).astype(np.float32)
    color_layer = np.zeros_like(base)
    alpha = np.zeros(label_map.shape, dtype=np.float32)
    for idx, label in enumerate(LABEL_ORDER, start=1):
        selected = label_map == idx
        color_layer[selected] = COLORS[label]
        alpha[selected] = 0.5
    overlay = (base * (1 - alpha[..., None]) + color_layer * alpha[..., None]).clip(0, 255).astype(np.uint8)
    add_legend(Image.fromarray(overlay), "No-training 8-class overlay", "Marker-gene mapping + H&E rules").save(
        output_dir / "no_training_8class_overlay_preview.png"
    )

    with (output_dir / "no_training_8class_records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "pixels", "rule"])
        writer.writeheader()
        for label in LABEL_ORDER:
            writer.writerow({"label": label, "pixels": int(masks[label].sum()), "rule": rules[label]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-side", type=int, default=2200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = resize_image(Image.open(args.he_image).convert("RGB"), args.max_side)
    masks, rules = build_class_masks(args.run_dir, image)
    render_outputs(image, masks, args.output_dir, rules)
    print(args.output_dir / "no_training_8class_labelmap_preview.png")
    print(args.output_dir / "no_training_8class_overlay_preview.png")
    print(args.output_dir / "no_training_8class_records.csv")


if __name__ == "__main__":
    main()
