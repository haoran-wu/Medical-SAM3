#!/usr/bin/env python3
"""Build an annotation-free 8-class map from all FICTURE-guided SAM runs.

Manual annotations are intentionally not read here. Candidate masks are selected
from completed SAM/Medical-SAM3 FICTURE refine runs by marker genes, the
candidate's own molecular-prior consistency metrics, and class-specific size
constraints. The output format mirrors the diagnostic 8-class figures so the
result can be inspected like the training-assisted upper bound.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "results" / "visium_hd_exp1" / "sam3_reference_candidate_refine"
DEFAULT_HE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "no_training_multirun_8class"

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

MARKERS: Dict[str, set[str]] = {
    "lung_bronchiola": {"SCGB1A1", "SCGB3A1", "BPIFB1", "MSMB", "SLPI", "MUC5B", "MUC5AC"},
    "immune_infiltration": {"PTPRC", "LYZ", "SPP1", "FTL", "APOE", "CTSB", "CHIT1", "IGKC", "IGHG1", "IGHM", "IGLC1", "CD74", "CXCR4", "IL7R"},
    "lung_alveoli_normal_adjacent": {"SFTPB", "SFTPC", "SFTPA1", "SFTPA2", "LPCAT1", "NPC2", "AGER", "CAV1"},
    "lung_vessels": {"MYL9", "TAGLN", "MYH11", "ACTA2", "DES", "RGS5", "PECAM1", "VWF", "KDR"},
    "stroma": {"COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "MGP", "FN1", "THY1", "MYL9", "TAGLN", "ACTA2"},
    "tumor": {"CEACAM5", "SPINK1", "WFDC2", "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "IFI6", "ECM1"},
}

CLASS_LIMITS = {
    "lung_bronchiola": (600, 60000, 3),
    "immune_infiltration": (400, 30000, 6),
    "lung_alveoli_normal_adjacent": (4000, 450000, 5),
    "lung_vessels": (1500, 50000, 2),
    "stroma": (40000, 1800000, 2),
    "tumor": (8000, 450000, 5),
}


@dataclass
class Candidate:
    run: str
    run_dir: Path
    factor: int
    rank: int
    score: float
    dice_prior: float
    recall_prior: float
    precision_prior: float
    pixels: int
    mask_path: Path
    top_genes: str


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    scale = min(float(max_side) / max(image.size), 1.0)
    size = (int(round(image.size[0] * scale)), int(round(image.size[1] * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_candidates(runs_root: Path) -> List[Candidate]:
    out: List[Candidate] = []
    for csv_path in sorted(runs_root.glob("*/ranked_sam3_refined_candidates.csv")):
        run_dir = csv_path.parent
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                mask_rel = row.get("mask_path", "")
                if not mask_rel:
                    continue
                mask_path = run_dir / mask_rel
                if not mask_path.exists():
                    continue
                out.append(
                    Candidate(
                        run=run_dir.name,
                        run_dir=run_dir,
                        factor=safe_int(row.get("factor")),
                        rank=safe_int(row.get("rank_within_factor")),
                        score=safe_float(row.get("score")),
                        dice_prior=safe_float(row.get("dice_vs_coordinate_prior")),
                        recall_prior=safe_float(row.get("recall_vs_coordinate_prior")),
                        precision_prior=safe_float(row.get("precision_vs_coordinate_prior")),
                        pixels=safe_int(row.get("prediction_pixels")),
                        mask_path=mask_path,
                        top_genes=row.get("top_genes_specific", "") or "",
                    )
                )
    return out


def gene_tokens(text: str) -> set[str]:
    tokens = text.replace(";", ",").replace("|", ",").split(",")
    return {token.strip().upper() for token in tokens if token.strip()}


def candidate_class_score(candidate: Candidate, label: str) -> float:
    genes = gene_tokens(candidate.top_genes)
    hits = len(genes & MARKERS[label])
    if hits == 0:
        return -1e9
    min_px, max_px, _ = CLASS_LIMITS[label]
    if candidate.pixels < min_px or candidate.pixels > max_px:
        size_penalty = -0.75
    else:
        center = math.sqrt(min_px * max_px)
        size_penalty = -0.08 * abs(math.log(max(candidate.pixels, 1) / center))
    run_bonus = 0.08 if "medical" in candidate.run else 0.0
    return (
        2.0 * hits
        + 1.25 * candidate.score
        + 0.70 * candidate.dice_prior
        + 0.20 * candidate.precision_prior
        + 0.10 * candidate.recall_prior
        + size_penalty
        + run_bonus
    )


def read_mask(candidate: Candidate, size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.open(candidate.mask_path).convert("L").resize(size, Image.Resampling.NEAREST)) > 127


def union_selected(selected: Sequence[Candidate], size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    out = np.zeros((height, width), dtype=bool)
    for candidate in selected:
        out |= read_mask(candidate, size)
    return out


def choose_nonredundant(
    candidates: Sequence[Candidate],
    label: str,
    size: Tuple[int, int],
    max_overlap: float,
) -> List[Candidate]:
    scored = sorted(
        ((candidate_class_score(candidate, label), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    _, _, limit = CLASS_LIMITS[label]
    selected: List[Candidate] = []
    selected_union = np.zeros((size[1], size[0]), dtype=bool)
    used_factor_rank: set[Tuple[int, int, str]] = set()
    for score, candidate in scored:
        if score < 0 or len(selected) >= limit:
            break
        key = (candidate.factor, candidate.rank, candidate.top_genes)
        if key in used_factor_rank:
            continue
        mask = read_mask(candidate, size)
        if not mask.any():
            continue
        overlap = float((mask & selected_union).sum()) / float(mask.sum())
        if overlap > max_overlap:
            continue
        selected.append(candidate)
        selected_union |= mask
        used_factor_rank.add(key)
    return selected


def build_tissue_mask(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    tissue = ((r < 245) | (g < 240) | (b < 245)) & ~((r > 238) & (g > 235) & (b > 235))
    tissue_img = Image.fromarray(tissue.astype(np.uint8) * 255)
    tissue_img = tissue_img.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(7))
    return np.array(tissue_img) > 127


def build_he_rule_masks(image: Image.Image, tissue: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = np.array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    pigment = ((r < 95) & (g < 82) & (b < 98) & ((r.astype(int) + g.astype(int) + b.astype(int)) < 235) & tissue)
    pigment = np.array(Image.fromarray(pigment.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 127

    erythrocytes = (
        (r > 150)
        & (g < 120)
        & (b < 155)
        & ((r.astype(int) - g.astype(int)) > 45)
        & ((r.astype(int) - b.astype(int)) > 15)
        & tissue
    )
    erythrocytes = np.array(
        Image.fromarray(erythrocytes.astype(np.uint8) * 255)
        .filter(ImageFilter.MinFilter(3))
        .filter(ImageFilter.MaxFilter(5))
    ) > 127
    return {"pigment": pigment, "erythorocytes": erythrocytes}


def build_class_masks(
    candidates: Sequence[Candidate],
    image: Image.Image,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Candidate]], Dict[str, str]]:
    size = image.size
    selected: Dict[str, List[Candidate]] = {}
    rules: Dict[str, str] = {}
    for label in [
        "lung_bronchiola",
        "lung_vessels",
        "immune_infiltration",
        "lung_alveoli_normal_adjacent",
        "tumor",
        "stroma",
    ]:
        max_overlap = 0.75 if label in {"lung_alveoli_normal_adjacent", "tumor", "stroma"} else 0.55
        selected[label] = choose_nonredundant(candidates, label, size, max_overlap=max_overlap)
        rules[label] = "marker-gene candidate selection across completed FICTURE-to-SAM runs"

    masks = {label: union_selected(items, size) for label, items in selected.items()}
    if masks["lung_vessels"].any():
        masks["stroma"] &= ~masks["lung_vessels"]
    if masks["tumor"].any():
        masks["lung_alveoli_normal_adjacent"] &= ~masks["tumor"]
    tissue = build_tissue_mask(image)
    masks.update(build_he_rule_masks(image, tissue))
    rules["pigment"] = "H&E unsupervised dark-pixel threshold clipped to tissue"
    rules["erythorocytes"] = "H&E unsupervised eosin/RBC color threshold clipped to tissue"
    return masks, selected, rules


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
    legend_width = 610
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


def render_outputs(
    image: Image.Image,
    masks: Dict[str, np.ndarray],
    selected: Dict[str, List[Candidate]],
    output_dir: Path,
    rules: Dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = make_label_map(masks)

    for label, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{label}_no_training_mask_preview.png")

    label_rgb = np.ones((*label_map.shape, 3), dtype=np.uint8) * 255
    for idx, label in enumerate(LABEL_ORDER, start=1):
        label_rgb[label_map == idx] = COLORS[label]
    add_legend(Image.fromarray(label_rgb), "No-training 8-class mask", "Multi-run FICTURE prior + SAM").save(
        output_dir / "no_training_8class_labelmap_preview.png"
    )

    base = np.array(image).astype(np.float32)
    color_layer = np.zeros_like(base)
    alpha = np.zeros(label_map.shape, dtype=np.float32)
    for idx, label in enumerate(LABEL_ORDER, start=1):
        selected_pixels = label_map == idx
        color_layer[selected_pixels] = COLORS[label]
        alpha[selected_pixels] = 0.5
    overlay = (base * (1 - alpha[..., None]) + color_layer * alpha[..., None]).clip(0, 255).astype(np.uint8)
    add_legend(Image.fromarray(overlay), "No-training 8-class overlay", "Marker-gene multi-run selection").save(
        output_dir / "no_training_8class_overlay_preview.png"
    )

    with (output_dir / "no_training_8class_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "pixels",
                "rule",
                "run",
                "factor",
                "rank",
                "score",
                "dice_prior",
                "precision_prior",
                "recall_prior",
                "mask_path",
                "top_genes_specific",
            ],
        )
        writer.writeheader()
        for label in LABEL_ORDER:
            items = selected.get(label, [])
            if not items:
                writer.writerow({"label": label, "pixels": int(masks[label].sum()), "rule": rules[label]})
                continue
            for candidate in items:
                writer.writerow(
                    {
                        "label": label,
                        "pixels": int(masks[label].sum()),
                        "rule": rules[label],
                        "run": candidate.run,
                        "factor": candidate.factor,
                        "rank": candidate.rank,
                        "score": f"{candidate.score:.4f}",
                        "dice_prior": f"{candidate.dice_prior:.4f}",
                        "precision_prior": f"{candidate.precision_prior:.4f}",
                        "recall_prior": f"{candidate.recall_prior:.4f}",
                        "mask_path": str(candidate.mask_path),
                        "top_genes_specific": candidate.top_genes,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-side", type=int, default=2200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = resize_image(Image.open(args.he_image).convert("RGB"), args.max_side)
    candidates = read_candidates(args.runs_root)
    if not candidates:
        raise RuntimeError(f"No ranked SAM/FICTURE candidates found under {args.runs_root}")
    masks, selected, rules = build_class_masks(candidates, image)
    render_outputs(image, masks, selected, args.output_dir, rules)
    print(args.output_dir / "no_training_8class_labelmap_preview.png")
    print(args.output_dir / "no_training_8class_overlay_preview.png")
    print(args.output_dir / "no_training_8class_records.csv")


if __name__ == "__main__":
    main()
