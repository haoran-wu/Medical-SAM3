#!/usr/bin/env python3
"""Prepare six-image VLM inputs for every frozen corrected TMA30 candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


Image.MAX_IMAGE_PIXELS = None
CLASSES = ("airway", "arteriole", "venule", "alveoli")
EXPECTED_COMPONENTS = {"airway": 3, "arteriole": 4, "venule": 4, "alveoli": 3}
EVALUABLE_PURITY = 0.50
IMAGE_NAMES = (
    "01_close_he.jpg",
    "02_close_ficture.png",
    "03_medium_he.jpg",
    "04_medium_ficture.png",
    "05_full_he.jpg",
    "06_full_ficture.png",
)


SYSTEM_PROMPT = """Classify one exact SAM3 candidate from a registered lung tissue microarray.
You will receive close, medium, and full views in both H&E and K=12 strict-official raw FICTURE. A thin neutral black-and-white boundary marks the candidate; it is only a locator.
Use H&E as the main morphology evidence and FICTURE as supporting molecular evidence. Choose exactly one target class. Return only the requested JSON object."""


USER_TEMPLATE = """Candidate ID: {candidate_id}

Images 1-6 are close H&E, close FICTURE, medium H&E, medium FICTURE, full H&E, and full FICTURE. The close pair shows the tissue inside the boundary. The medium pair shows its immediately connected structure. The full pair shows its location on the TMA.

Fixed K=12 FICTURE RGB reference:
{ficture_rgb_reference}

Choose exactly one class:
- airway: bronchiolar or airway epithelial wall around one airway lumen.
- arteriole: an organized, relatively thick muscular arterial wall around a usually small and regular vascular lumen.
- venule: a thinner venous wall around a usually larger or more irregular vascular lumen.
- alveoli: delicate alveolar septa surrounding many small open airspaces.

Score all four classes from 0 to 100. The predicted class must have the highest score. If scores are tied, decide from H&E morphology first, then local architecture, then FICTURE. A candidate may show only part of its immediately connected parent structure.

Return exactly this structure and do not add keys:
{{
  "prompt_version": "{prompt_version}",
  "candidate_id": "{candidate_id}",
  "class_scores": {{
    "airway": 0,
    "arteriole": 0,
    "venule": 0,
    "alveoli": 0
  }},
  "predicted_class": "REPLACE_WITH_ONE_OF_THE_FOUR_CLASSES",
  "reason": "At most three short evidence-based sentences"
}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--he", type=Path, required=True)
    parser.add_argument("--ficture", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--factor-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if not mask.any():
        raise RuntimeError(f"Empty mask: {path}")
    return mask


def context_bounds(mask: np.ndarray, scale: float, minimum_side: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    height, width = mask.shape
    x1, x2 = int(xx.min()), int(xx.max()) + 1
    y1, y2 = int(yy.min()), int(yy.max()) + 1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(minimum_side, round(max(x2 - x1, y2 - y1) * scale)), width, height)
    left = max(0, min(round(center_x - side / 2), width - side))
    top = max(0, min(round(center_y - side / 2), height - side))
    return left, top, left + side, top + side


def boundary_view(
    image: Image.Image,
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    ficture: bool,
    max_side: int,
) -> Image.Image:
    x1, y1, x2, y2 = bounds
    crop = image.crop(bounds).convert("RGB")
    binary = mask[y1:y2, x1:x2]
    scale = min(1.0, max_side / max(crop.size))
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    shown = crop.resize(size, Image.Resampling.NEAREST if ficture else Image.Resampling.LANCZOS)
    binary_small = np.asarray(
        Image.fromarray(np.uint8(binary) * 255).resize(size, Image.Resampling.NEAREST)
    ) > 0
    values = np.asarray(shown, dtype=np.uint8).copy()
    boundary = binary_small & ~ndi.binary_erosion(binary_small, iterations=1)
    white = ndi.binary_dilation(boundary, iterations=3) & ~boundary
    values[white] = (255, 255, 255)
    values[boundary] = (18, 24, 28)
    return Image.fromarray(values)


def render_inputs(folder: Path, he: Image.Image, ficture: Image.Image, mask: np.ndarray) -> None:
    close = context_bounds(mask, 1.6, 600)
    medium = context_bounds(mask, 3.0, 1100)
    full = (0, 0, he.width, he.height)
    images = (
        boundary_view(he, mask, close, ficture=False, max_side=900),
        boundary_view(ficture, mask, close, ficture=True, max_side=900),
        boundary_view(he, mask, medium, ficture=False, max_side=1024),
        boundary_view(ficture, mask, medium, ficture=True, max_side=1024),
        boundary_view(he, mask, full, ficture=False, max_side=1100),
        boundary_view(ficture, mask, full, ficture=True, max_side=1100),
    )
    for name, image in zip(IMAGE_NAMES, images, strict=True):
        if name.endswith(".jpg"):
            image.save(folder / name, quality=94, subsampling=0, optimize=True)
        else:
            image.save(folder / name, optimize=True)


def annotation_components(root: Path, shape: tuple[int, int]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for label in CLASSES:
        path = root / f"TMA30_authoritative_{label}_mask.png"
        mask = load_binary(path)
        if mask.shape != shape:
            raise RuntimeError(f"Annotation shape mismatch: {path}")
        labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if count != EXPECTED_COMPONENTS[label]:
            raise RuntimeError(f"{label}: expected {EXPECTED_COMPONENTS[label]} regions, found {count}")
        for number in range(1, count + 1):
            component = labels == number
            components.append(
                {
                    "class": label,
                    "number": number,
                    "mask": component,
                    "area": int(component.sum()),
                }
            )
    return components


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    input_root = args.output / "input_images"
    input_root.mkdir()
    asset_root = args.output / "assets"
    asset_root.mkdir()

    pool_rows = read_csv(args.pool_csv)
    if not pool_rows:
        raise RuntimeError("Frozen pool is empty")
    if any(row["selection_used_annotation"].lower() != "false" for row in pool_rows):
        raise RuntimeError("The frozen pool is not annotation-blind")
    he = Image.open(args.he).convert("RGB")
    ficture = Image.open(args.ficture).convert("RGB")
    if he.size != ficture.size:
        raise RuntimeError("Registered H&E and FICTURE sizes differ")
    shape = (he.height, he.width)
    annotations = annotation_components(args.annotation_root, shape)

    source_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for order, pool_row in enumerate(pool_rows, start=1):
        mask_path = args.pool_csv.parent / pool_row["mask_path"]
        if sha256(mask_path) != pool_row["mask_sha256"]:
            raise RuntimeError(f"Mask hash mismatch: {pool_row['candidate_id']}")
        mask = load_binary(mask_path)
        if mask.shape != shape or int(mask.sum()) != int(pool_row["area_pixels"]):
            raise RuntimeError(f"Mask geometry mismatch: {pool_row['candidate_id']}")
        best: dict[str, Any] | None = None
        for annotation in annotations:
            shared = int(np.logical_and(mask, annotation["mask"]).sum())
            if best is None or shared > best["shared_pixels"]:
                best = {
                    "official_class_for_evaluation": annotation["class"],
                    "official_annotation": f"{annotation['class'].replace('_', ' ').title()} region {annotation['number']}",
                    "shared_pixels": shared,
                    "posthoc_precision": shared / int(mask.sum()),
                    "posthoc_recall": shared / annotation["area"],
                    "posthoc_dice": 2 * shared / (int(mask.sum()) + annotation["area"]),
                }
        assert best is not None
        candidate_id = f"TMA30 C{order:02d}"
        folder_name = f"Candidate_{order:03d}"
        folder = input_root / folder_name
        folder.mkdir()
        Image.fromarray(np.uint8(mask) * 255).save(folder / "candidate_mask.png", optimize=True)
        render_inputs(folder, he, ficture, mask)
        status = "evaluable" if best["posthoc_precision"] >= EVALUABLE_PURITY else "not_evaluable"
        source_counts[pool_row["source"]] += 1
        rows.append(
            {
                "order": order,
                "candidate_id": candidate_id,
                "candidate_uid": pool_row["candidate_id"],
                "source": pool_row["source"],
                "prompt_index": pool_row["prompt_index"],
                "prompt_id": pool_row["prompt_id"],
                "gene_module": pool_row["gene_module"],
                "folder": folder_name,
                "mask_sha256": pool_row["mask_sha256"],
                "evaluation_status": status,
                "posthoc_evaluation_threshold": EVALUABLE_PURITY,
                **best,
            }
        )

    copies = (
        (args.he, "TMA30_he.png"),
        (args.ficture, "TMA30_ficture.png"),
        (args.factor_info, "factor_info.csv"),
    )
    for source, name in copies:
        shutil.copy2(source, asset_root / name)
    for label in CLASSES:
        source = args.annotation_root / f"TMA30_authoritative_{label}_mask.png"
        shutil.copy2(source, asset_root / source.name)
    (args.output / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (args.output / "user_prompt_template.txt").write_text(USER_TEMPLATE, encoding="utf-8")
    (args.output / "vlm_source_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "frozen",
        "tma": "TMA30",
        "candidate_count": len(rows),
        "source_counts": dict(source_counts),
        "image_count_per_candidate": len(IMAGE_NAMES),
        "image_names": list(IMAGE_NAMES),
        "prompt_version": "tma30-fixed-tma39-six-image-four-class-v1",
        "official_annotation_region_count": len(annotations),
        "evaluable_rule": "candidate-to-best-GT purity at least 0.50",
        "evaluable_candidate_count": sum(row["evaluation_status"] == "evaluable" for row in rows),
        "selection_used_annotation": False,
        "input_hashes": {
            "he": sha256(asset_root / "TMA30_he.png"),
            "ficture": sha256(asset_root / "TMA30_ficture.png"),
            "factor_info": sha256(asset_root / "factor_info.csv"),
        },
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
