#!/usr/bin/env python3
"""Build a context-aware multi-image VLM pool from the compact piece pool."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 127


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def expanded_square_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    scale: float,
    min_side: int,
    max_side: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = int(max(x2 - x1, y2 - y1) * scale)
    side = max(min_side, min(max_side, side))
    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = nx1 + side
    ny2 = ny1 + side
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > width:
        nx1 -= nx2 - width
        nx2 = width
    if ny2 > height:
        ny1 -= ny2 - height
        ny2 = height
    return (max(0, nx1), max(0, ny1), min(width, nx2), min(height, ny2))


def fit_max_side(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
    return image


def context_crop(
    image: Image.Image,
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    max_side: int,
) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    crop = image.crop(crop_box).convert("RGBA")
    crop_mask = mask[y1:y2, x1:x2]
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    rgba = np.zeros((crop.height, crop.width, 4), dtype=np.uint8)
    rgba[crop_mask] = (0, 115, 255, 62)
    overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, mode="RGBA"))
    edge = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.FIND_EDGES)
    edge_rgba = np.zeros((crop.height, crop.width, 4), dtype=np.uint8)
    edge_np = np.array(edge) > 0
    edge_rgba[edge_np] = (255, 215, 0, 230)
    overlay = Image.alpha_composite(overlay, Image.fromarray(edge_rgba, mode="RGBA"))
    out = Image.alpha_composite(crop, overlay).convert("RGB")
    return fit_max_side(out, max_side)


def roi_locator(image: Image.Image, bbox: tuple[int, int, int, int], max_side: int) -> Image.Image:
    base = image.convert("RGB")
    scale = min(max_side / max(base.size), 1.0)
    small = base.resize(
        (max(1, int(base.width * scale)), max(1, int(base.height * scale))),
        Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(small)
    x1, y1, x2, y2 = bbox
    rect = [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)]
    for offset in range(4):
        draw.rectangle(
            [rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset],
            outline=(255, 215, 0),
        )
    return small


def copy_relative(src_root: Path, rel: str, dst_root: Path) -> str:
    src = src_root / rel
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return rel


def ensure_fields(fields: list[str], extra: list[str]) -> list[str]:
    out = list(fields)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--he-roi", type=Path, required=True)
    parser.add_argument("--ficture-roi", type=Path, required=True)
    parser.add_argument("--context-scale", type=float, default=4.0)
    parser.add_argument("--context-min-side", type=int, default=512)
    parser.add_argument("--context-max-side-fullres", type=int, default=1400)
    parser.add_argument("--output-max-side", type=int, default=768)
    parser.add_argument("--locator-max-side", type=int, default=900)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_dir = args.output_dir / "context_images"
    context_dir.mkdir(exist_ok=True)

    he = Image.open(args.he_roi).convert("RGB")
    ficture = Image.open(args.ficture_roi).convert("RGB")
    if he.size != ficture.size:
        raise SystemExit(f"H&E/FICTURE size mismatch: {he.size} vs {ficture.size}")

    public_rows = read_rows(args.source_pool / "public_vlm_requests.csv")
    hidden_rows = read_rows(args.source_pool / "hidden_candidate_truth.csv")
    if len(public_rows) != len(hidden_rows):
        raise SystemExit(f"public/hidden row count mismatch: {len(public_rows)} vs {len(hidden_rows)}")

    extra_fields = [
        "image1_rel",
        "image2_rel",
        "image3_rel",
        "image4_rel",
        "image5_rel",
        "image_sequence",
        "context_bbox_xyxy",
    ]
    public_fields = ensure_fields(list(public_rows[0].keys()), extra_fields)
    hidden_fields = ensure_fields(list(hidden_rows[0].keys()), extra_fields)

    for pub, hidden in zip(public_rows, hidden_rows):
        if pub.get("candidate_uid") != hidden.get("candidate_uid"):
            raise SystemExit(f"candidate_uid mismatch: {pub.get('candidate_uid')} vs {hidden.get('candidate_uid')}")
        uid = pub["candidate_uid"]
        mask = load_mask(Path(hidden["mask_path"]), he.size)
        bbox = mask_bbox(mask)
        crop_box = expanded_square_bbox(
            bbox,
            he.size,
            args.context_scale,
            args.context_min_side,
            args.context_max_side_fullres,
        )

        he_context_rel = f"context_images/{uid}_he_local_context.png"
        fic_context_rel = f"context_images/{uid}_ficture_local_context.png"
        locator_rel = f"context_images/{uid}_he_roi_locator.png"
        context_crop(he, mask, crop_box, args.output_max_side).save(args.output_dir / he_context_rel)
        context_crop(ficture, mask, crop_box, args.output_max_side).save(args.output_dir / fic_context_rel)
        roi_locator(he, bbox, args.locator_max_side).save(args.output_dir / locator_rel)

        copy_relative(args.source_pool, pub["he_crop_rel"], args.output_dir)
        copy_relative(args.source_pool, pub["ficture_crop_rel"], args.output_dir)

        values = {
            "image1_rel": pub["he_crop_rel"],
            "image2_rel": pub["ficture_crop_rel"],
            "image3_rel": he_context_rel,
            "image4_rel": fic_context_rel,
            "image5_rel": locator_rel,
            "image_sequence": (
                "1=H&E reverse-blur close-up; 2=FICTURE reverse-blur close-up; "
                "3=H&E local context with candidate outlined; "
                "4=FICTURE local context with candidate outlined; "
                "5=low-resolution H&E ROI locator."
            ),
            "context_bbox_xyxy": str(list(crop_box)),
        }
        pub.update(values)
        hidden.update(values)

    write_rows(args.output_dir / "public_vlm_requests.csv", public_rows, public_fields)
    write_rows(args.output_dir / "hidden_candidate_truth.csv", hidden_rows, hidden_fields)
    (args.output_dir / "README.txt").write_text(
        "Context-aware piece-first VLM pool.\n"
        "The candidate mask is unchanged from compact167; only extra context images were added.\n"
        f"rows={len(public_rows)}\n"
        "image1/image2 are original reverse-blur close-ups. image3/image4 add local context. "
        "image5 is a full-ROI locator.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(public_rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
