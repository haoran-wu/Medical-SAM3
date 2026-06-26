#!/usr/bin/env python3
"""Build high-resolution Xenium Silica H&E crops from the original H&E TIFF zip.

The older manual-alignment reports cropped H&E from the morphology-level
``*_wholeslide_rotated_he_grid.png`` images.  Those are correct for registration
debugging, but they are 7-8x downsampled from the original H&E TIFFs.  This
script keeps the same manually accepted TMA boxes, maps them back to the
original H&E pixel grid, and writes full-resolution oriented H&E crops.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import xenium_wholeslide_tif_he_translation_registration as ws_align


DEFAULT_MANUAL_MANIFEST = (
    ROOT
    / "output/aaai_xenium_silica_20260623/manual_accepted_alignment_annotation_tmas_20260624/"
    / "manual_accepted_alignment_manifest.json"
)
DEFAULT_REG_ROOT = ROOT / "output/xenium_silica_wholeslide_tif_he_translation_registration_20260624"
DEFAULT_OUT = ROOT / "output/aaai_xenium_silica_20260623/highres_he_crops_20260626"


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def grid_bbox_to_raw_bbox(
    bbox: list[int],
    *,
    slide: ws_align.base.SlideConfig,
    grid_size_wh: tuple[int, int],
) -> tuple[list[int], list[float]]:
    """Map an oriented low-res H&E-grid crop bbox back to raw H&E coordinates."""

    x0, y0, x1, y1 = [float(v) for v in bbox]
    grid_w, grid_h = grid_size_wh
    he_h, he_w = slide.he_shape[:2]

    oriented_raw_w = he_h
    oriented_raw_h = he_w
    sx = oriented_raw_w / float(grid_w)
    sy = oriented_raw_h / float(grid_h)

    ox0 = x0 * sx
    ox1 = x1 * sx
    oy0 = y0 * sy
    oy1 = y1 * sy

    # Inverse of rot90_ccw_lr_mirror:
    # raw bbox -> oriented bbox:
    # [he_h - y1, he_w - x1, he_h - y0, he_w - x0]
    raw_x0 = he_w - oy1
    raw_x1 = he_w - oy0
    raw_y0 = he_h - ox1
    raw_y1 = he_h - ox0

    raw_bbox = [
        max(0, int(math.floor(raw_x0))),
        max(0, int(math.floor(raw_y0))),
        min(he_w, int(math.ceil(raw_x1))),
        min(he_h, int(math.ceil(raw_y1))),
    ]
    oriented_raw_bbox = [ox0, oy0, ox1, oy1]
    return raw_bbox, oriented_raw_bbox


def crop_oriented_highres(slide: ws_align.base.SlideConfig, raw_bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = raw_bbox
    arr = ws_align.base.he_memmap(slide)
    raw_crop = np.asarray(arr[y0:y1, x0:x1, :]).copy()
    oriented = ws_align.roi_align.orient_he_crop(raw_crop, ws_align.ORIENTATION)
    return Image.fromarray(oriented)


def make_contact_sheet(records: list[dict], out_path: Path) -> None:
    tile_w, tile_h = 430, 360
    header_h = 42
    cols = 2
    rows = math.ceil(len(records) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)
    small_font = font(12)
    for idx, rec in enumerate(records):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), fill="white", outline="#cbd5e1")
        draw.text((x + 12, y + 9), f"{rec['tma']} {rec['slide']} high-res H&E", fill="#0f172a", font=title_font)
        draw.text(
            (x + 12, y + 29),
            f"{rec['highres_size_px']} from raw {rec['raw_he_member_px_unrotated']}",
            fill="#475569",
            font=small_font,
        )
        im = Image.open(rec["highres_he_path"]).convert("RGB")
        im.thumbnail((tile_w - 24, tile_h - header_h - 16), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + (tile_w - im.width) // 2, y + header_h + 8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    out = DEFAULT_OUT
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(DEFAULT_MANUAL_MANIFEST.read_text(encoding="utf-8"))
    zip_path = ws_align.base.HE_ZIP
    zip_members: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for slide in ws_align.base.SLIDES.values():
            zip_members[slide.zip_member] = zf.getinfo(slide.zip_member).file_size

    records: list[dict] = []
    for rec in manifest["records"]:
        tma = int(rec["tma"])
        slide_id = rec["slide"]
        slide = ws_align.base.SLIDES[slide_id]
        grid_path = DEFAULT_REG_ROOT / f"{slide_id}_wholeslide_rotated_he_grid.png"
        grid_w, grid_h = Image.open(grid_path).size
        bbox = rec["manual_alignment"]["crop_bbox_xyxy_in_oriented_he_grid"]
        raw_bbox, oriented_raw_bbox = grid_bbox_to_raw_bbox(bbox, slide=slide, grid_size_wh=(grid_w, grid_h))

        highres = crop_oriented_highres(slide, raw_bbox)
        highres_path = image_dir / f"TMA{tma:02d}_{slide_id}_he_highres.png"
        highres.save(highres_path)

        old_he = (
            ROOT
            / "output/aaai_xenium_silica_20260623/manual_accepted_alignment_annotation_tmas_20260624"
            / rec["outputs"]["he"]
        )
        old_w, old_h = Image.open(old_he).size
        row = {
            "tma": f"TMA{tma:02d}",
            "slide": slide_id,
            "old_lowres_he_path": str(old_he),
            "old_lowres_size_px": f"{old_w}x{old_h}",
            "highres_he_path": str(highres_path),
            "highres_size_px": f"{highres.width}x{highres.height}",
            "crop_bbox_in_rotated_he_grid_xyxy": json.dumps(bbox),
            "oriented_raw_bbox_xyxy": json.dumps([round(v, 2) for v in oriented_raw_bbox]),
            "raw_he_bbox_unrotated_xyxy": json.dumps(raw_bbox),
            "raw_he_zip": str(zip_path),
            "raw_he_member": slide.zip_member,
            "raw_he_member_px_unrotated": f"{slide.he_shape[1]}x{slide.he_shape[0]}",
            "raw_he_member_bytes": zip_members[slide.zip_member],
            "grid_source_path": str(grid_path),
            "grid_source_size_px": f"{grid_w}x{grid_h}",
            "grid_to_raw_scale_x": round(slide.he_shape[0] / float(grid_w), 6),
            "grid_to_raw_scale_y": round(slide.he_shape[1] / float(grid_h), 6),
        }
        records.append(row)

    with (out / "highres_he_crop_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "highres_he_crop_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    make_contact_sheet(records, out / "highres_he_contact_sheet.png")
    print(f"Wrote {len(records)} high-res H&E crops to {out}")
    for rec in records:
        print(rec["tma"], rec["slide"], rec["old_lowres_size_px"], "->", rec["highres_size_px"])


if __name__ == "__main__":
    main()
