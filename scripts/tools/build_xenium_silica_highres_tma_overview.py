#!/usr/bin/env python3
"""Build a high-resolution TMA overview for Xenium Silica.

Each row shows:

1. High-resolution H&E crop from the original H&E TIFF.
2. K=12 molecule-level FICTURE/punkst map using the official cmap48 first colors.
3. Continuous cell-boundary annotation overlay on the high-resolution H&E crop.

This report is for visual QC.  The annotation overlay is evaluation ground
truth only; it is not an input to FICTURE/SAM/VLM.
"""

from __future__ import annotations

import csv
import html
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import xenium_wholeslide_tif_he_translation_registration as ws_align

AAAI_ROOT = ROOT / "output/aaai_xenium_silica_20260623"
HIGHRES_ROOT = AAAI_ROOT / "highres_he_crops_20260626"
FICTURE_ROOT = (
    AAAI_ROOT
    / "xenium_true_molecule_ficture_sync_20260626"
    / "xenium_silica_true_molecule_ficture_slim_20260626"
    / "ksweep"
)
GT_ROOT = AAAI_ROOT / "continuous_gt_masks_cellboundary_20260626"
OLD_OVERVIEW_ROOT = AAAI_ROOT / "tma_sample_overview_20260626"
OUT_ROOT = AAAI_ROOT / "tma_sample_overview_highres_20260626"


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


def rel(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        return list(csv.DictReader(handle, delimiter=delimiter))


def first_genes(text: str, n: int = 5) -> str:
    genes = [g.strip() for g in str(text).split(",") if g.strip()]
    return ", ".join(genes[:n])


def load_factor_legend(sample_dir: Path) -> list[dict[str, str]]:
    color_rows = read_csv_rows(sample_dir / "h12.k12.color.rgb.tsv")[:12]
    info_path = sample_dir / "source_matched_factor_info.csv"
    info_by_factor: dict[int, dict[str, str]] = {}
    if info_path.exists():
        for row in read_csv_rows(info_path):
            try:
                info_by_factor[int(row["Factor"])] = row
            except Exception:
                continue
    legend: list[dict[str, str]] = []
    for idx, color in enumerate(color_rows):
        info = info_by_factor.get(idx, {})
        legend.append(
            {
                "factor": str(idx),
                "hex": color.get("Color_hex") or f"#{int(color['R']):02x}{int(color['G']):02x}{int(color['B']):02x}",
                "celltype": info.get("Celltype2", ""),
                "compartment": info.get("Major Compartment", ""),
                "top_genes": first_genes(info.get("TopGene_specific") or info.get("TopGene_pval", "")),
            }
        )
    return legend


def annotation_counts_for_tma(tma_dir: Path) -> dict[str, int]:
    manifest = json.loads((tma_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("fill_counts") or manifest.get("annotation_counts") or {}
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items()}
    return {}


def make_original_slide_previews(out_root: Path) -> list[dict[str, str]]:
    """Create lightweight previews directly from the two original H&E TIFFs."""

    out_dir = out_root / "original_slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_sizes: dict[str, int] = {}
    with zipfile.ZipFile(ws_align.base.HE_ZIP) as zf:
        for slide in ws_align.base.SLIDES.values():
            zip_sizes[slide.zip_member] = int(zf.getinfo(slide.zip_member).file_size)

    records: list[dict[str, str]] = []
    for slide_id in ["770-1", "771-1"]:
        slide = ws_align.base.SLIDES[slide_id]
        arr = ws_align.base.he_memmap(slide)
        stride = 40
        thumb = np.asarray(arr[::stride, ::stride, :]).copy()
        im = Image.fromarray(thumb)
        stem = Path(slide.zip_member).stem
        out_path = out_dir / f"{slide_id}_{stem}_original_he_thumbnail.jpg"
        im.save(out_path, quality=92)
        records.append(
            {
                "slide": slide_id,
                "raw_member": slide.zip_member,
                "raw_size_px": f"{slide.he_shape[1]}x{slide.he_shape[0]}",
                "raw_member_bytes": str(zip_sizes.get(slide.zip_member, "")),
                "preview_size_px": f"{im.width}x{im.height}",
                "sampling_stride_px": str(stride),
                "preview_abs": str(out_path),
                "preview_rel": rel(out_path, out_root),
            }
        )
    (out_root / "source_slides_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (out_root / "source_slides_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "slide",
            "raw_member",
            "raw_size_px",
            "raw_member_bytes",
            "preview_size_px",
            "sampling_stride_px",
            "preview_abs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec[field] for field in fields})
    return records


def alpha_blend_annotation_on_he(he: Image.Image, lowres_mask_path: Path) -> tuple[Image.Image, Image.Image]:
    """Resize a low-res mask to high-res and blend colored non-white pixels."""

    he_rgba = he.convert("RGBA")
    mask = Image.open(lowres_mask_path).convert("RGB").resize(he.size, Image.Resampling.NEAREST)
    white = Image.new("RGB", he.size, "white")
    nonwhite = ImageChops.difference(mask, white).convert("L").point(lambda v: 0 if v < 8 else 1)

    color_rgba = mask.convert("RGBA")
    alpha = nonwhite.point(lambda v: 115 if v else 0)
    color_rgba.putalpha(alpha)
    overlay = Image.alpha_composite(he_rgba, color_rgba).convert("RGB")
    return overlay, mask


def make_panel_preview(he: Image.Image, fic: Image.Image, ann: Image.Image, out_path: Path) -> None:
    tile_w, tile_h = 460, 430
    header_h = 34
    canvas = Image.new("RGB", (tile_w * 3, tile_h), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)
    for idx, (title, image) in enumerate(
        [
            ("High-res H&E", he),
            ("K=12 molecule FICTURE", fic),
            ("Cell-boundary annotation", ann),
        ]
    ):
        x = idx * tile_w
        draw.rectangle((x, 0, x + tile_w - 1, tile_h - 1), fill="white", outline="#cbd5e1")
        draw.text((x + 12, 8), title, fill="#0f172a", font=title_font)
        thumb = image.copy()
        thumb.thumbnail((tile_w - 24, tile_h - header_h - 12), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (tile_w - thumb.width) // 2, header_h + 6))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_contact_sheet(records: list[dict], out_path: Path) -> None:
    tile_w, tile_h = 520, 410
    cols = 2
    rows = (len(records) + cols - 1) // cols
    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), "#eef2f7")
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)
    small_font = font(12)
    for idx, rec in enumerate(records):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), fill="white", outline="#cbd5e1")
        draw.text((x + 12, y + 8), f"{rec['tma']} {rec['slide']} high-res overview", fill="#0f172a", font=title_font)
        draw.text((x + 12, y + 29), f"H&E {rec['he_size']} | FICTURE {rec['ficture_native_size']}", fill="#475569", font=small_font)
        img = Image.open(rec["preview_panel_abs"]).convert("RGB")
        img.thumbnail((tile_w - 24, tile_h - 54), Image.Resampling.LANCZOS)
        canvas.paste(img, (x + (tile_w - img.width) // 2, y + 48))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def factor_legend_html(rows: list[dict[str, str]]) -> str:
    out = ["<table class='legend'><thead><tr><th>Factor</th><th>Color</th><th>Cell type</th><th>Top genes</th></tr></thead><tbody>"]
    for row in rows:
        out.append(
            "<tr>"
            f"<td>F{html.escape(row['factor'])}</td>"
            f"<td><span class='swatch' style='background:{html.escape(row['hex'])}'></span>{html.escape(row['hex'])}</td>"
            f"<td>{html.escape(row.get('celltype') or row.get('compartment') or 'unlabeled')}</td>"
            f"<td>{html.escape(row.get('top_genes') or '')}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def annotation_legend_html(counts: dict[str, int]) -> str:
    if not counts:
        return "<p class='muted'>No annotation counts found.</p>"
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(counts.values())
    return "<div class='ann-counts'>" + "".join(
        f"<span><b>{html.escape(label)}</b>: {count:,} cells ({count / total:.1%})</span>" for label, count in rows
    ) + "</div>"


def source_slide_html(records: list[dict[str, str]]) -> str:
    parts = [
        "<section class='source-section'>",
        "<h2>Original Whole-Slide H&amp;E Sources</h2>",
        "<p class='meta'>These two previews are sampled directly from the full-resolution H&amp;E TIFF members inside <code>OneDrive_1_4-24-2026.zip</code>. They are source previews for QC; the TMA panels below use full-resolution crops from these raw TIFFs.</p>",
        "<div class='source-grid'>",
    ]
    for rec in records:
        raw_gb = ""
        try:
            raw_gb = f" | {int(rec['raw_member_bytes']) / (1024 ** 3):.2f} GiB"
        except Exception:
            pass
        parts.append(
            "<figure>"
            f"<figcaption>Slide {html.escape(rec['slide'])}: {html.escape(rec['raw_member'])}<br>"
            f"<span class='caption-sub'>raw {html.escape(rec['raw_size_px'])}{html.escape(raw_gb)}; preview sampled every {html.escape(rec['sampling_stride_px'])} px</span></figcaption>"
            f"<a href='{html.escape(rec['preview_rel'])}'><img class='panel' src='{html.escape(rec['preview_rel'])}'></a>"
            "</figure>"
        )
    parts.append("</div></section>")
    return "\n".join(parts)


def main() -> None:
    image_dir = OUT_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    source_slide_records = make_original_slide_previews(OUT_ROOT)
    highres_rows = read_csv_rows(HIGHRES_ROOT / "highres_he_crop_manifest.csv")
    records: list[dict] = []
    for row in highres_rows:
        tma = row["tma"]
        tma_num = int(tma.replace("TMA", ""))
        slide = row["slide"]
        sample = f"{tma}_{slide}"
        he_path = Path(row["highres_he_path"])
        he = Image.open(he_path).convert("RGB")

        sample_dir = FICTURE_ROOT / sample / "hex_12_k12"
        fic_native_path = sample_dir / "h12.k12.cmap48.pixel.png"
        if not fic_native_path.exists():
            fic_native_path = sample_dir / "h12.k12.pixel.png"
        fic_native = Image.open(fic_native_path).convert("RGB")
        fic_high = fic_native.resize(he.size, Image.Resampling.NEAREST)
        fic_high_path = image_dir / f"{tma}_ficture_k12_cmap48_highres_display.png"
        fic_high.save(fic_high_path)

        gt_dir = GT_ROOT / f"{tma}_{slide}"
        mask_low_path = gt_dir / "images" / f"{tma}_cell_boundary_fill_mask_only.png"
        ann_overlay, ann_mask = alpha_blend_annotation_on_he(he, mask_low_path)
        ann_overlay_path = image_dir / f"{tma}_annotation_cellboundary_highres_overlay.png"
        ann_mask_path = image_dir / f"{tma}_annotation_cellboundary_highres_mask_only.png"
        ann_overlay.save(ann_overlay_path)
        ann_mask.save(ann_mask_path)

        preview_path = image_dir / f"{tma}_highres_three_panel_preview.jpg"
        make_panel_preview(he, fic_high, ann_overlay, preview_path)

        factor_legend = load_factor_legend(sample_dir)
        annotation_counts = annotation_counts_for_tma(gt_dir)
        rec = {
            "tma": tma,
            "tma_num": tma_num,
            "slide": slide,
            "status": "",
            "he_abs": str(he_path),
            "he_rel": rel(he_path, OUT_ROOT),
            "he_size": f"{he.width}x{he.height}",
            "ficture_native_abs": str(fic_native_path),
            "ficture_native_rel": rel(fic_native_path, OUT_ROOT),
            "ficture_native_size": f"{fic_native.width}x{fic_native.height}",
            "ficture_high_abs": str(fic_high_path),
            "ficture_high_rel": rel(fic_high_path, OUT_ROOT),
            "annotation_overlay_abs": str(ann_overlay_path),
            "annotation_overlay_rel": rel(ann_overlay_path, OUT_ROOT),
            "annotation_mask_abs": str(ann_mask_path),
            "annotation_mask_rel": rel(ann_mask_path, OUT_ROOT),
            "preview_panel_abs": str(preview_path),
            "preview_panel_rel": rel(preview_path, OUT_ROOT),
            "factor_legend": factor_legend,
            "annotation_counts": annotation_counts,
            "raw_he_member": row["raw_he_member"],
            "raw_he_bbox_unrotated_xyxy": row["raw_he_bbox_unrotated_xyxy"],
            "old_lowres_size_px": row["old_lowres_size_px"],
            "old_lowres_he_path": row["old_lowres_he_path"],
        }
        records.append(rec)

    records.sort(key=lambda r: r["tma_num"])
    make_contact_sheet(records, OUT_ROOT / "highres_tma_overview_contact_sheet.jpg")

    manifest_rows = []
    for rec in records:
        slim = {k: v for k, v in rec.items() if k not in {"factor_legend", "annotation_counts"}}
        slim["annotation_counts"] = rec["annotation_counts"]
        manifest_rows.append(slim)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    with (OUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "tma",
            "slide",
            "he_size",
            "old_lowres_size_px",
            "ficture_native_size",
            "he_abs",
            "ficture_high_abs",
            "annotation_overlay_abs",
            "raw_he_member",
            "raw_he_bbox_unrotated_xyxy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec[field] for field in fields})

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Xenium Silica High-Resolution TMA Overview</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#f1f5f9;color:#111827}",
        "h1{font-size:24px;margin:0 0 8px} h2{font-size:20px;margin:28px 0 8px}",
        ".note{background:white;border:1px solid #d7dee8;padding:14px 16px;margin:14px 0 22px;line-height:1.45}",
        ".source-section{background:white;border:1px solid #cbd5e1;margin:12px 0 22px;padding:14px}",
        ".source-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}",
        ".row{background:white;border:1px solid #cbd5e1;margin:22px 0;padding:14px}",
        ".grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;align-items:start}",
        "figure{margin:0;border:1px solid #d7dee8;background:#fff}",
        "figcaption{font-size:13px;padding:8px 10px;border-bottom:1px solid #e5e7eb;font-weight:700;color:#334155}",
        ".caption-sub{font-weight:400;color:#64748b}",
        "img.panel{width:100%;height:auto;display:block}",
        ".meta{font-size:13px;color:#475569;line-height:1.45;margin:8px 0 12px}",
        ".legend-wrap{display:grid;grid-template-columns:1.5fr 1fr;gap:14px;margin-top:12px}",
        "table.legend{border-collapse:collapse;width:100%;font-size:12px}",
        "table.legend th,table.legend td{border:1px solid #e5e7eb;padding:5px 6px;text-align:left;vertical-align:top}",
        ".swatch{display:inline-block;width:18px;height:18px;border:1px solid #475569;margin-right:6px;vertical-align:middle}",
        ".ann-counts{font-size:12px;display:flex;flex-wrap:wrap;gap:7px}",
        ".ann-counts span{border:1px solid #d1d5db;border-radius:4px;background:#f8fafc;padding:5px 7px}",
        ".muted{color:#64748b}",
        "a{color:#075985;text-decoration:none} a:hover{text-decoration:underline}",
        "</style></head><body>",
        "<h1>Xenium Silica High-Resolution TMA Overview</h1>",
        source_slide_html(source_slide_records),
        "<div class='note'><b>Experiment design.</b> Goal: inspect each TMA sample using original-resolution H&E crops. Model inputs shown here are high-resolution H&E and K=12 molecule-level FICTURE/punkst maps. The right panel is a filled cell-boundary annotation overlay used only as evaluation ground truth, not as model input. FICTURE means a false-color molecular feature map built from Xenium transcript locations and gene identities; it is not a cell-count or cell-centroid proxy.</div>",
        f"<p><a href='{html.escape(rel(OUT_ROOT / 'highres_tma_overview_contact_sheet.jpg', OUT_ROOT))}'>Contact sheet</a> | <a href='manifest.csv'>Manifest CSV</a></p>",
    ]

    for rec in records:
        html_parts.extend(
            [
                f"<section class='row'><h2>{html.escape(rec['tma'])} | slide {html.escape(rec['slide'])}</h2>",
                "<div class='meta'>"
                f"H&E now uses original TIFF crop <code>{html.escape(rec['he_size'])}</code>; previous low-res H&E was <code>{html.escape(rec['old_lowres_size_px'])}</code>. "
                f"Raw member: <code>{html.escape(rec['raw_he_member'])}</code>. "
                f"Raw bbox: <code>{html.escape(rec['raw_he_bbox_unrotated_xyxy'])}</code>. "
                f"FICTURE native map: <code>{html.escape(rec['ficture_native_size'])}</code>; display-fit image is resized to match the high-res H&E canvas."
                "</div>",
                "<div class='grid'>",
                f"<figure><figcaption>High-resolution H&amp;E ROI</figcaption><a href='{html.escape(rec['he_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['he_rel'])}'></a></figure>",
                f"<figure><figcaption>K=12 molecule-level FICTURE, cmap48</figcaption><a href='{html.escape(rec['ficture_high_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['ficture_high_rel'])}'></a></figure>",
                f"<figure><figcaption>Cell-boundary annotation overlay on high-res H&amp;E</figcaption><a href='{html.escape(rec['annotation_overlay_rel'])}'><img class='panel' loading='lazy' src='{html.escape(rec['annotation_overlay_rel'])}'></a></figure>",
                "</div>",
                "<div class='legend-wrap'>",
                "<div><h3>FICTURE factor legend</h3>",
                factor_legend_html(rec["factor_legend"]),
                "</div>",
                "<div><h3>Annotation labels</h3>",
                annotation_legend_html(rec["annotation_counts"]),
                f"<p class='muted'>Mask-only file: <a href='{html.escape(rec['annotation_mask_rel'])}'>open</a>. Preview panel: <a href='{html.escape(rec['preview_panel_rel'])}'>open</a>.</p>",
                "</div></div></section>",
            ]
        )

    html_parts.append("</body></html>")
    (OUT_ROOT / "index.html").write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Wrote {len(records)} high-res TMA rows to {OUT_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
