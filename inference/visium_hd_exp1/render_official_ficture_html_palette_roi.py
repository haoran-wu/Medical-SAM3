#!/usr/bin/env python3
"""Render the PASS_OFFICIAL FICTURE ROI using colors from pixel.info.html.

This does not redo registration. It reuses the already aligned factor-index ROI
and only replaces factor colors with the RGB values from the FICTURE HTML info
table. That keeps the coordinate system fixed while making the map color legend
match the source HTML exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
DEFAULT_HTML = PROJECT_ROOT / "data/visium_hd_exp1/pixel_cell_type_image/hex_12.k12.pixel.info.html"
DEFAULT_OFFICIAL_SUMMARY = (
    PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "html_palette"


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.current_cell: List[str] = []
        self.current_row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag = tag.lower()
        if tag in {"td", "th"} and self.in_cell:
            text = " ".join("".join(self.current_cell).split())
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def parse_rgb(text: str) -> tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) < 3:
        raise ValueError(f"Could not parse RGB value: {text!r}")
    r, g, b = nums[:3]
    if not all(0 <= value <= 255 for value in (r, g, b)):
        raise ValueError(f"RGB value out of range: {text!r}")
    return r, g, b


def load_html_factor_table(path: Path) -> List[dict]:
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    if not parser.rows:
        raise RuntimeError(f"No table rows parsed from {path}")
    header = parser.rows[0]
    rows: List[dict] = []
    for raw in parser.rows[1:]:
        if len(raw) != len(header):
            continue
        row = dict(zip(header, raw))
        if not row.get("Factor") or not row.get("RGB"):
            continue
        row["factor"] = int(row["Factor"])
        row["rgb_tuple"] = parse_rgb(row["RGB"])
        row["hex"] = "#{:02X}{:02X}{:02X}".format(*row["rgb_tuple"])
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No factor rows with RGB values parsed from {path}")
    return rows


def verify_official(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {path}")
    failed = [key for key, value in payload.get("status_checks", {}).items() if not value]
    if failed:
        raise SystemExit(f"Official FICTURE summary has failed checks: {failed}")
    return payload


def save_legend(rows: List[dict], output_dir: Path) -> None:
    fields = [
        "Factor",
        "Major Compartment",
        "Celltype",
        "RGB",
        "hex",
        "Weight",
        "PostUMI",
        "TopGene_pval",
        "TopGene_specific",
        "TopGene_fc",
        "TopGene_weight",
    ]
    with (output_dir / "html_palette_factor_legend.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["Factor"])):
            writer.writerow({field: row.get(field, "") for field in fields})
    (output_dir / "html_palette_factor_legend.json").write_text(
        json.dumps(sorted(rows, key=lambda item: int(item["Factor"])), indent=2),
        encoding="utf-8",
    )


def make_overlay(he: np.ndarray, ficture: np.ndarray, mask: np.ndarray, alpha: float = 0.50) -> Image.Image:
    out = he.astype(np.float32).copy()
    fic = ficture.astype(np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * fic[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_checker(he: np.ndarray, ficture: np.ndarray, mask: np.ndarray, tile: int = 192) -> Image.Image:
    yy, xx = np.indices(mask.shape)
    use_fic = ((xx // tile + yy // tile) % 2 == 0) & mask
    out = he.copy()
    out[use_fic] = ficture[use_fic]
    return Image.fromarray(out)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def thumbnail(img: Image.Image, width: int) -> Image.Image:
    w, h = img.size
    return img.resize((width, int(h * width / w)), Image.Resampling.LANCZOS)


def write_qc_panel(
    he: Image.Image,
    ficture: Image.Image,
    overlay: Image.Image,
    checker: Image.Image,
    output_path: Path,
    html_path: Path,
    factors: int,
) -> None:
    panels = [
        ("1. H&E ROI", he),
        ("2. FICTURE map recolored from HTML RGB", ficture),
        ("3. HTML-color FICTURE over H&E", overlay),
        ("4. Checkerboard: H&E / HTML-color FICTURE", checker),
    ]
    thumbs = [(title, thumbnail(img, 760)) for title, img in panels]
    title_font = font(30)
    body_font = font(22)
    small_font = font(19)
    pad = 24
    label_h = 42
    note_h = 155
    cols = 2
    cell_w = thumbs[0][1].size[0]
    cell_h = thumbs[0][1].size[1] + label_h
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, note_h + 2 * cell_h + 3 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), "FICTURE color check: RGB comes from hex_12.k12.pixel.info.html", fill=(20, 20, 20), font=title_font)
    draw.text((pad, 60), f"Same PASS_OFFICIAL ROI and same factor index; only the colors are replaced. Factors parsed: {factors}.", fill=(45, 45, 45), font=body_font)
    draw.text((pad, 92), f"HTML source: {html_path}", fill=(65, 65, 65), font=small_font)
    for idx, (label, img) in enumerate(thumbs):
        x = pad + (idx % cols) * (cell_w + pad)
        y = note_h + (idx // cols) * (cell_h + pad)
        draw.text((x, y), label, fill=(20, 20, 20), font=body_font)
        canvas.paste(img, (x, y + label_h))
        draw.rectangle((x, y + label_h, x + img.size[0] - 1, y + label_h + img.size[1] - 1), outline=(180, 180, 180))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--info-html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--official-summary", type=Path, default=DEFAULT_OFFICIAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    verify_official(args.official_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    he_path = args.input_dir / "he_roi_matching_official_ficture_coverage.png"
    index_path = args.input_dir / "ficture_official_filtered_roi_factor_index.npy"
    he = Image.open(he_path).convert("RGB")
    factor_index = np.load(index_path)
    if he.size != (3144, 3327) or factor_index.shape != (3327, 3144):
        raise SystemExit(f"Unexpected ROI geometry: he={he.size}, factor_index={factor_index.shape}")

    rows = load_html_factor_table(args.info_html)
    palette: Dict[int, tuple[int, int, int]] = {row["factor"]: row["rgb_tuple"] for row in rows}
    missing = sorted(int(x) for x in np.unique(factor_index) if x >= 0 and int(x) not in palette)
    if missing:
        raise SystemExit(f"Factor ids missing from HTML palette: {missing}")

    ficture = np.zeros((*factor_index.shape, 3), dtype=np.uint8)
    mask = factor_index >= 0
    for factor_id, rgb in palette.items():
        ficture[factor_index == factor_id] = rgb
    ficture_img = Image.fromarray(ficture)
    he_np = np.array(he)
    overlay = make_overlay(he_np, ficture, mask)
    checker = make_checker(he_np, ficture, mask)

    ficture_img.save(args.output_dir / "ficture_official_filtered_roi_rgb_from_info_html.png")
    overlay.save(args.output_dir / "ficture_official_filtered_overlay_roi_from_info_html.png")
    checker.save(args.output_dir / "ficture_official_filtered_checkerboard_roi_from_info_html.png")
    save_legend(rows, args.output_dir)
    write_qc_panel(
        he=he,
        ficture=ficture_img,
        overlay=overlay,
        checker=checker,
        output_path=args.output_dir / "ficture_html_palette_qc_panel.png",
        html_path=args.info_html,
        factors=len(rows),
    )

    summary = {
        "status": "PASS_OFFICIAL_HTML_PALETTE_RENDERED",
        "meaning": "Same PASS_OFFICIAL FICTURE ROI and factor index; colors are replaced with RGB values from the HTML factor table.",
        "info_html": str(args.info_html),
        "official_summary": str(args.official_summary),
        "he_roi": str(he_path),
        "factor_index": str(index_path),
        "roi_size_wh": list(he.size),
        "factor_index_shape_hw": list(factor_index.shape),
        "factors_in_html": len(rows),
        "outputs": {
            "ficture_rgb": str(args.output_dir / "ficture_official_filtered_roi_rgb_from_info_html.png"),
            "overlay_on_he": str(args.output_dir / "ficture_official_filtered_overlay_roi_from_info_html.png"),
            "checkerboard": str(args.output_dir / "ficture_official_filtered_checkerboard_roi_from_info_html.png"),
            "qc_panel": str(args.output_dir / "ficture_html_palette_qc_panel.png"),
            "legend_csv": str(args.output_dir / "html_palette_factor_legend.csv"),
            "legend_json": str(args.output_dir / "html_palette_factor_legend.json"),
        },
    }
    (args.output_dir / "summary_html_palette.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output_dir / "ficture_html_palette_qc_panel.png")


if __name__ == "__main__":
    main()
