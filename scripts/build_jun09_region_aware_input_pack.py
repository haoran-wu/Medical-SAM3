#!/usr/bin/env python3
"""Build a region-aware input pack for Jun09 piece scoring.

Each candidate gets three visual scales:

1. existing piece reverse-blur crop,
2. larger local context crop with the candidate contour,
3. full ROI locator view with the candidate box.

The masks are not changed.  This is only an input-design ablation to test
whether tissue recognition failed because isolated pieces lacked context.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
DEFAULT_POOL = BASE / "corrected_pool"
DEFAULT_OUT = BASE / "region_aware_piece_context_locator_pack"
HE_ROI = (
    ROOT
    / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
    "he_roi_matching_official_ficture_coverage.png"
)
FICTURE_ROI = (
    ROOT
    / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
    "ficture_official_filtered_roi_rgb.png"
)
LEGEND_CSV = ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_prompt_legend_from_html.csv"

CLASSES = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
CLASS_TEXT = {
    "bronchiola": "bronchiolar airway tissue, airway-like lumen, epithelial lining.",
    "alveoli": "alveolar lung parenchyma, open air spaces, thin septa.",
    "vessels": "blood vessel or vascular wall, lumen-like vascular structure, smooth muscle vessel wall.",
    "tumor": "tumor region.",
    "stroma": "stromal / mesenchymal tissue, collagen, fibroblast, smooth-muscle-like tissue.",
    "immune_infiltration": "immune-cell-rich region, small round-cell aggregates, macrophage / lymphoid / plasma-cell rich tissue.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    scale: float,
    min_side: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    bw = max(x1 - x0, min_side)
    bh = max(y1 - y0, min_side)
    side = max(bw, bh)
    new_w = min(width, int(round(side * scale)))
    new_h = min(height, int(round(side * scale)))
    nx0 = max(0, int(round(cx - new_w / 2)))
    ny0 = max(0, int(round(cy - new_h / 2)))
    nx1 = min(width, nx0 + new_w)
    ny1 = min(height, ny0 + new_h)
    nx0 = max(0, nx1 - new_w)
    ny0 = max(0, ny1 - new_h)
    return nx0, ny0, nx1, ny1


def draw_contour(image: Image.Image, mask: np.ndarray, title: str, color=(0, 96, 255)) -> Image.Image:
    base = image.convert("RGBA")
    m = mask.astype(np.uint8)
    edge = m.astype(bool) ^ (
        np.roll(m, 1, axis=0).astype(bool)
        & np.roll(m, -1, axis=0).astype(bool)
        & np.roll(m, 1, axis=1).astype(bool)
        & np.roll(m, -1, axis=1).astype(bool)
    )
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay[edge] = (*color, 255)
    overlay[mask] = (*color, 38)
    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 980), 30), fill=(255, 255, 255, 230))
    draw.text((8, 9), title, fill=(0, 0, 0, 255), font=ImageFont.load_default())
    return out.convert("RGB")


def make_local_context(
    roi: Image.Image,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    title: str,
    out_path: Path,
    max_side: int,
) -> None:
    crop = roi.crop(box)
    crop_mask = mask[box[1] : box[3], box[0] : box[2]]
    out = draw_contour(crop, crop_mask, title)
    scale = min(1.0, max_side / max(out.size))
    if scale < 1.0:
        out = out.resize((int(out.width * scale), int(out.height * scale)), Image.Resampling.BICUBIC)
    out.save(out_path, quality=90)


def make_locator(
    roi: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    out_path: Path,
    max_side: int,
) -> None:
    out = roi.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    for offset in range(0, 10):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=(255, 0, 0), width=2)
    draw.rectangle((0, 0, min(out.width, 980), 34), fill=(255, 255, 255))
    draw.text((10, 10), title, fill=(0, 0, 0), font=ImageFont.load_default())
    scale = min(1.0, max_side / max(out.size))
    if scale < 1.0:
        out = out.resize((int(out.width * scale), int(out.height * scale)), Image.Resampling.BICUBIC)
    out.save(out_path, quality=90)


def legend_text(max_rows: int = 12) -> str:
    if not LEGEND_CSV.exists():
        return ""
    rows = read_csv(LEGEND_CSV)[:max_rows]
    lines: list[str] = []
    for row in rows:
        rgb = row.get("rgb") or row.get("RGB") or row.get("color_rgb") or ""
        comp = row.get("major_compartment") or row.get("Major Compartment") or row.get("compartment") or ""
        cell = row.get("cell_type") or row.get("cell type") or row.get("Cell type") or row.get("celltype") or ""
        idx = row.get("factor") or row.get("Factor") or row.get("color") or row.get("factor_id") or ""
        lines.append(f"Color {idx}: RGB {rgb}; compartment: {comp}; cell type: {cell}.")
    return "\n".join(lines)


def prompt_text(include_ficture: bool = True) -> str:
    class_lines = "\n".join(f"{c} = {CLASS_TEXT[c]}" for c in CLASSES)
    images = [
        "1. H&E piece crop: the candidate piece is sharp; outside is gray reverse-blur.",
    ]
    if include_ficture:
        images.extend(
            [
                "2. FICTURE piece crop: official FICTURE cell-type colors for the same candidate piece.",
                "3. H&E local context: a larger neighborhood; the blue contour/fill marks the same candidate piece.",
                "4. FICTURE local context: larger neighborhood with the same candidate contour.",
                "5. H&E full ROI locator: the red box shows where this candidate is located.",
                "6. FICTURE full ROI locator: red box shows candidate location.",
            ]
        )
    else:
        images.extend(
            [
                "2. H&E local context: a larger neighborhood; the blue contour/fill marks the same candidate piece.",
                "3. H&E full ROI locator: the red box shows where this candidate is located.",
            ]
        )
    legend = legend_text() if include_ficture else ""
    legend_block = f"\nFICTURE color legend:\n{legend}\n" if legend else ""
    return f"""You are given aligned views of one candidate region in a human lung cancer tissue section.
The target to score is always the marked candidate piece, not the surrounding tissue.

Images:
{chr(10).join(images)}
{legend_block}
Tissue classes:
{class_lines}

Score how likely the marked candidate piece belongs to each tissue class.

Rules:
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
- Each value must be an integer from 0 to 100.
- Higher means more likely for the marked candidate piece.
- Use local context to understand lumen, vessel wall, alveolar spaces, tumor nests, stroma, or immune aggregates.
- Return only one JSON object.
"""


def html_index(out_dir: Path, rows: list[dict[str, object]]) -> None:
    by_class: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_class.setdefault(str(row["true_class"]), []).append(row)
    sections: list[str] = []
    for cls in ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]:
        cards = []
        for row in by_class.get(cls, []):
            imgs = [
                row["he_piece_rel"],
                row["ficture_piece_rel"],
                row["he_local_context_rel"],
                row["ficture_local_context_rel"],
                row["he_full_locator_rel"],
                row["ficture_full_locator_rel"],
            ]
            img_html = "".join(f"<img src='{html.escape(str(src))}' alt='{html.escape(str(src))}'>" for src in imgs)
            cards.append(
                "<article class='card'>"
                f"<h3>{html.escape(str(row['candidate_uid']))}</h3>"
                f"<p>true={html.escape(cls)} | component={html.escape(str(row.get('matched_annotation_component_id','')))} | "
                f"D/P/R={html.escape(str(row.get('component_dice','')))} / {html.escape(str(row.get('component_precision','')))} / {html.escape(str(row.get('component_recall','')))}</p>"
                f"<p class='small'>bbox={html.escape(str(row['bbox_xyxy']))} | area={html.escape(str(row.get('piece_area','')))}</p>"
                f"<div class='imgs'>{img_html}</div>"
                "</article>"
            )
        sections.append(f"<h2>{html.escape(cls)}</h2><div class='grid'>{''.join(cards)}</div>")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jun09 region-aware piece input pack</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 34px; color: #1f2937; }}
p {{ line-height: 1.5; max-width: 1100px; }}
.callout {{ background:#eef6ff; border-left:4px solid #2563eb; padding:12px 16px; margin:16px 0 24px; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:20px; }}
.card {{ border:1px solid #e5e7eb; border-radius:8px; padding:16px; }}
.imgs {{ display:grid; grid-template-columns:repeat(3, minmax(220px, 1fr)); gap:12px; align-items:start; }}
img {{ width:100%; border:1px solid #ddd; background:white; }}
.small {{ color:#6b7280; font-size:13px; }}
pre {{ background:#f7f7f8; padding:14px; white-space:pre-wrap; }}
</style>
</head>
<body>
<h1>Jun09 Region-Aware Piece Scoring Input Pack</h1>
<div class="callout">Purpose: test whether VLM/CLIP failed because each candidate piece was shown too isolated. The marked mask is unchanged; only the visual context changes.</div>
<p>Each candidate has six views: H&E piece crop, FICTURE piece crop, H&E local context, FICTURE local context, H&E full ROI locator, and FICTURE full ROI locator. Hidden annotation labels are shown only for audit and are not part of the model prompt.</p>
<h2>Prompt template</h2>
<pre>{html.escape(prompt_text(include_ficture=True))}</pre>
{''.join(sections)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-scale", type=float, default=3.0)
    parser.add_argument("--local-min-side", type=int, default=640)
    parser.add_argument("--view-max-side", type=int, default=900)
    parser.add_argument("--locator-max-side", type=int, default=900)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    view_dir = args.out_dir / "views"
    view_dir.mkdir(exist_ok=True)

    public = read_csv(args.pool_dir / "public_vlm_requests.csv")
    hidden = read_csv(args.pool_dir / "hidden_candidate_truth.csv")
    hidden_by_uid = {row["candidate_uid"]: row for row in hidden}

    he = Image.open(HE_ROI).convert("RGB")
    fic = Image.open(FICTURE_ROI).convert("RGB")
    if he.size != fic.size:
        raise ValueError(f"ROI size mismatch: {he.size} vs {fic.size}")

    rows: list[dict[str, object]] = []
    for row in public:
        uid = row["candidate_uid"]
        truth = hidden_by_uid.get(uid, {})
        mask_path = Path(row["mask_path"])
        mask = load_mask(mask_path, he.size)
        tight = bbox(mask)
        local_box = expand_bbox(tight, he.width, he.height, args.local_scale, args.local_min_side)

        prefix = uid.replace("/", "_")
        he_local = view_dir / f"{prefix}_he_local_context.jpg"
        fic_local = view_dir / f"{prefix}_ficture_local_context.jpg"
        he_locator = view_dir / f"{prefix}_he_full_locator.jpg"
        fic_locator = view_dir / f"{prefix}_ficture_full_locator.jpg"
        he_piece = view_dir / f"{prefix}_he_piece_reverse_blur.png"
        fic_piece = view_dir / f"{prefix}_ficture_piece_reverse_blur.png"
        shutil.copy2(args.pool_dir / row["he_crop_rel"], he_piece)
        shutil.copy2(args.pool_dir / row["ficture_crop_rel"], fic_piece)
        make_local_context(he, mask, local_box, f"H&E local context: {uid}", he_local, args.view_max_side)
        make_local_context(fic, mask, local_box, f"FICTURE local context: {uid}", fic_local, args.view_max_side)
        make_locator(he, tight, f"H&E full ROI locator: {uid}", he_locator, args.locator_max_side)
        make_locator(fic, tight, f"FICTURE full ROI locator: {uid}", fic_locator, args.locator_max_side)

        out = {
            **row,
            "true_class": truth.get("classification_true_display", truth.get("classification_true_label", "")),
            "matched_annotation_component_id": truth.get("matched_annotation_component_id", ""),
            "component_dice": truth.get("component_dice", ""),
            "component_precision": truth.get("component_precision", ""),
            "component_recall": truth.get("component_recall", ""),
            "piece_area": truth.get("piece_area", ""),
            "bbox_xyxy": ",".join(str(v) for v in tight),
            "local_context_bbox_xyxy": ",".join(str(v) for v in local_box),
            "he_piece_rel": str(he_piece.relative_to(args.out_dir)),
            "ficture_piece_rel": str(fic_piece.relative_to(args.out_dir)),
            "he_local_context_rel": str(he_local.relative_to(args.out_dir)),
            "ficture_local_context_rel": str(fic_local.relative_to(args.out_dir)),
            "he_full_locator_rel": str(he_locator.relative_to(args.out_dir)),
            "ficture_full_locator_rel": str(fic_locator.relative_to(args.out_dir)),
            "image1_rel": str(he_piece.relative_to(args.out_dir)),
            "image2_rel": str(fic_piece.relative_to(args.out_dir)),
            "image3_rel": str(he_local.relative_to(args.out_dir)),
            "image4_rel": str(fic_local.relative_to(args.out_dir)),
            "image5_rel": str(he_locator.relative_to(args.out_dir)),
            "image6_rel": str(fic_locator.relative_to(args.out_dir)),
        }
        rows.append(out)

    write_csv(args.out_dir / "public_region_aware_requests.csv", rows)
    shutil.copy2(args.pool_dir / "hidden_candidate_truth.csv", args.out_dir / "hidden_candidate_truth.csv")
    (args.out_dir / "prompt_region_aware_he_ficture.txt").write_text(prompt_text(include_ficture=True))
    (args.out_dir / "prompt_region_aware_he_only.txt").write_text(prompt_text(include_ficture=False))
    (args.out_dir / "README.md").write_text(
        "# Jun09 region-aware piece scoring input pack\n\n"
        "This package tests whether piece-first tissue recognition improves when the model sees local context and a full ROI locator. "
        "The candidate masks are unchanged; only the input views are richer.\n\n"
        "Image order for VLM scoring is fixed and must not be changed:\n\n"
        "1. H&E piece reverse-blur crop\n"
        "2. FICTURE piece reverse-blur crop\n"
        "3. H&E local context crop\n"
        "4. FICTURE local context crop\n"
        "5. H&E full ROI locator\n"
        "6. FICTURE full ROI locator\n\n"
        "The model sees public_region_aware_requests.csv and the prompt files. "
        "hidden_candidate_truth.csv is only for post-hoc evaluation and assembly.\n\n"
        "Failure gates after scoring:\n"
        "- parse failures must be near zero;\n"
        "- score vectors must not collapse to all-zero/all-tie;\n"
        "- one predicted class should not dominate most candidates unless the input pool truly does;\n"
        "- assembly must report selected pieces and Dice / Precision / Recall per class.\n"
    )
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "pool_dir": str(args.pool_dir),
                "he_roi": str(HE_ROI),
                "ficture_roi": str(FICTURE_ROI),
                "n_candidates": len(rows),
                "local_scale": args.local_scale,
                "local_min_side": args.local_min_side,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    html_index(args.out_dir, rows)
    print(args.out_dir / "index.html")
    print(args.out_dir / "public_region_aware_requests.csv")


if __name__ == "__main__":
    main()
