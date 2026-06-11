#!/usr/bin/env python3
"""Build H&E visualization ablation inputs for Jun10 Step2.

The downstream VLM runner expects request CSV rows with ``he_crop_rel`` pointing
under ``corrected_pool``. This script creates small, style-specific corrected
pool roots for the same balanced candidate rows so the only changed variable is
how the H&E candidate mask is displayed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
JUN09_BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
JUN10_BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill"
REQUESTS = JUN10_BASE / "cell_type_first_requests_balanced18.csv"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
OUT_BASE = JUN10_BASE / "he_view_ablation_inputs_clean"
PAD = 72

STYLES = [
    "gray_reverse_blur",
    "blue_fill_only",
    "light_gray_masked",
    "outline_only",
    "bbox_only",
]


def crop_box(mask: np.ndarray, pad: int = PAD) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def mask_border(mask_img: Image.Image) -> Image.Image:
    dilated = mask_img.filter(ImageFilter.MaxFilter(5))
    eroded = mask_img.filter(ImageFilter.MinFilter(5))
    border = Image.fromarray((np.array(dilated) > np.array(eroded)).astype(np.uint8) * 255)
    return border


def render_blue_fill_only(he_crop: Image.Image, mask_crop: Image.Image) -> Image.Image:
    out = he_crop.convert("RGBA")
    mask = np.array(mask_crop) > 0
    overlay = Image.new("RGBA", he_crop.size, (0, 115, 255, 0))
    alpha = Image.fromarray(mask.astype(np.uint8) * 95, mode="L")
    overlay.putalpha(alpha)
    out = Image.alpha_composite(out, overlay)
    return out.convert("RGB")


def render_light_gray_masked(he_crop: Image.Image, mask_crop: Image.Image) -> Image.Image:
    he = he_crop.convert("RGB")
    mask = np.array(mask_crop) > 0
    out = Image.new("RGB", he.size, (244, 244, 244))
    out_arr = np.array(out)
    he_arr = np.array(he)
    out_arr[mask] = he_arr[mask]
    return Image.fromarray(out_arr)


def render_outline_only(he_crop: Image.Image, mask_crop: Image.Image) -> Image.Image:
    out = he_crop.convert("RGB")
    border = mask_border(mask_crop)
    draw = ImageDraw.Draw(out)
    draw.bitmap((0, 0), border, fill=(0, 90, 255))
    return out


def render_bbox_only(he_crop: Image.Image, mask_crop: Image.Image) -> Image.Image:
    out = he_crop.convert("RGB")
    arr = np.array(mask_crop) > 0
    ys, xs = np.where(arr)
    draw = ImageDraw.Draw(out)
    if len(xs):
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        draw.rectangle((x0, y0, x1, y1), outline=(0, 90, 255), width=5)
    return out


def main() -> None:
    df = pd.read_csv(REQUESTS)
    he_roi = Image.open(HE_ROI).convert("RGB")
    summary_rows: list[dict[str, object]] = []

    for style in STYLES:
        style_root = OUT_BASE / style
        crop_dir = style_root / "corrected_pool/candidate_pair_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        out_df = df.copy()

        for idx, row in df.iterrows():
            uid = str(row["candidate_uid"])
            mask_path = Path(str(row["mask_path"]))
            mask_full = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_full) > 0
            box = crop_box(mask_arr)
            he_crop = he_roi.crop(box)
            mask_crop = mask_full.crop(box)

            if style == "gray_reverse_blur":
                source = JUN09_BASE / "corrected_pool" / str(row["he_crop_rel"])
                rendered = Image.open(source).convert("RGB")
            elif style == "blue_fill_only":
                rendered = render_blue_fill_only(he_crop, mask_crop)
            elif style == "light_gray_masked":
                rendered = render_light_gray_masked(he_crop, mask_crop)
            elif style == "outline_only":
                rendered = render_outline_only(he_crop, mask_crop)
            elif style == "bbox_only":
                rendered = render_bbox_only(he_crop, mask_crop)
            else:
                raise KeyError(style)

            name = f"{uid}_he_{style}.png"
            rendered.save(crop_dir / name)
            out_df.loc[idx, "he_crop_rel"] = f"candidate_pair_crops/{name}"
            summary_rows.append(
                {
                    "style": style,
                    "candidate_uid": uid,
                    "true_class": row["true_class"],
                    "image_rel": f"{style}/corrected_pool/candidate_pair_crops/{name}",
                    "width": rendered.size[0],
                    "height": rendered.size[1],
                }
            )

        csv_path = OUT_BASE / f"cell_type_first_requests_balanced18_{style}.csv"
        out_df.to_csv(csv_path, index=False)

    pd.DataFrame(summary_rows).to_csv(OUT_BASE / "he_view_ablation_manifest.csv", index=False)
    print(OUT_BASE)


if __name__ == "__main__":
    main()
