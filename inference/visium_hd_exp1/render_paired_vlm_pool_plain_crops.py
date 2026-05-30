#!/usr/bin/env python3
"""Re-render a paired VLM hit-test pool without colored mask overlays.

The existing paired pool stores the candidate table and hidden truth. This
script keeps the same rows but replaces the H&E/FICTURE images with SaLIP-style
masked crops: crop around the candidate mask, keep pixels inside the mask, and
replace the outside with a neutral light background. This preserves the true
H&E/FICTURE colors inside the candidate instead of adding a blue fill overlay.
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageFilter


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.array(image) > 127


def bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    pad: int,
    max_side: int,
    background: Tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    x1, y1, x2, y2 = bbox(mask)
    h, w = mask.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2].astype(np.uint8)
    crop_mask = mask[y1:y2, x1:x2]
    bg = np.zeros_like(crop, dtype=np.uint8)
    bg[..., 0] = background[0]
    bg[..., 1] = background[1]
    bg[..., 2] = background[2]
    out = np.where(crop_mask[..., None], crop, bg)
    img = Image.fromarray(out)
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return img


def reverse_blur_gray_crop(
    image: np.ndarray,
    mask: np.ndarray,
    pad: int,
    max_side: int,
    blur_radius: float,
) -> Image.Image:
    x1, y1, x2, y2 = bbox(mask)
    h, w = mask.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = Image.fromarray(image[y1:y2, x1:x2].astype(np.uint8)).convert("RGB")
    crop_mask = mask[y1:y2, x1:x2]
    gray_blur = crop.convert("L").convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = np.array(gray_blur)
    crop_np = np.array(crop)
    out[crop_mask] = crop_np[crop_mask]
    img = Image.fromarray(out.astype(np.uint8))
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return img


def image_tag(path: str, alt: str) -> str:
    return f'<img src="{html.escape(path)}" alt="{html.escape(alt)}">'


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def write_html(output_dir: Path, rows: list[dict]) -> None:
    labels = []
    seen = set()
    for row in rows:
        key = (row["label"], row["display"])
        if key not in seen:
            labels.append(key)
            seen.add(key)

    bucket_counts = {}
    for row in rows:
        key = (row.get("label", ""), row.get("sample_bucket", ""))
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
    per_label_text = ", ".join(
        f"{display}: {sum(bucket_counts.get((label, bucket), 0) for bucket in ['GOOD', 'MID', 'BAD'])}"
        for label, display in labels
    )

    sections = []
    for label, display in labels:
        sections.append(f"<h2>{html.escape(display)}</h2><div class='grid'>")
        for row in [r for r in rows if r["label"] == label]:
            sections.append(
                "<article>"
                f"<h3>{html.escape(row['sample_bucket'])}: "
                f"{html.escape(row['source'])} / {html.escape(row['setting'])} / "
                f"{html.escape(str(row['candidate_id']))}</h3>"
                "<div class='imgs'>"
                + image_tag(row["he_crop_rel"], "H&E masked crop")
                + image_tag(row["ficture_crop_rel"], "FICTURE masked crop")
                + "</div>"
                "<p>这里没有蓝色填充高亮。浅灰色是 mask 外部，彩色/组织像素只保留在 candidate mask 内部。</p>"
                f"<details><summary>VLM prompt</summary><pre>{html.escape(row.get('prompt_text', ''))}</pre></details>"
                "</article>"
            )
        sections.append("</div>")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>No-highlight paired H&E/FICTURE VLM pool</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #20242a; }}
.note {{ max-width: 1020px; line-height: 1.6; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
article {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: white; }}
.imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
img {{ width: 100%; height: auto; border: 1px solid #eee; background: #fafafa; }}
pre {{ white-space: pre-wrap; font-size: 12px; }}
</style>
</head>
<body>
<h1>无蓝色高亮的本地 VLM 直接检索输入池</h1>
<div class="note">
<p><b>目的：</b>测试去掉蓝色 overlay 后，本地 VLM 能不能更好地从 GOOD/MID/BAD candidate 里把真正好的 mask 排到前面。</p>
<p><b>输入：</b>这个 pool 一共有 {len(rows)} 个 candidate。每个 candidate 仍然有两张图：H&E masked crop 和官方 FICTURE masked crop。两张图来自同一个 mask、同一个 ROI；mask 外部用浅灰遮掉，不再用蓝色填充。</p>
<p><b>每类数量：</b>{html.escape(per_label_text)}。</p>
<p><b>评价：</b>VLM 只输出 0-1 分数；隐藏 Dice/Precision/Recall 不进 prompt，只在打分后用来判断有没有提升。</p>
</div>
{''.join(sections)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def pool_count_summary(rows: list[dict]) -> str:
    labels = []
    seen = set()
    for row in rows:
        key = (row["label"], row["display"])
        if key not in seen:
            labels.append(key)
            seen.add(key)
    bucket_counts = {}
    for row in rows:
        key = (row.get("label", ""), row.get("sample_bucket", ""))
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
    return ", ".join(
        f"{display}: {sum(bucket_counts.get((label, bucket), 0) for bucket in ['GOOD', 'MID', 'BAD'])}"
        for label, display in labels
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--pad", type=int, default=72)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument(
        "--mode",
        choices=["plain_masked", "reverse_blur_gray"],
        default="plain_masked",
        help="plain_masked replaces outside-mask pixels with light gray; reverse_blur_gray keeps outside context as grayscale blur.",
    )
    parser.add_argument("--blur-radius", type=float, default=7.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "candidate_pair_crops"
    image_dir.mkdir(exist_ok=True)

    ficture = np.array(Image.open(args.ficture_image).convert("RGB"))
    shape_hw = ficture.shape[:2]
    he = np.array(Image.open(args.he_image).convert("RGB"))
    if he.shape[:2] != shape_hw:
        he = np.array(
            Image.fromarray(he).resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR)
        )

    public_path = args.source_pool / "public_vlm_requests.csv"
    hidden_path = args.source_pool / "hidden_candidate_truth.csv"
    rows = list(csv.DictReader(public_path.open()))
    fieldnames = list(rows[0].keys())
    if "image_cue" not in fieldnames:
        fieldnames.append("image_cue")

    if args.mode == "reverse_blur_gray":
        image_cue = (
            "Image 1: H&E reverse-blur crop. The candidate mask remains sharp and full color; "
            "surrounding context outside the candidate is grayscale and blurred; there is no colored overlay.\n"
            "Image 2: official FICTURE factor-color reverse-blur crop for the same candidate mask. "
            "The candidate mask remains sharp and full color; surrounding context outside the candidate is grayscale and blurred; "
            "there is no colored overlay."
        )
        html_note = "这里没有蓝色填充高亮。candidate mask 内部保留原始颜色和清晰度，mask 外部保留灰度模糊上下文。"
    else:
        image_cue = (
            "Image 1: H&E masked crop. Pixels outside the candidate mask are replaced by light gray; "
            "there is no colored overlay.\n"
            "Image 2: official FICTURE factor-color masked crop for the same candidate mask. "
            "Pixels outside the candidate mask are replaced by light gray; there is no colored overlay."
        )
        html_note = "这里没有蓝色填充高亮。浅灰色是 mask 外部，彩色/组织像素只保留在 candidate mask 内部。"

    for row in rows:
        mask_path = Path(row["mask_path"])
        mask = read_mask(mask_path, shape_hw)
        uid = row["candidate_uid"]
        he_name = f"{uid}_he_{args.mode}.png"
        ficture_name = f"{uid}_ficture_{args.mode}.png"
        if args.mode == "reverse_blur_gray":
            reverse_blur_gray_crop(he, mask, args.pad, args.max_side, args.blur_radius).save(image_dir / he_name)
            reverse_blur_gray_crop(ficture, mask, args.pad, args.max_side, args.blur_radius).save(image_dir / ficture_name)
        else:
            masked_crop(he, mask, args.pad, args.max_side).save(image_dir / he_name)
            masked_crop(ficture, mask, args.pad, args.max_side).save(image_dir / ficture_name)
        row["he_crop_rel"] = f"candidate_pair_crops/{he_name}"
        row["ficture_crop_rel"] = f"candidate_pair_crops/{ficture_name}"
        row["image_cue"] = image_cue
        row["prompt_text"] = (
            f"Target tissue class: {row['display']} ({row['label']}).\n"
            f"Plain meaning: {row.get('target_description', '')}.\n"
            f"{image_cue}\n"
            f"FICTURE color/cell-type/gene hints for this target: {row.get('target_factor_hints') or 'not available'}.\n"
            f"Candidate FICTURE color composition: {row.get('candidate_factor_composition') or 'not available'}.\n"
            "Score whether the candidate mask crop is a good candidate for the target class. "
            "Return JSON only: {\"score\": 0.0-1.0, \"precision\": 0.0-1.0, \"recall\": 0.0-1.0, \"reason\": \"short reason\"}."
        )

    write_csv(args.output_dir / "public_vlm_requests.csv", rows, fieldnames)
    shutil.copy2(hidden_path, args.output_dir / "hidden_candidate_truth.csv")
    summary_src = args.source_pool / "hit_test_pool_summary.csv"
    if summary_src.exists():
        shutil.copy2(summary_src, args.output_dir / "hit_test_pool_summary.csv")
    write_html(args.output_dir, rows)
    index_path = args.output_dir / "index.html"
    index_text = index_path.read_text().replace(
        "这里没有蓝色填充高亮。浅灰色是 mask 外部，彩色/组织像素只保留在 candidate mask 内部。",
        html_note,
    )
    if args.mode == "reverse_blur_gray":
        index_text = index_text.replace(
            "mask 外部用浅灰遮掉，不再用蓝色填充",
            "mask 外部保留灰度模糊上下文，不再用蓝色填充",
        )
    index_path.write_text(index_text)
    per_label_text = pool_count_summary(rows)
    (args.output_dir / "README_中文.md").write_text(
        "# 无高亮 paired H&E/FICTURE VLM pool\n\n"
        f"这份 pool 有 {len(rows)} 个 candidate，只改变给 VLM 看的图：\n"
        f"每类 candidate 数量：{per_label_text}。\n"
        f"渲染模式：`{args.mode}`。不再用蓝色半透明 overlay。\n"
        "隐藏 Dice/Precision/Recall 不给模型，只用于最后评价排序是否变好。\n"
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
