#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a plain-language HTML report for source-image CLIP top-k retrieval.

The report visualizes the exact image crop that CLIP saw for each top-ranked
candidate:

candidate mask -> crop source image with mask outside set to light gray ->
CLIP image/text score -> top-k mask evaluation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image


LABELS = [
    ("lung_bronchiola", "bronchiola", "支气管/细支气管"),
    ("lung_alveoli_normal_adjacent", "alveoli", "肺泡"),
    ("lung_vessels", "vessels", "血管"),
    ("tumor", "tumor", "肿瘤"),
    ("stroma", "stroma", "间质"),
    ("immune_infiltration", "immune infiltration", "免疫浸润"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def resize_bool(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(im) > 0


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clip_crop(image: np.ndarray, mask: np.ndarray, pad: int = 24) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox(mask)
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if crop_mask.any():
        bg = np.full_like(crop, 245)
        crop = np.where(crop_mask[..., None], crop, bg)
    return Image.fromarray(crop.astype(np.uint8)), (x1, y1, x2, y2)


def overlay_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 112, 255),
    alpha: float = 0.45,
) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2].copy().astype(np.float32)
    crop_mask = mask[y1:y2, x1:x2]
    crop[crop_mask] = crop[crop_mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))


def mask_thumb(mask: np.ndarray, crop_box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    crop_mask = mask[y1:y2, x1:x2]
    arr = np.ones((crop_mask.shape[0], crop_mask.shape[1], 3), dtype=np.uint8) * 255
    arr[crop_mask] = np.array([0, 112, 255], dtype=np.uint8)
    return Image.fromarray(arr)


def fit_image(image: Image.Image, max_w: int = 260, max_h: int = 190) -> Image.Image:
    image = image.copy()
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), "white")
    canvas.paste(image, ((max_w - image.width) // 2, (max_h - image.height) // 2))
    return canvas


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def fmt(value: str | float) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def load_scores(run_dir: Path, label_slugs: set[str]) -> Dict[str, list[dict]]:
    rows_by_label = {slug: [] for slug in label_slugs}
    for row in read_csv(run_dir / "source_image_salip_clip_candidate_scores.csv"):
        if row["label"] not in label_slugs:
            continue
        row["score_float"] = float(row["score"])
        rows_by_label[row["label"]].append(row)
    for slug in rows_by_label:
        rows_by_label[slug].sort(key=lambda row: row["score_float"], reverse=True)
    return rows_by_label


def html_table(rows: Iterable[str]) -> str:
    return "<table>" + "\n".join(rows) + "</table>"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "topk_visual_report_chinese")
    thumb_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    label_slugs = {slug for slug, _, _ in LABELS}
    summary = json.loads((run_dir / "source_image_salip_clip_summary.json").read_text())
    source_images = {source: Path(path) for source, path in summary["source_images"].items()}
    source_arrays = {source: np.asarray(Image.open(path).convert("RGB")) for source, path in source_images.items()}

    scores_by_label = load_scores(run_dir, label_slugs)
    selection_rows = [row for row in read_csv(run_dir / "source_image_salip_clip_selection_metrics.csv") if row["label"] in label_slugs]
    best_rows = [row for row in read_csv(run_dir / "source_image_salip_clip_best_by_label.csv") if row["label"] in label_slugs]

    for label_slug, _, _ in LABELS:
        for rank, row in enumerate(scores_by_label[label_slug][: args.top_n], 1):
            source = row["source"]
            image = source_arrays[source]
            mask = resize_bool(read_mask(Path(row["mask_path"])), image.shape[:2])
            crop, crop_box = clip_crop(image, mask)
            overlay = overlay_crop(image, mask, crop_box)
            mask_only = mask_thumb(mask, crop_box)
            prefix = (
                f"{label_slug}__rank{rank:02d}__{source}__"
                f"{safe_slug(row['setting'])}__{int(row['candidate_id']):03d}"
            )
            fit_image(crop).save(thumb_dir / f"{prefix}_clip_crop.jpg", quality=92)
            fit_image(overlay).save(thumb_dir / f"{prefix}_overlay.jpg", quality=92)
            fit_image(mask_only).save(thumb_dir / f"{prefix}_mask.jpg", quality=92)
            row["thumb_prefix"] = prefix

    css = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;margin:24px;color:#202124;line-height:1.45;background:#fff}
h1{font-size:28px;margin:0 0 8px} h2{font-size:22px;margin:34px 0 8px;border-top:2px solid #eee;padding-top:22px} h3{font-size:17px;margin:20px 0 8px}
.note{background:#f7f9fc;border-left:5px solid #4776d0;padding:12px 14px;margin:14px 0;max-width:1180px}.warn{background:#fff7e6;border-left-color:#e09000}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f1f3f4;border-radius:5px;padding:1px 5px}
table{border-collapse:collapse;margin:10px 0 18px;max-width:1250px;width:100%;font-size:14px} th,td{border-bottom:1px solid #ddd;padding:7px 8px;text-align:left;vertical-align:top} th{background:#f6f6f6;font-weight:700}.num{text-align:right;font-variant-numeric:tabular-nums}
.source-he{background:#e9f3ff}.source-ficture{background:#fff0e0}.badge{border-radius:999px;padding:2px 7px;font-size:12px;font-weight:700}.cardgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:14px;margin:10px 0 28px}.card{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fff}.cardhead{font-weight:700;margin-bottom:8px}.imgs{display:flex;gap:8px;align-items:flex-start}.thumbblock{font-size:12px;color:#555;text-align:center}.thumbblock img{display:block;width:190px;max-height:145px;object-fit:contain;border:1px solid #eee;background:white;margin-bottom:3px}.small{font-size:13px;color:#555}.topklist{font-size:13px;max-width:650px;word-break:break-word}.legend{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 14px}
"""

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>CLIP crop direct retrieval top-k report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>CLIP 直接检索报告：candidate mask -> 原图 crop -> 文本相似度 -> top-k mask</h1>",
        (
            "<div class='note'><b>实验设计（给不懂项目的人看）</b><br>"
            "目标：测试 OpenAI CLIP 能不能从 H&E + FICTURE 的 candidate pool 里，直接把每个组织类别想要的 mask 捞出来。<br>"
            "输入：每个 candidate mask 先裁原图。H&E candidate 裁 H&E ROI；FICTURE candidate 裁 FICTURE map ROI。"
            "裁剪时 mask 外区域变成浅灰背景，这就是 CLIP 实际看到的图。<br>"
            "输出：CLIP 给每个 crop 和组织类别文字 prompt 算一个图文相似度分数。分数越高，排名越靠前。<br>"
            "top-k：按 CLIP 分数取前 k 个 crop，把它们对应的 mask 合并，再用 annotation 计算 Dice/Precision/Recall。"
            "annotation 只用于最后评价，不参与打分。<br>"
            "官方约束：FICTURE 使用 PASS_OFFICIAL 的同一 3144 x 3327 ROI；旧 FICTURE root 没有用于这里。</div>"
        ),
        (
            "<div class='note warn'><b>一句话结论：</b>"
            "这次 direct CLIP crop retrieval 流程已经跑完，但 CLIP 按 crop/text 相似度选出来的 top-k 多数没有命中真正好的 mask；"
            "vessels 相对最好，其他类尤其 alveoli、immune 明显捞偏。</div>"
        ),
        "<div class='legend'><span class='badge source-he'>source=H&E pool</span><span class='badge source-ficture'>source=FICTURE pool</span><span class='mono'>score</span> 是 CLIP 图文相似度，不是 Dice。</div>",
        "<h2>六类 top-k 评价总表</h2>",
        (
            "<p class='small'>这里显示每个 pool_scope 下 CLIP 最好的 top-k。"
            "<span class='mono'>all_sources</span> 是 H&E + FICTURE 合并；"
            "<span class='mono'>he_only</span> 和 <span class='mono'>ficture_only</span> 是分别只看一个 pool。</p>"
        ),
    ]

    summary_rows = [
        "<tr><th>pool scope</th><th>class</th><th>best top-k</th><th>Dice</th><th>Precision</th><th>Recall</th><th>selected source sequence</th></tr>"
    ]
    for scope in ["all_sources", "he_only", "ficture_only"]:
        for label_slug, display, cn_name in LABELS:
            row = next((item for item in best_rows if item["pool_scope"] == scope and item["label"] == label_slug), None)
            if not row:
                continue
            summary_rows.append(
                "<tr>"
                f"<td><span class='mono'>{scope}</span></td>"
                f"<td>{html.escape(cn_name)}<br><span class='small'>{html.escape(display)}</span></td>"
                f"<td class='num'>{html.escape(row['top_k'])}</td>"
                f"<td class='num'>{fmt(row['dice'])}</td>"
                f"<td class='num'>{fmt(row['precision'])}</td>"
                f"<td class='num'>{fmt(row['recall'])}</td>"
                f"<td class='topklist'>{html.escape(row['selected_sources'])}</td>"
                "</tr>"
            )
    parts.append(html_table(summary_rows))

    for label_slug, display, cn_name in LABELS:
        parts.append(f"<h2>{html.escape(cn_name)} <span class='small'>({html.escape(display)})</span></h2>")
        parts.append("<h3>这个 class 的 all_sources top-k Dice</h3>")
        rows = ["<tr><th>top-k</th><th>Dice</th><th>Precision</th><th>Recall</th><th>CLIP 选中的 mask 来源/ID</th></tr>"]
        for top_k in ["1", "2", "3", "5", "8", "12", "20"]:
            row = next(
                (
                    item
                    for item in selection_rows
                    if item["pool_scope"] == "all_sources" and item["label"] == label_slug and item["top_k"] == top_k
                ),
                None,
            )
            if not row:
                continue
            selected = row["selected"].split(";") if row.get("selected") else []
            selected_short = "; ".join(selected[:8]) + ("; ..." if len(selected) > 8 else "")
            rows.append(
                "<tr>"
                f"<td class='num'>{top_k}</td>"
                f"<td class='num'>{fmt(row['dice'])}</td>"
                f"<td class='num'>{fmt(row['precision'])}</td>"
                f"<td class='num'>{fmt(row['recall'])}</td>"
                f"<td class='topklist'>{html.escape(selected_short)}</td>"
                "</tr>"
            )
        parts.append(html_table(rows))
        parts.append("<h3>CLIP 分数最高的前 20 个 candidate crop</h3><div class='cardgrid'>")
        for rank, row in enumerate(scores_by_label[label_slug][: args.top_n], 1):
            source_class = "source-he" if row["source"] == "he" else "source-ficture"
            source_label = "H&E pool" if row["source"] == "he" else "FICTURE pool"
            prefix = row["thumb_prefix"]
            parts.append("<div class='card'>")
            parts.append(
                f"<div class='cardhead'>Rank {rank} | score={float(row['score']):.6f} | "
                f"<span class='badge {source_class}'>{source_label}</span></div>"
            )
            parts.append(
                "<div class='small'>"
                f"setting: <span class='mono'>{html.escape(row['setting'])}</span> &nbsp; "
                f"candidate id: <span class='mono'>{int(row['candidate_id']):03d}</span><br>"
                f"run: <span class='mono'>{html.escape(row['run'])}</span>"
                "</div>"
            )
            parts.append("<div class='imgs'>")
            parts.append(
                f"<div class='thumbblock'><img src='thumbs/{prefix}_clip_crop.jpg'><b>CLIP 实际输入 crop</b></div>"
            )
            parts.append(
                f"<div class='thumbblock'><img src='thumbs/{prefix}_overlay.jpg'><b>mask overlay</b></div>"
            )
            parts.append(f"<div class='thumbblock'><img src='thumbs/{prefix}_mask.jpg'><b>mask only</b></div>")
            parts.append("</div></div>")
        parts.append("</div>")

    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")
    (out_dir / "README_中文.md").write_text(
        "# CLIP crop direct retrieval top-k report\n\n"
        "这个报告展示 source_image_salip_clip：candidate mask -> 裁对应原图 -> CLIP 图文相似度 -> top-k mask。\n\n"
        "- `index.html`: 中文可视化报告\n"
        "- `thumbs/`: 每个 top-20 candidate 的 CLIP 实际输入 crop、overlay、mask\n",
        encoding="utf-8",
    )
    print(out_dir)
    print(out_dir / "index.html")
    print(f"thumbs={len(list(thumb_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
