#!/usr/bin/env python3
"""Render a Chinese full-ROI report for the feature-aware CLIP/VLM hit test.

This is a direct retrieval report: CLIP and VLM score the same small candidate
pool, then we check whether the hidden GOOD candidate is ranked first.  The
figures intentionally show the whole official ROI instead of local crops so
alignment can be inspected by eye.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent.parent / "output" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OFFICIAL_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
OFFICIAL_INPUT_DIR = PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "output/visium_hd_exp1/vlm_topk_selection_audit/manual_direct_clip_vlm_hit_test_20260521"
)

LABEL_ORDER = [
    ("bronchiola", "lung_bronchiola", "支气管/细支气管"),
    ("alveoli", "lung_alveoli_normal_adjacent", "肺泡"),
    ("vessels", "lung_vessels", "血管"),
    ("tumor", "tumor", "肿瘤"),
    ("stroma", "stroma", "间质"),
    ("immune infiltration", "immune_infiltration", "免疫浸润"),
]

FORBIDDEN_PATH_FRAGMENTS = [
    "ficture_coord_scaled_hires",
    "ficture_corrected",
    "agent_verified_filtered_ficture_he_align",
    "deprecated_ficture_pre_official",
]

SCORE_NAME_ZH = {
    "CLIP mean": "CLIP",
    "Qwen2.5-VL-7B VLM": "Qwen2.5-VL",
    "Gemma-3-27B VLM": "Gemma",
    "MedGemma-4B VLM": "MedGemma",
    "Llama-3.2-11B-Vision VLM": "Llama 3.2 Vision",
}

DEFAULT_SELECTOR_COLUMNS = [
    "CLIP mean",
    "Qwen2.5-VL-7B VLM",
    "Gemma-3-27B VLM",
    "MedGemma-4B VLM",
    "Llama-3.2-11B-Vision VLM",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check_no_forbidden(paths: Iterable[Path]) -> None:
    bad: list[str] = []
    for path in paths:
        text = str(path)
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in text:
                bad.append(text)
    if bad:
        raise SystemExit("Refusing deprecated FICTURE path(s):\n" + "\n".join(sorted(set(bad))))


def verify_pass_official(summary_path: Path, input_dir: Path) -> dict:
    check_no_forbidden([summary_path, input_dir])
    summary = load_json(summary_path)
    if summary.get("status") != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {summary.get('status')}")

    he_path = input_dir / "he_roi_matching_official_ficture_coverage.png"
    ficture_path = input_dir / "ficture_official_filtered_roi_rgb.png"
    factor_path = input_dir / "ficture_official_filtered_roi_factor_index.npy"
    region_summary_path = input_dir / "region_summary_official_filtered_roi.json"
    for path in [he_path, ficture_path, factor_path, region_summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    he_size = Image.open(he_path).size
    ficture_size = Image.open(ficture_path).size
    factor_shape = np.load(factor_path, mmap_mode="r").shape
    if he_size != (3144, 3327):
        raise SystemExit(f"Unexpected official H&E ROI size: {he_size}; expected (3144, 3327)")
    if ficture_size != he_size:
        raise SystemExit(f"H&E/FICTURE ROI size mismatch: {he_size} vs {ficture_size}")
    if factor_shape != (he_size[1], he_size[0]):
        raise SystemExit(f"Factor index shape mismatch: {factor_shape}; expected {(he_size[1], he_size[0])}")

    return {
        "summary": summary,
        "he_path": he_path,
        "ficture_path": ficture_path,
        "factor_path": factor_path,
        "region_summary_path": region_summary_path,
        "roi_size_wh": he_size,
        "factor_shape": factor_shape,
    }


def resolve_path(value: str, report_dir: Path) -> Path:
    raw = Path(value)
    candidates = [
        raw,
        PROJECT_ROOT / raw,
        report_dir / raw,
    ]
    # Some HPC paths include the project root prefix; strip it for local copies.
    text = str(value)
    marker = "Medical-SAM3/"
    if marker in text:
        rel = Path(text.split(marker, 1)[1])
        candidates.extend([PROJECT_ROOT / rel, report_dir / rel])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(value)


def load_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def load_rgb(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if shape_hw is not None and image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(image)


def downsample_rgb(image: np.ndarray, display_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= display_height:
        return image
    new_w = int(round(w * display_height / h))
    return np.asarray(Image.fromarray(image).resize((new_w, display_height), Image.Resampling.BILINEAR))


def downsample_mask(mask: np.ndarray, display_height: int) -> np.ndarray:
    h, w = mask.shape
    if h <= display_height:
        return mask
    new_w = int(round(w * display_height / h))
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((new_w, display_height), Image.Resampling.NEAREST)) > 127


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def solid_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def score_columns(rows: list[dict], allowed_columns: list[str] | None = None) -> list[str]:
    if not rows:
        return []
    if allowed_columns is not None:
        return [column for column in allowed_columns if column in rows[0]]
    columns = [c for c in ["CLIP mean"] if c in rows[0]]
    columns.extend(c for c in rows[0] if c in DEFAULT_SELECTOR_COLUMNS and c.endswith(" VLM"))
    return columns


def score_label(column: str) -> str:
    if column in SCORE_NAME_ZH:
        return SCORE_NAME_ZH[column]
    if column.endswith(" VLM"):
        return column.replace(" VLM", " 直接VLM打分")
    return column


def selector_example_text() -> str:
    return """
<h2>一个具体例子：bronchiola GOOD candidate</h2>
<p class='lead'>例如 bronchiola 的 GOOD candidate 是 <code>ficture/base_official_points_step32/303</code>。模型并不知道它是 GOOD；GOOD 只是我们事后用真实 annotation 算 Dice 时给它贴的隐藏标签。</p>
<table><thead><tr><th>方法</th><th>这个例子里实际给模型看的东西</th></tr></thead><tbody>
<tr><td><b>CLIP</b></td><td>图像输入是两张 crop：一张 H&E crop，一张 FICTURE color crop。两张 crop 都是围绕 candidate mask 的外接框裁出来，外扩 24 px；mask 外区域变浅灰。文字输入包括：<br><code>a histology crop showing a bronchiolar airway with a visible lumen and epithelial lining</code><br><code>a lung tissue region containing a branching bronchiole or airway structure</code><br><code>bronchiolar tissue in H and E stained lung histology</code><br>以及 FICTURE 语义提示，例如 <code>F7 bronchiolar secretory markers SCGB1A1, SCGB3A1</code>、<code>F9 neuroendocrine airway markers GRP, KRT5</code>。</td></tr>
<tr><td><b>VLM</b></td><td>图像输入是三张 crop：H&E 上高亮 candidate、FICTURE map 上高亮同一个 candidate、bronchiola gene/FICTURE prior heatmap 上高亮同一个 candidate。文字输入会说明目标是 bronchiola，并给出类似这样的上下文：<br><code>Target tissue class: lung_bronchiola. Meaning: bronchiolar airway tissue with lumen and epithelial lining.</code><br><code>FICTURE interpretation: factor 7 RGB 0,170,170 bronchiolar secretory / club airway epithelium, genes SCGB1A1, SCGB3A1, BPIFB1, MSMB; factor 9 RGB 255,0,127 neuroendocrine airway, genes GRP, KRT5, SCGB1A1, TPH1.</code><br>然后要求模型只返回 JSON 分数，例如 <code>{"score": 0.73, "recall": 0.68, "precision": 0.82, "reason": "short reason"}</code>。</td></tr>
</tbody></table>
"""


def float_value(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_selectors(rows: list[dict], allowed_columns: list[str] | None = None) -> list[dict]:
    summaries: list[dict] = []
    for display, _slug, zh in LABEL_ORDER:
        label_rows = [row for row in rows if row["label"] == display]
        if not label_rows:
            continue
        for column in score_columns(rows, allowed_columns):
            scored = [(row, float_value(row, column)) for row in label_rows]
            scored = [(row, score) for row, score in scored if score is not None]
            if not scored:
                summaries.append(
                    {
                        "label": display,
                        "label_zh": zh,
                        "selector": score_label(column),
                        "top_pick": "没有分数",
                        "good_rank": "",
                        "good_score": "",
                        "top_dice": "",
                        "unique_scores": "0",
                        "interpretation": "没有输出",
                    }
                )
                continue
            scored.sort(key=lambda item: (item[1], float(item[0]["true_dice"])), reverse=True)
            good_score = next((score for row, score in scored if row["manual_pick"] == "GOOD"), None)
            if good_score is None:
                good_rank = ""
            else:
                good_rank = 1 + sum(1 for _row, score in scored if score > good_score)
            top_score = scored[0][1]
            top_rows = [row for row, score in scored if abs(score - top_score) < 1e-9]
            unique_scores = len({round(score, 8) for _row, score in scored})
            top_pick = " / ".join(row["manual_pick"] for row in top_rows)
            if unique_scores <= 1:
                interpretation = "分数几乎一样，不能说明模型真的会挑"
            elif good_rank == 1 and len(top_rows) == 1:
                interpretation = "挑中了 GOOD"
            elif good_rank == 1:
                interpretation = "GOOD 并列第一"
            else:
                interpretation = "没挑中 GOOD"
            summaries.append(
                {
                    "label": display,
                    "label_zh": zh,
                    "selector": score_label(column),
                    "top_pick": top_pick,
                    "good_rank": good_rank,
                    "good_score": f"{good_score:.4f}" if good_score is not None else "",
                    "top_dice": f"{float(top_rows[0]['true_dice']):.3f}",
                    "unique_scores": str(unique_scores),
                    "interpretation": interpretation,
                }
            )
    return summaries


def render_candidate_panel(
    row: dict,
    label_zh: str,
    he_full: np.ndarray,
    ficture_full: np.ndarray,
    annotation_full: np.ndarray,
    mask_full: np.ndarray,
    output_path: Path,
    display_height: int,
) -> None:
    he = downsample_rgb(he_full, display_height)
    ficture = downsample_rgb(ficture_full, display_height)
    annotation = downsample_mask(annotation_full, display_height)
    mask = downsample_mask(mask_full, display_height)

    cand_color = (0, 112, 255)
    ann_color = (0, 185, 95)
    panels = [
        (he, "H&E ROI only", "black"),
        (overlay(he, mask, cand_color, 0.50), "Candidate on H&E ROI", "blue"),
        (solid_mask(mask, cand_color), "Candidate only", "blue"),
        (overlay(he, annotation, ann_color, 0.48), "Annotation on H&E ROI", "green"),
        (solid_mask(annotation, ann_color), "Annotation only", "green"),
        (ficture, "Official FICTURE map", "black"),
    ]

    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "Songti SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 6, figsize=(30, 7.4))
    for ax, (image, title, color) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10, color=color, loc="left")
        ax.axis("off")

    title = (
        f"{label_zh} / {row['label']}  {row['manual_pick']} candidate, full official ROI\n"
        f"Dice {float(row['true_dice']):.3f} | Precision {float(row['true_precision']):.3f} | "
        f"Recall {float(row['true_recall']):.3f} | {row['candidate']}"
    )
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.86], w_pad=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keep = {label for label, _slug, _zh in LABEL_ORDER}
    rows = [row for row in rows if row.get("label") in keep]
    order = {label: i for i, (label, _slug, _zh) in enumerate(LABEL_ORDER)}
    pick_order = {"GOOD": 0, "MID": 1, "BAD": 2}
    rows.sort(key=lambda row: (order[row["label"]], pick_order.get(row["manual_pick"], 99)))
    return rows


def write_html(
    rows: list[dict],
    summaries: list[dict],
    output_html: Path,
    panel_rel_paths: dict[tuple[str, str], str],
    preflight: dict,
    legend_path: Path,
    allowed_columns: list[str],
) -> None:
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Heiti SC", Arial, sans-serif; margin: 28px; color: #202124; line-height: 1.48; }
h1 { font-size: 30px; margin: 0 0 8px; }
h2 { margin-top: 32px; border-top: 1px solid #ddd; padding-top: 18px; }
h3 { margin-top: 24px; }
.lead { max-width: 1180px; font-size: 16px; color: #3c4043; }
.box { border: 1px solid #ddd; background: #fafafa; padding: 16px 18px; margin: 18px 0; max-width: 1180px; }
.box p { margin: 8px 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 22px; font-size: 14px; }
th, td { border-bottom: 1px solid #e0e0e0; padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 700; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.hit { color: #137333; font-weight: 700; }
.miss { color: #b3261e; font-weight: 700; }
.tie { color: #9a6700; font-weight: 700; }
.card { margin: 20px 0 34px; }
.card img { width: 100%; max-width: 1800px; border: 1px solid #d0d7de; background: white; }
.meta { color: #57606a; font-size: 13px; margin: 4px 0 10px; }
.good { color: #137333; font-weight: 700; }
.mid { color: #9a6700; font-weight: 700; }
.bad { color: #b3261e; font-weight: 700; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
"""
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>六类 feature-aware CLIP/VLM 候选池命中测试</title>",
        f"<style>{css}</style>",
        "<h1>六类 feature-aware CLIP/VLM 候选池命中测试</h1>",
        "<p class='lead'>这份报告只回答一个问题：给定一小组已经混入好/中/坏的候选 mask，CLIP 或 VLM 能不能在不知道真实标注的情况下，把真正好的候选挑出来。</p>",
        "<div class='box'>",
        "<h2>实验设计</h2>",
        "<p><b>目标：</b>把 CLIP 和 VLM 放到同一个任务里比较：给每个 candidate mask 打一个分，然后看 GOOD candidate 能不能排第一。这里不是最终 segmentation，也不是第二阶段融合重排。</p>",
        "<p><b>候选池：</b>六个类别，每类 3 个候选：GOOD 是已知比较好的 mask，MID 是中等 mask，BAD 是从别的类别混进来的错误 mask。真实 Dice/Precision/Recall 只用来事后检查模型有没有挑对，模型打分时看不到。</p>",
        "<p><b>这次看 top 几：</b>这个小实验每类只有 GOOD/MID/BAD 三个候选，所以当前只看 <b>top-1</b>：模型最高分挑出来的那个到底是 GOOD、MID 还是 BAD。这里不是之前大规模 ranking 里的 top-1/top-3/top-5/top-20 union。</p>",
        "<p><b>H&E 是什么：</b>H&E 是普通病理切片图像，不是 candidate mask。candidate mask 是算法画出来的蓝色区域。模型输入会用 candidate mask 决定看哪一块 H&E 或 FICTURE 图。</p>",
        "<p><b>这份 HTML 里的六联图是不是模型输入：</b>不是。六联图是给人看的，用来检查候选 mask 在整张组织 ROI 上的位置和对齐情况。CLIP/VLM 实际打分时用的是从同一张官方 ROI 里裁出来的 candidate crop。</p>",
        "<p><b>官方对齐约束：</b>本报告先检查 <code>PASS_OFFICIAL</code>；H&E、FICTURE、annotation 都使用官方 ROI <code>3144×3327</code>。candidate mask 原始分辨率可能不同，但都先按 nearest-neighbor 放回官方 ROI 尺寸；不使用旧的 deprecated FICTURE 路径。</p>",
        "</div>",
        "<h2>模型到底看了什么</h2>",
        "<table><thead><tr><th>模型/打分方法</th><th>图像输入</th><th>文字输入</th><th>输出分数是什么意思</th></tr></thead><tbody>",
        "<tr><td><b>CLIP</b></td><td>两张 candidate crop：第一张来自 H&E，第二张来自官方 FICTURE 颜色图。裁剪方法是先把 candidate mask 放回官方 ROI，再取 mask 的外接框，四周加 24 px padding。crop 里面只保留 mask 内的图像，mask 外背景置成浅灰色。这样做是为了让 CLIP 主要看“这个候选 mask 里面是什么”，而不是被周围大面积背景或邻近组织影响。</td><td>目标类别的短文本描述，例如 bronchiola 会包含“有腔的细支气管/支气管上皮”等描述。第二路 FICTURE 语义文本还会加入颜色 factor 的细胞类型和 marker gene 提示。</td><td>CLIP 计算图像 crop 和目标类别文本的相似度。报告里的 <b>CLIP</b> 是 H&E crop 分数和 FICTURE 语义 crop 分数的平均值，所以它代表“形态图像证据 + FICTURE 颜色/基因语义证据”的综合 CLIP 判断。分数越高，CLIP 越认为这个候选像目标类别。</td></tr>",
        "<tr><td><b>Qwen2.5-VL / Gemma / MedGemma / Llama 3.2 Vision</b></td><td>三张对齐 crop：1. H&E crop，看病理形态；2. 官方 FICTURE factor-map crop，看空间转录组颜色/因子；3. 目标类别的 gene/FICTURE prior heatmap crop，看哪些区域有该类别相关基因支持。三张都高亮同一个 candidate mask，因为 VLM 需要知道“你让我判断的是哪一块”，否则它只能看整张 crop 里的组织，无法知道要给哪个 mask 打分。裁剪方法也是取 candidate mask 外接框，四周加 48 px padding，最长边最多缩到 768 px。</td><td>文字里明确告诉 VLM：目标类别是什么、目标类别的病理含义、候选来源和大小、以及 FICTURE factor legend 中颜色、推断细胞类型和 marker genes 的上下文。文字里也附带几个辅助数字：molecular 是候选区域是否落在目标类别相关基因热区，factor semantic 是候选覆盖的 FICTURE 颜色是否像该类别，H&E heuristic 是粗略病理图像规则，shape 是面积/形状先验。</td><td>VLM 直接返回 JSON：<code>score</code>、<code>recall</code>、<code>precision</code> 和简短理由。这里我们只用纯 <code>score</code> 排序，不用融合后的二次分数，也不把它和其他分数再融合。分数越高，VLM 越认为这个候选 mask 是目标类别。</td></tr>",
        "</tbody></table>",
        "<h2>完整 pipeline 怎么跑</h2>",
        "<ol>",
        "<li>先从官方 FICTURE candidate pool 里为每个类别挑 3 个候选：GOOD、MID、BAD。</li>",
        "<li>对每个候选，把 mask 放回同一张官方 ROI 坐标系。</li>",
        "<li>给 CLIP 做两张裁剪图：H&E crop 和 FICTURE crop；给 VLM 做三张裁剪图：H&E、FICTURE、gene-prior heatmap。</li>",
        "<li>把目标类别文字和 FICTURE 颜色/细胞类型/marker gene 说明一起给模型。</li>",
        "<li>模型给每个候选一个分数。每个类别有 3 个候选，所以把 3 个分数从高到低排序。</li>",
        "<li>只看 top-1：最高分候选如果是 GOOD，就是这个模型在这个小 pool 里捞对了；如果最高分是 MID 或 BAD，就是没捞对。</li>",
        "</ol>",
        "<h2>颜色/细胞类型语义怎么来的</h2>",
        "<p class='lead'>来源是 <code>hex_12.k12.pixel.info.tsv</code>：每个 FICTURE false-color factor 都有 RGB 颜色和 marker genes，例如 <code>TopGene_specific</code>/<code>TopGene_pval</code>。我先根据这些 marker genes 的已知生物学含义推断这个颜色大概代表什么细胞类型，然后把“颜色 → 细胞类型 → marker genes”的说明放进 CLIP 文本提示和 VLM 文本上下文。也就是说，细胞类型名称是 marker-gene 推断，不是原始 HTML 直接给出的标准答案。</p>",
        selector_example_text(),
        "<h2>这份图怎么读</h2>",
        "<p class='lead'>每张候选图都是整张组织 ROI，而不是局部裁剪。六个面板依次是：正常 H&E、候选 mask 叠到 H&E 上、候选 mask 单独图、人工 annotation 叠到 H&E 上、annotation 单独图、官方 FICTURE map。蓝色是候选，绿色是真实标注。看蓝色和绿色在整张组织里的位置是否一致，就能判断是否对齐、是否抓到了目标结构。</p>",
        "<h2>模型是否挑中 GOOD</h2>",
        "<p class='lead'>下面的“模型/打分方法”就是用来给候选 mask 排序的方法。最高分候选如果是 GOOD，说明这个模型在这个小 pool 里成功把好候选捞上来了。</p>",
        "<p class='lead'><b>表格列怎么读：</b>“模型 top-1 挑出来的是”就是该模型最高分的候选类型；“GOOD 在 3 个候选里排第几”是 GOOD candidate 按模型分数排序后的名次，1 表示模型把 GOOD 放第一；“GOOD 的模型分数”是 GOOD candidate 自己拿到的分；“top-1 候选真实 Dice”是模型挑出来的那个候选事后和人工 annotation 比出来的 Dice；“3 个候选被打成几种分数”用来判断模型有没有真的区分候选，如果只有 1 种分数，说明三个候选几乎同分，这种情况不能算模型真的会挑。</p>",
        "<table><thead><tr><th>类别</th><th>模型/打分方法</th><th>模型 top-1 挑出来的是</th><th>GOOD 在 3 个候选里排第几</th><th>GOOD 的模型分数</th><th>top-1 候选真实 Dice</th><th>3 个候选被打成几种分数</th><th>结论</th></tr></thead><tbody>",
    ]
    for row in summaries:
        interp = row["interpretation"]
        klass = "hit" if "挑中了" in interp or "并列第一" in interp else "tie" if "一样" in interp else "miss"
        parts.append(
            "<tr>"
            f"<td>{html.escape(row['label_zh'])}<br><code>{html.escape(row['label'])}</code></td>"
            f"<td>{html.escape(row['selector'])}</td>"
            f"<td>{html.escape(row['top_pick'])}</td>"
            f"<td class='num'>{html.escape(str(row['good_rank']))}</td>"
            f"<td class='num'>{html.escape(str(row['good_score']))}</td>"
            f"<td class='num'>{html.escape(str(row['top_dice']))}</td>"
            f"<td class='num'>{html.escape(str(row['unique_scores']))}</td>"
            f"<td class='{klass}'>{html.escape(interp)}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    parts.extend(
        [
            "<h2>候选池全 ROI 图</h2>",
            "<p class='lead'>下面每类固定展示 GOOD、MID、BAD 三个候选。这里的重点不是追求 best，而是看模型能不能从这个小 pool 里把 GOOD 捞上来。</p>",
        ]
    )

    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for label, _slug, label_zh in LABEL_ORDER:
        parts.append(f"<h3>{html.escape(label_zh)} / <code>{html.escape(label)}</code></h3>")
        for row in by_label.get(label, []):
            key = (row["label"], row["manual_pick"])
            rel = panel_rel_paths[key]
            pick_class = row["manual_pick"].lower()
            parts.append("<div class='card'>")
            parts.append(
                f"<div><span class='{pick_class}'>{html.escape(row['manual_pick'])}</span> "
                f"Dice {float(row['true_dice']):.3f}, Precision {float(row['true_precision']):.3f}, "
                f"Recall {float(row['true_recall']):.3f}</div>"
            )
            parts.append(
                f"<div class='meta'>候选来源：<code>{html.escape(row['candidate'])}</code>；原本最适合类别：<code>{html.escape(row['original_best_for'])}</code></div>"
            )
            parts.append(f"<img src='{html.escape(rel)}' alt='{html.escape(label)} {html.escape(row['manual_pick'])} full ROI panel'>")
            parts.append("</div>")

    parts.extend(
        [
            "<h2>输出文件</h2>",
            "<ul>",
            f"<li>官方状态：<code>{html.escape(preflight['summary']['status'])}</code>，ROI <code>{preflight['roi_size_wh'][0]}×{preflight['roi_size_wh'][1]}</code>，factor index <code>{tuple(preflight['factor_shape'])}</code></li>",
            f"<li>本 HTML 展示的方法：{html.escape('、'.join(score_label(column) for column in allowed_columns if column in rows[0]))}</li>",
            f"<li>颜色/细胞类型语义表：<a href='{html.escape(os.path.relpath(legend_path, output_html.parent))}'>{html.escape(legend_path.name)}</a></li>",
            "</ul>",
        ]
    )
    output_html.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scores-csv", type=Path, default=None)
    parser.add_argument("--official-summary", type=Path, default=OFFICIAL_SUMMARY)
    parser.add_argument("--official-input-dir", type=Path, default=OFFICIAL_INPUT_DIR)
    parser.add_argument("--display-height", type=int, default=1050)
    parser.add_argument(
        "--selector-column",
        action="append",
        default=None,
        help="Score column to show. Repeat to override the default clean selector set.",
    )
    args = parser.parse_args()

    report_dir = args.report_dir if args.report_dir.is_absolute() else PROJECT_ROOT / args.report_dir
    output_dir = args.output_dir if args.output_dir is not None else report_dir
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = args.scores_csv or report_dir / "manual_direct_retrieval_scores.csv"
    output_html = output_dir / "chinese_feature_aware_6class_full_roi_pool_report.html"
    panel_dir = output_dir / "full_roi_candidate_panels"
    source_legend_path = report_dir / "feature_color_celltype_legend.csv"
    legend_path = output_dir / "feature_color_celltype_legend.csv"
    if source_legend_path.exists() and source_legend_path.resolve() != legend_path.resolve():
        shutil.copy2(source_legend_path, legend_path)
    if scores_csv.exists():
        score_copy = output_dir / scores_csv.name
        if scores_csv.resolve() != score_copy.resolve():
            shutil.copy2(scores_csv, score_copy)

    preflight = verify_pass_official(
        args.official_summary if args.official_summary.is_absolute() else PROJECT_ROOT / args.official_summary,
        args.official_input_dir if args.official_input_dir.is_absolute() else PROJECT_ROOT / args.official_input_dir,
    )
    rows = read_rows(scores_csv)
    allowed_columns = args.selector_column or DEFAULT_SELECTOR_COLUMNS
    summaries = summarize_selectors(rows, allowed_columns)

    shape_hw = (preflight["roi_size_wh"][1], preflight["roi_size_wh"][0])
    he_full = load_rgb(preflight["he_path"], shape_hw)
    ficture_full = load_rgb(preflight["ficture_path"], shape_hw)
    region_summary = load_json(preflight["region_summary_path"])
    masks_by_slug = {
        (item.get("slug") or item["label"]): resolve_path(item["mask_path"], report_dir)
        for item in region_summary["labels"]
    }

    panel_rel_paths: dict[tuple[str, str], str] = {}
    for label, slug, label_zh in LABEL_ORDER:
        annotation = load_mask(masks_by_slug[slug], shape_hw)
        label_rows = [row for row in rows if row["label"] == label]
        for row in label_rows:
            mask_path = resolve_path(row["mask_path"], report_dir)
            check_no_forbidden([mask_path])
            candidate = load_mask(mask_path, shape_hw)
            panel_path = panel_dir / f"{slug}__{row['manual_pick'].lower()}__full_roi_6panel.png"
            render_candidate_panel(
                row,
                label_zh,
                he_full,
                ficture_full,
                annotation,
                candidate,
                panel_path,
                args.display_height,
            )
            panel_rel_paths[(label, row["manual_pick"])] = os.path.relpath(panel_path, output_html.parent)

    write_html(rows, summaries, output_html, panel_rel_paths, preflight, legend_path, allowed_columns)
    manifest = {
        "status": "WROTE_CHINESE_FEATURE_AWARE_FULL_ROI_POOL_REPORT",
        "html": str(output_html),
        "panel_dir": str(panel_dir),
        "n_rows": len(rows),
        "labels": [label for label, _slug, _zh in LABEL_ORDER],
        "selector_columns": [column for column in allowed_columns if column in rows[0]],
        "official_summary": str(args.official_summary),
        "official_input_dir": str(args.official_input_dir),
        "notes": [
            "Direct retrieval hit-test, not fused reranking.",
            "Figures use full official ROI; candidate masks are resized back to 3144x3327 for visualization.",
        ],
    }
    (output_dir / "chinese_feature_aware_6class_full_roi_pool_report_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"WROTE {output_html}")
    print(f"WROTE {panel_dir}")
    print(f"rows={len(rows)} labels={len(LABEL_ORDER)}")


if __name__ == "__main__":
    main()
