#!/usr/bin/env python3
"""Render a plain-Chinese VLM-only direct retrieval report."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "x"


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_pool_dir(item: str) -> Tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"Expected image_mode=/path/to/pool, got {item}")
    name, path = item.split("=", 1)
    return name, Path(path)


def parse_run(item: str) -> Tuple[str, str, Path]:
    parts = item.split("|", 2)
    if len(parts) != 3:
        raise SystemExit(f"Expected image_mode|model_name|/path/to/result, got {item}")
    mode, model, path = parts
    return mode, model, Path(path)


def score_warning(scores: List[float]) -> Tuple[int, int, str]:
    rounded = [round(score, 6) for score in scores]
    unique_count = len(set(rounded))
    top_score = max(rounded) if rounded else 0.0
    top1_tie_size = sum(1 for score in rounded if score == top_score)
    if unique_count <= 1:
        return unique_count, top1_tie_size, "无效：所有 candidate 分数一样"
    if unique_count <= 2 or top1_tie_size >= 5:
        return unique_count, top1_tie_size, "警告：分数太粗或 top1 并列太多"
    return unique_count, top1_tie_size, ""


def copy_crop(pool_dir: Path, rel_path: str, out_path: Path) -> str:
    src = pool_dir / rel_path
    if not src.exists():
        return ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    return str(out_path)


def analyze_run(
    image_mode: str,
    model_name: str,
    result_dir: Path,
    pool_dir: Path,
    figure_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    rows = read_rows(result_dir / "vlm_candidate_scores.csv")
    metrics: List[dict] = []
    top5_rows: List[dict] = []
    for label, display in LABEL_ORDER:
        label_rows = [row for row in rows if row["label"] == label]
        ranked = sorted(label_rows, key=lambda row: float(row.get("vlm_score") or 0.0), reverse=True)
        if not ranked:
            continue
        top1 = ranked[0]
        top5 = ranked[:5]
        good_ranks = [idx + 1 for idx, row in enumerate(ranked) if row["sample_bucket"] == "GOOD"]
        unique_count, tie_size, warning = score_warning([float(row.get("vlm_score") or 0.0) for row in ranked])
        metrics.append(
            {
                "image_input": image_mode,
                "model": model_name,
                "class": display,
                "n_candidates": len(ranked),
                "top1_is_good": "YES" if top1["sample_bucket"] == "GOOD" else "NO",
                "top1_bucket": top1["sample_bucket"],
                "top1_source": top1["source"],
                "top1_dice": f"{float(top1['hidden_dice']):.3f}",
                "top5_good_count": sum(1 for row in top5 if row["sample_bucket"] == "GOOD"),
                "best_good_rank": min(good_ranks) if good_ranks else "",
                "top5_mean_dice": f"{sum(float(row['hidden_dice']) for row in top5) / len(top5):.3f}",
                "score_unique_count": unique_count,
                "top1_tie_size": tie_size,
                "tie_warning": warning,
                "result_dir": str(result_dir),
            }
        )
        for rank, row in enumerate(top5, start=1):
            base = f"{slug(image_mode)}__{slug(model_name)}__{slug(display)}__rank{rank}"
            he_out = figure_dir / f"{base}_he.png"
            fic_out = figure_dir / f"{base}_ficture.png"
            he_rel = copy_crop(pool_dir, row["he_crop_rel"], he_out)
            fic_rel = copy_crop(pool_dir, row["ficture_crop_rel"], fic_out)
            top5_rows.append(
                {
                    "image_input": image_mode,
                    "model": model_name,
                    "class": display,
                    "rank": rank,
                    "bucket": row["sample_bucket"],
                    "source": row["source"],
                    "dice": f"{float(row['hidden_dice']):.3f}",
                    "score": f"{float(row.get('vlm_score') or 0.0):.3f}",
                    "candidate": f"{row['source']}/{row['setting']}/{row['candidate_id']}",
                    "reason": row.get("vlm_reason", ""),
                    "he_image": Path(he_rel).name if he_rel else "",
                    "ficture_image": Path(fic_rel).name if fic_rel else "",
                }
            )
    return metrics, top5_rows


def render_html(
    out_dir: Path,
    summary: List[dict],
    metrics: List[dict],
    top5_rows: List[dict],
    failed_notes: List[str],
) -> None:
    metric_defs = [
        ("Top1 是不是 GOOD", "VLM 排第一的 candidate 是不是隐藏答案里的 GOOD。"),
        ("Top5 里有几个 GOOD", "VLM 前 5 名里面有几个 GOOD，满分是 5。"),
        ("Best GOOD Rank", "第一个 GOOD 出现在第几名；越小越好，1 表示第一名就是 GOOD。"),
        ("Top5 平均 Dice", "前 5 名 candidate 的真实 Dice 平均值；Dice 没给模型看，只用于评价。"),
        ("Tie warning", "如果很多 candidate 分数一样，说明模型没有真正区分，结果不能过度相信。"),
    ]
    summary_head = list(summary[0].keys()) if summary else []
    metric_head = [
        "image_input",
        "model",
        "class",
        "top1_is_good",
        "top1_bucket",
        "top1_source",
        "top1_dice",
        "top5_good_count",
        "best_good_rank",
        "top5_mean_dice",
        "tie_warning",
    ]
    summary_rows = [
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in summary_head) + "</tr>"
        for row in summary
    ]
    metric_rows = [
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in metric_head) + "</tr>"
        for row in metrics
    ]
    def_rows = [
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(desc)}</td></tr>" for name, desc in metric_defs
    ]
    failed_html = "".join(f"<li>{html.escape(note)}</li>" for note in failed_notes)
    per_class_counts = sorted({int(row.get("n_candidates") or 0) for row in metrics if row.get("n_candidates")})
    if len(per_class_counts) == 1:
        pool_size_text = f"每类 {per_class_counts[0]} 个 candidate，总共约 {per_class_counts[0] * 6} 个。"
    elif per_class_counts:
        pool_size_text = f"每类 candidate 数量不完全一样，范围是 {per_class_counts[0]} 到 {per_class_counts[-1]} 个。"
    else:
        pool_size_text = "candidate 数量见下方每类结果表。"

    details = []
    run_keys = []
    for row in top5_rows:
        key = (row["image_input"], row["model"], row["class"])
        if key not in run_keys:
            run_keys.append(key)
    for image_mode, model, display in run_keys:
        rows = [row for row in top5_rows if (row["image_input"], row["model"], row["class"]) == (image_mode, model, display)]
        cards = []
        for row in rows:
            he_img = f"figures/{html.escape(row['he_image'])}" if row["he_image"] else ""
            fic_img = f"figures/{html.escape(row['ficture_image'])}" if row["ficture_image"] else ""
            cards.append(
                "<article>"
                f"<h4>Rank {row['rank']} | {html.escape(row['bucket'])} | {html.escape(row['source'])} | Dice {html.escape(row['dice'])} | score {html.escape(row['score'])}</h4>"
                "<div class='imgs'>"
                f"<img src='{he_img}' alt='H&E crop'>"
                f"<img src='{fic_img}' alt='FICTURE crop'>"
                "</div>"
                f"<p><b>candidate:</b> {html.escape(row['candidate'])}</p>"
                f"<p><b>VLM reason:</b> {html.escape(row['reason'])}</p>"
                "</article>"
            )
        details.append(
            f"<details><summary>{html.escape(image_mode)} / {html.escape(model)} / {html.escape(display)} Top5</summary>"
            "<div class='grid'>"
            + "".join(cards)
            + "</div></details>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>VLM 直接检索 GOOD candidate 小测试</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #20242a; line-height: 1.55; }}
.note {{ max-width: 1120px; }}
table {{ border-collapse: collapse; margin: 18px 0; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px 9px; text-align: left; vertical-align: top; }}
code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
details {{ margin: 14px 0; padding: 10px 0; border-top: 1px solid #eee; }}
summary {{ cursor: pointer; font-weight: 700; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; margin-top: 12px; }}
article {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: white; }}
.imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
img {{ width: 100%; border: 1px solid #eee; background: #fafafa; }}
</style>
</head>
<body>
<h1>VLM 直接检索 GOOD candidate 小测试</h1>
<div class="note">
<h2>实验设计</h2>
<p><b>Goal：</b>测试 VLM 能不能从 candidate pool 里把真正好的 mask 排到前面。</p>
<p><b>Candidate pool：</b>6 个组织类别，{html.escape(pool_size_text)}GOOD/MID/BAD 是按隐藏 Dice 分出来的，模型看不到这些标签。</p>
<p><b>Model inputs：</b>每个 candidate 给 VLM 两张图：一张 H&E crop，一张官方 PASS_OFFICIAL FICTURE crop。两张图来自同一个 mask、同一个 ROI。这里比较两种输入图：<code>light-gray masked crop</code> 和 <code>gray reverse blur</code>，不使用蓝色 overlay。</p>
<p><b>Model output：</b>VLM 只输出一个 0-1 分数。分数越高，表示它越认为这个 candidate 像目标类别。排序按这个分数从高到低。</p>
<p><b>Ground truth use：</b>Dice/Precision/Recall 和 GOOD/MID/BAD 只在模型打分后用于评价，没有放进 prompt。</p>
<p><b>Official-data guardrail：</b>FICTURE 图只使用 PASS_OFFICIAL 的 same-ROI 版本，不使用旧的 deprecated FICTURE 输出。</p>
</div>
{"<h2>失败或未完成模型</h2><ul>" + failed_html + "</ul>" if failed_html else ""}
<h2>指标解释</h2>
<table><thead><tr><th>指标</th><th>意思</th></tr></thead><tbody>{''.join(def_rows)}</tbody></table>
<h2>模型总览</h2>
<table><thead><tr>{''.join(f'<th>{html.escape(key)}</th>' for key in summary_head)}</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<h2>每个类别结果</h2>
<table><thead><tr>{''.join(f'<th>{html.escape(key)}</th>' for key in metric_head)}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<h2>Top5 可视化</h2>
{''.join(details)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", action="append", required=True, help="image_mode=/path/to/pool")
    parser.add_argument("--run", action="append", required=True, help="image_mode|model_name|/path/to/result")
    parser.add_argument("--failed-note", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    pool_dirs: Dict[str, Path] = dict(parse_pool_dir(item) for item in args.pool_dir)
    metrics: List[dict] = []
    top5_rows: List[dict] = []
    for item in args.run:
        image_mode, model_name, result_dir = parse_run(item)
        if image_mode not in pool_dirs:
            raise SystemExit(f"No pool-dir provided for image mode {image_mode}")
        run_metrics, run_top5 = analyze_run(image_mode, model_name, result_dir, pool_dirs[image_mode], figure_dir)
        metrics.extend(run_metrics)
        top5_rows.extend(run_top5)

    summary: List[dict] = []
    run_keys = []
    for row in metrics:
        key = (row["image_input"], row["model"])
        if key not in run_keys:
            run_keys.append(key)
    for image_mode, model_name in run_keys:
        rows = [row for row in metrics if (row["image_input"], row["model"]) == (image_mode, model_name)]
        summary.append(
            {
                "image_input": image_mode,
                "model": model_name,
                "Top1_GOOD_classes": f"{sum(1 for row in rows if row['top1_is_good'] == 'YES')}/6",
                "GOOD_in_Top5_total": f"{sum(int(row['top5_good_count']) for row in rows)}/30",
                "mean_top1_Dice": f"{sum(float(row['top1_dice']) for row in rows) / len(rows):.3f}",
                "mean_top5_Dice": f"{sum(float(row['top5_mean_dice']) for row in rows) / len(rows):.3f}",
                "warning_classes": sum(1 for row in rows if row["tie_warning"]),
            }
        )

    write_csv(
        args.output_dir / "per_class_vlm_retrieval_metrics.csv",
        metrics,
        list(metrics[0].keys()) if metrics else [],
    )
    write_csv(
        args.output_dir / "model_summary.csv",
        summary,
        list(summary[0].keys()) if summary else [],
    )
    write_csv(
        args.output_dir / "top5_candidates.csv",
        top5_rows,
        list(top5_rows[0].keys()) if top5_rows else [],
    )
    render_html(args.output_dir, summary, metrics, top5_rows, args.failed_note)
    (args.output_dir / "README_中文.md").write_text(
        "# VLM 直接检索 GOOD candidate 小测试\n\n"
        "打开 `index.html`。表格只使用 VLM 分数排序，不混入 CLIP 或 fused score。\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
