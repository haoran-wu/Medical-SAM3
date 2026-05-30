#!/usr/bin/env python3
"""Aggregate paired CLIP/VLM hit-test results into CSV and a Chinese HTML report."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, List, Tuple


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", action="append", required=True, help="display_name=/path/to/result_root")
    parser.add_argument("--failed-note", action="append", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots: Dict[str, Path] = {}
    for item in args.model_root:
        if "=" not in item:
            raise SystemExit(f"Expected display_name=/path mapping, got {item}")
        name, path = item.split("=", 1)
        roots[name] = Path(path)

    combined: List[dict] = []
    first_root = next(iter(roots.values()))
    for row in read_rows(first_root / "retrieval_comparison_metrics.csv"):
        if row["method"].startswith("OpenAI CLIP"):
            out = dict(row)
            out["model_group"] = "CLIP baseline"
            combined.append(out)

    for model, root in roots.items():
        path = root / "retrieval_comparison_metrics.csv"
        if not path.exists():
            continue
        for row in read_rows(path):
            if row["method"].startswith("VLM direct score"):
                out = dict(row)
                out["model_group"] = model
                out["method"] = model
                combined.append(out)

    fieldnames = [
        "model_group",
        "method",
        "label",
        "display",
        "n_candidates",
        "top1_bucket",
        "top1_hidden_dice",
        "top1_hidden_precision",
        "top1_hidden_recall",
        "top1_score",
        "top1_candidate",
        "good_count_top3",
        "good_count_top5",
        "mean_hidden_dice_top5",
        "best_good_rank",
    ]
    write_csv(args.output_dir / "combined_clip_vlm_retrieval_metrics.csv", combined, fieldnames)

    method_keys: List[Tuple[str, str]] = []
    for row in combined:
        key = (row["model_group"], row["method"])
        if key not in method_keys:
            method_keys.append(key)
    summary: List[dict] = []
    for group, method in method_keys:
        rows = [row for row in combined if row["model_group"] == group and row["method"] == method]
        if not rows:
            continue
        unique_scores = ""
        warning = ""
        if group in roots:
            score_csv = roots[group] / "vlm_candidate_scores.csv"
            if score_csv.exists():
                scores = {round(float(row.get("vlm_score") or 0.0), 6) for row in read_rows(score_csv)}
                unique_scores = str(len(scores))
                if len(scores) <= 1:
                    warning = "uninformative tie: all VLM scores are identical"
                elif len(scores) <= 2:
                    warning = "very coarse scores: top ranks may be unstable"
        summary.append(
            {
                "model_group": group,
                "method": method,
                "top1_good_labels": str(sum(1 for row in rows if row["top1_bucket"] == "GOOD")),
                "top1_mid_labels": str(sum(1 for row in rows if row["top1_bucket"] == "MID")),
                "top1_bad_labels": str(sum(1 for row in rows if row["top1_bucket"] == "BAD")),
                "good_in_top5_total": str(sum(int(row["good_count_top5"]) for row in rows)),
                "mean_top1_hidden_dice": f"{sum(float(row['top1_hidden_dice']) for row in rows) / len(rows):.6f}",
                "mean_top5_hidden_dice": f"{sum(float(row['mean_hidden_dice_top5']) for row in rows) / len(rows):.6f}",
                "score_unique_values_total": unique_scores,
                "warning": warning,
            }
        )
    write_csv(args.output_dir / "method_summary.csv", summary, list(summary[0].keys()))

    summary_rows = []
    for row in summary:
        summary_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in row.keys())
            + "</tr>"
        )
    class_rows = []
    for row in combined:
        cols = ["method", "display", "top1_bucket", "top1_hidden_dice", "good_count_top5", "best_good_rank", "top1_candidate"]
        class_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in cols)
            + "</tr>"
        )
    failed = "".join(f"<li>{html.escape(note)}</li>" for note in args.failed_note)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>CLIP/VLM direct retrieval summary</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; line-height: 1.55; color: #20242a; }}
table {{ border-collapse: collapse; margin: 18px 0; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px 10px; text-align: left; }}
code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>CLIP/VLM 小 pool 直接检索结果</h1>
<p><b>实验设计：</b>6 个类别，每类 15 个候选 mask，包括 GOOD/MID/BAD 各 5 个。模型看到同一个候选的 H&E crop、FICTURE crop 和文字提示；隐藏 Dice 只用于最后评价。目标是看模型能不能把 GOOD 排到前面，不是 fused reranking。</p>
<p><b>候选池：</b><code>{html.escape(str(args.pool_dir))}</code></p>
<p><b>指标解释：</b><code>top1_good_labels</code> 是 6 个类别里 top1 选中 GOOD 的类别数，满分 6；<code>good_in_top5_total</code> 是所有类别 top5 中 GOOD 的总数，满分 30。</p>
{"<ul>" + failed + "</ul>" if failed else ""}
<h2>方法总览</h2>
<table><thead><tr>{''.join(f'<th>{html.escape(k)}</th>' for k in summary[0].keys())}</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<h2>逐类别 top1 / top5</h2>
<table><thead><tr><th>method</th><th>class</th><th>top1 bucket</th><th>top1 hidden Dice</th><th>GOOD in top5</th><th>best GOOD rank</th><th>top1 candidate</th></tr></thead><tbody>{''.join(class_rows)}</tbody></table>
</body>
</html>
"""
    (args.output_dir / "index.html").write_text(html_text)
    (args.output_dir / "README_中文.md").write_text(
        "# CLIP/VLM direct retrieval summary\n\n"
        "See `index.html`, `method_summary.csv`, and `combined_clip_vlm_retrieval_metrics.csv`.\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
