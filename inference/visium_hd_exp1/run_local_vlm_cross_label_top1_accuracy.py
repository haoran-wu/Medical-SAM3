#!/usr/bin/env python3
"""Local VLM cross-label tissue classification for paired H&E/FICTURE crops."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image

from run_paired_vlm_hit_test import generate, load_vlm, write_csv
from vlm_prompt_contract import DEFAULT_FACTOR_LEGEND_CSV, DEFAULT_POOL_CSV, SYSTEM_PROMPT, build_user_prompt


CLASS_KEYS = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune_infiltration",
]

TRUE_LABEL_TO_CLASS = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
}

CLASS_DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}

def row_key(row: dict) -> Tuple[str, str, str, str, str]:
    return (row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"]))


def row_key_string(row: dict) -> str:
    return "__".join(row_key(row))


def parse_scores(text: str) -> Tuple[Dict[str, int], str]:
    stripped = text.strip()
    object_candidates = re.findall(r"\{[^{}]*\}", stripped, flags=re.S)
    if not object_candidates:
        raise ValueError("No JSON object found")
    object_text = object_candidates[-1]
    for candidate in reversed(object_candidates):
        if all(key in candidate for key in CLASS_KEYS):
            object_text = candidate
            break
    try:
        data = json.loads(object_text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(object_text)
        except (SyntaxError, ValueError):
            data = {}
            for key in CLASS_KEYS:
                key_match = re.search(
                    rf"(?:\"{key}\"|'{key}'|{key})\s*:\s*(-?\d+(?:\.\d+)?)",
                    object_text,
                )
                if key_match:
                    data[key] = key_match.group(1)
    if set(data.keys()) != set(CLASS_KEYS):
        raise ValueError(f"Expected keys {CLASS_KEYS}, got {sorted(data.keys())}")
    numeric_values: Dict[str, float] = {}
    for key in CLASS_KEYS:
        value = data[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} is boolean, not a numeric score")
        if isinstance(value, str):
            value = value.strip()
        numeric_values[key] = float(value)
    values = list(numeric_values.values())
    if all(0.0 <= value <= 1.0 for value in values):
        numeric_values = {key: value * 100.0 for key, value in numeric_values.items()}
    scores: Dict[str, int] = {}
    for key, numeric in numeric_values.items():
        integer = int(round(numeric))
        if integer < 0 or integer > 100:
            raise ValueError(f"{key} score out of range: {integer}")
        scores[key] = integer
    return scores, json.dumps(scores, sort_keys=True)


def predicted_from_scores(scores: Dict[str, int]) -> Tuple[str, bool, int]:
    max_score = max(scores.values())
    winners = [key for key, value in scores.items() if value == max_score]
    return (winners[0] if len(winners) == 1 else ""), len(winners) > 1, max_score


def accuracy_row(name: str, rows: Sequence[dict]) -> dict:
    denominator = len(rows)
    correct = sum(1 for row in rows if row["is_correct"] == "true")
    return {
        "group": name,
        "n": denominator,
        "correct": correct,
        "top1_accuracy": f"{correct / denominator:.6f}" if denominator else "",
    }


def write_accuracy_tables(output_dir: Path, prediction_rows: List[dict]) -> None:
    valid_rows = [row for row in prediction_rows if row.get("parse_status") == "ok"]
    write_csv(output_dir / "overall_accuracy.csv", [accuracy_row("overall", valid_rows)])
    buckets = sorted({row.get("sample_bucket", "") for row in valid_rows if row.get("sample_bucket", "")})
    bucket_rows = [accuracy_row(bucket, [row for row in valid_rows if row["sample_bucket"] == bucket]) for bucket in buckets]
    write_csv(output_dir / "bucket_accuracy.csv", bucket_rows)
    per_class = []
    for class_key in CLASS_KEYS:
        class_rows = [row for row in valid_rows if row["true_class"] == class_key]
        item = accuracy_row(CLASS_DISPLAY[class_key], class_rows)
        item["true_class"] = class_key
        per_class.append(item)
    write_csv(output_dir / "per_class_accuracy.csv", per_class)


def score_warning(scores: Sequence[int]) -> Tuple[int, int, str]:
    unique_count = len(set(scores))
    top_score = max(scores) if scores else 0
    top1_tie_size = sum(1 for score in scores if score == top_score)
    if unique_count <= 1:
        return unique_count, top1_tie_size, "all same score"
    if unique_count <= 2 or top1_tie_size >= 5:
        return unique_count, top1_tie_size, "coarse or tied scores"
    return unique_count, top1_tie_size, ""


def write_same_class_retrieval_tables(output_dir: Path, prediction_rows: List[dict]) -> None:
    valid_rows = [row for row in prediction_rows if row.get("parse_status") == "ok"]
    metrics_rows: List[dict] = []
    score_rows: List[dict] = []
    for class_key in CLASS_KEYS:
        label_rows = [row for row in valid_rows if row["true_class"] == class_key]
        for row in label_rows:
            score_rows.append(
                {
                    "row_index": row["row_index"],
                    "candidate_uid": row["candidate_uid"],
                    "target_class": class_key,
                    "sample_bucket": row["sample_bucket"],
                    "source": row["source"],
                    "run": row["run"],
                    "setting": row["setting"],
                    "candidate_id": row["candidate_id"],
                    "target_class_score": row[class_key],
                }
            )
        ranked = sorted(label_rows, key=lambda item: int(item[class_key]), reverse=True)
        if not ranked:
            continue
        target_scores = [int(row[class_key]) for row in ranked]
        unique_count, top1_tie_size, warning = score_warning(target_scores)
        top1 = ranked[0]
        top5 = ranked[:5]
        good_ranks = [idx + 1 for idx, row in enumerate(ranked) if row["sample_bucket"] == "GOOD"]
        metrics_rows.append(
            {
                "target_class": class_key,
                "display": CLASS_DISPLAY[class_key],
                "n_candidates": len(ranked),
                "top1_good": "true" if top1["sample_bucket"] == "GOOD" and top1_tie_size == 1 else "false",
                "top1_bucket": top1["sample_bucket"],
                "good_in_top5": sum(1 for row in top5 if row["sample_bucket"] == "GOOD"),
                "best_good_rank": min(good_ranks) if good_ranks else "",
                "unique_scores": f"{unique_count}/{len(ranked)}",
                "top1_tie_size": top1_tie_size,
                "tie_warning": warning,
                "top1_score": top1[class_key],
                "top1_candidate": f"{top1['source']}/{top1['setting']}/{top1['candidate_id']}",
            }
        )
    write_csv(output_dir / "same_class_retrieval_scores.csv", score_rows)
    write_csv(output_dir / "same_class_retrieval_metrics.csv", metrics_rows)


def write_html_report(output_dir: Path, prediction_rows: List[dict], model: str, pool_path: Path) -> None:
    overall = list(csv.DictReader((output_dir / "overall_accuracy.csv").open()))
    buckets = list(csv.DictReader((output_dir / "bucket_accuracy.csv").open()))
    classes = list(csv.DictReader((output_dir / "per_class_accuracy.csv").open()))
    failed = list(csv.DictReader((output_dir / "failed_rows.csv").open())) if (output_dir / "failed_rows.csv").exists() else []
    retrieval = (
        list(csv.DictReader((output_dir / "same_class_retrieval_metrics.csv").open()))
        if (output_dir / "same_class_retrieval_metrics.csv").exists()
        else []
    )

    def table(rows: Iterable[dict], cols: Sequence[str]) -> str:
        out = ["<table><thead><tr>"]
        out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
        out.append("</tr></thead><tbody>")
        for row in rows:
            out.append("<tr>")
            out.extend(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in cols)
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    preview_cols = [
        "candidate_uid",
        "sample_bucket",
        "true_class",
        "predicted_class",
        "top_score",
        "top_score_tie",
        "is_correct",
    ]
    preview = prediction_rows[:24]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Local VLM Cross-label and Same-class Retrieval</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:28px; color:#20242a; line-height:1.55; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
th,td {{ border-bottom:1px solid #ddd; padding:7px 9px; text-align:left; vertical-align:top; }}
.note {{ max-width:1050px; }}
code {{ background:#f4f4f4; padding:1px 4px; border-radius:4px; }}
</style>
</head>
<body>
<h1>Local VLM Cross-label and Same-class Retrieval</h1>
<div class="note">
<h2>Experiment Design</h2>
<p><b>Goal:</b> 对每个 candidate 在 6 个 tissue class 上同时打分。同一套分数同时用于两个实验：Test1 取最高分做 tissue classification；Test2 在同一 true class 内按对应 class score 排序，看 GOOD mask 能不能排到前面。</p>
<p><b>Candidate pool:</b> 90 个 gray reverse blur candidate，6 个 true class 各 15 个。GOOD/MID/BAD 只用于抽样分组，不给模型看。</p>
<p><b>Model inputs:</b> 每个 candidate 给两张图：H&amp;E crop 和 official FICTURE color crop。mask 内清晰，mask 外灰色并模糊；没有蓝色 overlay。</p>
<p><b>Model outputs:</b> 模型返回 6 个整数 score：bronchiola、alveoli、vessels、tumor、stroma、immune_infiltration。外部脚本取最高分作为 predicted label。</p>
<p><b>Ground truth use:</b> true label、GOOD/MID/BAD、Dice/Precision/Recall 都对模型隐藏，只在打分结束后用于评价。</p>
<p><b>Official-data guardrail:</b> FICTURE 使用 PASS_OFFICIAL / same-ROI 数据，不使用 deprecated FICTURE roots。</p>
<p><b>Model:</b> {html.escape(model)}</p>
<p><b>Input pool:</b> <code>{html.escape(str(pool_path))}</code></p>
</div>

<h2>Overall Top1 Accuracy</h2>
{table(overall, ["group", "n", "correct", "top1_accuracy"])}

<h2>GOOD / MID / BAD Groups</h2>
<p>这里 GOOD/MID/BAD 不是模型预测目标，只是按照原先 candidate 的 Dice 质量分出来的抽样组。</p>
{table(buckets, ["group", "n", "correct", "top1_accuracy"])}

<h2>Accuracy By True Class</h2>
{table(classes, ["true_class", "group", "n", "correct", "top1_accuracy"])}

<h2>Test2: Same-class Candidate Retrieval</h2>
<p>这里不再额外调用模型。对于每个 true class 的 15 个 candidate，直接用该 class 的 score 排序。例如 bronchiola 的 15 个 candidate 按 bronchiola score 排序。</p>
{table(retrieval, ["target_class", "n_candidates", "top1_good", "good_in_top5", "unique_scores", "top1_bucket", "top1_score", "top1_candidate", "tie_warning"])}

<h2>Prediction Preview</h2>
{table(preview, preview_cols)}

<h2>Failures</h2>
<p>{len(failed)} unresolved local generation/parse failures.</p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-legend-csv", type=Path, default=DEFAULT_FACTOR_LEGEND_CSV)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    user_prompt = build_user_prompt(args.factor_legend_csv)
    pool_dir = args.pool_csv.parent
    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.limit:
        rows = rows[: args.limit]
    label_counts = Counter(row["label"] for row in rows)
    bucket_counts = Counter(row["sample_bucket"] for row in rows)
    if not rows:
        raise SystemExit("Input pool has no rows")
    if set(label_counts) - set(TRUE_LABEL_TO_CLASS):
        raise SystemExit(f"Unexpected labels in input pool: {sorted(set(label_counts) - set(TRUE_LABEL_TO_CLASS))}")
    if len(label_counts) != len(CLASS_KEYS):
        raise SystemExit(f"Expected six labels in input pool, got {dict(label_counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT)
    (args.output_dir / "prompt_user.txt").write_text(user_prompt)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(args.pool_csv),
                "factor_legend_csv": str(args.factor_legend_csv),
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "class_keys": CLASS_KEYS,
            },
            indent=2,
        )
    )

    cross_path = args.output_dir / "cross_label_scores.csv"
    pred_path = args.output_dir / "per_candidate_predictions.csv"
    failed_path = args.output_dir / "failed_rows.csv"
    responses_path = args.output_dir / "raw_responses.jsonl"

    prediction_rows: List[dict] = []
    done_keys = set()
    if args.resume and pred_path.exists():
        prediction_rows = list(csv.DictReader(pred_path.open()))
        done_keys = {row["row_key"] for row in prediction_rows if row.get("parse_status") == "ok"}

    failed_rows: List[dict] = []
    if args.resume and failed_path.exists():
        failed_rows = list(csv.DictReader(failed_path.open()))

    vlm = load_vlm(args.model, args.device)

    for idx, row in enumerate(rows, start=1):
        key = row_key_string(row)
        if key in done_keys:
            continue
        raw = ""
        try:
            he_image = Image.open(pool_dir / row["he_crop_rel"]).convert("RGB")
            ficture_image = Image.open(pool_dir / row["ficture_crop_rel"]).convert("RGB")
            raw = generate(vlm, args.device, [he_image, ficture_image], SYSTEM_PROMPT, user_prompt, args.max_new_tokens)
            scores, parsed_json = parse_scores(raw)
            predicted, top_tie, top_score = predicted_from_scores(scores)
            true_class = TRUE_LABEL_TO_CLASS[row["label"]]
            is_correct = bool(predicted and predicted == true_class and not top_tie)
            out = {
                "row_index": idx,
                "row_key": key,
                "candidate_uid": row["candidate_uid"],
                "true_label": row["label"],
                "true_class": true_class,
                "display": row["display"],
                "sample_bucket": row["sample_bucket"],
                "source": row["source"],
                "run": row["run"],
                "setting": row["setting"],
                "candidate_id": row["candidate_id"],
                "predicted_class": predicted,
                "top_score": top_score,
                "top_score_tie": "true" if top_tie else "false",
                "is_correct": "true" if is_correct else "false",
                "parse_status": "ok",
                **{class_key: scores[class_key] for class_key in CLASS_KEYS},
            }
            prediction_rows = [old for old in prediction_rows if old["row_key"] != key] + [out]
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            done_keys.add(key)
            with responses_path.open("a") as handle:
                handle.write(json.dumps({"row_key": key, "parsed_json": parsed_json, "raw_response": raw}) + "\n")
            print(f"Scored {idx}/{len(rows)} true={true_class} pred={predicted or 'TIE'} correct={is_correct}", flush=True)
        except Exception as exc:
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            failed_rows.append(
                {
                    "row_index": idx,
                    "row_key": key,
                    "candidate_uid": row["candidate_uid"],
                    "true_label": row["label"],
                    "sample_bucket": row["sample_bucket"],
                    "error": str(exc)[:2400],
                    "raw_response": raw[:2400],
                }
            )
            print(f"FAILED {idx}/{len(rows)} {key}: {str(exc)[:200]}", flush=True)
        prediction_rows = sorted(prediction_rows, key=lambda item: int(item["row_index"]))
        write_csv(pred_path, prediction_rows)
        cross_rows = [
            {
                "row_index": pred["row_index"],
                "row_key": pred["row_key"],
                "candidate_uid": pred["candidate_uid"],
                "true_class": pred["true_class"],
                "sample_bucket": pred["sample_bucket"],
                **{class_key: pred[class_key] for class_key in CLASS_KEYS},
            }
            for pred in prediction_rows
        ]
        write_csv(cross_path, cross_rows)
        write_csv(
            failed_path,
            failed_rows,
            fieldnames=["row_index", "row_key", "candidate_uid", "true_label", "sample_bucket", "error"],
        )

    if failed_rows:
        print(f"{len(failed_rows)} rows failed; rerun with --resume after fixing the issue.", flush=True)
    write_accuracy_tables(args.output_dir, prediction_rows)
    write_same_class_retrieval_tables(args.output_dir, prediction_rows)
    write_html_report(args.output_dir, prediction_rows, args.model, args.pool_csv)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
