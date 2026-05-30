#!/usr/bin/env python3
"""Local VLM cross-label tissue classification for paired H&E/FICTURE crops."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image

from run_paired_vlm_hit_test import generate, load_vlm, write_csv


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

SYSTEM_PROMPT = (
    "You are a pathology image classifier. Score which tissue class the shown candidate region "
    "most likely belongs to. Return only the requested JSON object."
)

USER_PROMPT = """You are given two aligned crops of the same candidate region:

Image 1: H&E reverse-blur crop. The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: FICTURE reverse-blur crop. The candidate region is sharp and full color; the outside region is grayscale and blurred.

Score how likely this candidate belongs to each class.

Classes:
bronchiola = bronchiolar airway tissue, airway-like lumen, epithelial lining.
alveoli = alveolar lung parenchyma, open air spaces, thin septa.
vessels = blood vessel or vascular wall, lumen-like vascular structure.
tumor = malignant epithelial tumor region, dense tumor nests.
stroma = stromal or mesenchymal tissue, collagen/fibroblast/smooth-muscle-like tissue.
immune_infiltration = immune-cell-rich region, small round-cell aggregates.

Rules:
- Return exactly one JSON object.
- Use exactly these six keys: bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
- Each value must be an integer from 0 to 100.
- Higher means more likely.
- Do not include explanation, reason, precision, recall, Dice, markdown, or extra text.
- Do not give all classes the same score unless there is truly no visible evidence."""


def row_key(row: dict) -> Tuple[str, str, str, str, str]:
    return (row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"]))


def row_key_string(row: dict) -> str:
    return "__".join(row_key(row))


def parse_scores(text: str) -> Tuple[Dict[str, int], str]:
    stripped = text.strip()
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError("No JSON object found")
    data = json.loads(match.group(0))
    if set(data.keys()) != set(CLASS_KEYS):
        raise ValueError(f"Expected keys {CLASS_KEYS}, got {sorted(data.keys())}")
    scores: Dict[str, int] = {}
    for key in CLASS_KEYS:
        value = data[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} is boolean, not an integer score")
        if isinstance(value, str):
            value = value.strip()
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError(f"{key} score is not integer: {value!r}")
        integer = int(numeric)
        if integer < 0 or integer > 100:
            raise ValueError(f"{key} score out of range: {integer}")
        scores[key] = integer
    return scores, match.group(0)


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
    bucket_rows = [
        accuracy_row(bucket, [row for row in valid_rows if row["sample_bucket"] == bucket])
        for bucket in ["GOOD", "MID", "BAD"]
    ]
    write_csv(output_dir / "bucket_accuracy.csv", bucket_rows)
    per_class = []
    for class_key in CLASS_KEYS:
        class_rows = [row for row in valid_rows if row["true_class"] == class_key]
        item = accuracy_row(CLASS_DISPLAY[class_key], class_rows)
        item["true_class"] = class_key
        per_class.append(item)
    write_csv(output_dir / "per_class_accuracy.csv", per_class)


def write_html_report(output_dir: Path, prediction_rows: List[dict], model: str, pool_path: Path) -> None:
    overall = list(csv.DictReader((output_dir / "overall_accuracy.csv").open()))
    buckets = list(csv.DictReader((output_dir / "bucket_accuracy.csv").open()))
    classes = list(csv.DictReader((output_dir / "per_class_accuracy.csv").open()))
    failed = list(csv.DictReader((output_dir / "failed_rows.csv").open())) if (output_dir / "failed_rows.csv").exists() else []

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
<title>Local VLM Cross-Label Top1 Accuracy</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:28px; color:#20242a; line-height:1.55; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
th,td {{ border-bottom:1px solid #ddd; padding:7px 9px; text-align:left; vertical-align:top; }}
.note {{ max-width:1050px; }}
code {{ background:#f4f4f4; padding:1px 4px; border-radius:4px; }}
</style>
</head>
<body>
<h1>Local VLM Cross-Label Top1 Accuracy</h1>
<div class="note">
<h2>Experiment Design</h2>
<p><b>Goal:</b> 让本地部署的 VLM 对每个 candidate 在 6 个 tissue class 上同时打分，然后用最高分对应的 class 作为预测 label。</p>
<p><b>Candidate pool:</b> 90 个 gray reverse blur candidate，6 个 true class 各 15 个。GOOD/MID/BAD 只用于抽样分组，不给模型看。</p>
<p><b>Model inputs:</b> 每个 candidate 给两张图：H&amp;E crop 和 official FICTURE factor-color crop。mask 内清晰，mask 外灰色并模糊；没有蓝色 overlay。</p>
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
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pool_dir = args.pool_csv.parent
    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.limit:
        rows = rows[: args.limit]
    label_counts = Counter(row["label"] for row in rows)
    bucket_counts = Counter(row["sample_bucket"] for row in rows)
    if args.limit == 0:
        if len(rows) != 90:
            raise SystemExit(f"Expected 90 rows, got {len(rows)}")
        if sorted(label_counts.values()) != [15, 15, 15, 15, 15, 15]:
            raise SystemExit(f"Expected 15 rows per label, got {dict(label_counts)}")
        if dict(bucket_counts) != {"GOOD": 30, "MID": 30, "BAD": 30}:
            raise SystemExit(f"Expected GOOD/MID/BAD counts 30 each, got {dict(bucket_counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT)
    (args.output_dir / "prompt_user.txt").write_text(USER_PROMPT)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(args.pool_csv),
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
        try:
            he_image = Image.open(pool_dir / row["he_crop_rel"]).convert("RGB")
            ficture_image = Image.open(pool_dir / row["ficture_crop_rel"]).convert("RGB")
            raw = generate(vlm, args.device, [he_image, ficture_image], SYSTEM_PROMPT, USER_PROMPT, args.max_new_tokens)
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
    write_html_report(args.output_dir, prediction_rows, args.model, args.pool_csv)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
