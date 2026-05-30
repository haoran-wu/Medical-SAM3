#!/usr/bin/env python3
"""Run paired H&E/FICTURE VLM direct retrieval through OpenRouter."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]

SCORE_RUBRIC_SCORE_ONLY = """Give one integer similarity score from 0 to 100.
Use the full scale when the visual evidence differs across candidates:
- near 0 means clearly not the target tissue class
- around 25 means weak or mostly unrelated match
- around 50 means mixed or partial match
- around 75 means likely target, but imperfect boundary or coverage
- near 100 means near-perfect target candidate
Judge this candidate independently from the images. Do not use a default score.
Your entire answer must be only the integer score. Do not return JSON, markdown, words, precision, recall, or a reason."""

SCORE_RUBRIC_WITH_REASON = """Give one integer similarity score from 0 to 100.
Use the full scale when the visual evidence differs across candidates. Also provide one short reason.
Return one valid JSON object with exactly two keys: score and reason. Do not include precision or recall."""


def score_rubric(include_reason: bool) -> str:
    return SCORE_RUBRIC_WITH_REASON if include_reason else SCORE_RUBRIC_SCORE_ONLY


def row_key(row: dict) -> Tuple[str, str, str, str, str]:
    return (row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"]))


def parse_json_score(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
        raw_score = float(nums[0]) if nums else 0.0
        score = raw_score / 100.0 if raw_score > 1.0 else raw_score
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "raw_score": raw_score,
            "reason": text[:360],
            "parse_status": "fallback_number" if nums else "no_score_found",
        }
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"score": 0.0, "raw_score": 0.0, "reason": text[:360], "parse_status": "malformed_json"}
    data["reason"] = str(data.get("reason", ""))[:500]
    try:
        raw_score = float(data.get("score", 0.0))
        data["raw_score"] = raw_score
        data["score"] = max(0.0, min(1.0, raw_score / 100.0 if raw_score > 1.0 else raw_score))
        data["parse_status"] = "ok"
    except Exception:
        data["score"] = 0.0
        data["raw_score"] = 0.0
        data["parse_status"] = "bad_score"
    return data


def image_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_description_only_prompt(row: dict, include_reason: bool = False) -> str:
    image_cue = row.get("image_cue") or (
        "Image 1: H&E histology crop.\n"
        "Image 2: official FICTURE factor-color crop for the same candidate mask."
    )
    return f"""You are given two aligned image crops for the same candidate mask.
{image_cue}

Target tissue class: {row['display']} ({row['label']})
Meaning of the target class: {row['target_description']}

Use only the two images, the candidate mask/crop, and the target-class description above.
Do not use any FICTURE color/cell-type/gene legend. Do not assume the candidate is correct.
Score how well the candidate region matches the target class.

{score_rubric(include_reason)}"""


def load_factor_legend(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["Factor"]))
    return rows


def format_factor_legend(rows: Sequence[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            "Factor {factor}: RGB {rgb}; Major Compartment: {major}; Celltype2: {celltype}".format(
                factor=row["Factor"],
                rgb=row["RGB"],
                major=row["Major Compartment"],
                celltype=row["Celltype2"],
            )
        )
    return "\n".join(lines)


def build_full_legend_prompt(row: dict, factor_legend_text: str, include_reason: bool = False) -> str:
    return f"""You are given two aligned image crops for the same candidate mask.

Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: official FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

The FICTURE image uses this factor legend:
{factor_legend_text}

Target tissue class: {row['display']} ({row['label']})
Meaning of the target class: {row['target_description']}

Score how well this candidate mask matches the target tissue class.
Use both H&E morphology and FICTURE factor colors.

{score_rubric(include_reason)}"""


def openrouter_chat(
    api_key: str,
    model: str,
    he_data_url: str,
    ficture_data_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
    response_format: bool,
) -> Tuple[str, dict]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful pathology image judge. Your task is direct retrieval: "
                    "score whether the shown candidate mask/crop is a good visual match for the requested target class. "
                    "You must output one continuous ranking score, not precision/recall estimates."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": he_data_url}},
                    {"type": "image_url", "image_url": {"url": ficture_data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/Medical-SAM3",
        "X-Title": "Medical-SAM3 VLM candidate hit-test",
    }
    data = json.dumps(body).encode("utf-8")
    url = "https://openrouter.ai/api/v1/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(str(item.get("text", item)) for item in content)
            if content is None:
                diagnostic = {
                    "error": "openrouter_returned_no_message_content",
                    "finish_reason": choice.get("finish_reason"),
                    "native_finish_reason": choice.get("native_finish_reason"),
                    "message_keys": sorted(message.keys()),
                }
                for key in ("refusal", "reasoning", "reasoning_details"):
                    if message.get(key):
                        diagnostic[key] = str(message.get(key))[:1000]
                content = json.dumps(diagnostic)
            return str(content), payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            body_text = exc.read().decode("utf-8", errors="replace")[:1200]
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body_text}") from exc
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_warning(scores: List[float]) -> Tuple[int, int, str]:
    rounded = [round(score, 6) for score in scores]
    unique_count = len(set(rounded))
    top_score = max(rounded) if rounded else 0.0
    top1_tie_size = sum(1 for score in rounded if score == top_score)
    if unique_count <= 1:
        return unique_count, top1_tie_size, "uninformative tie: all scores are identical"
    if unique_count <= 2 or top1_tie_size >= 5:
        return unique_count, top1_tie_size, "coarse/tied scores: ranking may be unstable"
    return unique_count, top1_tie_size, ""


def retrieval_metrics(rows: List[dict], score_key: str, method: str) -> List[dict]:
    out: List[dict] = []
    for label, display in LABEL_ORDER:
        label_rows = [row for row in rows if row["label"] == label]
        ranked = sorted(label_rows, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)
        if not ranked:
            continue
        top1 = ranked[0]
        top3 = ranked[:3]
        top5 = ranked[:5]
        good_ranks = [i + 1 for i, row in enumerate(ranked) if row["sample_bucket"] == "GOOD"]
        score_unique_count, top1_tie_size, warning = score_warning(
            [float(row.get(score_key) or 0.0) for row in ranked]
        )
        out.append(
            {
                "method": method,
                "label": label,
                "display": display,
                "n_candidates": len(ranked),
                "top1_bucket": top1["sample_bucket"],
                "top1_hidden_dice": top1["hidden_dice"],
                "top1_hidden_precision": top1["hidden_precision"],
                "top1_hidden_recall": top1["hidden_recall"],
                "top1_score": f"{float(top1.get(score_key) or 0.0):.6f}",
                "top1_candidate": f"{top1['source']}/{top1['setting']}/{top1['candidate_id']}",
                "good_count_top3": sum(1 for row in top3 if row["sample_bucket"] == "GOOD"),
                "good_count_top5": sum(1 for row in top5 if row["sample_bucket"] == "GOOD"),
                "mean_hidden_dice_top5": f"{sum(float(row['hidden_dice']) for row in top5) / max(len(top5), 1):.6f}",
                "best_good_rank": min(good_ranks) if good_ranks else "",
                "score_unique_count": score_unique_count,
                "top1_tie_size": top1_tie_size,
                "tie_warning": warning,
            }
        )
    return out


def write_html(output_dir: Path, model: str, rows: List[dict], metrics_rows: List[dict]) -> None:
    prompt_mode = rows[0].get("prompt_mode", "description_only") if rows else "description_only"
    vals = sorted({round(float(row.get("vlm_score") or 0.0), 6) for row in rows})
    top1_good = sum(1 for row in metrics_rows if row["top1_bucket"] == "GOOD")
    good_top5 = sum(int(row["good_count_top5"]) for row in metrics_rows)
    warning = ""
    if len(vals) == 1:
        warning = "所有 candidate 分数完全一样，是无信息 tie。"
    elif len(vals) <= 3:
        warning = "分数种类很少，排序可能较粗，需要谨慎。"

    metric_table = []
    for row in metrics_rows:
        metric_table.append(
            "<tr>"
            f"<td>{html.escape(row['display'])}</td>"
            f"<td>{html.escape(row['top1_bucket'])}</td>"
            f"<td>{float(row['top1_hidden_dice']):.3f}</td>"
            f"<td>{html.escape(str(row['good_count_top5']))} / 5</td>"
            f"<td>{html.escape(str(row['best_good_rank']))}</td>"
            f"<td>{html.escape(str(row.get('score_unique_count') or ''))}</td>"
            f"<td>{html.escape(row.get('tie_warning') or '')}</td>"
            f"<td><code>{html.escape(row['top1_candidate'])}</code></td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>OpenRouter VLM direct retrieval result</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:30px; color:#20242a; line-height:1.55; }}
h2 {{ margin-top:30px; border-bottom:1px solid #dfe3e7; padding-bottom:6px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:14px 0; }}
th,td {{ border-bottom:1px solid #dfe3e7; padding:8px 10px; text-align:left; vertical-align:top; }}
th {{ background:#fafafa; }} code {{ background:#f1f3f5; padding:2px 5px; border-radius:4px; }}
.box {{ background:#f7f8fa; border:1px solid #dfe3e7; border-radius:8px; padding:14px 16px; }}
</style>
</head>
<body>
<h1>OpenRouter VLM direct retrieval</h1>
<div class="box">
<p><b>模型：</b>{html.escape(model)}</p>
<p><b>Prompt 模式：</b>{html.escape(prompt_mode)}</p>
<p><b>实验设计：</b>90 个 gray reverse-blur candidate，6 类，每类 GOOD/MID/BAD 各 5 个。模型看到 H&E crop、FICTURE crop，以及目标类别描述。GOOD/MID/BAD 和 Dice 只在评价时使用，不给模型看。</p>
<p><b>总览：</b>top1 GOOD = {top1_good}/6；GOOD in top5 = {good_top5}/30；score 种类数 = {len(vals)}。{html.escape(warning)}</p>
</div>
<h2>逐类别结果</h2>
<table><thead><tr><th>class</th><th>top1 是什么</th><th>top1 Dice</th><th>top5 GOOD</th><th>第一个 GOOD 排名</th><th>score 种类数</th><th>tie warning</th><th>top1 candidate</th></tr></thead><tbody>
{''.join(metric_table)}
</tbody></table>
<h2>文件</h2>
<ul>
<li><code>vlm_candidate_scores.csv</code>: 每个 candidate 的原始模型分数和 reason</li>
<li><code>retrieval_comparison_metrics.csv</code>: 逐类别 top1/top5 评价</li>
</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-reason", action="store_true", help="Ask the VLM to include a short reason for debugging.")
    parser.add_argument(
        "--prompt-mode",
        choices=["description_only", "full_legend"],
        default="description_only",
        help="Prompt style for the VLM retrieval test.",
    )
    parser.add_argument(
        "--factor-legend-csv",
        type=Path,
        help="CSV with FICTURE factor legend. Required when --prompt-mode full_legend.",
    )
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send OpenRouter response_format=json_object. Useful for models that return empty content with JSON mode.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing {args.api_key_env}")
    factor_legend_text = ""
    if args.prompt_mode == "full_legend":
        if not args.factor_legend_csv:
            raise SystemExit("--factor-legend-csv is required for --prompt-mode full_legend")
        factor_legend_text = format_factor_legend(load_factor_legend(args.factor_legend_csv))

    requests = list(csv.DictReader((args.pool_dir / "public_vlm_requests.csv").open()))
    hidden = {row_key(row): row for row in csv.DictReader((args.pool_dir / "hidden_candidate_truth.csv").open())}
    if args.limit:
        requests = requests[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing: Dict[Tuple[str, str, str, str, str], dict] = {}
    scores_path = args.output_dir / "vlm_candidate_scores.csv"
    if args.resume and scores_path.exists():
        for row in csv.DictReader(scores_path.open()):
            existing[row_key(row)] = row

    rows: List[dict] = []
    for idx, row in enumerate(requests, start=1):
        key = row_key(row)
        if key in existing:
            rows.append(existing[key])
            print(f"Reused {idx}/{len(requests)} {row['display']} {row['sample_bucket']}", flush=True)
            continue

        he_data = image_data_url(args.pool_dir / row["he_crop_rel"], args.max_side, args.jpeg_quality)
        ficture_data = image_data_url(args.pool_dir / row["ficture_crop_rel"], args.max_side, args.jpeg_quality)
        if args.prompt_mode == "full_legend":
            prompt = build_full_legend_prompt(row, factor_legend_text, args.include_reason)
        else:
            prompt = build_description_only_prompt(row, args.include_reason)
        raw, payload = openrouter_chat(
            api_key=api_key,
            model=args.model,
            he_data_url=he_data,
            ficture_data_url=ficture_data,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
            response_format=not args.disable_response_format,
        )
        parsed = parse_json_score(raw)
        truth = hidden[key]
        usage = payload.get("usage") or {}
        out = dict(row)
        out.update(
            {
                "model": args.model,
                "provider": "openrouter",
                "prompt_mode": args.prompt_mode,
                "actual_prompt_text": prompt,
                "vlm_score": f"{float(parsed['score']):.9f}",
                "vlm_score_raw": f"{float(parsed.get('raw_score', parsed['score'])):.9f}",
                "vlm_parse_status": parsed.get("parse_status", ""),
                "raw_response": raw,
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "hidden_dice": truth["hidden_dice"],
                "hidden_precision": truth["hidden_precision"],
                "hidden_recall": truth["hidden_recall"],
                "hidden_iou": truth["hidden_iou"],
            }
        )
        if args.include_reason:
            out["vlm_reason"] = parsed.get("reason", "")
        rows.append(out)
        write_csv(scores_path, rows, list(rows[0].keys()))
        print(f"Scored {idx}/{len(requests)} {row['display']} {row['sample_bucket']} score={out['vlm_score']}", flush=True)

    write_csv(scores_path, rows, list(rows[0].keys()))
    metrics_rows = retrieval_metrics(rows, "vlm_score", f"OpenRouter {args.prompt_mode}: {args.model}")
    write_csv(args.output_dir / "retrieval_comparison_metrics.csv", metrics_rows)
    write_html(args.output_dir, args.model, rows, metrics_rows)
    (args.output_dir / "README_中文.md").write_text(
        "# OpenRouter VLM direct retrieval\n\n"
        f"Model: `{args.model}`\n\n"
        f"Prompt mode: `{args.prompt_mode}`\n\n"
        "Input per candidate: H&E crop, FICTURE crop, and target tissue prompt.\n"
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
